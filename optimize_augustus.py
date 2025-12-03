#!/usr/bin/env python3
"""
optimize_augustus.py - Fast AUGUSTUS parameter optimization using Bayesian Optimization

This is a Python reimplementation of BGM2AT.optimize_augustus from GETA pipeline.
Key improvements:
1. Bayesian Optimization (Optuna) instead of grid search - 5-10x fewer evaluations
2. Minimal disk I/O - symlinks instead of full config copies
3. Efficient parallelization with ProcessPoolExecutor
4. Optional RAM-based temp files (/dev/shm)

Author: Rewritten for Won's lab
Original: chenlianfu/geta
"""

import argparse
import os
import sys
import re
import shutil
import tempfile
import subprocess
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Using grid search fallback.", file=sys.stderr)
    print("Install with: pip install optuna", file=sys.stderr)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class GeneModel:
    """Represents a gene model from GenBank format"""
    content: str
    gene_name: str
    cds_length: int


@dataclass 
class AccuracyMetrics:
    """AUGUSTUS prediction accuracy metrics"""
    nucleotide_sensitivity: float = 0.0
    nucleotide_specificity: float = 0.0
    exon_sensitivity: float = 0.0
    exon_specificity: float = 0.0
    gene_sensitivity: float = 0.0
    gene_specificity: float = 0.0
    
    @property
    def weighted_accuracy(self) -> float:
        """Calculate weighted accuracy score (same formula as original)"""
        return (
            self.nucleotide_sensitivity * 3 +
            self.nucleotide_specificity * 2 +
            self.exon_sensitivity * 4 +
            self.exon_specificity * 3 +
            self.gene_sensitivity * 2 +
            self.gene_specificity * 1
        ) / 15


class AugustusOptimizer:
    """Main optimizer class for AUGUSTUS parameters"""
    
    # 28 parameters to optimize (from metapars.cfg)
    PARAM_DEFINITIONS = {
        '/IntronModel/d': {'type': 'int', 'default_range': (100, 950)},
        '/IntronModel/e': {'type': 'int', 'default_range': (10, 100)},
        '/IntronModel/slope_of_bandwidth': {'type': 'float', 'default_range': (0.1, 0.5)},
        '/IntronModel/minwindowcount': {'type': 'int', 'default_range': (1, 6)},
        '/IntronModel/aession_i': {'type': 'int', 'default_range': (0, 500)},
        '/IntronModel/ass_upwindow_size': {'type': 'int', 'default_range': (10, 50)},
        '/IntronModel/ass_start': {'type': 'int', 'default_range': (1, 5)},
        '/IntronModel/ass_end': {'type': 'int', 'default_range': (1, 5)},
        '/IntronModel/dss_start': {'type': 'int', 'default_range': (1, 5)},
        '/IntronModel/dss_end': {'type': 'int', 'default_range': (0, 4)},
        '/IGenicModel/k': {'type': 'int', 'default_range': (1, 5)},
        '/IGenicModel/verbosity': {'type': 'int', 'default_range': (0, 3)},
        '/ExonModel/k': {'type': 'int', 'default_range': (1, 5)},
        '/ExonModel/minPatSum': {'type': 'int', 'default_range': (10, 200)},
        '/ExonModel/slope_of_bandwidth': {'type': 'float', 'default_range': (0.1, 1.0)},
        '/ExonModel/minwindowcount': {'type': 'int', 'default_range': (1, 10)},
        '/ExonModel/etorder': {'type': 'int', 'default_range': (0, 2)},
        '/ExonModel/etpseudocount': {'type': 'int', 'default_range': (1, 10)},
        '/ExonModel/exonlengthD': {'type': 'int', 'default_range': (500, 4000)},
        '/ExonModel/maxexonlength': {'type': 'int', 'default_range': (5000, 30000)},
        '/ExonModel/minexonlength': {'type': 'int', 'default_range': (1, 10)},
        '/UtrModel/d': {'type': 'int', 'default_range': (100, 1000)},
        '/UtrModel/e': {'type': 'int', 'default_range': (50, 200)},
        '/UtrModel/slope_of_bandwidth': {'type': 'float', 'default_range': (0.1, 0.8)},
        '/UtrModel/minwindowcount': {'type': 'int', 'default_range': (1, 5)},
        '/Constant/decomp_num_steps': {'type': 'int', 'default_range': (1, 3)},
        '/Constant/min_coding_len': {'type': 'int', 'default_range': (50, 300)},
        '/Constant/probNinCoding': {'type': 'float', 'default_range': (0.1, 0.4)},
    }
    
    def __init__(
        self,
        species_name: str,
        train_gb: str,
        augustus_config_path: Optional[str] = None,
        onlytrain_gb: Optional[str] = None,
        cpu: int = 8,
        n_trials: int = 100,
        test_ratio: float = 0.2,
        min_test_genes: int = 100,
        max_test_genes: int = 600,
        genes_per_test: int = 50,
        min_intron_len: int = 30,
        use_memory: bool = False,
        output_dir: str = "optimize_output"
    ):
        self.species_name = species_name
        self.train_gb = Path(train_gb).resolve()
        self.onlytrain_gb = Path(onlytrain_gb).resolve() if onlytrain_gb else None
        self.cpu = cpu
        self.n_trials = n_trials
        self.test_ratio = test_ratio
        self.min_test_genes = min_test_genes
        self.max_test_genes = max_test_genes
        self.genes_per_test = genes_per_test
        self.min_intron_len = min_intron_len
        self.use_memory = use_memory
        self.output_dir = Path(output_dir).resolve()
        
        # Set AUGUSTUS_CONFIG_PATH
        if augustus_config_path:
            self.config_path = Path(augustus_config_path).resolve()
            os.environ['AUGUSTUS_CONFIG_PATH'] = str(self.config_path)
        else:
            self.config_path = Path(os.environ.get('AUGUSTUS_CONFIG_PATH', ''))
        
        if not self.config_path.exists():
            raise ValueError(f"AUGUSTUS_CONFIG_PATH not found: {self.config_path}")
        
        self.species_dir = self.config_path / 'species' / species_name
        self.params_file = self.species_dir / f'{species_name}_parameters.cfg'
        self.metapars_file = self.species_dir / f'{species_name}_metapars.cfg'
        
        if not self.params_file.exists():
            raise ValueError(f"Parameters file not found: {self.params_file}")
        
        # Load gene models
        self.test_genes: List[GeneModel] = []
        self.train_genes: List[GeneModel] = []
        self.onlytrain_genes: List[GeneModel] = []
        
        # Temp directory
        if use_memory and Path('/dev/shm').exists():
            self.tmp_base = Path('/dev/shm') / f'augustus_opt_{os.getpid()}'
        else:
            self.tmp_base = self.output_dir / 'tmp'
        
        # Parameter ranges (will be updated from metapars.cfg)
        self.param_ranges: Dict[str, dict] = {}
        
        # Best accuracy tracking
        self.best_accuracy = 0.0
        self.best_params: Dict[str, float] = {}
        
    def setup(self):
        """Initialize directories and load data"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_base.mkdir(parents=True, exist_ok=True)
        
        # Backup original config files
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        shutil.copy(self.params_file, f"{self.params_file}.{timestamp}.bak")
        if self.metapars_file.exists():
            shutil.copy(self.metapars_file, f"{self.metapars_file}.{timestamp}.bak")
        
        # Load parameter ranges
        self._load_metapars()
        
        # Load gene models
        self._load_gene_models()
        
        logger.info(f"Loaded {len(self.test_genes)} test genes, {len(self.train_genes)} train genes")
        if self.onlytrain_genes:
            logger.info(f"Loaded {len(self.onlytrain_genes)} onlytrain genes")
    
    def _load_metapars(self):
        """Parse metapars.cfg to get parameter ranges"""
        if not self.metapars_file.exists():
            logger.warning(f"metapars.cfg not found, using defaults")
            self.param_ranges = self.PARAM_DEFINITIONS.copy()
            return
        
        with open(self.metapars_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                match = re.match(r'^\s*(\S+)\s+(.+)', line)
                if not match:
                    continue
                
                param_name, range_str = match.groups()
                
                # Parse range: "100"-"950" (int) or "0.1"_"0.5" (float)
                int_match = re.search(r'"([^"]+)"-"([^"]+)"', range_str)
                float_match = re.search(r'"([^"]+)"_"([^"]+)"', range_str)
                
                if int_match:
                    low, high = int(int_match.group(1)), int(int_match.group(2))
                    self.param_ranges[param_name] = {
                        'type': 'int',
                        'range': (low, high)
                    }
                elif float_match:
                    low, high = float(float_match.group(1)), float(float_match.group(2))
                    self.param_ranges[param_name] = {
                        'type': 'float', 
                        'range': (low, high)
                    }
                else:
                    # Discrete values
                    values = re.findall(r'"([^"]+)"', range_str)
                    if values:
                        self.param_ranges[param_name] = {
                            'type': 'categorical',
                            'choices': values
                        }
        
        logger.info(f"Loaded {len(self.param_ranges)} parameter ranges")
    
    def _load_gene_models(self):
        """Load gene models from GenBank files"""
        all_genes = self._parse_genbank(self.train_gb)
        
        if self.onlytrain_gb:
            self.onlytrain_genes = self._parse_genbank(self.onlytrain_gb)
        
        # Determine test set size
        n_genes = len(all_genes)
        n_test = int(n_genes * self.test_ratio)
        n_test = max(min(n_test, self.max_test_genes), min(n_genes, self.min_test_genes))
        
        # Random split
        random.shuffle(all_genes)
        self.test_genes = all_genes[:n_test]
        self.train_genes = all_genes[n_test:]
        
    def _parse_genbank(self, filepath: Path) -> List[GeneModel]:
        """Parse GenBank format file into gene models"""
        genes = []
        
        with open(filepath) as f:
            content = f.read()
        
        # Split by record separator
        records = content.split('//\n')
        
        for record in records:
            if not record.strip() or 'LOCUS' not in record:
                continue
            
            # Extract gene name
            gene_match = re.search(r'gene="([^"]+)"', record)
            gene_name = gene_match.group(1) if gene_match else 'unknown'
            
            # Calculate CDS length
            cds_length = 0
            # Remove whitespace for CDS parsing
            compact = re.sub(r'\s+', '', record)
            join_match = re.search(r'CDS.*join\(([^)]+)\)', compact)
            if join_match:
                for span in join_match.group(1).split(','):
                    coord_match = re.search(r'(\d+)\.\.(\d+)', span)
                    if coord_match:
                        cds_length += abs(int(coord_match.group(2)) - int(coord_match.group(1))) + 1
            
            genes.append(GeneModel(
                content=record + '//\n',
                gene_name=gene_name,
                cds_length=cds_length
            ))
        
        return genes
    
    def _create_temp_config(self, params: Dict[str, float], tmp_dir: Path) -> Path:
        """Create temporary config with modified parameters - using symlinks"""
        config_dir = tmp_dir / 'config'
        species_subdir = config_dir / 'species' / self.species_name
        species_subdir.mkdir(parents=True, exist_ok=True)
        
        # Symlink most directories (no copy!)
        for item in ['cgp', 'extrinsic', 'model', 'parameters', 'profile']:
            src = self.config_path / item
            dst = config_dir / item
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
        
        # Symlink species directory contents except parameters.cfg
        src_species = self.config_path / 'species' / self.species_name
        for item in src_species.iterdir():
            dst = species_subdir / item.name
            if item.name != f'{self.species_name}_parameters.cfg' and not dst.exists():
                dst.symlink_to(item)
        
        # Symlink other species (for generic templates)
        for other_species in (self.config_path / 'species').iterdir():
            if other_species.name != self.species_name:
                dst = config_dir / 'species' / other_species.name
                if not dst.exists():
                    dst.symlink_to(other_species)
        
        # Create modified parameters.cfg
        with open(self.params_file) as f:
            params_content = f.read()
        
        for param_name, value in params.items():
            # Replace parameter value
            pattern = rf'({re.escape(param_name)}\s+)\S+'
            if isinstance(value, float):
                replacement = rf'\g<1>{value:.6f}'
            else:
                replacement = rf'\g<1>{value}'
            params_content = re.sub(pattern, replacement, params_content)
        
        modified_params = species_subdir / f'{self.species_name}_parameters.cfg'
        with open(modified_params, 'w') as f:
            f.write(params_content)
        
        return config_dir
    
    def _run_single_test(
        self,
        test_genes: List[GeneModel],
        train_genes: List[GeneModel],
        config_dir: Path,
        work_dir: Path
    ) -> AccuracyMetrics:
        """Run etraining + augustus for a single test batch"""
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Write gene model files
        test_file = work_dir / 'test.gb'
        train_file = work_dir / 'train.gb'
        
        with open(test_file, 'w') as f:
            for gene in test_genes:
                f.write(gene.content)
        
        with open(train_file, 'w') as f:
            for gene in train_genes:
                f.write(gene.content)
            for gene in self.onlytrain_genes:
                f.write(gene.content)
        
        # Run etraining
        etraining_cmd = [
            'etraining',
            f'--min_intron_len={self.min_intron_len}',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={config_dir}',
            str(train_file)
        ]
        
        try:
            subprocess.run(
                etraining_cmd,
                capture_output=True,
                check=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"etraining timeout in {work_dir}")
            return AccuracyMetrics()
        except subprocess.CalledProcessError as e:
            logger.warning(f"etraining failed: {e.stderr.decode()[:200]}")
            return AccuracyMetrics()
        
        # Run augustus
        augustus_cmd = [
            'augustus',
            f'--min_intron_len={self.min_intron_len}',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={config_dir}',
            str(test_file)
        ]
        
        try:
            result = subprocess.run(
                augustus_cmd,
                capture_output=True,
                check=True,
                timeout=600
            )
            output = result.stdout.decode()
        except subprocess.TimeoutExpired:
            logger.warning(f"augustus timeout in {work_dir}")
            return AccuracyMetrics()
        except subprocess.CalledProcessError as e:
            logger.warning(f"augustus failed: {e.stderr.decode()[:200]}")
            return AccuracyMetrics()
        
        # Parse accuracy from output
        return self._parse_augustus_output(output, test_genes)
    
    def _parse_augustus_output(
        self,
        output: str,
        test_genes: List[GeneModel]
    ) -> AccuracyMetrics:
        """Parse AUGUSTUS output to extract accuracy metrics"""
        metrics = AccuracyMetrics()
        total_cds_length = sum(g.cds_length for g in test_genes)
        
        for line in output.split('\n'):
            # nucleotide level |  0.95 |  0.89 |
            nuc_match = re.search(r'nucleotide level[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
            if nuc_match:
                metrics.nucleotide_sensitivity = float(nuc_match.group(1))
                metrics.nucleotide_specificity = float(nuc_match.group(2))
            
            # exon level       | pred | annot | TP |
            exon_match = re.search(r'exon level[\s|]+([\d.]+)[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
            if exon_match:
                pred, annot, tp = map(float, exon_match.groups())
                if annot > 0 and pred > 0:
                    metrics.exon_sensitivity = tp / annot
                    metrics.exon_specificity = tp / pred
            
            # gene level
            gene_match = re.search(r'gene level[\s|]+([\d.]+)[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
            if gene_match:
                pred, annot, tp = map(float, gene_match.groups())
                if annot > 0 and pred > 0:
                    metrics.gene_sensitivity = tp / annot
                    metrics.gene_specificity = tp / pred
        
        return metrics
    
    def evaluate_params(self, params: Dict[str, float]) -> float:
        """Evaluate a parameter set by running cross-validation"""
        # Create temp directory for this evaluation
        eval_id = random.randint(100000, 999999)
        eval_dir = self.tmp_base / f'eval_{eval_id}'
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create config with modified parameters
            config_dir = self._create_temp_config(params, eval_dir)
            
            # Split test genes into batches
            test_genes_shuffled = self.test_genes.copy()
            random.shuffle(test_genes_shuffled)
            
            batches = []
            for i in range(0, len(test_genes_shuffled), self.genes_per_test):
                batch = test_genes_shuffled[i:i + self.genes_per_test]
                if batch:
                    batches.append(batch)
            
            # Run tests in parallel
            all_metrics = []
            
            with ProcessPoolExecutor(max_workers=min(self.cpu, len(batches))) as executor:
                futures = []
                for i, test_batch in enumerate(batches):
                    # Training genes = all except current test batch
                    train_batch = [g for g in self.train_genes]
                    for j, other_batch in enumerate(batches):
                        if j != i:
                            train_batch.extend(other_batch)
                    
                    work_dir = eval_dir / f'batch_{i}'
                    futures.append(executor.submit(
                        self._run_single_test,
                        test_batch,
                        train_batch,
                        config_dir,
                        work_dir
                    ))
                
                for future in as_completed(futures):
                    try:
                        metrics = future.result()
                        all_metrics.append(metrics)
                    except Exception as e:
                        logger.warning(f"Batch evaluation failed: {e}")
            
            # Average metrics
            if not all_metrics:
                return 0.0
            
            avg_accuracy = sum(m.weighted_accuracy for m in all_metrics) / len(all_metrics)
            return avg_accuracy
            
        finally:
            # Cleanup
            shutil.rmtree(eval_dir, ignore_errors=True)
    
    def optimize_with_optuna(self) -> Dict[str, float]:
        """Run Bayesian optimization with Optuna"""
        if not OPTUNA_AVAILABLE:
            return self.optimize_grid_search()
        
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            for param_name, param_info in self.param_ranges.items():
                if param_info['type'] == 'int':
                    low, high = param_info['range']
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif param_info['type'] == 'float':
                    low, high = param_info['range']
                    params[param_name] = trial.suggest_float(param_name, low, high)
                elif param_info['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_info['choices']
                    )
            
            accuracy = self.evaluate_params(params)
            
            # Track best
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_params = params.copy()
                logger.info(f"New best accuracy: {accuracy:.4f}")
            
            return accuracy
        
        # Create study
        sampler = TPESampler(seed=42, n_startup_trials=10)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='augustus_optimization'
        )
        
        # Optimize
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=1,  # We parallelize within each trial
            show_progress_bar=True
        )
        
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def optimize_grid_search(self) -> Dict[str, float]:
        """Fallback grid search optimization (if Optuna not available)"""
        logger.info("Running grid search optimization (fallback mode)")
        
        best_params = {}
        
        # Read current parameter values as starting point
        with open(self.params_file) as f:
            params_content = f.read()
        
        for param_name, param_info in self.param_ranges.items():
            # Get current value
            match = re.search(rf'{re.escape(param_name)}\s+(\S+)', params_content)
            if match:
                current_value = match.group(1)
                try:
                    if param_info['type'] == 'float':
                        best_params[param_name] = float(current_value)
                    else:
                        best_params[param_name] = int(current_value)
                except ValueError:
                    best_params[param_name] = current_value
        
        # Iterative optimization
        for round_num in range(3):
            logger.info(f"Grid search round {round_num + 1}")
            improved = False
            
            for param_name, param_info in self.param_ranges.items():
                # Generate test values
                if param_info['type'] == 'int':
                    low, high = param_info['range']
                    test_values = [int(low + i * (high - low) / 5) for i in range(6)]
                elif param_info['type'] == 'float':
                    low, high = param_info['range']
                    test_values = [low + i * (high - low) / 5 for i in range(6)]
                else:
                    test_values = param_info.get('choices', [])
                
                best_value = best_params.get(param_name, test_values[0])
                best_acc = 0.0
                
                for value in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = value
                    
                    acc = self.evaluate_params(test_params)
                    if acc > best_acc:
                        best_acc = acc
                        best_value = value
                
                if best_value != best_params.get(param_name):
                    improved = True
                    best_params[param_name] = best_value
                    logger.info(f"  {param_name}: {best_value} (accuracy: {best_acc:.4f})")
            
            if not improved:
                logger.info("No improvement in this round, stopping")
                break
        
        self.best_params = best_params
        return best_params
    
    def apply_best_params(self):
        """Apply best parameters to the original config file"""
        if not self.best_params:
            logger.warning("No best parameters to apply")
            return
        
        with open(self.params_file) as f:
            content = f.read()
        
        for param_name, value in self.best_params.items():
            pattern = rf'({re.escape(param_name)}\s+)\S+'
            if isinstance(value, float):
                replacement = rf'\g<1>{value:.6f}'
            else:
                replacement = rf'\g<1>{value}'
            content = re.sub(pattern, replacement, content)
        
        with open(self.params_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Applied best parameters to {self.params_file}")
    
    def update_stop_codon_probs(self):
        """Update stop codon probabilities based on training data"""
        # Run etraining to get stop codon frequencies
        all_gb = self.output_dir / 'all_training.gb'
        with open(all_gb, 'w') as f:
            for gene in self.test_genes + self.train_genes + self.onlytrain_genes:
                f.write(gene.content)
        
        result = subprocess.run(
            [
                'etraining',
                f'--min_intron_len={self.min_intron_len}',
                f'--species={self.species_name}',
                f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
                str(all_gb)
            ],
            capture_output=True
        )
        
        output = result.stdout.decode()
        
        # Parse stop codon probabilities
        tag_prob = taa_prob = tga_prob = None
        for line in output.split('\n'):
            if line.startswith('tag:'):
                match = re.search(r'\(([\d.]+)\)', line)
                if match:
                    tag_prob = float(match.group(1))
            elif line.startswith('taa:'):
                match = re.search(r'\(([\d.]+)\)', line)
                if match:
                    taa_prob = float(match.group(1))
            elif line.startswith('tga:'):
                match = re.search(r'\(([\d.]+)\)', line)
                if match:
                    tga_prob = float(match.group(1))
        
        if all([tag_prob, taa_prob, tga_prob]):
            with open(self.params_file) as f:
                content = f.read()
            
            content = re.sub(r'(/Constant/amberprob\s+)\S+', rf'\g<1>{tag_prob}', content)
            content = re.sub(r'(/Constant/ochreprob\s+)\S+', rf'\g<1>{taa_prob}', content)
            content = re.sub(r'(/Constant/opalprob\s+)\S+', rf'\g<1>{tga_prob}', content)
            
            with open(self.params_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Updated stop codon probs: TAG={tag_prob}, TAA={taa_prob}, TGA={tga_prob}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.tmp_base.exists():
            shutil.rmtree(self.tmp_base, ignore_errors=True)
    
    def run(self):
        """Run full optimization pipeline"""
        logger.info("=" * 60)
        logger.info("AUGUSTUS Parameter Optimization (Python/Optuna)")
        logger.info("=" * 60)
        
        self.setup()
        
        try:
            if OPTUNA_AVAILABLE:
                best_params = self.optimize_with_optuna()
            else:
                best_params = self.optimize_grid_search()
            
            self.apply_best_params()
            self.update_stop_codon_probs()
            
            # Save results
            results_file = self.output_dir / 'optimization_results.txt'
            with open(results_file, 'w') as f:
                f.write(f"Best accuracy: {self.best_accuracy:.4f}\n\n")
                f.write("Best parameters:\n")
                for param, value in sorted(best_params.items()):
                    f.write(f"  {param}: {value}\n")
            
            logger.info(f"Results saved to {results_file}")
            logger.info(f"Final best accuracy: {self.best_accuracy:.4f}")
            
        finally:
            self.cleanup()
        
        return self.best_accuracy, self.best_params


def main():
    parser = argparse.ArgumentParser(
        description='Fast AUGUSTUS parameter optimization using Bayesian Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s species_name train.gb
    %(prog)s --cpu 32 --n-trials 200 --use-memory species_name train.gb
    %(prog)s --onlytrain onlytrain.gb --output-dir ./results species_name train.gb
        """
    )
    
    parser.add_argument('species', help='AUGUSTUS species name')
    parser.add_argument('train_gb', help='GenBank file with training gene models')
    
    parser.add_argument('--onlytrain', help='GenBank file with genes only for training (not testing)')
    parser.add_argument('--augustus-config-path', help='AUGUSTUS config path (default: $AUGUSTUS_CONFIG_PATH)')
    parser.add_argument('--cpu', type=int, default=8, help='Number of CPUs (default: 8)')
    parser.add_argument('--n-trials', type=int, default=100, 
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Ratio of genes for testing (default: 0.2)')
    parser.add_argument('--min-test-genes', type=int, default=100,
                        help='Minimum genes for testing (default: 100)')
    parser.add_argument('--max-test-genes', type=int, default=600,
                        help='Maximum genes for testing (default: 600)')
    parser.add_argument('--genes-per-test', type=int, default=50,
                        help='Genes per test batch (default: 50)')
    parser.add_argument('--min-intron-len', type=int, default=30,
                        help='Minimum intron length (default: 30)')
    parser.add_argument('--use-memory', action='store_true',
                        help='Use /dev/shm for temp files (faster but uses RAM)')
    parser.add_argument('--output-dir', default='optimize_output',
                        help='Output directory (default: optimize_output)')
    
    args = parser.parse_args()
    
    optimizer = AugustusOptimizer(
        species_name=args.species,
        train_gb=args.train_gb,
        augustus_config_path=args.augustus_config_path,
        onlytrain_gb=args.onlytrain,
        cpu=args.cpu,
        n_trials=args.n_trials,
        test_ratio=args.test_ratio,
        min_test_genes=args.min_test_genes,
        max_test_genes=args.max_test_genes,
        genes_per_test=args.genes_per_test,
        min_intron_len=args.min_intron_len,
        use_memory=args.use_memory,
        output_dir=args.output_dir
    )
    
    accuracy, best_params = optimizer.run()
    
    print(f"\nOptimization complete!")
    print(f"Best accuracy: {accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
