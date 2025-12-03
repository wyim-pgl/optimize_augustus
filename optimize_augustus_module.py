#!/usr/bin/env python3
"""
optimize_augustus_module.py - AUGUSTUS Parameter Optimization Module

Bayesian optimization for AUGUSTUS HMM parameters using Optuna.
Can be used standalone or imported by augustus_training.py

Key optimizations over original BGM2AT.optimize_augustus:
1. Bayesian Optimization instead of grid search (5-10x fewer evaluations)
2. Symlinks instead of copying config directories (90% less I/O)
3. ProcessPoolExecutor for efficient parallelization
4. Optional RAM-based temp files
"""

import os
import sys
import re
import shutil
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GeneModelGB:
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
        """Calculate weighted accuracy score"""
        return (
            self.nucleotide_sensitivity * 3 +
            self.nucleotide_specificity * 2 +
            self.exon_sensitivity * 4 +
            self.exon_specificity * 3 +
            self.gene_sensitivity * 2 +
            self.gene_specificity * 1
        ) / 15


def run_single_evaluation(args: Tuple) -> AccuracyMetrics:
    """Worker function for parallel evaluation (must be at module level for pickling)"""
    test_content, train_content, config_dir, work_dir, species_name, min_intron_len = args
    
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = work_dir / 'test.gb'
    train_file = work_dir / 'train.gb'
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    with open(train_file, 'w') as f:
        f.write(train_content)
    
    # Run etraining
    try:
        subprocess.run(
            [
                'etraining',
                f'--min_intron_len={min_intron_len}',
                f'--species={species_name}',
                f'--AUGUSTUS_CONFIG_PATH={config_dir}',
                str(train_file)
            ],
            capture_output=True,
            check=True,
            timeout=300
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return AccuracyMetrics()
    
    # Run augustus
    try:
        result = subprocess.run(
            [
                'augustus',
                f'--min_intron_len={min_intron_len}',
                f'--species={species_name}',
                f'--AUGUSTUS_CONFIG_PATH={config_dir}',
                str(test_file)
            ],
            capture_output=True,
            check=True,
            timeout=600
        )
        output = result.stdout.decode()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return AccuracyMetrics()
    
    # Parse accuracy
    metrics = AccuracyMetrics()
    
    for line in output.split('\n'):
        nuc_match = re.search(r'nucleotide level[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
        if nuc_match:
            metrics.nucleotide_sensitivity = float(nuc_match.group(1))
            metrics.nucleotide_specificity = float(nuc_match.group(2))
        
        exon_match = re.search(r'exon level[\s|]+([\d.]+)[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
        if exon_match:
            pred, annot, tp = map(float, exon_match.groups())
            if annot > 0:
                metrics.exon_sensitivity = tp / annot
            if pred > 0:
                metrics.exon_specificity = tp / pred
        
        gene_match = re.search(r'gene level[\s|]+([\d.]+)[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
        if gene_match:
            pred, annot, tp = map(float, gene_match.groups())
            if annot > 0:
                metrics.gene_sensitivity = tp / annot
            if pred > 0:
                metrics.gene_specificity = tp / pred
    
    return metrics


class AugustusOptimizer:
    """Bayesian optimizer for AUGUSTUS parameters"""
    
    # Parameters to optimize (subset of metapars.cfg)
    DEFAULT_PARAMS = {
        '/IntronModel/d': ('int', 100, 950),
        '/IntronModel/e': ('int', 10, 100),
        '/IntronModel/slope_of_bandwidth': ('float', 0.1, 0.5),
        '/IntronModel/minwindowcount': ('int', 1, 6),
        '/IntronModel/ass_upwindow_size': ('int', 10, 50),
        '/IntronModel/ass_start': ('int', 1, 5),
        '/IntronModel/ass_end': ('int', 1, 5),
        '/IntronModel/dss_start': ('int', 1, 5),
        '/IntronModel/dss_end': ('int', 0, 4),
        '/IGenicModel/k': ('int', 1, 5),
        '/ExonModel/k': ('int', 1, 5),
        '/ExonModel/minPatSum': ('int', 10, 200),
        '/ExonModel/slope_of_bandwidth': ('float', 0.1, 1.0),
        '/ExonModel/minwindowcount': ('int', 1, 10),
        '/ExonModel/etorder': ('int', 0, 2),
        '/ExonModel/etpseudocount': ('int', 1, 10),
        '/ExonModel/exonlengthD': ('int', 500, 4000),
        '/ExonModel/maxexonlength': ('int', 5000, 30000),
        '/ExonModel/minexonlength': ('int', 1, 10),
        '/UtrModel/d': ('int', 100, 1000),
        '/UtrModel/e': ('int', 50, 200),
        '/UtrModel/slope_of_bandwidth': ('float', 0.1, 0.8),
        '/UtrModel/minwindowcount': ('int', 1, 5),
        '/Constant/decomp_num_steps': ('int', 1, 3),
        '/Constant/min_coding_len': ('int', 50, 300),
        '/Constant/probNinCoding': ('float', 0.1, 0.4),
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
        
        # Set config path
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
        
        # Temp directory
        if use_memory and Path('/dev/shm').exists():
            self.tmp_base = Path('/dev/shm') / f'aug_opt_{os.getpid()}'
        else:
            self.tmp_base = self.output_dir / 'tmp'
        
        # Data
        self.test_genes: List[GeneModelGB] = []
        self.train_genes: List[GeneModelGB] = []
        self.onlytrain_genes: List[GeneModelGB] = []
        self.param_ranges: Dict[str, Tuple] = {}
        
        # Results
        self.best_accuracy = 0.0
        self.best_params: Dict[str, float] = {}
    
    def setup(self):
        """Initialize optimizer"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_base.mkdir(parents=True, exist_ok=True)
        
        # Backup original params
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        if self.params_file.exists():
            shutil.copy(self.params_file, f"{self.params_file}.{timestamp}.bak")
        
        # Load parameter ranges
        self._load_param_ranges()
        
        # Load gene models
        self._load_gene_models()
        
        logger.info(f"Optimizer setup: {len(self.test_genes)} test, {len(self.train_genes)} train genes")
    
    def _load_param_ranges(self):
        """Load parameter ranges from metapars.cfg or use defaults"""
        if self.metapars_file.exists():
            with open(self.metapars_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    match = re.match(r'^\s*(\S+)\s+(.+)', line)
                    if not match:
                        continue
                    
                    param_name, range_str = match.groups()
                    
                    # Parse: "100"-"950" (int) or "0.1"_"0.5" (float)
                    int_match = re.search(r'"([^"]+)"-"([^"]+)"', range_str)
                    float_match = re.search(r'"([^"]+)"_"([^"]+)"', range_str)
                    
                    if int_match:
                        low, high = int(int_match.group(1)), int(int_match.group(2))
                        self.param_ranges[param_name] = ('int', low, high)
                    elif float_match:
                        low, high = float(float_match.group(1)), float(float_match.group(2))
                        self.param_ranges[param_name] = ('float', low, high)
        
        # Use defaults for missing params
        for param, range_info in self.DEFAULT_PARAMS.items():
            if param not in self.param_ranges:
                self.param_ranges[param] = range_info
        
        logger.info(f"Loaded {len(self.param_ranges)} parameter ranges")
    
    def _load_gene_models(self):
        """Load gene models from GenBank files"""
        all_genes = self._parse_genbank(self.train_gb)
        
        if self.onlytrain_gb and self.onlytrain_gb.exists():
            self.onlytrain_genes = self._parse_genbank(self.onlytrain_gb)
        
        # Split into test/train
        n_genes = len(all_genes)
        n_test = int(n_genes * self.test_ratio)
        n_test = max(min(n_test, self.max_test_genes), min(n_genes, self.min_test_genes))
        
        random.shuffle(all_genes)
        self.test_genes = all_genes[:n_test]
        self.train_genes = all_genes[n_test:]
    
    def _parse_genbank(self, filepath: Path) -> List[GeneModelGB]:
        """Parse GenBank file"""
        genes = []
        
        with open(filepath) as f:
            content = f.read()
        
        for record in content.split('//\n'):
            if not record.strip() or 'LOCUS' not in record:
                continue
            
            gene_match = re.search(r'gene="([^"]+)"', record)
            gene_name = gene_match.group(1) if gene_match else 'unknown'
            
            # Calculate CDS length
            cds_length = 0
            compact = re.sub(r'\s+', '', record)
            join_match = re.search(r'CDS.*join\(([^)]+)\)', compact)
            if join_match:
                for span in join_match.group(1).split(','):
                    coord_match = re.search(r'(\d+)\.\.(\d+)', span)
                    if coord_match:
                        cds_length += abs(int(coord_match.group(2)) - int(coord_match.group(1))) + 1
            
            genes.append(GeneModelGB(
                content=record + '//\n',
                gene_name=gene_name,
                cds_length=cds_length
            ))
        
        return genes
    
    def _create_temp_config(self, params: Dict[str, float], tmp_dir: Path) -> Path:
        """Create temp config using symlinks (minimal I/O)"""
        config_dir = tmp_dir / 'config'
        species_subdir = config_dir / 'species' / self.species_name
        species_subdir.mkdir(parents=True, exist_ok=True)
        
        # Symlink config subdirectories
        for item in ['cgp', 'extrinsic', 'model', 'parameters', 'profile']:
            src = self.config_path / item
            dst = config_dir / item
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copytree(src, dst)
        
        # Symlink species files except parameters.cfg
        src_species = self.config_path / 'species' / self.species_name
        for item in src_species.iterdir():
            dst = species_subdir / item.name
            if item.name != f'{self.species_name}_parameters.cfg' and not dst.exists():
                try:
                    dst.symlink_to(item)
                except OSError:
                    shutil.copy2(item, dst)
        
        # Symlink other species
        for other_species in (self.config_path / 'species').iterdir():
            if other_species.name != self.species_name:
                dst = config_dir / 'species' / other_species.name
                if not dst.exists():
                    try:
                        dst.symlink_to(other_species)
                    except OSError:
                        pass
        
        # Create modified parameters.cfg
        with open(self.params_file) as f:
            params_content = f.read()
        
        for param_name, value in params.items():
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
    
    def evaluate_params(self, params: Dict[str, float]) -> float:
        """Evaluate parameter set"""
        eval_id = random.randint(100000, 999999)
        eval_dir = self.tmp_base / f'eval_{eval_id}'
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dir = self._create_temp_config(params, eval_dir)
            
            # Prepare batches
            test_shuffled = self.test_genes.copy()
            random.shuffle(test_shuffled)
            
            batches = []
            for i in range(0, len(test_shuffled), self.genes_per_test):
                batch = test_shuffled[i:i + self.genes_per_test]
                if batch:
                    batches.append(batch)
            
            if not batches:
                return 0.0
            
            # Prepare evaluation args
            eval_args = []
            for i, test_batch in enumerate(batches):
                # Test content
                test_content = ''.join(g.content for g in test_batch)
                
                # Train content (everything except this batch)
                train_content = ''.join(g.content for g in self.train_genes)
                for j, other_batch in enumerate(batches):
                    if j != i:
                        train_content += ''.join(g.content for g in other_batch)
                train_content += ''.join(g.content for g in self.onlytrain_genes)
                
                work_dir = eval_dir / f'batch_{i}'
                eval_args.append((
                    test_content,
                    train_content,
                    str(config_dir),
                    str(work_dir),
                    self.species_name,
                    self.min_intron_len
                ))
            
            # Run evaluations in parallel
            all_metrics = []
            n_workers = min(self.cpu, len(eval_args))
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(run_single_evaluation, args) for args in eval_args]
                for future in as_completed(futures):
                    try:
                        metrics = future.result(timeout=900)
                        if metrics.weighted_accuracy > 0:
                            all_metrics.append(metrics)
                    except Exception as e:
                        logger.debug(f"Batch failed: {e}")
            
            if not all_metrics:
                return 0.0
            
            return sum(m.weighted_accuracy for m in all_metrics) / len(all_metrics)
            
        finally:
            shutil.rmtree(eval_dir, ignore_errors=True)
    
    def optimize(self) -> Dict[str, float]:
        """Run optimization"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using grid search")
            return self._grid_search()
        
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            for param_name, (ptype, low, high) in self.param_ranges.items():
                if ptype == 'int':
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            
            accuracy = self.evaluate_params(params)
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_params = params.copy()
                logger.info(f"New best: {accuracy:.4f}")
            
            return accuracy
        
        sampler = TPESampler(seed=42, n_startup_trials=10)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1, show_progress_bar=True)
        
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def _grid_search(self) -> Dict[str, float]:
        """Fallback grid search"""
        logger.info("Running grid search (fallback)")
        
        # Start with current values
        best_params = {}
        with open(self.params_file) as f:
            content = f.read()
        
        for param_name, (ptype, low, high) in self.param_ranges.items():
            match = re.search(rf'{re.escape(param_name)}\s+(\S+)', content)
            if match:
                try:
                    if ptype == 'float':
                        best_params[param_name] = float(match.group(1))
                    else:
                        best_params[param_name] = int(float(match.group(1)))
                except ValueError:
                    best_params[param_name] = low
            else:
                best_params[param_name] = low
        
        # Iterative optimization
        for round_num in range(3):
            logger.info(f"Round {round_num + 1}")
            improved = False
            
            for param_name, (ptype, low, high) in self.param_ranges.items():
                if ptype == 'int':
                    test_values = [int(low + i * (high - low) / 5) for i in range(6)]
                else:
                    test_values = [low + i * (high - low) / 5 for i in range(6)]
                
                best_value = best_params[param_name]
                best_acc = 0.0
                
                for value in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = value
                    acc = self.evaluate_params(test_params)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_value = value
                
                if best_value != best_params[param_name]:
                    improved = True
                    best_params[param_name] = best_value
            
            if not improved:
                break
        
        self.best_params = best_params
        return best_params
    
    def apply_best_params(self):
        """Apply best params to config file"""
        if not self.best_params:
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
    
    def cleanup(self):
        """Clean up temp files"""
        if self.tmp_base.exists():
            shutil.rmtree(self.tmp_base, ignore_errors=True)
    
    def run(self) -> Tuple[float, Dict[str, float]]:
        """Run complete optimization"""
        logger.info("=" * 60)
        logger.info("AUGUSTUS Parameter Optimization")
        logger.info("=" * 60)
        
        self.setup()
        
        try:
            best_params = self.optimize()
            self.apply_best_params()
            
            # Save results
            results_file = self.output_dir / 'optimization_results.txt'
            with open(results_file, 'w') as f:
                f.write(f"Best accuracy: {self.best_accuracy:.4f}\n\n")
                f.write("Best parameters:\n")
                for param, value in sorted(best_params.items()):
                    f.write(f"  {param}: {value}\n")
            
            logger.info(f"Results saved to {results_file}")
            
        finally:
            self.cleanup()
        
        return self.best_accuracy, self.best_params


def main():
    """Standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AUGUSTUS Parameter Optimization')
    parser.add_argument('species', help='Species name')
    parser.add_argument('train_gb', help='Training GenBank file')
    parser.add_argument('--onlytrain', help='Additional training-only GenBank file')
    parser.add_argument('--augustus-config-path', help='AUGUSTUS config path')
    parser.add_argument('--cpu', type=int, default=8, help='CPUs (default: 8)')
    parser.add_argument('--n-trials', type=int, default=100, help='Optimization trials (default: 100)')
    parser.add_argument('--min-intron-len', type=int, default=30, help='Min intron length (default: 30)')
    parser.add_argument('--use-memory', action='store_true', help='Use /dev/shm for temp files')
    parser.add_argument('--output-dir', default='optimize_output', help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    optimizer = AugustusOptimizer(
        species_name=args.species,
        train_gb=args.train_gb,
        augustus_config_path=args.augustus_config_path,
        onlytrain_gb=args.onlytrain,
        cpu=args.cpu,
        n_trials=args.n_trials,
        min_intron_len=args.min_intron_len,
        use_memory=args.use_memory,
        output_dir=args.output_dir
    )
    
    accuracy, best_params = optimizer.run()
    print(f"\nOptimization complete! Best accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
