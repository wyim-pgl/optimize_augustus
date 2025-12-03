#!/usr/bin/env python3
"""
augustus_training.py - Complete AUGUSTUS HMM Training Pipeline

A Python reimplementation of BGM2AT from the GETA pipeline.
Standalone pipeline for training AUGUSTUS gene prediction models.

Features:
- No external Perl dependencies (gff2gbSmallDNA.pl, filterGenes.pl, etc.)
- Bayesian optimization for parameter tuning (Optuna)
- Efficient parallelization and minimal disk I/O
- Comprehensive logging and checkpointing

Author: Rewritten for Won's lab
Original: chenlianfu/geta BGM2AT
"""

import argparse
import os
import sys
import re
import shutil
import subprocess
import random
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Setup logging
def setup_logging(log_file: Path):
    """Configure logging to both file and stderr"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr)
        ]
    )
    return logging.getLogger(__name__)


@dataclass
class GeneModel:
    """Represents a gene model from GFF3"""
    gene_id: str
    seqid: str
    start: int
    end: int
    strand: str
    cds_regions: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def cds_length(self) -> int:
        return sum(end - start + 1 for start, end in self.cds_regions)
    
    @property
    def gene_length(self) -> int:
        return self.end - self.start + 1


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
    
    def __str__(self):
        return (
            f"Level          Sensitivity    Specificity\n"
            f"nucleotide     {self.nucleotide_sensitivity:.3f}          {self.nucleotide_specificity:.3f}\n"
            f"exon           {self.exon_sensitivity:.3f}          {self.exon_specificity:.3f}\n"
            f"gene           {self.gene_sensitivity:.3f}          {self.gene_specificity:.3f}\n"
            f"Weighted Accuracy: {self.weighted_accuracy:.4f}"
        )


try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not installed. Install with: pip install biopython", file=sys.stderr)


# Globals for the ProcessPoolExecutor worker
_worker_fasta_reader: Optional['FastaReader'] = None
_worker_gb_writer: Optional['GenBankWriter'] = None


def _initialize_gb_worker(fasta_path: Path, flanking_length: int):
    """
    Initializer for the GenBank conversion worker process.
    Creates a single FastaReader and GenBankWriter instance per worker.
    """
    global _worker_fasta_reader, _worker_gb_writer
    # Each worker process gets its own reader and writer instance
    _worker_fasta_reader = FastaReader(fasta_path)
    _worker_gb_writer = GenBankWriter(_worker_fasta_reader, flanking_length)


def _run_gb_conversion(gene: GeneModel) -> Tuple[str, str]:
    """
    The actual worker function that converts a gene to GenBank format.
    Relies on the globally initialized writer in each worker process.
    """
    global _worker_gb_writer
    if _worker_gb_writer is None:
        # This should not happen if the initializer is called correctly
        return (gene.gene_id, "")
    
    try:
        gb_string = _worker_gb_writer.gene_to_genbank(gene)
        return (gene.gene_id, gb_string)
    except Exception as e:
        # Log the error and return an empty string
        logging.error(f"Failed to convert gene {gene.gene_id} in worker: {e}")
        return (gene.gene_id, "")


class FastaReader:
    """FASTA file reader with lazy loading (memory efficient)

    Builds an index on first read, then loads only requested subsequences.
    Avoids loading entire genome into memory.
    """

    def __init__(self, fasta_path: Path):
        self.fasta_path = fasta_path
        self.index: Dict[str, Tuple[int, int]] = {}  # seqid -> (length, header_line_number)
        self._seq_cache: Dict[str, str] = {}  # Cache for full sequences (per-worker)
        self._build_index()

    def _build_index(self):
        """Build index by reading file once, storing seqid and position"""
        with open(self.fasta_path) as f:
            for line_num, line in enumerate(f):
                if line.startswith('>'):
                    seqid = line[1:].split()[0]
                    self.index[seqid] = (0, line_num)  # Will be updated when sequence ends

    def _ensure_sequence_loaded(self, seqid: str):
        """Load sequence into cache if not already loaded"""
        if seqid in self._seq_cache:
            return

        if seqid not in self.index:
            raise KeyError(f"Sequence {seqid} not found in FASTA")

        # Load this specific sequence from file
        seq_lines = []
        found = False
        with open(self.fasta_path) as f:
            for line in f:
                if line.startswith(f'>{seqid}'):
                    found = True
                    continue
                if found:
                    if line.startswith('>'):
                        break
                    seq_lines.append(line.rstrip('\n\r'))

        if found:
            self._seq_cache[seqid] = ''.join(seq_lines)
        else:
            raise KeyError(f"Sequence {seqid} not found in FASTA")

    def get_sequence(self, seqid: str, start: int = None, end: int = None) -> str:
        """Get sequence or subsequence (1-based coordinates)"""
        # Ensure sequence is loaded
        self._ensure_sequence_loaded(seqid)

        seq = self._seq_cache[seqid]
        seq_len = len(seq)

        # Handle defaults
        if start is None:
            start = 1
        if end is None:
            end = seq_len

        # Validate coordinates
        if start < 1 or end > seq_len or start > end:
            raise ValueError(f"Invalid coordinates: {start}-{end} for sequence of length {seq_len}")

        # Return substring (convert to 0-based)
        return seq[start-1:end]

    def get_seq_length(self, seqid: str) -> int:
        """Get sequence length (loads sequence if needed)"""
        self._ensure_sequence_loaded(seqid)
        return len(self._seq_cache[seqid])


class GFF3Parser:
    """Parse GFF3 files to extract gene models"""
    
    @staticmethod
    def parse(gff3_path: Path) -> List[GeneModel]:
        """Parse GFF3 file and return list of gene models"""
        genes: Dict[str, GeneModel] = {}
        gene_to_mrna: Dict[str, List[str]] = defaultdict(list)
        mrna_to_cds: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        mrna_info: Dict[str, dict] = {}
        
        with open(gff3_path) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                seqid, source, feature, start, end, score, strand, phase, attributes = parts
                start, end = int(start), int(end)
                
                # Parse attributes
                attrs = {}
                for attr in attributes.split(';'):
                    if '=' in attr:
                        key, value = attr.split('=', 1)
                        attrs[key] = value
                
                if feature == 'gene':
                    gene_id = attrs.get('ID', f'gene_{len(genes)}')
                    genes[gene_id] = GeneModel(
                        gene_id=gene_id,
                        seqid=seqid,
                        start=start,
                        end=end,
                        strand=strand
                    )
                
                elif feature == 'mRNA' or feature == 'transcript':
                    mrna_id = attrs.get('ID', '')
                    parent = attrs.get('Parent', '')
                    if parent and mrna_id:
                        gene_to_mrna[parent].append(mrna_id)
                        mrna_info[mrna_id] = {'seqid': seqid, 'start': start, 'end': end, 'strand': strand}
                
                elif feature == 'CDS':
                    parent = attrs.get('Parent', '')
                    if parent:
                        # Handle multiple parents
                        for p in parent.split(','):
                            mrna_to_cds[p].append((start, end))
        
        # Assign CDS to genes (use first/longest mRNA)
        for gene_id, gene in genes.items():
            mrnas = gene_to_mrna.get(gene_id, [])
            if mrnas:
                # Find mRNA with most CDS
                best_mrna = max(mrnas, key=lambda m: len(mrna_to_cds.get(m, [])))
                cds_list = mrna_to_cds.get(best_mrna, [])
                gene.cds_regions = sorted(cds_list, key=lambda x: x[0])
        
        # Filter genes with CDS
        valid_genes = [g for g in genes.values() if g.cds_regions]
        
        return valid_genes


class GenBankWriter:
    """Convert gene models to GenBank format using BioPython"""
    
    def __init__(self, fasta_reader: FastaReader, flanking_length: int = 1000):
        self.fasta = fasta_reader
        self.flanking_length = flanking_length
    
    def gene_to_genbank(self, gene: GeneModel) -> str:
        """Convert a single gene model to GenBank format using BioPython"""
        if not BIOPYTHON_AVAILABLE:
            return self._gene_to_genbank_fallback(gene)
        
        # Calculate region with flanking
        seq_length = self.fasta.get_seq_length(gene.seqid)
        region_start = max(1, gene.start - self.flanking_length)
        region_end = min(seq_length, gene.end + self.flanking_length)
        
        # Get sequence
        sequence = self.fasta.get_sequence(gene.seqid, region_start, region_end)
        
        # Create SeqRecord
        locus_name = f"{gene.seqid}_{region_start}_{region_end}"[:16]
        record = SeqRecord(
            Seq(sequence),
            id=locus_name,
            name=locus_name,
            description="",
            annotations={"molecule_type": "DNA"}
        )
        
        # Adjust coordinates relative to extracted region (0-based for BioPython)
        offset = region_start - 1
        
        # Add source feature
        source_feature = SeqFeature(
            FeatureLocation(0, len(sequence)),
            type="source",
            qualifiers={"organism": ["unknown"], "mol_type": ["genomic DNA"]}
        )
        record.features.append(source_feature)
        
        # Add gene feature
        gene_start_adj = gene.start - offset - 1  # 0-based
        gene_end_adj = gene.end - offset
        strand = -1 if gene.strand == '-' else 1
        
        gene_feature = SeqFeature(
            FeatureLocation(gene_start_adj, gene_end_adj, strand=strand),
            type="gene",
            qualifiers={"gene": [gene.gene_id]}
        )
        record.features.append(gene_feature)
        
        # Add CDS feature (compound location for multiple exons)
        if len(gene.cds_regions) > 1:
            locations = []
            for cds_start, cds_end in gene.cds_regions:
                adj_start = cds_start - offset - 1  # 0-based
                adj_end = cds_end - offset
                locations.append(FeatureLocation(adj_start, adj_end, strand=strand))
            
            if strand == -1:
                locations = locations[::-1]  # Reverse for minus strand
            cds_location = CompoundLocation(locations)
        else:
            cds_start, cds_end = gene.cds_regions[0]
            adj_start = cds_start - offset - 1
            adj_end = cds_end - offset
            cds_location = FeatureLocation(adj_start, adj_end, strand=strand)
        
        cds_feature = SeqFeature(
            cds_location,
            type="CDS",
            qualifiers={"gene": [gene.gene_id]}
        )
        record.features.append(cds_feature)
        
        # Write to string
        from io import StringIO
        output = StringIO()
        SeqIO.write(record, output, "genbank")
        return output.getvalue()
    
    def _gene_to_genbank_fallback(self, gene: GeneModel) -> str:
        """Fallback GenBank writer without BioPython"""
        seq_length = len(self.fasta.sequences.get(gene.seqid, ''))
        region_start = max(1, gene.start - self.flanking_length)
        region_end = min(seq_length, gene.end + self.flanking_length)
        sequence = self.fasta.get_sequence(gene.seqid, region_start, region_end)
        offset = region_start - 1
        
        cds_parts = []
        for cds_start, cds_end in gene.cds_regions:
            adj_start = cds_start - offset
            adj_end = cds_end - offset
            cds_parts.append(f"{adj_start}..{adj_end}")
        
        if gene.strand == '-':
            cds_location = f"complement(join({','.join(cds_parts)}))"
        else:
            cds_location = f"join({','.join(cds_parts)})"
        
        gb_lines = []
        locus_name = f"{gene.seqid}_{region_start}_{region_end}"[:16]
        gb_lines.append(f"LOCUS       {locus_name:16} {len(sequence):>6} bp    DNA")
        gb_lines.append("FEATURES             Location/Qualifiers")
        gb_lines.append(f"     source          1..{len(sequence)}")
        gb_lines.append(f'                     /organism="unknown"')
        
        gene_start = gene.start - offset
        gene_end = gene.end - offset
        if gene.strand == '-':
            gb_lines.append(f"     gene            complement({gene_start}..{gene_end})")
        else:
            gb_lines.append(f"     gene            {gene_start}..{gene_end}")
        gb_lines.append(f'                     /gene="{gene.gene_id}"')
        gb_lines.append(f"     CDS             {cds_location}")
        gb_lines.append(f'                     /gene="{gene.gene_id}"')
        gb_lines.append("ORIGIN")
        
        for i in range(0, len(sequence), 60):
            chunk = sequence[i:i+60].lower()
            groups = [chunk[j:j+10] for j in range(0, len(chunk), 10)]
            gb_lines.append(f"{i+1:>9} {' '.join(groups)}")
        
        gb_lines.append("//")
        return '\n'.join(gb_lines)
    
    def write_genbank(self, genes: List[GeneModel], output_path: Path):
        """Write multiple gene models to GenBank file"""
        if BIOPYTHON_AVAILABLE:
            records = []
            for gene in genes:
                try:
                    # Parse back to SeqRecord for batch writing
                    from io import StringIO
                    gb_str = self.gene_to_genbank(gene)
                    record = SeqIO.read(StringIO(gb_str), "genbank")
                    records.append(record)
                except Exception as e:
                    logging.warning(f"Failed to convert gene {gene.gene_id}: {e}")
            
            SeqIO.write(records, output_path, "genbank")
        else:
            with open(output_path, 'w') as f:
                for gene in genes:
                    try:
                        gb_entry = self.gene_to_genbank(gene)
                        f.write(gb_entry + '\n')
                    except Exception as e:
                        logging.warning(f"Failed to convert gene {gene.gene_id}: {e}")


class AugustusTrainer:
    """Main AUGUSTUS training pipeline"""
    
    def __init__(
        self,
        gff3_file: Path,
        genome_file: Path,
        species_name: str,
        output_dir: Path,
        augustus_config_path: Optional[Path] = None,
        onlytrain_gff3: Optional[Path] = None,
        augustus_species_start_from: Optional[str] = None,
        flanking_length: Optional[int] = None,
        min_gene_number: int = 500,
        test_gene_number: int = 300,
        cpu: int = 8,
        n_trials: int = 100,
        min_intron_len: int = 30,
        use_memory: bool = False,
        optimize_method: int = 3,
        stop_after_first: bool = False,
        start_codons: str = 'ATG'
    ):
        self.gff3_file = Path(gff3_file).resolve()
        self.genome_file = Path(genome_file).resolve()
        self.species_name = species_name
        self.output_dir = Path(output_dir).resolve()
        self.onlytrain_gff3 = Path(onlytrain_gff3).resolve() if onlytrain_gff3 else None
        self.augustus_species_start_from = augustus_species_start_from
        self.flanking_length = flanking_length
        self.min_gene_number = min_gene_number
        self.test_gene_number = test_gene_number
        self.cpu = cpu
        self.n_trials = n_trials
        self.min_intron_len = min_intron_len
        self.use_memory = use_memory
        self.optimize_method = optimize_method
        self.stop_after_first = stop_after_first
        self.start_codons = [c.strip().upper() for c in start_codons.split(',') if c.strip()]
        
        # Setup AUGUSTUS config path
        if augustus_config_path:
            self.config_path = Path(augustus_config_path).resolve()
        else:
            env_path = os.environ.get('AUGUSTUS_CONFIG_PATH', '')
            if env_path:
                self.config_path = Path(env_path)
            else:
                raise ValueError("AUGUSTUS_CONFIG_PATH not set")
        
        self.config_path_orig = Path(os.environ.get('AUGUSTUS_CONFIG_PATH', self.config_path))
        
        # Will be set during setup
        self.logger = None
        self.fasta_reader = None
        self.genes: List[GeneModel] = []
        self.onlytrain_genes: List[GeneModel] = []
        self.train_genes: List[GeneModel] = []
        self.test_genes: List[GeneModel] = []
        
        # GenBank content cache (minimal - only keep what we need)
        # Avoid loading all genes into memory at once
        self.gb_gene_ids: Set[str] = set()  # Track which genes have valid GB files
        
    def setup(self):
        """Initialize the training environment"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / 'augustus_training.log'
        self.logger = setup_logging(log_file)
        
        self.logger.info("=" * 70)
        self.logger.info("AUGUSTUS HMM Training Pipeline (Python)")
        self.logger.info("=" * 70)
        self.logger.info(f"Species: {self.species_name}")
        self.logger.info(f"GFF3: {self.gff3_file}")
        self.logger.info(f"Genome: {self.genome_file}")
        self.logger.info(f"Output: {self.output_dir}")
        
        # Load genome
        self.logger.info("Loading genome FASTA...")
        self.fasta_reader = FastaReader(self.genome_file)
        self.logger.info(f"Loaded {len(self.fasta_reader.sequences)} sequences")
        
        # Parse GFF3
        self.logger.info("Parsing GFF3 file...")
        self.genes = GFF3Parser.parse(self.gff3_file)
        self.logger.info(f"Found {len(self.genes)} gene models with CDS")
        
        if self.onlytrain_gff3:
            self.logger.info("Parsing onlytrain GFF3...")
            self.onlytrain_genes = GFF3Parser.parse(self.onlytrain_gff3)
            # Remove genes that are in main training set
            main_ids = {g.gene_id for g in self.genes}
            self.onlytrain_genes = [g for g in self.onlytrain_genes if g.gene_id not in main_ids]
            self.logger.info(f"Found {len(self.onlytrain_genes)} additional onlytrain genes")
        
        # Calculate flanking length if not specified
        if self.flanking_length is None:
            gene_lengths = sorted([g.gene_length for g in self.genes])
            self.flanking_length = gene_lengths[len(gene_lengths) // 2]
            self.logger.info(f"Auto-calculated flanking length: {self.flanking_length}")
        
        # Save flanking length
        with open(self.output_dir / 'flanking_length.txt', 'w') as f:
            f.write(str(self.flanking_length))
    
    def _check_ok_file(self, step_name: str) -> bool:
        """Check if a step has already completed"""
        ok_file = self.output_dir / f'{step_name}.ok'
        return ok_file.exists()
    
    def _mark_ok(self, step_name: str):
        """Mark a step as completed"""
        ok_file = self.output_dir / f'{step_name}.ok'
        ok_file.touch()
    
    def step1_prepare_config(self):
        """Step 1: Prepare AUGUSTUS configuration files"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 1: Prepare AUGUSTUS configuration")
        self.logger.info("=" * 50)
        
        if self._check_ok_file('step1'):
            self.logger.info("Step 1 already completed, skipping...")
            return
        
        # Create species directory if needed
        species_dir = self.config_path / 'species' / self.species_name
        
        if self.config_path != self.config_path_orig:
            # Copy necessary config directories
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            for subdir in ['species', 'cgp', 'extrinsic', 'model', 'parameters', 'profile']:
                src = self.config_path_orig / subdir
                dst = self.config_path / subdir
                if src.exists() and not dst.exists():
                    if subdir == 'species':
                        dst.mkdir(parents=True, exist_ok=True)
                        # Copy generic species
                        generic_src = src / 'generic'
                        generic_dst = dst / 'generic'
                        if generic_src.exists() and not generic_dst.exists():
                            shutil.copytree(generic_src, generic_dst)
                    else:
                        shutil.copytree(src, dst)
                    self.logger.info(f"Copied {subdir} to {dst}")
        
        # Create new species if not exists
        if not species_dir.exists():
            cmd = [
                'new_species.pl',
                f'--species={self.species_name}',
                f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
                '--ignore'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning(f"new_species.pl warning: {result.stderr}")
            self.logger.info(f"Created new species: {self.species_name}")
        
        # Copy parameters from starting species if specified
        if self.augustus_species_start_from:
            src_params = self.config_path_orig / 'species' / self.augustus_species_start_from / f'{self.augustus_species_start_from}_parameters.cfg'
            dst_params = species_dir / f'{self.species_name}_parameters.cfg'
            
            if src_params.exists():
                with open(src_params) as f:
                    content = f.read()
                content = content.replace(self.augustus_species_start_from, self.species_name)
                with open(dst_params, 'w') as f:
                    f.write(content)
                self.logger.info(f"Copied parameters from {self.augustus_species_start_from}")
        
        os.environ['AUGUSTUS_CONFIG_PATH'] = str(self.config_path)
        self._mark_ok('step1')
    
    def step2_convert_to_genbank(self):
        """Step 2: Convert GFF3 to GenBank format (in parallel)

        Optimization: Stream results directly to file instead of caching in memory.
        This avoids loading all genes into self.gb_content.
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 2: Convert GFF3 to GenBank format (in parallel)")
        self.logger.info("=" * 50)

        if self._check_ok_file('step2'):
            self.logger.info("Step 2 already completed, skipping...")
            self._mark_gene_ids_from_files()
            return

        all_genes_to_process = self.genes + self.onlytrain_genes
        total_genes = len(all_genes_to_process)
        self.logger.info(f"Converting {total_genes} total genes to GenBank format using {self.cpu} workers...")

        # Output files
        genes_gb_path = self.output_dir / 'genes.raw.gb'
        onlytrain_gb_path = self.output_dir / 'genes.onlytrain.gb'

        # Create mapping of gene_id -> file path for output
        gene_output_files = {}
        for gene in self.genes:
            gene_output_files[gene.gene_id] = genes_gb_path
        for gene in self.onlytrain_genes:
            gene_output_files[gene.gene_id] = onlytrain_gb_path

        # Use a process pool to parallelize the conversion
        # Stream results directly to files (no intermediate caching)
        with ProcessPoolExecutor(
            max_workers=self.cpu,
            initializer=_initialize_gb_worker,
            initargs=(self.genome_file, self.flanking_length)
        ) as executor:

            futures = {executor.submit(_run_gb_conversion, gene): gene.gene_id
                      for gene in all_genes_to_process}

            # Open output files for streaming writes
            file_handles = {
                genes_gb_path: open(genes_gb_path, 'w'),
                onlytrain_gb_path: open(onlytrain_gb_path, 'w')
            }

            try:
                processed_count = 0
                for future in as_completed(futures):
                    gene_id = futures[future]
                    try:
                        _, gb_string = future.result()
                        if gb_string and gene_id in gene_output_files:
                            # Stream directly to file (no memory accumulation)
                            output_file = gene_output_files[gene_id]
                            file_handles[output_file].write(gb_string)
                            self.gb_gene_ids.add(gene_id)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert gene {gene_id}: {e}")

                    processed_count += 1
                    if processed_count % 100 == 0 or processed_count == total_genes:
                        self.logger.info(f"Converted {processed_count}/{total_genes} genes...")
            finally:
                # Close output files
                for f in file_handles.values():
                    f.close()

        self.logger.info("GenBank files written.")
        self._mark_ok('step2')
    
    def _mark_gene_ids_from_files(self):
        """Mark which genes have valid GB files (without loading into memory)"""
        genes_gb = self.output_dir / 'genes.raw.gb'
        if genes_gb.exists():
            self._scan_genbank_gene_ids(genes_gb)

        onlytrain_gb = self.output_dir / 'genes.onlytrain.gb'
        if onlytrain_gb.exists():
            self._scan_genbank_gene_ids(onlytrain_gb)

    def _scan_genbank_gene_ids(self, gb_file: Path):
        """Scan GenBank file to find gene IDs without loading entire file into memory"""
        with open(gb_file) as f:
            for line in f:
                match = re.search(r'/gene="([^"]+)"', line)
                if match:
                    self.gb_gene_ids.add(match.group(1))

    def _copy_filtered_genbank(self, in_file, out_file, valid_genes: List[GeneModel]):
        """Copy GenBank records for valid genes only (streaming, no full file load)

        Reads GenBank file line by line, copies records matching valid gene IDs.
        """
        valid_gene_ids = {g.gene_id for g in valid_genes}
        current_gene_id = None
        in_record = False

        for line in in_file:
            # Detect gene ID at start of record
            if '/gene="' in line:
                match = re.search(r'/gene="([^"]+)"', line)
                if match:
                    current_gene_id = match.group(1)
                    in_record = current_gene_id in valid_gene_ids

            # Write line if this record is valid
            if in_record:
                out_file.write(line)

            # Detect end of record
            if line.strip() == '//':
                in_record = False
                current_gene_id = None
    
    def step3_filter_bad_genes(self):
        """Step 3: Remove incorrect gene models using etraining"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 3: Filter bad gene models")
        self.logger.info("=" * 50)
        
        if self._check_ok_file('step3'):
            self.logger.info("Step 3 already completed, skipping...")
            self._load_filtered_genes()
            return
        
        # Create temporary species for validation
        temp_species = f'temp_validate_{os.getpid()}'
        temp_species_dir = self.config_path / 'species' / temp_species
        
        try:
            # Create temp species
            subprocess.run([
                'new_species.pl',
                f'--species={temp_species}',
                f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
                '--ignore'
            ], capture_output=True)
            
            # Run etraining to find bad genes
            genes_gb = self.output_dir / 'genes.raw.gb'
            etraining_err = self.output_dir / 'etraining.validate.err'
            
            cmd = [
                'etraining',
                f'--min_intron_len={self.min_intron_len}',
                f'--species={temp_species}',
                f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
                '--stopCodonExcludedFromCDS=false',
                str(genes_gb)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            with open(etraining_err, 'w') as f:
                f.write(result.stderr)
            
            # Parse bad genes from stderr
            bad_genes: Set[str] = set()
            for line in result.stderr.split('\n'):
                match = re.search(r'in sequence (\S+):', line)
                if match:
                    bad_genes.add(match.group(1))
            
            self.logger.info(f"Found {len(bad_genes)} problematic gene models")
            
            # Save bad genes list
            with open(self.output_dir / 'badgenes.lst', 'w') as f:
                for gene_id in sorted(bad_genes):
                    f.write(f"{gene_id}\n")
            
            # Filter genes
            self.genes = [g for g in self.genes if g.gene_id not in bad_genes]
            self.logger.info(f"Remaining valid genes: {len(self.genes)}")

            # Write filtered GenBank (copy from source, only valid genes)
            filtered_gb = self.output_dir / 'genes.gb'
            with open(filtered_gb, 'w') as out_f:
                with open(genes_gb) as in_f:
                    self._copy_filtered_genbank(in_f, out_f, self.genes)
            
        finally:
            # Cleanup temp species
            if temp_species_dir.exists():
                shutil.rmtree(temp_species_dir)
        
        self._mark_ok('step3')
    
    def _load_filtered_genes(self):
        """Load filtered genes from existing files"""
        bad_genes_file = self.output_dir / 'badgenes.lst'
        bad_genes = set()
        if bad_genes_file.exists():
            with open(bad_genes_file) as f:
                bad_genes = {line.strip() for line in f if line.strip()}
        
        self.genes = [g for g in self.genes if g.gene_id not in bad_genes]
    
    def step4_split_train_test(self):
        """Step 4: Split genes into training and testing sets"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 4: Split train/test sets")
        self.logger.info("=" * 50)
        
        if self._check_ok_file('step4'):
            self.logger.info("Step 4 already completed, loading existing splits...")
            self._load_train_test_split()
            return
        
        n_genes = len(self.genes)
        
        if n_genes >= self.min_gene_number:
            # Proper split
            n_test = min(self.test_gene_number, n_genes // 2)
            random.shuffle(self.genes)
            self.test_genes = self.genes[:n_test]
            self.train_genes = self.genes[n_test:]
            self.logger.info(f"Split: {len(self.train_genes)} training, {len(self.test_genes)} testing")
        else:
            # Use all for both (warning: unreliable accuracy)
            self.test_genes = self.genes.copy()
            self.train_genes = self.genes.copy()
            self.logger.warning(f"Insufficient genes ({n_genes} < {self.min_gene_number})")
            self.logger.warning("Using all genes for both training and testing (unreliable accuracy)")
        
        # Write split files
        self._write_genbank_list(self.train_genes, self.output_dir / 'genes.gb.train')
        self._write_genbank_list(self.test_genes, self.output_dir / 'genes.gb.test')
        
        # Write etraining file (train + onlytrain, stream from source files)
        etraining_genes = self.train_genes + self.onlytrain_genes
        etraining_file = self.output_dir / 'genes.gb.etraining'
        self._write_genbank_list(etraining_genes, etraining_file)
        
        # Save split info
        split_info = {
            'n_train': len(self.train_genes),
            'n_test': len(self.test_genes),
            'n_onlytrain': len(self.onlytrain_genes),
            'train_ids': [g.gene_id for g in self.train_genes],
            'test_ids': [g.gene_id for g in self.test_genes]
        }
        with open(self.output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self._mark_ok('step4')
    
    def _write_genbank_list(self, genes: List[GeneModel], output_path: Path):
        """Write list of genes to GenBank file (stream from source, no memory cache)"""
        valid_gene_ids = {g.gene_id for g in genes}

        # Source files to read from
        source_files = []
        genes_gb = self.output_dir / 'genes.gb'
        if genes_gb.exists():
            source_files.append(genes_gb)

        onlytrain_gb = self.output_dir / 'genes.onlytrain.gb'
        if onlytrain_gb.exists():
            source_files.append(onlytrain_gb)

        # Stream copy from source files
        with open(output_path, 'w') as out_f:
            for source_file in source_files:
                with open(source_file) as in_f:
                    self._copy_filtered_genbank(in_f, out_f, genes)
    
    def _load_train_test_split(self):
        """Load existing train/test split"""
        split_file = self.output_dir / 'split_info.json'
        if split_file.exists():
            with open(split_file) as f:
                split_info = json.load(f)
            
            train_ids = set(split_info['train_ids'])
            test_ids = set(split_info['test_ids'])
            
            self.train_genes = [g for g in self.genes if g.gene_id in train_ids]
            self.test_genes = [g for g in self.genes if g.gene_id in test_ids]
    
    def step5_first_training(self) -> AccuracyMetrics:
        """Step 5: First etraining and accuracy test"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 5: First AUGUSTUS training")
        self.logger.info("=" * 50)
        
        first_metrics_file = self.output_dir / 'first_accuracy.json'
        
        if self._check_ok_file('step5'):
            self.logger.info("Step 5 already completed, loading results...")
            with open(first_metrics_file) as f:
                data = json.load(f)
            return AccuracyMetrics(**data)
        
        # Run etraining
        etraining_file = self.output_dir / 'genes.gb.etraining'
        etraining_out = self.output_dir / 'etraining.out1'
        
        cmd = [
            'etraining',
            f'--min_intron_len={self.min_intron_len}',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
            str(etraining_file)
        ]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(etraining_out, 'w') as f:
            f.write(result.stdout)
        
        # Update stop codon frequencies
        self._update_stop_codon_freqs(result.stdout)
        
        # Run augustus test
        test_file = self.output_dir / 'genes.gb.test'
        test_out = self.output_dir / 'firsttest.out'
        
        cmd = [
            'augustus',
            f'--min_intron_len={self.min_intron_len}',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
            str(test_file)
        ]
        
        self.logger.info(f"Running accuracy test...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(test_out, 'w') as f:
            f.write(result.stdout)
        
        # Parse accuracy
        metrics = self._parse_accuracy(result.stdout)
        self.logger.info(f"First training accuracy:\n{metrics}")
        
        # Backup HMM files
        hmm_backup = self.output_dir / 'hmm_files_bak01'
        species_dir = self.config_path / 'species' / self.species_name
        if hmm_backup.exists():
            shutil.rmtree(hmm_backup)
        shutil.copytree(species_dir, hmm_backup)
        
        # Save metrics
        with open(first_metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        self._mark_ok('step5')
        return metrics
    
    def _update_stop_codon_freqs(self, etraining_output: str):
        """Update stop codon frequencies in parameters.cfg"""
        freqs = {}
        for line in etraining_output.split('\n'):
            for codon in ['tag', 'taa', 'tga']:
                if line.startswith(f'{codon}:'):
                    match = re.search(r'\(([\d.]+)\)', line)
                    if match:
                        freqs[codon] = float(match.group(1))
        
        if len(freqs) == 3:
            params_file = self.config_path / 'species' / self.species_name / f'{self.species_name}_parameters.cfg'
            
            with open(params_file) as f:
                content = f.read()
            
            content = re.sub(r'(/Constant/amberprob\s+)\S+', rf'\g<1>{freqs["tag"]}', content)
            content = re.sub(r'(/Constant/ochreprob\s+)\S+', rf'\g<1>{freqs["taa"]}', content)
            content = re.sub(r'(/Constant/opalprob\s+)\S+', rf'\g<1>{freqs["tga"]}', content)
            
            with open(params_file, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Updated stop codon freqs: TAG={freqs['tag']:.4f}, TAA={freqs['taa']:.4f}, TGA={freqs['tga']:.4f}")
    
    def _parse_accuracy(self, augustus_output: str) -> AccuracyMetrics:
        """Parse AUGUSTUS output to extract accuracy metrics"""
        metrics = AccuracyMetrics()
        
        for line in augustus_output.split('\n'):
            # nucleotide level |  0.95 |  0.89 |
            nuc_match = re.search(r'nucleotide level[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
            if nuc_match:
                metrics.nucleotide_sensitivity = float(nuc_match.group(1))
                metrics.nucleotide_specificity = float(nuc_match.group(2))
            
            # exon level
            exon_match = re.search(r'exon level[\s|]+([\d.]+)[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
            if exon_match:
                pred, annot, tp = map(float, exon_match.groups())
                if annot > 0:
                    metrics.exon_sensitivity = tp / annot
                if pred > 0:
                    metrics.exon_specificity = tp / pred
            
            # gene level
            gene_match = re.search(r'gene level[\s|]+([\d.]+)[\s|]+([\d.]+)[\s|]+([\d.]+)', line)
            if gene_match:
                pred, annot, tp = map(float, gene_match.groups())
                if annot > 0:
                    metrics.gene_sensitivity = tp / annot
                if pred > 0:
                    metrics.gene_specificity = tp / pred
        
        return metrics
    
    def step6_optimize_parameters(self):
        """Step 6: Optimize AUGUSTUS parameters"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 6: Optimize AUGUSTUS parameters")
        self.logger.info("=" * 50)
        
        if self._check_ok_file('step6'):
            self.logger.info("Step 6 already completed, skipping...")
            return
        
        if self.optimize_method == 0:
            self.logger.info("Parameter optimization disabled (--optimize-method 0)")
            self._mark_ok('step6')
            return
        
        train_file = self.output_dir / 'genes.gb.train'
        onlytrain_file = self.output_dir / 'genes.gb.etraining'
        
        if self.optimize_method in [1, 3]:
            # Use our Python Bayesian optimization
            self.logger.info("Running Bayesian optimization (method 1)...")
            self._run_bayesian_optimization(train_file, onlytrain_file)
        
        if self.optimize_method in [2, 3]:
            # Use AUGUSTUS optimize_augustus.pl
            self.logger.info("Running optimize_augustus.pl (method 2)...")
            self._run_perl_optimization(train_file, onlytrain_file)
        
        self._mark_ok('step6')
    
    def _run_bayesian_optimization(self, train_file: Path, onlytrain_file: Path):
        """Run Bayesian optimization for parameters"""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, skipping Bayesian optimization")
            return
        
        # Import the optimizer module
        from optimize_augustus_module import AugustusOptimizer
        
        optimizer = AugustusOptimizer(
            species_name=self.species_name,
            genes_for_cv_gb=str(train_file),
            augustus_config_path=str(self.config_path),
            onlytrain_gb=str(onlytrain_file) if onlytrain_file.exists() else None,
            cpu=self.cpu,
            n_trials=self.n_trials,
            min_intron_len=self.min_intron_len,
            use_memory=self.use_memory,
            output_dir=str(self.output_dir / 'optimization')
        )
        
        optimizer.run()
    
    def _run_perl_optimization(self, train_file: Path, onlytrain_file: Path):
        """Run AUGUSTUS optimize_augustus.pl"""
        # Prepare files
        n_train = len(self.train_genes)
        n_onlytrain = max(100, n_train - 200)
        
        cmd = [
            'optimize_augustus.pl',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
            '--rounds=3',
            f'--cpus={self.cpu}',
            f'--kfold={min(8, self.cpu)}',
            f'--onlytrain={onlytrain_file}',
            str(train_file)
        ]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        opt_log = self.output_dir / 'optimize_augustus.log'
        with open(opt_log, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        if result.returncode != 0:
            self.logger.warning(f"optimize_augustus.pl finished with return code {result.returncode}")
    
    def step7_second_training(self) -> AccuracyMetrics:
        """Step 7: Second etraining and accuracy test"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 7: Second AUGUSTUS training")
        self.logger.info("=" * 50)
        
        second_metrics_file = self.output_dir / 'second_accuracy.json'
        
        if self._check_ok_file('step7'):
            self.logger.info("Step 7 already completed, loading results...")
            with open(second_metrics_file) as f:
                data = json.load(f)
            return AccuracyMetrics(**data)
        
        # Run etraining
        etraining_file = self.output_dir / 'genes.gb.etraining'
        
        cmd = [
            'etraining',
            f'--min_intron_len={self.min_intron_len}',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
            str(etraining_file)
        ]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(self.output_dir / 'etraining.out2', 'w') as f:
            f.write(result.stdout)
        
        # Set start codons based on user input
        self._set_start_codons()
        
        # Run augustus test
        test_file = self.output_dir / 'genes.gb.test'
        
        cmd = [
            'augustus',
            f'--min_intron_len={self.min_intron_len}',
            f'--species={self.species_name}',
            f'--AUGUSTUS_CONFIG_PATH={self.config_path}',
            str(test_file)
        ]
        
        self.logger.info("Running accuracy test...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(self.output_dir / 'secondtest.out', 'w') as f:
            f.write(result.stdout)
        
        # Parse accuracy
        metrics = self._parse_accuracy(result.stdout)
        self.logger.info(f"Second training accuracy:\n{metrics}")
        
        # Backup HMM files
        hmm_backup = self.output_dir / 'hmm_files_bak02'
        species_dir = self.config_path / 'species' / self.species_name
        if hmm_backup.exists():
            shutil.rmtree(hmm_backup)
        shutil.copytree(species_dir, hmm_backup)
        
        # Save metrics
        with open(second_metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        self._mark_ok('step7')
        return metrics
    
    def _set_start_codons(self):
        """Set start codons in exon_probs.pbl based on user input."""
        exon_probs_file = self.config_path / 'species' / self.species_name / f'{self.species_name}_exon_probs.pbl'
        
        if not exon_probs_file.exists() or not self.start_codons:
            self.logger.warning(f"Cannot set start codons: {exon_probs_file} not found or no codons specified.")
            return
        
        with open(exon_probs_file) as f:
            content = f.read()
        
        # Build the new STARTCODONS section
        num_codons = len(self.start_codons)
        prob = 1.0 / num_codons if num_codons > 0 else 0
        
        new_section_lines = ["[STARTCODONS]", f"# number of start codons:\n{num_codons}", "# start codons and their probabilities"]
        for codon in self.start_codons:
            new_section_lines.append(f"{codon.upper()}\t{prob}")
        
        new_section = "\n".join(new_section_lines) + "\n\n# Length distributions\n[LENGTH]"
        
        # Replace the existing [STARTCODONS] section up to [LENGTH]
        # The re.DOTALL flag allows '.' to match newlines
        content = re.sub(
            r'\[STARTCODONS\].*?\[LENGTH\]',
            new_section,
            content,
            flags=re.DOTALL
        )
        
        with open(exon_probs_file, 'w') as f:
            f.write(content)
        
        self.logger.info(f"Set start codons to: {', '.join(self.start_codons)} with uniform probability.")
    
    def step8_compare_and_finalize(self, first_metrics: AccuracyMetrics, second_metrics: AccuracyMetrics):
        """Step 8: Compare results and finalize"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Step 8: Compare and finalize")
        self.logger.info("=" * 50)
        
        first_acc = first_metrics.weighted_accuracy
        second_acc = second_metrics.weighted_accuracy
        
        self.logger.info(f"First training accuracy:  {first_acc:.4f}")
        self.logger.info(f"Second training accuracy: {second_acc:.4f}")
        
        # Determine which is better
        species_dir = self.config_path / 'species' / self.species_name
        
        if second_acc < first_acc:
            self.logger.warning("Optimization did not improve accuracy!")
            self.logger.info("Rolling back to first training results...")
            
            # Restore from backup
            hmm_backup = self.output_dir / 'hmm_files_bak01'
            if hmm_backup.exists():
                shutil.rmtree(species_dir)
                shutil.copytree(hmm_backup, species_dir)
            
            final_metrics = first_metrics
            final_backup = 'hmm_files_bak01'
        else:
            self.logger.info("Optimization improved accuracy!")
            final_metrics = second_metrics
            final_backup = 'hmm_files_bak02'
        
        # Create symlink to best backup
        best_link = self.output_dir / 'hmm_files_bak'
        if best_link.exists():
            best_link.unlink()
        best_link.symlink_to(final_backup)
        
        # Write final report
        report_file = self.output_dir / 'accuracy_of_AUGUSTUS_HMM_Training.txt'
        with open(report_file, 'w') as f:
            f.write(f"AUGUSTUS HMM Training Results\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Species: {self.species_name}\n")
            f.write(f"Training genes: {len(self.train_genes)}\n")
            f.write(f"Testing genes: {len(self.test_genes)}\n\n")
            f.write(f"Final Accuracy: {final_metrics.weighted_accuracy * 100:.2f}%\n\n")
            f.write(f"{final_metrics}\n\n")
            f.write(f"HMM files location: {species_dir}\n")
        
        self.logger.info(f"\nFinal accuracy: {final_metrics.weighted_accuracy * 100:.2f}%")
        self.logger.info(f"HMM files saved to: {species_dir}")
        self.logger.info(f"Report saved to: {report_file}")
        
        return final_metrics
    
    def run(self):
        """Run the complete training pipeline"""
        self.setup()
        
        # Step 1: Prepare config
        self.step1_prepare_config()
        
        # Step 2: Convert to GenBank
        self.step2_convert_to_genbank()
        
        # Step 3: Filter bad genes
        self.step3_filter_bad_genes()
        
        # Step 4: Split train/test
        self.step4_split_train_test()
        
        # Step 5: First training
        first_metrics = self.step5_first_training()
        
        if self.stop_after_first:
            self.logger.info("Stopping after first training (--stop-after-first)")
            return first_metrics
        
        # Step 6: Optimize parameters
        self.step6_optimize_parameters()
        
        # Step 7: Second training
        second_metrics = self.step7_second_training()
        
        # Step 8: Compare and finalize
        final_metrics = self.step8_compare_and_finalize(first_metrics, second_metrics)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("AUGUSTUS HMM Training completed successfully!")
        self.logger.info("=" * 70)
        
        return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Complete AUGUSTUS HMM Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    %(prog)s genes.gff3 genome.fasta species_name
    
    # With all options
    %(prog)s \\
        --output-dir ./training_output \\
        --cpu 32 \\
        --n-trials 150 \\
        --flanking-length 1000 \\
        --use-memory \\
        genes.gff3 genome.fasta my_species
    
    # Using existing species as starting point
    %(prog)s \\
        --augustus-species-start-from arabidopsis \\
        genes.gff3 genome.fasta new_plant_species
        """
    )
    
    # Required arguments
    parser.add_argument('gff3', help='GFF3 file with gene models for training')
    parser.add_argument('genome', help='Genome FASTA file')
    parser.add_argument('species', help='AUGUSTUS species name to create/update')
    
    # Optional arguments
    parser.add_argument('-o', '--output-dir', default='augustus_training_output',
                        help='Output directory (default: augustus_training_output)')
    parser.add_argument('--augustus-config-path',
                        help='AUGUSTUS config path (default: $AUGUSTUS_CONFIG_PATH)')
    parser.add_argument('--augustus-species-start-from',
                        help='Start from existing species parameters')
    parser.add_argument('--onlytrain-gff3',
                        help='Additional GFF3 with genes only for training (not testing)')
    parser.add_argument('--flanking-length', type=int,
                        help='Flanking length for GenBank conversion (default: auto)')
    parser.add_argument('--min-gene-number', type=int, default=500,
                        help='Minimum genes for proper train/test split (default: 500)')
    parser.add_argument('--test-gene-number', type=int, default=300,
                        help='Number of genes for testing (default: 300)')
    parser.add_argument('--cpu', type=int, default=8,
                        help='Number of CPUs (default: 8)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optimization trials for Bayesian optimization (default: 100)')
    parser.add_argument('--min-intron-len', type=int, default=30,
                        help='Minimum intron length (default: 30)')
    parser.add_argument('--use-memory', action='store_true',
                        help='Use /dev/shm for temp files (faster)')
    parser.add_argument('--optimize-method', type=int, default=1, choices=[0, 1, 2, 3],
                        help='0=none, 1=Bayesian, 2=optimize_augustus.pl, 3=both (default: 1)')
    parser.add_argument('--stop-after-first', action='store_true',
                        help='Stop after first etraining (skip optimization)')
    parser.add_argument('--start-codons', default='ATG',
                        help='Comma-separated list of allowed start codons (e.g., "ATG,CTG,TTG"). '
                             'Probabilities will be distributed uniformly. (default: ATG)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.gff3).exists():
        print(f"Error: GFF3 file not found: {args.gff3}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.genome).exists():
        print(f"Error: Genome file not found: {args.genome}", file=sys.stderr)
        sys.exit(1)
    
    # Run pipeline
    trainer = AugustusTrainer(
        gff3_file=args.gff3,
        genome_file=args.genome,
        species_name=args.species,
        output_dir=args.output_dir,
        augustus_config_path=args.augustus_config_path,
        onlytrain_gff3=args.onlytrain_gff3,
        augustus_species_start_from=args.augustus_species_start_from,
        flanking_length=args.flanking_length,
        min_gene_number=args.min_gene_number,
        test_gene_number=args.test_gene_number,
        cpu=args.cpu,
        n_trials=args.n_trials,
        min_intron_len=args.min_intron_len,
        use_memory=args.use_memory,
        optimize_method=args.optimize_method,
        stop_after_first=args.stop_after_first,
        start_codons=args.start_codons
    )
    
    try:
        metrics = trainer.run()
        print(f"\nTraining completed! Final accuracy: {metrics.weighted_accuracy * 100:.2f}%")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
