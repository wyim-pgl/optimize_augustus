# AUGUSTUS Training Pipeline (Python)

Complete Python reimplementation of GETA's BGM2AT pipeline for AUGUSTUS HMM training.

## Features

| Feature | Original (Perl) | Python Version |
|---------|-----------------|----------------|
| Parameter optimization | Grid search (6 rounds × 28 params × 6 values) | Bayesian optimization (100 trials) |
| Config file handling | Full directory copy (~2000×) | Symlinks (minimal I/O) |
| Parallelization | ParaFly (file-based) | ProcessPoolExecutor (memory) |
| Dependencies | GETA + Perl modules | Python 3.8+ only |
| Speed | 4-6 hours | 30-60 minutes |

## Installation

```bash
# Required
pip install optuna --break-system-packages

# AUGUSTUS must be installed and in PATH
# $AUGUSTUS_CONFIG_PATH must be set
which augustus etraining
echo $AUGUSTUS_CONFIG_PATH
```

## Files

- `augustus_training.py` - Main pipeline (replaces BGM2AT)
- `optimize_augustus_module.py` - Parameter optimization module (replaces BGM2AT.optimize_augustus)

## Usage

### Basic Usage

```bash
# Minimal command
python augustus_training.py genes.gff3 genome.fasta my_species

# Output will be in ./augustus_training_output/
```

### Full Options

```bash
python augustus_training.py \
    --output-dir ./training_results \
    --cpu 32 \
    --n-trials 150 \
    --flanking-length 1000 \
    --min-intron-len 20 \
    --use-memory \
    --augustus-species-start-from arabidopsis \
    genes.gff3 genome.fasta my_new_species
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | augustus_training_output | Output directory |
| `--cpu` | 8 | Number of CPUs |
| `--n-trials` | 100 | Bayesian optimization trials |
| `--flanking-length` | auto | Flanking sequence length for GenBank |
| `--min-gene-number` | 500 | Minimum genes for train/test split |
| `--test-gene-number` | 300 | Number of genes for testing |
| `--min-intron-len` | 30 | Minimum intron length |
| `--use-memory` | False | Use /dev/shm for temp files |
| `--optimize-method` | 1 | 0=none, 1=Bayesian, 2=optimize_augustus.pl, 3=both |
| `--stop-after-first` | False | Stop after first training |
| `--augustus-config-path` | $AUGUSTUS_CONFIG_PATH | Custom config path |
| `--augustus-species-start-from` | None | Copy params from existing species |
| `--onlytrain-gff3` | None | Additional genes for training only |

### Optimization Method Options

```bash
# No optimization (just training)
python augustus_training.py --optimize-method 0 genes.gff3 genome.fasta species

# Bayesian optimization only (fastest, recommended)
python augustus_training.py --optimize-method 1 genes.gff3 genome.fasta species

# AUGUSTUS optimize_augustus.pl only (slower but thorough)
python augustus_training.py --optimize-method 2 genes.gff3 genome.fasta species

# Both methods (slowest, most thorough)
python augustus_training.py --optimize-method 3 genes.gff3 genome.fasta species
```

## Pipeline Steps

1. **Prepare Config** - Set up AUGUSTUS species directory
2. **Convert GFF3 → GenBank** - No external dependencies needed
3. **Filter Bad Genes** - Remove problematic gene models via etraining
4. **Split Train/Test** - Random stratified split
5. **First Training** - Initial etraining + accuracy test
6. **Optimize Parameters** - Bayesian optimization (Optuna)
7. **Second Training** - Final etraining + accuracy test
8. **Finalize** - Compare and select best model

## Output Files

```
augustus_training_output/
├── augustus_training.log      # Complete log
├── genes.gb                   # Filtered GenBank file
├── genes.gb.train             # Training set
├── genes.gb.test              # Test set
├── firsttest.out              # First accuracy test
├── secondtest.out             # Second accuracy test
├── hmm_files_bak01/           # First training backup
├── hmm_files_bak02/           # Second training backup
├── hmm_files_bak -> bak0X/    # Link to best
├── accuracy_of_AUGUSTUS_HMM_Training.txt  # Final report
└── optimization/              # Optimization results
    └── optimization_results.txt
```

## Standalone Optimizer

You can also use just the optimizer:

```bash
python optimize_augustus_module.py \
    --cpu 32 \
    --n-trials 200 \
    --use-memory \
    species_name train.gb
```

## Speed Comparison

| Scenario | Original BGM2AT | Python Version |
|----------|-----------------|----------------|
| 500 genes, 8 CPU | ~4-6 hours | ~30-60 min |
| 500 genes, 32 CPU | ~2-3 hours | ~15-30 min |
| 1000 genes, 32 CPU | ~6-8 hours | ~45-90 min |

## Tips

1. **For fungi/small genomes**: Use `--flanking-length 100 --min-intron-len 20`
2. **For speed**: Use `--use-memory --optimize-method 1 --n-trials 50`
3. **For accuracy**: Use `--optimize-method 3 --n-trials 200`
4. **Checkpointing**: Pipeline creates `.ok` files; restart from where it stopped

## Requirements

- Python 3.8+
- optuna (`pip install optuna`)
- AUGUSTUS 3.5.0+
- $AUGUSTUS_CONFIG_PATH set correctly

## Troubleshooting

### "AUGUSTUS_CONFIG_PATH not found"
```bash
export AUGUSTUS_CONFIG_PATH=/path/to/augustus/config
```

### "etraining: command not found"
```bash
# Make sure AUGUSTUS is in PATH
export PATH=/path/to/augustus/bin:$PATH
```

### Memory issues with large genomes
```bash
# Don't use --use-memory, let it use disk
python augustus_training.py genes.gff3 genome.fasta species
```

## License

Same as original GETA (GPL-3.0)

## Credits

- Original BGM2AT: chenlianfu/geta
- Python rewrite: For Won's lab
