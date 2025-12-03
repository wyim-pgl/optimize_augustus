# AUGUSTUS HMM Training Pipeline (Python)

A Python reimplementation of the BGM2AT pipeline for training AUGUSTUS Hidden Markov Models (HMMs). This script provides a standalone, efficient, and modern pipeline for generating new species-specific gene prediction models for AUGUSTUS.

It is designed to be highly memory-efficient, fast, and easy to use, removing all external Perl dependencies from the original pipeline.

## Key Features

- **High Performance**: Utilizes modern parallelization (`ProcessPoolExecutor`) and efficient I/O to be significantly faster than the original Perl implementation.
- **Memory Efficient**: Employs lazy-loading for FASTA files and a streaming I/O model, allowing it to process huge genomes (10GB+) with a minimal memory footprint.
- **Advanced Parameter Optimization**: Integrates Optuna for sophisticated Bayesian optimization of AUGUSTUS metaparameters, leading to better models in fewer iterations than grid search.
- **Full Isoform Support**: By default, it treats all alternative splice variants (isoforms) from a GFF3 file as unique training examples, enriching the training data and improving prediction accuracy for organisms with complex splicing.
- **Zero Perl Dependencies**: A pure Python implementation, simplifying installation and environment management.
- **Robust & Resumable**: Features comprehensive logging and checkpointing, allowing the pipeline to be safely stopped and resumed from the last completed step.

## Requirements

- Python 3.8+
- AUGUSTUS (3.5.0+ recommended) must be installed with its scripts (`etraining`, `new_species.pl`, etc.) in your system's `PATH`.
- The `AUGUSTUS_CONFIG_PATH` environment variable must be set to point to the AUGUSTUS `config` directory.
- `optuna`: A Python library for optimization.

## Installation

1.  **Install AUGUSTUS**:
    If you don't have AUGUSTUS installed, you can do so via `apt`, `conda`, or from source.

    *   **Conda (Recommended)**:
        ```bash
        conda create -n augustus -c bioconda -c conda-forge augustus
        conda activate augustus
        # Set the config path for your conda environment
        export AUGUSTUS_CONFIG_PATH=$CONDA_PREFIX/config
        ```
    *   **Build from Source**:
        ```bash
        git clone https://github.com/Gaius-Augustus/Augustus.git
        cd Augustus
        make
        # Add augustus to your PATH and set the config path
        export PATH="$PWD/bin:$PATH"
        export AUGUSTUS_CONFIG_PATH="$PWD/config"
        ```

2.  **Install Python Dependencies**:
    ```bash
    pip install optuna
    ```
    *BioPython is optional but recommended for the most robust GenBank file handling. If not found, a fallback is used.*
    ```bash
    pip install biopython
    ```

## Usage

The script is run from the command line, with the GFF3, genome FASTA, and a new species name as the primary inputs.

```bash
python augustus_training.py [OPTIONS] <genes.gff3> <genome.fasta> <species_name>
```

### Command-Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| **`gff3`** | **(Required)** GFF3 file with gene models for training. | |
| **`genome`** | **(Required)** Genome FASTA file. | |
| **`species`** | **(Required)** AUGUSTUS species name to create/update. | |
| `-o`, `--output-dir` | Output directory to store all results and logs. | `augustus_training_output` |
| `--cpu` | Number of CPUs to use for parallel tasks. | `8` |
| `--augustus-config-path` | Override the `$AUGUSTUS_CONFIG_PATH` environment variable. | `$AUGUSTUS_CONFIG_PATH` |
| `--augustus-species-start-from`| Create the new species by copying parameters from an existing species (e.g., `arabidopsis`). | `None` |
| `--onlytrain-gff3` | An additional GFF3 file with genes to be used only for training (not for testing accuracy). | `None` |
| `--flanking-length` | Length of flanking DNA to include around each gene in the GenBank file. | `auto` |
| `--min-gene-number` | Minimum number of valid genes required for a proper train/test split. | `500` |
| `--test-gene-number`| Number of genes to hold out for the test set. | `300` |
| `--n-trials` | Number of trials for the Bayesian optimization of parameters. | `100` |
| `--min-intron-len` | Minimum intron length for `etraining`. | `30` |
| `--start-codons` | A comma-separated list of allowed start codons. Probabilities are distributed uniformly. | `ATG` |
| `--no-isoforms` | Flag to use only the longest isoform per gene, disabling the default behavior of using all isoforms. | `False` |
| `--optimize-method` | Optimization strategy: `0`=None, `1`=Bayesian (Optuna), `2`=optimize_augustus.pl, `3`=Both. | `1` |
| `--use-memory` | If flagged, use `/dev/shm` for temporary files during optimization. Can be faster but requires sufficient RAM. | `False` |
| `--stop-after-first` | If flagged, the pipeline will stop after the first training round and skip optimization. | `False` |

---

## Examples

#### Basic Run
Train a new model for `my_species` using all defaults.
```bash
python augustus_training.py genes.gff3 genome.fasta my_species
```

#### Advanced Run for a Complex Eukaryote
This example uses 32 cores, starts from *Arabidopsis* parameters, provides extra training genes, defines multiple start codons, and runs 200 optimization trials.
```bash
python augustus_training.py \
    --output-dir ./training_results \
    --cpu 32 \
    --n-trials 200 \
    --augustus-species-start-from arabidopsis \
    --onlytrain-gff3 extra_genes.gff3 \
    --start-codons "ATG,CTG,TTG" \
    genes.gff3 genome.fasta my_new_plant
```

#### Run for a Simple Organism (e.g., some fungi)
This example disables the multi-isoform feature, as it may not be necessary for organisms with simple gene structures.
```bash
python augustus_training.py \
    --no-isoforms \
    --min-intron-len 20 \
    genes.gff3 genome.fasta my_fungus
```

## Pipeline Overview

The pipeline automates the following sequence of steps:

1.  **Prepare Config**: Sets up the AUGUSTUS species directory and copies base parameters if specified.
2.  **Convert GFF3 to GenBank**: A memory-efficient, parallelized conversion of GFF3 annotations into the GenBank format required by AUGUSTUS.
3.  **Filter Bad Genes**: Runs a preliminary `etraining` to identify and exclude gene models that would cause errors (e.g., in-frame stop codons).
4.  **Split Train/Test Sets**: Randomly splits the valid gene models into training and testing sets.
5.  **First Training**: Runs `etraining` on the training set and evaluates the accuracy of the resulting model against the test set.
6.  **Optimize Parameters**: Performs parameter optimization using the selected method (default: Bayesian optimization with Optuna).
7.  **Second Training**: Re-runs `etraining` with the optimized parameters.
8.  **Finalize**: Compares the accuracy before and after optimization, keeps the better of the two models, and generates a final report.

This entire process is resumable. If the script is interrupted, it will pick up from the last successfully completed step upon re-running.