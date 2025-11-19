# AMR-MoEGA: Antimicrobial Resistance Prediction using Mixture of Experts Genetic Algorithms

<div align="center">

**End-to-End Bioinformatics → Feature Engineering → MoE-GA Modeling Pipeline for Antimicrobial Resistance Prediction**

[![Repository](https://img.shields.io/badge/Repository-AMR--MoEGA-blue)](https://github.com/anshul-2010/AMR-MoEGA)
[![Python](https://img.shields.io/badge/Python-3.7+-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---
The official repository for the paper "AMR-MoEGA: Antimicrobial Resistance Prediction using Mixture of Experts and Genetic Algorithms". 
ArXiv link: https://arxiv.org/abs/2511.12223

Abstract
Antimicrobial resistance (AMR) poses a mounting global health crisis, requiring rapid and reliable prediction frameworks that capture its complex evolutionary dynamics. Traditional antimicrobial susceptibility testing (AST), while accurate, remains laborious and time-consuming, limiting its clinical scalability. Existing computational approaches, primarily reliant on single nucleotide polymorphism (SNP)-based analysis, fail to account for evolutionary drivers such as horizontal gene transfer (HGT) and genome-level interactions.
This study introduces a novel Evolutionary Mixture of Experts (Evo-MoE) framework that integrates genomic sequence analysis, machine learning, and evolutionary algorithms to model and predict AMR evolution. A Mixture of Experts model, trained on labeled genomic data for multiple antibiotics, serves as the predictive core, estimating the likelihood of resistance for each genome. This model is embedded as a fitness function within a Genetic Algorithm designed to simulate AMR development across generations. Each genome, encoded as an individual in the population, undergoes mutation, crossover, and selection guided by predicted resistance probabilities.
The resulting evolutionary trajectories reveal dynamic pathways of resistance acquisition, offering mechanistic insights into genomic evolution under selective antibiotic pressure. Sensitivity analysis of mutation rates and selection pressures demonstrates the model's robustness and biological plausibility. Validation against curated AMR databases and literature evidence further substantiates the framework's predictive fidelity.
This integrative approach bridges genomic prediction and evolutionary simulation, offering a powerful tool for understanding and anticipating AMR dynamics, and potentially guiding rational antibiotic design and policy interventions.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [System Requirements](#system-requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Quick Start](#quick-start)
8. [Pipeline Components](#pipeline-components)
9. [Command-Line Usage](#command-line-usage)
10. [Data Formats](#data-formats)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting](#troubleshooting)
13. [Citation](#citation)

---

## Overview

**AMR-MoEGA** is a comprehensive, production-ready computational pipeline for predicting antimicrobial resistance (AMR) from whole-genome sequencing (WGS) data. It automates the entire workflow from raw sequencing reads to high-confidence AMR predictions through:

- **Integrated Bioinformatics Processing**: Automated quality control, read trimming, genome alignment, and variant annotation
- **Advanced Feature Engineering**: Comprehensive genomic feature extraction (SNP matrices, gene presence/absence, functional annotations, PCA embeddings)
- **Machine Learning with Mixture-of-Experts (MoE)**: Intelligent ensemble approach combining multiple specialized expert models with a learned gating network
- **Evolutionary Algorithm Optimization**: Custom MoE-Genetic Algorithm (MoEGA) for simultaneous hyperparameter optimization and genomic feature selection
- **Comprehensive Evaluation Framework**: Statistical testing, cross-validation, and publication-ready visualizations

## Key Features

| Feature | Description |
|---------|-------------|
| **End-to-End Automation** | Complete pipeline from raw FASTQ reads to AMR predictions |
| **Advanced Genomic Features** | SNP matrices, gene presence/absence, functional annotations, PCA dimensionality reduction |
| **Mixture-of-Experts Architecture** | Multiple expert classifiers with learned gating network for adaptive ensemble prediction |
| **Evolutionary Optimization** | GA-based simultaneous feature selection and hyperparameter tuning |
| **Robust Evaluation** | Cross-validation, statistical significance testing, comparative metrics |
| **Configuration-Driven** | YAML-based configuration for reproducible, customizable execution |
| **Publication-Ready Analysis** | Decision boundaries, fitness trajectories, PCA visualizations, feature importance plots |
| **Reproducible Science** | Fixed random seeds, logging, and experiment tracking |

## Repository Structure

```
AMR-MoEGA/
│
├── bioinformatics_pipeline/              # Complete bioinformatics workflow
│   ├── a_download/                       # Data download modules
│   │   └── download_microbigge_data.py   # Download from MicroBIGG-E database
│   ├── b_preprocessing/                  # Read QC & alignment
│   │   ├── trim_reads.py                 # fastp-based read trimming
│   │   ├── align_reads.py                # BWA read alignment
│   │   ├── coverage_stats.py             # Coverage analysis
│   │   ├── post_alignment_qc.py          # BAM quality control
│   │   └── *.sh                          # Shell wrapper scripts
│   ├── c_variants/                       # Variant calling & annotation
│   │   ├── variant_calling_and_refinement.py  # BCFtools/GATK calling
│   │   ├── filter_variants.py            # Variant quality filtering
│   │   ├── snpeff_annotation.py          # SnpEff functional annotation
│   │   └── *.sh                          # External tool scripts
│   └── bioinfo.py                        # Bioinformatics pipeline orchestrator
│
├── ml_pipeline/                          # Machine learning & feature engineering
│   ├── d_feature_engineering/            # Genomic feature extraction
│   │   ├── build_snp_matrix.py           # SNP matrix construction
│   │   ├── build_snp_matrix_ml_iamr.py   # Alternative SNP encoding
│   │   ├── gene_presence_absence.py      # Gene presence/absence matrix
│   │   ├── functional_feature_builder.py # Functional annotation features
│   │   └── pca_feature_reduction.py      # PCA-based dimensionality reduction
│   ├── e_modeling/                       # ML modeling engines
│   │   ├── baseline_models.py            # Standard ML classifiers (RF, SVM, XGB)
│   │   ├── static_ml_training.py         # Baseline model training
│   │   ├── ga_static_fitness.py          # GA with static fitness
│   │   ├── moe_ga_engine.py              # MoE-GA optimizer
│   │   ├── gating_network.py             # Expert gating mechanism
│   │   ├── scoring_metrics.py            # Performance metrics
│   │   └── expert_models/                # Individual expert definitions
│   ├── feature_engineering.py            # FE pipeline orchestrator
│   └── modeling.py                       # Modeling pipeline orchestrator
│
├── moega_pipeline/                       # MoE-GA implementation
│   ├── chromosome.py                     # Chromosome representation
│   ├── ga_engine.py                      # GA main loop
│   ├── genetic_operators.py              # Selection, crossover, mutation
│   ├── hgt_crossover.py                  # Horizontal Gene Transfer crossover
│   ├── fitness.py                        # Fitness evaluation
│   ├── gating.py                         # Gating network training
│   ├── search_space.py                   # Feature/hyperparameter search space
│   ├── experts.py                        # Expert model definitions
│   ├── trainer.py                        # Data loading & splitting
│   ├── logging_utils.py                  # Logging utilities
│   └── __init__.py
│
├── analysis_pipeline/                    # Post-analysis & evaluation
│   ├── f_post_analysis/                  # Post-model analysis
│   │   ├── decision_boundary_plots.py    # Decision boundary visualization
│   │   ├── fitness_trajectory_plots.py   # GA fitness progression
│   │   ├── pca_visualization.py          # PCA scatter plots
│   │   ├── feature_importance.py         # Feature importance analysis
│   │   └── genome_evolution_summary.py   # Evolutionary trajectory summary
│   ├── g_evaluation/                     # Model evaluation
│   │   ├── cross_validation.py           # k-fold cross-validation
│   │   ├── statistical_tests.py          # Statistical significance testing
│   │   ├── comparative_results.py        # Comparative model metrics
│   │   └── roc_analysis.py               # ROC/AUC analysis
│   └── evaluation.py                     # Evaluation pipeline orchestrator
│
├── configs/                              # Configuration files
│   ├── config.yaml                       # Main pipeline configuration
│   └── bioinfo_config.yaml               # Bioinformatics-specific config
│
├── data/                                 # Data directory
│   ├── raw/                              # Raw input data
│   │   ├── fastq/                        # Raw FASTQ reads
│   │   ├── genomes/                      # Reference genomes
│   │   ├── metadata/                     # Sample metadata
│   │   ├── AMR_labels/                   # Ground truth AMR labels
│   │   └── reference/                    # Reference genome (indexed)
│   ├── intermediate/                     # Intermediate processing outputs
│   │   ├── trimmed_reads/                # Trimmed FASTQ
│   │   ├── aligned_reads/                # BAM files & QC
│   │   ├── variants/                     # Raw VCF files
│   │   └── annotated_variants/           # Annotated VCF files
│   ├── processed/                        # Final processed data
│   │   ├── features/                     # Feature matrices
│   │   ├── labels/                       # Processed labels
│   │   ├── pca_embeddings/               # PCA-reduced features
│   │   └── train_test_split/             # Train/test indices
│   └── Giessen_dataset/                  # Example dataset
│       ├── cip_ctx_ctz_gen_multi_data.csv
│       └── cip_ctx_ctz_gen_pheno.csv
│
├── scripts/                              # Utility shell scripts
│   ├── install_dependencies.sh           # Dependency installation
│   ├── prepare_training_data.sh          # Data preparation
│   ├── convert_vcf_to_matrix.sh          # VCF format conversion
│   ├── generate_plots.sh                 # Plot generation
│   └── evaluate_model.sh                 # Model evaluation
│
├── docs/                                 # Documentation
│   ├── bioinfo_pipeline.md               # Bioinformatics details
│   ├── methodology.md                    # Methods description
│   ├── overview.md                       # High-level overview
│   └── results_summary.md                # Results interpretation
│
├── utils/                                # Utility modules
│   ├── file_io.py                        # File I/O operations
│   ├── genetic_operators.py              # Genetic algorithm utilities
│   ├── helpers.py                        # General helper functions
│   ├── logger.py                         # Logging configuration
│   ├── plot_utils.py                     # Plotting utilities
│   └── stats_utils.py                    # Statistical utilities
│
├── pipeline_cli.py                       # Main CLI entry point
├── requirements.txt                      # Python dependencies
├── CITATION.cff                          # Citation metadata
├── LICENSE                               # License file
└── README.md                             # This file
```

---

## System Requirements

### Minimum Requirements
- **OS**: Linux/macOS/Windows (with WSL2 recommended for Windows)
- **Python**: 3.7 or higher
- **RAM**: 16 GB minimum (32 GB recommended for large datasets)
- **Storage**: 50+ GB for intermediate files

### External Bioinformatics Tools
The pipeline uses the following tools (installed separately):
- **fastp** ≥ 0.20: Read quality control and adapter trimming
- **bwa** ≥ 0.7.17: Read alignment
- **samtools** ≥ 1.10: BAM manipulation and indexing
- **bcftools** ≥ 1.10: Variant calling and filtering
- **gatk** ≥ 4.1.0: Alternative variant calling
- **snpEff** ≥ 4.3: Variant functional annotation

### Python Dependencies
See `requirements.txt` for detailed versions:
```
numpy
pandas
scikit-learn
joblib
matplotlib
pyyaml
cyvcf2
scipy
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anshul-2010/AMR-MoEGA.git
cd AMR-MoEGA
```

### 2. Create Python Environment
**Using conda (recommended):**
```bash
conda create -n amr-moega python=3.9
conda activate amr-moega
```

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Bioinformatics Tools

**Option A: Using conda (recommended)**
```bash
conda install -c bioconda fastp bwa samtools bcftools snpeff
```

**Option B: Manual installation**
See individual tool documentation for installation instructions:
- [fastp](https://github.com/OpenGene/fastp)
- [bwa](http://bio-bwa.sourceforge.net/)
- [samtools](http://www.htslib.org/)
- [bcftools](http://www.htslib.org/)
- [snpEff](http://snpEff.sourceforge.net/)

### 5. Verify Installation
```bash
# Check Python packages
python -c "import pandas, sklearn, yaml; print('✓ Python dependencies OK')"

# Check bioinformatics tools (Linux/macOS)
which fastp bwa samtools bcftools
```

---

## Configuration

### Main Configuration File: `configs/config.yaml`

This YAML file controls all pipeline parameters. Key sections:

#### Project & Paths
```yaml
project:
  name: "AMR-Evolution-Prediction"
  description: "End-to-end AMR prediction pipeline"

paths:
  data_dir: "data"
  raw: "data/raw"
  intermediate: "data/intermediate"
  processed: "data/processed"
```

#### Bioinformatics Settings
```yaml
bioinformatics:
  threads: 8
  memory_gb: 32
  
  tools:
    fastp: "fastp"
    bwa: "bwa"
    samtools: "samtools"
    bcftools: "bcftools"
    snpeff: "snpEff"
  
  trimming:
    min_length: 50
    quality_cutoff: 20
    detect_adapters: true
  
  alignment:
    reference_genome: "data/raw/reference/reference.fasta"
    mark_duplicates: true
    sort_bam: true
```

#### Feature Engineering
```yaml
feature_engineering:
  snp_matrix:
    encoding: "binary"
    min_maf: 0.01
    max_missing_rate: 0.2
  
  pca:
    enabled: true
    n_components: 50
    whiten: true
```

#### ML & MoE-GA Settings
```yaml
modeling:
  train_static_ml: true
  train_ga_baseline: true
  train_moega: true
  
  moega:
    population_size: 30
    generations: 20
    elitism: 2
    mutation_rate: 0.1
    crossover_rate: 0.8
```

### Bioinformatics Config: `configs/bioinfo_config.yaml`

Detailed bioinformatics settings:
```yaml
data:
  raw_fastq_dir: "data/raw/fastq"
  trimmed_fastq_dir: "data/intermediate/trimmed_reads"
  bam_dir: "data/intermediate/aligned_reads"

reference:
  fasta: "data/raw/reference.fa"
  bwa_index_prefix: "data/raw/reference/bwa_index/ref"
  snpeff_db: "Escherichia_coli_K12"
```

---

## Quick Start

### Full Pipeline Execution (Single Command)
```bash
# Run entire pipeline from raw reads to MoE-GA predictions
python pipeline_cli.py bioinfo -c configs/bioinfo_config.yaml && \
python pipeline_cli.py features -c configs/config.yaml && \
python pipeline_cli.py model -c configs/config.yaml && \
python pipeline_cli.py eval -c configs/config.yaml
```

### Step-by-Step Execution

#### Step 1: Bioinformatics Processing
```bash
python pipeline_cli.py bioinfo -c configs/bioinfo_config.yaml
```
**Outputs:**
- Trimmed reads: `data/intermediate/trimmed_reads/`
- Aligned BAM files: `data/intermediate/aligned_reads/`
- Variant VCF files: `data/intermediate/variants/`
- Annotated variants: `data/intermediate/annotated_variants/`

#### Step 2: Feature Engineering
```bash
python pipeline_cli.py features -c configs/config.yaml
```
**Outputs:**
- SNP matrix: `data/processed/features/snp_matrix.csv`
- Gene features: `data/processed/features/gene_features.csv`
- PCA embeddings: `data/processed/pca_embeddings/pca_features.pkl`

#### Step 3: Model Training (Static ML + MoE-GA)
```bash
python pipeline_cli.py model -c configs/config.yaml
```
**Outputs:**
- Trained models: `models/`
- MoE-GA run results: `experiments/moega_run/`
- Fitness logs: `experiments/moega_run/fitness.log`

#### Step 4: Evaluation & Analysis
```bash
python pipeline_cli.py eval -c configs/config.yaml
```
**Outputs:**
- Cross-validation results: `results/cross_validation.csv`
- Comparative metrics: `results/comparative_results.csv`
- Visualizations: `results/plots/`

---

## Pipeline Components

### 1. Bioinformatics Pipeline (`bioinformatics_pipeline/`)

Automated processing of raw WGS data:

**Read Trimming & QC**
```bash
python -m bioinformatics_pipeline.b_preprocessing.trim_reads
# Removes low-quality bases, adapters using fastp
```

**Read Alignment**
```bash
python -m bioinformatics_pipeline.b_preprocessing.align_reads
# Maps reads to reference using BWA, generates sorted BAM
```

**Variant Calling**
```bash
python -m bioinformatics_pipeline.c_variants.variant_calling_and_refinement
# Calls variants using BCFtools/GATK
```

**Variant Annotation**
```bash
python -m bioinformatics_pipeline.c_variants.snpeff_annotation
# Annotates variants with functional impact (HIGH/MODERATE/LOW)
```

### 2. Feature Engineering Pipeline (`ml_pipeline/d_feature_engineering/`)

Extracts diverse genomic features:

**SNP Matrix**
```bash
python -m ml_pipeline.d_feature_engineering.build_snp_matrix
# Binary encoding: 0=reference, 1=alternate, NA=missing
```

**Gene Presence/Absence**
```bash
python -m ml_pipeline.d_feature_engineering.gene_presence_absence
# Binary matrix: 1=gene present, 0=gene absent
```

**Functional Features**
```bash
python -m ml_pipeline.d_feature_engineering.functional_feature_builder
# Pathway annotations, mutation types, impact categories
```

**PCA Dimensionality Reduction**
```bash
python -m ml_pipeline.d_feature_engineering.pca_feature_reduction
# Reduces SNP matrix from N features to 50 principal components
```

### 3. Machine Learning Models (`ml_pipeline/e_modeling/`)

**Baseline Models**
```bash
python -m ml_pipeline.e_modeling.static_ml_training
# Random Forest, SVM, XGBoost with default hyperparameters
```

**Mixture-of-Experts Architecture**
- Multiple expert classifiers (Random Forest, SVM, XGBoost)
- Learned gating network (Neural Network)
- Soft ensemble predictions: `final_pred = Σ(expert_pred × gate_weight)`

**MoE-GA Optimizer**
```bash
python moega_pipeline/ga_engine.py \
  --features data/processed/features/snp_matrix.csv \
  --labels data/processed/labels/labels.csv \
  --pop 30 --gens 20 --out experiments/moega_run
```
**Search space:**
- Feature selection (which genomic features to use)
- Hyperparameter optimization (learning rates, tree depths, regularization)
- Expert weighting (importance of each expert)

### 4. Evaluation Pipeline (`analysis_pipeline/`)

**Cross-Validation**
```bash
python -c "from analysis_pipeline.g_evaluation.cross_validation import run_cross_validation; run_cross_validation(config)"
# 5-fold/10-fold stratified cross-validation
```

**Statistical Tests**
```bash
python -c "from analysis_pipeline.g_evaluation.statistical_tests import run_stats_tests; run_stats_tests(config)"
# McNemar test, paired t-tests for model comparison
```

**Visualizations**
```bash
python -c "from analysis_pipeline.f_post_analysis.decision_boundary_plots import plot_decision_boundaries"
# Decision boundaries, fitness trajectories, PCA plots, feature importance
```

---

## Command-Line Usage Guide

### Main CLI: `pipeline_cli.py`

```bash
usage: pipeline_cli.py [-h] {bioinfo,features,model,eval} ...

positional arguments:
  {bioinfo,features,model,eval}
    bioinfo              Run the bioinformatics pipeline
    features             Run feature engineering
    model                Run ML/MoE-GA training
    eval                 Run evaluation
```

### 1. Bioinformatics Command
```bash
python pipeline_cli.py bioinfo -c configs/bioinfo_config.yaml

# Optional flags:
# --skip-trimming         Skip read trimming step
# --skip-alignment        Skip read alignment step
# --skip-variant-calling  Skip variant calling step
# --skip-annotation       Skip SnpEff annotation step
```

### 2. Feature Engineering Command
```bash
python pipeline_cli.py features -c configs/config.yaml

# Example with custom paths:
python pipeline_cli.py features -c configs/config.yaml \
  --vcf-dir data/intermediate/annotated_variants \
  --output-dir data/processed/features
```

### 3. Model Training Command
```bash
python pipeline_cli.py model -c configs/config.yaml

# Example with custom MoE-GA parameters:
python pipeline_cli.py model -c configs/config.yaml \
  --moega-pop 50 \
  --moega-gens 30 \
  --seed 42
```

### 4. Evaluation Command
```bash
python pipeline_cli.py eval -c configs/config.yaml

# Example with custom CV folds:
python pipeline_cli.py eval -c configs/config.yaml \
  --cv-folds 10 \
  --test-size 0.2
```

---

## Data Formats

### Input Formats

**FASTQ Files** (raw sequencing reads)
```
@read_id
ACGTACGTACGT...
+
IIIIIIIIIIII...
```
- Location: `data/raw/fastq/`
- Format: Paired-end (suffix: `_R1.fastq.gz`, `_R2.fastq.gz`)

**Reference Genome** (FASTA)
```
>sequence_id
ACGTACGTACGT...
```
- Location: `data/raw/reference/reference.fasta` (must be BWA-indexed)

**AMR Labels** (CSV)
```
sample_id,resistance_label
sample_001,resistant
sample_002,susceptible
```
- Location: `data/raw/AMR_labels/`

### Output Formats

**SNP Matrix** (CSV)
```
sample_id,snp_1,snp_2,snp_3,...
sample_001,0,1,0,...
sample_002,1,0,1,...
```
- Location: `data/processed/features/snp_matrix.csv`
- Values: 0=reference, 1=alternate, NaN=missing

**Feature Matrix** (CSV)
```
sample_id,feature_1,feature_2,...
sample_001,0.123,0.456,...
sample_002,0.789,0.012,...
```
- Location: `data/processed/features/`

**VCF Files** (Variant Call Format)
```
##fileformat=VCFv4.2
#CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO
NC_000913.3  100  .  A  T  60  PASS  ANN=T|missense_variant|...
```
- Raw VCF: `data/intermediate/variants/`
- Annotated VCF: `data/intermediate/annotated_variants/`

**Model Results** (JSON/pickle)
```json
{
  "model_type": "MoE-GA",
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.95,
  "auc": 0.96,
  "selected_features": ["SNP_123", "SNP_456", ...],
  "expert_weights": [0.3, 0.4, 0.3],
  "hyperparameters": {...}
}
```
- Location: `experiments/moega_run/`

---

## Advanced Usage

### Custom Feature Selection
Edit `configs/config.yaml`:
```yaml
feature_engineering:
  build_snp_matrix: true
  gene_presence_absence: false      # Skip gene features
  functional_features: true
  pca_reduction: true
```

### MoE-GA with Custom Parameters
```bash
python moega_pipeline/ga_engine.py \
  --features data/processed/features/snp_matrix.csv \
  --labels data/processed/labels/labels.csv \
  --pop 50 \                        # Larger population
  --gens 50 \                       # More generations
  --mut_mask_rate 0.05 \            # Feature mutation rate
  --mut_param_rate 0.2 \            # Parameter mutation rate
  --elitism 5 \                     # Keep top 5 individuals
  --tourney_k 5 \                   # Tournament size
  --seed 42 \                       # Reproducibility
  --out experiments/moega_custom
```

### Cross-Validation with Custom Folds
```python
from analysis_pipeline.g_evaluation.cross_validation import run_cross_validation
from sklearn.model_selection import StratifiedKFold

config['evaluation'] = {
    'cv_folds': 10,
    'stratified': True,
    'shuffle': True,
    'random_state': 42
}
run_cross_validation(config)
```

### Parallel Feature Engineering
```bash
# Run feature modules independently and combine
python -m ml_pipeline.d_feature_engineering.build_snp_matrix &
python -m ml_pipeline.d_feature_engineering.gene_presence_absence &
python -m ml_pipeline.d_feature_engineering.functional_feature_builder &
wait
```

### Experiment Tracking
Enable logging in config:
```yaml
logging:
  level: "DEBUG"
  file: "experiments/run.log"
  console: true
```

Access logs:
```bash
tail -f experiments/run.log
```

---

## Troubleshooting

### Issue: "Command not found: fastp"
**Solution:** Install bioinformatics tools
```bash
conda install -c bioconda fastp bwa samtools bcftools
```

### Issue: "Memory error during alignment"
**Solution:** Reduce threads in `config.yaml`:
```yaml
bioinformatics:
  threads: 4      # Reduce from 8
  memory_gb: 16   # Reduce from 32
```

### Issue: "No such file: reference.fasta"
**Solution:** Ensure reference genome exists and is indexed:
```bash
# Check file exists
ls -la data/raw/reference/reference.fasta*

# If not indexed, create index:
bwa index data/raw/reference/reference.fasta
samtools faidx data/raw/reference/reference.fasta
```

### Issue: "ValueError: y contains unseen labels"
**Solution:** Ensure train/test sets have all classes:
```python
# In configs/config.yaml, use stratified split:
feature_engineering:
  train_test_split:
    stratify: true
    test_size: 0.2
    random_state: 42
```

### Issue: "MoE-GA training too slow"
**Solution:** Reduce search space or population size:
```bash
python moega_pipeline/ga_engine.py \
  --features data/processed/features/snp_matrix.csv \
  --labels data/processed/labels/labels.csv \
  --pop 20 \        # Reduce population
  --gens 10 \       # Reduce generations
  --out experiments/quick_run
```

### Issue: "YAML config parsing error"
**Solution:** Validate YAML syntax:
```bash
# Python check
python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"

# Or use online YAML validator
```

---

## License

This project is licensed under the MIT License - see `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact & Support

- **Issues/Bugs**: [GitHub Issues](https://github.com/anshul-2010/AMR-MoEGA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/anshul-2010/AMR-MoEGA/discussions)
- **Email**: Open an issue for contact information

---

## Related Resources

- [Bioinformatics Best Practices](https://bioinformatics.ca/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [SnpEff Tutorial](http://snpeff.sourceforge.net/)
- [Genetic Algorithm Optimization](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Mixture of Experts](https://en.wikipedia.org/wiki/Mixture_of_experts)

## Contact
For questions or collaborations:
- Anshul Bagaria
- Dual Degree Student / Researcher – IIT Madras
- Email: be21b005@smail.iitm.ac.in

## Note on Automation Status
- The core logic and individual components within the pipeline files (*.py, etc.) are functionally correct and have been verified. However, the fully automated, end-to-end streaming process (e.g., scheduled runs, continuous data flow between stages) is still under active development.
- If you encounter issues with automated runs, please try executing the pipeline steps manually and sequentially. The full automation layer will be completed soon.
