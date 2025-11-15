# AMR-MoEGA: Antimicrobial Resistance Prediction using Mixture of Experts and Genetic Algorithms
### A Full Bioinformatics → Feature Engineering → MoE-GA Modeling Pipeline for Antimicrobial Resistance Prediction

This repository implements a complete, end-to-end computational pipeline for predicting antimicrobial resistance (AMR) from whole-genome sequencing (WGS) data, integrating:
- Bioinformatics workflow (QC, trimming, alignment, variant calling, SnpEff annotation)
- Genomic feature engineering (SNP matrix, gene presence/absence, PCA reduction)
- Mixture-of-Experts (MoE) AMR classifier
- Custom Evolutionary Algorithm (MoEGA) for hyperparameter + genomic feature selection
- Unified CLI for reproducible execution

Repository Structure

.
├── src
│   ├── components
│   │   ├── Button.js
│   │   └── Card.js
│   ├── pages
│   │   ├── HomePage.js
│   │   └── AboutPage.js
│   └── App.js
├── public
│   ├── index.html
│   └── assets
│       └── logo.png
├── .gitignore
├── package.json
└── README.md

AMR-MoEGA/
||
||── pipeline/
|│   ├── bioinfo/                 # All bioinformatics modules
|│   ├── features/                # Feature engineering
|│   ├── moega/                   # GA + MoE modeling engine
|│   ├── utils/                   # Common pipeline utilities
|│   └── cli.py                   # Top-level CLI
|│
|├── config/
|│   ├── config.yaml              # Global runtime config
|│   └── bioinfo_config.yaml      # Reference genomes, SnpEff DB, tools
|│
|├── notebooks/                   # Analysis & visualization
|├── data/                        # raw → intermediate → processed
|└── README.md

Installation
1. Clone
```python
git clone https://github.com/anshul-2010/AMR-Evolution-Prediction.git
cd AMR-Evolution-Prediction
```

2. Create Conda Environment
```python
conda env create -f environment.yml
conda activate amr-evo
```

3. Install Repo
pip install -e .

⚙️ Configuration Files
config/config.yaml

Controls the overall pipeline:

Directory paths

Pipeline steps

Feature engineering options

MoE-GA settings

Logging

Dataset splits

config/bioinfo_config.yaml

Controls:

Paths to reference genome

BWA, fastp, samtools executable paths

SnpEff database ID

Variant calling parameters

You must edit these paths before running the pipeline.

🧪 Running the Full Pipeline (One Command)

The entire workflow—from genomes → variants → features → MoE model → evaluation—can be run with:

python -m pipeline.cli run --config config/config.yaml


This executes:

Bioinformatics pipeline

Feature engineering

Model training (MoE or GA or both)

Evaluation

Visualization

All outputs are stored in:

data/intermediate
data/processed
experiments/results

🧬 1. Running the Bioinformatics Pipeline Only

This performs:

QC with fastp

Alignment with BWA-MEM

Sorting/indexing with samtools

Variant calling (bcftools mpileup + call)

Refinement (QUAL filters)

SnpEff annotation

Run:

python -m pipeline.cli bioinfo \
    --config config/config.yaml \
    --bioinfo-config config/bioinfo_config.yaml


The pipeline writes:

data/intermediate/trimmed_reads/
data/intermediate/aligned_reads/
data/intermediate/variants/
data/intermediate/annotated_variants/

🔬 2. Feature Engineering

Generates:

SNP binary presence/absence matrix

Gene presence/absence matrix

Functional features (synonymous vs nonsynonymous)

PCA embeddings

Run:

python -m pipeline.cli features --config config/config.yaml


Output written to:

data/processed/features/
data/processed/PCA_embeddings/
data/processed/train_test_split/

🧠 3. Train the MoE AMR Classifier (XGBoost + LightGBM + RandomForest)

The MoE model contains:

Three experts

XGBoostClassifier

LightGBMClassifier

RandomForestClassifier

Adaptive gating network (PyTorch) that learns instance-wise expert weights

Weighted expert fusion for final AMR prediction

To train the MoE model:

python -m pipeline.cli moe --config config/config.yaml


Artifacts saved in:

experiments/model_checkpoints/moe/
experiments/results/

🧬⚙️ 4. Run the Evolutionary Algorithm (MoEGA)

The EA performs:

Joint optimization of
✔ ML hyperparameters
✔ Feature subset selection

Fitness = MoE model accuracy (trained each generation)

Uses
✔ Tournament selection
✔ Adaptive mutation
✔ Adaptive crossover
✔ Expert-informed routing for genetic operators

Run:

python -m pipeline.cli ga --config config/config.yaml


Outputs:

experiments/results/moega/
experiments/logs/moega/


Includes:

Fitness curves

Best chromosome hyperparameters

Selected genomic feature set

Best MoE model checkpoint

📊 5. Evaluation + Plotting

Evaluate trained models:

python -m pipeline.cli evaluate --config config/config.yaml


Generate PCA, feature importance, decision boundaries:

python -m pipeline.cli visualize --config config/config.yaml


Plots saved under:

experiments/results/plots/

📁 Structure of Key CLI Commands
Command	Description
python -m pipeline.cli run	Full pipeline
python -m pipeline.cli bioinfo	Bioinformatics pipeline only
python -m pipeline.cli features	Feature extraction
python -m pipeline.cli moe	MoE model training
python -m pipeline.cli ga	Evolutionary optimization (MoEGA)
python -m pipeline.cli evaluate	Test-set evaluation
python -m pipeline.cli visualize	All plots
🧪 Example Full Workflow
python -m pipeline.cli bioinfo \
    --config config/config.yaml \
    --bioinfo-config config/bioinfo_config.yaml

python -m pipeline.cli features --config config/config.yaml

python -m pipeline.cli moe --config config/config.yaml

python -m pipeline.cli ga --config config/config.yaml

python -m pipeline.cli evaluate --config config/config.yaml

🧾 Citing This Work

A proper CITATION.cff is included:

CITATION.cff

🤝 Contributing

Open to PRs for:

new variant callers

additional expert models

faster mutation/crossover kernels

more feature encoders

📧 Contact

For questions or collaborations:

Your Name
PhD / Researcher – IIT Madras
Email: your_email@domain
