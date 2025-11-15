AMR-Evolution-Prediction/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml                # Conda environment with all deps
├── requirements.txt               # Optional pip dependencies
├── .gitignore
│
├── docs/
│   ├── overview.md
│   ├── methodology.md
│   ├── pipeline_diagram.png
│   ├── results_summary.md
│   └── api_documentation.md
│
├── data/
│   ├── raw/
│   │   ├── genomes/              # Downloaded MicroBIGG-E genomes
│   │   ├── metadata/             # Species/antibiotic metadata
│   │   └── AMR_labels/           # Phenotypic resistance labels
│   │
│   ├── intermediate/
│   │   ├── aligned_reads/        # BWA mem output
│   │   ├── trimmed_reads/        # fastp output
│   │   ├── variants/             # VCF + refined outputs
│   │   └── annotated_variants/   # After SnpEff
│   │
│   └── processed/
│       ├── features/             # Feature matrices (SNPs, gene presence, etc.)
│       ├── PCA_embeddings/
│       ├── train_test_split/
│       └── labels/
│
├── pipeline/
│   ├── 00_download/
│   │   └── download_microbigg_data.py
│   │
│   ├── 01_preprocessing/
│   │   ├── trim_reads.py
│   │   ├── align_reads.py
│   │   ├── post_alignment_qc.py
│   │   └── coverage_stats.py
│   │
│   ├── 02_variants/
│   │   ├── variant_calling_and_refinement.py
│   │   ├── filter_variants.py
│   │   └── snpeff_annotation.py
│   │
│   ├── 03_feature_engineering/
│   │   ├── build_snp_matrix.py
│   │   ├── gene_presence_absence.py
│   │   ├── functional_feature_builder.py
│   │   └── pca_feature_reduction.py
│   │
│   ├── 04_modeling/
│   │   ├── baseline_models.py       # RandomForest, XGBoost, LR, SVM
│   │   ├── static_ML_training.py
│   │   ├── ga_static_fitness.py
│   │   ├── moe_ga_engine.py         # The main MoE–GA algorithm
│   │   ├── expert_models/
│   │   │   ├── expert_clf.py
│   │   │   └── expert_reg.py
│   │   ├── gating_network.py
│   │   └── scoring_metrics.py
│   │
│   ├── 05_post_analysis/
│   │   ├── pca_visualization.py
│   │   ├── feature_importance.py
│   │   ├── fitness_trajectory_plots.py
│   │   ├── decision_boundary_plots.py
│   │   └── genome_evolution_summary.py
│   │
│   ├── 06_evaluation/
│   │   ├── cross_validation.py
│   │   ├── comparative_results.py
│   │   └── statistical_tests.py
│   │
│   ├── run_pipeline.sh              # End-to-end wrapper
│   └── config.yaml                  # Central config file
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_PCA_visualization.ipynb
│   ├── 03_variant_stats.ipynb
│   ├── 04_feature_importance.ipynb
│   ├── 05_expert_model_analysis.ipynb
│   ├── 06_MoE_GA_fitness_trajectories.ipynb
│   └── 07_final_results_and_plots.ipynb
│
├── experiments/
│   ├── configs/
│   │   ├── exp1_baseline.yaml
│   │   ├── exp2_GA_static.yaml
│   │   ├── exp3_MoEGA.yaml
│   │   └── exp4_ablation.yaml
│   │
│   ├── logs/
│   ├── model_checkpoints/
│   └── results/
│       ├── exp1/
│       ├── exp2/
│       ├── exp3/
│       └── exp4/
│
├── scripts/
│   ├── install_dependencies.sh
│   ├── convert_vcf_to_matrix.sh
│   ├── prepare_training_data.sh
│   ├── evaluate_model.sh
│   └── generate_plots.sh
│
└── utils/
    ├── file_io.py
    ├── logger.py
    ├── helpers.py
    ├── stats_utils.py
    ├── genetic_operators.py      # mutation, crossover, selection
    └── plot_utils.py