"""
Driver for feature engineering modules under 03_feature_engineering.
"""

from ml_pipeline.d_feature_engineering.build_snp_matrix import build_snp_matrix
from ml_pipeline.d_feature_engineering.gene_presence_absence import build_gene_features
from ml_pipeline.d_feature_engineering.functional_feature_builder import (
    build_functional_features,
)
from ml_pipeline.d_feature_engineering.pca_feature_reduction import run_pca


def run_feature_engineering(config):
    fe = config.get("feature_engineering", {})

    print("\n========== FEATURE ENGINEERING PIPELINE ==========\n")

    if fe.get("build_snp_matrix", True):
        print("[1/4] Building SNP matrix...")
        build_snp_matrix(config)

    if fe.get("gene_presence_absence", True):
        print("[2/4] Building gene presence/absence features...")
        build_gene_features(config)

    if fe.get("functional_features", True):
        print("[3/4] Adding functional annotations...")
        build_functional_features(config)

    if fe.get("pca_reduction", True):
        print("[4/4] Running PCA reduction...")
        run_pca(config)

    print("\n Feature engineering completed!\n")
