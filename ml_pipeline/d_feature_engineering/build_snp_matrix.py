#!/usr/bin/env python3
import os
import yaml
import pandas as pd
from cyvcf2 import VCF


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_matrix(vcf_dir, output_csv):
    sample_names = []
    pos2genotype = {}

    for fname in os.listdir(vcf_dir):
        if not fname.endswith(".vcf.gz") and not fname.endswith(".vcf"):
            continue
        path = os.path.join(vcf_dir, fname)
        sample = os.path.splitext(os.path.splitext(fname)[0])[0]
        sample_names.append(sample)
        vcf = VCF(path)
        for rec in vcf:
            key = f"{rec.CHROM}:{rec.POS}"
            # this is simplistic; adapt to represent genotype properly
            gt = rec.genotypes[0][0]  # 0 for ref, 1 for alt
            pos2genotype.setdefault(key, {})[sample] = gt

    df = pd.DataFrame.from_dict(pos2genotype, orient="index", columns=sample_names)
    df = df.fillna(0).astype(int)
    df.to_csv(output_csv)
    print(f"Wrote SNP matrix: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vcf_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_csv = cfg["snp_matrix"]["output_csv"]
    build_matrix(args.vcf_dir, out_csv)
