"""
High-level orchestration for the bioinformatics pipeline.
All low-level steps are implemented in the submodules under 01_preprocessing and 02_variants.
"""

import os
from pathlib import Path

# Import step modules
from bioinformatics_pipeline.b_preprocessing.trim_reads import trim_reads
from bioinformatics_pipeline.b_preprocessing.align_reads import align_reads
from bioinformatics_pipeline.b_preprocessing.post_alignment_qc import (
    run_post_alignment_qc,
)
from bioinformatics_pipeline.b_preprocessing.coverage_stats import compute_coverage

from bioinformatics_pipeline.c_variants.variant_calling_and_refinement import (
    call_and_refine_variants,
)
from bioinformatics_pipeline.c_variants.filter_variants import filter_variants
from bioinformatics_pipeline.c_variants.snpeff_annotation import annotate_variants


# Main Entrypoint
def run_bioinfo_pipeline(config):
    """
    Executes the entire bioinformatics workflow defined in bioinfo_config.yaml.
    """

    bio = config.get("bioinformatics", {})
    if not bio:
        raise ValueError("bioinformatics section missing in config YAML")

    print("\n========== BIOINFORMATICS PIPELINE ==========\n")

    # Trimming
    if bio["trimming"]["enabled"]:
        print("[1/5] Running read trimming...")
        trim_reads(bio)
    else:
        print("[1/5] Skipping trimming.")

    # Alignment
    if bio["alignment"]["enabled"]:
        print("[2/5] Running alignment...")
        align_reads(bio)
        run_post_alignment_qc(bio)
        compute_coverage(bio)
    else:
        print("[2/5] Skipping alignment.")

    # Variant Calling
    if bio["variant_calling"]["enabled"]:
        print("[3/5] Running variant calling + refinement...")
        call_and_refine_variants(bio)
    else:
        print("[3/5] Skipping variant calling.")

    # Variant Filtering
    if bio["refinement"]["enabled"]:
        print("[4/5] Running variant filtering...")
        filter_variants(bio)
    else:
        print("[4/5] Skipping variant filtering.")

    # SnpEff Annotation
    if bio["annotation"]["enabled"]:
        print("[5/5] Running SnpEff annotation...")
        annotate_variants(bio)
    else:
        print("[5/5] Skipping annotation.")

    print("\n🎉 Bioinformatics pipeline completed successfully!\n")
