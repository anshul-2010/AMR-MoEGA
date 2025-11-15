## Bioinformatics Pipeline Integration

We have integrated the core AMR-bioinformatics logic from the [ML-iAMR](https://github.com/YunxiaoRen/ML-iAMR) repository into this modular pipeline.

### Steps

1. **Trim & Align**  
   Run `bioinformatics_pipeline/b_preprocessing/align_and_trim.sh config/bioinf_config.yaml path/to/raw_fastq out/bam_dir`  
2. **Variant Calling**  
   Call variants using: `bioinformatics_pipeline/c_variants/call_variants.sh config/bioinf_config.yaml out/bam_dir out/vcf_dir`  
3. **Filter & Annotate**  
   Filter low-DP / low-QUAL variants and annotate:  
   `bioinformatics_pipeline/c_variants/filter_and_annotate.sh config/bioinf_config.yaml out/vcf_dir out/filtered out/annotated`  
4. **Build SNP Matrix**  
   Use `ml_pipeline/d_feature_engineering/build_snp_matrix.py --config config/bioinf_config.yaml --vcf_dir out/filtered` to generate a samples × variant matrix.

### Configuration

In `config/bioinf_config.yaml`, configure:
- `data.reference`: path to reference genome FASTA  
- `pipeline.threads`: number of threads  
- `variant_calling.min_dp`, `variant_calling.min_qual`  
- `annotation.snpeff_db`: name of SnpEff database  
- `snp_matrix.output_csv`: output CSV path  