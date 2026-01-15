def flow_I(config_path):
    """
    High-level function to call for converting raw sequencing data to an adata object. 
    Command line accesses this through smftools load <config_path>
    Works for nanopore pod5, fast5, and unaligned modBAM data types for direct SMF workflows.
    Works for nanopore pod5, fast5, unaligned BAM for conversion SMF workflows.
    Also works for illumina fastq and unaligned BAM for conversion SMF workflows.

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        None
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad, make_dirs
    from ..config import LoadExperimentConfig, ExperimentConfig
    from .load_adata import load_adata
    from .preprocess_adata import preprocess_adata
    from .spatial_adata import spatial_adata

    import numpy as np
    import pandas as pd
    import anndata as ad
    from smftools.optional_imports import require

    sc = require("scanpy", extra="omics", purpose="archived CLI workflows")

    import os
    from importlib import resources
    from pathlib import Path

    from datetime import datetime
    date_str = datetime.today().strftime("%y%m%d")
    ################################### 1) General params and input organization ###################################
    # Load experiment config parameters into global variables
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, report = ExperimentConfig.from_var_dict(loader.var_dict, date_str=date_str, defaults_dir=defaults_dir)

    # General config variable init - Necessary user passed inputs
    smf_modality = cfg.smf_modality # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    input_data_path = Path(cfg.input_data_path)  # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = Path(cfg.output_directory)  # Path to the output directory to make for the analysis. Necessary.
    fasta = Path(cfg.fasta)  # Path to reference FASTA. Necessary.
    split_dir = Path(cfg.split_dir) # Relative path to directory for demultiplexing reads
    split_path = output_directory / split_dir # Absolute path to directory for demultiplexing reads

    # Make initial output directory
    make_dirs([output_directory])

    bam_suffix = cfg.bam_suffix
    strands = cfg.strands

    # General config variable init - Optional user passed inputs for enzyme base specificity
    mod_target_bases = cfg.mod_target_bases  # Nucleobases of interest that may be modified. ['GpC', 'CpG', 'C', 'A']

    # Conversion/deamination specific variable init
    conversion_types = cfg.conversion_types  # 5mC
    conversions = cfg.conversions

    # Common Anndata accession params
    reference_column = cfg.reference_column

    # If conversion_types is passed:
    if conversion_types:
        conversions += conversion_types

    ############################################### smftools load start ###############################################
    initial_adata, initial_adata_path = load_adata(config_path)

    # Initial adata path info
    initial_backup_dir = initial_adata_path.parent / 'adata_accessory_data'
    ############################################### smftools load end ###############################################

    ############################################### smftools preprocess start ###############################################
    pp_adata, pp_adata_path, pp_dedup_adata, pp_dup_rem_adata_path = preprocess_adata(config_path)

    # Preprocessed adata path info
    pp_adata_basename = initial_adata_path.with_suffix("").name + '_preprocessed.h5ad.gz'
    pp_adata_path = initial_adata_path / pp_adata_basename
    pp_backup_dir = pp_adata_path.parent / 'pp_adata_accessory_data'

    # Preprocessed duplicate removed adata path info
    pp_dup_rem_adata_basename = pp_adata_path.with_suffix("").name + '_duplicates_removed.h5ad.gz'
    pp_dup_rem_adata_path = pp_adata_path / pp_dup_rem_adata_basename
    pp_dup_rem_backup_dir= pp_adata_path.parent / 'pp_dup_rem_adata_accessory_data'
    ############################################### smftools preprocess end ###############################################

    ############################################### smftools spatial start ###############################################
    # Preprocessed duplicate removed adata with basic analyses appended path info
    basic_analyzed_adata_basename = pp_dup_rem_adata_path.with_suffix("").name + '_analyzed_I.h5ad.gz'
    basic_analyzed_adata_path = pp_dup_rem_adata_path / basic_analyzed_adata_basename
    basic_analyzed_backup_dir= pp_dup_rem_adata_path.parent /'duplicate_removed_analyzed_adata_I_accessory_data'

    spatial_adata, spatial_adata_path = spatial_adata(config_path)
    ############################################### smftools spatial end ###############################################
