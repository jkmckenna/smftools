def load_adata(config_path):
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
    from ..informatics.discover_input_files import discover_input_files 
    from ..informatics.bam_functions import concatenate_fastqs_to_bam, extract_read_features_from_bam
    from ..informatics.pod5_functions import fast5_to_pod5
    from ..informatics.fasta_functions import subsample_fasta_from_bed

    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc

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

    # Detect the input filetype
    if Path(input_data_path).is_file():
            suffix = input_data_path.suffix.lower()
            suffixes = [s.lower() for s in input_data_path.suffixes]  # handles multi-part extensions

            # recognize multi-suffix cases like .fastq.gz or .fq.gz
            input_is_pod5  = any(s in ['.pod5', '.p5'] for s in suffixes)
            input_is_fast5 = any(s in ['.fast5', '.f5'] for s in suffixes)
            input_is_fastq = any(s in ['.fastq', '.fq'] for s in suffixes)
            input_is_bam   = suffix == bam_suffix.lower()

            if input_is_fastq:
                fastq_paths = [str(input_data_path)]
    elif Path(input_data_path).is_dir():
        found = discover_input_files(input_data_path, bam_suffix=bam_suffix, recursive=cfg.recursive_input_search)

        input_is_pod5 = found["input_is_pod5"]
        input_is_fast5 = found["input_is_fast5"]
        input_is_fastq = found["input_is_fastq"]
        input_is_bam = found["input_is_bam"]

        pod5_paths = found["pod5_paths"]
        fast5_paths = found["fast5_paths"]
        fastq_paths = found["fastq_paths"]
        bam_paths = found["bam_paths"]

        print(f"Found {found['all_files_searched']} files; fastq={len(fastq_paths)}, bam={len(bam_paths)}, pod5={len(pod5_paths)}, fast5={len(fast5_paths)}")

    # If the input files are not pod5 files, and they are fast5 files, convert the files to a pod5 file before proceeding.
    if input_is_fast5 and not input_is_pod5:
        # take the input directory of fast5 files and write out a single pod5 file into the output directory.
        output_pod5 = output_directory / 'FAST5s_to_POD5.pod5'
        print(f'Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_pod5}')
        fast5_to_pod5(input_data_path, output_pod5)
        # Reassign the pod5_dir variable to point to the new pod5 file.
        input_data_path = output_pod5
        input_is_pod5 = True
        input_is_fast5 = False
    
    # If the input is a fastq or a directory of fastqs, concatenate them into an unaligned BAM and save the barcode
    elif input_is_fastq:
        # Output file for FASTQ concatenation.
        output_bam = output_directory / 'FASTQs_concatenated_into_BAM.bam'

        summary = concatenate_fastqs_to_bam(
            fastq_paths,
            output_bam,
            barcode_tag='BC',
            gzip_suffixes=('.gz','.gzip'),
            barcode_map=cfg.fastq_barcode_map,
            add_read_group=True,
            rg_sample_field=None,
            progress=False,
            auto_pair=cfg.fastq_auto_pairing)
        
        print(f"Found the following barcodes: {summary['barcodes']}")

        # Set the input data path to the concatenated BAM.
        input_data_path = output_bam
        input_is_bam = True
        input_is_fastq = False

    # Determine if the input data needs to be basecalled
    if input_is_pod5:
        basecall = True
    elif input_is_bam:
        basecall = False
    else:
        print('Error, can not find input bam or pod5')

    # Generate the base name of the unaligned bam without the .bam suffix
    if basecall:
        model_basename = Path(cfg.model).name
        model_basename = str(model_basename).replace('.', '_')
        if smf_modality == 'direct':
            mod_string = "_".join(cfg.mod_list)
            bam = output_directory / f"{model_basename}_{mod_string}_calls"
        else:
            bam = output_directory / f"{model_basename}_canonical_basecalls"
    else:
        bam_base = input_data_path.stem
        bam = output_directory / bam_base

    # Generate path names for the unaligned, aligned, as well as the aligned/sorted bam.
    unaligned_output = bam.with_suffix(bam_suffix)
    aligned_BAM = bam.with_name(bam.stem + "_aligned")
    aligned_output = aligned_BAM.with_suffix(bam_suffix)
    aligned_sorted_BAM =aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(bam_suffix)

    # Naming of the demultiplexed output directory
    if cfg.barcode_both_ends:
        split_path = split_path.with_name(split_path.name + '_both_ends_barcoded')
    else:
        split_path = split_path.with_name(split_path.name + '_at_least_one_end_barcoded')

    # Direct methylation detection SMF specific parameters
    if smf_modality == 'direct':
        mod_bed_dir = split_path / "split_mod_beds"
        mod_tsv_dir = split_path / "split_mod_tsvs"
        bam_qc_dir = split_path / "bam_qc"
        mod_map = {'6mA': '6mA', '5mC_5hmC': '5mC'}
        mods = [mod_map[mod] for mod in cfg.mod_list]
    ########################################################################################################################

    ################################### 2) FASTA Handling ###################################
    from ..informatics.fasta_functions import generate_converted_FASTA, get_chromosome_lengths

    # If fasta_regions_of_interest bed is passed, subsample the input FASTA on regions of interest and use the subsampled FASTA.
    if cfg.fasta_regions_of_interest and '.bed' in cfg.fasta_regions_of_interest:
        fasta_basename = fasta.parent / fasta.stem
        bed_basename_minus_suffix = Path(cfg.fasta_regions_of_interest).stem
        output_FASTA = fasta_basename.with_name(fasta_basename.name + '_subsampled_by_' + bed_basename_minus_suffix + '.fasta')
        subsample_fasta_from_bed(fasta, cfg.fasta_regions_of_interest, output_directory, output_FASTA)
        fasta = output_directory / output_FASTA

    # For conversion style SMF, make a converted reference FASTA
    if smf_modality == 'conversion':
        fasta_basename = fasta.parent / fasta.stem
        converted_FASTA_basename = fasta_basename.with_name(fasta_basename.name + '_converted.fasta') 
        converted_FASTA = output_directory / converted_FASTA_basename
        if 'converted.fa' in fasta:
            print(f'{fasta} is already converted. Using existing converted FASTA.')
            converted_FASTA = fasta
        elif converted_FASTA.exists():
            print(f'{converted_FASTA} already exists. Using existing converted FASTA.')
        else:
            generate_converted_FASTA(fasta, conversions, strands, converted_FASTA)
        fasta = converted_FASTA

    # Make a FAI and .chrom.names file for the fasta
    get_chromosome_lengths(fasta)
    ########################################################################################################################

    ################################### 3) Basecalling ###################################
    from ..informatics.basecalling import modcall, canoncall
    # 1) Basecall using dorado
    if basecall and cfg.sequencer == 'ont':
        if unaligned_output.exists():
            print(f'{unaligned_output} already exists. Using existing basecalled BAM.')
        elif smf_modality != 'direct':
            canoncall(str(cfg.model_dir), cfg.model, str(input_data_path), cfg.barcode_kit, str(bam), bam_suffix, cfg.barcode_both_ends, cfg.trim, cfg.device)
        else:
            modcall(str(cfg.model_dir), cfg.model, str(input_data_path), cfg.barcode_kit, cfg.mod_list, str(bam), bam_suffix, cfg.barcode_both_ends, cfg.trim, cfg.device)
    elif basecall:
        print(f"Basecalling is currently only supported for ont sequencers and not pacbio.")
    else:
        pass
    ########################################################################################################################

    ################################### 4) Alignment and sorting #############################################
    from ..informatics.bam_functions import align_and_sort_BAM
    from ..informatics.bed_functions import aligned_BAM_to_bed
    # 3) Align the BAM to the reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    if os.path.exists(aligned_output) and os.path.exists(aligned_sorted_output):
        print(f'{aligned_sorted_output} already exists. Using existing aligned/sorted BAM.')
    else:
        align_and_sort_BAM(fasta, unaligned_output, bam_suffix, output_directory, cfg.make_bigwigs, cfg.threads, cfg.aligner, cfg.aligner_args)

    # Make beds and provide basic histograms
    bed_dir = os.path.join(output_directory, 'beds')
    if os.path.isdir(bed_dir):
        print(f'{bed_dir} already exists. Skipping BAM -> BED conversion for {aligned_sorted_output}')
    else:
        aligned_BAM_to_bed(aligned_output, output_directory, fasta, cfg.make_bigwigs, cfg.threads)
    ########################################################################################################################

    ################################### 5) Demultiplexing ######################################################################
    from ..informatics.bam_functions import demux_and_index_BAM, split_and_index_BAM
    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if split_path.is_dir():
        print(f"{split_path} already exists. Using existing demultiplexed BAMs.")

        bam_files = sorted(
            p for p in split_path.iterdir()
            if p.is_file()
            and p.suffix == ".bam"
            and "unclassified" not in p.name
        )
    else:
        make_dirs([split_path])
        if cfg.input_already_demuxed:
            split_and_index_BAM(aligned_sorted_BAM, 
                                split_path, 
                                bam_suffix)
        else:
            bam_files = demux_and_index_BAM(aligned_sorted_BAM, 
                                            split_path, bam_suffix, 
                                            cfg.barcode_kit, 
                                            cfg.barcode_both_ends, 
                                            cfg.trim, 
                                            fasta, 
                                            cfg.make_bigwigs, 
                                            cfg.threads)

    # Make beds and provide basic histograms
    bed_dir = os.path.join(split_path, 'beds')
    if os.path.isdir(bed_dir):
        print(f'{bed_dir} already exists. Skipping BAM -> BED conversion for demultiplexed bams')
    else:
        for bam in bam_files:
            aligned_BAM_to_bed(bam, split_path, fasta, cfg.make_bigwigs, cfg.threads)
    ########################################################################################################################

    ################################### 6) SAMTools based BAM QC ######################################################################
    from ..informatics.bam_functions import bam_qc
    # 5) Samtools QC metrics on split BAM files
    bam_qc_dir = split_path / "bam_qc"
    if os.path.isdir(bam_qc_dir):
        print( f'{bam_qc_dir} already exists. Using existing BAM QC calculations.')
    else:
        make_dirs([bam_qc_dir])
        bam_qc(bam_files, bam_qc_dir, cfg.threads, modality=smf_modality)
    ######################################################################################################################## 

    ################################### 7) AnnData loading ######################################################################
    if smf_modality != 'direct':
        from ..informatics.converted_BAM_to_adata import converted_BAM_to_adata
        # 6) Take the converted BAM and load it into an adata object.
        if smf_modality == 'deaminase':
            deaminase_footprinting = True
        else:
            deaminase_footprinting = False
        raw_adata, raw_adata_path = converted_BAM_to_adata(fasta, 
                                                                  split_path, 
                                                                  cfg.mapping_threshold, 
                                                                  cfg.experiment_name, 
                                                                  conversions, 
                                                                  bam_suffix, 
                                                                  cfg.device, 
                                                                  cfg.threads, 
                                                                  deaminase_footprinting,
                                                                  delete_intermediates=cfg.delete_intermediate_hdfs) 
    else:
        if mod_bed_dir.is_dir():
            print(f'{mod_bed_dir} already exists, skipping making modbeds')
        else:
            from ..informatics.modkit_functions import modQC, make_modbed
            make_dirs([mod_bed_dir])  

            modQC(aligned_sorted_output, 
                  cfg.thresholds) # get QC metrics for mod calls
            
            make_modbed(aligned_sorted_output, 
                        cfg.thresholds, 
                        mod_bed_dir) # Generate bed files of position methylation summaries for every sample
            
        if mod_tsv_dir.is_dir():
            print(f'{mod_tsv_dir} already exists, skipping making modtsvs')
        else:
            from ..informatics.modkit_functions import extract_mods
            make_dirs([mod_tsv_dir])

            extract_mods(cfg.thresholds, 
                         mod_tsv_dir, 
                         split_path, 
                         bam_suffix, 
                         skip_unclassified=cfg.skip_unclassified, 
                         modkit_summary=False,
                         threads=cfg.threads) # Extract methylations calls for split BAM files into split TSV files
            
        from ..informatics.modkit_extract_to_adata import modkit_extract_to_adata

        #6 Load the modification data from TSVs into an adata object
        raw_adata, raw_adata_path = modkit_extract_to_adata(fasta, 
                                                                split_path, 
                                                                cfg.mapping_threshold, 
                                                                cfg.experiment_name, 
                                                                mods, 
                                                                cfg.batch_size, 
                                                                mod_tsv_dir, 
                                                                cfg.delete_batch_hdfs, 
                                                                cfg.threads)

    ########################################################################################################################

    ############################################### MultiQC HTML Report ###############################################
    from ..informatics.run_multiqc import run_multiqc
    # multiqc ###
    mqc_dir = split_path / "multiqc"
    if mqc_dir.is_dir():
        print(f'{mqc_dir} already exists, skipping multiqc')
    else:
        run_multiqc(split_path, mqc_dir)
    ########################################################################################################################

    return raw_adata, raw_adata_path