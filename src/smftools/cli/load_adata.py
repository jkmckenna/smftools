from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from smftools.logging_utils import get_logger

from .helpers import AdataPaths

logger = get_logger(__name__)


def check_executable_exists(cmd: str) -> bool:
    """Return True if a command-line executable is available in PATH."""
    return shutil.which(cmd) is not None


def delete_tsvs(
    tsv_dir: Union[str, Path, Iterable[str], None],
    *,
    dry_run: bool = False,
    verbose: bool = True,
):
    """
    Delete intermediate tsv files.

    Parameters
    ----------

    tsv_dir : str | Path | None
        Path to a directory to remove recursively (e.g. a tsv dir created earlier).
    dry_run : bool
        If True, print what *would* be removed but do not actually delete.
    verbose : bool
        Print progress / warnings.
    """

    # Helper: remove a single file path (Path-like or string)
    def _maybe_unlink(p: Path):
        if not p.exists():
            if verbose:
                logger.info(f"[skip] not found: {p}")
            return
        if not p.is_file():
            if verbose:
                logger.info(f"[skip] not a file: {p}")
            return
        if dry_run:
            logger.info(f"[dry-run] would remove file: {p}")
            return
        try:
            p.unlink()
            if verbose:
                logger.info(f"Removed file: {p}")
        except Exception as e:
            logger.warning(f"Failed to remove file {p}: {e}")

    # Remove tmp_dir recursively (if provided)
    if tsv_dir is not None:
        td = Path(tsv_dir)
        if not td.exists():
            if verbose:
                logger.info(f"[skip] tsv_dir not found: {td}")
        else:
            if not td.is_dir():
                if verbose:
                    logger.info(f"[skip] tsv_dir is not a directory: {td}")
            else:
                if dry_run:
                    logger.info(f"[dry-run] would remove directory tree: {td}")
                else:
                    try:
                        shutil.rmtree(td)
                        if verbose:
                            logger.info(f"Removed directory tree: {td}")
                    except Exception as e:
                        logger.warning(f"[error] failed to remove tmp dir {td}: {e}")


def load_adata(config_path: str):
    """
    CLI-facing wrapper for the load pipeline.

    - Reads config CSV into ExperimentConfig
    - Computes canonical paths for all downstream AnnData stages
    - Registers those in the summary CSV
    - Applies stage-skipping logic (hmm > spatial > pp_dedup > pp > raw)
    - If needed, calls the core pipeline to actually build the raw AnnData

    Returns
    -------
    adata : anndata.AnnData | None
        Newly created AnnData object, or None if we skipped because a later-stage
        AnnData already exists.
    adata_path : pathlib.Path
        Path to the "current" AnnData that should be used downstream.
    cfg : ExperimentConfig
        Config object for downstream steps.
    """
    from datetime import datetime
    from importlib import resources

    from ..config import ExperimentConfig, LoadExperimentConfig
    from ..readwrite import add_or_update_column_in_csv, make_dirs
    from .helpers import get_adata_paths

    date_str = datetime.today().strftime("%y%m%d")

    # -----------------------------
    # 1) Load config into cfg
    # -----------------------------
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, report = ExperimentConfig.from_var_dict(
        loader.var_dict, date_str=date_str, defaults_dir=defaults_dir
    )

    # Ensure base output dir
    make_dirs([cfg.output_directory])

    # -----------------------------
    # 2) Compute and register paths
    # -----------------------------
    paths = get_adata_paths(cfg)

    # experiment-level metadata in summary CSV
    add_or_update_column_in_csv(cfg.summary_file, "experiment_name", cfg.experiment_name)
    add_or_update_column_in_csv(cfg.summary_file, "config_path", config_path)
    add_or_update_column_in_csv(cfg.summary_file, "input_data_path", cfg.input_data_path)
    add_or_update_column_in_csv(cfg.summary_file, "input_files", [cfg.input_files])

    # AnnData stage paths
    add_or_update_column_in_csv(cfg.summary_file, "load_adata", paths.raw)
    add_or_update_column_in_csv(cfg.summary_file, "pp_adata", paths.pp)
    add_or_update_column_in_csv(cfg.summary_file, "pp_dedup_adata", paths.pp_dedup)
    add_or_update_column_in_csv(cfg.summary_file, "spatial_adata", paths.spatial)
    add_or_update_column_in_csv(cfg.summary_file, "hmm_adata", paths.hmm)

    # -----------------------------
    # 3) Stage skipping logic
    # -----------------------------
    if not getattr(cfg, "force_redo_load_adata", False):
        if paths.hmm.exists():
            logger.debug(f"HMM AnnData already exists: {paths.hmm}\nSkipping smftools load")
            return None, paths.hmm, cfg
        if paths.spatial.exists():
            logger.debug(f"Spatial AnnData already exists: {paths.spatial}\nSkipping smftools load")
            return None, paths.spatial, cfg
        if paths.pp_dedup.exists():
            logger.debug(
                f"Preprocessed deduplicated AnnData already exists: {paths.pp_dedup}\n"
                f"Skipping smftools load"
            )
            return None, paths.pp_dedup, cfg
        if paths.pp.exists():
            logger.debug(f"Preprocessed AnnData already exists: {paths.pp}\nSkipping smftools load")
            return None, paths.pp, cfg
        if paths.raw.exists():
            logger.debug(
                f"Raw AnnData from smftools load already exists: {paths.raw}\nSkipping smftools load"
            )
            return None, paths.raw, cfg

    # If we get here, we actually want to run the full load pipeline
    adata, adata_path, cfg = load_adata_core(cfg, paths, config_path=config_path)

    return adata, adata_path, cfg


def load_adata_core(cfg, paths: AdataPaths, config_path: str | None = None):
    """
    Core load pipeline.

    Assumes:
    - cfg is a fully initialized ExperimentConfig
    - paths is an AdataPaths object describing canonical h5ad stage paths
    - No stage-skipping or early returns based on existing AnnDatas are done here
      (that happens in the wrapper).

    Does:
    - handle input format (fast5/pod5/fastq/bam/h5ad)
    - basecalling / alignment / demux / BAM QC
    - optional bed + bigwig generation
    - AnnData construction (conversion or direct modality)
    - basic read-level QC annotations
    - write raw AnnData to paths.raw
    - run MultiQC
    - optional deletion of intermediate BAMs

    Returns
    -------
    raw_adata : anndata.AnnData
        Newly created raw AnnData object.
    raw_adata_path : Path
        Path where the raw AnnData was written (paths.raw).
    cfg : ExperimentConfig
        (Same object, possibly with some fields updated, e.g. fasta path.)
    """

    from ..informatics.bam_functions import (
        align_and_sort_BAM,
        bam_qc,
        concatenate_fastqs_to_bam,
        demux_and_index_BAM,
        extract_read_features_from_bam,
        split_and_index_BAM,
    )
    from ..informatics.basecalling import canoncall, modcall
    from ..informatics.bed_functions import aligned_BAM_to_bed
    from ..informatics.converted_BAM_to_adata import converted_BAM_to_adata
    from ..informatics.fasta_functions import (
        generate_converted_FASTA,
        get_chromosome_lengths,
        subsample_fasta_from_bed,
    )
    from ..informatics.h5ad_functions import add_read_length_and_mapping_qc
    from ..informatics.modkit_extract_to_adata import modkit_extract_to_adata
    from ..informatics.modkit_functions import extract_mods, make_modbed, modQC
    from ..informatics.pod5_functions import fast5_to_pod5
    from ..informatics.run_multiqc import run_multiqc
    from ..metadata import record_smftools_metadata
    from ..readwrite import add_or_update_column_in_csv, make_dirs
    from .helpers import write_gz_h5ad

    ################################### 1) General params and input organization ###################################
    output_directory = Path(cfg.output_directory)
    make_dirs([output_directory])

    raw_adata_path = paths.raw
    pp_adata_path = paths.pp
    pp_dup_rem_adata_path = paths.pp_dedup
    spatial_adata_path = paths.spatial
    hmm_adata_path = paths.hmm

    # Naming of the demultiplexed output directory
    double_barcoded_path = cfg.split_path / "both_ends_barcoded"
    single_barcoded_path = cfg.split_path / "at_least_one_end_barcoded"

    # Direct methylation detection SMF specific parameters
    if cfg.smf_modality == "direct":
        mod_bed_dir = cfg.output_directory / "mod_beds"
        add_or_update_column_in_csv(cfg.summary_file, "mod_bed_dir", mod_bed_dir)
        mod_tsv_dir = cfg.output_directory / "mod_tsvs"
        add_or_update_column_in_csv(cfg.summary_file, "mod_tsv_dir", mod_tsv_dir)
        bam_qc_dir = cfg.output_directory / "bam_qc"
        mods = [cfg.mod_map[mod] for mod in cfg.mod_list]

        if not check_executable_exists("dorado"):
            raise RuntimeError(
                "Error: 'dorado' is not installed or not in PATH. "
                "Install from https://github.com/nanoporetech/dorado"
            )
        if not check_executable_exists("modkit"):
            raise RuntimeError(
                "Error: 'modkit' is not installed or not in PATH. "
                "Install from https://github.com/nanoporetech/modkit"
            )
    else:
        mod_bed_dir = None
        mod_tsv_dir = None
        mods = None

    # demux / aligner executables
    if (not cfg.input_already_demuxed) or cfg.aligner == "dorado":
        if not check_executable_exists("dorado"):
            raise RuntimeError(
                "Error: 'dorado' is not installed or not in PATH. "
                "Install from https://github.com/nanoporetech/dorado"
            )

    if cfg.aligner == "minimap2":
        if not check_executable_exists("minimap2"):
            raise RuntimeError(
                "Error: 'minimap2' is not installed or not in PATH. Install minimap2"
            )

    # # Detect the input filetypes
    # If the input files are fast5 files, convert the files to a pod5 file before proceeding.
    if cfg.input_type == "fast5":
        # take the input directory of fast5 files and write out a single pod5 file into the output directory.
        output_pod5 = cfg.output_directory / "FAST5s_to_POD5.pod5"
        if output_pod5.exists():
            pass
        else:
            logger.info(
                f"Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_pod5}"
            )
            fast5_to_pod5(cfg.input_data_path, output_pod5)
        # Reassign the pod5_dir variable to point to the new pod5 file.
        cfg.input_data_path = output_pod5
        cfg.input_type = "pod5"
    # If the input is a fastq or a directory of fastqs, concatenate them into an unaligned BAM and save the barcode
    elif cfg.input_type == "fastq":
        # Output file for FASTQ concatenation.
        output_bam = cfg.output_directory / "canonical_basecalls.bam"
        if output_bam.exists():
            logger.debug("Output BAM already exists")
        else:
            logger.info("Concatenating FASTQ files into a single BAM file")
            summary = concatenate_fastqs_to_bam(
                cfg.input_files,
                output_bam,
                barcode_tag="BC",
                gzip_suffixes=(".gz", ".gzip"),
                barcode_map=cfg.fastq_barcode_map,
                add_read_group=True,
                rg_sample_field=None,
                progress=False,
                auto_pair=cfg.fastq_auto_pairing,
                samtools_backend=cfg.samtools_backend,
            )

            logger.info(f"Found the following barcodes in FASTQ inputs: {summary['barcodes']}")

        # Set the input data path to the concatenated BAM.
        cfg.input_data_path = output_bam
        cfg.input_type = "bam"
    elif cfg.input_type == "h5ad":
        pass
    else:
        pass

    add_or_update_column_in_csv(cfg.summary_file, "input_data_path", cfg.input_data_path)

    # Determine if the input data needs to be basecalled
    if cfg.input_type == "pod5":
        logger.info(f"Detected pod5 inputs: {cfg.input_files}")
        basecall = True
    elif cfg.input_type in ["bam"]:
        logger.info(f"Detected bam input: {cfg.input_files}")
        basecall = False
    else:
        logger.info("Error, can not find input bam or pod5")

    # Generate the base name of the unaligned bam without the .bam suffix
    if basecall:
        model_basename = Path(cfg.model).name
        model_basename = str(model_basename).replace(".", "_")
        if cfg.smf_modality == "direct":
            mod_string = "_".join(cfg.mod_list)
            bam = cfg.output_directory / f"{model_basename}_{mod_string}_calls"
        else:
            bam = cfg.output_directory / f"{model_basename}_canonical_basecalls"
    else:
        bam_base = cfg.input_data_path.name
        bam = cfg.output_directory / bam_base

    # Generate path names for the unaligned, aligned, as well as the aligned/sorted bam.
    unaligned_output = bam.with_suffix(cfg.bam_suffix)
    aligned_BAM = (
        cfg.output_directory / (bam.stem + "_aligned")
    )  # doing this allows specifying an input bam in a seperate directory as the aligned output bams
    aligned_output = aligned_BAM.with_suffix(cfg.bam_suffix)
    aligned_sorted_BAM = aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(cfg.bam_suffix)

    add_or_update_column_in_csv(cfg.summary_file, "basecalled_bam", unaligned_output)
    add_or_update_column_in_csv(cfg.summary_file, "aligned_bam", aligned_output)
    add_or_update_column_in_csv(cfg.summary_file, "sorted_bam", aligned_sorted_output)
    ########################################################################################################################

    ################################### 2) FASTA Handling ###################################

    try:
        cfg.fasta = Path(cfg.fasta)
    except Exception:
        logger.warning("Need to provide an input FASTA path to proceed with smftools load")

    # If fasta_regions_of_interest bed is passed, subsample the input FASTA on regions of interest and use the subsampled FASTA.
    if cfg.fasta_regions_of_interest and ".bed" in cfg.fasta_regions_of_interest:
        fasta_stem = cfg.fasta.stem
        bed_stem = Path(cfg.fasta_regions_of_interest).stem
        output_FASTA = cfg.output_directory / f"{fasta_stem}_subsampled_by_{bed_stem}.fasta"

        logger.info("Subsampling FASTA records using the provided BED file")
        subsample_fasta_from_bed(
            cfg.fasta, cfg.fasta_regions_of_interest, cfg.output_directory, output_FASTA
        )
        fasta = output_FASTA
    else:
        logger.info("Using the full FASTA file")
        fasta = cfg.fasta

    # For conversion style SMF, make a converted reference FASTA
    if cfg.smf_modality == "conversion":
        fasta_stem = fasta.stem
        converted_FASTA_basename = f"{fasta_stem}_converted.fasta"
        converted_FASTA = cfg.output_directory / converted_FASTA_basename

        if "converted.fa" in fasta.name:
            logger.info(f"{fasta} is already converted. Using existing converted FASTA.")
            converted_FASTA = fasta
        elif converted_FASTA.exists():
            logger.info(f"{converted_FASTA} already exists. Using existing converted FASTA.")
        else:
            logger.info(f"Converting FASTA base sequences")
            generate_converted_FASTA(fasta, cfg.conversion_types, cfg.strands, converted_FASTA)
        fasta = converted_FASTA

    add_or_update_column_in_csv(cfg.summary_file, "fasta", fasta)

    # Make a FAI and .chrom.names file for the fasta
    get_chromosome_lengths(fasta)
    ########################################################################################################################

    ################################### 3) Basecalling ###################################

    # 1) Basecall using dorado
    if basecall and cfg.sequencer == "ont":
        try:
            cfg.model_dir = Path(cfg.model_dir)
        except Exception:
            logger.warning(
                "Need to provide a valid path to a dorado model directory to use dorado basecalling"
            )
        if aligned_sorted_output.exists():
            logger.info(
                f"{aligned_sorted_output} already exists. Using existing basecalled, aligned, sorted BAM."
            )
        elif unaligned_output.exists():
            logger.info(f"{unaligned_output} already exists. Using existing basecalled BAM.")
        elif cfg.smf_modality != "direct":
            logger.info("Running canonical basecalling using dorado")
            canoncall(
                str(cfg.model_dir),
                cfg.model,
                str(cfg.input_data_path),
                cfg.barcode_kit,
                str(bam),
                cfg.bam_suffix,
                cfg.barcode_both_ends,
                cfg.trim,
                cfg.device,
            )
        else:
            logger.info("Running modified basecalling using dorado")
            modcall(
                str(cfg.model_dir),
                cfg.model,
                str(cfg.input_data_path),
                cfg.barcode_kit,
                cfg.mod_list,
                str(bam),
                cfg.bam_suffix,
                cfg.barcode_both_ends,
                cfg.trim,
                cfg.device,
            )
    elif basecall:
        logger.error("Basecalling is currently only supported for ont sequencers and not pacbio.")
    else:
        pass
    ########################################################################################################################

    ################################### 4) Alignment and sorting #############################################

    # 3) Align the BAM to the reference FASTA and sort the bam on positional coordinates. Also make an index and a bed file of mapped reads
    if aligned_sorted_output.exists():
        logger.debug(f"{aligned_sorted_output} already exists. Using existing aligned/sorted BAM.")
    else:
        logger.info(f"Aligning and sorting reads")
        align_and_sort_BAM(fasta, unaligned_output, cfg)
        # Deleted the unsorted aligned output
        aligned_output.unlink()

    if cfg.make_beds:
        # Make beds and provide basic histograms
        bed_dir = cfg.output_directory / "beds"
        if bed_dir.is_dir():
            logger.debug(
                f"{bed_dir} already exists. Skipping BAM -> BED conversion for {aligned_sorted_output}"
            )
        else:
            logger.info("Making bed files from the aligned and sorted BAM file")
            aligned_BAM_to_bed(
                aligned_sorted_output,
                cfg.output_directory,
                fasta,
                cfg.make_bigwigs,
                cfg.threads,
                samtools_backend=cfg.samtools_backend,
                bedtools_backend=cfg.bedtools_backend,
                bigwig_backend=cfg.bigwig_backend,
            )
    ########################################################################################################################

    ################################### 5) Demultiplexing ######################################################################

    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if cfg.input_already_demuxed:
        if cfg.split_path.is_dir():
            logger.debug(f"{cfg.split_path} already exists. Using existing demultiplexed BAMs.")

            all_bam_files = sorted(
                p for p in cfg.split_path.iterdir() if p.is_file() and p.suffix == cfg.bam_suffix
            )
            unclassified_bams = [p for p in all_bam_files if "unclassified" in p.name]
            bam_files = [p for p in all_bam_files if "unclassified" not in p.name]

        else:
            make_dirs([cfg.split_path])
            logger.info("Demultiplexing samples into individual aligned/sorted BAM files")
            all_bam_files = split_and_index_BAM(
                aligned_sorted_BAM,
                cfg.split_path,
                cfg.bam_suffix,
                samtools_backend=cfg.samtools_backend,
            )

            unclassified_bams = [p for p in all_bam_files if "unclassified" in p.name]
            bam_files = sorted(p for p in all_bam_files if "unclassified" not in p.name)

        se_bam_files = bam_files
        bam_dir = cfg.split_path

    else:
        if single_barcoded_path.is_dir():
            logger.debug(
                f"{single_barcoded_path} already exists. Using existing single ended demultiplexed BAMs."
            )

            all_se_bam_files = sorted(
                p
                for p in single_barcoded_path.iterdir()
                if p.is_file() and p.suffix == cfg.bam_suffix
            )
            unclassified_se_bams = [p for p in all_se_bam_files if "unclassified" in p.name]
            se_bam_files = [p for p in all_se_bam_files if "unclassified" not in p.name]
        else:
            make_dirs([cfg.split_path, single_barcoded_path])
            logger.info(
                "Demultiplexing samples into individual aligned/sorted BAM files based on single end barcode status with Dorado"
            )
            all_se_bam_files = demux_and_index_BAM(
                aligned_sorted_BAM,
                single_barcoded_path,
                cfg.bam_suffix,
                cfg.barcode_kit,
                False,
                cfg.trim,
                cfg.threads,
            )

            unclassified_se_bams = [p for p in all_se_bam_files if "unclassified" in p.name]
            se_bam_files = [p for p in all_se_bam_files if "unclassified" not in p.name]

        if double_barcoded_path.is_dir():
            logger.debug(
                f"{double_barcoded_path} already exists. Using existing double ended demultiplexed BAMs."
            )

            all_de_bam_files = sorted(
                p
                for p in double_barcoded_path.iterdir()
                if p.is_file() and p.suffix == cfg.bam_suffix
            )
            unclassified_de_bams = [p for p in all_de_bam_files if "unclassified" in p.name]
            de_bam_files = [p for p in all_de_bam_files if "unclassified" not in p.name]
        else:
            make_dirs([cfg.split_path, double_barcoded_path])
            logger.info(
                "Demultiplexing samples into individual aligned/sorted BAM files based on double end barcode status with Dorado"
            )
            all_de_bam_files = demux_and_index_BAM(
                aligned_sorted_BAM,
                double_barcoded_path,
                cfg.bam_suffix,
                cfg.barcode_kit,
                True,
                cfg.trim,
                cfg.threads,
            )

            unclassified_de_bams = [p for p in all_de_bam_files if "unclassified" in p.name]
            de_bam_files = [p for p in all_de_bam_files if "unclassified" not in p.name]

        bam_files = se_bam_files + de_bam_files
        unclassified_bams = unclassified_se_bams + unclassified_de_bams
        bam_dir = single_barcoded_path

    add_or_update_column_in_csv(cfg.summary_file, "demuxed_bams", [se_bam_files])

    if cfg.make_beds:
        # Make beds and provide basic histograms
        bed_dir = cfg.split_path / "beds"
        if bed_dir.is_dir():
            logger.debug(
                f"{bed_dir} already exists. Skipping BAM -> BED conversion for demultiplexed bams"
            )
        else:
            logger.info("Making BED files from BAM files for each sample")
            for bam in bam_files:
                aligned_BAM_to_bed(
                    bam,
                    cfg.split_path,
                    fasta,
                    cfg.make_bigwigs,
                    cfg.threads,
                    samtools_backend=cfg.samtools_backend,
                    bedtools_backend=cfg.bedtools_backend,
                    bigwig_backend=cfg.bigwig_backend,
                )
    ########################################################################################################################

    ################################### 6) SAMTools based BAM QC ######################################################################

    # 5) Samtools QC metrics on split BAM files
    bam_qc_dir = cfg.split_path / "bam_qc"
    if bam_qc_dir.is_dir():
        logger.debug(f"{bam_qc_dir} already exists. Using existing BAM QC calculations.")
    else:
        make_dirs([bam_qc_dir])
        logger.info("Performing BAM QC")
        bam_qc(
            bam_files,
            bam_qc_dir,
            cfg.threads,
            modality=cfg.smf_modality,
            samtools_backend=cfg.samtools_backend,
        )
    ########################################################################################################################

    ################################### 7) AnnData loading ######################################################################
    if cfg.smf_modality != "direct":
        from ..informatics.converted_BAM_to_adata import converted_BAM_to_adata

        # 6) Take the converted BAM and load it into an adata object.
        if cfg.smf_modality == "deaminase":
            deaminase_footprinting = True
        else:
            deaminase_footprinting = False

        logger.info(f"Loading Anndata from BAM files for {cfg.smf_modality} footprinting")
        raw_adata, raw_adata_path = converted_BAM_to_adata(
            fasta,
            bam_dir,
            cfg.output_directory,
            cfg.input_already_demuxed,
            cfg.mapping_threshold,
            cfg.experiment_name,
            cfg.conversion_types,
            cfg.bam_suffix,
            cfg.device,
            cfg.threads,
            deaminase_footprinting,
            delete_intermediates=cfg.delete_intermediate_hdfs,
            double_barcoded_path=double_barcoded_path,
            samtools_backend=cfg.samtools_backend,
        )
    else:
        if mod_bed_dir.is_dir():
            logger.debug(f"{mod_bed_dir} already exists, skipping making modbeds")
        else:
            from ..informatics.modkit_functions import make_modbed, modQC

            make_dirs([mod_bed_dir])

            logger.info("Performing modQC for direct footprinting samples")

            modQC(aligned_sorted_output, cfg.thresholds)  # get QC metrics for mod calls

            logger.info("Making modified BED files for direct footprinting samples")

            make_modbed(
                aligned_sorted_output, cfg.thresholds, mod_bed_dir
            )  # Generate bed files of position methylation summaries for every sample

        from ..informatics.modkit_functions import extract_mods

        make_dirs([mod_tsv_dir])

        logger.info(
            "Extracting single read modification states into TSVs for direct footprinting samples"
        )

        extract_mods(
            cfg.thresholds,
            mod_tsv_dir,
            bam_dir,
            cfg.bam_suffix,
            skip_unclassified=cfg.skip_unclassified,
            modkit_summary=False,
            threads=cfg.threads,
        )  # Extract methylations calls for split BAM files into split TSV files

        from ..informatics.modkit_extract_to_adata import modkit_extract_to_adata

        logger.info("Making Anndata for direct modification detection SMF samples")

        # 6 Load the modification data from TSVs into an adata object
        raw_adata, raw_adata_path = modkit_extract_to_adata(
            fasta,
            bam_dir,
            cfg.output_directory,
            cfg.input_already_demuxed,
            cfg.mapping_threshold,
            cfg.experiment_name,
            mods,
            cfg.batch_size,
            mod_tsv_dir,
            cfg.delete_batch_hdfs,
            cfg.threads,
            double_barcoded_path,
            cfg.samtools_backend,
        )
        if cfg.delete_intermediate_tsvs:
            delete_tsvs(mod_tsv_dir)

    raw_adata.obs["Experiment_name"] = [cfg.experiment_name] * raw_adata.shape[0]
    raw_adata.obs["Experiment_name_and_barcode"] = (
        raw_adata.obs["Experiment_name"].astype(str) + "_" + raw_adata.obs["Barcode"].astype(str)
    )

    ########################################################################################################################

    ############################################### Add basic read length, read quality, mapping quality stats ###############################################

    logger.info("Adding read length, mapping quality, and modification signal to Anndata")
    add_read_length_and_mapping_qc(
        raw_adata,
        se_bam_files,
        extract_read_features_from_bam_callable=extract_read_features_from_bam,
        bypass=cfg.bypass_add_read_length_and_mapping_qc,
        force_redo=cfg.force_redo_add_read_length_and_mapping_qc,
        samtools_backend=cfg.samtools_backend,
    )

    raw_adata.obs["Raw_modification_signal"] = np.nansum(raw_adata.X, axis=1)
    ########################################################################################################################

    ############################################### if input data type was pod5, append the pod5 file origin to each read ###############################################
    from ..informatics.h5ad_functions import annotate_pod5_origin

    if cfg.input_type == "pod5":
        logger.info("Adding the POD5 origin file to each read into Anndata")
        annotate_pod5_origin(
            raw_adata,
            cfg.input_data_path,
            n_jobs=cfg.threads,
            csv_path=output_directory / "read_to_pod5_origin_mapping.csv",
        )
    ########################################################################################################################

    ############################################### Save final adata ###############################################
    logger.info(f"Saving AnnData to {raw_adata_path}")
    record_smftools_metadata(
        raw_adata,
        step_name="load",
        cfg=cfg,
        config_path=config_path,
        output_path=raw_adata_path,
    )
    write_gz_h5ad(raw_adata, raw_adata_path)
    ########################################################################################################################

    ############################################### MultiQC HTML Report ###############################################

    # multiqc ###
    mqc_dir = cfg.split_path / "multiqc"
    if mqc_dir.is_dir():
        logger.info(f"{mqc_dir} already exists, skipping multiqc")
    else:
        logger.info("Running multiqc")
        run_multiqc(cfg.split_path, mqc_dir)
    ########################################################################################################################

    ############################################### delete intermediate BAM files ###############################################
    if cfg.delete_intermediate_bams:
        logger.info("Deleting intermediate BAM files")
        # delete aligned and sorted bam
        aligned_sorted_output.unlink()
        bai = aligned_sorted_output.parent / (aligned_sorted_output.name + ".bai")
        bai.unlink()
        # delete the demultiplexed bams. Keep the demultiplexing summary files and directories to faciliate demultiplexing in the future with these files
        for bam in bam_files:
            bai = bam.parent / (bam.name + ".bai")
            bam.unlink()
            bai.unlink()
        for bam in unclassified_bams:
            bai = bam.parent / (bam.name + ".bai")
            bam.unlink()
            bai.unlink()
        logger.info("Finished deleting intermediate BAM files")
    ########################################################################################################################

    return raw_adata, raw_adata_path, cfg
