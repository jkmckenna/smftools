from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from smftools.constants import BARCODE_KIT_ALIASES, LOAD_DIR, LOGGING_DIR, UMI_KIT_ALIASES
from smftools.logging_utils import get_logger, setup_logging

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

    from ..readwrite import make_dirs
    from .helpers import get_adata_paths, load_experiment_config

    # -----------------------------
    # 1) Load config into cfg
    # -----------------------------
    cfg = load_experiment_config(config_path)

    # Ensure base output dir
    output_directory = Path(cfg.output_directory)
    make_dirs([output_directory])

    # -----------------------------
    # 2) Compute and register paths
    # -----------------------------
    paths = get_adata_paths(cfg)

    # -----------------------------
    # 3) Stage skipping logic
    # -----------------------------
    if not getattr(cfg, "force_redo_load_adata", False):
        if paths.raw.exists():
            logger.info(
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
    from datetime import datetime

    from ..informatics.bam_functions import (
        BarcodeKitConfig,
        _bam_has_barcode_info_tags,
        _build_flanking_from_adapters,
        _get_dorado_version,
        align_and_sort_BAM,
        annotate_demux_type_from_bi_tag,
        annotate_umi_tags_in_bam,
        bam_qc,
        concatenate_fastqs_to_bam,
        demux_and_index_BAM,
        extract_and_assign_barcodes_in_bam,
        extract_read_features_from_bam,
        extract_read_tags_from_bam,
        load_barcode_references_from_yaml,
        load_umi_config_from_yaml,
        resolve_barcode_config,
        resolve_umi_config,
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
    from ..informatics.h5ad_functions import (
        add_demux_type_from_bm_tag,
        add_read_length_and_mapping_qc,
        add_read_tag_annotations,
        add_secondary_supplementary_alignment_flags,
        expand_bi_tag_columns,
    )
    from ..informatics.modkit_extract_to_adata import modkit_extract_to_adata
    from ..informatics.modkit_functions import extract_mods, make_modbed, modQC
    from ..informatics.pod5_functions import fast5_to_pod5
    from ..informatics.run_multiqc import run_multiqc
    from ..metadata import record_smftools_metadata
    from ..readwrite import add_or_update_column_in_csv, make_dirs
    from .helpers import write_gz_h5ad

    ################################### 1) General params and input organization ###################################
    date_str = datetime.today().strftime("%y%m%d")
    now = datetime.now()
    time_str = now.strftime("%H%M%S")

    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    output_directory = Path(cfg.output_directory)
    load_directory = output_directory / LOAD_DIR
    logging_directory = load_directory / LOGGING_DIR

    make_dirs([output_directory, load_directory])

    if cfg.emit_log_file:
        log_file = logging_directory / f"{date_str}_{time_str}_log.log"
        make_dirs([logging_directory])
    else:
        log_file = None

    setup_logging(level=log_level, log_file=log_file, reconfigure=log_file is not None)

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
        mod_bed_dir = load_directory / "mod_beds"
        mod_tsv_dir = load_directory / "mod_tsvs"
        bam_qc_dir = load_directory / "bam_qc"
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
        output_pod5 = load_directory / "FAST5s_to_POD5.pod5"
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
        output_bam = load_directory / "canonical_basecalls.bam"
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
            bam = load_directory / f"{model_basename}_{mod_string}_calls"
        else:
            bam = load_directory / f"{model_basename}_canonical_basecalls"
    else:
        bam_base = cfg.input_data_path.stem
        bam = cfg.input_data_path.parent / bam_base

    # Generate path names for the unaligned, aligned, as well as the aligned/sorted bam.
    unaligned_output = bam.with_suffix(cfg.bam_suffix)

    aligned_BAM = (
        load_directory / (bam.stem + "_aligned")
    )  # doing this allows specifying an input bam in a seperate directory as the aligned output bams

    aligned_output = aligned_BAM.with_suffix(cfg.bam_suffix)
    aligned_sorted_BAM = aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(cfg.bam_suffix)

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
        output_FASTA = load_directory / f"{fasta_stem}_subsampled_by_{bed_stem}.fasta"

        logger.info("Subsampling FASTA records using the provided BED file")
        subsample_fasta_from_bed(
            cfg.fasta, cfg.fasta_regions_of_interest, load_directory, output_FASTA
        )
        fasta = output_FASTA
    else:
        logger.info("Using the full FASTA file")
        fasta = cfg.fasta

    # For conversion style SMF, make a converted reference FASTA
    if cfg.smf_modality == "conversion":
        fasta_stem = fasta.stem
        converted_FASTA_basename = f"{fasta_stem}_converted.fasta"
        converted_FASTA = load_directory / converted_FASTA_basename

        if "converted.fa" in fasta.name:
            logger.info(f"{fasta} is already converted. Using existing converted FASTA.")
            converted_FASTA = fasta
        elif converted_FASTA.exists():
            logger.info(f"{converted_FASTA} already exists. Using existing converted FASTA.")
        else:
            logger.info(f"Converting FASTA base sequences")
            generate_converted_FASTA(fasta, cfg.conversion_types, cfg.strands, converted_FASTA)
        fasta = converted_FASTA

    # Make a FAI and .chrom.names file for the fasta
    get_chromosome_lengths(fasta)
    ########################################################################################################################

    ################################### 3) Basecalling ###################################

    demux_backend = str(getattr(cfg, "demux_backend", "dorado") or "dorado").strip().lower()
    if demux_backend not in {"smftools", "dorado"}:
        raise ValueError("demux_backend must be one of: smftools, dorado")

    # Validate demux configuration up front for clearer errors.
    if not cfg.input_already_demuxed:
        if demux_backend == "smftools":
            if not cfg.barcode_kit:
                raise ValueError("demux_backend='smftools' requires barcode_kit to be set.")
            if cfg.barcode_kit == "custom" and not cfg.custom_barcode_yaml:
                raise ValueError(
                    "demux_backend='smftools' with barcode_kit='custom' requires custom_barcode_yaml."
                )
            if cfg.barcode_kit != "custom" and cfg.barcode_kit not in BARCODE_KIT_ALIASES:
                raise ValueError(
                    "demux_backend='smftools' requires barcode_kit to be 'custom' with custom_barcode_yaml, "
                    f"or one of BARCODE_KIT_ALIASES: {list(BARCODE_KIT_ALIASES.keys())}"
                )
        else:
            if not cfg.barcode_kit:
                raise ValueError("demux_backend='dorado' requires barcode_kit.")
            if cfg.barcode_kit == "custom":
                raise ValueError(
                    "demux_backend='dorado' does not support barcode_kit='custom'. "
                    "Use demux_backend='smftools' with custom_barcode_yaml."
                )

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
            dorado_kit_name = cfg.barcode_kit if cfg.barcode_kit != "custom" else None
            canoncall(
                str(cfg.model_dir),
                cfg.model,
                str(cfg.input_data_path),
                dorado_kit_name,
                str(bam),
                cfg.bam_suffix,
                cfg.barcode_both_ends,
                cfg.trim,
                cfg.device,
                cfg.emit_moves,
            )
        else:
            logger.info("Running modified basecalling using dorado")
            dorado_kit_name = cfg.barcode_kit if cfg.barcode_kit != "custom" else None
            modcall(
                str(cfg.model_dir),
                cfg.model,
                str(cfg.input_data_path),
                dorado_kit_name,
                cfg.mod_list,
                str(bam),
                cfg.bam_suffix,
                cfg.barcode_both_ends,
                cfg.trim,
                cfg.device,
                cfg.emit_moves,
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
        align_and_sort_BAM(fasta, unaligned_output, aligned_output, cfg)
        # Deleted the unsorted aligned output
        aligned_output.unlink()

    if cfg.make_beds:
        # Make beds and provide basic histograms
        bed_dir = load_directory / "beds"
        if bed_dir.is_dir():
            logger.debug(
                f"{bed_dir} already exists. Skipping BAM -> BED conversion for {aligned_sorted_output}"
            )
        else:
            logger.info("Making bed files from the aligned and sorted BAM file")
            aligned_BAM_to_bed(
                aligned_sorted_output,
                load_directory,
                fasta,
                cfg.make_bigwigs,
                cfg.threads,
                samtools_backend=cfg.samtools_backend,
                bedtools_backend=cfg.bedtools_backend,
                bigwig_backend=cfg.bigwig_backend,
            )
    ########################################################################################################################

    ################################### 4.5) Optional UMI annotation #############################################
    if getattr(cfg, "use_umi", False):
        logger.info("Annotating UMIs in aligned and sorted BAM before demultiplexing")

        # Resolve UMI kit alias or custom YAML path
        umi_kit_config = None
        umi_kit = getattr(cfg, "umi_kit", None)
        umi_yaml_path = getattr(cfg, "umi_yaml", None)
        if umi_kit and umi_kit != "custom":
            if umi_kit not in UMI_KIT_ALIASES:
                raise ValueError(
                    f"Unknown umi_kit '{umi_kit}'. "
                    f"Available aliases: {list(UMI_KIT_ALIASES.keys())} or use 'custom' with umi_yaml."
                )
            umi_yaml_path = UMI_KIT_ALIASES[umi_kit]
            logger.info(f"Using UMI kit alias '{umi_kit}' -> {umi_yaml_path}")
        elif umi_kit == "custom" and not umi_yaml_path:
            raise ValueError("umi_kit='custom' requires umi_yaml path to be specified.")
        if umi_yaml_path:
            logger.info(f"Loading UMI config from YAML: {umi_yaml_path}")
            umi_kit_config = load_umi_config_from_yaml(umi_yaml_path)
        resolved_umi = resolve_umi_config(umi_kit_config, cfg)

        annotate_umi_tags_in_bam(
            aligned_sorted_output,
            use_umi=True,
            umi_kit_config=umi_kit_config,
            umi_length=getattr(cfg, "umi_length", None),
            umi_search_window=getattr(cfg, "umi_search_window", 200),
            umi_adapter_matcher=getattr(cfg, "umi_adapter_matcher", "edlib"),
            umi_adapter_max_edits=resolved_umi["umi_adapter_max_edits"],
            samtools_backend=cfg.samtools_backend,
            umi_ends=resolved_umi["umi_ends"],
            umi_flank_mode=resolved_umi["umi_flank_mode"],
            umi_amplicon_max_edits=resolved_umi["umi_amplicon_max_edits"],
            same_orientation=resolved_umi.get("same_orientation", False),
            threads=cfg.threads,
        )
    ########################################################################################################################

    ################################### 4.6) Optional smftools barcode extraction #############################################
    use_smftools_demux = demux_backend == "smftools"
    if use_smftools_demux and cfg.barcode_kit:
        # Resolve barcode YAML path from kit alias or custom path
        if cfg.barcode_kit == "custom":
            if not cfg.custom_barcode_yaml:
                raise ValueError(
                    "barcode_kit='custom' requires custom_barcode_yaml path to be specified"
                )
            barcode_yaml_path = cfg.custom_barcode_yaml
        elif cfg.barcode_kit in BARCODE_KIT_ALIASES:
            barcode_yaml_path = BARCODE_KIT_ALIASES[cfg.barcode_kit]
            logger.info(f"Using barcode kit alias '{cfg.barcode_kit}' -> {barcode_yaml_path}")
        else:
            raise ValueError(
                f"Unknown barcode_kit '{cfg.barcode_kit}' for smftools demux backend. "
                f"Available aliases: {list(BARCODE_KIT_ALIASES.keys())} or use 'custom' with custom_barcode_yaml."
            )

        logger.info("Loading barcode references from YAML")
        yaml_result = load_barcode_references_from_yaml(barcode_yaml_path)

        # Handle both old format (tuple) and new format (BarcodeKitConfig)
        if isinstance(yaml_result, BarcodeKitConfig):
            barcode_kit_config = yaml_result
            barcode_references = barcode_kit_config.barcodes
            barcode_length = barcode_kit_config.barcode_length
        else:
            barcode_references, barcode_length = yaml_result
            # Build a BarcodeKitConfig from legacy adapters for flanking support
            legacy_adapters = getattr(cfg, "barcode_adapters", [None, None])
            flanking = (
                _build_flanking_from_adapters(legacy_adapters)
                if any(a is not None for a in (legacy_adapters or []))
                else None
            )
            barcode_kit_config = BarcodeKitConfig(
                barcodes=barcode_references,
                barcode_length=barcode_length,
                flanking=flanking,
            )

        logger.info(
            f"Loaded {len(barcode_references)} barcode references (length={barcode_length})"
        )
        resolved_bc = resolve_barcode_config(barcode_kit_config, cfg)

        logger.info("Extracting and assigning barcodes to aligned BAM using smftools backend")
        barcoded_bam = extract_and_assign_barcodes_in_bam(
            aligned_sorted_output,
            barcode_adapters=getattr(cfg, "barcode_adapters", [None, None]),
            barcode_references=barcode_references,
            barcode_length=barcode_length,
            barcode_search_window=getattr(cfg, "barcode_search_window", 200),
            barcode_max_edit_distance=resolved_bc["barcode_max_edit_distance"],
            barcode_adapter_matcher=getattr(cfg, "barcode_adapter_matcher", "edlib"),
            barcode_composite_max_edits=resolved_bc["barcode_composite_max_edits"],
            barcode_min_separation=resolved_bc.get("barcode_min_separation"),
            require_both_ends=getattr(cfg, "barcode_both_ends", False),
            min_barcode_score=getattr(cfg, "barcode_min_score", None),
            samtools_backend=cfg.samtools_backend,
            barcode_kit_config=barcode_kit_config,
            barcode_ends=resolved_bc["barcode_ends"],
            barcode_amplicon_gap_tolerance=resolved_bc["barcode_amplicon_gap_tolerance"],
        )
        # Update aligned_sorted_output to point to the barcoded BAM
        aligned_sorted_output = barcoded_bam
        logger.info(f"smftools barcode extraction complete: {barcoded_bam}")
    ########################################################################################################################

    ################################### 5) Demultiplexing ######################################################################

    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if cfg.input_already_demuxed or use_smftools_demux:
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
        double_barcoded_path = None

    else:
        # --- Dorado demux: version-aware branching ---
        dorado_version = _get_dorado_version()
        use_single_pass = dorado_version is not None and dorado_version >= (1, 3, 1)

        if use_single_pass:
            # Check what barcode tags are already present in the BAM
            tag_info = _bam_has_barcode_info_tags(aligned_sorted_output)

            if tag_info["has_bc"] and tag_info["has_bi"]:
                # Best case: basecalling already classified with per-end scoring info
                logger.info(
                    "Dorado basecalling already classified barcodes with scoring info (bi/bv tags). "
                    "Using --no-classify for demux."
                )
                demux_mode = "no_classify"
            elif tag_info["has_bc"]:
                # BC tags from older basecalling, but new dorado available — re-classify
                logger.info(
                    "BC tags present but no bi/bv scoring tags. "
                    "Re-classifying barcodes with dorado >= 1.3.1 to get per-end scoring info."
                )
                demux_mode = "classify"
            else:
                # No BC tags — need full classification
                logger.info("No existing barcode tags. Running full dorado demux classification.")
                demux_mode = "classify"

            # Single-pass demux into split_path directly (no se_/de_ subdirectories)
            if cfg.split_path.is_dir():
                logger.debug(f"{cfg.split_path} already exists. Using existing demultiplexed BAMs.")
                all_bam_files = sorted(
                    p
                    for p in cfg.split_path.iterdir()
                    if p.is_file() and p.suffix == cfg.bam_suffix
                )
                unclassified_bams = [p for p in all_bam_files if "unclassified" in p.name]
                bam_files = [p for p in all_bam_files if "unclassified" not in p.name]
            else:
                make_dirs([cfg.split_path])
                logger.info(
                    "Demultiplexing with dorado (single-pass, version %s)",
                    ".".join(str(v) for v in dorado_version),
                )
                all_bam_files = demux_and_index_BAM(
                    aligned_sorted_BAM,
                    cfg.split_path,
                    cfg.bam_suffix,
                    cfg.barcode_kit,
                    barcode_both_ends=False,
                    trim=cfg.trim,
                    threads=cfg.threads,
                    no_classify=(demux_mode == "no_classify"),
                    file_prefix="",  # no se_/de_ prefix for single-pass
                )
                unclassified_bams = [p for p in all_bam_files if "unclassified" in p.name]
                bam_files = [p for p in all_bam_files if "unclassified" not in p.name]

            # Annotate BM tag from bi per-end scores on each demuxed BAM
            for bam in bam_files:
                if "unclassified" not in bam.name:
                    annotate_demux_type_from_bi_tag(bam)

            se_bam_files = bam_files
            bam_dir = cfg.split_path
            double_barcoded_path = None

        else:
            # Old dorado (< 1.3.1) or dorado not found: use existing 2-pass approach
            if dorado_version is not None:
                logger.warning(
                    "Dorado version %s detected (< 1.3.1). Using 2-pass demux. "
                    "Upgrade to dorado >= 1.3.1 for faster single-pass demux with per-end scoring.",
                    ".".join(str(v) for v in dorado_version),
                )

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
    bam_qc_dir = load_directory / "bam_qc"
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
            load_directory,
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
            demux_backend=getattr(cfg, "demux_backend", None),
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
            load_directory,
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
            demux_backend=getattr(cfg, "demux_backend", None),
        )
        if cfg.delete_intermediate_tsvs:
            delete_tsvs(mod_tsv_dir)

    raw_adata.obs["Experiment_name"] = [cfg.experiment_name] * raw_adata.shape[0]
    raw_adata.obs["Experiment_name_and_barcode"] = (
        raw_adata.obs["Experiment_name"].astype(str) + "_" + raw_adata.obs["Barcode"].astype(str)
    )

    # Store experiment-specific BAM paths for POD5 plotting
    if "bam_paths" not in raw_adata.uns:
        raw_adata.uns["bam_paths"] = {}
    if unaligned_output.exists():
        raw_adata.uns["bam_paths"][f"{cfg.experiment_name}_unaligned"] = str(unaligned_output)
    if aligned_sorted_output.exists():
        raw_adata.uns["bam_paths"][f"{cfg.experiment_name}_aligned"] = str(aligned_sorted_output)

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

    # Build default tag list: always NM/MD, MM/ML only for direct modality
    default_tags = ["NM", "MD", "fn"]
    if cfg.smf_modality == "direct":
        default_tags.extend(["MM", "ML"])
    # Add UMI tags if UMI extraction was enabled
    if getattr(cfg, "use_umi", False):
        default_tags.extend(["U1", "U2", "US", "UE", "RX"])
    # Add barcode tags if smftools barcode extraction was used
    if demux_backend == "smftools" and cfg.barcode_kit:
        default_tags.extend(["BC", "BM", "B1", "B2", "B3", "B4", "B5", "B6"])
    # Add barcode tags from dorado single-pass demux (BM annotated from bi tag)
    elif demux_backend == "dorado" and cfg.barcode_kit and not cfg.input_already_demuxed:
        dorado_ver = _get_dorado_version()
        if dorado_ver is not None and dorado_ver >= (1, 3, 1):
            default_tags.extend(["BC", "BM", "bi"])
    bam_tag_names = getattr(cfg, "bam_tag_names", default_tags)

    logger.info("Adding BAM tags and BAM flags to adata.obs")
    add_read_tag_annotations(
        raw_adata,
        se_bam_files,
        tag_names=bam_tag_names,
        include_flags=True,
        include_cigar=True,
        extract_read_tags_from_bam_callable=extract_read_tags_from_bam,
        samtools_backend=cfg.samtools_backend,
    )

    # Expand dorado bi array tag into individual float score columns
    if "bi" in bam_tag_names:
        expand_bi_tag_columns(raw_adata, bi_column="bi")

    # Derive demux_type from BM tag when using smftools or dorado single-pass backend
    _derive_bm = False
    if demux_backend == "smftools" and cfg.barcode_kit and not cfg.input_already_demuxed:
        _derive_bm = True
    elif demux_backend == "dorado" and cfg.barcode_kit and not cfg.input_already_demuxed:
        dorado_ver = _get_dorado_version()
        if dorado_ver is not None and dorado_ver >= (1, 3, 1):
            _derive_bm = True
    if _derive_bm:
        logger.info("Deriving demux_type from BM tag")
        add_demux_type_from_bm_tag(raw_adata, bm_column="BM")

    if getattr(cfg, "annotate_secondary_supplementary", False):
        logger.info("Annotating secondary/supplementary alignments from aligned BAM")
        add_secondary_supplementary_alignment_flags(
            raw_adata,
            aligned_sorted_output,
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
            csv_path=load_directory / "read_to_pod5_origin_mapping.csv",
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
    mqc_dir = load_directory / "multiqc"
    if mqc_dir.is_dir():
        logger.info(f"{mqc_dir} already exists, skipping multiqc")
    else:
        logger.info("Running multiqc")
        run_multiqc(bam_qc_dir, mqc_dir)
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
