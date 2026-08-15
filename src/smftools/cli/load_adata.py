from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from smftools.constants import (
    BARCODE_KIT_ALIASES,
    LOAD_DIR,
    LOGGING_DIR,
    RAW_DIR,
    UMI_KIT_ALIASES,
)
from smftools.logging_utils import get_logger, setup_logging

from .helpers import AdataPaths

logger = get_logger(__name__)


def check_executable_exists(cmd: str) -> bool:
    """Return True if a command-line executable is available in PATH."""
    return shutil.which(cmd) is not None


def _validate_alignment_executables(cfg) -> None:
    """Require non-adapter Dorado use only when the request needs it."""
    if getattr(cfg, "alignment_mode", "align") == "existing":
        return
    if (
        cfg.input_type in {"fast5", "pod5"} or not cfg.input_already_demuxed
    ) and not check_executable_exists("dorado"):
        raise RuntimeError(
            "Error: 'dorado' is not installed or not in PATH. "
            "Install from https://github.com/nanoporetech/dorado"
        )


def _alignment_source_layout(resolved_input_manifest) -> str:
    """Return the adapter-facing layout after canonical BAM normalization."""
    paired_rows = [
        row
        for row in resolved_input_manifest.rows
        if bool(getattr(row, "pair_id", ""))
        or str(getattr(row, "mate", "unpaired")) in {"R1", "R2"}
    ]
    unpaired_rows = [
        row
        for row in resolved_input_manifest.rows
        if not bool(getattr(row, "pair_id", ""))
        and str(getattr(row, "mate", "unpaired")) not in {"R1", "R2"}
    ]
    if paired_rows and unpaired_rows:
        return "mixed_bam"
    return "paired_bam" if paired_rows else "single_bam"


def _probe_alignment_adapter(cfg):
    """Select and probe an adapter before any task output is created."""
    if getattr(cfg, "alignment_mode", "align") == "existing":
        return None
    from ..informatics.alignment_adapters import get_alignment_adapter

    adapter = get_alignment_adapter(cfg.aligner)
    environment = adapter.validate_environment(cfg.samtools_backend)
    return adapter, environment


def _resolve_alignment_adapter(cfg, resolved_input_manifest, probed_adapter):
    """Validate the resolved source layout before alignment staging or execution."""
    if probed_adapter is None:
        return None
    from ..informatics.alignment_adapters import AlignmentRequest

    adapter, environment = probed_adapter
    source_layout = _alignment_source_layout(resolved_input_manifest)
    adapter.validate_request(
        AlignmentRequest(
            reference_fasta=Path(cfg.fasta),
            input_bam=Path(cfg.input_data_path or "input.bam"),
            aligned_bam=Path("aligned.bam"),
            source_layout=source_layout,
            modality=str(cfg.smf_modality),
            aligner_args=tuple(cfg.aligner_args or ()),
            threads=cfg.threads,
            align_from_bam=bool(cfg.align_from_bam),
        )
    )
    return adapter, environment, source_layout


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


def load_dense_cache(config_path: str):
    """Ensure raw artifacts exist, then build the optional dense zarr cache."""
    from ..informatics.partition_store import write_dense_cache_from_spine
    from ..readwrite import safe_read_h5ad
    from .raw_adata import raw_adata

    _spine, spine_path, cfg = raw_adata(config_path)
    cache_paths = write_dense_cache_from_spine(
        spine_path, output_dir=Path(cfg.output_directory) / LOAD_DIR
    )
    spine, _ = safe_read_h5ad(cache_paths["spine"])
    return spine, cache_paths["spine"], cfg


def _publish_canonical_barcode_identity(
    *,
    output_directory: str | Path,
    aligned_bam: str | Path,
    resolved_input_manifest,
    route_sidecar: str | Path | None,
    classifier_source: str,
    sidecar_manifest: str | Path,
    force_redo: bool,
    sidecar_key_suffix: str = "",
) -> tuple[Path, Path]:
    """Publish or reuse the canonical barcode/sample identity intermediate."""
    from ..informatics.barcode_sidecar import (
        BARCODE_IDENTITY_SCHEMA_VERSION,
        publish_barcode_identity_sidecar,
    )
    from ..informatics.raw_intermediate_manifest import (
        IntermediateSpec,
        artifact_checksum,
        commit_intermediate,
        committed_output,
        prepare_intermediate,
    )
    from ..informatics.sidecar_manifest import register_sidecar

    aligned_bam = Path(aligned_bam)
    identity_inputs = [
        ("aligned-bam", artifact_checksum(aligned_bam)),
        ("input-manifest", resolved_input_manifest.digest),
    ]
    route_sidecar = Path(route_sidecar) if route_sidecar is not None else None
    if route_sidecar is not None and route_sidecar.is_file():
        identity_inputs.append(("route-barcode-sidecar", artifact_checksum(route_sidecar)))
    identity_spec = IntermediateSpec(
        operation="canonical-barcode-identity",
        input_artifacts=tuple(identity_inputs),
        operation_config={
            "classifier_source": classifier_source,
            "schema_version": BARCODE_IDENTITY_SCHEMA_VERSION,
        },
    )
    workspace = prepare_intermediate(
        output_directory,
        identity_spec,
        force_redo=force_redo,
    )
    if workspace.reusable:
        barcode_sidecar = committed_output(workspace, "barcode")
        report = committed_output(workspace, "report")
        if barcode_sidecar is None or report is None:
            raise RuntimeError("Validated barcode identity commit is missing an output.")
        logger.info("Reusing validated canonical barcode identity: %s", workspace.root)
    else:
        barcode_sidecar, report = publish_barcode_identity_sidecar(
            aligned_bam,
            workspace.root / "barcode.parquet",
            input_manifest=resolved_input_manifest,
            classifier_sidecar=route_sidecar,
            classifier_source=classifier_source,
        )
        commit_intermediate(workspace, {"barcode": barcode_sidecar, "report": report})
    key_suffix = f":{sidecar_key_suffix}" if sidecar_key_suffix else ""
    register_sidecar(
        sidecar_manifest,
        f"barcode{key_suffix}",
        barcode_sidecar,
        metadata={
            "schema_version": BARCODE_IDENTITY_SCHEMA_VERSION,
            "source": "canonical_identity",
        },
    )
    register_sidecar(
        sidecar_manifest,
        f"barcode_identity_report{key_suffix}",
        report,
        metadata={"schema_version": BARCODE_IDENTITY_SCHEMA_VERSION},
    )
    return Path(barcode_sidecar), Path(report)


def _barcode_classifier_source(cfg, *, demux_backend: str) -> str:
    """Return the provenance tier for route-specific barcode evidence."""
    if cfg.input_already_demuxed:
        return "filename"
    if not cfg.barcode_kit:
        return "filename"
    return "sequence:smftools" if demux_backend == "smftools" else "sequence:dorado"


def _attach_dense_barcode_identity(
    raw_adata, sidecar_path: str | Path, experiment_name: str
) -> None:
    """Attach canonical identity fields and namespace-aware grouping to dense raw data."""
    from ..informatics.barcode_sidecar import read_barcode_identity_sidecar

    identity = read_barcode_identity_sidecar(sidecar_path).set_index("read_name")
    identity = identity.reindex(raw_adata.obs_names)
    for column in [
        "barcode",
        "barcode_source",
        "barcode_confidence",
        "sample",
        "sample_source",
        "sample_confidence",
        "read_group",
        "namespace",
        "identity_status",
        "identity_conflicts",
        "identity_schema_version",
        "BC",
        "BM",
        "bi",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
    ]:
        if column in identity.columns:
            raw_adata.obs[column] = identity[column].values
    namespace = pd.Series(str(experiment_name), index=raw_adata.obs_names)
    if "namespace" in raw_adata.obs:
        declared = raw_adata.obs["namespace"].fillna("").astype(str)
        namespace = declared.where(declared != "", namespace)
    barcode_column = "barcode" if "barcode" in raw_adata.obs else "Barcode"
    raw_adata.obs["Experiment_name_and_barcode"] = (
        namespace.astype(str) + "_" + raw_adata.obs[barcode_column].astype(str)
    )


def _prepare_existing_alignment(
    *,
    output_directory: str | Path,
    source_bam: str | Path,
    reference_fasta: str | Path,
    reference_bundle,
    resolved_input_manifest,
    sidecar_manifest: str | Path,
    modality: str,
    threads: int | None,
    force_redo: bool,
    source_row=None,
    sidecar_key_suffix: str = "",
) -> tuple[Path, Path, Path]:
    """Validate, normalize, and own one existing alignment without realignment."""
    from ..informatics.alignment_manifest import (
        ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        read_alignment_manifest,
        write_alignment_manifest,
    )
    from ..informatics.alignment_validation import normalize_existing_alignment
    from ..informatics.bam_functions import _require_pysam
    from ..informatics.raw_intermediate_manifest import (
        IntermediateSpec,
        artifact_checksum,
        commit_intermediate,
        committed_output,
        prepare_intermediate,
    )
    from ..informatics.sidecar_manifest import register_sidecar

    pysam = _require_pysam()
    source_bam = Path(source_bam)
    source_rows = tuple(getattr(resolved_input_manifest, "rows", ()))
    selected_row = source_row or (source_rows[0] if len(source_rows) == 1 else None)
    source_checksum = (
        str(selected_row.sha256)
        if selected_row is not None and getattr(selected_row, "sha256", None)
        else artifact_checksum(source_bam)
    )
    reference_checksum = artifact_checksum(reference_fasta)
    spec = IntermediateSpec(
        operation="existing-alignment-normalization",
        input_artifacts=(
            ("input-manifest", resolved_input_manifest.digest),
            ("source-bam", source_checksum),
            ("alignment-reference-bundle", reference_bundle["digest"]),
            ("alignment-fasta", reference_checksum),
        ),
        operation_config={
            "alignment_manifest_schema": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
            "alignment_mode": "existing",
            "modality": str(modality),
            "normalization": "copy-or-coordinate-sort-and-index",
        },
        tool_versions={"pysam": str(pysam.__version__)},
    )
    workspace = prepare_intermediate(output_directory, spec, force_redo=force_redo)
    if workspace.reusable:
        normalized_bam = committed_output(workspace, "aligned-sorted-bam")
        normalized_bai = committed_output(workspace, "aligned-sorted-bai")
        manifest_path = committed_output(workspace, "alignment-manifest")
        if normalized_bam is None or normalized_bai is None or manifest_path is None:
            raise RuntimeError("Validated existing-alignment commit is missing an output.")
        read_alignment_manifest(manifest_path)
        logger.info("Reusing validated existing alignment: %s", workspace.root)
    else:
        normalized_bam = workspace.root / "aligned.sorted.bam"
        normalized_bam, normalized_bai, source_summary, normalized_summary = (
            normalize_existing_alignment(
                source_bam,
                normalized_bam,
                reference_fasta,
                modality=modality,
                threads=threads,
            )
        )
        manifest_path = write_alignment_manifest(
            workspace.root / "alignment_manifest.json",
            input_manifest_digest=resolved_input_manifest.digest,
            reference_bundle=reference_bundle,
            prepared_reference_sha256=reference_checksum,
            source_bam=source_bam,
            source_sha256=source_checksum,
            normalized_bam=normalized_bam,
            normalized_bai=normalized_bai,
            validation={
                "source": source_summary.to_dict(),
                "normalized": normalized_summary.to_dict(),
                "normalization_applied": (
                    not source_summary.coordinate_sorted or source_bam.suffix.lower() == ".cram"
                ),
            },
        )
        commit_intermediate(
            workspace,
            {
                "aligned-sorted-bam": normalized_bam,
                "aligned-sorted-bai": normalized_bai,
                "alignment-manifest": manifest_path,
            },
        )
    register_sidecar(
        sidecar_manifest,
        f"alignment_manifest:{sidecar_key_suffix}" if sidecar_key_suffix else "alignment_manifest",
        manifest_path,
        metadata={
            "alignment_mode": "existing",
            "schema_version": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        },
    )
    return Path(normalized_bam), Path(normalized_bai), Path(manifest_path)


def load_adata_core(
    cfg,
    paths: AdataPaths,
    config_path: str | None = None,
    *,
    raw_only: bool = False,
):
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
        annotate_umi_tags_in_bam,
        bam_qc,
        build_barcode_sidecar_from_split_bams,
        concatenate_fastqs_to_bam,
        demux_and_index_BAM,
        derive_bm_from_bi_to_sidecar,
        derive_umi_orientation_tags_from_aligned_bam,
        extract_and_assign_barcodes_in_bam,
        extract_read_features_from_bam,
        extract_read_tags_from_bam,
        load_barcode_references_from_yaml,
        load_umi_config_from_yaml,
        rebuild_barcode_sidecar_via_dorado_classification,
        resolve_barcode_config,
        resolve_umi_config,
        split_and_index_BAM,
        subsample_split_bams,
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
    from ..informatics.input_manifest import (
        materialize_input_view,
        resolve_input_manifest,
        subset_input_manifest,
    )
    from ..informatics.modkit_extract_to_adata import modkit_extract_to_adata
    from ..informatics.modkit_functions import extract_mods, make_modbed, modQC
    from ..informatics.partition_read import relative_uns_path
    from ..informatics.raw_intermediate_manifest import (
        IntermediateSpec,
        alignment_reference_bundle,
        artifact_checksum,
        commit_intermediate,
        committed_output,
        executable_version,
        prepare_intermediate,
    )
    from ..informatics.region_catalog import (
        REFERENCE_INTERVAL_MAP_SCHEMA_VERSION,
        REGION_CATALOG_SCHEMA_VERSION,
        write_normalized_alignment_bed,
        write_reference_interval_map,
        write_region_catalogs,
    )
    from ..informatics.run_multiqc import run_multiqc
    from ..informatics.sidecar_manifest import (
        register_sidecar,
        resolve_sidecar,
        sidecar_manifest_path,
    )
    from ..memory_guard import require_memory_headroom
    from ..metadata import record_smftools_metadata
    from ..readwrite import add_or_update_column_in_csv, make_dirs
    from .helpers import write_gz_h5ad

    ################################### 1) General params and input organization ###################################
    date_str = datetime.today().strftime("%y%m%d")
    now = datetime.now()
    time_str = now.strftime("%H%M%S")

    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    output_directory = Path(cfg.output_directory)
    load_directory = output_directory / (RAW_DIR if raw_only else LOAD_DIR)
    bam_outputs_directory = Path(cfg.bam_outputs_path)
    fasta_outputs_directory = Path(cfg.fasta_outputs_path)
    bed_outputs_directory = Path(cfg.bed_outputs_path)
    modkit_outputs_directory = Path(cfg.modkit_outputs_path)
    sidecar_manifest = sidecar_manifest_path(load_directory)
    logging_directory = load_directory / LOGGING_DIR

    # Fail tool/version checks before creating any task output. Source-layout
    # validation follows canonical manifest resolution but still precedes the
    # alignment workspace.
    _validate_alignment_executables(cfg)
    probed_alignment_adapter = _probe_alignment_adapter(cfg)

    make_dirs(
        [
            output_directory,
            load_directory,
            bam_outputs_directory,
            fasta_outputs_directory,
        ]
    )

    if cfg.emit_log_file and not raw_only:
        log_file = logging_directory / f"{date_str}_{time_str}_log.log"
        make_dirs([logging_directory])
    else:
        log_file = None

    # ``raw_adata`` owns raw's lifecycle-scoped human/performance log. Do not
    # rotate to a second legacy load log after that wrapper has configured it.
    setup_logging(
        level=log_level,
        log_file=log_file,
        reconfigure=log_file is not None and not raw_only,
    )

    requested_input_data_path = cfg.input_data_path
    resolved_input_manifest = resolve_input_manifest(
        output_directory=output_directory,
        input_manifest_path=cfg.input_manifest_path,
        input_paths=None if cfg.input_manifest_path else cfg.input_files,
        alignment_mode=cfg.alignment_mode,
        modality=cfg.smf_modality,
        barcode_map=cfg.fastq_barcode_map,
        auto_pair=cfg.fastq_auto_pairing,
    )
    full_input_manifest = resolved_input_manifest
    cfg.input_manifest_digest = full_input_manifest.digest
    append_source_ids = tuple(getattr(cfg, "_raw_append_source_ids", ()))
    if append_source_ids:
        resolved_input_manifest = subset_input_manifest(
            full_input_manifest,
            append_source_ids,
        )
        logger.info(
            "Append-only raw execution selected %d new source(s) from %d total source(s)",
            len(resolved_input_manifest.rows),
            len(full_input_manifest.rows),
        )
    cfg.input_type = resolved_input_manifest.input_type
    cfg.input_files = [Path(row.path) for row in resolved_input_manifest.rows]
    cfg._resolved_input_manifest = resolved_input_manifest
    source_artifacts = (
        ("input-manifest", resolved_input_manifest.digest),
        *((f"source:{row.source_id}", row.sha256) for row in resolved_input_manifest.rows),
    )
    force_redo_intermediates = bool(getattr(cfg, "force_redo_load_adata", False))
    if cfg.input_manifest_path:
        if len(cfg.input_files) == 1:
            cfg.input_data_path = cfg.input_files[0]
        elif cfg.input_type in {"pod5", "fast5"}:
            cfg.input_data_path = materialize_input_view(resolved_input_manifest, output_directory)

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
        mod_bed_dir = modkit_outputs_directory / "mod_beds"
        mod_tsv_dir = modkit_outputs_directory / "mod_tsvs"
        mods = [cfg.mod_map[mod] for mod in cfg.mod_list]

        if cfg.direct_signal_backend == "modkit" and not check_executable_exists("modkit"):
            raise RuntimeError(
                "Error: 'modkit' is not installed or not in PATH. "
                "Install from https://github.com/nanoporetech/modkit"
            )
    else:
        mod_bed_dir = None
        mod_tsv_dir = None
        mods = None

    alignment_adapter_context = _resolve_alignment_adapter(
        cfg, resolved_input_manifest, probed_alignment_adapter
    )

    # # Detect the input filetypes
    # If the input files are fast5 files, convert the files to a pod5 file before proceeding.
    if cfg.input_type == "fast5":
        from ..informatics.pod5_functions import fast5_to_pod5

        conversion_spec = IntermediateSpec(
            operation="fast5-to-pod5",
            input_artifacts=source_artifacts,
            operation_config={"output_format": "pod5"},
            tool_versions={"pod5": executable_version("pod5")},
        )
        conversion_workspace = prepare_intermediate(
            output_directory,
            conversion_spec,
            force_redo=force_redo_intermediates,
        )
        if conversion_workspace.reusable:
            output_pod5 = committed_output(conversion_workspace, "pod5")
            if output_pod5 is None:
                raise RuntimeError("Validated FAST5 conversion commit has no POD5 output.")
            logger.info("Reusing validated FAST5-to-POD5 intermediate: %s", output_pod5)
        else:
            output_pod5 = conversion_workspace.root / "converted.pod5"
            logger.info(
                f"Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_pod5}"
            )
            fast5_to_pod5(cfg.input_data_path, output_pod5)
            commit_intermediate(conversion_workspace, {"pod5": output_pod5})
        # Reassign the pod5_dir variable to point to the new pod5 file.
        cfg.input_data_path = output_pod5
        cfg.input_type = "pod5"
    # If the input is a fastq or a directory of fastqs, concatenate them into an unaligned BAM and save the barcode
    elif cfg.input_type == "fastq":
        normalization_spec = IntermediateSpec(
            operation="fastq-to-unaligned-bam",
            input_artifacts=source_artifacts,
            operation_config={
                "add_read_group": True,
                "auto_pair": False,
                "barcode_map": {
                    **(cfg.fastq_barcode_map or {}),
                    **resolved_input_manifest.fastq_barcode_map(),
                },
                "barcode_tag": "BC",
                "identity_metadata_schema": 1,
                "samtools_backend": cfg.samtools_backend,
            },
            tool_versions={"samtools": executable_version("samtools")},
        )
        normalization_workspace = prepare_intermediate(
            output_directory,
            normalization_spec,
            force_redo=force_redo_intermediates,
        )
        if normalization_workspace.reusable:
            output_bam = committed_output(normalization_workspace, "bam")
            if output_bam is None:
                raise RuntimeError("Validated FASTQ normalization commit has no BAM output.")
            logger.info("Reusing validated FASTQ-to-BAM intermediate: %s", output_bam)
        else:
            output_bam = normalization_workspace.root / "canonical_basecalls.bam"
            logger.info("Concatenating FASTQ files into a single BAM file")
            summary = concatenate_fastqs_to_bam(
                resolved_input_manifest.fastq_inputs(),
                output_bam,
                barcode_tag="BC",
                gzip_suffixes=(".gz", ".gzip"),
                barcode_map={
                    **(cfg.fastq_barcode_map or {}),
                    **resolved_input_manifest.fastq_barcode_map(),
                },
                read_group_map=resolved_input_manifest.fastq_read_group_map(),
                sample_map=resolved_input_manifest.fastq_sample_map(),
                add_read_group=True,
                rg_sample_field=None,
                progress=False,
                auto_pair=False,
                samtools_backend=cfg.samtools_backend,
            )

            logger.info(f"Found the following barcodes in FASTQ inputs: {summary['barcodes']}")
            commit_intermediate(normalization_workspace, {"bam": output_bam})

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

    if not basecall:
        # Preserve the exact BAM input path. Path.with_suffix() would incorrectly
        # turn names such as ``sample.repaired.bam`` into ``sample.bam`` after the
        # terminal .bam suffix has already been removed.
        unaligned_output = Path(
            cfg.input_files[0]
            if cfg.alignment_mode == "existing" and len(cfg.input_files) > 1
            else cfg.input_data_path
        )
        bam = unaligned_output.with_suffix("")

    ########################################################################################################################

    ################################### 2) FASTA Handling ###################################

    try:
        cfg.fasta = Path(cfg.fasta)
    except Exception:
        logger.warning("Need to provide an input FASTA path to proceed with smftools load")

    original_fasta = Path(cfg.fasta)
    reference_bundle = alignment_reference_bundle(cfg)
    run_root = load_directory.parent
    region_catalog_paths = write_region_catalogs(
        cfg,
        original_fasta=original_fasta,
        run_root=run_root,
    )
    alignment_catalog = None
    if "alignment" in region_catalog_paths:
        alignment_catalog = pd.read_parquet(region_catalog_paths["alignment"])

    # Alignment BEDs always use original FASTA coordinates. The deprecated
    # fasta_regions_of_interest alias has already been resolved into this field
    # by ExperimentConfig.
    if cfg.alignment_regions_bed:
        fasta_stem = cfg.fasta.stem
        bed_stem = Path(cfg.alignment_regions_bed).stem
        source_sha = str(alignment_catalog["source_sha256"].iloc[0])
        output_FASTA = fasta_outputs_directory / (
            f"{fasta_stem}_subsampled_by_{bed_stem}_{source_sha[:12]}_"
            f"{reference_bundle['digest'][:12]}.fasta"
        )
        normalized_bed = write_normalized_alignment_bed(
            alignment_catalog,
            fasta_outputs_directory / f"{output_FASTA.stem}.bed",
        )

        logger.info("Subsampling FASTA records using the provided BED file")
        if not output_FASTA.exists():
            subsample_fasta_from_bed(cfg.fasta, normalized_bed, load_directory, output_FASTA)
        fasta = output_FASTA
    else:
        logger.info("Using the full FASTA file")
        fasta = cfg.fasta

    # For conversion style SMF, make a converted reference FASTA
    if cfg.smf_modality == "conversion":
        fasta_stem = fasta.stem
        converted_FASTA_basename = f"{fasta_stem}_converted_{reference_bundle['digest'][:12]}.fasta"
        converted_FASTA = fasta_outputs_directory / converted_FASTA_basename

        if "converted.fa" in fasta.name:
            logger.info(f"{fasta} is already converted. Using existing converted FASTA.")
            converted_FASTA = fasta
        elif converted_FASTA.exists():
            logger.info(f"{converted_FASTA} already exists. Using existing converted FASTA.")
        else:
            logger.info(f"Converting FASTA base sequences")
            generate_converted_FASTA(fasta, cfg.conversion_types, cfg.strands, converted_FASTA)
        fasta = converted_FASTA

    reference_interval_map = write_reference_interval_map(
        run_root=run_root,
        original_fasta=original_fasta,
        alignment_fasta=fasta,
        modality=str(cfg.smf_modality),
        conversions=cfg.conversion_types,
        strands=cfg.strands,
        alignment_catalog=alignment_catalog,
    )
    register_sidecar(
        sidecar_manifest,
        "reference_interval_map",
        reference_interval_map,
        metadata={"schema_version": REFERENCE_INTERVAL_MAP_SCHEMA_VERSION},
    )
    for scope, catalog_path in region_catalog_paths.items():
        register_sidecar(
            sidecar_manifest,
            f"{scope}_regions",
            catalog_path,
            metadata={
                "schema_version": REGION_CATALOG_SCHEMA_VERSION,
                "coordinate_system": "0-based-half-open-original-fasta",
            },
        )

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
        if getattr(cfg, "max_basecall_reads", None) is not None:
            from ..informatics.pod5_functions import subsample_pod5_for_basecalling

            cfg.input_data_path = subsample_pod5_for_basecalling(
                cfg.input_data_path,
                cfg.max_basecall_reads,
                load_directory,
            )
        basecall_spec = IntermediateSpec(
            operation="dorado-basecalling",
            input_artifacts=(("pod5-input", artifact_checksum(cfg.input_data_path)),),
            operation_config={
                "barcode_both_ends": bool(cfg.barcode_both_ends),
                "barcode_kit": cfg.barcode_kit if cfg.barcode_kit != "custom" else None,
                "device": str(cfg.device),
                "emit_moves": bool(cfg.emit_moves),
                "model": str(cfg.model),
                "modifications": list(cfg.mod_list or []) if cfg.smf_modality == "direct" else [],
                "modality": str(cfg.smf_modality),
                "trim": bool(cfg.trim),
            },
            tool_versions={"dorado": executable_version("dorado")},
        )
        basecall_workspace = prepare_intermediate(
            output_directory,
            basecall_spec,
            force_redo=force_redo_intermediates,
        )
        if basecall_workspace.reusable:
            unaligned_output = committed_output(basecall_workspace, "bam")
            if unaligned_output is None:
                raise RuntimeError("Validated basecalling commit has no BAM output.")
            logger.info("Reusing validated dorado basecalling intermediate: %s", unaligned_output)
        else:
            bam = basecall_workspace.root / "basecalls"
            unaligned_output = bam.with_suffix(cfg.bam_suffix)
        if basecall_workspace.reusable:
            pass
        elif cfg.smf_modality != "direct":
            require_memory_headroom(
                cfg,
                operation_label="dorado canonical basecalling",
                estimator="external_basecalling_peak",
            )
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
            commit_intermediate(basecall_workspace, {"bam": unaligned_output})
        else:
            require_memory_headroom(
                cfg,
                operation_label="dorado modified basecalling",
                estimator="external_basecalling_peak",
            )
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
            commit_intermediate(basecall_workspace, {"bam": unaligned_output})
    elif basecall:
        logger.error("Basecalling is currently only supported for ont sequencers and not pacbio.")
    else:
        pass
    ########################################################################################################################

    ################################### 4) Alignment and sorting #############################################

    # Existing mode owns a validated copy (or coordinate-sorted normalization)
    # without invoking an aligner or changing alignment placement.
    alignment_workspace = None
    alignment_manifest_path = None
    alignment_partitions = None
    if cfg.alignment_mode == "existing":
        from ..informatics.alignment_validation import validate_alignment_partitions

        alignment_rows = resolved_input_manifest.alignment_inputs()
        if len(alignment_rows) > 1:
            unsupported = []
            if not getattr(cfg, "skip_bam_split", False):
                unsupported.append("skip_bam_split=False")
            if not cfg.input_already_demuxed:
                unsupported.append("input_already_demuxed=False")
            if getattr(cfg, "use_umi", False):
                unsupported.append("use_umi=True")
            if getattr(cfg, "make_beds", False):
                unsupported.append("make_beds=True")
            if (
                str(cfg.smf_modality) == "direct"
                and str(getattr(cfg, "direct_signal_backend", "modkit")) == "modkit"
            ):
                unsupported.append("direct_signal_backend='modkit'")
            if unsupported:
                raise ValueError(
                    "Partitioned existing-alignment ingestion does not support this processing "
                    "route yet: " + ", ".join(unsupported)
                )
        validate_alignment_partitions(alignment_rows, fasta, modality=cfg.smf_modality)
        prepared_partitions = []
        for source_row in alignment_rows:
            prepared_partitions.append(
                (
                    *_prepare_existing_alignment(
                        output_directory=output_directory,
                        source_bam=source_row.path,
                        reference_fasta=fasta,
                        reference_bundle=reference_bundle,
                        resolved_input_manifest=resolved_input_manifest,
                        sidecar_manifest=sidecar_manifest,
                        modality=cfg.smf_modality,
                        threads=cfg.threads,
                        force_redo=force_redo_intermediates,
                        source_row=source_row,
                        sidecar_key_suffix=(
                            source_row.source_id if len(alignment_rows) > 1 else ""
                        ),
                    ),
                    source_row,
                )
            )
        aligned_sorted_output, alignment_bai, alignment_manifest_path, _first_row = (
            prepared_partitions[0]
        )
        alignment_partitions = [
            (bam_path, row) for bam_path, _bai, _manifest, row in prepared_partitions
        ]
        aligned_sorted_BAM = aligned_sorted_output.with_suffix("")
        unaligned_output = aligned_sorted_output
    else:
        from ..informatics.alignment_adapters import AlignmentRequest
        from ..informatics.alignment_manifest import (
            ALIGNMENT_MANIFEST_SCHEMA_VERSION,
            read_alignment_manifest,
            write_alignment_manifest,
        )
        from ..informatics.alignment_validation import validate_existing_alignment

        assert alignment_adapter_context is not None
        adapter, adapter_environment, source_layout = alignment_adapter_context
        alignment_reference_sha256 = artifact_checksum(fasta)
        reference_index_plan = adapter.reference_plan(
            alignment_reference_sha256, adapter_environment
        )
        alignment_spec = IntermediateSpec(
            operation="alignment-sort-index",
            input_artifacts=(
                ("unaligned-bam", artifact_checksum(unaligned_output)),
                ("alignment-reference-bundle", reference_bundle["digest"]),
                ("alignment-fasta", alignment_reference_sha256),
            ),
            operation_config={
                "alignment_adapter_schema": 1,
                "alignment_manifest_schema": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
                "align_from_bam": bool(getattr(cfg, "align_from_bam", False)),
                "adapter": adapter.name,
                "aligner_args": list(getattr(cfg, "aligner_args", None) or []),
                "bam_suffix": str(cfg.bam_suffix),
                "reference_index_identity": reference_index_plan["identity"],
                "rescue_min_margin_bp": int(getattr(cfg, "rescue_min_margin_bp", 0)),
                "rescue_min_margin_fraction": float(
                    getattr(cfg, "rescue_min_margin_fraction", 0.0)
                ),
                "rescue_secondary_alignments": bool(
                    getattr(cfg, "rescue_secondary_alignments", False)
                ),
                "samtools_backend": str(cfg.samtools_backend),
                "source_layout": source_layout,
            },
            tool_versions={
                adapter.name: adapter_environment.adapter_version,
                f"sort-index:{adapter_environment.samtools_backend}": (
                    adapter_environment.sort_index_version
                ),
            },
        )
        alignment_workspace = prepare_intermediate(
            output_directory,
            alignment_spec,
            force_redo=force_redo_intermediates,
        )
        aligned_BAM = alignment_workspace.root / "aligned"
        aligned_output = aligned_BAM.with_suffix(cfg.bam_suffix)
        aligned_sorted_BAM = aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
        aligned_sorted_output = aligned_sorted_BAM.with_suffix(cfg.bam_suffix)
        alignment_bai = Path(str(aligned_sorted_output) + ".bai")
        if alignment_workspace.reusable:
            committed_bam = committed_output(alignment_workspace, "aligned-sorted-bam")
            committed_bai = committed_output(alignment_workspace, "aligned-sorted-bai")
            alignment_manifest_path = committed_output(alignment_workspace, "alignment-manifest")
            if committed_bam is None or committed_bai is None or alignment_manifest_path is None:
                raise RuntimeError(
                    "Validated alignment commit is missing BAM, BAI, or manifest output."
                )
            read_alignment_manifest(alignment_manifest_path)
            aligned_sorted_output = committed_bam
            aligned_sorted_BAM = aligned_sorted_output.with_suffix("")
            alignment_bai = committed_bai
            logger.info(
                "Reusing validated alignment/sort/index intermediate: %s", aligned_sorted_output
            )
        else:
            require_memory_headroom(
                cfg,
                operation_label=f"{cfg.aligner} alignment and sorting",
                estimator="external_alignment_peak",
            )
            logger.info("Aligning and sorting reads")
            adapter_result = adapter.execute(
                AlignmentRequest(
                    reference_fasta=Path(fasta),
                    input_bam=Path(unaligned_output),
                    aligned_bam=aligned_output,
                    source_layout=source_layout,
                    modality=str(cfg.smf_modality),
                    aligner_args=tuple(cfg.aligner_args or ()),
                    threads=cfg.threads,
                    align_from_bam=bool(cfg.align_from_bam),
                ),
                adapter_environment,
                alignment_reference_sha256,
            )
            aligned_sorted_output = adapter_result.aligned_sorted_bam
            aligned_sorted_BAM = aligned_sorted_output.with_suffix("")
            alignment_bai = adapter_result.aligned_sorted_bai
            alignment_adapter_provenance = adapter_result.provenance

        alignment_partitions = [(aligned_sorted_output, resolved_input_manifest.rows[0])]

    alignment_was_generated = alignment_workspace is not None and not alignment_workspace.reusable

    # Optional: rescue reads whose primary alignment lost to a worse-covering
    # secondary alignment (e.g. minimap2 preferring a truncated match against
    # a wild-type contig over a full-length match against a shorter deletion-
    # allele contig). Runs before anything downstream reads aligned_sorted_
    # output, so a corrected Reference_strand is the only thing raw ingestion
    # ever sees. See src/smftools/informatics/alignment_rescue.py.
    if getattr(cfg, "rescue_secondary_alignments", False) and alignment_was_generated:
        rescue_summary_path = aligned_sorted_BAM.with_name(
            aligned_sorted_BAM.stem + "_rescue_summary.csv"
        )
        if rescue_summary_path.exists():
            logger.debug(
                f"{rescue_summary_path} already exists. Skipping secondary-alignment rescue."
            )
        else:
            logger.info("Rescuing reads misassigned by minimap2's primary-alignment pick")
            from ..informatics.alignment_rescue import (
                build_record_chromosome_map,
                rescue_secondary_alignments,
            )

            record_chromosome = build_record_chromosome_map(
                fasta, cfg.smf_modality, cfg.conversion_types
            )
            rescued_tmp = aligned_sorted_BAM.with_name(
                aligned_sorted_BAM.stem + "_rescue_tmp"
            ).with_suffix(cfg.bam_suffix)
            summary = rescue_secondary_alignments(
                aligned_sorted_output,
                rescued_tmp,
                record_chromosome,
                min_margin_bp=cfg.rescue_min_margin_bp,
                min_margin_fraction=cfg.rescue_min_margin_fraction,
                threads=cfg.threads,
            )
            # Swap the corrected BAM (+ its freshly-built index) into place so
            # every downstream consumer sees it at the original path with no
            # further plumbing.
            rescued_tmp_bai = Path(str(rescued_tmp) + ".bai")
            final_bai = Path(str(aligned_sorted_output) + ".bai")
            rescued_tmp.replace(aligned_sorted_output)
            if rescued_tmp_bai.exists():
                rescued_tmp_bai.replace(final_bai)
            summary.to_dataframe().to_csv(rescue_summary_path, index=False)

    if alignment_was_generated:
        normalized_summary = validate_existing_alignment(
            aligned_sorted_output,
            fasta,
            modality=cfg.smf_modality,
        )
        alignment_manifest_path = write_alignment_manifest(
            alignment_workspace.root / "alignment_manifest.json",
            input_manifest_digest=resolved_input_manifest.digest,
            reference_bundle=reference_bundle,
            prepared_reference_sha256=artifact_checksum(fasta),
            source_bam=unaligned_output,
            source_sha256=artifact_checksum(unaligned_output),
            normalized_bam=aligned_sorted_output,
            normalized_bai=alignment_bai,
            validation={"normalized": normalized_summary.to_dict()},
            alignment_mode="align",
            adapter=alignment_adapter_provenance,
        )
        alignment_outputs = {
            "aligned-sorted-bam": aligned_sorted_output,
            "aligned-sorted-bai": alignment_bai,
            "alignment-manifest": alignment_manifest_path,
        }
        if getattr(cfg, "rescue_secondary_alignments", False):
            alignment_outputs["rescue-summary"] = rescue_summary_path
        assert alignment_workspace is not None
        commit_intermediate(alignment_workspace, alignment_outputs)

    if alignment_was_generated and alignment_manifest_path is not None:
        register_sidecar(
            sidecar_manifest,
            "alignment_manifest",
            alignment_manifest_path,
            metadata={
                "alignment_mode": cfg.alignment_mode,
                "schema_version": 1,
            },
        )

    if cfg.make_beds:
        # Make beds and provide basic histograms
        aligned_bed_output_root = bed_outputs_directory / "aligned"
        bed_dir = aligned_bed_output_root / "beds"
        if bed_dir.is_dir():
            logger.debug(
                f"{bed_dir} already exists. Skipping BAM -> BED conversion for {aligned_sorted_output}"
            )
        else:
            require_memory_headroom(
                cfg,
                operation_label="BAM to BED conversion",
                estimator="external_bam_to_bed_peak",
            )
            logger.info("Making bed files from the aligned and sorted BAM file")
            aligned_BAM_to_bed(
                aligned_sorted_output,
                aligned_bed_output_root,
                fasta,
                cfg.make_bigwigs,
                cfg.threads,
                samtools_backend=cfg.samtools_backend,
                bedtools_backend=cfg.bedtools_backend,
                bigwig_backend=cfg.bigwig_backend,
            )
    ########################################################################################################################

    ################################### 4.5) Optional UMI annotation #############################################
    umi_sidecar = None
    barcode_sidecar = None
    if getattr(cfg, "use_umi", False):
        logger.info("Extracting positional UMIs (US/UE) from unaligned BAM before demultiplexing")

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
        umi_inputs = [
            ("unaligned-bam", artifact_checksum(unaligned_output)),
            ("aligned-bam", artifact_checksum(aligned_sorted_output)),
        ]
        if umi_yaml_path:
            umi_inputs.append(("umi-config", artifact_checksum(umi_yaml_path)))
        umi_spec = IntermediateSpec(
            operation="umi-sidecars",
            input_artifacts=tuple(umi_inputs),
            operation_config={
                "adapter_matcher": getattr(cfg, "umi_adapter_matcher", "edlib"),
                "amplicon_max_edits": resolved_umi["umi_amplicon_max_edits"],
                "ends": resolved_umi["umi_ends"],
                "flank_mode": resolved_umi["umi_flank_mode"],
                "length": getattr(cfg, "umi_length", None),
                "max_edits": resolved_umi["umi_adapter_max_edits"],
                "same_orientation": resolved_umi.get("same_orientation", False),
                "search_window": getattr(cfg, "umi_search_window", 200),
            },
            tool_versions={"samtools": executable_version("samtools")},
        )
        umi_workspace = prepare_intermediate(
            output_directory,
            umi_spec,
            force_redo=force_redo_intermediates,
        )
        if umi_workspace.reusable:
            umi_positional_sidecar = committed_output(umi_workspace, "umi-positional")
            umi_sidecar = committed_output(umi_workspace, "umi-oriented")
            if umi_positional_sidecar is None or umi_sidecar is None:
                raise RuntimeError("Validated UMI commit is missing a sidecar output.")
            logger.info("Reusing validated UMI sidecars: %s", umi_workspace.root)
        else:
            generated_positional = annotate_umi_tags_in_bam(
                unaligned_output,
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
            umi_positional_sidecar = umi_workspace.root / "positional.parquet"
            shutil.copy2(generated_positional, umi_positional_sidecar)
            logger.info("Deriving orientation-aware UMI tags (U1/U2/RX/FC) from aligned BAM")
            umi_sidecar = derive_umi_orientation_tags_from_aligned_bam(
                umi_positional_sidecar,
                aligned_sorted_output,
                output_sidecar_path=umi_workspace.root / "oriented.parquet",
                samtools_backend=cfg.samtools_backend,
            )
            commit_intermediate(
                umi_workspace,
                {"umi-oriented": umi_sidecar, "umi-positional": umi_positional_sidecar},
            )
        register_sidecar(
            sidecar_manifest,
            "umi_positional",
            umi_positional_sidecar,
            metadata={"source_bam": str(unaligned_output)},
        )
        register_sidecar(
            sidecar_manifest,
            "umi_oriented",
            umi_sidecar,
            metadata={"source_bam": str(aligned_sorted_output)},
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
        barcode_spec = IntermediateSpec(
            operation="smftools-barcode-sidecars",
            input_artifacts=(
                ("unaligned-bam", artifact_checksum(unaligned_output)),
                ("barcode-config", artifact_checksum(barcode_yaml_path)),
            ),
            operation_config={
                "adapter_matcher": getattr(cfg, "barcode_adapter_matcher", "edlib"),
                "amplicon_gap_tolerance": resolved_bc["barcode_amplicon_gap_tolerance"],
                "barcode_ends": resolved_bc["barcode_ends"],
                "composite_max_edits": resolved_bc["barcode_composite_max_edits"],
                "max_edit_distance": resolved_bc["barcode_max_edit_distance"],
                "min_score": getattr(cfg, "barcode_min_score", None),
                "min_separation": resolved_bc.get("barcode_min_separation"),
                "require_both_ends": bool(getattr(cfg, "barcode_both_ends", False)),
                "search_window": getattr(cfg, "barcode_search_window", 200),
            },
            tool_versions={"samtools": executable_version("samtools")},
        )
        barcode_workspace = prepare_intermediate(
            output_directory,
            barcode_spec,
            force_redo=force_redo_intermediates,
        )
        if barcode_workspace.reusable:
            barcode_positional_sidecar = committed_output(barcode_workspace, "barcode-positional")
            barcode_sidecar = committed_output(barcode_workspace, "barcode")
            if barcode_positional_sidecar is None or barcode_sidecar is None:
                raise RuntimeError("Validated barcode commit is missing a sidecar output.")
            logger.info("Reusing validated smftools barcode sidecars: %s", barcode_workspace.root)
        else:
            logger.info(
                "Extracting and assigning barcodes from unaligned BAM using smftools backend"
            )
            generated_barcode_sidecar = extract_and_assign_barcodes_in_bam(
                unaligned_output,
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
                threads=cfg.threads,
            )
            barcode_positional_sidecar = barcode_workspace.root / "positional.parquet"
            shutil.copy2(generated_barcode_sidecar, barcode_positional_sidecar)
            barcode_sidecar = barcode_workspace.root / "aligned.parquet"
            shutil.copy2(generated_barcode_sidecar, barcode_sidecar)
            commit_intermediate(
                barcode_workspace,
                {
                    "barcode": barcode_sidecar,
                    "barcode-positional": barcode_positional_sidecar,
                },
            )
        register_sidecar(
            sidecar_manifest,
            "barcode_positional",
            barcode_positional_sidecar,
            metadata={"source_bam": str(unaligned_output)},
        )
        register_sidecar(
            sidecar_manifest,
            "barcode",
            barcode_sidecar,
            metadata={"source_bam": str(aligned_sorted_output)},
        )
        logger.info(f"smftools barcode extraction complete: {barcode_sidecar}")
    ########################################################################################################################

    ################################### 5) Demultiplexing ######################################################################

    skip_bam_split = getattr(cfg, "skip_bam_split", False)
    dorado_barcode_workspace = None
    if demux_backend == "dorado" and cfg.barcode_kit and not cfg.input_already_demuxed:
        dorado_barcode_spec = IntermediateSpec(
            operation="dorado-barcode-sidecar",
            input_artifacts=(("aligned-bam", artifact_checksum(aligned_sorted_output)),),
            operation_config={
                "barcode_both_ends": bool(getattr(cfg, "barcode_both_ends", False)),
                "barcode_kit": str(cfg.barcode_kit),
                "bm_score_threshold": float(getattr(cfg, "dorado_bm_score_threshold", 0.65)),
                "skip_bam_split": bool(skip_bam_split),
                "trim": bool(cfg.trim),
            },
            tool_versions={"dorado": executable_version("dorado")},
        )
        dorado_barcode_workspace = prepare_intermediate(
            output_directory,
            dorado_barcode_spec,
            force_redo=force_redo_intermediates,
        )
        if dorado_barcode_workspace.reusable:
            barcode_sidecar = committed_output(dorado_barcode_workspace, "barcode")
            if barcode_sidecar is None:
                raise RuntimeError("Validated Dorado barcode commit has no sidecar output.")
            logger.info("Reusing validated Dorado barcode sidecar: %s", barcode_sidecar)

    # 3) Split the aligned and sorted BAM files by barcode (BC Tag) into the split_BAM directory
    if skip_bam_split:
        logger.info("skip_bam_split=True: skipping BAM splitting, using aligned source partitions")
        assert alignment_partitions is not None
        se_bam_files = [bam_path for bam_path, _row in alignment_partitions]
        bam_files = list(se_bam_files)
        unclassified_bams = []
        bam_dir = None
        double_barcoded_path = None
        # For dorado backend in non-split mode:
        # - if BC+bi tags are present on the aligned BAM, derive BM and write sidecar
        # - if bi is absent, keep BC-only sidecar behavior and warn that BM/demux_type is unavailable
        if (
            demux_backend == "dorado"
            and cfg.barcode_kit
            and not cfg.input_already_demuxed
            and not dorado_barcode_workspace.reusable
        ):
            tag_info = _bam_has_barcode_info_tags(aligned_sorted_output)
            if barcode_sidecar is None:
                barcode_sidecar = aligned_sorted_output.with_suffix(".barcode_tags.parquet")
            if tag_info.get("has_bc") and tag_info.get("has_bi"):
                derive_bm_from_bi_to_sidecar(
                    aligned_sorted_output,
                    barcode_sidecar,
                    threshold=float(getattr(cfg, "dorado_bm_score_threshold", 0.65)),
                    samtools_backend=cfg.samtools_backend,
                )
                register_sidecar(
                    sidecar_manifest,
                    "barcode",
                    barcode_sidecar,
                    metadata={"source": "dorado_skip_bam_split_bi_tags"},
                )
                logger.info(
                    "Built barcode sidecar with BC/BM/bi from aligned BAM in non-split dorado mode: %s",
                    barcode_sidecar,
                )
            else:
                # The aligned BAM was never actually barcode-classified at basecall
                # time (no reliable BC/bi tags to read) -- scanning it for BC tags,
                # as this branch used to do, silently produces an almost-empty
                # sidecar (observed: 156 reads recovered out of >1M) rather than a
                # loud failure, since a handful of stray/mismatched tags is enough
                # to avoid the `if read_to_barcode:` empty-dict check. Run real
                # dorado classification instead, into a throwaway split directory
                # purely to harvest a reliable read->barcode sidecar from -- the
                # split BAMs themselves are discarded, honoring skip_bam_split's
                # promise that the aligned BAM stays the one artifact used
                # downstream. See dev/pipeline_scaling_audit.md.
                logger.info(
                    "skip_bam_split=True with dorado backend: bi tag not found on aligned BAM. "
                    "Running dorado demux classification into a temporary directory to build "
                    "a reliable barcode sidecar (BM tag/demux_type still unavailable; use "
                    "skip_bam_split=False for per-end barcode scoring)."
                )
                rebuild_barcode_sidecar_via_dorado_classification(
                    aligned_sorted_output,
                    barcode_sidecar,
                    barcode_kit=cfg.barcode_kit,
                    barcode_both_ends=getattr(cfg, "barcode_both_ends", False),
                    trim=cfg.trim,
                    threads=cfg.threads,
                    samtools_backend=cfg.samtools_backend,
                )
                register_sidecar(
                    sidecar_manifest,
                    "barcode",
                    barcode_sidecar,
                    metadata={"source": "dorado_skip_bam_split_reclassified"},
                )
                logger.info(
                    "Built barcode sidecar via dorado demux classification: %s",
                    barcode_sidecar,
                )
    elif cfg.input_already_demuxed or use_smftools_demux:
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
                barcode_sidecar=barcode_sidecar,
            )

            unclassified_bams = [p for p in all_bam_files if "unclassified" in p.name]
            bam_files = sorted(p for p in all_bam_files if "unclassified" not in p.name)

        se_bam_files = bam_files
        bam_dir = cfg.split_path
        double_barcoded_path = None

        # Ensure barcode sidecar exists for input_already_demuxed paths
        # (smftools demux already produces one above; this covers the remaining case)
        if barcode_sidecar is None and cfg.input_already_demuxed and bam_files:
            barcode_sidecar = aligned_sorted_output.with_suffix(".barcode_tags.parquet")
            build_barcode_sidecar_from_split_bams(
                bam_files, barcode_sidecar, samtools_backend=cfg.samtools_backend
            )
            register_sidecar(
                sidecar_manifest,
                "barcode",
                barcode_sidecar,
                metadata={"source": "input_already_demuxed_split_bams"},
            )

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

            # Derive BM from bi into sidecar (without modifying BAMs).
            # For no_classify: bi/BC are on the aligned_sorted BAM.
            # For classify: bi/BC are on the split BAMs after dorado re-classification.
            if not dorado_barcode_workspace.reusable:
                barcode_sidecar = aligned_sorted_output.with_suffix(".barcode_tags.parquet")
                if demux_mode == "no_classify":
                    derive_bm_from_bi_to_sidecar(
                        aligned_sorted_output,
                        barcode_sidecar,
                        samtools_backend=cfg.samtools_backend,
                    )
                    sidecar_source = "dorado_aligned_sorted_bam"
                else:
                    # Derive from each split BAM and concatenate
                    sidecar_dfs = []
                    for bam_file in bam_files:
                        if "unclassified" in bam_file.name:
                            continue
                        per_bam_sidecar = bam_file.with_suffix(".barcode_tags.parquet")
                        derive_bm_from_bi_to_sidecar(
                            bam_file,
                            per_bam_sidecar,
                            samtools_backend=cfg.samtools_backend,
                        )
                        sidecar_dfs.append(pd.read_parquet(per_bam_sidecar))
                    if sidecar_dfs:
                        bc_df = pd.concat(sidecar_dfs, ignore_index=True)
                        bc_df = bc_df.drop_duplicates(subset=["read_name"], keep="first")
                    else:
                        bc_df = pd.DataFrame(columns=["read_name", "BC", "BM"])
                    bc_df.to_parquet(barcode_sidecar, index=False)
                    sidecar_source = "dorado_single_pass_demux_bams"

                register_sidecar(
                    sidecar_manifest,
                    "barcode",
                    barcode_sidecar,
                    metadata={"source": sidecar_source},
                )
                logger.info("dorado barcode sidecar written: %s", barcode_sidecar)

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

            # Build barcode sidecar from split BAMs for old dorado (< 1.3.1)
            if barcode_sidecar is None and bam_files:
                barcode_sidecar = aligned_sorted_output.with_suffix(".barcode_tags.parquet")
                build_barcode_sidecar_from_split_bams(
                    bam_files, barcode_sidecar, samtools_backend=cfg.samtools_backend
                )
                register_sidecar(
                    sidecar_manifest,
                    "barcode",
                    barcode_sidecar,
                    metadata={"source": "dorado_legacy_split_bams"},
                )

    if dorado_barcode_workspace is not None and not dorado_barcode_workspace.reusable:
        if barcode_sidecar is None:
            raise RuntimeError("Dorado barcode processing did not produce a sidecar.")
        committed_barcode = dorado_barcode_workspace.root / "barcode.parquet"
        shutil.copy2(barcode_sidecar, committed_barcode)
        barcode_sidecar = committed_barcode
        commit_intermediate(dorado_barcode_workspace, {"barcode": barcode_sidecar})
        register_sidecar(
            sidecar_manifest,
            "barcode",
            barcode_sidecar,
            metadata={"source": "dorado_committed_intermediate"},
        )

    # Resolve every route-specific authority into one canonical identity
    # sidecar before QC and raw metadata consume it. This also covers the
    # input_already_demuxed=True + skip_bam_split=True route, where no split
    # BAM filenames exist from which to reconstruct labels.
    classifier_source = _barcode_classifier_source(cfg, demux_backend=demux_backend)
    route_barcode_sidecar = barcode_sidecar
    partition_barcode_sidecars: dict[Path, Path] = {}
    assert alignment_partitions is not None
    if len(alignment_partitions) == 1:
        barcode_sidecar, _barcode_identity_report = _publish_canonical_barcode_identity(
            output_directory=output_directory,
            aligned_bam=aligned_sorted_output,
            resolved_input_manifest=resolved_input_manifest,
            route_sidecar=route_barcode_sidecar,
            classifier_source=classifier_source,
            sidecar_manifest=sidecar_manifest,
            force_redo=force_redo_intermediates,
        )
        partition_barcode_sidecars[Path(aligned_sorted_output)] = barcode_sidecar
    else:
        for partition_bam, source_row in alignment_partitions:
            partition_manifest = replace(
                resolved_input_manifest,
                rows=(source_row,),
                digest=str(source_row.source_id),
            )
            partition_sidecar, _report = _publish_canonical_barcode_identity(
                output_directory=output_directory,
                aligned_bam=partition_bam,
                resolved_input_manifest=partition_manifest,
                route_sidecar=None,
                classifier_source=classifier_source,
                sidecar_manifest=sidecar_manifest,
                force_redo=force_redo_intermediates,
                sidecar_key_suffix=str(source_row.source_id),
            )
            partition_barcode_sidecars[Path(partition_bam)] = partition_sidecar
        barcode_sidecar = partition_barcode_sidecars[Path(aligned_sorted_output)]

    add_or_update_column_in_csv(cfg.summary_file, "demuxed_bams", [se_bam_files])

    if not skip_bam_split and getattr(cfg, "max_reads_per_barcode", None) is not None:
        logger.info(f"Subsampling split BAMs to max {cfg.max_reads_per_barcode} reads per barcode.")
        subsample_split_bams(
            bam_files,
            cfg.max_reads_per_barcode,
            samtools_backend=cfg.samtools_backend,
        )

    if cfg.make_beds and not skip_bam_split:
        # Make beds and provide basic histograms
        demux_bed_output_root = bed_outputs_directory / "demultiplexed"
        bed_dir = demux_bed_output_root / "beds"
        if bed_dir.is_dir():
            logger.debug(
                f"{bed_dir} already exists. Skipping BAM -> BED conversion for demultiplexed bams"
            )
        else:
            logger.info("Making BED files from BAM files for each sample")
            for bam in bam_files:
                aligned_BAM_to_bed(
                    bam,
                    demux_bed_output_root,
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
    bam_qc_dir = bam_outputs_directory / "bam_qc"
    skip_bam_qc = getattr(cfg, "skip_bam_qc", False)
    if skip_bam_qc:
        logger.info("skip_bam_qc=True: skipping BAM QC")
    elif bam_qc_dir.is_dir():
        logger.debug(f"{bam_qc_dir} already exists. Using existing BAM QC calculations.")
    else:
        make_dirs([bam_qc_dir])
        logger.info("Performing BAM QC")
        _qc_barcodes = None
        _qc_barcode_readname_map = None
        if skip_bam_split and barcode_sidecar and Path(barcode_sidecar).exists():
            _bc_df = pd.read_parquet(barcode_sidecar)
            _qc_barcodes = sorted(_bc_df["BC"].dropna().unique().tolist())
            _qc_barcodes = [b for b in _qc_barcodes if b != "unclassified"]
            _bc_df = _bc_df[_bc_df["BC"].isin(_qc_barcodes)]
            _qc_barcode_readname_map = {
                str(bc): set(group["read_name"].astype(str).tolist())
                for bc, group in _bc_df.groupby("BC", observed=True)
            }
            del _bc_df
        bam_qc(
            bam_files,
            bam_qc_dir,
            cfg.threads,
            modality=cfg.smf_modality,
            samtools_backend=cfg.samtools_backend,
            barcodes=_qc_barcodes,
            barcode_readname_map=_qc_barcode_readname_map,
        )
    ########################################################################################################################

    if raw_only:
        direct_signal_backend = str(getattr(cfg, "direct_signal_backend", "modkit"))
        direct_uses_modkit = cfg.smf_modality == "direct" and direct_signal_backend == "modkit"
        mod_tsv_paths: list[Path] | None = None
        if direct_uses_modkit:
            from ..informatics.modkit_functions import extract_mods, make_modbed, modQC

            direct_inputs = [("aligned-bam", artifact_checksum(aligned_sorted_output))]
            if barcode_sidecar is not None:
                direct_inputs.append(("barcode-sidecar", artifact_checksum(barcode_sidecar)))
            direct_mod_spec = IntermediateSpec(
                operation="direct-modification-extraction",
                input_artifacts=tuple(direct_inputs),
                operation_config={
                    "bam_suffix": str(cfg.bam_suffix),
                    "skip_unclassified": bool(cfg.skip_unclassified),
                    "thresholds": list(cfg.thresholds or []),
                },
                tool_versions={"modkit": executable_version("modkit")},
            )
            direct_mod_workspace = prepare_intermediate(
                output_directory,
                direct_mod_spec,
                force_redo=force_redo_intermediates,
            )
            if direct_mod_workspace.reusable:
                mod_bed_dir = committed_output(direct_mod_workspace, "mod-beds")
                mod_tsv_dir = committed_output(direct_mod_workspace, "mod-tsvs")
                if mod_bed_dir is None or mod_tsv_dir is None:
                    raise RuntimeError(
                        "Validated direct-mod commit is missing an output directory."
                    )
                logger.info(
                    "Reusing validated direct-modification extraction: %s",
                    direct_mod_workspace.root,
                )
            else:
                mod_bed_dir = direct_mod_workspace.root / "mod_beds"
                mod_tsv_dir = direct_mod_workspace.root / "mod_tsvs"
                require_memory_headroom(
                    cfg,
                    operation_label="modkit raw extraction",
                    estimator="external_modkit_peak",
                )
                make_dirs([mod_bed_dir])
                modQC(aligned_sorted_output, cfg.thresholds)
                make_modbed(aligned_sorted_output, cfg.thresholds, mod_bed_dir)
                make_dirs([mod_tsv_dir])
                extract_mods(
                    cfg.thresholds,
                    mod_tsv_dir,
                    bam_dir if bam_dir is not None else cfg.split_path,
                    cfg.bam_suffix,
                    skip_unclassified=cfg.skip_unclassified,
                    modkit_summary=False,
                    threads=cfg.threads,
                    single_bam=aligned_sorted_output,
                )
                commit_intermediate(
                    direct_mod_workspace,
                    {"mod-beds": mod_bed_dir, "mod-tsvs": mod_tsv_dir},
                )
            mod_tsv_paths = sorted(mod_tsv_dir.glob("*.tsv")) + sorted(mod_tsv_dir.glob("*.tsv.gz"))

        from ..readwrite import safe_read_h5ad

        logger.info("Extracting read-relative raw records from aligned BAM")
        # Streaming for every modality/backend combination: never holds more
        # than one reference's ragged data in memory at once (conversion/
        # deaminase, and direct modality's pysam backend), and for direct
        # modality's modkit backend, never holds more than one read-id
        # bucket's slice of the whole-experiment modkit-extract TSV either
        # (build_ragged_records_streaming splits+buckets it up front -- see
        # _split_modkit_tsv_by_bucket).
        frame = None
        from ..informatics.raw_store import write_raw_store_streaming
        from .raw_adata import (
            build_partitioned_ragged_records_streaming,
            build_ragged_records_streaming,
        )

        assert alignment_partitions is not None
        append_partition_identity = bool(getattr(cfg, "_raw_append_source_ids", ())) and bool(
            getattr(alignment_partitions[0][1], "namespace", "")
        )
        if len(alignment_partitions) == 1 and not append_partition_identity:
            reference_frames, reference_lengths, extra_uns = build_ragged_records_streaming(
                cfg,
                fasta=fasta,
                aligned_bam=aligned_sorted_output,
                barcode_sidecar=barcode_sidecar,
                umi_sidecar=umi_sidecar,
                mod_tsv_paths=mod_tsv_paths,
            )
        else:
            reference_frames, reference_lengths, extra_uns = (
                build_partitioned_ragged_records_streaming(
                    cfg,
                    fasta=fasta,
                    partitions=[
                        (
                            partition_bam,
                            str(source_row.namespace),
                            partition_barcode_sidecars[Path(partition_bam)],
                        )
                        for partition_bam, source_row in alignment_partitions
                    ],
                    umi_sidecar=umi_sidecar,
                )
            )
            extra_uns["alignment_source_partitions"] = {
                str(source_row.source_id): {
                    "source_id": str(source_row.source_id),
                    "namespace": str(source_row.namespace),
                }
                for partition_bam, source_row in alignment_partitions
            }
        extra_uns["experiment_id"] = cfg.experiment_id
        extra_uns["experiment"] = cfg.experiment_id
        extra_uns["reference_interval_map"] = relative_uns_path(reference_interval_map, run_root)
        extra_uns["region_catalogs"] = {
            scope: relative_uns_path(path, run_root)
            for scope, path in sorted(region_catalog_paths.items())
        }
        extra_uns["region_catalog_schema_version"] = REGION_CATALOG_SCHEMA_VERSION
        extra_uns["reference_interval_map_schema_version"] = REFERENCE_INTERVAL_MAP_SCHEMA_VERSION
        raw_paths = write_raw_store_streaming(
            reference_frames,
            load_directory,
            reference_lengths=reference_lengths,
            shard_size=int(getattr(cfg, "raw_parquet_shard_size", 100_000)),
            start_bin_size=int(getattr(cfg, "parquet_start_bin_size", 1_000_000)),
            analysis_mode=getattr(cfg, "analysis_mode", "auto"),
            load_cache_mode=getattr(cfg, "load_cache_mode", "auto"),
            max_full_matrix_gb=float(getattr(cfg, "max_full_matrix_gb", 8.0)),
            genome_tile_size=int(getattr(cfg, "genome_tile_size", 10_000)),
            genome_tile_halo=int(getattr(cfg, "genome_tile_halo", 1_000)),
            bam_path=aligned_sorted_output,
            extra_uns=extra_uns,
            refresh_experiment_spine=False,
        )
        # The streaming path never materializes one experiment-wide frame
        # (that's the whole point) -- downstream steps below that used to
        # read `frame` (molecule count, the chimera-rate plot) read the
        # just-written spine.obs instead, which is already documented as
        # an equivalent, and cheaper since it's scalar-only, no ragged
        # array columns (see plot_reference_barcode_chimera_rate's own
        # docstring: "the raw spine obs or ragged frame").
        n_molecules = safe_read_h5ad(raw_paths["spine"], verbose=False)[0].n_obs

        # Consolidated provenance manifest (dev/experiment_storage_schema.md, Phase 2):
        # config-by-value, input/FASTA paths, and a readable stage-completion index --
        # none of which spine.uns previously captured (only a hash, not the values, and
        # nothing for input_data_path at all).
        from ..informatics.experiment_manifest import (
            update_experiment_manifest,
        )

        if append_source_ids:
            cfg.input_files = [Path(row.path) for row in full_input_manifest.rows]
            cfg._resolved_input_manifest = full_input_manifest
            cfg.input_data_path = requested_input_data_path
        resolved_config = cfg.to_dict()
        update_experiment_manifest(
            run_root,
            experiment_id=cfg.experiment_id,
            experiment=cfg.experiment_id,
            modality=extra_uns.get("modality"),
            input_data_path=(
                relative_uns_path(cfg.input_data_path, run_root) if cfg.input_data_path else None
            ),
            input_manifest={
                "schema_version": 1,
                "digest": cfg.input_manifest_digest,
                "artifacts": {
                    "csv": relative_uns_path(
                        output_directory
                        / RAW_DIR
                        / "input_manifest"
                        / "resolved_input_manifest.csv",
                        run_root,
                    ),
                    "json": relative_uns_path(
                        output_directory
                        / RAW_DIR
                        / "input_manifest"
                        / "resolved_input_manifest.json",
                        run_root,
                    ),
                    "resolution_report": relative_uns_path(
                        output_directory
                        / RAW_DIR
                        / "input_manifest"
                        / "input_resolution_report.json",
                        run_root,
                    ),
                },
            },
            fasta_path=relative_uns_path(fasta, run_root) if fasta else None,
            reference_uids=extra_uns.get("reference_uids"),
            reference_lengths={str(k): int(v) for k, v in reference_lengths.items()},
            reference_interval_map=relative_uns_path(reference_interval_map, run_root),
            region_catalogs={
                scope: relative_uns_path(path, run_root)
                for scope, path in sorted(region_catalog_paths.items())
            },
            region_catalog_schema_version=REGION_CATALOG_SCHEMA_VERSION,
            reference_interval_map_schema_version=REFERENCE_INTERVAL_MAP_SCHEMA_VERSION,
            config=resolved_config,
        )
        spine, _ = safe_read_h5ad(raw_paths["spine"])
        if str(cfg.smf_modality) == "deaminase" and not getattr(
            cfg, "bypass_raw_chimera_rate_plot", False
        ):
            try:
                from ..plotting import plot_reference_barcode_chimera_rate

                # frame is None on the streaming path (deaminase always takes
                # it) -- spine.obs is a documented-equivalent input (see
                # plot_reference_barcode_chimera_rate's own docstring) and is
                # what's actually available without re-materializing the
                # whole experiment. Its barcode column is the canonicalized
                # "Barcode" (constants.BARCODE), not the ragged frame's
                # lowercase "barcode" the function defaults to -- must be
                # passed explicitly or every read falls out of the group-by.
                plot_reference_barcode_chimera_rate(
                    frame if frame is not None else spine.obs,
                    load_directory / "plots",
                    barcode_column="barcode" if frame is not None else "Barcode",
                    min_events_per_span=cfg.deaminase_chimera_min_events_per_span,
                    min_segment_purity=cfg.deaminase_chimera_min_segment_purity,
                    max_single_strand_fraction=cfg.deaminase_chimera_max_single_strand_fraction,
                )
            except Exception:
                logger.warning("Failed to plot reference x barcode chimera rate.", exc_info=True)

        mqc_dir = bam_outputs_directory / "multiqc"
        if skip_bam_qc:
            logger.info("skip_bam_qc=True: skipping multiqc")
        elif not mqc_dir.is_dir():
            require_memory_headroom(
                cfg,
                operation_label="MultiQC",
                estimator="external_multiqc_peak",
            )
            run_multiqc(bam_qc_dir, mqc_dir)
        return spine, raw_paths["spine"], cfg

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
            bam_dir if bam_dir is not None else cfg.split_path,
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
            single_bam=aligned_sorted_output,
            barcode_sidecar=barcode_sidecar,
        )
    else:
        require_memory_headroom(
            cfg,
            operation_label="modkit dense extraction",
            estimator="external_modkit_peak",
        )
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
            bam_dir if bam_dir is not None else cfg.split_path,
            cfg.bam_suffix,
            skip_unclassified=cfg.skip_unclassified,
            modkit_summary=False,
            threads=cfg.threads,
            single_bam=aligned_sorted_output,
        )  # Extract methylations calls for split BAM files into split TSV files

        from ..informatics.modkit_extract_to_adata import modkit_extract_to_adata

        logger.info("Making Anndata for direct modification detection SMF samples")

        # 6 Load the modification data from TSVs into an adata object
        raw_adata, raw_adata_path = modkit_extract_to_adata(
            fasta,
            bam_dir if bam_dir is not None else cfg.split_path,
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
            single_bam=aligned_sorted_output,
            barcode_sidecar=barcode_sidecar,
            max_workers=getattr(cfg, "direct_max_workers", None),
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
    from functools import partial as _partial

    se_bam_files = [aligned_sorted_output]
    _extract_features = _partial(extract_read_features_from_bam, primary_only=True)
    add_read_length_and_mapping_qc(
        raw_adata,
        se_bam_files,
        extract_read_features_from_bam_callable=_extract_features,
        bypass=cfg.bypass_add_read_length_and_mapping_qc,
        force_redo=cfg.force_redo_add_read_length_and_mapping_qc,
        samtools_backend=cfg.samtools_backend,
    )

    # Build default tag list: always NM/MD, MM/ML only for direct modality
    default_tags = ["NM", "MD", "fn"]
    if cfg.smf_modality == "direct":
        default_tags.extend(["MM", "ML"])
    # UMI tags are loaded from Parquet sidecar below (not from BAM)
    # Barcode tags (BC/BM) are loaded from Parquet sidecar below (not from BAM)
    # Only extract BC and bi from BAM for dorado; BM is derived into sidecar
    if demux_backend == "dorado" and cfg.barcode_kit and not cfg.input_already_demuxed:
        dorado_ver = _get_dorado_version()
        if dorado_ver is not None and dorado_ver >= (1, 3, 1):
            default_tags.extend(["BC", "bi"])
    bam_tag_names = getattr(cfg, "bam_tag_names", default_tags)

    logger.info("Adding BAM tags and BAM flags to adata.obs")
    _extract_tags = _partial(extract_read_tags_from_bam, primary_only=True)
    add_read_tag_annotations(
        raw_adata,
        se_bam_files,
        tag_names=bam_tag_names,
        include_flags=True,
        include_cigar=True,
        extract_read_tags_from_bam_callable=_extract_tags,
        samtools_backend=cfg.samtools_backend,
    )

    # Load UMI tags from Parquet sidecar (written by annotate_umi_tags_in_bam)
    if getattr(cfg, "use_umi", False) and (not umi_sidecar or not Path(umi_sidecar).exists()):
        _resolved_umi_sidecar = resolve_sidecar(sidecar_manifest, "umi_oriented")
        if _resolved_umi_sidecar is not None:
            umi_sidecar = _resolved_umi_sidecar

    if getattr(cfg, "use_umi", False) and umi_sidecar and Path(umi_sidecar).exists():
        logger.info("Loading UMI tags from Parquet sidecar: %s", umi_sidecar)
        umi_df = pd.read_parquet(umi_sidecar).set_index("read_name")
        umi_df = umi_df.reindex(raw_adata.obs_names)
        for col in ["U1", "U2", "RX", "FC", "US", "UE"]:
            if col in umi_df.columns:
                raw_adata.obs[col] = umi_df[col].values
        del umi_df

    # Load the canonical barcode/sample identity sidecar for every route.
    if not barcode_sidecar or not Path(barcode_sidecar).exists():
        _resolved_barcode_sidecar = resolve_sidecar(sidecar_manifest, "barcode")
        if _resolved_barcode_sidecar is not None:
            barcode_sidecar = _resolved_barcode_sidecar

    if barcode_sidecar and Path(barcode_sidecar).exists():
        logger.info("Loading barcode/sample identity from Parquet sidecar: %s", barcode_sidecar)
        _attach_dense_barcode_identity(raw_adata, barcode_sidecar, cfg.experiment_name)

    # Expand dorado bi array tag into individual float score columns
    if "bi" in raw_adata.obs.columns or "bi" in bam_tag_names:
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
    mqc_dir = bam_outputs_directory / "multiqc"
    if skip_bam_qc:
        logger.info("skip_bam_qc=True: skipping multiqc")
    elif mqc_dir.is_dir():
        logger.info(f"{mqc_dir} already exists, skipping multiqc")
    else:
        require_memory_headroom(
            cfg,
            operation_label="MultiQC",
            estimator="external_multiqc_peak",
        )
        logger.info("Running multiqc")
        run_multiqc(bam_qc_dir, mqc_dir)
    ########################################################################################################################

    ############################################### delete intermediate BAM files ###############################################
    if cfg.delete_intermediate_bams:
        logger.info("Deleting intermediate BAM files")
        logger.info(
            "Retaining committed alignment BAM and index so their immutable revision remains reusable"
        )
        # delete the demultiplexed bams. Keep the demultiplexing summary files and directories to faciliate demultiplexing in the future with these files
        for bam in bam_files:
            if Path(bam).resolve() == aligned_sorted_output.resolve():
                continue
            bai = bam.parent / (bam.name + ".bai")
            if bam.exists():
                bam.unlink()
            if bai.exists():
                bai.unlink()
        for bam in unclassified_bams:
            bai = bam.parent / (bam.name + ".bai")
            if bam.exists():
                bam.unlink()
            if bai.exists():
                bai.unlink()
        logger.info("Finished deleting intermediate BAM files")
    ########################################################################################################################

    return raw_adata, raw_adata_path, cfg
