from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import anndata as ad

from smftools.constants import (
    BAM_OUTPUTS_DIR,
    BED_OUTPUTS_DIR,
    CHIMERIC_DIR,
    FASTA_OUTPUTS_DIR,
    H5_DIR,
    HMM_DIR,
    LATENT_DIR,
    LOAD_DIR,
    MODKIT_OUTPUTS_DIR,
    PREPROCESS_DIR,
    RAW_DIR,
    SPATIAL_DIR,
    SPLIT_DIR,
    VARIANT_DIR,
)
from smftools.logging_utils import get_logger

from ..metadata import write_runtime_schema_yaml
from ..readwrite import safe_write_h5ad

logger = get_logger(__name__)
_RESOURCE_ENVELOPE_CACHE: dict[tuple[Any, ...], Any] = {}

_NON_SEMANTIC_STAGE_CONFIG_KEYS = {
    "bam_outputs_path",
    "bed_outputs_path",
    "device",
    "emit_log_file",
    "emit_perf_log",
    "fasta_outputs_path",
    "full_run_latent",
    "hmm_device",
    "informatics_outputs_path",
    "log_level",
    "memory_reserve_gb",
    "output_directory",
    "modkit_outputs_path",
    "perf_log_sample_interval_seconds",
    "plot_threads_fraction",
    "split_path",
    "summary_file",
    "threads",
    "max_memory_gb",
    "max_memory_percent",
    "target_task_memory_mb",
}
#: Settings that only change how results are *displayed*, never what is
#: computed (`F30`). `reindex_references_adata` adds a `<reference>_reindexed`
#: var column and its docstring is explicit that it "never touches
#: X/layers/var_names -- it is purely a reinterpretation of the reindexed
#: coordinate value". Correcting a TSS offset previously invalidated the raw
#: stage, forcing a full rebuild from FASTQ to relabel a plot axis.
_DISPLAY_ONLY_CONFIG_KEYS = {
    "reindexing_offsets",
    "reindexing_invert",
    "reindexed_var_suffix",
    # Colour maps and which layer a clustermap draws change the picture, never
    # the matrix behind it. They reach four stages apiece, so leaving them
    # semantic made a palette tweak invalidate hmm and spatial together
    # (`F46`).
    "clustermap_cmap_a",
    "clustermap_cmap_c",
    "clustermap_cmap_cpg",
    "clustermap_cmap_gpc",
    "layer_for_clustermap_plotting",
}

#: Stage order, used to keep a stage from being invalidated by settings that
#: belong to stages running after it (`F30`). Turning on a spatial plot
#: previously marked the raw store stale.
_STAGE_ORDER = ("raw", "preprocess", "spatial", "hmm", "latent")


#: Name fragments that identify which stage a config key belongs to, for keys
#: that do not carry a stage prefix (`F46`).
#:
#: `_downstream_config_prefixes` below only strips keys literally prefixed
#: `preprocess_`/`spatial_`/`hmm_`/`latent_`, and most settings do not follow
#: that convention: `bypass_deamination_segmentation`,
#: `read_len_filter_thresholds` and `deaminase_segment_penalty_scale` are all
#: preprocess-owned and all sat in raw's hash. Raw carried 279 keys, so a
#: filter-threshold tweak forced a 36-minute re-extraction.
#:
#: Fragments must be specific enough that they cannot match a key an *earlier*
#: stage owns. A false match here excludes a genuinely relevant key and lets a
#: stale stage be reused silently, which is the failure this file's denylist
#: design exists to avoid; a missed fragment merely recomputes too often.
_STAGE_OWNED_KEY_MARKERS: dict[str, tuple[str, ...]] = {
    "preprocess": (
        "deaminase",
        "deamination",
        "chimera",
        "duplicate",
        "filter_thresholds",
        "clean_nan",
        "base_context",
        "mismatch_frequency",
        "read_modification_stats",
        "complexity_analysis",
        "binary_layer",
        "variant_",
    ),
    "spatial": ("spatial_", "autocorr", "matrix_corr"),
    "hmm": ("hmm_",),
    "latent": ("latent_", "umap", "leiden"),
}


def _stage_owning_key(key: str) -> str | None:
    """The stage a config key belongs to by name, or ``None`` when unclear.

    ``bypass_x`` and ``force_redo_x`` are resolved through ``x``: a switch for
    one stage's work cannot affect a stage that runs before it.
    """
    name = str(key)
    for prefix in ("bypass_", "force_redo_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for stage, markers in _STAGE_OWNED_KEY_MARKERS.items():
        if any(marker in name for marker in markers):
            return stage
    return None


def _is_downstream_owned(key: str, stage: str) -> bool:
    """True when ``key`` belongs to a stage that runs after ``stage``."""
    if stage not in _STAGE_ORDER:
        return False
    owner = _stage_owning_key(key)
    if owner is None or owner not in _STAGE_ORDER:
        return False
    return _STAGE_ORDER.index(owner) > _STAGE_ORDER.index(stage)


def _downstream_config_prefixes(stage: str) -> tuple[str, ...]:
    """Config-key prefixes owned by stages that run after ``stage``."""
    if stage not in _STAGE_ORDER:
        return ()
    return tuple(f"{name}_" for name in _STAGE_ORDER[_STAGE_ORDER.index(stage) + 1 :])


_STAGE_NON_SEMANTIC_CONFIG_KEYS = {
    stage: {"plot_regions_bed", "plot_allow_unanalyzed_gaps", "plot_subsample_seed"}
    for stage in ("preprocess", "spatial", "hmm", "latent", "full")
}
_STAGE_NON_SEMANTIC_CONFIG_KEYS["raw"] = {
    "alignment_regions_bed",
    "fasta",
    "input_data_path",
    "input_files",
    "input_manifest_digest",
    "input_manifest_path",
}
_STAGE_SEMANTIC_CONFIG_KEYS = {
    "latent": {
        "from_adata_stage",
        "latent_cp_iterations",
        "latent_cp_memory_policy",
        "latent_cp_rank",
        "latent_execution_mode",
        "latent_knn_neighbors",
        "latent_leiden_resolution",
        "latent_max_fit_reads",
        "latent_min_reads",
        "latent_n_pcs",
        "latent_nmf_components",
        "latent_nmf_max_iter",
        "latent_random_state",
        "latent_run_cp",
        "latent_run_nmf",
        "latent_run_pca_umap",
        "latent_transform_chunk_reads",
        "layer_for_umap_plotting",
        "mod_target_bases",
        "smf_modality",
    },
}
_STAGE_PLOT_CONFIG_KEYS = {
    "latent": {
        "sample_name_col_for_plotting",
        "latent_plot_max_reads",
        "umap_layers_to_plot",
    }
}

# Canonical mapping from user-facing stage aliases to AdataPaths attribute names
STAGE_MAP = {
    "raw": "raw",
    "load": "raw",
    "pp": "pp",
    "preprocess": "pp",
    "pp_dedup": "pp_dedup",
    "preprocess_dedup": "pp_dedup",
    "spatial": "spatial",
    "hmm": "hmm",
    "latent": "latent",
    "variant": "variant",
    "chimeric": "chimeric",
}


@dataclass
class AdataPaths:
    raw: Path
    pp: Path
    pp_dedup: Path
    spatial: Path
    hmm: Path
    latent: Path
    variant: Path
    chimeric: Path
    # Optional dense zarr cache for the load stage, built on demand by
    # `smftools load` / cli.load_adata.load_dense_cache (see
    # informatics/partition_store.write_dense_cache_from_spine). Not written by
    # the default raw_adata()/full_flow path.
    store: Path | None = None
    spine: Path | None = None
    catalog: Path | None = None
    raw_spine: Path | None = None
    preprocess_spine: Path | None = None
    spatial_spine: Path | None = None
    hmm_spine: Path | None = None
    latent_spine: Path | None = None


@dataclass
class ArtifactPaths:
    """Canonical path bundle for split `smftools load` sub-steps.

    This path resolver centralizes commonly shared files so CLI
    raw sub-steps (`basecall`, `align`, `barcode`, `umi`, `modbase`) use a
    single source of truth for input/output locations.
    """

    output_directory: Path
    load_directory: Path
    bam_outputs_directory: Path
    fasta_outputs_directory: Path
    bed_outputs_directory: Path
    modkit_outputs_directory: Path
    split_directory: Path
    bam_qc_directory: Path
    mod_tsv_directory: Path
    mod_bed_directory: Path
    sidecar_manifest: Path
    raw_directory: Path
    spine: Path
    dense_store: Path
    dense_catalog: Path

    unaligned_bam: Path
    aligned_bam: Path
    aligned_sorted_bam: Path
    aligned_sorted_bai: Path

    barcode_sidecar: Path
    barcode_positional_sidecar: Path
    umi_positional_sidecar: Path
    umi_oriented_sidecar: Path

    def as_dict(self) -> dict[str, str]:
        """Serialize all path fields as strings."""
        return {k: str(v) for k, v in self.__dict__.items()}


def resolved_stage_config(cfg, stage: str | None = None) -> dict[str, Any]:
    """Return semantic config values used for stage compatibility checks."""
    if str(stage) == "preprocess":
        from ..preprocessing.semantic_upgrade import preprocess_stage_compute_config

        return preprocess_stage_compute_config(cfg)
    if hasattr(cfg, "to_dict"):
        values = dict(cfg.to_dict())
    else:
        values = dict(vars(cfg))
    ignored = (
        _NON_SEMANTIC_STAGE_CONFIG_KEYS
        | _STAGE_NON_SEMANTIC_CONFIG_KEYS.get(str(stage), set())
        | _DISPLAY_ONLY_CONFIG_KEYS
    )
    # A stage cannot be affected by settings belonging to stages that run after
    # it, so those must not invalidate it (`F30`). Subtracting known-irrelevant
    # keys rather than enumerating relevant ones is deliberate: an allowlist
    # that is too narrow silently reuses a stage that should have been
    # recomputed, which is far worse than recomputing one too often.
    downstream = _downstream_config_prefixes(str(stage))
    resolved = {
        key: value
        for key, value in values.items()
        if key not in ignored
        and not key.startswith("force_redo_")
        and not key.endswith("_max_workers")
        and not (downstream and key.startswith(downstream))
        and not _is_downstream_owned(key, str(stage))
    }
    selected = _STAGE_SEMANTIC_CONFIG_KEYS.get(str(stage))
    if selected is not None:
        return {key: resolved[key] for key in sorted(selected) if key in resolved}
    return resolved


def resolved_stage_plot_config(cfg, stage: str) -> dict[str, Any]:
    """Return plot-only config values that do not invalidate stage computation."""
    if hasattr(cfg, "to_dict"):
        values = dict(cfg.to_dict())
    else:
        values = dict(vars(cfg))
    selected = _STAGE_PLOT_CONFIG_KEYS.get(str(stage), set())
    return {key: values[key] for key in sorted(selected) if key in values}


def stage_config_hash(cfg, stage: str | None = None) -> str:
    """Hash semantic stage configuration without machine-local resource limits."""
    from ..informatics.experiment_manifest import config_hash

    return config_hash(resolved_stage_config(cfg, stage))


def stage_plot_config_hash(cfg, stage: str) -> str:
    """Hash plot-only stage configuration independently from compute settings."""
    from ..informatics.experiment_manifest import config_hash

    return config_hash(resolved_stage_plot_config(cfg, stage))


def stage_input_artifact_ids(
    run_root: str | Path,
    source_path: str | Path | None,
    *,
    include_region_catalogs: bool = False,
) -> list[str]:
    """Return stable source-file and upstream-stage identities for compatibility."""
    if source_path is None:
        return []
    from ..informatics.experiment_manifest import (
        artifact_record,
        config_hash,
        read_experiment_manifest,
    )

    run_root = Path(run_root)
    source_path = Path(source_path)
    source = artifact_record(source_path, run_root, checksum=True)
    identities = [f"path:{config_hash(source)}"]
    if include_region_catalogs and source_path.suffix == ".h5ad":
        from ..informatics.partition_read import resolve_relative_path
        from ..readwrite import safe_read_h5ad

        spine, _ = safe_read_h5ad(source_path)
        configured_catalogs = spine.uns.get("region_catalogs", {})
        if isinstance(configured_catalogs, dict):
            identities.append(f"region-config:{config_hash(dict(configured_catalogs))}")
            for scope, value in sorted(configured_catalogs.items()):
                path = resolve_relative_path(value, run_root)
                if path is not None and path.is_file():
                    record = artifact_record(path, run_root, checksum=True)
                    identities.append(f"region:{scope}:{config_hash(record)}")
    stage_by_dir = {
        RAW_DIR: "raw",
        PREPROCESS_DIR: "preprocess",
        SPATIAL_DIR: "spatial",
        HMM_DIR: "hmm",
        LATENT_DIR: "latent",
    }
    source_stage_dir = source_path.parent.name
    if source_path.parent.parent.name == "generations":
        source_stage_dir = source_path.parent.parent.parent.name
    source_stage = stage_by_dir.get(source_stage_dir)
    if source_stage is not None:
        entry = read_experiment_manifest(run_root).get("stages", {}).get(source_stage)
        if isinstance(entry, dict):
            provenance = {
                key: entry.get(key)
                for key in (
                    "config_hash",
                    "completed_at",
                    "generation_id",
                    "input_artifact_ids",
                    "schema_versions",
                )
            }
            identities.append(f"stage:{source_stage}:{config_hash(provenance)}")
    return identities


def raw_input_artifact_ids(cfg: Any) -> list[str]:
    """Return ordered content identities for raw sources and alignment reference."""
    from ..informatics.input_manifest import resolve_input_manifest_readonly
    from ..informatics.raw_intermediate_manifest import alignment_reference_bundle

    input_manifest_path = getattr(cfg, "input_manifest_path", None)
    input_files = getattr(cfg, "input_files", None)
    identities: list[str] = []
    if input_manifest_path or input_files:
        resolved = resolve_input_manifest_readonly(
            input_manifest_path=input_manifest_path,
            input_paths=None if input_manifest_path else input_files,
            alignment_mode=getattr(cfg, "alignment_mode", "align"),
            modality=getattr(cfg, "smf_modality", ""),
            barcode_map=getattr(cfg, "fastq_barcode_map", None),
            auto_pair=bool(getattr(cfg, "fastq_auto_pairing", True)),
        )
        identities.append(f"input-manifest:{resolved.digest}")
        identities.extend(f"source:{row.source_id}:{row.sha256}" for row in resolved.rows)
    if getattr(cfg, "fasta", None):
        reference = alignment_reference_bundle(cfg)
        identities.append(f"alignment-reference-bundle:{reference['digest']}")
    return identities


def stage_lifecycle(
    cfg,
    stage: str,
    source_path: str | Path | None = None,
    *,
    input_artifact_ids: list[str] | None = None,
):
    """Create a lifecycle context for one partitioned CLI stage."""
    from ..informatics.experiment_manifest import StageLifecycle

    run_root = Path(cfg.output_directory)
    return StageLifecycle(
        run_root,
        str(stage),
        config_hash=stage_config_hash(cfg, stage),
        input_artifact_ids=(
            input_artifact_ids
            if input_artifact_ids is not None
            else stage_input_artifact_ids(
                run_root,
                source_path,
                include_region_catalogs=str(stage) == "latent",
            )
        ),
    )


def publish_stage_outputs(
    lifecycle,
    outputs: dict[str, Path],
    *,
    required: tuple[str, ...],
    task_catalog_key: str | None = "task_catalog",
    checksum_keys: tuple[str, ...] = ("manifest", "task_catalog"),
    schema_versions: dict[str, int] | None = None,
    task_count: int | None = None,
    extra: dict[str, Any] | None = None,
    nonempty_directory_keys: tuple[str, ...] = (),
) -> None:
    """Validate executor outputs and publish the terminal complete record."""
    from ..informatics.experiment_manifest import artifact_record, stage_is_complete

    missing = [key for key in required if key not in outputs or not Path(outputs[key]).exists()]
    if missing:
        raise RuntimeError(f"stage did not publish required artifact(s): {missing}")

    artifacts = {
        key: artifact_record(
            path,
            lifecycle.run_root,
            checksum=key in checksum_keys,
            artifact_id=f"{lifecycle.stage}:{key}:{lifecycle.config_hash}",
            require_nonempty=key in nonempty_directory_keys,
        )
        for key, path in outputs.items()
        if path is not None and isinstance(path, (str, Path)) and Path(path).exists()
    }
    expected_tasks = task_count
    if task_catalog_key is not None and task_catalog_key in outputs:
        import pandas as pd

        expected_tasks = len(pd.read_parquet(outputs[task_catalog_key]))
    completion_extra = dict(extra or {})
    from ..pipeline.experiment_graph import experiment_stage_result_metadata

    completion_extra.update(
        experiment_stage_result_metadata(
            lifecycle.stage,
            stage_config_hash=lifecycle.config_hash,
            input_artifact_ids=lifecycle.input_artifact_ids,
            artifacts=artifacts,
            schema_versions=dict(schema_versions or {}),
        )
    )
    lifecycle.complete(
        artifacts=artifacts,
        expected_tasks=expected_tasks,
        successful_tasks=expected_tasks,
        schema_versions=dict(schema_versions or {}),
        **completion_extra,
    )
    if not stage_is_complete(
        lifecycle.run_root,
        lifecycle.stage,
        config_hash=lifecycle.config_hash,
        required_artifacts=required,
    ):
        raise RuntimeError(f"published {lifecycle.stage} stage record failed validation")


def partitioned_stage_is_complete(
    cfg,
    stage: str,
    *,
    required: tuple[str, ...],
    source_path: str | Path | None = None,
    extra_matches: dict[str, Any] | None = None,
    allow_previous_complete: bool = False,
) -> bool:
    """Check the compatible completion record used by partitioned CLI skips.

    This gate decides whether a stage actually reruns, so it must agree with the
    planner about what "complete" means. It used to compare config hash, input
    artifacts and required artifacts but *not* the algorithm version, so
    `smftools experiment plan` would report `stale_algorithm` and the run would
    skip the stage anyway -- making every version bump advisory (`F41`).

    A manifest entry predating the field records nothing there, compares
    unequal, and reruns. That is the safe direction.
    """
    from ..informatics.experiment_manifest import stage_is_complete
    from ..pipeline.experiment_graph import stage_algorithm_version

    if str(stage) == "raw":
        input_artifact_ids = raw_input_artifact_ids(cfg)
    else:
        input_artifact_ids = (
            stage_input_artifact_ids(
                cfg.output_directory,
                source_path,
                include_region_catalogs=str(stage) == "latent",
            )
            if source_path is not None
            else None
        )
    matches = dict(extra_matches or {})
    algorithm_version = stage_algorithm_version(stage)
    if algorithm_version is not None:
        matches.setdefault("semantic_algorithm_version", algorithm_version)
    return stage_is_complete(
        cfg.output_directory,
        stage,
        config_hash=stage_config_hash(cfg, stage),
        input_artifact_ids=input_artifact_ids,
        required_artifacts=required,
        extra_matches=matches,
        allow_previous_complete=allow_previous_complete,
    )


def get_adata_paths(
    cfg,
    *,
    allow_invalid_raw: bool = False,
    lineage_generations: Mapping[str, str] | None = None,
) -> AdataPaths:
    """Compute all standard AnnData paths for an experiment.

    Args:
        cfg: Loaded experiment configuration.
        allow_invalid_raw: Fall back to the legacy raw spine when the current
            generation selector is invalid. Reserved for raw-stage recovery.
        lineage_generations: Optional ``stage -> generation id`` map pinning
            which generation each stage resolves. This is the `D1` selector: a
            re-basecalling lineage reads its own descendant generations while
            ``current.json`` keeps answering for everyone else.

    Returns:
        The canonical and generation-selected artifact paths.
    """
    lineage_generations = dict(lineage_generations or {})
    output_directory = Path(cfg.output_directory)

    # Raw and Preprocessed adata file pathes will have set names.
    raw = output_directory / LOAD_DIR / H5_DIR / f"{cfg.experiment_name}.h5ad.gz"
    pp = output_directory / PREPROCESS_DIR / H5_DIR / f"{cfg.experiment_name}_preprocessed.h5ad.gz"

    if cfg.smf_modality == "direct":
        # direct SMF: duplicate-removed path is just preprocessed path
        pp_dedup = pp
    else:
        pp_dedup = (
            output_directory
            / PREPROCESS_DIR
            / H5_DIR
            / f"{cfg.experiment_name}_preprocessed_duplicates_removed.h5ad.gz"
        )

    pp_dedup_base = pp_dedup.name.removesuffix(".h5ad.gz")

    # All of the following just append a new suffix to the preprocessesed_deduplicated base name
    spatial = output_directory / SPATIAL_DIR / H5_DIR / f"{pp_dedup_base}_spatial.h5ad.gz"
    hmm = output_directory / HMM_DIR / H5_DIR / f"{pp_dedup_base}_hmm.h5ad.gz"
    latent = output_directory / LATENT_DIR / H5_DIR / f"{pp_dedup_base}_latent.h5ad.gz"
    variant = output_directory / VARIANT_DIR / H5_DIR / f"{pp_dedup_base}_variant.h5ad.gz"
    chimeric = output_directory / CHIMERIC_DIR / H5_DIR / f"{pp_dedup_base}_chimeric.h5ad.gz"

    # Dense-cache artifacts live in the load directory (output/LOAD_DIR), matching
    # write_dense_cache_from_spine(output_dir=load_directory) in load_dense_cache.
    load_dir = output_directory / LOAD_DIR
    store = load_dir / "store"
    spine = load_dir / "spine.h5ad"
    catalog = load_dir / "catalog.parquet"
    raw_output_dir = output_directory / RAW_DIR
    from ..informatics.generation import resolve_stage_generation
    from ..informatics.raw_generation import RawGenerationError, resolve_current_raw_generation

    def _stage_spine(stage: str, stage_dir: Path) -> Path:
        """Resolve one stage's spine, honouring a lineage pin when present."""
        pinned = lineage_generations.get(stage)
        if pinned is None:
            return stage_dir / "spine.h5ad"
        resolved = resolve_stage_generation(stage_dir, pinned)
        if resolved is None:
            raise RawGenerationError(f"{stage} lineage generation {pinned!r} could not be resolved")
        return resolved[0] / "spine.h5ad"

    if "raw" in lineage_generations:
        current_raw_generation = None
        raw_spine = _stage_spine("raw", raw_output_dir)
    else:
        try:
            current_raw_generation = resolve_current_raw_generation(raw_output_dir)
        except RawGenerationError:
            if not allow_invalid_raw:
                raise
            current_raw_generation = None
        raw_spine = (
            current_raw_generation[0] / "spine.h5ad"
            if current_raw_generation is not None
            else raw_output_dir / "spine.h5ad"
        )
    preprocess_spine = _stage_spine("preprocess", output_directory / PREPROCESS_DIR)
    spatial_spine = _stage_spine("spatial", output_directory / SPATIAL_DIR)
    hmm_spine = _stage_spine("hmm", output_directory / HMM_DIR)
    latent_spine = _stage_spine("latent", output_directory / LATENT_DIR)

    return AdataPaths(
        raw=raw,
        pp=pp,
        pp_dedup=pp_dedup,
        spatial=spatial,
        hmm=hmm,
        latent=latent,
        variant=variant,
        chimeric=chimeric,
        store=store,
        spine=spine,
        catalog=catalog,
        raw_spine=raw_spine,
        preprocess_spine=preprocess_spine,
        spatial_spine=spatial_spine,
        hmm_spine=hmm_spine,
        latent_spine=latent_spine,
    )


def _derive_load_bam_stem(cfg, load_directory: Path) -> str:
    """Infer canonical BAM stem used by load sub-steps.

    Mirrors current `load_adata` naming conventions:
    - basecalling inputs: model-based stem
    - BAM input: input BAM stem
    """
    input_type = str(getattr(cfg, "input_type", "") or "").lower()
    if input_type == "pod5":
        model_basename = Path(str(getattr(cfg, "model", "model"))).name.replace(".", "_")
        if str(getattr(cfg, "smf_modality", "")).lower() == "direct":
            mod_list = list(getattr(cfg, "mod_list", []) or [])
            mod_string = "_".join(mod_list) if mod_list else "mods"
            return f"{model_basename}_{mod_string}_calls"
        return f"{model_basename}_canonical_basecalls"

    input_data_path = getattr(cfg, "input_data_path", None)
    if input_data_path:
        return Path(str(input_data_path)).stem

    experiment_name = str(getattr(cfg, "experiment_name", "smftools"))
    return f"{experiment_name}_canonical_basecalls"


def get_artifact_paths(cfg, bam_stem: str | None = None) -> ArtifactPaths:
    """Resolve canonical artifact paths for split `load` subcommands.

    Parameters
    ----------
    cfg
        ExperimentConfig-like object with at least `output_directory`,
        `split_path`, and `bam_suffix` fields.
    bam_stem
        Optional BAM stem override. If omitted, inferred from cfg.
    """
    output_directory = Path(cfg.output_directory)
    load_directory = output_directory / LOAD_DIR
    informatics_outputs_directory = Path(
        getattr(cfg, "informatics_outputs_path", output_directory / RAW_DIR)
    )
    bam_outputs_directory = Path(
        getattr(cfg, "bam_outputs_path", informatics_outputs_directory / BAM_OUTPUTS_DIR)
    )
    fasta_outputs_directory = Path(
        getattr(cfg, "fasta_outputs_path", informatics_outputs_directory / FASTA_OUTPUTS_DIR)
    )
    bed_outputs_directory = Path(
        getattr(cfg, "bed_outputs_path", informatics_outputs_directory / BED_OUTPUTS_DIR)
    )
    modkit_outputs_directory = Path(
        getattr(cfg, "modkit_outputs_path", informatics_outputs_directory / MODKIT_OUTPUTS_DIR)
    )
    split_directory = Path(getattr(cfg, "split_path", bam_outputs_directory / SPLIT_DIR))
    bam_qc_directory = bam_outputs_directory / "bam_qc"
    mod_tsv_directory = modkit_outputs_directory / "mod_tsvs"
    mod_bed_directory = modkit_outputs_directory / "mod_beds"
    sidecar_manifest = output_directory / RAW_DIR / "sidecar_manifest.json"

    bam_suffix = str(getattr(cfg, "bam_suffix", ".bam") or ".bam")
    if not bam_suffix.startswith("."):
        bam_suffix = f".{bam_suffix}"

    stem = bam_stem or _derive_load_bam_stem(cfg, load_directory)
    unaligned_bam = bam_outputs_directory / f"{stem}{bam_suffix}"
    aligned_bam = bam_outputs_directory / f"{stem}_aligned{bam_suffix}"
    aligned_sorted_bam = bam_outputs_directory / f"{stem}_aligned_sorted{bam_suffix}"
    aligned_sorted_bai = Path(f"{aligned_sorted_bam}.bai")

    return ArtifactPaths(
        output_directory=output_directory,
        load_directory=load_directory,
        bam_outputs_directory=bam_outputs_directory,
        fasta_outputs_directory=fasta_outputs_directory,
        bed_outputs_directory=bed_outputs_directory,
        modkit_outputs_directory=modkit_outputs_directory,
        split_directory=split_directory,
        bam_qc_directory=bam_qc_directory,
        mod_tsv_directory=mod_tsv_directory,
        mod_bed_directory=mod_bed_directory,
        sidecar_manifest=sidecar_manifest,
        raw_directory=output_directory / RAW_DIR,
        spine=load_directory / "spine.h5ad",
        dense_store=load_directory / "store",
        dense_catalog=load_directory / "catalog.parquet",
        unaligned_bam=unaligned_bam,
        aligned_bam=aligned_bam,
        aligned_sorted_bam=aligned_sorted_bam,
        aligned_sorted_bai=aligned_sorted_bai,
        barcode_sidecar=aligned_sorted_bam.with_suffix(".barcode_tags.parquet"),
        barcode_positional_sidecar=unaligned_bam.with_suffix(".barcode_tags.parquet"),
        umi_positional_sidecar=unaligned_bam.with_suffix(".umi_tags.parquet"),
        umi_oriented_sidecar=aligned_sorted_bam.with_suffix(".umi_tags.parquet"),
    )


def artifact_manifest_path(output_directory: str | Path) -> Path:
    """Return canonical artifact manifest path for load sub-steps."""
    return Path(output_directory) / LOAD_DIR / "artifacts_manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Read artifact manifest JSON (returns default scaffold if missing)."""
    p = Path(path)
    if not p.exists():
        return {
            "version": 1,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "artifacts": {},
            "steps": [],
        }
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "artifacts" not in data:
        data["artifacts"] = {}
    if "steps" not in data:
        data["steps"] = []
    if "version" not in data:
        data["version"] = 1
    return data


def write_artifact_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write the load artifact manifest atomically."""
    from ..readwrite import atomic_write_json

    p = Path(path)
    manifest = dict(manifest)
    manifest["updated_at"] = _utc_now_iso()
    if "created_at" not in manifest:
        manifest["created_at"] = manifest["updated_at"]
    return atomic_write_json(p, manifest)


def register_artifact(
    manifest: dict[str, Any],
    *,
    key: str,
    path: str | Path,
    producer_step: str,
    status: str = "ready",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upsert a single artifact entry in-memory."""
    if "artifacts" not in manifest:
        manifest["artifacts"] = {}
    manifest["artifacts"][key] = {
        "path": str(Path(path)),
        "producer_step": producer_step,
        "status": status,
        "metadata": dict(metadata or {}),
        "updated_at": _utc_now_iso(),
    }
    return manifest


def record_artifact_step(
    manifest: dict[str, Any],
    *,
    step: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a step execution record to the manifest."""
    if "steps" not in manifest:
        manifest["steps"] = []
    manifest["steps"].append(
        {
            "step": step,
            "timestamp": _utc_now_iso(),
            "inputs": list(inputs or []),
            "outputs": list(outputs or []),
            "params": dict(params or {}),
        }
    )
    return manifest


def artifact_is_ready(manifest: dict[str, Any], key: str) -> bool:
    """Return True if artifact exists in manifest and status is `ready`."""
    artifacts = manifest.get("artifacts", {})
    item = artifacts.get(key)
    return bool(item) and str(item.get("status", "")).lower() == "ready"


def load_experiment_config(config_path: str):
    """Load ExperimentConfig without invoking any pipeline stages."""
    from datetime import datetime
    from importlib import resources

    from ..config import ExperimentConfig, LoadExperimentConfig
    from ..memory_guard import activate_resource_envelope, resolve_resource_envelope

    date_str = datetime.today().strftime("%y%m%d")
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, _ = ExperimentConfig.from_var_dict(
        loader.var_dict, date_str=date_str, defaults_dir=defaults_dir
    )
    envelope_key = (
        str(Path(config_path).resolve()),
        cfg.threads,
        cfg.max_memory_percent,
        cfg.max_memory_gb,
        cfg.memory_reserve_gb,
        cfg.target_task_memory_mb,
    )
    envelope = _RESOURCE_ENVELOPE_CACHE.get(envelope_key)
    if envelope is None:
        envelope = activate_resource_envelope(resolve_resource_envelope(cfg))
        _RESOURCE_ENVELOPE_CACHE[envelope_key] = envelope
    cfg._resource_envelope = envelope
    # Every existing Python pool, external tool, and downstream library reads
    # cfg.threads, so replacing the request with the resolved ceiling applies
    # affinity/cgroup/scheduler limits without duplicating caps at each caller.
    cfg.threads = envelope.resolved_threads
    logger.info(
        "ResourceEnvelope: cpu requested=%d resolved=%d "
        "(logical=%d affinity=%s cgroup=%s scheduler=%s); memory requested=%.2f GiB "
        "resolved=%.2f GiB (total=%.2f available=%.2f reserve=%.2f); "
        "enforcement=%s active=%s",
        envelope.requested_threads,
        envelope.resolved_threads,
        envelope.logical_cpu_count,
        envelope.affinity_cpu_count,
        envelope.cgroup_cpu_count,
        envelope.scheduler_cpu_count,
        envelope.requested_memory_bytes / (1024**3),
        envelope.resolved_memory_bytes / (1024**3),
        envelope.total_memory_bytes / (1024**3),
        envelope.available_memory_bytes / (1024**3),
        envelope.memory_reserve_bytes / (1024**3),
        envelope.enforcement_mode,
        envelope.enforcement_active,
    )
    return cfg


def write_gz_h5ad(adata: ad.AnnData, path: Path) -> Path:
    if path.suffix != ".gz":
        path = path.with_name(path.name + ".gz")
    # Despite the ".gz" name (kept for compatibility with AdataPaths/stage-
    # resolution and existing on-disk files), this writes an uncompressed
    # HDF5 container, not a gzip archive. Every file this function produces
    # is read back by the next pipeline stage -- see safe_write_h5ad's
    # docstring for why HDF5-internal gzip is the wrong default for that.
    safe_write_h5ad(adata, path, compression=None, backup=True)
    write_runtime_schema_yaml(adata, path, step_name="runtime")
    return path


_DEFAULT_PRIORITY = ("hmm", "latent", "spatial", "chimeric", "variant", "pp_dedup", "pp", "raw")


def resolve_adata_stage(
    cfg,
    paths: AdataPaths,
    min_stage: str = "raw",
) -> tuple[Path | None, str | None]:
    """Resolve which AnnData file to load.

    If ``cfg.from_adata_stage`` is set, force that stage.  Otherwise fall back
    to the standard priority order:
    hmm > latent > spatial > chimeric > variant > pp_dedup > pp > raw.

    Parameters
    ----------
    cfg : ExperimentConfig

    paths : AdataPaths

    min_stage : str, default "raw"
        The lowest stage to consider in the fallback chain.  Stages below this
        in the priority list are skipped.  For example, ``min_stage="pp"``
        excludes ``raw``.

    Returns
    -------
    (path, stage_name) or (None, None) if no file is found.
    """
    if cfg.from_adata_stage is not None:
        key = STAGE_MAP.get(cfg.from_adata_stage.lower())
        if key is None:
            raise ValueError(
                f"Unknown from_adata_stage '{cfg.from_adata_stage}'. "
                f"Valid values: {', '.join(sorted(STAGE_MAP))}"
            )
        p = getattr(paths, key)
        if p.exists():
            logger.info(f"from_adata_stage override: loading '{key}' from {p}")
            return p, key
        logger.warning(f"from_adata_stage='{cfg.from_adata_stage}' requested but {p} not found")
        return None, None

    # Default priority, truncated at min_stage
    try:
        cutoff = _DEFAULT_PRIORITY.index(min_stage) + 1
    except ValueError:
        cutoff = len(_DEFAULT_PRIORITY)
    stages = _DEFAULT_PRIORITY[:cutoff]

    for stage in stages:
        p = getattr(paths, stage)
        if p.exists():
            logger.info(f"Auto-resolved AnnData stage '{stage}' from {p}")
            return p, stage

    return None, None
