from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

_NON_SEMANTIC_STAGE_CONFIG_KEYS = {
    "bam_outputs_path",
    "bed_outputs_path",
    "device",
    "emit_log_file",
    "fasta_outputs_path",
    "hmm_device",
    "informatics_outputs_path",
    "log_level",
    "output_directory",
    "modkit_outputs_path",
    "split_path",
    "summary_file",
    "threads",
    "max_memory_gb",
    "max_memory_percent",
    "target_task_memory_mb",
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


def resolved_stage_config(cfg) -> dict[str, Any]:
    """Return semantic config values used for stage compatibility checks."""
    if hasattr(cfg, "to_dict"):
        values = dict(cfg.to_dict())
    else:
        values = dict(vars(cfg))
    return {
        key: value
        for key, value in values.items()
        if key not in _NON_SEMANTIC_STAGE_CONFIG_KEYS
        and not key.startswith("force_redo_")
        and not key.endswith("_max_workers")
    }


def stage_config_hash(cfg) -> str:
    """Hash semantic stage configuration without machine-local resource limits."""
    from ..informatics.experiment_manifest import config_hash

    return config_hash(resolved_stage_config(cfg))


def stage_lifecycle(cfg, stage: str, source_path: str | Path | None = None):
    """Create a lifecycle context for one partitioned CLI stage."""
    from ..informatics.experiment_manifest import StageLifecycle, artifact_record, config_hash

    run_root = Path(cfg.output_directory)
    input_artifact_ids = []
    if source_path is not None:
        source = artifact_record(source_path, run_root)
        input_artifact_ids.append(f"path:{config_hash(source)}")
    return StageLifecycle(
        run_root,
        str(stage),
        config_hash=stage_config_hash(cfg),
        input_artifact_ids=input_artifact_ids,
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
    lifecycle.complete(
        artifacts=artifacts,
        expected_tasks=expected_tasks,
        successful_tasks=expected_tasks,
        schema_versions=dict(schema_versions or {}),
        **dict(extra or {}),
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
) -> bool:
    """Check the compatible completion record used by partitioned CLI skips."""
    from ..informatics.experiment_manifest import stage_is_complete

    return stage_is_complete(
        cfg.output_directory,
        stage,
        config_hash=stage_config_hash(cfg),
        required_artifacts=required,
    )


def get_adata_paths(cfg) -> AdataPaths:
    """
    Central helper: given cfg, compute all standard AnnData paths.
    """
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
    raw_spine = output_directory / RAW_DIR / "spine.h5ad"
    preprocess_spine = output_directory / PREPROCESS_DIR / "spine.h5ad"
    spatial_spine = output_directory / SPATIAL_DIR / "spine.h5ad"
    hmm_spine = output_directory / HMM_DIR / "spine.h5ad"
    latent_spine = output_directory / LATENT_DIR / "spine.h5ad"

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
    from ..memory_guard import enable_aggregate_memory_cap, resolve_memory_budget_bytes

    date_str = datetime.today().strftime("%y%m%d")
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, _ = ExperimentConfig.from_var_dict(
        loader.var_dict, date_str=date_str, defaults_dir=defaults_dir
    )
    # Refine the aggregate memory cap (a no-op generic default until now, set
    # at CLI startup before any config was available) with this experiment's
    # actual max_memory_percent/max_memory_gb. Single hook point: every CLI
    # entry point loads its config through this function. No-op on non-Linux
    # platforms (see smftools.memory_guard) -- macOS enforcement instead
    # happens per-worker-pool via resolve_max_workers/start_worker_watchdog.
    enable_aggregate_memory_cap(budget_bytes=resolve_memory_budget_bytes(cfg))
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
