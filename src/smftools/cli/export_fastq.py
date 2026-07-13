"""CLI logic for exporting per-barcode FASTQ files of QC-passed reads.

Sequence and quality are read directly from the raw ragged store (no BAM
re-parsing); the QC-passed read set is resolved from the most complete
preprocessing artifact available for each experiment.
"""

from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

_QC_PASS_COLUMNS = ("passes_dedup", "passes_qc", "passes_read_qc")


def _resolve_qc_obs(paths, *, allow_unfiltered: bool):
    """Return ``(label_obs, accepted_read_ids_or_None, source_description)``.

    Tries, in order: the partitioned preprocess spine's QC columns, the legacy
    deduplicated AnnData's read set, the legacy QC-filtered (pre-dedup) AnnData's
    read set. Falls back to every read in the raw spine (unfiltered) only when
    ``allow_unfiltered`` is set.
    """
    import anndata as ad

    from ..informatics.partition_read import load_spine

    if paths.preprocess_spine and paths.preprocess_spine.exists():
        spine = load_spine(paths.preprocess_spine)
        for column in _QC_PASS_COLUMNS:
            if column in spine.obs.columns:
                accepted = set(spine.obs.index[spine.obs[column].astype(bool)].astype(str))
                return spine.obs, accepted, f"{column}@{paths.preprocess_spine}"
        logger.warning(
            "preprocess spine at %s has no QC columns; falling through", paths.preprocess_spine
        )

    for label, path_attr in (("pp_dedup", "pp_dedup"), ("pp", "pp")):
        candidate = getattr(paths, path_attr, None)
        if candidate and candidate.exists():
            backed = ad.read_h5ad(candidate, backed="r")
            try:
                obs = backed.obs.copy()
                accepted = set(backed.obs_names.astype(str))
            finally:
                backed.file.close()
            return obs, accepted, f"{label}@{candidate}"

    if not allow_unfiltered:
        raise ValueError(
            "no QC-passed read set found (preprocess spine / pp_dedup / pp not present); "
            "run `smftools experiment preprocess <config_path>` first, or pass allow_unfiltered=True"
        )
    logger.warning("no preprocessing artifacts found; exporting ALL raw reads (unfiltered)")
    from ..informatics.partition_read import load_spine as _load_spine

    raw_spine = _load_spine(paths.raw_spine)
    return raw_spine.obs, None, f"raw_spine@{paths.raw_spine} (unfiltered)"


def _resolve_group_by(label_obs, group_by: str | None, cfg) -> str:
    candidates = [
        group_by,
        str(getattr(cfg, "sample_name_col_for_plotting", "Sample")),
        "Sample",
        "Barcode",
    ]
    for candidate in candidates:
        if candidate and candidate in label_obs.columns:
            if group_by and candidate != group_by:
                logger.warning(
                    "group_by=%r not found on resolved QC obs; falling back to %r",
                    group_by,
                    candidate,
                )
            return candidate
    raise KeyError(
        f"none of {[c for c in candidates if c]} found on the resolved QC obs "
        f"(columns: {list(label_obs.columns)})"
    )


def export_fastq_for_experiment(
    config_path: str,
    outdir: str | Path,
    *,
    group_by: str | None = None,
    allow_unfiltered: bool = False,
    gzip_output: bool = True,
) -> Path:
    """Write one FASTQ per barcode of QC-passed reads for one experiment.

    Args:
        config_path: Path to the experiment config.
        outdir: Directory to write ``<barcode>.fastq[.gz]`` + a manifest CSV into.
        group_by: obs column to group reads by. Defaults to
            ``cfg.sample_name_col_for_plotting``, falling back to ``Sample`` then
            ``Barcode``.
        allow_unfiltered: If no QC-passed read set is available, write every raw
            read instead of raising.
        gzip_output: Whether to gzip-compress the FASTQ output.

    Returns:
        Path: ``outdir``.

    Raises:
        FileNotFoundError: If no raw ragged spine exists for this config (run
            ``smftools experiment raw <config_path>`` first).
        ValueError: If no QC-passed read set is available and
            ``allow_unfiltered`` is not set.
    """
    from ..informatics.fastq_export import write_fastq_manifest, write_fastq_per_barcode
    from ..informatics.partition_read import load_spine
    from .helpers import get_adata_paths, load_experiment_config

    cfg = load_experiment_config(config_path)
    paths = get_adata_paths(cfg)

    if not paths.raw_spine or not paths.raw_spine.exists():
        raise FileNotFoundError(
            f"no raw ragged spine found at {paths.raw_spine}; "
            f"run `smftools experiment raw {config_path}` first"
        )
    raw_spine = load_spine(paths.raw_spine)
    base_dir = paths.raw_spine.parent

    label_obs, accepted_ids, source = _resolve_qc_obs(paths, allow_unfiltered=allow_unfiltered)
    resolved_group_by = _resolve_group_by(label_obs, group_by, cfg)
    labels = label_obs[resolved_group_by].reindex(raw_spine.obs.index)

    logger.info(
        "Exporting FASTQ: %d QC-passed read(s) from %s, grouped by %r",
        len(accepted_ids) if accepted_ids is not None else raw_spine.n_obs,
        source,
        resolved_group_by,
    )

    outdir = Path(outdir)
    manifest = write_fastq_per_barcode(
        raw_spine.obs,
        base_dir,
        outdir,
        read_ids=accepted_ids,
        group_labels=labels,
        gzip_output=gzip_output,
    )
    write_fastq_manifest(outdir, manifest)
    return outdir


def export_fastq_for_project(
    project_dir: str | Path,
    outdir: str | Path,
    *,
    experiments: list[str] | None = None,
    allow_unfiltered: bool = False,
    gzip_output: bool = True,
) -> Path:
    """Write one FASTQ per ``<experiment>__<barcode>`` of QC-passed reads across a project.

    Only supports experiments that have run partitioned preprocessing (a
    ``preprocess_adata_outputs/spine.h5ad`` sibling to the registered raw spine);
    experiments without one are skipped with a warning unless ``allow_unfiltered``.
    Barcode labels are namespaced by experiment id to avoid collisions.

    Args:
        project_dir: Project directory (see ``smftools project init``).
        outdir: Directory to write FASTQs + a manifest CSV into.
        experiments: Optional explicit subset of registered experiment ids.
        allow_unfiltered: For experiments lacking a preprocess spine, write every
            raw read instead of skipping.
        gzip_output: Whether to gzip-compress the FASTQ output.

    Returns:
        Path: ``outdir``.
    """
    from ..informatics.fastq_export import write_fastq_manifest, write_fastq_per_barcode
    from ..informatics.partition_read import load_spine
    from ..project.registry import list_experiments

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    entries = list_experiments(project_dir, active_only=True)
    if experiments is not None:
        wanted = {str(e) for e in experiments}
        entries = [entry for entry in entries if entry["id"] in wanted]

    overall_manifest: dict[str, dict[str, object]] = {}
    for entry in entries:
        exp_id = str(entry["id"])
        raw_spine_path = entry.get("spines", {}).get("raw")
        if not raw_spine_path or not Path(raw_spine_path).exists():
            logger.warning("skipping %r: raw spine not found at %s", exp_id, raw_spine_path)
            continue
        raw_spine_path = Path(raw_spine_path)

        preprocess_spine_path = entry.get("spines", {}).get("preprocess")
        if preprocess_spine_path and not Path(preprocess_spine_path).exists():
            preprocess_spine_path = None
        if preprocess_spine_path is None and not allow_unfiltered:
            logger.warning(
                "skipping %r: no preprocess spine found; run partitioned preprocessing "
                "for this experiment, or pass allow_unfiltered=True",
                exp_id,
            )
            continue

        raw_spine = load_spine(raw_spine_path)
        base_dir = raw_spine_path.parent

        if preprocess_spine_path is not None:
            spine = load_spine(Path(preprocess_spine_path))
            accepted = None
            for column in _QC_PASS_COLUMNS:
                if column in spine.obs.columns:
                    accepted = set(spine.obs.index[spine.obs[column].astype(bool)].astype(str))
                    break
            label_obs = spine.obs
        else:
            logger.warning("exporting %r unfiltered (no preprocess spine found)", exp_id)
            accepted = None
            label_obs = raw_spine.obs

        group_by = "Sample" if "Sample" in label_obs.columns else "Barcode"
        labels = label_obs[group_by].reindex(raw_spine.obs.index).astype(str)
        labels = f"{exp_id}__" + labels

        logger.info(
            "Exporting FASTQ for experiment %r: %d QC-passed read(s)",
            exp_id,
            len(accepted) if accepted is not None else raw_spine.n_obs,
        )

        manifest = write_fastq_per_barcode(
            raw_spine.obs,
            base_dir,
            outdir,
            read_ids=accepted,
            group_labels=labels,
            gzip_output=gzip_output,
        )
        overall_manifest.update(manifest)

    write_fastq_manifest(outdir, overall_manifest)
    return outdir
