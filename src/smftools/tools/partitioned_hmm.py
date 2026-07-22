"""Bounded HMM execution over partitioned preprocessing/spatial spines."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.constants import REFERENCE_STRAND
from smftools.logging_utils import get_logger

from ..cli.hmm_adata import (
    HMMTrainer,
    _feature_ranges_for_merged_layer,
    _resolve_pos_mask_for_methbase,
    build_hmm_tasks,
    build_multi_channel_union,
    build_single_channel,
    resolve_torch_device,
)
from ..cli.stage_artifacts import (
    PLOT_CATALOG_COLUMNS,
    prepare_analysis_plot_layout,
    register_plot_artifact,
)
from ..hmm.fit_plan import HMMFitPlan, HMMModelSpec, build_hmm_fit_plan
from ..hmm.HMM import mask_layers_outside_read_span, normalize_hmm_feature_sets
from ..informatics.experiment_spine import write_experiment_spine
from ..informatics.partition_read import load_spine, materialize, relative_uns_path
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..preprocessing.dispatch_plan import plan_preprocess_tasks
from ..readwrite import safe_read_zarr, safe_write_h5ad, safe_write_zarr

logger = get_logger(__name__)

HMM_SPINE_FILENAME = "spine.h5ad"
HMM_TASK_CATALOG = "task_catalog.parquet"
HMM_FIT_CATALOG = "fit_catalog.parquet"
HMM_FIT_SELECTION_CATALOG = "fit_selection.parquet"
HMM_PARTIAL_SUBDIR = "store"
HMM_MODEL_SUBDIR = "models"


def _component(value: object) -> str:
    from urllib.parse import quote

    return quote(str(value), safe="._-")


def _task_path(output_dir: Path, task) -> Path:
    return (
        output_dir
        / HMM_PARTIAL_SUBDIR
        / f"reference={_component(task.reference)}"
        / f"core={task.core_start:012d}-{task.core_end:012d}"
        / f"barcode={_component(task.barcode)}"
        / f"chunk={task.chunk_index:05d}"
    )


def _apply_merges(adata, model, prefix: str, feature_sets: dict, cfg) -> None:
    merged_suffix = str(getattr(cfg, "hmm_merged_suffix", "_merged"))
    for core_layer, distance in getattr(cfg, "hmm_merge_layer_features", []) or []:
        base_layer = f"{prefix}_{core_layer}"
        if base_layer not in adata.layers:
            continue
        merged_base = model.merge_intervals_to_new_layer(
            adata,
            base_layer,
            distance_threshold=int(distance),
            suffix=merged_suffix,
            overwrite=True,
        )
        masked_layers = [merged_base, f"{merged_base}_lengths"]
        feature_ranges = _feature_ranges_for_merged_layer(core_layer, feature_sets)
        if feature_ranges:
            masked_layers.extend(
                model.write_size_class_layers_from_binary(
                    adata,
                    merged_base,
                    out_prefix=prefix,
                    feature_ranges=feature_ranges,
                    suffix=merged_suffix,
                    overwrite=True,
                )
            )
        mask_layers_outside_read_span(
            adata,
            masked_layers,
            use_original_var_names=True,
        )


def _configured_model_specs(cfg) -> list[HMMModelSpec]:
    """Expand configured HMM definitions into deterministic model specs."""
    feature_sets = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))
    specs = []
    for task in build_hmm_tasks(cfg):
        if not any(group in feature_sets for group in task.feature_groups):
            continue
        multichannel = len(task.signals) > 1
        label = str(task.output_prefix or ("Combined" if multichannel else task.signals[0]))
        architecture = (
            "multi"
            if multichannel
            else (
                "single_distance_binned"
                if bool(getattr(cfg, "hmm_distance_aware", False))
                else "single"
            )
        )
        specs.append(
            HMMModelSpec(
                name=str(task.name),
                label=label,
                signals=tuple(map(str, task.signals)),
                feature_groups=tuple(map(str, task.feature_groups)),
                architecture=architecture,
            )
        )
    return specs


def _prepare_model_input(adata, reference: str, spec: HMMModelSpec, cfg):
    """Materialize one configured HMM signal matrix and its position mask."""
    smf_modality = str(getattr(cfg, "smf_modality", "conversion"))
    if len(spec.signals) == 1:
        signal = spec.signals[0]
        values, coordinates = build_single_channel(
            adata,
            ref=reference,
            methbase=signal,
            smf_modality=smf_modality,
            cfg=cfg,
        )
        position_mask = _resolve_pos_mask_for_methbase(adata, reference, signal)
        return values, coordinates, position_mask

    values, coordinates, used_signals = build_multi_channel_union(
        adata,
        ref=reference,
        methbases=spec.signals,
        smf_modality=smf_modality,
        cfg=cfg,
    )
    position_mask = None
    for signal in used_signals:
        signal_mask = _resolve_pos_mask_for_methbase(adata, reference, signal)
        position_mask = signal_mask if position_mask is None else position_mask | signal_mask
    return values, coordinates, position_mask


def _annotate_task(
    adata,
    task,
    cfg,
    models_dir: Path,
    model_assignments: dict[str, dict[str, object]],
) -> list[str]:
    """Apply only pre-fitted immutable HMM artifacts to one materialization."""
    trainer = HMMTrainer(cfg=cfg, models_dir=models_dir)
    # hmm_device (default "cpu") overrides the general device setting for HMM
    # specifically -- see its definition in config/experiment_config.py for why.
    device = resolve_torch_device(
        getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto")
    )
    feature_sets_all = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))
    probability_threshold = float(getattr(cfg, "hmm_feature_prob_threshold", 0.5))
    decode = str(getattr(cfg, "hmm_decode", "marginal"))
    write_posterior = bool(getattr(cfg, "hmm_write_posterior", True))
    posterior_state = getattr(cfg, "hmm_posterior_state", "Modified")
    force_apply = bool(getattr(cfg, "force_redo_hmm_apply", False))
    uns_key = "hmm_appended_layers"
    adata.uns[uns_key] = []
    adata.uns["hmm_model_artifacts"] = []

    def record_model_artifact(layers: list[str]) -> None:
        if trainer.last_artifact is None:
            return
        artifact = dict(trainer.last_artifact)
        artifact["checkpoint"] = (Path(HMM_MODEL_SUBDIR) / str(artifact["checkpoint"])).as_posix()
        artifact["metadata"] = (Path(HMM_MODEL_SUBDIR) / str(artifact["metadata"])).as_posix()
        artifact["layers"] = list(layers)
        adata.uns["hmm_model_artifacts"].append(artifact)

    for spec in _configured_model_specs(cfg):
        feature_sets = {
            name: feature_sets_all[name] for name in spec.feature_groups if name in feature_sets_all
        }
        try:
            values, coordinates, position_mask = _prepare_model_input(
                adata, str(task.reference), spec, cfg
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping HMM signal %s for %s: %s", spec.label, task.task_id, exc)
            continue

        assignment = model_assignments.get(spec.label)
        if assignment is None:
            raise RuntimeError(
                f"HMM apply task {task.task_id} has no planned model for {spec.label!r}"
            )
        if bool(assignment.get("skipped", False)):
            logger.warning(
                "Skipping HMM signal %s for %s because its fit was skipped: %s",
                spec.label,
                task.task_id,
                assignment.get("reason", "unknown fit error"),
            )
            continue

        layers_before = set(adata.uns.get(uns_key, []))
        model = trainer.load_artifact(assignment, device=device)
        prefix = spec.label
        if bool(getattr(cfg, "bypass_hmm_apply", False)):
            record_model_artifact([])
            continue
        model.annotate_adata(
            adata,
            prefix=prefix,
            X=values,
            coords=coordinates,
            var_mask=position_mask,
            span_fill=True,
            config=cfg,
            decode=decode,
            write_posterior=write_posterior,
            posterior_state=posterior_state,
            feature_sets=feature_sets,
            prob_threshold=probability_threshold,
            uns_key=uns_key,
            uns_flag=f"hmm_annotated_{prefix}",
            force_redo=force_apply,
            # Read-span masking (NaN outside each read's own alignment span,
            # so clustermaps grey those positions out via cmap.set_bad and
            # _plot_feature_fractions' isfinite-based counts exclude them) is
            # a per-read positional concern, independent of channel count --
            # always needed, not just for multi-channel signals.
            mask_to_read_span=True,
            mask_use_original_var_names=True,
        )
        _apply_merges(adata, model, prefix, feature_sets, cfg)
        record_model_artifact(
            [
                name
                for name in adata.uns.get(uns_key, [])
                if name not in layers_before and name in adata.layers
            ]
        )
    return [name for name in adata.uns.get(uns_key, []) if name in adata.layers]


def execute_hmm_fit_task(
    spine_path,
    plan: HMMFitPlan,
    cfg,
    models_dir,
    parent_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fit exactly one planned immutable model before apply workers start."""
    adata = materialize(
        spine_path,
        references=plan.reference,
        read_ids=plan.selected_read_ids,
        start=plan.load_start,
        end=plan.load_end,
    )
    for column in (REFERENCE_STRAND, "Barcode", "Sample"):
        if column in adata.obs and isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
            adata.obs[column] = adata.obs[column].cat.remove_unused_categories()
    try:
        values, coordinates, _ = _prepare_model_input(adata, plan.reference, plan.model_spec, cfg)
    except (KeyError, ValueError) as exc:
        return {
            "fit_id": plan.fit_id,
            "skipped": True,
            "reason": str(exc),
        }

    trainer = HMMTrainer(cfg=cfg, models_dir=Path(models_dir))
    device = resolve_torch_device(
        getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto")
    )
    if plan.fit_kind == "ADAPT":
        if parent_artifact is None or bool(parent_artifact.get("skipped", False)):
            return {
                "fit_id": plan.fit_id,
                "skipped": True,
                "reason": "shared parent HMM fit was unavailable",
            }
        base_model = trainer.load_artifact(parent_artifact, device=device)
        trainer.adapt_or_load(
            base_model=base_model,
            sample=plan.barcode,
            reference=plan.reference,
            label=plan.model_spec.label,
            arch=plan.model_spec.architecture,
            X=values,
            coords=coordinates,
            device=device,
            core_start=plan.core_start,
            core_end=plan.core_end,
            training_selection=plan.training_metadata(),
        )
    else:
        trainer.fit_or_load(
            sample=plan.barcode,
            ref=plan.reference,
            label=plan.model_spec.label,
            arch=plan.model_spec.architecture,
            X=values,
            coords=coordinates,
            device=device,
            reference=plan.reference,
            core_start=plan.core_start,
            core_end=plan.core_end,
            training_selection=plan.training_metadata(),
            fit_kind=plan.fit_kind,
        )
    artifact = dict(trainer.last_artifact or {})
    artifact.update(
        {
            "fit_id": plan.fit_id,
            "parent_fit_id": plan.parent_fit_id,
            "skipped": False,
        }
    )
    return artifact


def execute_hmm_task(
    spine_path,
    task,
    cfg,
    output_dir,
    models_dir,
    model_assignments: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Materialize, annotate, core-crop, and persist one HMM task.

    Layers are streamed to disk (``incremental_zarr.append_zarr_layer``) one
    at a time and freed from ``adata.layers`` immediately after, instead of
    being collected into a second, core-cropped copy (``result``'s ``layers=``
    dict) that used to sit in memory alongside the still-live original --
    same rationale, and same pattern, as ``preprocessing.partitioned_
    executor.execute_preprocess_task``. ``model.annotate_adata`` itself still
    writes every layer for one HMM task (accessibility/cpg) onto ``adata``
    before returning, so this doesn't lower the peak during annotation, but
    it removes the double-materialization that followed it.
    """
    from ..informatics.incremental_zarr import append_zarr_layer, consolidate_zarr_store

    adata = materialize(
        spine_path,
        references=task.reference,
        read_ids=task.read_ids,
        start=task.load_start,
        end=task.load_end,
    )
    for column in (REFERENCE_STRAND, "Barcode", "Sample"):
        if column in adata.obs and isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
            adata.obs[column] = adata.obs[column].cat.remove_unused_categories()
    appended_layers = _annotate_task(
        adata,
        task,
        cfg,
        Path(models_dir),
        model_assignments,
    )
    model_artifacts = list(adata.uns.get("hmm_model_artifacts", []))
    layer_model_map: dict[str, str] = {}
    for artifact in model_artifacts:
        model_id = str(artifact["model_id"])
        for layer in artifact.get("layers", []):
            layer = str(layer)
            existing = layer_model_map.get(layer)
            if existing is not None and existing != model_id:
                raise RuntimeError(
                    f"HMM layer {layer!r} was produced by conflicting models "
                    f"{existing!r} and {model_id!r}"
                )
            layer_model_map[layer] = model_id
    if model_artifacts:
        unmapped = sorted(set(appended_layers).difference(layer_model_map))
        if unmapped:
            raise RuntimeError(f"HMM layers lack immutable model provenance: {unmapped}")
    positions = np.asarray(adata.var_names, dtype=np.int64)
    core_mask = (positions >= task.core_start) & (positions < task.core_end)
    core_obs = adata.obs.copy()
    core_var = adata.var.loc[core_mask].copy()
    n_positions = int(core_mask.sum())

    import anndata as ad

    skeleton = ad.AnnData(obs=core_obs, var=core_var)
    skeleton.uns.update(
        {
            "task_id": task.task_id,
            "reference": task.reference,
            "barcode": task.barcode,
            "core_start": task.core_start,
            "core_end": task.core_end,
            "load_start": task.load_start,
            "load_end": task.load_end,
            "hmm_appended_layers": appended_layers,
            "hmm_model_artifacts_json": json.dumps(
                model_artifacts, sort_keys=True, separators=(",", ":")
            ),
            "hmm_layer_model_map_json": json.dumps(
                layer_model_map, sort_keys=True, separators=(",", ":")
            ),
        }
    )
    path = _task_path(Path(output_dir), task)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_zarr(skeleton, path, backup=False, verbose=False, zarr_format=3)

    for index, name in enumerate(appended_layers):
        cropped = np.asarray(adata.layers[name])[:, core_mask]
        del adata.layers[name]
        append_zarr_layer(path, name, cropped, consolidate=(index == len(appended_layers) - 1))
        del cropped
    if not appended_layers:
        consolidate_zarr_store(path)

    return {
        **task.to_dict(include_read_ids=False),
        "group_path": path.relative_to(output_dir).as_posix(),
        "n_positions": n_positions,
        "layers": appended_layers,
        "hmm_model_ids": [str(item["model_id"]) for item in model_artifacts],
        "hmm_model_checksums": [
            str(item.get("model_checksum", item["checkpoint_sha256"])) for item in model_artifacts
        ],
        "hmm_model_artifact_refs": [str(item["checkpoint"]) for item in model_artifacts],
        "hmm_model_artifacts_json": json.dumps(
            model_artifacts, sort_keys=True, separators=(",", ":")
        ),
        "hmm_layer_model_map_json": json.dumps(
            layer_model_map, sort_keys=True, separators=(",", ":")
        ),
    }


def _plot_feature_fractions(records, output_dir: Path, layout) -> None:
    """Plot barcode-level HMM feature fractions for each bounded reference core."""
    import matplotlib.pyplot as plt

    rows = []
    for record in records:
        result, _ = safe_read_zarr(output_dir / record["group_path"], verbose=False)
        for layer in record.get("layers", []):
            if any(token in str(layer) for token in ("_lengths", "_states", "_posterior")):
                continue
            values = np.asarray(result.layers[layer], dtype=float)
            valid = np.isfinite(values)
            rows.append(
                {
                    "reference": record["reference"],
                    "core_start": record["core_start"],
                    "core_end": record["core_end"],
                    "barcode": record["barcode"],
                    "layer": str(layer),
                    "modified": float(np.nansum(values > 0)),
                    "valid": int(valid.sum()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    keys = ["reference", "core_start", "core_end"]
    for (reference, core_start, core_end), window in frame.groupby(keys, sort=True, observed=True):
        reduced = (
            window.groupby(["barcode", "layer"], observed=True)[["modified", "valid"]]
            .sum()
            .reset_index()
        )
        reduced["fraction"] = reduced["modified"].div(reduced["valid"].replace(0, np.nan))
        pivot = reduced.pivot(index="barcode", columns="layer", values="fraction")
        figure, axis = plt.subplots(
            figsize=(max(8, 0.35 * len(pivot.columns)), max(5, 0.22 * len(pivot)))
        )
        image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        axis.set_xticks(
            range(len(pivot.columns)), pivot.columns.astype(str), rotation=90, fontsize=6
        )
        axis.set_yticks(range(len(pivot)), pivot.index.astype(str), fontsize=6)
        axis.set(title=f"{reference}:{core_start}-{core_end}", xlabel="HMM layer", ylabel="Barcode")
        figure.colorbar(image, ax=axis, label="Positive fraction", shrink=0.8)
        figure.tight_layout()
        path = layout.categories["features"] / (
            f"{_component(reference)}__{core_start}_{core_end}__feature_fraction.png"
        )
        figure.savefig(path, dpi=160)
        plt.close(figure)
        register_plot_artifact(
            layout,
            path,
            stage="hmm",
            category="features",
            plot_type="barcode_hmm_feature_fraction",
            reference=str(reference),
            core_start=int(core_start),
            core_end=int(core_end),
        )


def _feature_run_lengths(row: np.ndarray) -> np.ndarray:
    """Column-count lengths of each contiguous positive run in one row.

    NaN (masked outside a read's own alignment span -- see
    HMM.mask_layers_outside_read_span) and 0 both count as "not a feature".
    """
    valid = np.nan_to_num(row, nan=0.0) > 0.5
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.array([], dtype=int)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return ends - starts


def _plot_feature_count_size_histograms(records, output_dir: Path, layout) -> None:
    """Per-barcode histograms of feature count-per-read and feature size-per-
    read, for every footprint/accessible feature layer -- both the
    HMM-decoded (unmerged) and post-merge (``_apply_merges``) variants.

    "Feature count" is the number of discrete feature runs (e.g. accessible
    patches, bound stretches) found on one read; "feature size" is each
    individual run's length in grid positions, pooled across reads. Distinct
    from ``_plot_feature_fractions``, which reports the fraction of
    *positions* flagged, not counts/sizes of discrete feature instances.
    """
    import matplotlib.pyplot as plt

    def _is_group_feature_layer(name: str) -> bool:
        if name.endswith(("_lengths", "_states")) or "_posterior_" in name:
            return False
        return "_all_footprint_features" in name or "_all_accessible_features" in name

    rows = []
    for record in records:
        layer_names = [name for name in record.get("layers", []) if _is_group_feature_layer(name)]
        if not layer_names:
            continue
        result, _ = safe_read_zarr(output_dir / record["group_path"], verbose=False)
        for layer in layer_names:
            arr = np.asarray(result.layers[layer], dtype=float)
            for read_row in arr:
                sizes = _feature_run_lengths(read_row)
                rows.append(
                    {
                        "reference": record["reference"],
                        "core_start": record["core_start"],
                        "core_end": record["core_end"],
                        "barcode": str(record["barcode"]),
                        "layer": str(layer),
                        "n_features": int(sizes.size),
                        "sizes": sizes,
                    }
                )
    if not rows:
        return

    frame = pd.DataFrame(rows)
    keys = ["reference", "core_start", "core_end", "layer"]
    metrics = (
        (
            "count",
            lambda sub: sub["n_features"].to_numpy(),
            "Features per read",
            "hmm_feature_count_histogram",
        ),
        (
            "size",
            lambda sub: np.concatenate(sub["sizes"].to_numpy()) if len(sub) else np.array([]),
            "Feature size (positions)",
            "hmm_feature_size_histogram",
        ),
    )
    for (reference, core_start, core_end, layer), window in frame.groupby(keys, sort=True):
        barcodes = sorted(window["barcode"].unique())
        n_cols = min(4, len(barcodes))
        n_rows = -(-len(barcodes) // n_cols)  # ceil division

        for metric, values_fn, xlabel, plot_type in metrics:
            figure, axes = plt.subplots(
                n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), squeeze=False
            )
            for i, barcode in enumerate(barcodes):
                axis = axes[i // n_cols][i % n_cols]
                values = values_fn(window.loc[window["barcode"] == barcode])
                if values.size:
                    if metric == "count":
                        lo, hi = int(values.min()), int(values.max())
                        bins = np.arange(lo, hi + 2) - 0.5
                    else:
                        bins = 20
                    axis.hist(values, bins=bins, color="tab:blue")
                axis.set_title(str(barcode), fontsize=8)
                axis.tick_params(labelsize=6)
            for i in range(len(barcodes), n_rows * n_cols):
                axes[i // n_cols][i % n_cols].set_visible(False)
            figure.suptitle(f"{reference}:{core_start}-{core_end} [{layer}]", fontsize=10)
            figure.supxlabel(xlabel, fontsize=8)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            path = layout.categories["features"] / (
                f"{_component(reference)}__{core_start}_{core_end}__{_component(layer)}"
                f"__feature_{metric}_hist.png"
            )
            figure.savefig(path, dpi=160)
            plt.close(figure)
            register_plot_artifact(
                layout,
                path,
                stage="hmm",
                category="features",
                plot_type=plot_type,
                reference=str(reference),
                core_start=int(core_start),
                core_end=int(core_end),
            )


def _grouped_bar_plot(axis, group_labels, series: dict, *, ylabel: str, title: str) -> None:
    """One bar per (group, series-key) pair, grouped by ``group_labels`` on the
    x-axis and colored/offset by ``series`` key (here, barcode)."""
    keys = sorted(series)
    n_groups = len(group_labels)
    n_series = len(keys)
    width = 0.8 / max(1, n_series)
    x = np.arange(n_groups)
    for index, key in enumerate(keys):
        offset = (index - (n_series - 1) / 2) * width
        axis.bar(x + offset, series[key], width=width, label=str(key))
    axis.set_xticks(x, group_labels, rotation=45, ha="right", fontsize=8)
    axis.set(ylabel=ylabel, title=title, ylim=(0, 1.02))


def _plot_hmm_parameters_across_barcodes(records, models_dir: Path, cfg, layout) -> None:
    """Compare fitted HMM emission/transition parameters across barcodes, for
    each reference/window/signal-label combination that used a per-barcode
    fit.

    Diagnostic for whether ``cfg.hmm_fit_scope="per_sample"`` (the default: a
    separate model fit per barcode) is buying anything over a shared
    ``"global"`` fit -- if parameters converge to similar values across
    barcodes, that's direct evidence a global fit would work just as well, at
    a fraction of the per-task fitting cost (each per-sample fit is a fresh,
    uncached EM run -- see dev/pipeline_scaling_audit.md's HMM cost
    discussion). No-op when ``hmm_fit_scope="global"``: every barcode already
    shares the exact same fitted model in that case, so there's nothing to
    compare.
    """
    import matplotlib.pyplot as plt

    trainer = HMMTrainer(cfg=cfg, models_dir=Path(models_dir))
    scope = trainer._fit_scope()
    if scope == "global":
        return
    kind = "ADAPT" if scope == "global_then_adapt" else "PER"

    # hmm_device (default "cpu") overrides the general device setting for HMM
    # specifically -- see its definition in config/experiment_config.py for why.
    device = resolve_torch_device(
        getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto")
    )
    hmm_tasks = build_hmm_tasks(cfg)
    if not hmm_tasks:
        return
    feature_sets_all = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))

    windows: dict[tuple[str, int, int], set[str]] = {}
    for record in records:
        key = (str(record["reference"]), int(record["core_start"]), int(record["core_end"]))
        windows.setdefault(key, set()).add(str(record["barcode"]))

    for (reference, core_start, core_end), barcodes in sorted(windows.items()):
        model_reference = f"{reference}__{core_start}_{core_end}"
        for hmm_task in hmm_tasks:
            feature_sets = {
                name: feature_sets_all[name]
                for name in hmm_task.feature_groups
                if name in feature_sets_all
            }
            if not feature_sets:
                continue
            multichannel = len(hmm_task.signals) != 1
            label = str(
                hmm_task.output_prefix or ("Combined" if multichannel else hmm_task.signals[0])
            )
            arch = trainer.choose_arch(multichannel=multichannel)

            loaded = {}
            for barcode in sorted(barcodes):
                path = None
                for record in records:
                    if (
                        str(record.get("reference")) != reference
                        or str(record.get("barcode")) != barcode
                        or int(record.get("core_start", -1)) != core_start
                        or int(record.get("core_end", -1)) != core_end
                    ):
                        continue
                    raw_artifacts = record.get("hmm_model_artifacts_json", "[]")
                    artifacts = json.loads(raw_artifacts) if isinstance(raw_artifacts, str) else []
                    matching = [
                        item
                        for item in artifacts
                        if item.get("model_key", {}).get("fit_kind") == kind
                        and item.get("model_key", {}).get("label") == label
                    ]
                    if matching:
                        path = models_dir.parent / str(matching[0]["checkpoint"])
                        break
                if path is None:
                    # Backward-compatible discovery for checkpoints written before
                    # immutable model artifacts were introduced.
                    path = trainer._path(kind, barcode, model_reference, label)
                if not path.exists():
                    continue
                try:
                    loaded[barcode] = trainer._load(path, arch=arch, device=device)
                except Exception:
                    logger.warning(
                        "Could not load HMM model %s for parameter comparison plot",
                        path,
                        exc_info=True,
                    )
            if len(loaded) < 2:
                # Nothing to compare -- only one barcode fit a model for this
                # window/label, or the rest failed to load.
                continue

            n_states = int(next(iter(loaded.values())).n_states)
            state_labels = [f"State {i}" for i in range(n_states)]

            emission_by_barcode = {}
            n_channels = 1
            for barcode, model in loaded.items():
                arr = np.asarray(model.emission.detach().cpu().numpy()).reshape(n_states, -1)
                n_channels = max(n_channels, arr.shape[1])
                emission_by_barcode[barcode] = arr
            if n_channels == 1:
                emission_group_labels = state_labels
                emission_series = {
                    barcode: arr[:, 0].tolist() for barcode, arr in emission_by_barcode.items()
                }
            else:
                emission_group_labels = [
                    f"{state}-ch{channel}"
                    for state in state_labels
                    for channel in range(n_channels)
                ]
                emission_series = {
                    barcode: arr.reshape(-1).tolist()
                    for barcode, arr in emission_by_barcode.items()
                }

            trans_group_labels = [f"{i}→{j}" for i in range(n_states) for j in range(n_states)]
            trans_series = {
                barcode: np.asarray(model.trans.detach().cpu().numpy()).reshape(-1).tolist()
                for barcode, model in loaded.items()
            }

            figure, (emission_axis, trans_axis) = plt.subplots(
                1,
                2,
                figsize=(max(10, 0.6 * len(loaded) + 6), 5),
            )
            _grouped_bar_plot(
                emission_axis,
                emission_group_labels,
                emission_series,
                ylabel="P(obs=1 | state)",
                title="Emission",
            )
            _grouped_bar_plot(
                trans_axis,
                trans_group_labels,
                trans_series,
                ylabel="Transition probability",
                title="Transition",
            )
            handles, plot_labels = emission_axis.get_legend_handles_labels()
            figure.legend(
                handles,
                plot_labels,
                title="Barcode",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=min(6, len(loaded)),
                fontsize=8,
            )
            figure.suptitle(f"{reference}:{core_start}-{core_end} [{label}] ({kind.lower()} fit)")
            figure.tight_layout(rect=(0, 0.1, 1, 0.96))
            path = layout.categories["emissions"] / (
                f"{_component(reference)}__{core_start}_{core_end}__{_component(label)}"
                "__hmm_params_by_barcode.png"
            )
            # bbox_inches="tight": the figure-level legend sits below the
            # tight_layout'd axes area, outside the rect reserved for them --
            # without this, savefig can clip it regardless of exact anchor
            # coordinates.
            figure.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            register_plot_artifact(
                layout,
                path,
                stage="hmm",
                category="emissions",
                plot_type="hmm_parameters_across_barcodes",
                reference=str(reference),
                core_start=int(core_start),
                core_end=int(core_end),
            )


def _plot_hmm_fit_history(models_dir: Path, layout) -> None:
    """Plot per-iteration EM log-likelihood proxy for every fitted HMM checkpoint.

    ``HMMTrainer._save`` now stores each ``fit_em`` run's ``hist`` (see
    cli.hmm_adata) on the checkpoint payload as ``fit_history``. This reads it
    straight from the saved ``.pt`` files (no model reconstruction needed) so
    convergence can be inspected without refitting -- diagnostic for whether
    ``hmm_max_iter`` (default 50) is set higher than real data needs (see
    dev/pipeline_scaling_audit.md's HMM cost discussion).
    """
    import matplotlib.pyplot as plt
    import torch

    models_dir = Path(models_dir)
    curves: dict[str, list[float]] = {}
    for path in sorted(models_dir.rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            logger.warning("Could not load HMM checkpoint %s for fit-history plot", path)
            continue
        hist = payload.get("fit_history")
        if hist:
            curves[path.stem] = [float(v) for v in hist]
    if not curves:
        return

    max_len = max(len(hist) for hist in curves.values())
    figure, axis = plt.subplots(figsize=(max(8.0, 0.12 * max_len + 4), 5))
    for name, hist in sorted(curves.items()):
        axis.plot(range(1, len(hist) + 1), hist, marker="o", markersize=3, linewidth=1, label=name)
    axis.set_xlabel("EM iteration")
    axis.set_ylabel("Log-likelihood proxy")
    axis.set_title("HMM fit convergence")
    if len(curves) <= 20:
        axis.legend(fontsize=6, ncol=2, loc="lower right")
    figure.tight_layout()
    path = layout.categories["training"] / "hmm_fit_history.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    register_plot_artifact(
        layout,
        path,
        stage="hmm",
        category="training",
        plot_type="hmm_fit_history",
    )


def _matching_hmm_layers(records, roots, *, lengths: bool = False) -> list[str]:
    """Resolve configured feature roots against layers actually written by the tasks."""
    available = {str(layer) for record in records for layer in record.get("layers", [])}
    suffixes = [f"_{root}{'_lengths' if lengths else ''}" for root in roots]
    return sorted(
        layer for layer in available if any(layer.endswith(suffix) for suffix in suffixes)
    )


def _plot_feature_clustermaps(
    records,
    output_spine: Path,
    output_dir: Path,
    layout,
    cfg,
) -> None:
    """Plot configured HMM feature and length layers one bounded core at a time."""
    from ..cli.hmm_adata import _resolve_feature_colormap, _resolve_length_feature_ranges
    from ..plotting import combined_hmm_length_clustermap, combined_hmm_raw_clustermap
    from ..plotting.plotting_utils import subsample_read_ids

    # Imported directly from the submodule, not via `from ..preprocessing
    # import reindex_references_adata`: that lazy package-level attribute
    # collides with its own submodule's name, so it's order-dependent-fragile
    # -- anything elsewhere that imports smftools.preprocessing.
    # reindex_references_adata by qualified path (e.g. a smoke test) first
    # silently shadows the function with the submodule for the rest of the
    # process, since __getattr__ is never consulted once normal attribute
    # lookup already succeeds. Importing directly from the fully-qualified
    # submodule sidesteps the package-attribute ambiguity entirely.
    from ..preprocessing.reindex_references_adata import reindex_references_adata

    output_spine = Path(output_spine)
    output_dir = Path(output_dir)

    feature_roots = list(getattr(cfg, "hmm_clustermap_feature_layers", ["all_accessible_features"]))
    length_roots = list(getattr(cfg, "hmm_clustermap_length_layers", ["all_footprint_features"]))
    feature_layers = _matching_hmm_layers(records, feature_roots)
    length_layers = _matching_hmm_layers(records, length_roots, lengths=True)
    if not feature_layers and not length_layers:
        return

    raw_layer = str(getattr(cfg, "layer_for_clustermap_plotting", "nan0_0minus1"))
    requested_layers = [raw_layer, *feature_layers, *length_layers]
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for record in records:
        key = (
            str(record["reference"]),
            int(record["core_start"]),
            int(record["core_end"]),
        )
        grouped.setdefault(key, []).append(record)

    max_reads_per_plot = getattr(cfg, "clustermap_max_reads_per_plot", 5000)
    for (reference, core_start, core_end), core_records in sorted(grouped.items()):
        # Capped per barcode, not globally: every barcode's read_ids are
        # combined here before combined_hmm_raw_clustermap/_length_clustermap
        # split back out by (reference, sample) and subsample again -- without
        # this, materialize() below loads every barcode's *full* read set for
        # the whole reference just to have most of it discarded a few lines
        # later inside those functions' own per-group subsampling. Bounds this
        # reduce-phase materialize to O(n_barcodes * max_reads_per_plot)
        # instead of O(total reference reads); see
        # dev/pipeline_scaling_audit.md, finding E.
        by_barcode: dict[str, list[str]] = {}
        for record in core_records:
            task, _ = safe_read_zarr(output_dir / str(record["group_path"]), verbose=False)
            barcode = str(record.get("barcode", ""))
            by_barcode.setdefault(barcode, []).extend(map(str, task.obs_names))
        read_ids: list[str] = []
        for barcode_read_ids in by_barcode.values():
            deduped = list(dict.fromkeys(barcode_read_ids))
            read_ids.extend(subsample_read_ids(deduped, max_reads_per_plot))
        adata = materialize(
            output_spine,
            references=reference,
            read_ids=read_ids,
            start=core_start,
            end=core_end,
            layers=requested_layers,
        )
        for column in (REFERENCE_STRAND, "Barcode"):
            if column not in adata.obs:
                continue
            if not isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
                adata.obs[column] = adata.obs[column].astype("category")
            adata.obs[column] = adata.obs[column].cat.remove_unused_categories()

        # Ported from the legacy (non-partitioned) pipeline, where this ran
        # once on the whole-experiment adata (cli/hmm_adata.py). Purely
        # additive (writes a new var column, never touches X/layers), so it's
        # safe to run per task-window materialization here -- var_names are
        # still absolute genomic coordinates within a window, same as for a
        # full reference.
        index_suffix = str(getattr(cfg, "reindexed_var_suffix", None) or "") or None
        reindexing_offsets = getattr(cfg, "reindexing_offsets", None)
        reindexing_invert = getattr(cfg, "reindexing_invert", None)
        if index_suffix and (reindexing_offsets or reindexing_invert):
            reindex_references_adata(
                adata,
                reference_col=REFERENCE_STRAND,
                offsets=reindexing_offsets,
                new_col=index_suffix,
                invert=reindexing_invert,
            )
        if index_suffix and f"{reference}_{index_suffix}" not in adata.var:
            index_suffix = None

        base_dir = (
            layout.categories["clustermaps"]
            / f"reference={_component(reference)}"
            / f"core={core_start:012d}-{core_end:012d}"
        )
        common = dict(
            sample_col="Barcode",
            reference_col=REFERENCE_STRAND,
            layer_gpc=raw_layer,
            layer_cpg=raw_layer,
            layer_c=raw_layer,
            layer_a=raw_layer,
            cmap_gpc=str(getattr(cfg, "clustermap_cmap_gpc", "coolwarm")),
            cmap_cpg=str(getattr(cfg, "clustermap_cmap_cpg", "viridis")),
            cmap_c=str(getattr(cfg, "clustermap_cmap_c", "coolwarm")),
            cmap_a=str(getattr(cfg, "clustermap_cmap_a", "coolwarm")),
            min_quality=None,
            min_length=None,
            min_mapped_length_to_reference_length_ratio=None,
            min_position_valid_fraction=None,
            demux_types=None,
            sort_by=str(getattr(cfg, "hmm_clustermap_sortby", "hmm")),
            deaminase=str(getattr(cfg, "smf_modality", "conversion")) == "deaminase",
            index_col_suffix=index_suffix,
            omit_chimeric_reads=bool(getattr(cfg, "omit_chimeric_reads", True)),
            # Matches partitioned_spatial.py's established pattern: combined_hmm_raw_
            # clustermap/combined_hmm_length_clustermap already parallelize across
            # (reference, sample) groups internally via ProcessPoolExecutor -- this was
            # previously hardcoded to 1, leaving that dispatch unused.
            n_jobs=max(1, int(getattr(cfg, "threads", 1) or 1)),
            restrict_to_read_span=bool(getattr(cfg, "hmm_clustermap_restrict_to_read_span", False)),
            max_reads_per_plot=getattr(cfg, "clustermap_max_reads_per_plot", 5000),
            cfg=cfg,
        )
        for layer in feature_layers:
            plot_dir = base_dir / "features" / _component(layer)
            combined_hmm_raw_clustermap(
                adata,
                hmm_feature_layer=layer,
                cmap_hmm=_resolve_feature_colormap(
                    layer, cfg, str(getattr(cfg, "clustermap_cmap_hmm", "coolwarm"))
                ),
                save_path=plot_dir,
                normalize_hmm=False,
                **common,
            )
            for path in sorted(plot_dir.glob("*.png")):
                register_plot_artifact(
                    layout,
                    path,
                    stage="hmm",
                    category="clustermaps",
                    plot_type="hmm_accessible_feature_clustermap",
                    reference=reference,
                    sample=path.stem.rsplit("__", 1)[-1],
                    core_start=core_start,
                    core_end=core_end,
                )
        for layer in length_layers:
            plot_dir = base_dir / "lengths" / _component(layer)
            combined_hmm_length_clustermap(
                adata,
                length_layer=layer,
                cmap_lengths=_resolve_feature_colormap(layer, cfg, "Greens"),
                length_feature_ranges=_resolve_length_feature_ranges(layer, cfg, "Greens"),
                save_path=plot_dir,
                **common,
            )
            for path in sorted(plot_dir.glob("*.png")):
                register_plot_artifact(
                    layout,
                    path,
                    stage="hmm",
                    category="clustermaps",
                    plot_type="hmm_footprint_length_clustermap",
                    reference=reference,
                    sample=path.stem.rsplit("__", 1)[-1],
                    core_start=core_start,
                    core_end=core_end,
                )


def _fit_catalog_frame(plans, artifacts: dict[str, dict[str, object]] | None = None):
    """Return a scalar, portable catalog for planned or completed HMM fits."""
    artifacts = artifacts or {}
    rows = []
    for plan in plans:
        record = plan.to_dict(include_read_ids=False)
        spec = record.pop("model_spec")
        record.update(
            {
                "model_name": spec["name"],
                "model_label": spec["label"],
                "model_architecture": spec["architecture"],
                "signals": spec["signals"],
                "feature_groups": spec["feature_groups"],
            }
        )
        artifact = artifacts.get(plan.fit_id)
        if artifact is None:
            record.update(
                {
                    "fit_state": "planned",
                    "model_id": None,
                    "model_checksum": None,
                    "checkpoint_sha256": None,
                    "artifact_ref": None,
                    "artifact_json": None,
                }
            )
        elif bool(artifact.get("skipped", False)):
            record.update(
                {
                    "fit_state": "skipped",
                    "model_id": None,
                    "model_checksum": None,
                    "checkpoint_sha256": None,
                    "artifact_ref": None,
                    "artifact_json": json.dumps(artifact, sort_keys=True, separators=(",", ":")),
                }
            )
        else:
            portable = dict(artifact)
            portable["checkpoint"] = (
                Path(HMM_MODEL_SUBDIR) / str(portable["checkpoint"])
            ).as_posix()
            portable["metadata"] = (Path(HMM_MODEL_SUBDIR) / str(portable["metadata"])).as_posix()
            record.update(
                {
                    "fit_state": "complete",
                    "model_id": portable["model_id"],
                    "model_checksum": portable["model_checksum"],
                    "checkpoint_sha256": portable["checkpoint_sha256"],
                    "artifact_ref": portable["checkpoint"],
                    "artifact_json": json.dumps(portable, sort_keys=True, separators=(",", ":")),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _fit_selection_frame(plans) -> pd.DataFrame:
    """Return selected fit membership as one scalar row per molecule/model."""
    return pd.DataFrame(
        [
            {
                "fit_id": plan.fit_id,
                "fit_kind": plan.fit_kind,
                "reference": plan.reference,
                "barcode": plan.barcode,
                "core_start": plan.core_start,
                "core_end": plan.core_end,
                "model_label": plan.model_spec.label,
                "read_id": read_id,
                "selection_rank": rank,
                "selection_sha256": plan.selection_sha256,
            }
            for plan in plans
            for rank, read_id in enumerate(plan.selected_read_ids)
        ]
    )


def execute_partitioned_hmm(spine_path, cfg, output_dir) -> dict[str, Path]:
    """Run bounded HMM tasks and publish a linked thin spine."""
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spine = load_spine(spine_path)
    filter_mask = next(
        (column for column in ("passes_dedup", "passes_qc") if column in spine.obs),
        None,
    )
    tasks = plan_preprocess_tasks(
        spine,
        target_task_memory_mb=int(getattr(cfg, "target_task_memory_mb", 512)),
        partition_by_barcode=True,
        filter_mask=filter_mask,
    )
    if not tasks:
        raise RuntimeError("partitioned HMM has no non-empty tasks")
    models_dir = output_dir / HMM_MODEL_SUBDIR
    if bool(getattr(cfg, "force_redo_hmm_fit", False)):
        # One revision per stage invocation: every worker resolves the same new
        # immutable IDs, while a later forced invocation receives fresh IDs.
        setattr(cfg, "_hmm_fit_revision", uuid.uuid4().hex)
    from ..memory_guard import require_memory_headroom, run_tasks_parallel

    # A GPU device (MPS on Apple Silicon, confirmed via real-data testing;
    # CUDA not verified here but not assumed safe either) isn't something
    # multiple worker *processes* can safely initialize/use concurrently --
    # several processes racing to grab the same GPU context reliably crashed
    # the whole pool (BrokenProcessPool). Force sequential execution
    # whenever a non-CPU device is in play; CPU-only runs still get the
    # normal memory/thread-aware parallel dispatch.
    # hmm_device (default "cpu") overrides the general device setting for HMM
    # specifically -- see its definition in config/experiment_config.py for why.
    device = resolve_torch_device(
        getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto")
    )

    model_specs = _configured_model_specs(cfg)
    fit_planning = build_hmm_fit_plan(tasks, model_specs, cfg)
    fit_catalog_path = output_dir / HMM_FIT_CATALOG
    fit_selection_path = output_dir / HMM_FIT_SELECTION_CATALOG
    _fit_catalog_frame(fit_planning.all_plans).to_parquet(fit_catalog_path, index=False)
    _fit_selection_frame(fit_planning.all_plans).to_parquet(fit_selection_path, index=False)

    base_results = run_tasks_parallel(
        execute_hmm_fit_task,
        [(spine_path, plan, cfg, models_dir, None) for plan in fit_planning.base_plans],
        cfg=cfg,
        force_sequential=str(device) != "cpu",
        pool_label=f"hmm fit pool ({len(fit_planning.base_plans)} models, device={device})",
        per_item_memory_mb=max(
            (plan.estimated_memory_bytes for plan in fit_planning.base_plans),
            default=1,
        )
        / (1024**2),
        estimator="hmm_fit_plan_peak",
    )
    artifacts_by_fit = {
        plan.fit_id: artifact
        for plan, artifact in zip(fit_planning.base_plans, base_results, strict=True)
    }
    adaptation_results = run_tasks_parallel(
        execute_hmm_fit_task,
        [
            (
                spine_path,
                plan,
                cfg,
                models_dir,
                artifacts_by_fit.get(str(plan.parent_fit_id)),
            )
            for plan in fit_planning.adaptation_plans
        ],
        cfg=cfg,
        force_sequential=str(device) != "cpu",
        pool_label=(
            f"hmm adaptation pool ({len(fit_planning.adaptation_plans)} models, device={device})"
        ),
        per_item_memory_mb=max(
            (plan.estimated_memory_bytes for plan in fit_planning.adaptation_plans),
            default=1,
        )
        / (1024**2),
        estimator="hmm_adaptation_plan_peak",
    )
    artifacts_by_fit.update(
        {
            plan.fit_id: artifact
            for plan, artifact in zip(
                fit_planning.adaptation_plans, adaptation_results, strict=True
            )
        }
    )
    _fit_catalog_frame(fit_planning.all_plans, artifacts_by_fit).to_parquet(
        fit_catalog_path, index=False
    )

    apply_artifacts = {
        task_id: {label: artifacts_by_fit[fit_id] for label, fit_id in assignments.items()}
        for task_id, assignments in fit_planning.apply_assignments.items()
    }
    records = run_tasks_parallel(
        execute_hmm_task,
        [
            (
                spine_path,
                task,
                cfg,
                output_dir,
                models_dir,
                apply_artifacts[task.task_id],
            )
            for task in tasks
        ],
        cfg=cfg,
        force_sequential=str(device) != "cpu",
        pool_label=f"hmm task pool ({len(tasks)} tasks, device={device})",
        per_item_memory_mb=max(task.estimated_memory_bytes for task in tasks) / (1024**2),
        estimator="hmm_task_plan_peak",
    )
    catalog_path = output_dir / HMM_TASK_CATALOG
    pd.DataFrame(records).to_parquet(catalog_path, index=False)

    require_memory_headroom(
        cfg,
        operation_label="HMM plots",
        estimator="hmm_plot_peak",
    )
    layout = prepare_analysis_plot_layout(output_dir, stage="hmm", source_spine=spine_path)
    pd.DataFrame(columns=PLOT_CATALOG_COLUMNS).to_parquet(layout.catalog, index=False)
    _plot_feature_fractions(records, output_dir, layout)
    _plot_feature_count_size_histograms(records, output_dir, layout)
    _plot_hmm_parameters_across_barcodes(records, models_dir, cfg, layout)
    _plot_hmm_fit_history(models_dir, layout)

    output_spine = output_dir / HMM_SPINE_FILENAME
    hmm_spine = spine.copy()
    # Relative to the run's output_directory, not output_dir -- see
    # informatics.partition_read._run_root_from_spine_path.
    run_root = output_dir.parent
    hmm_spine.uns["hmm_source_spine"] = relative_uns_path(spine_path, run_root)
    hmm_spine.uns["hmm_catalog"] = relative_uns_path(catalog_path, run_root)
    hmm_spine.uns["hmm_store"] = relative_uns_path(output_dir / HMM_PARTIAL_SUBDIR, run_root)
    hmm_spine.uns["hmm_model_store"] = relative_uns_path(models_dir, run_root)
    hmm_spine.uns["hmm_fit_catalog"] = relative_uns_path(fit_catalog_path, run_root)
    hmm_spine.uns["hmm_fit_selection"] = relative_uns_path(fit_selection_path, run_root)
    hmm_spine.uns["hmm_filter_mask"] = filter_mask or ""
    hmm_spine.uns["hmm_layer_absent_fill"] = {
        layer: 0.0 for record in records for layer in record["layers"]
    }
    hmm_spine.uns["hmm_schema_version"] = 1
    safe_write_h5ad(hmm_spine, output_spine, backup=False, verbose=False)
    write_experiment_spine(run_root)
    _plot_feature_clustermaps(records, output_spine, output_dir, layout, cfg)

    manifest = sidecar_manifest_path(output_dir)
    register_sidecar(manifest, "hmm_spine", output_spine)
    register_sidecar(manifest, "hmm_source_spine", spine_path)
    register_sidecar(manifest, "hmm_task_catalog", catalog_path)
    register_sidecar(manifest, "hmm_store", output_dir / HMM_PARTIAL_SUBDIR)
    register_sidecar(manifest, "hmm_model_store", models_dir)
    register_sidecar(manifest, "hmm_fit_catalog", fit_catalog_path)
    register_sidecar(manifest, "hmm_fit_selection", fit_selection_path)
    register_sidecar(manifest, "hmm_plot_catalog", layout.catalog)
    logger.info("Wrote partitioned HMM stage with %d task(s)", len(tasks))
    return {
        "spine": output_spine,
        "task_catalog": catalog_path,
        "store": output_dir / HMM_PARTIAL_SUBDIR,
        "models": models_dir,
        "fit_catalog": fit_catalog_path,
        "fit_selection": fit_selection_path,
        "plots": layout.root,
        "plot_catalog": layout.catalog,
        "manifest": manifest,
    }
