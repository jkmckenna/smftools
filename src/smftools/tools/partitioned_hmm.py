"""Bounded HMM execution over partitioned preprocessing/spatial spines."""

from __future__ import annotations

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
from ..hmm.HMM import mask_layers_outside_read_span, normalize_hmm_feature_sets
from ..informatics.experiment_spine import write_experiment_spine
from ..informatics.partition_read import load_spine, materialize, relative_uns_path
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..preprocessing.dispatch_plan import plan_preprocess_tasks
from ..readwrite import safe_read_zarr, safe_write_h5ad, safe_write_zarr

logger = get_logger(__name__)

HMM_SPINE_FILENAME = "spine.h5ad"
HMM_TASK_CATALOG = "task_catalog.parquet"
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


def _annotate_task(adata, task, cfg, models_dir: Path) -> list[str]:
    """Apply configured HMM task definitions to one bounded materialization."""
    trainer = HMMTrainer(cfg=cfg, models_dir=models_dir)
    # hmm_device (default "cpu") overrides the general device setting for HMM
    # specifically -- see its definition in config/experiment_config.py for why.
    device = resolve_torch_device(getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto"))
    feature_sets_all = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))
    probability_threshold = float(getattr(cfg, "hmm_feature_prob_threshold", 0.5))
    decode = str(getattr(cfg, "hmm_decode", "marginal"))
    write_posterior = bool(getattr(cfg, "hmm_write_posterior", True))
    posterior_state = getattr(cfg, "hmm_posterior_state", "Modified")
    force_apply = bool(getattr(cfg, "force_redo_hmm_apply", False))
    smf_modality = str(getattr(cfg, "smf_modality", "conversion"))
    uns_key = "hmm_appended_layers"
    adata.uns[uns_key] = []
    model_reference = f"{task.reference}__{task.core_start}_{task.core_end}"

    for hmm_task in build_hmm_tasks(cfg):
        feature_sets = {
            name: feature_sets_all[name]
            for name in hmm_task.feature_groups
            if name in feature_sets_all
        }
        if not feature_sets:
            continue
        signals = list(hmm_task.signals)
        if len(signals) == 1:
            signal = str(signals[0])
            try:
                values, coordinates = build_single_channel(
                    adata,
                    ref=task.reference,
                    methbase=signal,
                    smf_modality=smf_modality,
                    cfg=cfg,
                )
                position_mask = _resolve_pos_mask_for_methbase(adata, task.reference, signal)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping HMM signal %s for %s: %s", signal, task.task_id, exc)
                continue
            prefix = str(hmm_task.output_prefix or signal)
            architecture = trainer.choose_arch(multichannel=False)
        else:
            try:
                values, coordinates, used_signals = build_multi_channel_union(
                    adata,
                    ref=task.reference,
                    methbases=signals,
                    smf_modality=smf_modality,
                    cfg=cfg,
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping multi-channel HMM for %s: %s", task.task_id, exc)
                continue
            position_mask = None
            for signal in used_signals:
                signal_mask = _resolve_pos_mask_for_methbase(adata, task.reference, signal)
                position_mask = (
                    signal_mask if position_mask is None else position_mask | signal_mask
                )
            prefix = str(hmm_task.output_prefix or "Combined")
            architecture = trainer.choose_arch(multichannel=True)

        model = trainer.fit_or_load(
            sample=task.barcode,
            ref=model_reference,
            label=prefix,
            arch=architecture,
            X=values,
            coords=coordinates,
            device=device,
        )
        if bool(getattr(cfg, "bypass_hmm_apply", False)):
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
    return [name for name in adata.uns.get(uns_key, []) if name in adata.layers]


def execute_hmm_task(spine_path, task, cfg, output_dir, models_dir) -> dict[str, object]:
    """Materialize, annotate, core-crop, and persist one HMM task."""
    import anndata as ad

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
    appended_layers = _annotate_task(adata, task, cfg, Path(models_dir))
    positions = np.asarray(adata.var_names, dtype=np.int64)
    core_mask = (positions >= task.core_start) & (positions < task.core_end)
    core = adata[:, core_mask]
    result = ad.AnnData(
        obs=core.obs.copy(),
        var=core.var.copy(),
        layers={name: np.asarray(core.layers[name]) for name in appended_layers},
    )
    result.uns.update(
        {
            "task_id": task.task_id,
            "reference": task.reference,
            "barcode": task.barcode,
            "core_start": task.core_start,
            "core_end": task.core_end,
            "load_start": task.load_start,
            "load_end": task.load_end,
            "hmm_appended_layers": appended_layers,
        }
    )
    path = _task_path(Path(output_dir), task)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_zarr(result, path, backup=False, verbose=False, zarr_format=3)
    return {
        **task.to_dict(include_read_ids=False),
        "group_path": path.relative_to(output_dir).as_posix(),
        "n_positions": result.n_vars,
        "layers": appended_layers,
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
        layer_names = [
            name for name in record.get("layers", []) if _is_group_feature_layer(name)
        ]
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
            lambda sub: (
                np.concatenate(sub["sizes"].to_numpy()) if len(sub) else np.array([])
            ),
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
    device = resolve_torch_device(getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto"))
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
                hmm_task.output_prefix
                or ("Combined" if multichannel else hmm_task.signals[0])
            )
            arch = trainer.choose_arch(multichannel=multichannel)

            loaded = {}
            for barcode in sorted(barcodes):
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
    for path in sorted(models_dir.glob("*.pt")):
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

    for (reference, core_start, core_end), core_records in sorted(grouped.items()):
        read_ids: list[str] = []
        for record in core_records:
            task, _ = safe_read_zarr(output_dir / str(record["group_path"]), verbose=False)
            read_ids.extend(map(str, task.obs_names))
        adata = materialize(
            output_spine,
            references=reference,
            read_ids=list(dict.fromkeys(read_ids)),
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

        index_suffix = getattr(cfg, "reindexed_var_suffix", None)
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
            restrict_to_read_span=bool(
                getattr(cfg, "hmm_clustermap_restrict_to_read_span", False)
            ),
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
    from ..memory_guard import run_tasks_parallel

    # A GPU device (MPS on Apple Silicon, confirmed via real-data testing;
    # CUDA not verified here but not assumed safe either) isn't something
    # multiple worker *processes* can safely initialize/use concurrently --
    # several processes racing to grab the same GPU context reliably crashed
    # the whole pool (BrokenProcessPool). Force sequential execution
    # whenever a non-CPU device is in play; CPU-only runs still get the
    # normal memory/thread-aware parallel dispatch.
    # hmm_device (default "cpu") overrides the general device setting for HMM
    # specifically -- see its definition in config/experiment_config.py for why.
    device = resolve_torch_device(getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto"))
    records = run_tasks_parallel(
        execute_hmm_task,
        [(spine_path, task, cfg, output_dir, models_dir) for task in tasks],
        cfg=cfg,
        force_sequential=str(device) != "cpu",
    )
    catalog_path = output_dir / HMM_TASK_CATALOG
    pd.DataFrame(records).to_parquet(catalog_path, index=False)

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
    register_sidecar(manifest, "hmm_plot_catalog", layout.catalog)
    logger.info("Wrote partitioned HMM stage with %d task(s)", len(tasks))
    return {
        "spine": output_spine,
        "task_catalog": catalog_path,
        "store": output_dir / HMM_PARTIAL_SUBDIR,
        "models": models_dir,
        "plots": layout.root,
        "plot_catalog": layout.catalog,
        "manifest": manifest,
    }
