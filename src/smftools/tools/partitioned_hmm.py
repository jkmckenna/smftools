"""Bounded HMM execution over partitioned preprocessing/spatial spines."""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from smftools.constants import (
    BARCODE,
    COVERED_BASE_MASK,
    DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT,
    HMM_COVERED_BASE_MASK,
    REFERENCE_STRAND,
    SAMPLE_COL,
)
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
    write_plot_source_manifest,
)
from ..hmm.features import FeatureSettings
from ..hmm.fit_plan import HMMFitPlan, HMMModelSpec, build_hmm_fit_plan
from ..hmm.HMM import mask_layers_outside_read_span, normalize_hmm_feature_sets
from ..informatics.partition_read import (
    load_spine,
    materialize,
    relative_uns_path,
    run_root_from_stage_dir,
)
from ..informatics.plot_region_stitching import (
    mask_unanalyzed_gaps,
    resolve_plot_region_plans,
    select_plot_reads,
)
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..preprocessing.dispatch_plan import plan_preprocess_tasks
from ..readwrite import safe_read_zarr, safe_write_h5ad, safe_write_zarr

logger = get_logger(__name__)

HMM_SPINE_FILENAME = "spine.h5ad"
HMM_TASK_CATALOG = "task_catalog.parquet"
HMM_FIT_CATALOG = "fit_catalog.parquet"
HMM_FIT_SELECTION_CATALOG = "fit_selection.parquet"
HMM_PARTIAL_SUBDIR = "store"
HMM_COVERED_LAYER = HMM_COVERED_BASE_MASK
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


#: Decode working memory, measured on three 260820 tasks (bp axis, 42 output
#: layers per position; 1.0M / 1.6M / 10.4M read-positions -> +1.82 / 1.85 /
#: 5.93 GB, 2026-10-17): ~1.4 GB per task plus ~440 bytes per read-position,
#: i.e. ~10 bytes per output layer. Set slightly above. The preprocess figure
#: (8 bytes per position) made the largest HMM task ~70x its estimate.
HMM_INPUT_BYTES_PER_POSITION = 16
HMM_BYTES_PER_LAYER_POSITION = 12
HMM_TASK_BASE_BYTES = int(1.5 * 1024**3)


def hmm_output_layers_per_position(cfg, model_specs) -> int:
    """Dense layers one decode task holds per read-position: per model, the
    states and (optionally) the posterior, and per feature group each size
    class and the aggregate with their lengths -- doubled when merged."""
    total = 0
    for spec in model_specs:
        spec_cfg = spec.config(cfg)
        feature_sets = normalize_hmm_feature_sets(getattr(spec_cfg, "hmm_feature_sets", None))
        merged = {
            str(core) for core, _ in (getattr(spec_cfg, "hmm_merge_layer_features", []) or [])
        }
        total += 1 + int(bool(getattr(spec_cfg, "hmm_write_posterior", True)))
        for group in spec.feature_groups:
            classes = (feature_sets.get(group) or {}).get("features", {}) or {}
            layers = 2 * len(classes) + 2
            total += 2 * layers if f"all_{group}_features" in merged else layers
    return total


def hmm_bytes_per_position(cfg, model_specs) -> int:
    return HMM_INPUT_BYTES_PER_POSITION + HMM_BYTES_PER_LAYER_POSITION * (
        hmm_output_layers_per_position(cfg, model_specs)
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

        def base_architecture(settings) -> str:
            if multichannel:
                return "multi"
            return (
                "single_distance_binned"
                if bool(getattr(settings, "hmm_distance_aware", False))
                else "single"
            )

        # The endogenous CpG model carries its own settings (`HPS-15`).
        task_settings = hmm_task_settings(cfg, str(task.name))
        task_overrides = tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in sorted(task_settings.items())
        )
        task_cfg = HMMModelSpec(
            name="",
            label="",
            signals=(),
            feature_groups=(),
            architecture="",
            overrides=task_overrides,
        ).config(cfg)
        architecture = base_architecture(task_cfg)
        base = HMMModelSpec(
            name=str(task.name),
            label=label,
            signals=tuple(map(str, task.signals)),
            feature_groups=tuple(map(str, task.feature_groups)),
            architecture=_context_architecture(task_cfg, architecture, multichannel),
            base_label=label,
            overrides=task_overrides,
        )
        specs.append(base)
        for variant, settings in hmm_variants(cfg).items():
            if multichannel and any(key.startswith(VARIANT_SETTINGS_PREFIX) for key in settings):
                continue  # context emissions are single-channel
            overrides = tuple(
                sorted(
                    {
                        **dict(task_overrides),
                        **{
                            key: tuple(value) if isinstance(value, list) else value
                            for key, value in settings.items()
                        },
                    }.items()
                )
            )
            variant_spec = HMMModelSpec(
                name=f"{base.name}_{variant}",
                label=f"{label}_{variant}",
                signals=base.signals,
                feature_groups=base.feature_groups,
                architecture="",
                variant=variant,
                base_label=label,
                overrides=overrides,
            )
            specs.append(
                dataclasses.replace(
                    variant_spec,
                    architecture=_context_architecture(
                        variant_spec.config(cfg),
                        base_architecture(variant_spec.config(cfg)),
                        multichannel,
                    ),
                )
            )
    only = getattr(cfg, "_hmm_only_variant", None)
    if only:  # a layer set of one plan (`HPS-07`)
        specs = [spec for spec in specs if spec.variant == only]
    return specs


VARIANT_SETTINGS_PREFIX = "hmm_context_"
# A variant is a named fitting plan (`HPS-05`): besides emissions it may set
# how the model is fitted. Keys that name layers or features stay global.
VARIANT_PLAN_SETTINGS = frozenset(
    {
        "hmm_axis",
        "hmm_emission_scope",
        "hmm_emission_groupby",
        "hmm_efficiency_groupby",
        "hmm_efficiency_covariates",
        "hmm_efficiency_prior_sd",
        "hmm_tied_rounds",
        "hmm_tied_tol",
        "hmm_tied_transition_iters",
        "hmm_reference_groups",
        "hmm_transition_scope",
        "hmm_fit_reads_per_sample",
        "hmm_fit_strategy",
        "hmm_groupby",
        "hmm_shared_scope",
        "hmm_distance_aware",
        "hmm_distance_bins",
        "hmm_init_emission_probs",
        "hmm_init_transition_probs",
        "hmm_init_start_probs",
        "hmm_max_iter",
        "hmm_tol",
        "hmm_max_fit_reads",
        "hmm_fit_selection_seed",
        "hmm_adapt_emissions",
        "hmm_adapt_startprobs",
        "hmm_emission_adapt_iters",
    }
)


def hmm_task_settings(cfg, task_name: str) -> dict:
    """A model's own fitting settings over the stage's (`HPS-15`): today the
    endogenous CpG model's ``hmm_cpg_settings``; validated as variant settings."""
    if task_name != "cpg":
        return {}
    settings = dict(getattr(cfg, "hmm_cpg_settings", None) or {})
    unknown = [
        key
        for key in settings
        if not key.startswith(VARIANT_SETTINGS_PREFIX) and key not in VARIANT_PLAN_SETTINGS
    ]
    if unknown:
        raise ValueError(
            f"hmm_cpg_settings may set {VARIANT_SETTINGS_PREFIX}* and fitting settings "
            f"({', '.join(sorted(VARIANT_PLAN_SETTINGS))}), not {unknown}"
        )
    return settings


def hmm_variants(cfg) -> dict[str, dict]:
    """Configured variants, validated: name -> settings. A variant is a named
    fitting plan (`HCE-06` emissions, `HPS-05` fitting): ``hmm_context_*``
    and :data:`VARIANT_PLAN_SETTINGS` keys."""
    variants = getattr(cfg, "hmm_variants", None) or {}
    feature_sets = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))
    # A variant's layers are <label>_<variant>_<feature>: a name that is the
    # leading word of a feature layer would make layers ambiguous.
    reserved = {"all", "merged"} | {
        str(feature).split("_")[0]
        for group in feature_sets.values()
        for feature in dict(group.get("features", {}) or {})
    }
    for name, settings in variants.items():
        if not str(name).isidentifier() or str(name).startswith("_"):
            raise ValueError(f"hmm variant name {name!r} must be a plain identifier")
        if str(name) in reserved:
            raise ValueError(f"hmm variant name {name!r} clashes with HMM feature layer names")
        unknown = [
            key
            for key in (settings or {})
            if not key.startswith(VARIANT_SETTINGS_PREFIX) and key not in VARIANT_PLAN_SETTINGS
        ]
        if unknown:
            raise ValueError(
                f"hmm variant {name!r} may set {VARIANT_SETTINGS_PREFIX}* and fitting settings "
                f"({', '.join(sorted(VARIANT_PLAN_SETTINGS))}), not {unknown}"
            )
    return dict(variants)


def _context_architecture(cfg, architecture: str, multichannel: bool) -> str:
    """The single-channel architecture, or the context model when one is configured."""
    if context_model(cfg) == "none" or multichannel:
        return architecture
    if architecture != "single":
        raise ValueError(
            "hmm_context_model needs the single-channel HMM; "
            "it does not combine with hmm_distance_aware"
        )
    return "context_single"


def context_model(cfg) -> str:
    """``hmm_context_model``: none | table | learned (`HCE-04`)."""
    mode = str(getattr(cfg, "hmm_context_model", "none") or "none").lower()
    if mode not in ("none", "table", "learned"):
        raise ValueError(f"hmm_context_model must be none, table or learned, not {mode!r}")
    return mode


def context_setup(
    adata, reference: str, cfg, *, start: int | None = None, end: int | None = None
) -> dict[str, object]:
    """Sequence contexts of one fit (`HCE-04`): ContextBernoulliHMM.set_contexts arguments.

    Codes cover every position of ``reference`` (strand from its name), read
    from the materialization's references. Learned mode starts every weight
    at 1; table mode reads the weights of the fit's group -- the
    ``hmm_context_table_group`` value its reads share (e.g. their enzyme).
    """
    from smftools.analysis.compute.site_context_bias import (
        context_kmers,
        read_weight_table,
        weights_for,
    )

    mode = context_model(cfg)
    k = int(getattr(cfg, "hmm_context_k", 3))
    codes, cpg_codes = reference_context_codes(adata, reference, cfg, start=start, end=end)
    kmers = context_kmers(k)
    if mode == "learned":
        log_weights = np.zeros(len(kmers))
    else:
        path = getattr(cfg, "hmm_context_table", None)
        if not path:
            raise ValueError("hmm_context_model 'table' needs hmm_context_table")
        column = str(getattr(cfg, "hmm_context_table_group", "enzyme"))
        if column not in adata.obs:
            raise KeyError(f"reads carry no {column!r} column to choose context weights by")
        groups = sorted(set(adata.obs[column].dropna().astype(str)))
        if len(groups) != 1:
            raise ValueError(f"one fit's reads span {column} values {groups}; expected one")
        log_weights = np.log(weights_for(read_weight_table(path), groups[0], k))
    return {
        "log_weights": log_weights,
        "position_codes": codes,
        "cpg_codes": cpg_codes,
        "position_offset": int(start or 0),
    }


def reference_context_codes(
    adata, reference: str, cfg, *, start: int | None = None, end: int | None = None
) -> tuple[np.ndarray, list[int]]:
    """Context code of every position of ``reference`` in ``[start, end)``
    (default: all of it) and the CpG codes. Windows keep a long reference --
    a chromosome in genome mode (`HPS-12`) -- from being coded whole."""
    from smftools.analysis.compute.site_context_bias import (
        context_index,
        context_kmers,
        strand_of,
    )
    from smftools.tools.site_context_bias import sequences_from_uns

    k = int(getattr(cfg, "hmm_context_k", 3))
    base, strand = strand_of(str(reference))
    sequences = sequences_from_uns(adata.uns)
    if base not in sequences:
        raise KeyError(f"no reference sequence for {base!r} to read contexts from")
    sequence = sequences[base]
    lo = 0 if start is None else max(0, int(start))
    hi = len(sequence) if end is None else min(len(sequence), int(end))
    codes, _ = context_index(sequence, strand, np.arange(lo, hi), k=k)
    kmers = context_kmers(k)
    cpg_codes = [code for code, kmer in enumerate(kmers) if k > 1 and kmer[k // 2 + 1] == "G"]
    return codes, cpg_codes


def _pooled_model_input(spine_path, plan: HMMFitPlan, cfg, variant_cfg):
    """Training input of a fit pooled across references (`HXR-03`).

    Each member's reads are materialized on its own reference and stacked
    left-aligned (NaN padding): the HMM's transitions and levels do not depend
    on position, so rows of different lengths share one fit. A context model
    gets one coordinate row per read pointing into the members' concatenated
    context codes, so each read keeps its own reference's contexts.
    """
    blocks, setups = [], []
    for member in plan.members:
        adata = materialize(
            spine_path,
            references=member.reference,
            read_ids=member.selected_read_ids,
            start=member.load_start,
            end=member.load_end,
        )
        for column in (REFERENCE_STRAND, "Barcode", "Sample"):
            if column in adata.obs and isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
                adata.obs[column] = adata.obs[column].cat.remove_unused_categories()
        values, coordinates, _ = _prepare_model_input(adata, member.reference, plan.model_spec, cfg)
        blocks.append((np.asarray(values, dtype=float), np.asarray(coordinates, dtype=np.int64)))
        if plan.model_spec.architecture == "context_single":
            setups.append(
                context_setup(
                    adata,
                    member.reference,
                    variant_cfg,
                    start=member.load_start,
                    end=member.load_end,
                )
            )
    width = max(values.shape[1] for values, _ in blocks)
    rows = sum(values.shape[0] for values, _ in blocks)
    stacked = np.full((rows, width, *blocks[0][0].shape[2:]), np.nan)
    row = 0
    for values, _ in blocks:
        stacked[row : row + values.shape[0], : values.shape[1]] = values
        row += values.shape[0]
    if not setups:
        return stacked, np.arange(width), None
    coords = np.full((rows, width), -1, dtype=np.int64)
    row, offset = 0, 0
    for (values, coordinates), setup in zip(blocks, setups, strict=True):
        # each member's codes cover its window; its reads index them at
        # (position - window start) + where the window sits in the table
        shifted = coordinates - int(setup["position_offset"]) + offset
        coords[row : row + values.shape[0], : coordinates.size] = shifted
        row += values.shape[0]
        offset += len(setup["position_codes"])
    first = setups[0]
    for setup in setups[1:]:
        if not np.array_equal(setup["log_weights"], first["log_weights"]):
            raise ValueError("pooled references resolved different context weight tables")
    pooled = {
        "log_weights": first["log_weights"],
        "position_codes": np.concatenate([setup["position_codes"] for setup in setups]),
        "cpg_codes": first["cpg_codes"],
        "position_offset": 0,
    }
    return stacked, coords, pooled


def hmm_axis(cfg) -> str:
    """``hmm_axis`` (`HPS-11`) as resolved for fitting: ``bp`` or ``sites``
    (``hmm_distance_aware`` implies ``sites``, with a warning)."""
    from ..hmm.model_artifacts import resolve_hmm_axis

    axis = resolve_hmm_axis(cfg)
    if axis == "sites" and str(getattr(cfg, "hmm_axis", "") or "").lower() == "bp":
        _warn_once("hmm_distance_aware is on: fitting on the site axis instead of bp")
    return axis


_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        logger.warning(message)


def _on_bp_grid(adata, values: np.ndarray, coordinates: np.ndarray):
    """Site calls placed on the window's full position grid, NaN elsewhere
    (`HPS-11`); the position mask then covers every position."""
    positions = np.asarray(adata.var_names, dtype=np.int64)
    column = {int(position): index for index, position in enumerate(positions)}
    columns = np.asarray([column[int(c)] for c in np.asarray(coordinates)], dtype=int)
    grid = np.full((values.shape[0], positions.size, *values.shape[2:]), np.nan)
    grid[:, columns] = values
    return grid, positions, np.ones(positions.size, dtype=bool)


def _prepare_model_input(adata, reference: str, spec: HMMModelSpec, cfg):
    """Materialize one configured HMM signal matrix and its position mask
    (on the per-bp grid under ``hmm_axis: bp``)."""
    cfg = spec.config(cfg)  # a fitting plan's own axis (`HPS-05`)
    values, coordinates, position_mask = _prepare_site_input(adata, reference, spec, cfg)
    if hmm_axis(cfg) == "bp":
        values, coordinates, position_mask = _on_bp_grid(adata, values, coordinates)
    return _mask_uncovered_model_input(adata, values, coordinates), coordinates, position_mask


def _prepare_site_input(adata, reference: str, spec: HMMModelSpec, cfg):
    """One configured HMM signal matrix on its site columns."""
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


# Feature layers summarised per molecule (`HCE-06`): the fraction of the read's
# own span in the feature, stored as `<layer>_fraction`.
FRACTION_LAYER_SUFFIXES = ("_all_accessible_features", "_all_footprint_features")


def molecule_fractions(adata, layers, core_mask) -> dict[str, np.ndarray]:
    """Per read: the fraction of its span (finite positions) inside each feature layer."""
    import warnings

    fractions = {}
    for name in layers:
        if not str(name).endswith(FRACTION_LAYER_SUFFIXES) or name not in adata.layers:
            continue
        values = np.asarray(adata.layers[name], dtype=float)[:, core_mask]
        with warnings.catch_warnings():  # a read with no positions in the core: NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            fractions[f"{name}_fraction"] = np.nanmean(
                np.where(np.isfinite(values), values > 0, np.nan), axis=1
            )
    return fractions


SITE_FRACTION_SUFFIX = "_site_modified_fraction"


SITE_FEATURE_SUFFIX = "_site_fraction"
RAW_COLOR = "#757575"


def molecule_site_fractions(adata, reference: str, core_mask, cfg, specs) -> dict[str, np.ndarray]:
    """Per read, over the model's observed sites in the core (`HCE-08`, `HCE-09`).

    ``<model>_site_modified_fraction``: modified / observed calls of the HMM
    input (one per model; variants share their input). ``<layer>_site_fraction``
    for each accessible / footprint layer of every variant: the share of the
    same observed sites inside the feature -- the HMM call at exactly the sites
    the raw fraction counts, where ``<layer>_fraction`` counts every position
    of the read's span (as the periodicity inputs ``accessible`` vs
    ``accessible_all``).
    """
    import warnings

    positions = np.asarray(adata.var_names, dtype=np.int64)
    core = positions[np.asarray(core_mask, dtype=bool)]
    inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fractions = {}
    with warnings.catch_warnings():  # a read with no observed site: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for spec in specs:
            if len(spec.signals) != 1:
                continue
            base = spec.base_label or spec.label
            if base not in inputs:
                values, coords, _ = _prepare_model_input(adata, reference, spec, cfg)
                coords = np.asarray(coords, dtype=np.int64)
                inside = np.isin(coords, core)
                inputs[base] = (
                    np.asarray(values, dtype=float)[:, inside],
                    np.searchsorted(positions, coords[inside]),
                )
            values, columns = inputs[base]
            observed = ~np.isnan(values)
            n_observed = np.where(observed.sum(axis=1) > 0, observed.sum(axis=1), np.nan)
            if not spec.variant:
                fractions[f"{spec.label}{SITE_FRACTION_SUFFIX}"] = np.nanmean(
                    np.where(observed, values >= 0.5, np.nan), axis=1
                )
            for suffix in FRACTION_LAYER_SUFFIXES:
                layer = f"{spec.label}{suffix}"
                if layer not in adata.layers:
                    continue
                state = np.nan_to_num(np.asarray(adata.layers[layer], dtype=float)[:, columns]) > 0
                fractions[f"{layer}{SITE_FEATURE_SUFFIX}"] = (observed & state).sum(
                    axis=1
                ) / n_observed
    return fractions


def _mask_uncovered_model_input(adata, values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Keep paired insert gaps missing in HMM training and inference matrices."""
    if COVERED_BASE_MASK not in adata.layers:
        return values
    positions = np.asarray(adata.var_names, dtype=np.int64)
    column_by_position = {int(position): index for index, position in enumerate(positions)}
    try:
        columns = np.asarray([column_by_position[int(position)] for position in coordinates])
    except KeyError as exc:
        raise ValueError(
            f"HMM coordinate {exc.args[0]} is absent from materialized coverage"
        ) from exc
    covered = np.asarray(adata.layers[COVERED_BASE_MASK], dtype=bool)[:, columns]
    result = np.asarray(values, dtype=float).copy()
    if result.ndim == 2:
        result[~covered] = np.nan
    else:
        result[~covered, :] = np.nan
    return result


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
        if hasattr(model, "set_contexts"):
            # Decode with this task's own contexts and the fitted weights: a fit
            # pooled across references (`HXR-04`) or tiles (`HPS-12`) was fitted
            # on other windows, and codes cover only the loaded window.
            codes, cpg_codes = reference_context_codes(
                adata, str(task.reference), cfg, start=task.load_start, end=task.load_end
            )
            model.set_contexts(
                log_weights=model.log_weights.detach().cpu().numpy(),
                position_codes=codes,
                cpg_codes=cpg_codes,
                position_offset=int(task.load_start),
            )
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
            feature_settings=FeatureSettings(
                feature_sets=feature_sets,
                posterior_state=str(posterior_state),
                prob_threshold=probability_threshold,
                merges=tuple(
                    (str(core), int(distance))
                    for core, distance in (getattr(cfg, "hmm_merge_layer_features", []) or [])
                ),
                merged_suffix=str(getattr(cfg, "hmm_merged_suffix", "_merged")),
            ),
        )
        if f"hmm_feature_settings_{prefix}" not in adata.uns:
            # Version-1 calling (Viterbi decoding, more than two states).
            _apply_merges(adata, model, prefix, feature_sets, cfg)
        record_model_artifact(
            [
                name
                for name in adata.uns.get(uns_key, [])
                if name not in layers_before and name in adata.layers
            ]
        )
    return [name for name in adata.uns.get(uns_key, []) if name in adata.layers]


def _fit_input(spine_path, plan: HMMFitPlan, cfg, variant_cfg):
    """A plan's training matrix, coordinates and context setup (None unless a
    context model). Raises KeyError / ValueError when the input is unusable."""
    if plan.members:  # pooled across references (`HXR-03`)
        return _pooled_model_input(spine_path, plan, cfg, variant_cfg)
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
    values, coordinates, _ = _prepare_model_input(adata, plan.reference, plan.model_spec, cfg)
    setup = (
        context_setup(adata, plan.reference, variant_cfg, start=plan.load_start, end=plan.load_end)
        if plan.model_spec.architecture == "context_single"
        else None
    )
    return values, coordinates, setup


def resolve_emission_scope(spine, cfg):
    """``cfg`` with ``hmm_emission_scope: auto`` made concrete (`HPS-01`):
    ``shared_efficiency`` when the sample sheet gives every sample the
    emission and efficiency columns, else ``fit_group`` with a warning. An
    explicit scope is returned unchanged (and stays strict)."""
    scope = str(getattr(cfg, "hmm_emission_scope", "fit_group") or "fit_group").lower()
    if scope != "auto":
        return cfg
    resolved = copy.copy(cfg)
    resolved.hmm_emission_scope = "shared_efficiency"
    needed = [
        *(getattr(cfg, "hmm_emission_groupby", None) or []),
        *(getattr(cfg, "hmm_efficiency_groupby", None) or []),
        *(getattr(cfg, "hmm_efficiency_covariates", None) or []),
    ]
    reason = None
    strategy = str(getattr(cfg, "hmm_fit_strategy", "per_group") or "per_group").lower()
    groupby = [str(v).lower() for v in (getattr(cfg, "hmm_groupby", None) or ["sample"])]
    if strategy != "per_group" or not ({"sample", "barcode"} & set(groupby)):
        logger.warning(
            "hmm_emission_scope auto: needs per-sample fits (hmm_fit_strategy per_group, "
            "'sample' in hmm_groupby) -- using fit_group"
        )
        resolved.hmm_emission_scope = "fit_group"
        return resolved
    try:
        attributes = hmm_sample_attributes(spine, resolved)
        barcodes = sorted(
            {str(b) for b in spine.obs[BARCODE].astype(str).unique()}
            if BARCODE in spine.obs
            else []
        )
        lacking = sorted(
            {
                column
                for barcode in barcodes
                for column in needed
                if column not in (attributes or {}).get(barcode, {})
                or pd.isna((attributes or {}).get(barcode, {}).get(column))
            }
        )
        if lacking:
            reason = f"the sample sheet lacks {lacking} for some samples"
        elif not any(
            [
                getattr(cfg, "hmm_efficiency_groupby", None),
                getattr(cfg, "hmm_efficiency_covariates", None),
            ]
        ):
            resolved.hmm_emission_scope = "shared"
    except ValueError as exc:
        reason = str(exc)
    if reason is not None:
        logger.warning(
            "hmm_emission_scope auto: %s -- every sample keeps its own emissions (fit_group); "
            "comparisons between samples then carry per-sample emission differences",
            reason,
        )
        resolved.hmm_emission_scope = "fit_group"
    return resolved


def _resolve_spec_scopes(spine, specs, cfg):
    """Specs with each one's ``auto`` emission scope made concrete (an
    override), and the sample attributes when any plan shares emissions."""
    resolved, shared_cfg = [], None
    for spec in specs:
        spec_cfg = spec.config(cfg)
        concrete = resolve_emission_scope(spine, spec_cfg)
        scope = str(getattr(concrete, "hmm_emission_scope", "fit_group") or "fit_group")
        if concrete is not spec_cfg:
            overrides = tuple(
                (key, value) for key, value in spec.overrides if key != "hmm_emission_scope"
            ) + (("hmm_emission_scope", scope),)
            spec = dataclasses.replace(spec, overrides=tuple(sorted(overrides)))
        # The sample sheet is read for the first plan whose emission groups or
        # efficiency offsets follow sheet columns; a plan sharing one emission
        # model across every sample (`HPS-15`'s CpG default) needs none.
        if scope != "fit_group" and shared_cfg is None and _sheet_columns(spec.config(cfg)):
            shared_cfg = spec.config(cfg)
        resolved.append(spec)
    attributes = hmm_sample_attributes(spine, shared_cfg) if shared_cfg is not None else None
    if attributes is None and any(
        str(getattr(spec.config(cfg), "hmm_emission_scope", "fit_group")) != "fit_group"
        for spec in resolved
    ):
        attributes = {}
    return resolved, attributes


def _sheet_columns(cfg) -> list[str]:
    """Sample-sheet columns a plan's emission groups or offsets follow."""
    from ..hmm.fit_plan import efficiency_columns_of, emission_scope

    columns = [str(c) for c in (getattr(cfg, "hmm_emission_groupby", None) or [])]
    if emission_scope(cfg) == "shared_efficiency":
        categorical, numeric = efficiency_columns_of(cfg)
        columns += categorical + numeric
    return columns


def _plan_settings_columns(frame, specs, cfg):
    """Each fit's resolved emission scope and fitting-settings hash."""
    from ..hmm.model_artifacts import hmm_fit_config_hash

    by_label = {spec.label: spec.config(cfg) for spec in specs}
    if "model_label" not in frame:
        return frame
    return frame.assign(
        emission_scope=frame["model_label"].map(
            lambda label: str(getattr(by_label[label], "hmm_emission_scope", "fit_group"))
        ),
        fit_config_hash=frame["model_label"].map(
            lambda label: hmm_fit_config_hash(by_label[label])
        ),
    )


def hmm_sample_attributes(spine, cfg) -> dict[str, dict[str, str]] | None:
    """Barcode -> sample-sheet columns, for shared emission scopes (`HPS-02`);
    None under the default scope. The sheet is ``sample_sheet_path``, matched
    to reads by ``sample_sheet_mapping_column``."""
    from ..hmm.fit_plan import emission_scope

    if emission_scope(cfg) == "fit_group":
        return None
    path = getattr(cfg, "sample_sheet_path", None)
    if not path:
        raise ValueError("a shared hmm_emission_scope needs sample_sheet_path")
    mapping = str(getattr(cfg, "sample_sheet_mapping_column", None) or SAMPLE_COL)
    sheet = pd.read_csv(path, dtype=str)
    if mapping not in sheet.columns:
        raise ValueError(f"sample sheet {path} has no {mapping!r} column")
    obs = spine.obs
    if mapping not in obs or BARCODE not in obs:
        raise ValueError(f"reads carry no {mapping!r} / {BARCODE!r} to match the sample sheet")
    columns = [BARCODE] if mapping == BARCODE else [BARCODE, mapping]
    pairs = obs[columns].astype(str).drop_duplicates()
    merged = pairs.merge(sheet, on=mapping, how="inner")
    attributes: dict[str, dict[str, str]] = {}
    for barcode, rows in merged.groupby(BARCODE):
        if len(rows) > 1:
            raise ValueError(f"barcode {barcode!r} matches {len(rows)} sample-sheet rows")
        attributes[str(barcode)] = {k: v for k, v in rows.iloc[0].items() if k != BARCODE}
    return attributes


def build_tied_member(spine_path, plan: HMMFitPlan, cfg, variant_cfg, device):
    """One emission-group member: its configured model and training reads.
    Raises KeyError / ValueError when the plan's input is unusable."""
    from ..hmm.HMM import create_hmm
    from ..hmm.tied_fit import TiedMember

    trainer = HMMTrainer(cfg=variant_cfg, models_dir=Path("."))
    values, coordinates, trainer.context_setup = _fit_input(spine_path, plan, cfg, variant_cfg)
    model = create_hmm(variant_cfg, arch=plan.model_spec.architecture, device=device)
    trainer._apply_contexts(model)
    trainer._configure_axis(model, values)
    trainer._per_bp_start(model, values)
    return TiedMember(
        model,
        values,
        coordinates,
        covariates=np.asarray(plan.efficiency_covariates, dtype=float),
        category=plan.efficiency_key or None,
    )


def execute_tied_fit_task(
    spine_path, plans, cfg, models_dir, workers: int = 1
) -> list[dict[str, object]]:
    """Fit one emission group jointly (`HPS-03`): shared emissions and
    efficiency offsets, each member's own transitions; publish one ordinary
    model per member, so apply is unchanged.

    With ``workers > 1`` (CPU) the members are built and fitted by that many
    persistent worker processes, each holding its share; only the pooled
    emission solve runs here. Results match the in-process fit."""
    from ..hmm.model_artifacts import checkpoint_path, metadata_path
    from ..hmm.tied_fit import LocalMembers, ShardedMembers, tied_em_backend

    variant_cfg = plans[0].model_spec.config(cfg)
    pooled = str(getattr(variant_cfg, "hmm_transition_scope", "fit_group")) == "pooled"
    device = resolve_torch_device(
        getattr(cfg, "hmm_device", None) or getattr(cfg, "device", "auto")
    )
    trainers, keys, inputs, artifacts = [], [], [], {}
    for plan in plans:
        trainer = HMMTrainer(cfg=variant_cfg, models_dir=Path(models_dir))
        key = trainer._key(
            kind="PER",
            sample=plan.barcode,
            reference=plan.reference,
            label=plan.model_spec.label,
            arch=plan.model_spec.architecture,
            core_start=plan.core_start,
            core_end=plan.core_end,
        )
        trainers.append(trainer)
        keys.append(key)
    published = all(
        checkpoint_path(Path(models_dir), key).exists()
        and metadata_path(checkpoint_path(Path(models_dir), key)).exists()
        for key in keys
    )
    member_plans, models = [], []
    result = None
    if not published:
        build_args = [(spine_path, plan, cfg, variant_cfg, device) for plan in plans]
        sharded = str(device) == "cpu" and int(workers) > 1 and len(plans) > 1
        if sharded:
            backend = ShardedMembers(
                build_tied_member,
                build_args,
                workers=workers,
                device=device,
                probe=functools.partial(_member_weight_masks, variant_cfg),
            )
            skipped, kept = backend.skipped, backend.indices
            masks = backend.probes.get(kept[0]) if kept else None
        else:
            built, skipped = [], {}
            for index, args in enumerate(build_args):
                try:
                    built.append(build_tied_member(*args))
                except (KeyError, ValueError) as exc:
                    skipped[index] = str(exc)
            backend = LocalMembers(built)
            kept = [i for i in range(len(plans)) if i not in skipped]
            masks = _member_weight_masks(variant_cfg, built[0]) if built else None
        try:
            for index, reason in skipped.items():
                artifacts[plans[index].fit_id] = {
                    "fit_id": plans[index].fit_id,
                    "skipped": True,
                    "reason": reason,
                }
            member_plans = [plans[i] for i in kept]
            if member_plans:
                free_w, fixed_w = masks
                bounds = getattr(variant_cfg, "hmm_context_weight_bounds", None) or (0.1, 10.0)
                result, models = tied_em_backend(
                    backend,
                    covariates=[
                        np.asarray(p.efficiency_covariates, dtype=float) for p in member_plans
                    ],
                    categories=[p.efficiency_key or None for p in member_plans],
                    free_w=free_w,
                    fixed_w=fixed_w,
                    rounds=int(getattr(cfg, "hmm_tied_rounds", 50)),
                    tol=float(getattr(cfg, "hmm_tied_tol", 1e-5)),
                    transition_iters=int(getattr(cfg, "hmm_tied_transition_iters", 3)),
                    shrinkage=float(getattr(variant_cfg, "hmm_context_shrinkage", 50.0)),
                    bound=float(np.log(float(bounds[1]))),
                    prior_sd=float(getattr(cfg, "hmm_efficiency_prior_sd", 1.0)),
                    device=device,
                    label=(
                        f"{plans[0].model_spec.label} pooled transitions "
                        f"({len(member_plans)} members, "
                        f"{len({p.emission_group for p in member_plans})} emission groups)"
                        if pooled
                        else f"{plans[0].emission_group} ({len(member_plans)} members)"
                    ),
                    emission_groups=[p.emission_group for p in member_plans],
                    # `HPS-16`: one transition model per reference (or reference
                    # group / genome fit group), shared across emission groups.
                    transition_groups=(
                        [str(p.reference) for p in member_plans] if pooled else None
                    ),
                )
        finally:
            backend.close()
    fitted = {plan.fit_id: (model, i) for i, (plan, model) in enumerate(zip(member_plans, models))}
    for plan, trainer, key in zip(plans, trainers, keys, strict=True):
        if plan.fit_id in artifacts:
            continue
        model, index = fitted.get(plan.fit_id, (None, None))

        def fit_model(model=model):
            if model is None:
                raise RuntimeError("tied HMM member was not fitted")
            return model, list(result.history)

        trainer._load_or_fit_artifact(
            key=key,
            arch=plan.model_spec.architecture,
            device=device,
            fit_model=fit_model,
            training_selection=plan.training_metadata(),
        )
        artifact = dict(trainer.last_artifact or {})
        artifact.update(
            {
                "fit_id": plan.fit_id,
                "parent_fit_id": plan.parent_fit_id,
                "skipped": False,
                "emission_group": plan.emission_group,
                "emission_group_size": len(plans),
            }
        )
        if result is not None and index is not None:
            artifact.update(
                {
                    "efficiency_offset": float(result.offsets[index]),
                    "efficiency_slopes": [
                        float(b) for b in result.groups[plan.emission_group]["beta"]
                    ],
                    "efficiency_constant_covariates": list(
                        result.groups[plan.emission_group]["constant_covariates"]
                    ),
                    "transition_scope": "pooled" if pooled else "fit_group",
                    "tied_converged": bool(result.converged),
                    "tied_rounds_run": int(result.rounds_run),
                }
            )
        artifacts[plan.fit_id] = artifact
    return [artifacts[plan.fit_id] for plan in plans]


def _member_weight_masks(variant_cfg, member) -> tuple[np.ndarray, np.ndarray]:
    return _tied_weight_masks(member.model, variant_cfg)


def _tied_weight_masks(model, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Which context weights tied EM learns, and the fixed values of the rest:
    none without a context model; a table's weights stay fixed; learned
    weights are free for the modified state, or every state under
    ``hmm_context_states: all``."""
    n_states = int(model.n_states)
    weights = getattr(model, "log_weights", None)
    if weights is None:
        return np.zeros((n_states, 0), bool), np.zeros((n_states, 0))
    weights = weights.detach().cpu().numpy()
    n_contexts = weights.shape[-1]
    if weights.ndim == 1:
        scale = model._state_scale().detach().cpu().numpy()
        fixed = scale[:, None] * weights[None, :]
    else:
        fixed = weights.copy()
    free = np.zeros((n_states, n_contexts), bool)
    if context_model(cfg) == "learned":
        if str(getattr(cfg, "hmm_context_states", "modified")) == "all":
            free[:] = True
        else:
            free[model.modified_state_index()] = True
        fixed = np.where(free, 0.0, fixed)
    return free, fixed


def execute_hmm_fit_task(
    spine_path,
    plan: HMMFitPlan,
    cfg,
    models_dir,
    parent_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fit exactly one planned immutable model before apply workers start."""
    # A variant fits under its own settings (`HCE-06`); the default under cfg.
    variant_cfg = plan.model_spec.config(cfg)
    trainer = HMMTrainer(cfg=variant_cfg, models_dir=Path(models_dir))
    try:
        values, coordinates, trainer.context_setup = _fit_input(spine_path, plan, cfg, variant_cfg)
    except (KeyError, ValueError) as exc:
        return {"fit_id": plan.fit_id, "skipped": True, "reason": str(exc)}
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
    from ..informatics.incremental_zarr import (
        append_zarr_layer,
        consolidate_zarr_store,
    )
    from ..informatics.partition_query import (
        LAYER_ENCODING_ATTR,
        PROBABILITY_U8,
        encode_probability_u8,
    )
    from ..informatics.physical_layout import portable_matrix_chunks

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
    core_obs = core_obs.assign(**molecule_fractions(adata, appended_layers, core_mask))
    try:
        core_obs = core_obs.assign(
            **molecule_site_fractions(
                adata, task.reference, core_mask, cfg, _configured_model_specs(cfg)
            )
        )
    except Exception:
        logger.exception("Raw site fractions failed for task %s", task.task_id)
    context_qc_partial = ""
    if bool(getattr(cfg, "stage_context_qc", True)):
        # Sequence-context QC of the state calls (`SCQ-02`), while the
        # decoded layers and input calls are both at hand.
        from .hmm_context_qc import tally_hmm_task, write_task_partial

        try:
            context_qc_partial = write_task_partial(
                Path(output_dir),
                task.task_id,
                tally_hmm_task(
                    adata,
                    task.reference,
                    task.barcode,
                    core_mask,
                    cfg,
                    _configured_model_specs(cfg),
                ),
            )
        except Exception:
            logger.exception("HMM context QC tally failed for task %s", task.task_id)
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
            "analysis_core_id": task.analysis_core_id,
            "analysis_region_ids": list(task.analysis_region_ids),
            "analysis_planner_version": task.analysis_planner_version,
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
    chunks = portable_matrix_chunks(skeleton.shape, np.float32)
    safe_write_zarr(skeleton, path, backup=False, verbose=False, zarr_format=3, chunks=chunks)

    # `DLS-03`: feature layers are derivable from the posterior, so a locus task
    # stores the posterior, states and coverage and not the features (genome
    # tasks, or ``hmm_store_feature_layers: true``, still store them).
    derivable = _derivable_task_layers(adata)
    store_features = bool(getattr(cfg, "hmm_store_feature_layers", False)) or bool(
        getattr(cfg, "_hmm_tiled_references", False)
    )
    stored_layers = [name for name in appended_layers if store_features or name not in derivable]
    if not store_features and derivable and COVERED_BASE_MASK in adata.layers:
        adata.layers[HMM_COVERED_LAYER] = np.asarray(
            adata.layers[COVERED_BASE_MASK], dtype=bool
        ).astype(np.uint8)
        stored_layers.append(HMM_COVERED_LAYER)
    for name in appended_layers:
        if name not in stored_layers:
            del adata.layers[name]

    # `DLS-02`: the posterior is stored as uint8 (features were called from the
    # full-precision values above); readers decode it back to float.
    encode_posterior = str(getattr(cfg, "hmm_posterior_encoding", "uint8")) == "uint8"
    for index, name in enumerate(stored_layers):
        cropped = np.asarray(adata.layers[name])[:, core_mask]
        del adata.layers[name]
        encoded = encode_posterior and "_posterior_" in name
        if encoded:
            cropped = encode_probability_u8(cropped)
        last = index == len(stored_layers) - 1
        append_zarr_layer(
            path,
            name,
            cropped,
            chunks=portable_matrix_chunks(cropped.shape, cropped.dtype),
            consolidate=last and not encoded,
        )
        if encoded:
            import zarr

            zarr.open_group(str(path), mode="r+", use_consolidated=False)["layers"][name].attrs[
                LAYER_ENCODING_ATTR
            ] = PROBABILITY_U8
            if last:
                consolidate_zarr_store(path)
        del cropped
    if not stored_layers:
        consolidate_zarr_store(path)

    from ..informatics.derived_read_index import write_derived_read_index

    read_index_path = write_derived_read_index(
        output_dir,
        stage="hmm",
        task=task,
        obs=core_obs,
        group_path=path.relative_to(output_dir).as_posix(),
        stage_schema_version=2,
        model_artifacts=model_artifacts,
    )
    return {
        **task.to_dict(include_read_ids=False),
        "group_path": path.relative_to(output_dir).as_posix(),
        "n_positions": n_positions,
        "layers": stored_layers,
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
        "read_index_path": read_index_path.relative_to(output_dir).as_posix(),
        "context_qc_partial": context_qc_partial,
        # `DLS-03`: per model label, the settings its feature layers were called
        # with (absent for version-1 calling).
        "hmm_feature_settings_json": json.dumps(
            {
                key[len("hmm_feature_settings_") :]: json.loads(value)
                for key, value in adata.uns.items()
                if key.startswith("hmm_feature_settings_")
            },
            sort_keys=True,
        ),
    }


def tiled_references(spine, tasks) -> set[str]:
    """References decoded in more than one window, or in a window short of
    the whole reference: their feature layers cannot be re-derived exactly."""
    plans = spine.uns.get("reference_plans", {}) or {}
    windows: dict[str, set[tuple[int, int]]] = {}
    for task in tasks:
        windows.setdefault(str(task.reference), set()).add(
            (int(task.load_start), int(task.load_end))
        )
    tiled = set()
    for reference, spans in windows.items():
        length = int(dict(plans.get(reference, {}) or {}).get("reference_length") or 0)
        if len(spans) > 1 or (length and next(iter(spans)) != (0, length)):
            tiled.add(reference)
    return tiled


def _derivable_task_layers(adata) -> set[str]:
    """Feature layers this task's settings can re-derive from its posterior."""
    from ..hmm.features import FeatureSettings, feature_layer_names

    names: set[str] = set()
    for key, value in adata.uns.items():
        if key.startswith("hmm_feature_settings_"):
            prefix = key[len("hmm_feature_settings_") :]
            names.update(feature_layer_names(FeatureSettings.from_dict(json.loads(value)), prefix))
    return names


def record_layers(record) -> list[str]:
    """Every layer a task record offers: stored, plus derivable (`DLS-03`)."""
    from ..hmm.features import FeatureSettings, feature_layer_names

    names = [str(layer) for layer in record.get("layers", [])]
    for prefix, value in json.loads(record.get("hmm_feature_settings_json") or "{}").items():
        names.extend(feature_layer_names(FeatureSettings.from_dict(value), prefix))
    return list(dict.fromkeys(names))


def read_task_layers(record, output_dir: Path, names):
    """A task store's rows with ``names`` as layers: stored ones read, feature
    layers that were not stored derived from its posterior (`DLS-03`)."""
    from ..hmm.features import FeatureSettings, call_hmm_features, feature_layer_names
    from ..informatics.partition_query import read_zarr_subset

    names = list(dict.fromkeys(map(str, names)))
    stored = {str(layer) for layer in record.get("layers", [])}
    settings = {
        prefix: FeatureSettings.from_dict(value)
        for prefix, value in json.loads(record.get("hmm_feature_settings_json") or "{}").items()
    }
    wanted: dict[str, list[str]] = {}
    for prefix, setting in settings.items():
        derivable = feature_layer_names(setting, prefix)
        wanted[prefix] = [n for n in names if n in derivable and n not in stored]
    load = {n for n in names if n in stored}
    for prefix, derived in wanted.items():
        if derived:
            load.add(settings[prefix].posterior_layer(prefix))
            if HMM_COVERED_LAYER in stored:
                load.add(HMM_COVERED_LAYER)
    part = read_zarr_subset(Path(output_dir) / record["group_path"], layers=load)
    if not any(wanted.values()):
        return part
    positions = np.asarray(part.var_names, dtype=np.int64)
    spans = np.column_stack(
        [
            np.asarray(part.obs["reference_start"], dtype=np.int64),
            np.asarray(part.obs["reference_end"], dtype=np.int64),
        ]
    )
    covered = (
        np.asarray(part.layers[HMM_COVERED_LAYER], dtype=bool)
        if HMM_COVERED_LAYER in part.layers
        else None
    )
    for prefix, derived in wanted.items():
        if not derived:
            continue
        layers = call_hmm_features(
            np.asarray(part.layers[settings[prefix].posterior_layer(prefix)], dtype=np.float64),
            positions,
            spans,
            prefix=prefix,
            settings=settings[prefix],
            covered=covered,
            layers=derived,
        )
        for name, values in layers.items():
            part.layers[name] = values
    for name in list(part.layers):
        if name not in names:
            del part.layers[name]
    return part


def hmm_feature_settings(records) -> dict[str, dict]:
    """Per model label, the feature settings every task used (`DLS-03`)."""
    settings: dict[str, dict] = {}
    for record in records:
        for prefix, value in json.loads(record.get("hmm_feature_settings_json") or "{}").items():
            existing = settings.setdefault(prefix, value)
            if existing != value:
                raise ValueError(f"HMM tasks called {prefix!r} features with different settings")
    return settings


def _plot_feature_fractions(records, output_dir: Path, layout) -> None:
    """Plot barcode-level HMM feature fractions for each bounded reference core."""
    import matplotlib.pyplot as plt

    rows = []
    for record in records:
        layers = [
            layer
            for layer in record_layers(record)
            if not any(
                token in str(layer)
                for token in ("_lengths", "_states", "_posterior", "_end_truncated", "covered")
            )
        ]
        if not layers:
            continue
        result = read_task_layers(record, output_dir, layers)
        for layer in layers:
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


def _plot_feature_count_size_histograms(records, output_dir: Path, layout, specs=()) -> None:
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
        if name.endswith(("_lengths", "_states", "_end_truncated")) or "_posterior_" in name:
            return False
        return "_all_footprint_features" in name or "_all_accessible_features" in name

    histograms: dict[tuple[str, int, int, str, str], dict[str, dict[int, int]]] = {}
    for record in records:
        layer_names = [name for name in record_layers(record) if _is_group_feature_layer(name)]
        if not layer_names:
            continue
        result = read_task_layers(record, output_dir, layer_names)
        for layer in layer_names:
            arr = np.asarray(result.layers[layer], dtype=float)
            key = (
                str(record["reference"]),
                int(record["core_start"]),
                int(record["core_end"]),
                str(layer),
                str(record["barcode"]),
            )
            bucket = histograms.setdefault(key, {"count": {}, "size": {}})
            for read_row in arr:
                sizes = _feature_run_lengths(read_row)
                count = int(sizes.size)
                bucket["count"][count] = bucket["count"].get(count, 0) + 1
                for size in sizes:
                    value = int(size)
                    bucket["size"][value] = bucket["size"].get(value, 0) + 1
    if not histograms:
        return
    # One figure per feature (default layer); emission variants overlay it (`HCE-06`).
    variant_of = {
        layer: (base, variant)
        for base, members in variant_layer_groups({key[3] for key in histograms}, specs)
        for variant, layer in members
    }
    windows: dict[tuple[str, int, int, str], dict[str, dict[str, dict[str, dict[int, int]]]]] = {}
    for (reference, core_start, core_end, layer, barcode), values in histograms.items():
        base, variant = variant_of.get(layer, (layer, ""))
        by_variant = windows.setdefault((reference, core_start, core_end, base), {})
        by_variant.setdefault(variant, {})[barcode] = values
    metrics = (
        ("count", "Features per read", "hmm_feature_count_histogram"),
        ("size", "Feature size (positions)", "hmm_feature_size_histogram"),
    )
    for (reference, core_start, core_end, layer), by_variant in sorted(windows.items()):
        variants = list(by_variant)  # default first (variant_layer_groups order)
        barcodes = sorted({barcode for values in by_variant.values() for barcode in values})
        n_cols = min(4, len(barcodes))
        n_rows = -(-len(barcodes) // n_cols)  # ceil division

        for metric, xlabel, plot_type in metrics:
            figure, axes = plt.subplots(
                n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), squeeze=False
            )
            for i, barcode in enumerate(barcodes):
                axis = axes[i // n_cols][i % n_cols]
                present = [
                    (variant, by_variant[variant][barcode][metric])
                    for variant in variants
                    if barcode in by_variant[variant] and by_variant[variant][barcode][metric]
                ]
                if present:
                    low = min(min(counts) for _, counts in present)
                    high = max(max(counts) for _, counts in present)
                    bins = np.arange(low, high + 2) - 0.5  # shared across variants
                    for index, (variant, counts) in enumerate(present):
                        values = np.asarray(sorted(counts), dtype=int)
                        weights = np.asarray([counts[value] for value in values], dtype=int)
                        color = VARIANT_COLORS[variants.index(variant) % len(VARIANT_COLORS)]
                        if len(variants) == 1:
                            axis.hist(values, bins=bins, weights=weights, color="tab:blue")
                            continue
                        # Translucent fill, solid outline: overlaps stay legible.
                        axis.hist(
                            values,
                            bins=bins,
                            weights=weights,
                            color=color,
                            alpha=0.3,
                            histtype="stepfilled",
                            label=_variant_label(variant),
                        )
                        axis.hist(
                            values,
                            bins=bins,
                            weights=weights,
                            color=color,
                            histtype="step",
                            linewidth=1.1,
                        )
                axis.set_title(str(barcode), fontsize=8)
                axis.tick_params(labelsize=6)
            for i in range(len(barcodes), n_rows * n_cols):
                axes[i // n_cols][i % n_cols].set_visible(False)
            if len(variants) > 1:
                from matplotlib.patches import Patch

                handles = [
                    Patch(
                        facecolor=VARIANT_COLORS[i % len(VARIANT_COLORS)],
                        edgecolor=VARIANT_COLORS[i % len(VARIANT_COLORS)],
                        alpha=0.5,
                        label=_variant_label(variant),
                    )
                    for i, variant in enumerate(variants)
                ]
                figure.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.0, 0.5),
                    fontsize=7,
                    title="HMM",
                    frameon=False,
                )
            figure.suptitle(f"{reference}:{core_start}-{core_end} [{layer}]", fontsize=10)
            figure.supxlabel(xlabel, fontsize=8)
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            path = layout.categories["features"] / (
                f"{_component(reference)}__{core_start}_{core_end}__{_component(layer)}"
                f"__feature_{metric}_hist.png"
            )
            figure.savefig(path, dpi=160, bbox_inches="tight")
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


def _read_task_obs(path: Path, columns) -> pd.DataFrame:
    """Only the obs of a task store (its layers are not read)."""
    import anndata as ad
    import zarr

    obs = ad.io.read_elem(zarr.open_group(str(path), mode="r")["obs"])
    return obs[[column for column in columns if column in obs]]


def _plot_molecule_fractions(records, output_dir: Path, layout, specs=(), obs_reader=None) -> None:
    """Per barcode: each read's fraction in each feature, emission variants side by side.

    The HMM counterpart of preprocess's per-read modification-rate plots
    (`HCE-06`): one figure per reference window, a panel per feature
    (accessible, footprint), at each barcode one violin per variant -- its
    own colour, translucent with a solid edge and median -- over the reads'
    jittered values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Panels: each feature over the read's span (``<layer>_fraction``) and at
    # the model's observed sites (``<layer>_site_fraction`` -> ``<layer>_sites``),
    # the latter for accessible beside the raw modified-site fraction (`HCE-09`).
    raw_panels = {
        f"{spec.label}{SITE_FRACTION_SUFFIX}": f"{spec.label}_all_accessible_features_sites"
        for spec in specs
        if not spec.variant and len(spec.signals) == 1
    }
    rows, raw_rows = [], []
    for record in records:
        layers = [
            str(layer)
            for layer in record_layers(record)
            if str(layer).endswith(FRACTION_LAYER_SUFFIXES)
        ]
        if not layers:
            continue
        names = {f"{layer}_fraction": layer for layer in layers}
        names.update({f"{layer}{SITE_FEATURE_SUFFIX}": f"{layer}_sites" for layer in layers})
        obs = (obs_reader or _read_task_obs)(
            output_dir / record["group_path"], [*names, *raw_panels]
        )
        key = (
            str(record["reference"]),
            int(record["core_start"]),
            int(record["core_end"]),
            str(record["barcode"]),
        )
        for name in obs.columns:
            values = pd.to_numeric(obs[name], errors="coerce").dropna().to_numpy()
            if name in raw_panels:
                raw_rows.append((*key, raw_panels[name], values))
            else:
                rows.append((*key, names[name], values))
    if not rows:
        return
    variant_of = {
        layer: (base, variant)
        for base, members in variant_layer_groups({row[4] for row in rows}, specs)
        for variant, layer in members
    }
    windows: dict = {}
    for reference, core_start, core_end, barcode, layer, values in rows:
        base, variant = variant_of.get(layer, (layer, ""))
        cell = windows.setdefault((reference, core_start, core_end), {}).setdefault(base, {})
        cell.setdefault(variant, {}).setdefault(barcode, []).append(values)
    for reference, core_start, core_end, barcode, panel, values in raw_rows:
        cell = windows.get((reference, core_start, core_end), {}).get(panel)
        if cell is not None:
            cell.setdefault("raw", {}).setdefault(barcode, []).append(values)
    from smftools.preprocessing.stage_context_qc import _short_names

    rng = np.random.default_rng(0)
    short = _short_names(
        {
            b
            for by_feature in windows.values()
            for cell in by_feature.values()
            for by_barcode in cell.values()
            for b in by_barcode
        }
    )
    for (reference, core_start, core_end), by_feature in sorted(windows.items()):
        features = sorted(by_feature)
        variants = list(dict.fromkeys(v for feature in features for v in by_feature[feature]))
        # Raw first, in grey; the HMM variants keep their colours.
        hmm_variants = [v for v in variants if v != "raw"]
        variants = (["raw"] if "raw" in variants else []) + hmm_variants
        colors = {v: VARIANT_COLORS[i % len(VARIANT_COLORS)] for i, v in enumerate(hmm_variants)}
        colors["raw"] = RAW_COLOR
        barcodes = sorted(
            {b for feature in features for v in by_feature[feature].values() for b in v}
        )
        figure, axes = plt.subplots(
            len(features),
            1,
            figsize=(max(10, 0.6 * len(barcodes) * max(1, len(variants))), 3.6 * len(features)),
            squeeze=False,
        )
        width = 0.8 / max(1, len(variants))
        for axis, feature in zip(axes[:, 0], features, strict=True):
            present = [v for v in variants if v in by_feature[feature]]
            for v_index, variant in enumerate(present):
                color = colors[variant]
                offset = (v_index - (len(present) - 1) / 2) * width
                for b_index, barcode in enumerate(barcodes, start=1):
                    parts = by_feature[feature].get(variant, {}).get(barcode)
                    if not parts:
                        continue
                    values = np.concatenate(parts)
                    position = b_index + offset
                    if values.size >= 2 and np.std(values) > 0:
                        body = axis.violinplot(
                            [values],
                            positions=[position],
                            widths=width * 0.95,
                            showmedians=True,
                            showextrema=False,
                        )
                        for patch in body["bodies"]:
                            patch.set_facecolor(color)
                            patch.set_edgecolor(color)
                            patch.set_alpha(0.35)
                            patch.set_linewidth(1.0)
                        body["cmedians"].set_color(color)
                        body["cmedians"].set_linewidth(1.6)
                    shown = (
                        values if values.size <= 2000 else rng.choice(values, 2000, replace=False)
                    )
                    axis.scatter(
                        position + rng.uniform(-width / 4, width / 4, shown.size),
                        shown,
                        s=4,
                        alpha=0.35,
                        color=color,
                        edgecolors="none",
                        rasterized=True,
                    )
            axis.set_xticks(
                range(1, len(barcodes) + 1), [short[b] for b in barcodes], rotation=90, fontsize=6
            )
            axis.set_xlim(0.4, len(barcodes) + 0.6)
            axis.set_ylim(-0.02, 1.02)
            if feature.endswith("_sites"):
                label = f"{feature.removesuffix('_sites')}\nfraction of observed sites"
            else:
                label = f"{feature}\nfraction of read span"
            axis.set_ylabel(label, fontsize=7)
            axis.grid(axis="y", alpha=0.2)
        if len(variants) > 1:
            handles = [
                Patch(
                    facecolor=colors[v],
                    alpha=0.5,
                    edgecolor=colors[v],
                    label="raw (modified sites)" if v == "raw" else _variant_label(v),
                )
                for v in variants
            ]
            figure.legend(handles=handles, loc="upper right", fontsize=8, title="HMM")
        figure.suptitle(
            f"{reference}:{core_start}-{core_end} per-molecule feature fractions", fontsize=10
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = layout.categories["features"] / (
            f"{_component(reference)}__{core_start}_{core_end}__molecule_fractions.png"
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)
        register_plot_artifact(
            layout,
            path,
            stage="hmm",
            category="features",
            plot_type="hmm_molecule_fractions",
            reference=str(reference),
            core_start=int(core_start),
            core_end=int(core_end),
        )


def _plot_hmm_vs_raw_scatter(records, output_dir: Path, layout, specs=(), obs_reader=None) -> None:
    """Per reference window and model: each read's HMM accessible fraction
    against its raw modified-site fraction, a panel per barcode, every
    variant overlaid (`HCE-08`)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = {}
    for spec in specs:
        if len(spec.signals) == 1:
            base = spec.base_label or spec.label
            models.setdefault(base, []).append(
                (_variant_label(spec.variant), f"{spec.label}_all_accessible_features_fraction")
            )
    if not models:
        return
    windows: dict = {}
    for record in records:
        wanted = [
            column
            for base, members in models.items()
            for column in [f"{base}{SITE_FRACTION_SUFFIX}", *(c for _, c in members)]
        ]
        obs = (obs_reader or _read_task_obs)(output_dir / record["group_path"], wanted)
        if obs.empty:
            continue
        key = (str(record["reference"]), int(record["core_start"]), int(record["core_end"]))
        windows.setdefault(key, {}).setdefault(str(record["barcode"]), []).append(obs)
    from smftools.preprocessing.stage_context_qc import _short_names

    rng = np.random.default_rng(0)
    short = _short_names({b for by_barcode in windows.values() for b in by_barcode})
    for (reference, core_start, core_end), by_barcode in sorted(windows.items()):
        frames = {barcode: pd.concat(parts) for barcode, parts in by_barcode.items()}
        for base, members in models.items():
            raw = f"{base}{SITE_FRACTION_SUFFIX}"
            barcodes = sorted(b for b, frame in frames.items() if raw in frame)
            if not barcodes:
                continue
            n_cols = min(4, len(barcodes))
            n_rows = -(-len(barcodes) // n_cols)
            figure, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3.0 * n_cols, 3.0 * n_rows),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for index, barcode in enumerate(barcodes):
                axis = axes[index // n_cols][index % n_cols]
                frame = frames[barcode]
                handles = []
                for v_index, (variant, column) in enumerate(members):
                    if column not in frame:
                        continue
                    pair = frame[[raw, column]].apply(pd.to_numeric, errors="coerce").dropna()
                    if pair.empty:
                        continue
                    shown = pair if len(pair) <= 2000 else pair.sample(2000, random_state=rng)
                    color = VARIANT_COLORS[v_index % len(VARIANT_COLORS)]
                    axis.scatter(
                        shown[raw],
                        shown[column],
                        s=3,
                        alpha=0.3,
                        color=color,
                        edgecolors="none",
                        rasterized=True,
                    )
                    r = pair[raw].corr(pair[column]) if len(pair) > 2 else np.nan
                    handles.append(
                        Line2D(
                            [],
                            [],
                            marker="o",
                            linestyle="",
                            color=color,
                            label=f"{variant} r={r:.2f} (n={len(pair)})",
                        )
                    )
                axis.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, linestyle="--")
                axis.set_xlim(-0.02, 1.02)
                axis.set_ylim(-0.02, 1.02)
                axis.set_title(short[barcode], fontsize=8)
                axis.tick_params(labelsize=6)
                if handles:
                    axis.legend(handles=handles, fontsize=5, loc="upper left", frameon=False)
            for index in range(len(barcodes), n_rows * n_cols):
                axes[index // n_cols][index % n_cols].set_visible(False)
            figure.supxlabel(f"raw modified-site fraction ({base} sites)", fontsize=8)
            figure.supylabel("HMM accessible fraction", fontsize=8)
            figure.suptitle(
                f"{reference}:{core_start}-{core_end} [{base}] HMM vs raw per molecule",
                fontsize=10,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            path = layout.categories["features"] / (
                f"{_component(reference)}__{core_start}_{core_end}__{_component(base)}"
                "__hmm_vs_raw_scatter.png"
            )
            figure.savefig(path, dpi=150)
            plt.close(figure)
            register_plot_artifact(
                layout,
                path,
                stage="hmm",
                category="features",
                plot_type="hmm_vs_raw_molecule_scatter",
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
    uncached EM run -- see dev/plans/audits/pipeline_scaling_audit.md's HMM cost
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
                model_arch = arch
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
                        # The artifact's own architecture: a context model is
                        # not what ``choose_arch`` would pick.
                        model_arch = str(
                            matching[0].get("model_key", {}).get("architecture") or arch
                        )
                        break
                if path is None:
                    # Backward-compatible discovery for checkpoints written before
                    # immutable model artifacts were introduced.
                    path = trainer._path(kind, barcode, model_reference, label)
                if not path.exists():
                    continue
                try:
                    loaded[barcode] = trainer._load(path, arch=model_arch, device=device)
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
    dev/plans/audits/pipeline_scaling_audit.md's HMM cost discussion).
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


VARIANT_COLORS = ("#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#C62828", "#00838F")


def variant_layer_groups(layers, specs) -> list[tuple[str, list[tuple[str, str]]]]:
    """Layers grouped by the default layer they correspond to (`HCE-06`).

    ``[(base layer, [(variant, layer), ...]), ...]``: the default variant ("")
    first, then variants in configured order; a layer no spec claims is its
    own group.
    """
    variant_order = {"": 0}
    for spec in specs:
        variant_order.setdefault(spec.variant, len(variant_order))
    prefixes = sorted(specs, key=lambda spec: -len(spec.label))
    groups: dict[str, list[tuple[str, str]]] = {}
    for layer in sorted(layers):
        for spec in prefixes:
            if layer.startswith(f"{spec.label}_"):
                base = f"{spec.base_label or spec.label}{layer[len(spec.label) :]}"
                groups.setdefault(base, []).append((spec.variant, layer))
                break
        else:
            groups.setdefault(layer, []).append(("", layer))
    return [
        (base, sorted(members, key=lambda member: variant_order.get(member[0], 99)))
        for base, members in sorted(groups.items())
    ]


def _variant_label(variant: str) -> str:
    return variant or "default"


def _posterior_layers(layers, specs, cfg) -> dict[str, str]:
    """Each feature layer's model posterior layer (`HPS-18`): the posterior of
    the model label that prefixes it (the longest, so variants map to theirs)."""
    state = str(getattr(cfg, "hmm_posterior_state", "Modified"))
    suffix = state.strip().lower().replace(" ", "_").replace("-", "_")
    labels = sorted({spec.label for spec in specs}, key=len, reverse=True)
    out = {}
    for layer in layers:
        label = next((lab for lab in labels if str(layer).startswith(f"{lab}_")), None)
        if label is not None:
            out[layer] = f"{label}_posterior_{suffix}"
    return out


def _matching_hmm_layers(records, roots, *, lengths: bool = False) -> list[str]:
    """Resolve configured feature roots against layers actually written by the tasks."""
    available = {str(layer) for record in records for layer in record_layers(record)}
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
    # `HPS-18`: the posterior beside each feature panel.
    posterior_of = (
        _posterior_layers([*feature_layers, *length_layers], _configured_model_specs(cfg), cfg)
        if bool(getattr(cfg, "hmm_clustermap_posterior_panel", False))
        else {}
    )
    requested_layers = list(
        dict.fromkeys([raw_layer, *feature_layers, *length_layers, *posterior_of.values()])
    )
    spine = load_spine(output_spine, verbose=False)
    plans = resolve_plot_region_plans(
        spine,
        pd.DataFrame(records),
        spine_path=output_spine,
        allow_gaps=bool(getattr(cfg, "plot_allow_unanalyzed_gaps", False)),
    )
    read_index = output_dir / "read_index"
    # HMM figures draw at most this many reads per sample (`hmm_` key, so the
    # shared clustermap cap -- and earlier stages' fingerprints -- stay as they are).
    max_reads_per_plot = getattr(cfg, "hmm_clustermap_max_reads_per_plot", None) or getattr(
        cfg, "clustermap_max_reads_per_plot", DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT
    )
    selection_seed = int(getattr(cfg, "plot_subsample_seed", 0))
    for plan in plans:
        reference, core_start, core_end = plan.reference, plan.start, plan.end
        selection = select_plot_reads(
            read_index,
            plan,
            max_reads_per_barcode=max_reads_per_plot,
            seed=selection_seed,
        )
        if not selection.read_ids:
            continue
        adata = materialize(
            output_spine,
            references=reference,
            read_ids=selection.read_ids,
            start=core_start,
            end=core_end,
            layers=requested_layers,
        )
        mask_unanalyzed_gaps(adata, plan.gaps)
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
            / f"region={core_start:012d}-{core_end:012d}"
        )

        def _register_region_plot(path: Path, *, plot_type: str, layer: str) -> None:
            source_manifest = write_plot_source_manifest(
                layout,
                path,
                stage="hmm",
                plot_type=plot_type,
                region=plan.source_manifest(),
                layers=[raw_layer, layer],
                selection_seed=selection.seed,
                selection_sha256=selection.selection_sha256,
                selected_molecule_uids=selection.molecule_uids,
            )
            register_plot_artifact(
                layout,
                path,
                stage="hmm",
                category="clustermaps",
                plot_type=plot_type,
                reference=reference,
                sample=path.stem.rsplit("__", 1)[-1],
                core_start=core_start,
                core_end=core_end,
                source_manifest=source_manifest,
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
            max_reads_per_plot=max_reads_per_plot,
            cfg=cfg,
        )
        specs = _configured_model_specs(cfg)
        # One figure per feature: each emission variant's layer is a further
        # column in the same read order (`HCE-06`).
        for _, members in variant_layer_groups(feature_layers, specs):
            layer, extras = members[0][1], [name for _, name in members[1:]]
            if posterior_of.get(layer) and posterior_of[layer] != layer:
                extras = [*extras, posterior_of[layer]]
            plot_dir = base_dir / "features" / _component(layer)
            combined_hmm_raw_clustermap(
                adata,
                hmm_feature_layer=layer,
                extra_hmm_layers=extras,
                posterior_cmap=str(getattr(cfg, "hmm_posterior_cmap", None) or "RdBu_r"),
                cmap_hmm=(
                    str(getattr(cfg, "hmm_posterior_cmap", None) or "RdBu_r")
                    if "_posterior_" in layer
                    else None
                )
                or _resolve_feature_colormap(
                    layer, cfg, str(getattr(cfg, "clustermap_cmap_hmm", "coolwarm"))
                ),
                save_path=plot_dir,
                normalize_hmm=False,
                **common,
            )
            for path in sorted(plot_dir.glob("*.png")):
                _register_region_plot(
                    path,
                    plot_type="hmm_accessible_feature_clustermap",
                    layer=layer,
                )
        for _, members in variant_layer_groups(length_layers, specs):
            layer, extras = members[0][1], [name for _, name in members[1:]]
            if posterior_of.get(layer):
                extras = [*extras, posterior_of[layer]]
            plot_dir = base_dir / "lengths" / _component(layer)
            combined_hmm_length_clustermap(
                adata,
                length_layer=layer,
                extra_length_layers=extras,
                posterior_cmap=str(getattr(cfg, "hmm_posterior_cmap", None) or "RdBu_r"),
                cmap_lengths=_resolve_feature_colormap(layer, cfg, "Greens"),
                length_feature_ranges=_resolve_length_feature_ranges(layer, cfg, "Greens"),
                save_path=plot_dir,
                **common,
            )
            for path in sorted(plot_dir.glob("*.png")):
                _register_region_plot(
                    path,
                    plot_type="hmm_footprint_length_clustermap",
                    layer=layer,
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
                    "efficiency_offset": portable.get("efficiency_offset"),
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
                "read_reference": read_reference.get(read_id, plan.reference),
            }
            for plan in plans
            for read_reference in [
                {r: m.reference for m in plan.members for r in m.selected_read_ids}
            ]
            for rank, read_id in enumerate(plan.selected_read_ids)
        ]
    )


def execute_partitioned_hmm(spine_path, cfg, output_dir) -> dict[str, Path]:
    """Run bounded HMM tasks and publish a linked thin spine."""
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from ..informatics.derived_read_index import prepare_derived_read_index

    prepare_derived_read_index(output_dir)
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
        spine_path=spine_path,
        bytes_per_position=hmm_bytes_per_position(cfg, _configured_model_specs(cfg)),
    )
    if not tasks:
        raise RuntimeError("partitioned HMM has no non-empty tasks")
    # `DLS-03b`: feature layers are derived from the posterior over each read's
    # whole span, which equals the stage's own calls only when a reference is
    # decoded as one window. If any reference is tiled, every task stores its
    # feature layers (one rule per generation: a catalog never mixes).
    tiled = tiled_references(spine, tasks)
    if tiled:
        cfg = copy.copy(cfg)
        cfg._hmm_tiled_references = sorted(tiled)
        logger.info("HMM feature layers stored (tiled references: %s)", ", ".join(sorted(tiled)))
    models_dir = output_dir / HMM_MODEL_SUBDIR
    if bool(getattr(cfg, "force_redo_hmm_fit", False)):
        # One revision per stage invocation: every worker resolves the same new
        # immutable IDs, while a later forced invocation receives fresh IDs.
        setattr(cfg, "_hmm_fit_revision", uuid.uuid4().hex)
    from ..memory_guard import (
        require_memory_headroom,
        resource_envelope_for_config,
        run_tasks_parallel,
    )

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
    # Each fitting plan resolves its own `auto` emission scope (`HPS-05`).
    model_specs, sample_attributes = _resolve_spec_scopes(spine, model_specs, cfg)
    fit_planning = build_hmm_fit_plan(tasks, model_specs, cfg, sample_attributes=sample_attributes)
    fit_catalog_path = output_dir / HMM_FIT_CATALOG
    fit_selection_path = output_dir / HMM_FIT_SELECTION_CATALOG
    _fit_catalog_frame(fit_planning.all_plans).to_parquet(fit_catalog_path, index=False)
    _fit_selection_frame(fit_planning.all_plans).to_parquet(fit_selection_path, index=False)

    single_plans = [plan for plan in fit_planning.base_plans if not plan.emission_group]
    unpooled = sorted(
        {
            plan.model_spec.label
            for plan in single_plans
            if str(getattr(plan.model_spec.config(cfg), "hmm_transition_scope", "fit_group"))
            == "pooled"
        }
    )
    if unpooled:
        logger.warning(
            "hmm_transition_scope pooled needs a shared emission scope; %s fit their own "
            "transitions per sample",
            ", ".join(unpooled),
        )
    base_results = run_tasks_parallel(
        execute_hmm_fit_task,
        [(spine_path, plan, cfg, models_dir, None) for plan in single_plans],
        cfg=cfg,
        force_sequential=str(device) != "cpu",
        pool_label=f"hmm fit pool ({len(single_plans)} models, device={device})",
        per_item_memory_mb=max(
            (plan.estimated_memory_bytes for plan in single_plans),
            default=1,
        )
        / (1024**2),
        estimator="hmm_fit_plan_peak",
    )
    artifacts_by_fit = {
        plan.fit_id: artifact for plan, artifact in zip(single_plans, base_results, strict=True)
    }
    # Shared emission scopes (`HPS-03`): one tied fit per model x emission group.
    # One tied fit per model x emission group -- or, with pooled transitions
    # (`HPS-16`), one per model across its emission groups.
    tied_groups: dict[tuple[str, str], list[HMMFitPlan]] = {}
    for plan in fit_planning.base_plans:
        if plan.emission_group:
            pooled_spec = (
                str(getattr(plan.model_spec.config(cfg), "hmm_transition_scope", "fit_group"))
                == "pooled"
            )
            key = (plan.model_spec.label, "" if pooled_spec else plan.emission_group)
            tied_groups.setdefault(key, []).append(plan)
    tied_jobs = [tied_groups[key] for key in sorted(tied_groups)]
    # One group at a time, its members spread over every thread: a group's
    # rounds are synchronised, so one large group in one worker would leave
    # the others idle (and members of different groups never share work).
    tied_workers = max(1, int(resource_envelope_for_config(cfg).resolved_threads))
    tied_results = run_tasks_parallel(
        execute_tied_fit_task,
        [(spine_path, group, cfg, models_dir, tied_workers) for group in tied_jobs],
        cfg=cfg,
        force_sequential=True,
        pool_label=f"hmm tied fit pool ({len(tied_jobs)} emission groups, device={device})",
        per_item_memory_mb=max(
            (sum(plan.estimated_memory_bytes for plan in group) for group in tied_jobs),
            default=1,
        )
        / (1024**2),
        estimator="hmm_tied_fit_peak",
    )
    for group, results in zip(tied_jobs, tied_results, strict=True):
        for plan, artifact in zip(group, results, strict=True):
            artifacts_by_fit[plan.fit_id] = artifact
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
    _plan_settings_columns(
        _fit_catalog_frame(fit_planning.all_plans, artifacts_by_fit), model_specs, cfg
    ).to_parquet(fit_catalog_path, index=False)

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
        per_item_memory_mb=(
            max(task.estimated_memory_bytes for task in tasks) + HMM_TASK_BASE_BYTES
        )
        / (1024**2),
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
    _plot_feature_count_size_histograms(
        records, output_dir, layout, specs=_configured_model_specs(cfg)
    )
    _plot_molecule_fractions(records, output_dir, layout, specs=_configured_model_specs(cfg))
    _plot_hmm_vs_raw_scatter(records, output_dir, layout, specs=_configured_model_specs(cfg))
    _plot_hmm_parameters_across_barcodes(records, models_dir, cfg, layout)
    _plot_hmm_fit_history(models_dir, layout)
    from .hmm_context_qc import write_hmm_context_qc

    try:
        context_qc_dir = write_hmm_context_qc(output_dir, records, spine.uns, cfg, layout=layout)
    except Exception:
        context_qc_dir = None
        logger.exception("HMM context QC failed; generation still published")

    output_spine = output_dir / HMM_SPINE_FILENAME
    hmm_spine = spine.copy()
    # Relative to the run's output_directory, not output_dir -- see
    # informatics.partition_read._run_root_from_spine_path. Resolved via
    # run_root_from_stage_dir because output_dir may be a generation or staging
    # directory, which sits three levels below the run root rather than two;
    # anchoring on output_dir.parent there would bake `.staging/<id>/...` into
    # every pointer below and leave them dangling once the generation publishes.
    run_root = run_root_from_stage_dir(output_dir)
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
    from ..informatics.derived_read_index import DERIVED_READ_INDEX_DIRNAME

    read_index_dir = output_dir / DERIVED_READ_INDEX_DIRNAME
    hmm_spine.uns["hmm_read_index"] = relative_uns_path(read_index_dir, run_root)
    hmm_spine.uns["hmm_schema_version"] = 2
    # `HPS-17`: reads a model was fitted on, so analyses can hold them out.
    selection = pd.read_parquet(fit_selection_path)
    fit_reads = set(selection["read_id"].astype(str)) if "read_id" in selection.columns else set()
    read_ids = (
        hmm_spine.obs["read_id"].astype(str)
        if "read_id" in hmm_spine.obs
        else pd.Series(hmm_spine.obs_names.astype(str), index=hmm_spine.obs_names)
    )
    hmm_spine.obs["hmm_fit_read"] = read_ids.isin(fit_reads).to_numpy()
    # What the feature layers can be re-derived with (`DLS-03`), as JSON.
    hmm_spine.uns["hmm_feature_settings"] = json.dumps(
        hmm_feature_settings(records), sort_keys=True
    )
    safe_write_h5ad(hmm_spine, output_spine, backup=False, verbose=False)
    # The superset experiment spine is regenerated by the caller after this
    # generation is published, matching latent (cli/latent_adata.py), raw
    # (cli/raw_adata.py), and preprocess (preprocess_generation.py). Doing it
    # here would union a still-staged generation into the experiment spine and
    # leave it referenced even if validation then rejected the generation.
    _plot_feature_clustermaps(records, output_spine, output_dir, layout, cfg)

    manifest = sidecar_manifest_path(output_dir)
    register_sidecar(manifest, "hmm_spine", output_spine)
    register_sidecar(manifest, "hmm_source_spine", spine_path)
    register_sidecar(manifest, "hmm_task_catalog", catalog_path)
    register_sidecar(manifest, "hmm_read_index", read_index_dir)
    register_sidecar(manifest, "hmm_store", output_dir / HMM_PARTIAL_SUBDIR)
    register_sidecar(manifest, "hmm_model_store", models_dir)
    register_sidecar(manifest, "hmm_fit_catalog", fit_catalog_path)
    register_sidecar(manifest, "hmm_fit_selection", fit_selection_path)
    register_sidecar(manifest, "hmm_plot_catalog", layout.catalog)
    if context_qc_dir is not None:
        register_sidecar(manifest, "hmm_context_qc", context_qc_dir)
    logger.info("Wrote partitioned HMM stage with %d task(s)", len(tasks))
    return {
        "spine": output_spine,
        "task_catalog": catalog_path,
        "read_index": read_index_dir,
        "store": output_dir / HMM_PARTIAL_SUBDIR,
        "models": models_dir,
        "fit_catalog": fit_catalog_path,
        "fit_selection": fit_selection_path,
        "plots": layout.root,
        "plot_catalog": layout.catalog,
        "manifest": manifest,
    }
