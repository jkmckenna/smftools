"""What a CNN's detectors respond to: the detector catalogue (`MLR-03b`).

A position-agnostic CNN (`MLR-08`) is a bag of pattern detectors: each
final-layer channel, at every position, scores the input window it sees.
`detector_catalogue_run` describes those detectors for one model of a
published train run, out of fold (each held-out molecule read by the fold
model that held it out), and publishes an explain run naming the fold models:

- per molecule and detector, the maximum activation over positions and where
  it occurs (``data/maxima_NN.npz``, with the molecules in
  ``data/molecules.parquet``);
- ``windows.parquet``: per detector, its top molecules (one window each, at
  the position of the molecule's maximum): activation, centre coordinate,
  true class;
- ``patterns/fold_NN.npy``: per detector, the mean input pattern over its
  top windows (detectors x channels x window positions; NaN where never
  observed), the window spanning the detector's effective span;
- ``detectors.parquet``: per fold and detector -- AUROC of the per-molecule
  maximum alone (how predictive the detector is by itself), the active
  fraction of its top molecules against the base rate (log2 enrichment), mean
  and SD of its window centres on the locus, and its group: detectors whose
  per-molecule maxima correlate (Spearman >= ``group_similarity``) are
  grouped, so redundant ones collapse;
- ``summary.json`` and ``figures/detectors.png`` (`plot_detector_catalogue`,
  the most predictive detectors of the largest fold).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.readwrite import atomic_write_json

from ..artifacts import EnvironmentRecord, capture_environment_record, rebuild_workspace_indexes
from ..plan import parse_ml_plan
from ..workspace import MLWorkspace, resolve_ml_workspace
from .binding import FINAL_FOLD, bind_ml_job
from .comparison import roc_auc
from .contracts import (
    JobArtifact,
    JobExecutionContext,
    JobOperationResult,
    MLJobServiceError,
    ModelSelectionRequest,
    ResolvedJob,
)
from .explanations import PublishedExplanationRun, _read_run, _rows, _sample
from .resolution import resolve_model_selection
from .runs import MODELS, PREDICTIONS, TAGS, _load_model, _tags
from .service import run_explain_job

METHOD = "DetectorCatalogue"
REQUEST = "request.json"
MOLECULES = "data/molecules.parquet"
WINDOWS = "windows.parquet"
DETECTORS = "detectors.parquet"
PATTERN_INDEX = "patterns/index.json"
SUMMARY = "summary.json"
FIGURE = "figures/detectors.png"


def _maxima(fitted, data, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per molecule and final-layer channel: max activation and its position
    (column index), without holding the full activation map."""
    import torch

    from ..data.transforms import TorchFeatureTransform

    transform = TorchFeatureTransform(fitted.transform, device=fitted.resolved_device)
    model = fitted.model
    was_training = bool(model.training)
    model.eval()
    maxima, where = [], []
    try:
        transformed = transform(data)
        design = transformed.design_mask
        with torch.no_grad():
            for start in range(0, len(data.molecule_uids), batch_size):
                rows = slice(start, start + batch_size)
                features = model.forward_features(
                    transformed.values[rows],
                    observed_mask=transformed.observed_mask[rows],
                    availability_mask=transformed.availability_mask[rows],
                    design_mask=design[rows] if design.ndim == 3 else design,
                    padding_mask=transformed.padding_mask[rows],
                )
                value, index = features.max(dim=-1)
                maxima.append(value.cpu().numpy())
                where.append(index.cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(maxima).astype(np.float32), np.concatenate(where).astype(np.int32)


def _windows(values: np.ndarray, centres: np.ndarray, half: int) -> np.ndarray:
    """Input windows (molecules x channels x 2 half + 1) around ``centres``,
    NaN outside the molecule. ``values``: molecules x positions x channels."""
    n, positions, channels = values.shape
    offsets = np.arange(-half, half + 1)
    columns = centres[:, None] + offsets[None, :]
    inside = (columns >= 0) & (columns < positions)
    clipped = np.clip(columns, 0, positions - 1)
    picked = values[np.arange(n)[:, None], clipped]  # n x window x channels
    picked = np.where(inside[:, :, None], picked, np.nan)
    return np.transpose(picked, (0, 2, 1))


def _groups(maxima: np.ndarray, similarity: float) -> np.ndarray:
    """Group detectors whose per-molecule maxima rank-correlate >= similarity."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n_detectors = maxima.shape[1]
    if n_detectors == 1:
        return np.ones(1, dtype=int)
    ranks = pd.DataFrame(maxima).rank().to_numpy()
    correlation = np.corrcoef(ranks, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    distance = np.clip(1 - correlation, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    return fcluster(tree, t=1 - similarity, criterion="distance")


def detector_catalogue_run(
    run_id: str,
    *,
    model: str,
    workspace: MLWorkspace | None = None,
    project_dir: str | Path | None = None,
    max_per_fold: int | None = 2000,
    top_windows: int = 50,
    window: int | None = None,
    group_similarity: float = 0.8,
    seed: int = 0,
    tags: dict[str, Any] | None = None,
    policy=None,
    environment: EnvironmentRecord | None = None,
    rebuild_index: bool = True,
    figure: bool = True,
    device: str = "cpu",
) -> PublishedExplanationRun:
    """Catalogue the final-layer detectors of a residual CNN of a train run.

    Args:
        run_id / model: A completed train run and its residual CNN model key.
        max_per_fold: Held-out molecules read per fold (class-stratified, seeded).
        top_windows: Top molecules (one window each) per detector.
        window: Window width in positions (odd); default the fold model's
            recorded 90 % effective span (`MLR-08`), else its receptive field,
            at most 401.
        group_similarity: Spearman correlation of per-molecule maxima at or
            above which detectors are grouped.
        device: Where the CNN runs (cpu, cuda, mps or auto).
        Others: as `explain_run`.
    """
    if (workspace is None) == (project_dir is None):
        raise MLJobServiceError("pass exactly one of workspace or project_dir")
    if workspace is None:
        workspace = resolve_ml_workspace(project_dir=project_dir)
    manifest, run_path = _read_run(workspace, run_id)
    records = [
        item
        for item in json.loads((run_path / MODELS).read_text())
        if item["model"] == model and item["fold"] != FINAL_FOLD
    ]
    if not records:
        raise MLJobServiceError(f"run {run_id} has no fold models for {model!r}")
    if any(record["backend"] != "torch" for record in records):
        raise MLJobServiceError(f"{model!r} is not a torch model: no detectors to catalogue")
    train_plan = parse_ml_plan(json.loads((run_path / "resolved_plan.json").read_text()))
    train_job = manifest["job_name"]
    document = train_plan.to_dict()
    explain_job = f"catalogue_{model}"
    document["jobs"][explain_job] = {
        "action": "explain",
        "dataset": train_plan.jobs[train_job].dataset,
        "model": model,
        "explain": [METHOD],
        "source_job": train_job,
    }
    plan = parse_ml_plan(document)
    owner = {"project_dir": workspace.owner_root}
    if workspace.scope_kind != "project":
        owner = {"experiment_dir": workspace.owner_root}
    bound = bind_ml_job(train_plan, train_job, policy=policy, **owner)
    if bound.snapshot.snapshot_id != manifest["dataset_snapshot_id"]:
        raise MLJobServiceError(
            "the run's dataset has changed since training (snapshot differs); re-train first"
        )
    folds = {fold.fold_name or "single": fold for fold in bound.folds}
    for record in records:
        fold = folds.get(record["fold"])
        if fold is None or fold.split.split_id != record["split_id"]:
            raise MLJobServiceError(f"fold {record['fold']!r} no longer matches the run's split")
    selections = tuple(
        resolve_model_selection(
            workspace, ModelSelectionRequest(kind="exact", model_id=record["model_id"])
        )
        for record in records
    )
    predictions = pd.read_parquet(run_path / PREDICTIONS)
    predictions = predictions[predictions["model"] == model]
    environment = environment or capture_environment_record()
    tags = _tags(tags)
    settings = {
        "source_run_id": run_id,
        "model": model,
        "method": METHOD,
        "max_per_fold": max_per_fold,
        "top_windows": top_windows,
        "window": window,
        "group_similarity": group_similarity,
        "seed": seed,
    }
    job = ResolvedJob(
        plan=plan,
        workspace=workspace,
        job_name=explain_job,
        environment=environment,
        resolved_config={"tags": tags, **settings},
        dataset_snapshot_id=bound.snapshot.snapshot_id,
        model_selections=selections,
        seeds={"explanation": seed},
    )
    state: dict[str, Any] = {}

    def operation(context: JobExecutionContext) -> JobOperationResult[None]:
        rng = np.random.default_rng(seed)
        atomic_write_json(context.output_path(TAGS), tags)
        classes = {item.molecule_uid: item.class_id for item in bound.snapshot.observations}
        molecule_frames, window_frames, detector_frames, index_folds = [], [], [], []
        artifacts: list[JobArtifact] = []
        channels = coordinates = None
        positive = None
        for number, record in enumerate(records):
            fold_name = record["fold"]
            context.advance_phase(f"catalogue:{fold_name}")
            fold = folds[fold_name]
            fitted = _load_model(workspace, record["model_id"], record["backend"], device)
            config = getattr(fitted.model, "config", None)
            if config is None or not hasattr(config, "receptive_field"):
                raise MLJobServiceError(f"{model!r} is not a residual CNN")
            labels = fitted.label_schema
            positive = labels.positive_class or labels.class_order[-1]
            positive_id = list(labels.class_order).index(positive)
            roles = dict(fold.resolution.assignments)
            held_out = [uid for uid, role in roles.items() if role == "test"]
            chosen = _sample(held_out, classes, max_per_fold, rng)
            data = _rows(fold.dataset, "test", chosen)
            maxima, where = _maxima(fitted, data)
            channels = [channel.name for channel in fitted.input_schema.channels]
            coordinates = np.asarray(fitted.transform.coordinates, dtype=np.int64)
            truth = np.asarray(data.labels) == positive_id
            scale = record.get("detector_scale") or {}
            width = window or min(scale.get("effective_span_90") or config.receptive_field, 401)
            half = int(width) // 2
            values = np.where(
                np.asarray(data.observed_mask, dtype=bool),
                np.asarray(data.values, dtype=np.float32),
                np.float32(np.nan),
            )
            groups = _groups(maxima, group_similarity)
            patterns = np.full(
                (maxima.shape[1], len(channels), 2 * half + 1), np.nan, dtype=np.float32
            )
            base_rate = float(truth.mean()) if truth.size else float("nan")
            for detector in range(maxima.shape[1]):
                order = np.argsort(-maxima[:, detector], kind="stable")[:top_windows]
                order = order[maxima[order, detector] > 0]
                centres = where[order, detector]
                if order.size:
                    stack = _windows(values[order], centres, half)
                    with np.errstate(invalid="ignore"), _quiet():
                        patterns[detector] = np.nanmean(stack, axis=0)
                window_frames.append(
                    pd.DataFrame(
                        {
                            "fold": fold_name,
                            "detector": detector,
                            "rank": np.arange(order.size),
                            "molecule_uid": [data.molecule_uids[i] for i in order],
                            "activation": maxima[order, detector],
                            "centre": coordinates[centres] if order.size else [],
                            "truth_positive": truth[order],
                        }
                    )
                )
                top_positive = float(truth[order].mean()) if order.size else float("nan")
                detector_frames.append(
                    {
                        "fold": fold_name,
                        "model_id": record["model_id"],
                        "detector": detector,
                        "group": int(groups[detector]),
                        "auroc": roc_auc(truth, maxima[:, detector]),
                        "mean_max_positive": float(maxima[truth, detector].mean())
                        if truth.any()
                        else None,
                        "mean_max_other": float(maxima[~truth, detector].mean())
                        if (~truth).any()
                        else None,
                        "n_top": int(order.size),
                        "top_positive_fraction": top_positive,
                        "log2_enrichment": float(np.log2(top_positive / base_rate))
                        if order.size and top_positive > 0 and base_rate > 0
                        else None,
                        "centre_mean": float(coordinates[centres].mean()) if order.size else None,
                        "centre_sd": float(coordinates[centres].std()) if order.size else None,
                        "window": 2 * half + 1,
                    }
                )
            pattern_path = f"patterns/fold_{number:02d}.npy"
            np.save(context.output_path(pattern_path), patterns)
            maxima_path = f"data/maxima_{number:02d}.npz"
            np.savez_compressed(context.output_path(maxima_path), maxima=maxima, where=where)
            artifacts += [
                JobArtifact(f"patterns:{fold_name}", pattern_path, "application/x-npy"),
                JobArtifact(f"maxima:{fold_name}", maxima_path, "application/x-npz"),
            ]
            index_folds.append(
                {
                    "fold": fold_name,
                    "model_id": record["model_id"],
                    "patterns": pattern_path,
                    "maxima": maxima_path,
                    "n_molecules": len(data.molecule_uids),
                    "n_detectors": int(maxima.shape[1]),
                    "window": 2 * half + 1,
                }
            )
            scores = predictions[predictions["fold"] == fold_name].set_index("molecule_uid")
            molecule_frames.append(
                pd.DataFrame(
                    {
                        "fold": fold_name,
                        "row": np.arange(len(data.molecule_uids)),
                        "molecule_uid": list(data.molecule_uids),
                        "truth_positive": truth,
                        "score": scores[f"p_{positive}"]
                        .reindex(list(data.molecule_uids))
                        .to_numpy(),
                    }
                )
            )
        context.advance_phase("record")
        atomic_write_json(context.output_path(REQUEST), {**settings, "target_class": positive})
        pd.concat(molecule_frames, ignore_index=True).to_parquet(
            context.output_path(MOLECULES), index=False
        )
        pd.concat(window_frames, ignore_index=True).to_parquet(
            context.output_path(WINDOWS), index=False
        )
        detectors = pd.DataFrame(detector_frames)
        detectors.to_parquet(context.output_path(DETECTORS), index=False)
        atomic_write_json(
            context.output_path(PATTERN_INDEX),
            {
                "channels": channels,
                "axes": ["detector", "channel", "offset"],
                "folds": index_folds,
            },
        )
        best = detectors.assign(strength=(detectors["auroc"] - 0.5).abs()).sort_values(
            "strength", ascending=False
        )
        summary = {
            model: {
                "method": METHOD,
                "n_folds": len(index_folds),
                "n_detectors": int(detectors.groupby("fold")["detector"].nunique().max()),
                "n_groups_by_fold": {
                    str(fold): int(count)
                    for fold, count in detectors.groupby("fold")["group"].nunique().items()
                },
                "most_predictive": best.head(10)[
                    ["fold", "detector", "auroc", "log2_enrichment", "centre_mean"]
                ].to_dict("records"),
            }
        }
        atomic_write_json(context.output_path(SUMMARY), summary)
        state["summary"] = summary
        artifacts += [
            JobArtifact("tags", TAGS, "application/json"),
            JobArtifact("request", REQUEST, "application/json"),
            JobArtifact("molecules", MOLECULES, "application/vnd.apache.parquet"),
            JobArtifact("windows", WINDOWS, "application/vnd.apache.parquet"),
            JobArtifact("detectors", DETECTORS, "application/vnd.apache.parquet"),
            JobArtifact("pattern_index", PATTERN_INDEX, "application/json"),
            JobArtifact("summary", SUMMARY, "application/json"),
        ]
        if figure:
            from smftools.analysis.plot.ml_results import plot_detector_catalogue

            largest = max(index_folds, key=lambda item: item["n_molecules"])
            plot_detector_catalogue(
                detectors[detectors["fold"] == largest["fold"]],
                np.load(context.output_path(largest["patterns"])),
                channels=channels,
                title=f"{model}: detectors, held out {largest['fold']} ({run_id[:8]})",
                output_path=context.output_path(FIGURE),
            )
            artifacts.append(JobArtifact("figure", FIGURE, "image/png"))
        return JobOperationResult(value=None, artifacts=tuple(artifacts))

    outcome = run_explain_job(job, operation)
    if rebuild_index:
        rebuild_workspace_indexes(workspace)
    return PublishedExplanationRun(
        run_id=outcome.manifest.run_id,
        path=outcome.bundle.path,
        workspace=workspace,
        source_run_id=run_id,
        model=model,
        method=METHOD,
        summary=state["summary"],
        outcome=outcome,
    )


class _quiet:
    """Silence numpy's empty-slice warnings for windows never observed."""

    def __enter__(self):
        import warnings

        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)
