"""Out-of-fold explanations of a published train run (`MLR-03`).

`explain_run` explains one model of a train run with one method and publishes
the result as an explain run naming the run's fold models. Each held-out
molecule is explained by the fold model that held it out, so every
attribution is out-of-fold, like the run's predictions. Per fold, up to
``max_per_fold`` held-out molecules are explained (a seeded, class-stratified
sample); background-dependent methods (TreeSHAP interventional, integrated
gradients, ...) draw a seeded background from that fold's training molecules.

Payloads:

- ``tags.json``; ``request.json`` (method, parameters, target class, source
  run, sampling);
- ``data/molecules.parquet``: per explained molecule, fold, model id, row in
  that fold's matrix, truth and out-of-fold score (from the train run);
- ``attributions/index.json`` and ``attributions/fold_NN.npy``: per fold, a
  float32 molecules x channels x positions matrix -- each position's
  contribution to the target class (classical models: the sum over that
  position's transformed features, signal and indicators); absent for
  global methods (linear coefficients, permutation importance);
- ``importance.parquet``: per fold, channel and position, the global score --
  mean absolute and mean signed attribution, overall and per true class,
  for per-molecule methods; the method's own value for global ones;
- ``consistency.parquet``: per channel and fold pair, the Spearman rank
  correlation of position importance; ``summary.json``: its mean and the top
  positions by importance averaged over folds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.readwrite import atomic_write_json

from ..artifacts import (
    EnvironmentRecord,
    ExplanationMaskPolicy,
    ExplanationTarget,
    capture_environment_record,
    rebuild_workspace_indexes,
)
from ..data.partition_dataset import MLMaterializedPartitionData, PartitionDataset
from ..interpretability import (
    METHOD_CONTRACTS,
    ExplanationDecisionProvenance,
    InterpretabilityRequest,
    sample_training_background,
)
from ..plan import parse_ml_plan
from ..workspace import MLWorkspace, resolve_ml_workspace
from .actions import explain_partition_model
from .binding import FINAL_FOLD, BoundFold, bind_ml_job
from .contracts import (
    JobArtifact,
    JobExecutionContext,
    JobExecutionOutcome,
    JobOperationResult,
    MLJobServiceError,
    ModelSelectionRequest,
    ResolvedJob,
)
from .resolution import resolve_model_selection
from .runs import MODELS, PREDICTIONS, TAGS, _load_model, _tags
from .service import run_explain_job

REQUEST = "request.json"
MOLECULES = "data/molecules.parquet"
ATTRIBUTION_INDEX = "attributions/index.json"
IMPORTANCE = "importance.parquet"
CONSISTENCY = "consistency.parquet"
SUMMARY = "summary.json"
FIGURE = "figures/attributions.png"

# Method parameters used when the caller gives none.
DEFAULT_PARAMETERS: Mapping[str, Mapping[str, Any]] = {
    "NaiveBayesLogOdds": {},
    "LinearCoefficients": {"statistic": "coefficient"},
    "PermutationImportance": {"metric": "roc_auc", "n_repeats": 5},
    "TreeSHAP": {
        "model_output": "raw",
        "feature_perturbation": "tree_path_dependent",
        "check_additivity": False,
    },
    "Saliency": {"absolute": True, "example_batch_size": 64},
    "InputXGradient": {"example_batch_size": 64},
    "IntegratedGradients": {
        "baseline_reduction": "mean",
        "example_batch_size": 64,
        "integration_method": "gausslegendre",
        "internal_batch_size": 64,
        "n_steps": 32,
    },
    "DeepLift": {"baseline_reduction": "mean", "example_batch_size": 64},
    "GradientSHAP": {"example_batch_size": 64, "n_samples": 16, "stdevs": 0.0},
}


@dataclass(frozen=True)
class PublishedExplanationRun:
    """One published explain run of a train run's model."""

    run_id: str
    path: Path
    workspace: MLWorkspace
    source_run_id: str
    model: str
    method: str
    summary: Mapping[str, Any]
    outcome: JobExecutionOutcome

    def read(self, payload: str) -> Any:
        """One run payload (e.g. ``IMPORTANCE``): a DataFrame or parsed JSON."""
        path = self.path / payload
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())

    def attributions(self, fold: str) -> tuple[pd.DataFrame, np.ndarray]:
        """One fold's explained molecules (in matrix row order) and its
        molecules x channels x positions matrix."""
        index = self.read(ATTRIBUTION_INDEX)
        entry = next((item for item in index["folds"] if item["fold"] == fold), None)
        if entry is None or entry["file"] is None:
            raise KeyError(f"no per-molecule attributions for fold {fold!r}")
        molecules = self.read(MOLECULES)
        molecules = molecules[molecules["fold"] == fold].sort_values("row")
        return molecules.reset_index(drop=True), np.load(self.path / entry["file"])

    def inputs(self, fold: str) -> np.ndarray:
        """One fold's explained input values, molecules x channels x positions
        in `attributions` row order (NaN where unobserved)."""
        index = self.read(ATTRIBUTION_INDEX)
        entry = next((item for item in index["folds"] if item["fold"] == fold), None)
        if entry is None or not entry.get("inputs"):
            raise KeyError(f"no explained inputs for fold {fold!r}")
        return np.load(self.path / entry["inputs"])

    def pooled(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Every fold's molecules, attributions and inputs, concatenated."""
        index = self.read(ATTRIBUTION_INDEX)
        frames, matrices, inputs = [], [], []
        for entry in index["folds"]:
            molecules, matrix = self.attributions(entry["fold"])
            frames.append(molecules)
            matrices.append(matrix)
            inputs.append(self.inputs(entry["fold"]))
        return (
            pd.concat(frames, ignore_index=True),
            np.concatenate(matrices),
            np.concatenate(inputs),
        )


def _read_run(workspace: MLWorkspace, run_id: str) -> tuple[dict, Path]:
    path = workspace.runs_root / run_id
    manifest_path = path / "run_manifest.json"
    if not manifest_path.is_file():
        raise MLJobServiceError(f"no published run {run_id!r} in {workspace.runs_root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["action"] != "train" or manifest["state"] != "completed":
        raise MLJobServiceError(f"run {run_id} is not a completed train run")
    return manifest, path


def _rows(
    dataset: PartitionDataset, split: str, wanted: Sequence[str]
) -> MLMaterializedPartitionData:
    """The wanted molecules of one split, read batch by batch, in ``wanted``
    order -- without materializing the whole split."""
    position = {uid: index for index, uid in enumerate(wanted)}
    parts: list[tuple[np.ndarray, Any, np.ndarray]] = []
    found = 0
    for batch in dataset.iter_batches(split):
        rows = np.asarray(
            [index for index, uid in enumerate(batch.molecule_uids) if uid in position],
            dtype=np.int64,
        )
        if rows.size:
            order = np.asarray([position[batch.molecule_uids[i]] for i in rows])
            parts.append((rows, batch, order))
            found += rows.size
        if found == len(wanted):
            break
    if found != len(wanted):
        raise MLJobServiceError(f"{len(wanted) - found} requested {split} molecules were not read")
    target = np.concatenate([order for _rows, _batch, order in parts])
    arrange = np.argsort(target, kind="stable")

    def gather(name: str) -> np.ndarray:
        values = np.concatenate([getattr(batch, name)[rows] for rows, batch, _o in parts])
        return values[arrange]

    def gather_tuple(name: str) -> tuple:
        values = [getattr(batch, name)[i] for rows, batch, _o in parts for i in rows]
        return tuple(values[i] for i in arrange)

    first = parts[0][1]
    labels = None if first.labels is None else gather("labels")
    return MLMaterializedPartitionData(
        split=split,
        molecule_uids=gather_tuple("molecule_uids"),
        read_ids=gather_tuple("read_ids"),
        experiment_uids=gather_tuple("experiment_uids"),
        modalities=gather_tuple("modalities"),
        coordinates=first.coordinates,
        channel_names=first.channel_names,
        values=gather("values"),
        labels=labels,
        observed_mask=gather("observed_mask"),
        availability_mask=gather("availability_mask"),
        # A design mask is per position (2-D) unless it varies by molecule.
        design_mask=gather("design_mask") if first.design_mask.ndim == 3 else first.design_mask,
        padding_mask=gather("padding_mask"),
    )


def _sample(
    uids: Sequence[str], classes: Mapping[str, Any], limit: int | None, rng: np.random.Generator
) -> list[str]:
    """Up to ``limit`` molecules, stratified by class (proportional), seeded."""
    uids = sorted(uids)
    if limit is None or len(uids) <= limit:
        return uids
    frame = pd.DataFrame({"uid": uids, "cls": [classes.get(uid) for uid in uids]})
    chosen: list[str] = []
    for _cls, group in frame.groupby("cls", dropna=False, sort=True):
        take = max(1, round(limit * len(group) / len(frame)))
        chosen.extend(
            rng.choice(group["uid"].to_numpy(), size=min(take, len(group)), replace=False)
        )
    return sorted(chosen)


def _position_matrix(result, model) -> np.ndarray | None:
    """Per-molecule attributions as molecules x channels x positions (float32);
    ``None`` for a global (per-feature) result."""
    channels = [channel.name for channel in model.input_schema.channels]
    coordinates = [int(value) for value in model.transform.coordinates]
    values = np.asarray(result.values, dtype=np.float64)
    if result.axes == ("feature",):
        return None
    if result.axes == ("observation", "feature"):
        channel_index = {name: i for i, name in enumerate(channels)}
        coordinate_index = {value: i for i, value in enumerate(coordinates)}
        matrix = np.zeros((values.shape[0], len(channels), len(coordinates)), dtype=np.float64)
        for column, feature in enumerate(result.features):
            matrix[
                :, channel_index[feature.channel.name], coordinate_index[feature.coordinate]
            ] += values[:, column]
        return matrix.astype(np.float32)
    if result.axes == ("observation", "position", "channel"):
        return np.transpose(values, (0, 2, 1)).astype(np.float32)
    raise MLJobServiceError(f"explanation records do not support attribution axes {result.axes}")


def _global_vector(result, model) -> np.ndarray:
    """A global (per-feature) result summed per channel and position."""
    channels = [channel.name for channel in model.input_schema.channels]
    coordinates = [int(value) for value in model.transform.coordinates]
    channel_index = {name: i for i, name in enumerate(channels)}
    coordinate_index = {value: i for i, value in enumerate(coordinates)}
    vector = np.zeros((len(channels), len(coordinates)), dtype=np.float64)
    for value, feature in zip(np.asarray(result.values), result.features, strict=True):
        vector[channel_index[feature.channel.name], coordinate_index[feature.coordinate]] += value
    return vector


def _importance_rows(
    fold: str,
    channels: Sequence[str],
    coordinates: Sequence[int],
    statistics: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    frames = []
    for name, array in statistics.items():
        for c, channel in enumerate(channels):
            frames.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "channel": channel,
                        "coordinate": list(coordinates),
                        "statistic": name,
                        "value": array[c],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _consistency(importance: pd.DataFrame, statistic: str) -> pd.DataFrame:
    rows = []
    chosen = importance[importance["statistic"] == statistic]
    for channel, frame in chosen.groupby("channel"):
        wide = frame.pivot(index="coordinate", columns="fold", values="value")
        folds = list(wide.columns)
        for i, left in enumerate(folds):
            for right in folds[i + 1 :]:
                rho = wide[left].rank().corr(wide[right].rank())
                rows.append(
                    {
                        "channel": channel,
                        "fold_a": left,
                        "fold_b": right,
                        "statistic": statistic,
                        "spearman": None if pd.isna(rho) else float(rho),
                    }
                )
    return pd.DataFrame(rows, columns=["channel", "fold_a", "fold_b", "statistic", "spearman"])


def explain_run(
    run_id: str,
    *,
    model: str,
    method: str | None = None,
    workspace: MLWorkspace | None = None,
    project_dir: str | Path | None = None,
    parameters: Mapping[str, Any] | None = None,
    max_per_fold: int | None = 2000,
    background_size: int = 100,
    seed: int = 0,
    tags: Mapping[str, Any] | None = None,
    policy=None,
    environment: EnvironmentRecord | None = None,
    rebuild_index: bool = True,
    figure: bool = True,
    device: str = "cpu",
) -> PublishedExplanationRun:
    """Explain one model of a published train run, out-of-fold, and publish it.

    Args:
        run_id: A completed train run in the workspace.
        model: The run's model key (e.g. ``"nb"``).
        method: An interpretability method the model supports (default: the
            model family's ``default_explanation``, `MLR-09`), e.g.
            ``NaiveBayesLogOdds`` (naive Bayes), ``TreeSHAP`` (random
            forest), ``LinearCoefficients`` / ``PermutationImportance``
            (global), ``IntegratedGradients`` (torch).
        workspace / project_dir: The workspace holding the run.
        parameters: Method parameters (defaults: ``DEFAULT_PARAMETERS``).
        max_per_fold: Held-out molecules explained per fold (class-stratified,
            seeded); ``None`` explains every one.
        background_size: Training molecules per fold for background-dependent
            methods.
        seed: Seed for molecule and background sampling and the method.
        tags, policy, environment, rebuild_index: As `train_and_publish`.
        device: Where torch models run (cpu, cuda, mps or auto); gradient
            methods are much faster on a GPU.
        figure: Draw the default attribution clustermap into the run
            (``figures/attributions.png``: all folds, blocks by true class);
            `plot_explanation` draws others from the record.

    The run's data must be unchanged since training: the re-bound dataset
    snapshot and fold splits must equal the run's.
    """
    if (workspace is None) == (project_dir is None):
        raise MLJobServiceError("pass exactly one of workspace or project_dir")
    if workspace is None:
        workspace = resolve_ml_workspace(project_dir=project_dir)
    if method is not None and method not in METHOD_CONTRACTS:
        raise MLJobServiceError(f"unknown explanation method {method!r}")
    manifest, run_path = _read_run(workspace, run_id)
    records = [
        item
        for item in json.loads((run_path / MODELS).read_text())
        if item["model"] == model and item["fold"] != FINAL_FOLD
    ]
    if not records:
        raise MLJobServiceError(f"run {run_id} has no fold models for {model!r}")
    if method is None:
        from ..models.registry import BUILTIN_MODEL_REGISTRY

        family = records[0]["family"]
        method = (
            BUILTIN_MODEL_REGISTRY.definition(family).default_explanation
            if family in BUILTIN_MODEL_REGISTRY.names
            else None
        )
        if method is None:
            raise MLJobServiceError(f"family {family!r} has no default explanation; name a method")
    if method not in METHOD_CONTRACTS:
        raise MLJobServiceError(f"unknown explanation method {method!r}")
    contract = METHOD_CONTRACTS[method]
    if contract.layer_policy == "required":
        raise MLJobServiceError(f"{method} (layer attributions) is not recorded yet")
    train_plan = parse_ml_plan(json.loads((run_path / "resolved_plan.json").read_text()))
    train_job = manifest["job_name"]
    document = train_plan.to_dict()
    explain_job = f"explain_{model}"
    document["jobs"][explain_job] = {
        "action": "explain",
        "dataset": train_plan.jobs[train_job].dataset,
        "model": model,
        "explain": [method],
        "source_job": train_job,
    }
    plan = parse_ml_plan(document)
    owner = {"project_dir": workspace.owner_root}
    if workspace.scope_kind != "project":
        owner = {"experiment_dir": workspace.owner_root}
    # Bind with the run's own plan: the snapshot identity includes the plan
    # hash, which the added explain job changes.
    bound = bind_ml_job(train_plan, train_job, policy=policy, **owner)
    if bound.snapshot.snapshot_id != manifest["dataset_snapshot_id"]:
        raise MLJobServiceError(
            "the run's dataset has changed since training (snapshot differs); re-train first"
        )
    folds: dict[str, BoundFold] = {fold.fold_name or "single": fold for fold in bound.folds}
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
    parameters = dict(DEFAULT_PARAMETERS.get(method, {}) if parameters is None else parameters)
    predictions = pd.read_parquet(run_path / PREDICTIONS)
    predictions = predictions[predictions["model"] == model]
    environment = environment or capture_environment_record()
    tags = _tags(tags)
    job = ResolvedJob(
        plan=plan,
        workspace=workspace,
        job_name=explain_job,
        environment=environment,
        resolved_config={
            "tags": tags,
            "source_run_id": run_id,
            "method": method,
            "parameters": parameters,
            "max_per_fold": max_per_fold,
            "background_size": background_size,
        },
        dataset_snapshot_id=bound.snapshot.snapshot_id,
        model_selections=selections,
        seeds={"explanation": seed},
    )
    state: dict[str, Any] = {}

    def operation(context: JobExecutionContext) -> JobOperationResult[None]:
        rng = np.random.default_rng(seed)
        atomic_write_json(context.output_path(TAGS), tags)
        molecules, importance, index_folds = [], [], []
        channels = coordinates = channel_roles = None
        positive = None
        for number, record in enumerate(records):
            fold_name = record["fold"]
            context.advance_phase(f"explain:{fold_name}")
            fold = folds[fold_name]
            fitted = _load_model(workspace, record["model_id"], record["backend"], device)
            label_schema = fitted.label_schema
            positive = label_schema.positive_class or label_schema.class_order[-1]
            target_id = list(label_schema.class_order).index(positive)
            classes = {item.molecule_uid: item.class_id for item in bound.snapshot.observations}
            roles = dict(fold.resolution.assignments)
            held_out = [uid for uid, role in roles.items() if role == "test"]
            chosen = _sample(held_out, classes, max_per_fold, rng)
            data = _rows(fold.dataset, "test", chosen)
            background = None
            if contract.baseline_policy == "required" or (
                method == "TreeSHAP" and parameters.get("feature_perturbation") == "interventional"
            ):
                training = [uid for uid, role in roles.items() if role == "train"]
                picked = _sample(training, classes, background_size, rng)
                background = sample_training_background(
                    _rows(fold.dataset, "train", picked),
                    fitted.input_schema,
                    dataset_snapshot_id=fitted.dataset_snapshot_id,
                    split_id=fitted.split_id,
                    max_observations=background_size,
                    random_seed=seed,
                )
            mask_kinds = (
                ()
                if record["backend"] == "sklearn"
                else tuple(mask.kind for mask in fitted.input_schema.masks)
            )
            request = InterpretabilityRequest.create(
                method=method,
                model_id=record["model_id"],
                dataset_snapshot_id=fitted.dataset_snapshot_id,
                input_schema_hash=fitted.input_schema.schema_hash,
                split_role="test",
                cohort="held_out",
                observation_uids=data.molecule_uids,
                target=ExplanationTarget(
                    output_name=f"{positive}_probability", class_id=target_id, class_name=positive
                ),
                baseline=None if background is None else background.to_baseline(),
                mask_policy=ExplanationMaskPolicy.create(
                    mask_kinds=mask_kinds,
                    handling=(
                        "validity is represented by fitted transformed indicator features"
                        if record["backend"] == "sklearn"
                        else "forward masks through the model and zero invalid input attributions"
                    ),
                ),
                decision=ExplanationDecisionProvenance("fixed"),
                parameters=parameters,
                random_seed=seed,
            )
            result = explain_partition_model(fitted, data, request, background=background)
            channels = [channel.name for channel in fitted.input_schema.channels]
            channel_roles = [channel.biological_role for channel in fitted.input_schema.channels]
            coordinates = [int(value) for value in fitted.transform.coordinates]
            matrix = _position_matrix(result, fitted)
            entry = {
                "fold": fold_name,
                "model_id": record["model_id"],
                "n": len(data.molecule_uids),
                "file": None,
                "result_id": result.result_id,
            }
            truth = np.asarray(data.labels) if data.labels is not None else None
            if matrix is not None:
                entry["file"] = f"attributions/fold_{number:02d}.npy"
                np.save(context.output_path(entry["file"]), matrix)
                # The explained inputs (NaN where unobserved), so the record
                # draws its figures without re-reading the data (`MLR-04`).
                entry["inputs"] = f"attributions/inputs_{number:02d}.npy"
                observed = np.where(
                    np.asarray(data.observed_mask, dtype=bool),
                    np.asarray(data.values, dtype=np.float32),
                    np.float32(np.nan),
                )
                np.save(context.output_path(entry["inputs"]), np.transpose(observed, (0, 2, 1)))
                statistics = {
                    "mean_abs": np.abs(matrix).mean(axis=0),
                    "mean": matrix.mean(axis=0),
                }
                if truth is not None:
                    for class_id, class_name in enumerate(label_schema.class_order):
                        rows = truth == class_id
                        if rows.any():
                            statistics[f"mean_in_{class_name}"] = matrix[rows].mean(axis=0)
            else:
                vector = _global_vector(result, fitted)
                statistics = {"value": vector, "abs_value": np.abs(vector)}
            importance.append(_importance_rows(fold_name, channels, coordinates, statistics))
            index_folds.append(entry)
            scores = predictions[predictions["fold"] == fold_name].set_index("molecule_uid")
            molecules.append(
                pd.DataFrame(
                    {
                        "fold": fold_name,
                        "model_id": record["model_id"],
                        "row": np.arange(len(data.molecule_uids)),
                        "molecule_uid": list(data.molecule_uids),
                        "experiment_uid": list(data.experiment_uids),
                        "truth": (
                            None if truth is None else [label_schema.class_order[i] for i in truth]
                        ),
                        "score": scores[f"p_{positive}"]
                        .reindex(list(data.molecule_uids))
                        .to_numpy(),
                    }
                )
            )
        context.advance_phase("record")
        atomic_write_json(
            context.output_path(REQUEST),
            {
                "source_run_id": run_id,
                "model": model,
                "method": method,
                "method_version": contract.version,
                "parameters": parameters,
                "target_class": positive,
                "max_per_fold": max_per_fold,
                "background_size": background_size,
                "seed": seed,
                "sampling": "class-stratified, seeded, per fold, held-out molecules",
            },
        )
        pd.concat(molecules, ignore_index=True).to_parquet(
            context.output_path(MOLECULES), index=False
        )
        per_molecule = any(entry["file"] for entry in index_folds)
        atomic_write_json(
            context.output_path(ATTRIBUTION_INDEX),
            {
                "channels": channels,
                "channel_roles": channel_roles,
                "coordinates": coordinates,
                "axes": ["molecule", "channel", "position"],
                "value": (
                    f"contribution to {positive}"
                    + (
                        " (sum over each position's transformed features)"
                        if records[0]["backend"] == "sklearn"
                        else ""
                    )
                ),
                "per_molecule": per_molecule,
                "folds": index_folds,
            },
        )
        importance_frame = pd.concat(importance, ignore_index=True)
        importance_frame.to_parquet(context.output_path(IMPORTANCE), index=False)
        statistic = "mean_abs" if per_molecule else "abs_value"
        consistency = _consistency(importance_frame, statistic)
        consistency.to_parquet(context.output_path(CONSISTENCY), index=False)
        overall = (
            importance_frame[importance_frame["statistic"] == statistic]
            .groupby(["channel", "coordinate"])["value"]
            .mean()
            .sort_values(ascending=False)
        )
        summary = {
            model: {
                "method": method,
                "n_folds": len(index_folds),
                "n_molecules": int(sum(entry["n"] for entry in index_folds)),
                "importance_statistic": statistic,
                "fold_consistency_spearman": (
                    None
                    if consistency["spearman"].dropna().empty
                    else float(consistency["spearman"].dropna().mean())
                ),
                "top_positions": [
                    {"channel": channel, "coordinate": int(coordinate), "value": float(value)}
                    for (channel, coordinate), value in overall.head(20).items()
                ],
            }
        }
        atomic_write_json(context.output_path(SUMMARY), summary)
        state["summary"] = summary
        artifacts = [
            JobArtifact("tags", TAGS, "application/json"),
            JobArtifact("request", REQUEST, "application/json"),
            JobArtifact("molecules", MOLECULES, "application/vnd.apache.parquet"),
            JobArtifact("attribution_index", ATTRIBUTION_INDEX, "application/json"),
            JobArtifact("importance", IMPORTANCE, "application/vnd.apache.parquet"),
            JobArtifact("consistency", CONSISTENCY, "application/vnd.apache.parquet"),
            JobArtifact("summary", SUMMARY, "application/json"),
        ]
        artifacts += [
            JobArtifact(f"{kind}:{entry['fold']}", entry[key], "application/x-npy")
            for entry in index_folds
            if entry["file"]
            for kind, key in (("attributions", "file"), ("inputs", "inputs"))
        ]
        if figure and per_molecule:
            # Imported here: the plotting module selects a non-interactive backend.
            from smftools.analysis.plot.ml_results import plot_attribution_clustermap

            context.advance_phase("figure")
            frames, matrices, inputs = [], [], []
            for entry in index_folds:
                frame = pd.read_parquet(context.output_path(MOLECULES))
                frames.append(frame[frame["fold"] == entry["fold"]].sort_values("row"))
                matrices.append(np.load(context.output_path(entry["file"])))
                inputs.append(np.load(context.output_path(entry["inputs"])))
            plot_attribution_clustermap(
                pd.concat(frames, ignore_index=True),
                np.concatenate(matrices),
                inputs=np.concatenate(inputs),
                channels=channels,
                coordinates=coordinates,
                positive_class=positive,
                order="label",
                seed=seed,
                title=f"{model}: {method}, out-of-fold ({run_id[:8]})",
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
        method=method,
        summary=state["summary"],
        outcome=outcome,
    )


def plot_explanation(
    explained: PublishedExplanationRun, output_path: str | Path, **options: Any
) -> dict[str, Any]:
    """An attribution clustermap from a published explanation record: every
    fold's molecules, inputs beside attributions. ``options`` go to
    `plot_attribution_clustermap` (``order``, ``bins`` aligned with the pooled
    molecules -- see `PublishedExplanationRun.pooled` -- ``coordinate_labels``,
    ``extra_panels``, ``extra_strips``, ``max_rows``, ...)."""
    from smftools.analysis.plot.ml_results import plot_attribution_clustermap

    molecules, matrices, inputs = explained.pooled()
    index = explained.read(ATTRIBUTION_INDEX)
    request = explained.read(REQUEST)
    options.setdefault("positive_class", request["target_class"])
    options.setdefault(
        "channel_roles", index.get("channel_roles") or _source_roles(explained, index)
    )
    options.setdefault(
        "title",
        f"{explained.model}: {explained.method}, out-of-fold ({explained.source_run_id[:8]})",
    )
    return plot_attribution_clustermap(
        molecules,
        matrices,
        inputs=inputs,
        channels=index["channels"],
        coordinates=index["coordinates"],
        output_path=output_path,
        **options,
    )


def _source_roles(explained: PublishedExplanationRun, index: Mapping[str, Any]) -> list | None:
    """Channel roles from the source run's plan (records made before roles
    were stored in the attribution index)."""
    try:
        path = explained.workspace.runs_root / explained.source_run_id
        plan = parse_ml_plan(json.loads((path / "resolved_plan.json").read_text()))
        job = json.loads((path / "run_manifest.json").read_text())["job_name"]
        dataset = plan.datasets[plan.jobs[job].dataset]
        roles = {channel.name: channel.biological_role for channel in dataset.channels}
        return [roles.get(name) for name in index["channels"]]
    except (OSError, KeyError, ValueError):
        return None
