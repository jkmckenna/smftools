"""Deterministic plain-PyTorch training for registered model families."""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from smftools.optional_imports import require

from ..contracts import InputSchema, LabelSchema
from ..data.balancing import (
    BalanceResolution,
    resolve_role_balance,
    resolve_role_balance_from_plan,
)
from ..data.materialized_dataset import MLDatasetProtocol
from ..data.streaming_transforms import fit_feature_transform_streaming, plan_transform_fit
from ..data.transforms import (
    FeatureTransformSpec,
    FittedFeatureTransform,
    TorchFeatureTransform,
    TorchTransformedBatch,
    fit_feature_transform,
)
from ..models.protocols import TorchPredictor
from ..models.registry import BUILTIN_MODEL_REGISTRY, ModelRegistry, ResolvedModelDefinition
from ..plan import BalancingSpec

# Version 2 adds ``shuffle_buffer_batches``. It belongs in the persisted config
# rather than a call argument because it changes fitted weights: a streaming fit
# permutes rows within a window of decoded batches, so two runs with identical
# recorded provenance but different buffer sizes produce different models. A
# reproducibility record that does not determine the result is not one.
TORCH_TRAINING_CONFIG_VERSION = 2
_SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"})


class TorchTrainingError(ValueError):
    """Raised when plain-Torch training violates a resolved ML contract."""


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TorchTrainingError(f"{path} must be a positive integer")
    return value


def _nonnegative_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TorchTrainingError(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise TorchTrainingError(f"{path} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class TorchTrainingConfig:
    """Validated optimizer, early-stopping, device, and reproducibility policy."""

    schema_version: int = TORCH_TRAINING_CONFIG_VERSION
    max_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5
    min_delta: float = 0.0
    seed: int = 0
    device: str = "auto"
    deterministic: bool = True
    # Streaming fits only. Rows are permuted within this many decoded partition
    # batches; the materialized path shuffles globally and ignores it. Recorded
    # here because it changes fitted weights.
    shuffle_buffer_batches: int = 8

    def __post_init__(self) -> None:
        if self.schema_version != TORCH_TRAINING_CONFIG_VERSION:
            raise TorchTrainingError(
                f"unsupported Torch training config version {self.schema_version}; "
                f"expected {TORCH_TRAINING_CONFIG_VERSION}"
            )
        for name in ("max_epochs", "batch_size", "patience", "shuffle_buffer_batches"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise TorchTrainingError("seed must be a non-negative integer")
        learning_rate = _nonnegative_number(self.learning_rate, "learning_rate")
        if learning_rate == 0:
            raise TorchTrainingError("learning_rate must be positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(
            self,
            "weight_decay",
            _nonnegative_number(self.weight_decay, "weight_decay"),
        )
        object.__setattr__(self, "min_delta", _nonnegative_number(self.min_delta, "min_delta"))
        device = str(self.device).strip().lower()
        if device not in _SUPPORTED_DEVICES:
            raise TorchTrainingError(f"device must be one of {sorted(_SUPPORTED_DEVICES)}")
        if not isinstance(self.deterministic, bool):
            raise TorchTrainingError("deterministic must be boolean")
        object.__setattr__(self, "device", device)

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible training parameters."""
        return {
            "schema_version": self.schema_version,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "seed": self.seed,
            "device": self.device,
            "deterministic": self.deterministic,
            "shuffle_buffer_batches": self.shuffle_buffer_batches,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TorchTrainingConfig:
        """Strictly validate and restore a training configuration."""
        expected = {
            "schema_version",
            "max_epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "min_delta",
            "seed",
            "device",
            "deterministic",
            "shuffle_buffer_batches",
        }
        if set(raw) != expected:
            raise TorchTrainingError(f"Torch training fields must be exactly {sorted(expected)}")
        return cls(**{name: raw[name] for name in expected})


@dataclass(frozen=True)
class ClassificationTask:
    """Classification loss policy kept separate from model architecture."""

    label_schema: LabelSchema
    output_dim: int

    def __post_init__(self) -> None:
        n_classes = len(self.label_schema.class_order)
        if self.output_dim != n_classes and not (n_classes == 2 and self.output_dim == 1):
            raise TorchTrainingError(
                f"model output_dim {self.output_dim} is incompatible with {n_classes} classes"
            )

    @property
    def binary_logit(self) -> bool:
        """Return whether the task uses one binary class-1 logit."""
        return self.output_dim == 1

    def losses(self, logits: Any, labels: Any) -> Any:
        """Return one unreduced supervised loss per observation."""
        torch_functional = require(
            "torch.nn.functional",
            extra="ml-base",
            purpose="plain Torch classification losses",
        )
        if self.binary_logit:
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise TorchTrainingError("binary Torch logits must have shape (batch, 1)")
            return torch_functional.binary_cross_entropy_with_logits(
                logits[:, 0],
                labels.to(dtype=logits.dtype),
                reduction="none",
            )
        if logits.ndim != 2 or logits.shape[1] != self.output_dim:
            raise TorchTrainingError(
                f"multiclass Torch logits must have shape (batch, {self.output_dim})"
            )
        return torch_functional.cross_entropy(logits, labels, reduction="none")


@dataclass(frozen=True)
class TorchEpochRecord:
    """One tidy optimization epoch with train and validation losses."""

    epoch: int
    train_loss: float
    validation_loss: float

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch <= 0:
            raise TorchTrainingError("history epoch must be a positive integer")
        for name in ("train_loss", "validation_loss"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise TorchTrainingError(f"history {name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a tidy JSON-compatible epoch row."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TorchEpochRecord:
        """Strictly restore one epoch row."""
        expected = {"epoch", "train_loss", "validation_loss"}
        if set(raw) != expected:
            raise TorchTrainingError(f"history fields must be exactly {sorted(expected)}")
        return cls(
            epoch=raw["epoch"],
            train_loss=raw["train_loss"],
            validation_loss=raw["validation_loss"],
        )


@dataclass(frozen=True)
class FittedTorchModel:
    """Reusable plain Torch module plus immutable preprocessing and fit provenance."""

    family: str
    architecture: ResolvedModelDefinition
    model: Any
    transform: FittedFeatureTransform
    input_schema: InputSchema
    label_schema: LabelSchema
    dataset_snapshot_id: str
    split_id: str
    training_config: TorchTrainingConfig
    resolved_device: str
    best_epoch: int
    history: tuple[TorchEpochRecord, ...]
    validation_loss: float
    test_loss: float

    def __post_init__(self) -> None:
        if self.architecture.backend != "torch" or self.family != self.architecture.family:
            raise TorchTrainingError("fitted model must use its resolved Torch family")
        if self.transform.dataset_snapshot_id != self.dataset_snapshot_id:
            raise TorchTrainingError("transform dataset snapshot differs from fitted model")
        if self.transform.split_id != self.split_id:
            raise TorchTrainingError("transform split differs from fitted model")
        history = tuple(self.history)
        if not history or self.best_epoch not in {row.epoch for row in history}:
            raise TorchTrainingError("best_epoch must identify one recorded epoch")
        for name in ("validation_loss", "test_loss"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise TorchTrainingError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        output_dim = getattr(self.architecture.config, "output_dim", None)
        if output_dim is None:
            raise TorchTrainingError("Torch architecture config must declare output_dim")
        ClassificationTask(self.label_schema, output_dim)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "resolved_device", str(self.resolved_device))

    @property
    def predictor(self) -> TorchPredictor:
        """Return the backend-neutral adapter for channel-first transformed signals."""
        return TorchPredictor(
            model=self.model,
            input_schema=self.input_schema,
            label_schema=self.label_schema,
            capabilities=self.architecture.capabilities,
            device=self.resolved_device,
        )


@dataclass(frozen=True)
class TorchTrainingResult:
    """Fitted model and auditable train-role balancing outcome."""

    model: FittedTorchModel
    balance: BalanceResolution
    n_training_observations: int
    class_counts: tuple[int, ...]
    stopped_early: bool


def _resolve_device(requested: str) -> str:
    torch = require("torch", extra="ml-base", purpose="plain Torch device selection")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise TorchTrainingError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise TorchTrainingError("MPS was requested but is unavailable")
    return requested


@contextmanager
def _seeded_execution(config: TorchTrainingConfig):
    torch = require("torch", extra="ml-base", purpose="deterministic plain Torch training")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _loader(
    transformed: TorchTransformedBatch,
    *,
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
    sampler: Any = None,
):
    torch = require("torch", extra="ml-base", purpose="plain Torch data loading")
    torch_data = require("torch.utils.data", extra="ml-base", purpose="plain Torch data loading")
    if transformed.labels is None:
        raise TorchTrainingError("Torch classification batches require labels")
    selected = torch.tensor(indices, dtype=torch.long, device=transformed.values.device)
    design = transformed.design_mask
    if design.ndim == 2:
        design = design.unsqueeze(0).expand(len(transformed.values), -1, -1)
    tensors = (
        transformed.values.index_select(0, selected),
        transformed.labels.index_select(0, selected),
        transformed.observed_mask.index_select(0, selected),
        transformed.availability_mask.index_select(0, selected),
        design.index_select(0, selected),
        transformed.padding_mask.index_select(0, selected),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch_data.DataLoader(
        torch_data.TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=generator,
    )


def _batch_loss(model: Any, task: ClassificationTask, batch: tuple[Any, ...]) -> Any:
    values, labels, observed, availability, design, padding = batch
    logits = model(
        values,
        observed_mask=observed,
        availability_mask=availability,
        design_mask=design,
        padding_mask=padding,
    )
    return task.losses(logits, labels), labels


def _evaluate(model: Any, task: ClassificationTask, loader: Any) -> float:
    torch = require("torch", extra="ml-base", purpose="plain Torch evaluation")
    was_training = bool(model.training)
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                losses, _labels = _batch_loss(model, task, batch)
                total += float(losses.sum().item())
                count += int(losses.numel())
    finally:
        model.train(was_training)
    if count == 0:
        raise TorchTrainingError("evaluation loader produced no observations")
    return total / count


class _StreamingLoader:
    """Re-iterable loader that decodes partition batches instead of a tensor.

    Rebuilds its generator on every ``__iter__``, so it substitutes directly for
    the ``DataLoader`` the materialized path uses, including inside
    :func:`_evaluate`.

    Shuffling is **buffered, not global**. Random access over a partition store
    is what streaming exists to avoid, so training rows are permuted within a
    window of ``buffer_batches`` decoded batches rather than across the whole
    split. This changes SGD sample ordering and therefore the fitted weights; it
    is a genuine semantic difference from the materialized path, not an
    implementation detail. Widen the buffer to approach global shuffling at the
    cost of holding more rows.
    """

    def __init__(
        self,
        dataset: MLDatasetProtocol,
        transform: TorchFeatureTransform,
        split: str,
        *,
        batch_size: int,
        multiplicity: np.ndarray | None = None,
        shuffle_seed: int | None = None,
        buffer_batches: int = 8,
    ) -> None:
        self._dataset = dataset
        self._transform = transform
        self._split = split
        self._batch_size = batch_size
        self._multiplicity = multiplicity
        self._shuffle_seed = shuffle_seed
        self._buffer_batches = max(1, int(buffer_batches))
        self.epoch = 0

    def _decoded(self):
        torch = require("torch", extra="ml-base", purpose="plain Torch streaming")
        offset = 0
        for batch in self._dataset.iter_batches(self._split):
            rows = len(batch.molecule_uids)
            local = None
            if self._multiplicity is not None:
                counts = self._multiplicity[offset : offset + rows]
                offset += rows
                if not counts.any():
                    continue
                local = np.repeat(np.flatnonzero(counts), counts[counts > 0])
            else:
                offset += rows
            transformed = self._transform(batch)
            if transformed.labels is None:
                raise TorchTrainingError("Torch classification batches require labels")
            design = transformed.design_mask
            if design.ndim == 2:
                design = design.unsqueeze(0).expand(rows, -1, -1)
            tensors = (
                transformed.values,
                transformed.labels,
                transformed.observed_mask,
                transformed.availability_mask,
                design,
                transformed.padding_mask,
            )
            if local is not None:
                selected = torch.as_tensor(local, dtype=torch.long, device=tensors[0].device)
                tensors = tuple(tensor.index_select(0, selected) for tensor in tensors)
            yield tensors

    def __iter__(self):
        torch = require("torch", extra="ml-base", purpose="plain Torch streaming")

        def drain(chunks):
            if not chunks:
                return
            merged = tuple(torch.cat(parts, dim=0) for parts in zip(*chunks, strict=True))
            total = merged[0].shape[0]
            if self._shuffle_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(self._shuffle_seed + self.epoch)
                order = torch.randperm(total, generator=generator).to(merged[0].device)
                merged = tuple(tensor.index_select(0, order) for tensor in merged)
            for start in range(0, total, self._batch_size):
                stop = start + self._batch_size
                yield tuple(tensor[start:stop] for tensor in merged)

        buffer: list[tuple[Any, ...]] = []
        for tensors in self._decoded():
            buffer.append(tensors)
            if len(buffer) >= self._buffer_batches:
                yield from drain(buffer)
                buffer = []
        yield from drain(buffer)


def fit_torch_partition_model_streaming(
    dataset: MLDatasetProtocol,
    resolved_model: ResolvedModelDefinition,
    *,
    training_config: TorchTrainingConfig | None = None,
    transform_spec: FeatureTransformSpec | None = None,
    balancing: BalancingSpec | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> TorchTrainingResult:
    """Train a plain-Torch model without materializing any split.

    :func:`fit_torch_partition_model` materializes ``train``, ``validation``,
    and ``test``, so it is bounded by ``max_materialization_bytes`` three times
    over and cannot reach a production-scale experiment. This path decodes
    partition batches instead.

    Two differences from the materialized path are deliberate and affect
    results, so they are stated rather than buried:

    1. **Shuffling is buffered, not global** (see :class:`_StreamingLoader`).
       Fitted weights will not match the materialized path even at identical
       seeds. Weight parity is therefore asserted nowhere; what is asserted is
       that the locked-test contract, balance provenance, and transform
       identity are preserved.
    2. **Every epoch re-reads the store.** In-memory training pays one decode
       for all epochs; streaming pays one decode per epoch. It trades wall time
       for the ability to run at all.

    ``weighted_sampler`` balancing is refused: it draws with replacement across
    the whole split, which needs the random access streaming avoids.
    """
    if resolved_model.backend != "torch":
        raise TorchTrainingError("resolved_model must use the Torch backend")
    config = training_config or TorchTrainingConfig()
    input_schema = dataset.plan.dataset.input_schema
    label_schema = dataset.plan.dataset.label_schema
    if label_schema is None:
        raise TorchTrainingError("Torch classification requires a label schema")
    output_dim = getattr(resolved_model.config, "output_dim", None)
    task = ClassificationTask(label_schema, output_dim)
    device = _resolve_device(config.device)
    resolved_transform_spec = transform_spec or FeatureTransformSpec(indicators=())
    if resolved_transform_spec.indicators:
        raise TorchTrainingError(
            "Torch signal transforms cannot append mask indicators; masks remain separate inputs"
        )
    # Refuses non-streamable specs (imputation="median") before any read.
    plan_transform_fit(resolved_transform_spec)

    balance = resolve_role_balance_from_plan(
        dataset.plan,
        label_schema,
        balancing or BalancingSpec(),
        role="train",
        seed=config.seed,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
    )
    if balance.method == "weighted_sampler":
        raise TorchTrainingError(
            "weighted_sampler balancing cannot be streamed: it samples with replacement across "
            "the whole train split, which requires random access. Use class_weight for the same "
            "reweighting without global sampling, or downsample/upsample."
        )

    entries = dataset.plan.entries_for("train")
    fitted_transform = fit_feature_transform_streaming(
        dataset,
        resolved_transform_spec,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
        coordinates=tuple(int(value) for value in dataset.plan.coordinates),
        channel_names=tuple(channel.name for channel in input_schema.channels),
        molecule_uids=tuple(entry.molecule_uid for entry in entries),
    )

    torch = require("torch", extra="ml-base", purpose="plain Torch training")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    transform = TorchFeatureTransform(fitted_transform, device=device)

    multiplicity = np.bincount(balance.selected_indices, minlength=len(entries))
    train_loader = _StreamingLoader(
        dataset,
        transform,
        "train",
        batch_size=config.batch_size,
        multiplicity=multiplicity,
        shuffle_seed=config.seed,
        buffer_batches=config.shuffle_buffer_batches,
    )
    validation_loader = _StreamingLoader(
        dataset,
        transform,
        "validation",
        batch_size=config.batch_size,
    )

    model = registry.build(resolved_model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = (
        None
        if balance.method != "class_weight" or balance.class_weights is None
        else torch.tensor(balance.class_weights, dtype=torch.float32, device=device)
    )
    history: list[TorchEpochRecord] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0

    with _seeded_execution(config):
        for epoch in range(1, config.max_epochs + 1):
            train_loader.epoch = epoch
            model.train()
            train_numerator = 0.0
            train_denominator = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                losses, labels = _batch_loss(model, task, batch)
                weights = None if class_weights is None else class_weights[labels]
                numerator = losses.sum() if weights is None else (losses * weights).sum()
                denominator = (
                    float(losses.numel()) if weights is None else float(weights.sum().item())
                )
                loss = numerator / denominator
                loss.backward()
                optimizer.step()
                train_numerator += float(numerator.detach().item())
                train_denominator += denominator
            if train_denominator == 0:
                raise TorchTrainingError("streaming train loader produced no observations")
            validation_loss = _evaluate(model, task, validation_loader)
            history.append(
                TorchEpochRecord(
                    epoch=epoch,
                    train_loss=train_numerator / train_denominator,
                    validation_loss=validation_loss,
                )
            )
            if validation_loss < best_loss - config.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break

    if best_state is None:
        raise TorchTrainingError("training did not produce a finite validation state")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    validation_loss = _evaluate(model, task, validation_loader)

    # The locked test role stays unread until early stopping has selected and
    # restored the best validation state, exactly as in the materialized path.
    test_loader = _StreamingLoader(
        dataset,
        transform,
        "test",
        batch_size=config.batch_size,
    )
    test_loss = _evaluate(model, task, test_loader)
    fitted = FittedTorchModel(
        family=resolved_model.family,
        architecture=resolved_model,
        model=model,
        transform=fitted_transform,
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
        training_config=config,
        resolved_device=device,
        best_epoch=best_epoch,
        history=tuple(history),
        validation_loss=validation_loss,
        test_loss=test_loss,
    )
    return TorchTrainingResult(
        model=fitted,
        balance=balance,
        n_training_observations=len(balance.selected_indices),
        class_counts=balance.result_counts,
        stopped_early=len(history) < config.max_epochs,
    )


def fit_torch_partition_model(
    dataset: MLDatasetProtocol,
    resolved_model: ResolvedModelDefinition,
    *,
    training_config: TorchTrainingConfig | None = None,
    transform_spec: FeatureTransformSpec | None = None,
    balancing: BalancingSpec | None = None,
    registry: ModelRegistry = BUILTIN_MODEL_REGISTRY,
) -> TorchTrainingResult:
    """Fit one registered plain Torch model using immutable split roles.

    Train and validation roles are materialized only through the partition
    dataset's conservative memory preflight. The locked test role is not read
    until early stopping has selected and restored the best validation state.
    Optimizer state is intentionally not persisted by this inference-only MVP.
    """
    if resolved_model.backend != "torch":
        raise TorchTrainingError("resolved_model must use the Torch backend")
    config = training_config or TorchTrainingConfig()
    input_schema = dataset.plan.dataset.input_schema
    label_schema = dataset.plan.dataset.label_schema
    if label_schema is None:
        raise TorchTrainingError("Torch classification requires a label schema")
    output_dim = getattr(resolved_model.config, "output_dim", None)
    task = ClassificationTask(label_schema, output_dim)
    device = _resolve_device(config.device)
    resolved_transform_spec = transform_spec or FeatureTransformSpec(indicators=())
    if resolved_transform_spec.indicators:
        raise TorchTrainingError(
            "Torch signal transforms cannot append mask indicators; masks remain separate inputs"
        )

    train = dataset.materialize("train")
    validation = dataset.materialize("validation")
    if train.labels is None or validation.labels is None:
        raise TorchTrainingError("training and validation roles must contain labels")
    fitted_transform = fit_feature_transform(
        train,
        resolved_transform_spec,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
    )
    balance = resolve_role_balance(
        train,
        label_schema,
        balancing or BalancingSpec(),
        seed=config.seed,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
    )

    torch = require("torch", extra="ml-base", purpose="plain Torch training")
    # Seed before model construction so parameter initialization is part of the
    # persisted reproducibility contract. The execution context reseeds before
    # data-loader iteration and optimization.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    transform = TorchFeatureTransform(fitted_transform, device=device)
    transformed_train = transform(train)
    transformed_validation = transform(validation)
    sampler = balance.torch_weighted_sampler() if balance.method == "weighted_sampler" else None
    train_loader = _loader(
        transformed_train,
        indices=balance.selected_indices,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=balance.method != "weighted_sampler",
        sampler=sampler,
    )
    validation_loader = _loader(
        transformed_validation,
        indices=np.arange(len(validation.labels), dtype=np.int64),
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )

    model = registry.build(resolved_model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = (
        None
        if balance.method != "class_weight" or balance.class_weights is None
        else torch.tensor(balance.class_weights, dtype=torch.float32, device=device)
    )
    history: list[TorchEpochRecord] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0

    with _seeded_execution(config):
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            train_numerator = 0.0
            train_denominator = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                losses, labels = _batch_loss(model, task, batch)
                weights = None if class_weights is None else class_weights[labels]
                numerator = losses.sum() if weights is None else (losses * weights).sum()
                denominator = (
                    float(losses.numel()) if weights is None else float(weights.sum().item())
                )
                loss = numerator / denominator
                loss.backward()
                optimizer.step()
                train_numerator += float(numerator.detach().item())
                train_denominator += denominator
            if train_denominator == 0:
                raise TorchTrainingError("training loader produced no observations")
            validation_loss = _evaluate(model, task, validation_loader)
            history.append(
                TorchEpochRecord(
                    epoch=epoch,
                    train_loss=train_numerator / train_denominator,
                    validation_loss=validation_loss,
                )
            )
            if validation_loss < best_loss - config.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break

    if best_state is None:
        raise TorchTrainingError("training did not produce a finite validation state")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    validation_loss = _evaluate(model, task, validation_loader)

    test = dataset.materialize("test")
    if test.labels is None:
        raise TorchTrainingError("locked test role must contain labels")
    transformed_test = transform(test)
    test_loader = _loader(
        transformed_test,
        indices=np.arange(len(test.labels), dtype=np.int64),
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    test_loss = _evaluate(model, task, test_loader)
    fitted = FittedTorchModel(
        family=resolved_model.family,
        architecture=resolved_model,
        model=model,
        transform=fitted_transform,
        input_schema=input_schema,
        label_schema=label_schema,
        dataset_snapshot_id=dataset.plan.dataset.snapshot_id,
        split_id=dataset.plan.split.split_id,
        training_config=config,
        resolved_device=device,
        best_epoch=best_epoch,
        history=tuple(history),
        validation_loss=validation_loss,
        test_loss=test_loss,
    )
    return TorchTrainingResult(
        model=fitted,
        balance=balance,
        n_training_observations=len(balance.selected_indices),
        class_counts=balance.result_counts,
        stopped_early=len(history) < config.max_epochs,
    )
