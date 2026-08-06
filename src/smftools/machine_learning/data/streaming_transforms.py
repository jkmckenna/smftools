"""Fit train-only feature transforms from streamed batches (ML-204).

:func:`~smftools.machine_learning.data.transforms.fit_feature_transform` requires
a fully materialized train split, which is the reason no training path in the
package can consume a production-scale experiment: the materialization preflight
refuses long before the fit is reached.

Applying a fitted transform already streams -- ``FittedFeatureTransform.transform``
and ``_raw_feature_rows`` both accept an ``MLPartitionBatch``. Only fitting needed
a bounded implementation, and that is what this module provides.

How many passes a fit costs depends entirely on the spec, and the default spec
costs none::

    imputation / scaling                      passes  why
    ----------------------------------------  ------  ----------------------------------
    constant / none                                0  both statistics are declared
    constant / standard                            1  scaling needs imputed moments
    mean or most_frequent / none                   1  fill values need column statistics
    mean or most_frequent / standard               2  fill values precede imputation
    median / any                             refused  exact median needs the full column

Because ``FeatureTransformSpec`` defaults to ``imputation="constant"`` and
``scaling="none"``, the common case resolves the entire transform from read-plan
metadata without decoding a single batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .transforms import (
    ML_FEATURE_TRANSFORM_VERSION,
    FeatureTransformSpec,
    FittedFeatureTransform,
    MLTransformError,
    _digest,
    _feature_names,
    _identity_digest_payload,
    _molecule_digest,
    _raw_feature_rows,
    _sha256,
)

# Mode imputation accumulates per-column value counts. SMF calls are ``{0, 1}``,
# so this bound is generous; refusing above it keeps the accumulator bounded
# rather than letting a continuous layer silently allocate per distinct value.
MAX_MODE_CARDINALITY = 64


class StreamingBatchSource(Protocol):
    """Anything that can re-yield one split's batches in canonical order."""

    def iter_batches(self, split: str) -> Iterator[Any]: ...


@dataclass(frozen=True)
class TransformFitPlan:
    """How many data passes a spec's fit requires, and why."""

    passes: int
    needs_fill_pass: bool
    needs_scaling_pass: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "needs_fill_pass": self.needs_fill_pass,
            "needs_scaling_pass": self.needs_scaling_pass,
            "rationale": self.rationale,
        }


def plan_transform_fit(spec: FeatureTransformSpec) -> TransformFitPlan:
    """Decide the pass count for ``spec`` before any data is read.

    Refuses ``imputation="median"``: an exact median needs every value for a
    column simultaneously, which is the whole cost streaming exists to avoid.
    Silently substituting a near-median statistic would change fitted models
    with no signal to the caller, so this raises instead.
    """
    if spec.imputation == "median":
        raise MLTransformError(
            "imputation='median' cannot be fitted from streamed batches: an exact median "
            "requires the full column distribution in memory. Use imputation='mean' or "
            "imputation='most_frequent' for a streaming fit, or materialize the train split if the "
            "median is scientifically required and the split fits within "
            "max_materialization_bytes."
        )

    needs_fill_pass = spec.imputation in {"mean", "most_frequent"}
    needs_scaling_pass = spec.scaling == "standard"

    if not needs_fill_pass and not needs_scaling_pass:
        rationale = "constant fill and no scaling are declared by the spec; no data is read"
    elif not needs_fill_pass:
        rationale = "fill values are declared; one pass accumulates imputed moments for scaling"
    elif not needs_scaling_pass:
        rationale = "one pass accumulates per-column fill statistics"
    else:
        rationale = (
            "fill values must be known before imputing, so scaling moments need a second pass"
        )

    return TransformFitPlan(
        passes=int(needs_fill_pass) + int(needs_scaling_pass),
        needs_fill_pass=needs_fill_pass,
        needs_scaling_pass=needs_scaling_pass,
        rationale=rationale,
    )


class _FillAccumulator:
    """Per-column statistics for mean or most_frequent fill values."""

    def __init__(self, n_signal: int, method: str) -> None:
        self._method = method
        self._counts = np.zeros(n_signal, dtype=np.int64)
        self._sums = np.zeros(n_signal, dtype=np.float64)
        # value -> per-column occurrence counts, for mode only.
        self._value_counts: dict[float, np.ndarray] = {}
        self._n_signal = n_signal

    def update(self, raw: np.ndarray, valid: np.ndarray) -> None:
        self._counts += valid.sum(axis=0, dtype=np.int64)
        if self._method == "mean":
            self._sums += np.where(valid, raw, 0.0).sum(axis=0, dtype=np.float64)
            return
        observed = np.unique(raw[valid])
        for value in observed:
            key = float(value)
            if key not in self._value_counts:
                if len(self._value_counts) >= MAX_MODE_CARDINALITY:
                    raise MLTransformError(
                        f"imputation='most_frequent' saw more than {MAX_MODE_CARDINALITY} distinct values "
                        "while streaming; the accumulator is bounded on purpose. Use "
                        "imputation='mean' for continuous signal."
                    )
                self._value_counts[key] = np.zeros(self._n_signal, dtype=np.int64)
            self._value_counts[key] += (valid & (raw == value)).sum(axis=0, dtype=np.int64)

    def resolve(self, fill_value: float) -> np.ndarray:
        result = np.full(self._n_signal, float(fill_value), dtype=np.float64)
        seen = self._counts > 0
        if not seen.any():
            return result
        if self._method == "mean":
            result[seen] = self._sums[seen] / self._counts[seen]
            return result
        # Ties resolve to the smallest value, matching np.unique's sorted order
        # in the materialized implementation.
        best_counts = np.zeros(self._n_signal, dtype=np.int64)
        best_values = np.full(self._n_signal, np.nan, dtype=np.float64)
        for value in sorted(self._value_counts):
            counts = self._value_counts[value]
            takes = counts > best_counts
            best_counts = np.where(takes, counts, best_counts)
            best_values = np.where(takes, value, best_values)
        resolved = seen & np.isfinite(best_values)
        result[resolved] = best_values[resolved]
        return result


class _MomentAccumulator:
    """Streaming per-column mean and population variance (Chan's parallel form)."""

    def __init__(self, n_signal: int) -> None:
        self._n = 0
        self._mean = np.zeros(n_signal, dtype=np.float64)
        self._m2 = np.zeros(n_signal, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        rows = values.shape[0]
        if rows == 0:
            return
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0, dtype=np.float64)
        if self._n == 0:
            self._n = rows
            self._mean = batch_mean
            self._m2 = batch_m2
            return
        total = self._n + rows
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (rows / total)
        self._m2 = self._m2 + batch_m2 + (delta**2) * (self._n * rows / total)
        self._n = total

    def resolve(self) -> tuple[np.ndarray, np.ndarray]:
        if self._n == 0:
            raise MLTransformError("cannot fit scaling from an empty train split")
        centers = self._mean.copy()
        scales = np.sqrt(np.maximum(self._m2 / self._n, 0.0))
        scales[scales == 0] = 1.0
        return centers, scales


def fit_feature_transform_streaming(
    source: StreamingBatchSource,
    spec: FeatureTransformSpec,
    *,
    dataset_snapshot_id: str,
    split_id: str,
    coordinates: tuple[int, ...],
    channel_names: tuple[str, ...],
    molecule_uids: tuple[str, ...],
    split: str = "train",
) -> FittedFeatureTransform:
    """Fit a transform from streamed batches, holding only accumulators.

    Produces a :class:`FittedFeatureTransform` byte-identical to the materialized
    fit for the same rows: the returned ``transform_id`` is recomputed and
    validated by the dataclass, so a drift between the two paths surfaces as a
    construction failure rather than a silently different model.
    """
    if split != "train":
        raise MLTransformError("fitted transforms may only be fit on the 'train' role")
    plan = plan_transform_fit(spec)

    resolved_snapshot = _digest(dataset_snapshot_id, "dataset_snapshot_id")
    resolved_split = _digest(split_id, "split_id")
    coordinates = tuple(int(item) for item in coordinates)
    channel_names = tuple(str(item) for item in channel_names)
    n_signal = len(coordinates) * len(channel_names)

    if plan.needs_fill_pass:
        fill_accumulator = _FillAccumulator(n_signal, spec.imputation)
        for batch in source.iter_batches(split):
            raw, _masks, valid = _raw_feature_rows(batch)
            fill_accumulator.update(raw, valid)
        fill_values = fill_accumulator.resolve(spec.fill_value)
    else:
        fill_values = np.full(n_signal, float(spec.fill_value), dtype=np.float64)

    if plan.needs_scaling_pass:
        moments = _MomentAccumulator(n_signal)
        for batch in source.iter_batches(split):
            raw, _masks, valid = _raw_feature_rows(batch)
            imputed = np.where(valid, raw, fill_values[np.newaxis, :]).astype(
                np.float64, copy=False
            )
            moments.update(imputed)
        centers, scales = moments.resolve()
    else:
        centers = np.zeros(n_signal, dtype=np.float64)
        scales = np.ones(n_signal, dtype=np.float64)

    names = _feature_names(coordinates, channel_names, spec.indicators)
    molecule_digest = _molecule_digest(molecule_uids)
    identity = {
        "schema_version": ML_FEATURE_TRANSFORM_VERSION,
        "spec": spec.to_dict(),
        "dataset_snapshot_id": resolved_snapshot,
        "split_id": resolved_split,
        "fit_molecule_digest": molecule_digest,
        "n_positions": len(coordinates),
        "channel_names": list(channel_names),
        "coordinates": list(coordinates),
        "fill_values": fill_values.tolist(),
        "centers": centers.tolist(),
        "scales": scales.tolist(),
        "feature_names": list(names),
    }
    return FittedFeatureTransform(
        schema_version=ML_FEATURE_TRANSFORM_VERSION,
        transform_id=_sha256(_identity_digest_payload(identity)),
        spec=spec,
        dataset_snapshot_id=resolved_snapshot,
        split_id=resolved_split,
        fit_molecule_digest=molecule_digest,
        n_positions=len(coordinates),
        channel_names=channel_names,
        coordinates=coordinates,
        fill_values=fill_values,
        centers=centers,
        scales=scales,
        feature_names=names,
    )
