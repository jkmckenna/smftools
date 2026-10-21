"""HMM feature layers from a stored posterior (`DLS-03`).

One function turns a two-state HMM's posterior into every feature layer the
HMM stage publishes: per feature group (e.g. accessible / footprint), each size
class and the group aggregate, their run lengths, an end-truncated flag, and
the merged versions. The HMM stage calls it on the posterior it is about to
store, and readers call it on the stored posterior, so derived layers equal
the stage's own.

Rules (``FEATURE_DERIVATION_VERSION`` 2):

- a position belongs to the group whose state the posterior favours: the
  posterior's own state when ``p >= threshold``, the other state when
  ``p < 1 - threshold`` (a threshold of 0.5 partitions every position);
  both comparisons are exact on a posterior stored at 1/254 resolution
  (`DLS-02`) for thresholds on that grid;
- runs are found among each read's decoded columns *inside its read span*,
  and measured in reference bp (first to last decoded column). A run that
  reaches the first or last decoded column of the read is end-truncated: its
  true length is unknown, its size class is that of the observed length, and
  ``..._end_truncated`` marks it. (Version 1 measured runs on the posterior
  outside the read too, so read-end runs took lengths the read never saw.)
- feature layers are then masked to the read: NaN outside its span and at
  bases it does not cover; the posterior is not masked at uncovered bases --
  it is the decoder's estimate there, and runs bridge them as decoding did;
- merges join runs of an aggregate across gaps of at most N bp, re-split into
  size classes (``..._merged``), masked the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

FEATURE_DERIVATION_VERSION = 2
END_TRUNCATED_SUFFIX = "_end_truncated"


@dataclass(frozen=True)
class FeatureSettings:
    """Everything feature calling depends on besides the posterior."""

    feature_sets: Mapping[str, Mapping[str, Any]]
    posterior_state: str = "Modified"
    prob_threshold: float = 0.5
    merges: tuple[tuple[str, int], ...] = ()
    merged_suffix: str = "_merged"
    version: int = field(default=FEATURE_DERIVATION_VERSION)

    @classmethod
    def from_config(cls, cfg, feature_groups: Sequence[str] | None = None) -> "FeatureSettings":
        from .HMM import normalize_hmm_feature_sets

        sets = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))
        if feature_groups is not None:
            sets = {name: sets[name] for name in feature_groups if name in sets}
        return cls(
            feature_sets=sets,
            posterior_state=str(getattr(cfg, "hmm_posterior_state", "Modified")),
            prob_threshold=float(getattr(cfg, "hmm_feature_prob_threshold", 0.5)),
            merges=tuple(
                (str(core), int(distance))
                for core, distance in (getattr(cfg, "hmm_merge_layer_features", []) or [])
            ),
            merged_suffix=str(getattr(cfg, "hmm_merged_suffix", "_merged")),
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FeatureSettings":
        """Inverse of :meth:`to_dict` (e.g. from a spine's recorded settings)."""
        return cls(
            feature_sets={
                group: {
                    "state": spec.get("state", "Modified"),
                    "features": {
                        name: (float(lo), float(hi))
                        for name, (lo, hi) in (spec.get("features", {}) or {}).items()
                    },
                }
                for group, spec in (record.get("feature_sets", {}) or {}).items()
            },
            posterior_state=str(record.get("posterior_state", "Modified")),
            prob_threshold=float(record.get("prob_threshold", 0.5)),
            merges=tuple((str(c), int(d)) for c, d in (record.get("merges", []) or [])),
            merged_suffix=str(record.get("merged_suffix", "_merged")),
            version=int(record.get("version", FEATURE_DERIVATION_VERSION)),
        )

    def with_overrides(self, **overrides) -> "FeatureSettings":
        """A copy with some settings replaced (``merges``, ``prob_threshold``,
        ``feature_sets``, ...)."""
        from dataclasses import replace

        if "merges" in overrides:
            overrides["merges"] = tuple((str(c), int(d)) for c, d in overrides["merges"])
        return replace(self, **overrides)

    def posterior_layer(self, prefix: str) -> str:
        state = str(self.posterior_state).strip().lower().replace(" ", "_").replace("-", "_")
        return f"{prefix}_posterior_{state}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record of the settings (generation manifests, analyses)."""
        return {
            "version": int(self.version),
            "posterior_state": self.posterior_state,
            "prob_threshold": float(self.prob_threshold),
            "merges": [[core, int(d)] for core, d in self.merges],
            "merged_suffix": self.merged_suffix,
            "feature_sets": {
                group: {
                    "state": str(spec.get("state", "Modified")),
                    "features": {
                        name: [float(lo), float(hi)]
                        for name, (lo, hi) in (spec.get("features", {}) or {}).items()
                    },
                }
                for group, spec in self.feature_sets.items()
            },
        }


def feature_layer_names(settings: FeatureSettings, prefix: str) -> dict[str, str]:
    """Every layer ``call_hmm_features`` can produce -> the feature group it
    belongs to (merged layers map to their group too)."""
    names: dict[str, str] = {}
    merged = {core for core, _ in settings.merges}
    suffix = settings.merged_suffix
    for group, spec in settings.feature_sets.items():
        aggregate = f"{prefix}_all_{group}_features"
        stems = [aggregate] + [f"{prefix}_{name}" for name in (spec.get("features", {}) or {})]
        for stem in stems:
            names[stem] = group
            names[f"{stem}_lengths"] = group
        names[f"{aggregate}{END_TRUNCATED_SUFFIX}"] = group
        if f"all_{group}_features" in merged:
            for stem in stems:
                names[f"{stem}{suffix}"] = group
                names[f"{stem}{suffix}_lengths"] = group
            names[f"{aggregate}{suffix}{END_TRUNCATED_SUFFIX}"] = group
    return names


def _same_state(a: str, b: str) -> bool:
    def norm(value: str) -> str:
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    return norm(a) == norm(b)


#: Reads per block in :func:`call_hmm_features`.
FEATURE_CHUNK_READS = 1024


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index runs of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def run_lengths(binary: np.ndarray) -> np.ndarray:
    """Each run of nonzero values carries its length (columns) across the run."""
    binary = np.nan_to_num(np.asarray(binary, dtype=float)) > 0
    out = np.zeros(binary.shape, dtype=np.int32)
    for i in range(binary.shape[0]):
        for start, end in _runs(binary[i]):
            out[i, start:end] = end - start
    return out


def _size_class(length: float, ranges: Mapping[str, tuple[float, float]]) -> str | None:
    for name, (lo, hi) in ranges.items():
        if float(lo) <= float(length) < float(hi):
            return name
    return None


def _in_span(positions: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """(N, L) True where a position lies within the read's [start, end]."""
    starts = spans[:, :1].astype(np.int64)
    ends = spans[:, 1:2].astype(np.int64)
    return (positions[None, :] >= starts) & (positions[None, :] <= ends)


def _mask(layer: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.asarray(layer, dtype=np.float32).copy()
    out[~keep] = np.nan
    return out


def call_hmm_features(
    posterior: np.ndarray,
    positions: np.ndarray,
    spans: np.ndarray,
    *,
    prefix: str,
    settings: FeatureSettings,
    covered: np.ndarray | None = None,
    layers: Sequence[str] | None = None,
    chunk_reads: int | None = FEATURE_CHUNK_READS,
) -> dict[str, np.ndarray]:
    """Feature layers from a two-state posterior.

    Args:
        posterior: (N, L) posterior of ``settings.posterior_state``; NaN where
            a column was not decoded.
        positions: (L,) reference coordinates of the columns, ascending.
        spans: (N, 2) each read's reference ``[start, end]`` (inclusive).
        prefix: layer-name prefix (the model label, e.g. ``C``).
        settings: feature settings.
        covered: optional (N, L) bases each read covers; features are masked
            (NaN) where it is False, as well as outside the span.
        layers: optional names to return (see :func:`feature_layer_names`);
            only their feature groups and merges are computed. Default: all.
        chunk_reads: reads computed at a time. Every feature is a property
            of one read, so chunking changes nothing but the peak memory --
            the intermediates are ~12x the output. ``None``: all at once.

    Returns:
        layer name -> (N, L) array, float32 with NaN outside the read (and at
        uncovered bases).
    """
    n_total = np.shape(posterior)[0]
    if chunk_reads is not None and 0 < int(chunk_reads) < n_total:
        return _call_chunked(
            posterior,
            positions,
            spans,
            prefix=prefix,
            settings=settings,
            covered=covered,
            layers=layers,
            chunk_reads=int(chunk_reads),
        )
    posterior = np.asarray(posterior, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    spans = np.asarray(spans)
    n_reads, n_cols = posterior.shape
    in_span = _in_span(positions, spans)
    keep = in_span if covered is None else (in_span & np.asarray(covered, dtype=bool))
    decoded = in_span & np.isfinite(posterior)
    threshold = float(settings.prob_threshold)
    wanted = None
    if layers is not None:
        known = feature_layer_names(settings, prefix)
        unknown = sorted(set(layers).difference(known))
        if unknown:
            raise KeyError(f"not derivable from the {prefix} posterior: {unknown}")
        wanted = set(layers)
    groups = None if wanted is None else {known[name] for name in wanted}
    requested = layers
    layers: dict[str, np.ndarray] = {}

    for group, spec in settings.feature_sets.items():
        ranges = dict(spec.get("features", {}) or {})
        if not ranges or (groups is not None and group not in groups):
            continue
        favoured = _same_state(spec.get("state", "Modified"), settings.posterior_state)
        member = posterior >= threshold if favoured else posterior < 1.0 - threshold
        aggregate = f"{prefix}_all_{group}_features"
        out = {name: np.zeros((n_reads, n_cols), np.uint8) for name in ranges}
        all_layer = np.zeros((n_reads, n_cols), np.uint8)
        truncated = np.zeros((n_reads, n_cols), np.uint8)
        for i in range(n_reads):
            columns = np.flatnonzero(decoded[i])
            if columns.size == 0:
                continue
            row = member[i, columns]
            for start, end in _runs(row):
                first, last = columns[start], columns[end - 1]
                length = int(positions[last]) - int(positions[first]) + 1
                name = _size_class(length, ranges)
                if name is None:
                    continue
                out[name][i, first : last + 1] = 1
                all_layer[i, first : last + 1] = 1
                if start == 0 or end == columns.size:
                    truncated[i, first : last + 1] = 1
        layers[aggregate] = all_layer
        layers[f"{aggregate}_lengths"] = run_lengths(all_layer)
        layers[f"{aggregate}{END_TRUNCATED_SUFFIX}"] = truncated
        for name in ranges:
            layers[f"{prefix}_{name}"] = out[name]
            layers[f"{prefix}_{name}_lengths"] = run_lengths(out[name])

    merges = settings.merges
    if wanted is not None:
        suffix = settings.merged_suffix
        merges = tuple(
            (core, distance)
            for core, distance in merges
            if any(name.startswith(prefix) and suffix in name for name in wanted)
            and f"{prefix}_{core}" in layers
        )
    merged = _merge_layers(
        layers, positions, prefix=prefix, settings=settings.with_overrides(merges=merges), keep=keep
    )
    layers = {name: _mask(array, keep) for name, array in layers.items()}
    layers.update(merged)
    if requested is not None:
        layers = {name: layers[name] for name in requested}
    return layers


def _call_chunked(posterior, positions, spans, *, chunk_reads, covered, **kwargs):
    """:func:`call_hmm_features` over consecutive blocks of reads, stitched."""
    spans = np.asarray(spans)
    n_reads = np.shape(posterior)[0]
    out: dict[str, np.ndarray] = {}
    for start in range(0, n_reads, chunk_reads):
        stop = min(start + chunk_reads, n_reads)
        part = call_hmm_features(
            posterior[start:stop],
            positions,
            spans[start:stop],
            covered=None if covered is None else covered[start:stop],
            chunk_reads=None,
            **kwargs,
        )
        for name, values in part.items():
            if name not in out:
                out[name] = np.empty((n_reads, values.shape[1]), dtype=values.dtype)
            out[name][start:stop] = values
    return out


def _merge_layers(
    layers: Mapping[str, np.ndarray],
    positions: np.ndarray,
    *,
    prefix: str,
    settings: FeatureSettings,
    keep: np.ndarray,
) -> dict[str, np.ndarray]:
    """Merged aggregates (gaps <= N bp) and their size classes, masked."""
    out: dict[str, np.ndarray] = {}
    suffix = settings.merged_suffix
    for core, distance in settings.merges:
        base = f"{prefix}_{core}"
        if base not in layers:
            continue
        # The merge sees the aggregate as masked to the read (uncovered bases
        # are gaps), as the stage applied it.
        binary = (np.asarray(layers[base]) > 0) & keep
        truncated = (np.asarray(layers[f"{base}{END_TRUNCATED_SUFFIX}"]) > 0) & keep
        merged = np.zeros(binary.shape, np.uint8)
        merged_truncated = np.zeros(binary.shape, np.uint8)
        for i in range(binary.shape[0]):
            runs = _runs(binary[i])
            if not runs:
                continue
            groups = [[runs[0]]]
            for start, end in runs[1:]:
                gap = int(positions[start]) - int(positions[groups[-1][-1][1] - 1]) - 1
                if gap <= int(distance):
                    groups[-1].append((start, end))
                else:
                    groups.append([(start, end)])
            for group_runs in groups:
                first, last = group_runs[0][0], group_runs[-1][1]
                merged[i, first:last] = 1
                if any(truncated[i, s:e].any() for s, e in group_runs):
                    merged_truncated[i, first:last] = 1
        name = f"{base}{suffix}"
        out[name] = _mask(merged, keep)
        out[f"{name}_lengths"] = _mask(run_lengths(merged), keep)
        out[f"{name}{END_TRUNCATED_SUFFIX}"] = _mask(merged_truncated, keep)
        ranges = {}
        for group, spec in settings.feature_sets.items():
            if core == f"all_{group}_features":
                ranges = dict(spec.get("features", {}) or {})
        for feature in ranges:
            out[f"{prefix}_{feature}{suffix}"] = np.zeros(binary.shape, np.uint8)
        for i in range(merged.shape[0]):
            for start, end in _runs(merged[i] > 0):
                length = int(positions[end - 1]) - int(positions[start]) + 1
                feature = _size_class(length, ranges)
                if feature is not None:
                    out[f"{prefix}_{feature}{suffix}"][i, start:end] = 1
        for feature in ranges:
            layer = out[f"{prefix}_{feature}{suffix}"]
            out[f"{prefix}_{feature}{suffix}_lengths"] = _mask(run_lengths(layer), keep)
            out[f"{prefix}_{feature}{suffix}"] = _mask(layer, keep)
    return out
