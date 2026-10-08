"""Per-molecule motif occupancy from HMM feature classes (`MOT-04`).

For each read and motif instance, one state:

``uninformative``  the read does not span the motif, or has fewer than
                   ``min_sites`` observed sites within the motif +/- ``flank``
                   (no basis to call it bound or open)
``tf_bound``       a TF-sized footprint covers the motif
``medium_bound``   a medium footprint covers it
``nucleosome``     a nucleosome-sized (or larger) footprint covers it
``accessible``     an accessible feature covers it
``other``          spanned and informative, but no class covers ``min_cover``
                   of the motif (e.g. a footprint shorter than any class)

"Covers" means the class holds the largest share of the motif's positions,
at least ``min_cover``; ties go to the earlier state in ``CLASS_STATES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

STATES = ("uninformative", "tf_bound", "medium_bound", "nucleosome", "accessible", "other")
CLASS_STATES = ("tf_bound", "medium_bound", "nucleosome", "accessible")
UNINFORMATIVE, OTHER = 0, len(STATES) - 1
BOUND_STATES = ("tf_bound", "medium_bound", "nucleosome")


@dataclass(frozen=True)
class OccupancyRules:
    flank: int = 10
    min_sites: int = 2
    min_cover: float = 0.5

    def record(self) -> dict:
        return {"flank": self.flank, "min_sites": self.min_sites, "min_cover": self.min_cover}


@dataclass(frozen=True)
class Instance:
    """A motif instance resolved to the dataset's columns."""

    index: int
    core: slice  # columns of the motif itself
    window: slice  # columns of motif +/- flank (for informative sites)


def resolve_instances(
    hits: pd.DataFrame, coordinates: Sequence[int], flank: int
) -> tuple[list[Instance], pd.DataFrame]:
    """Instances whose every motif position is a dataset column, as column slices.

    Returns the instances and the kept rows of ``hits`` (with ``instance``,
    their row index in the result).
    """
    coordinates = np.asarray(coordinates, dtype=np.int64)
    kept, instances = [], []
    for row, hit in enumerate(hits.itertuples(index=False)):
        start, end = int(hit.start), int(hit.end)
        a, b = np.searchsorted(coordinates, [start, end])
        if b - a != end - start or b <= a:
            continue  # not wholly inside the dataset's positions
        w0, w1 = np.searchsorted(coordinates, [start - flank, end + flank])
        instances.append(Instance(len(kept), slice(int(a), int(b)), slice(int(w0), int(w1))))
        kept.append(row)
    table = hits.iloc[kept].reset_index(drop=True)
    table.insert(0, "instance", np.arange(len(table)))
    return instances, table


def classify(
    classes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    sites_observed: np.ndarray,
    instances: Sequence[Instance],
    rules: OccupancyRules = OccupancyRules(),
) -> np.ndarray:
    """States (codes into ``STATES``), reads x instances.

    ``classes`` maps each of ``CLASS_STATES`` present to ``(values, observed)``
    arrays, reads x columns (a state may combine several layers: values are
    in the class where any is > 0). ``sites_observed`` (reads x columns) marks
    observed raw sites.
    """
    present = [state for state in CLASS_STATES if state in classes]
    if not present:
        raise ValueError(f"no class channels; need some of {CLASS_STATES}")
    n_reads = sites_observed.shape[0]
    states = np.full((n_reads, len(instances)), UNINFORMATIVE, dtype=np.int8)
    in_class = {
        state: np.asarray(classes[state][1], bool)
        & (np.nan_to_num(np.asarray(classes[state][0], float), nan=0.0) > 0)
        for state in present
    }
    spanned_all = np.logical_and.reduce([np.asarray(classes[s][1], bool) for s in present])
    sites = np.asarray(sites_observed, dtype=bool)
    codes = np.array([STATES.index(state) for state in present], dtype=np.int8)
    for column, instance in enumerate(instances):
        width = instance.core.stop - instance.core.start
        spanned = spanned_all[:, instance.core].all(axis=1)
        informative = spanned & (sites[:, instance.window].sum(axis=1) >= rules.min_sites)
        cover = np.stack(
            [in_class[state][:, instance.core].sum(axis=1) / width for state in present], axis=1
        )
        best = np.argmax(cover, axis=1)  # first maximum: precedence on ties
        covered = cover[np.arange(n_reads), best] >= rules.min_cover
        states[:, column] = np.where(
            informative, np.where(covered, codes[best], OTHER), UNINFORMATIVE
        )
    return states


def wilson(successes: np.ndarray, trials: np.ndarray, z: float = 1.96):
    successes = np.asarray(successes, float)
    trials = np.asarray(trials, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = successes / trials
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return centre - half, centre + half


def group_occupancy(
    states: np.ndarray, groups: Sequence[str], instances: pd.DataFrame
) -> pd.DataFrame:
    """Per group and instance: reads per state, informative reads, and each
    state's fraction of informative reads with a Wilson interval for
    ``tf_bound`` and for any bound state."""
    groups = np.asarray([str(g) for g in groups])
    rows = []
    for group in np.unique(groups):
        block = states[groups == group]
        counts = np.stack([(block == code).sum(axis=0) for code in range(len(STATES))], axis=1)
        frame = instances.copy()
        frame.insert(0, "group", group)
        for code, state in enumerate(STATES):
            frame[f"n_{state}"] = counts[:, code]
        informative = counts[:, 1:].sum(axis=1)
        frame["n_reads"] = counts.sum(axis=1)
        frame["n_informative"] = informative
        with np.errstate(invalid="ignore", divide="ignore"):
            for code, state in enumerate(STATES[1:], start=1):
                frame[f"frac_{state}"] = np.where(
                    informative > 0, counts[:, code] / informative, np.nan
                )
        bound = sum(counts[:, STATES.index(s)] for s in BOUND_STATES)
        frame["frac_bound"] = np.where(informative > 0, bound / np.maximum(informative, 1), np.nan)
        for name, numerator in (
            ("tf_bound", counts[:, STATES.index("tf_bound")]),
            ("bound", bound),
        ):
            low, high = wilson(numerator, informative)
            frame[f"frac_{name}_low"], frame[f"frac_{name}_high"] = low, high
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
