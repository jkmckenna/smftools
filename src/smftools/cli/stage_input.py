"""Bounded input iteration shared by partition-aware downstream CLI stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from smftools.constants import REFERENCE_STRAND

from ..informatics.partition_read import load_spine, materialize
from ..plotting.plotting_utils import subsample_read_ids


@dataclass
class StageSlice:
    """One reference core plus its loaded halo and selected molecules."""

    reference: str
    analysis_mode: str
    core_start: int
    core_end: int
    load_start: int
    load_end: int
    adata: object

    def core(self):
        """Return a copy cropped to the non-overlapping output core."""
        positions = np.asarray(self.adata.var_names, dtype=np.int64)
        return self.adata[:, (positions >= self.core_start) & (positions < self.core_end)].copy()


def _reference_windows(plan: dict[str, object]) -> Iterable[tuple[int, int, int, int]]:
    reference_length = int(plan["reference_length"])
    if plan.get("analysis_mode") == "locus":
        yield 0, reference_length, 0, reference_length
        return
    tile_size = int(plan["tile_size"])
    tile_halo = int(plan["tile_halo"])
    for core_start in range(0, reference_length, tile_size):
        core_end = min(core_start + tile_size, reference_length)
        yield (
            core_start,
            core_end,
            max(0, core_start - tile_halo),
            min(reference_length, core_end + tile_halo),
        )


def iter_stage_slices(
    spine_path: str | Path,
    *,
    layers=None,
    samples=None,
    filter_mask: str | None = "auto",
    max_reads_per_sample: int | None = None,
) -> Iterable[StageSlice]:
    """Yield bounded, non-empty reference slices from a raw/load/preprocess spine.

    ``filter_mask='auto'`` prefers ``passes_dedup`` and then ``passes_qc``. Set it
    to ``None`` to retain every molecule.

    ``max_reads_per_sample``: when set, each (reference, core, sample) group's
    read_ids are randomly subsampled to at most this many *before*
    materializing, not after -- a caller that only plots a bounded number of
    reads per sample (e.g. ``_read_span_quality_plots``, capped to 300
    reads/barcode for legibility) would otherwise still pay to materialize
    every read for the whole reference first. Confirmed on real data: this
    made a single 43,735-read reference's materialize() call take 107.6s
    (of which only 300-per-barcode were ever used), with far larger
    references in the same experiment taking proportionally longer -- see
    dev/pipeline_scaling_audit.md. ``None`` (default) keeps every read,
    matching the previous unbounded behavior for callers that need it (this
    function currently has no other caller, but is a shared partition-aware
    utility).
    """
    spine_path = Path(spine_path)
    spine = load_spine(spine_path)
    plans = dict(spine.uns.get("reference_plans", {}))
    if not plans:
        raise ValueError("partition-aware stage input requires reference_plans")
    if filter_mask == "auto":
        filter_mask = next(
            (column for column in ("passes_dedup", "passes_qc") if column in spine.obs),
            None,
        )
    if filter_mask is not None and filter_mask not in spine.obs:
        raise KeyError(f"spine.obs lacks requested filter mask {filter_mask!r}")
    selected_samples = (
        None
        if samples is None
        else {samples}
        if isinstance(samples, str)
        else {str(value) for value in samples}
    )

    for reference, raw_plan in sorted(plans.items()):
        plan = dict(raw_plan)
        reference_obs = spine.obs.loc[spine.obs[REFERENCE_STRAND].astype(str) == str(reference)]
        if filter_mask is not None:
            reference_obs = reference_obs.loc[reference_obs[filter_mask].astype(bool)]
        if selected_samples is not None:
            sample_column = "Sample" if "Sample" in reference_obs else "Barcode"
            reference_obs = reference_obs.loc[
                reference_obs[sample_column].astype(str).isin(selected_samples)
            ]
        for core_start, core_end, load_start, load_end in _reference_windows(plan):
            core_obs = reference_obs.loc[
                (reference_obs["reference_start"].astype("int64") < core_end)
                & (reference_obs["reference_end"].astype("int64") > core_start)
            ]
            if core_obs.empty:
                continue
            if max_reads_per_sample is not None:
                cap_column = "Sample" if "Sample" in core_obs else "Barcode"
                read_ids = [
                    read_id
                    for _sample, group in core_obs.groupby(cap_column, sort=False, observed=True)
                    for read_id in subsample_read_ids(map(str, group.index), max_reads_per_sample)
                ]
            else:
                read_ids = core_obs.index
            adata = materialize(
                spine_path,
                references=reference,
                read_ids=read_ids,
                start=load_start,
                end=load_end,
                layers=layers,
            )
            yield StageSlice(
                reference=str(reference),
                analysis_mode=str(plan["analysis_mode"]),
                core_start=core_start,
                core_end=core_end,
                load_start=load_start,
                load_end=load_end,
                adata=adata,
            )
