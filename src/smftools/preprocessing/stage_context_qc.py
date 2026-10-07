"""Sequence-context QC of a preprocess generation (`SCQ-01`).

How strongly each barcode's modification calls depend on the bases around
the modified site -- the enzyme's (or chemistry's) sequence preference -- read
from the finished store: the site calls of the stage's site types, passing
reads only. QC and dedup flags are decided after the per-task pass, so the
tallies cannot be built inside the tasks; this closing step reads the store a
barcode at a time over worker processes. Memory scales with sites x barcodes.

Tables and statistics are those of ``smftools context-bias``
(`analysis.compute.site_context_bias`), per barcode within each reference
and site type, with a ``cpg`` flag (the site's strand-oriented next base is
G) so methylation reads as methylation rather than as enzyme preference.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTEXT_QC_SUBDIR = "context_qc"
CONTEXT_QC_CATEGORY = "context_qc"
PASSING_COLUMNS = ("passes_dedup", "passes_qc")
# Site types per modality: the flanks beyond a dinucleotide still carry the
# information, so conversion reports GpC and CpG separately.
MODALITY_SITE_TYPES = {
    "deaminase": ("C_site",),
    "conversion": ("GpC_site", "CpG_site"),
}
# Figures leave out barcodes with fewer calls than this (about ten reads of an
# amplicon); the tables keep every barcode.
MIN_PLOT_CALLS = 10_000
COUNT_COLUMNS = ["barcode", "physical_reference", "site_type", "position", "observed", "modified"]


def site_types_for(cfg) -> tuple[str, ...]:
    """The modality's site types; none for direct modalities (no use yet)."""
    modality = str(getattr(cfg, "smf_modality", "") or "").strip().lower()
    return MODALITY_SITE_TYPES.get(modality, ())


def tally_task_store(
    path: str | Path,
    *,
    barcode: str,
    reference: str,
    site_types: Iterable[str],
    passing: set | None,
) -> pd.DataFrame:
    """Per site of one task store: observed and modified calls of passing reads."""
    import anndata as ad
    import zarr

    group = zarr.open_group(str(path), mode="r")
    obs_names = ad.io.read_elem(group["obs"]).index.astype(str)
    var = ad.io.read_elem(group["var"])
    rows = np.ones(len(obs_names), dtype=bool)
    if passing is not None:
        rows = obs_names.isin(passing)
    frames = []
    if rows.any():
        values = np.asarray(ad.io.read_elem(group["X"]), dtype=float)[rows]
        observed = ~np.isnan(values)
        modified = observed & (np.nan_to_num(values, nan=0.0) >= 0.5)
        positions = var.index.astype(np.int64).to_numpy()
        for site_type in site_types:
            column = f"{reference}_{site_type}"
            if column not in var:
                continue
            mask = var[column].to_numpy(dtype=bool)
            n_observed = observed[:, mask].sum(axis=0)
            keep = n_observed > 0
            frames.append(
                pd.DataFrame(
                    {
                        "barcode": barcode,
                        "physical_reference": reference,
                        "site_type": site_type,
                        "position": positions[mask][keep],
                        "observed": n_observed[keep].astype(np.int64),
                        "modified": modified[:, mask].sum(axis=0)[keep].astype(np.int64),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COUNT_COLUMNS)


def _tally_barcode(paths, references, barcode, site_types, passing) -> pd.DataFrame:
    frames = [
        tally_task_store(
            path, barcode=barcode, reference=reference, site_types=site_types, passing=passing
        )
        for path, reference in zip(paths, references, strict=True)
    ]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame(columns=COUNT_COLUMNS)
    # Chunks of one window hold different reads at the same sites.
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["barcode", "physical_reference", "site_type", "position"], as_index=False)[
            ["observed", "modified"]
        ]
        .sum()
    )


def count_stage_sites(
    task_catalog: pd.DataFrame,
    task_path,
    obs: pd.DataFrame,
    site_types: Iterable[str],
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """Site tallies of passing reads, per barcode, reference and site type.

    ``task_path(row)`` locates a catalog row's store; ``obs`` (the stage's
    final read table) decides which reads pass -- dedup where it ran, else QC.
    """
    site_types = tuple(site_types)
    column = next((name for name in PASSING_COLUMNS if name in obs), None)
    barcodes = obs["Barcode"].astype(str) if "Barcode" in obs else None
    passes = obs[column].astype(bool) if column else pd.Series(True, index=obs.index)
    jobs = []
    for barcode, tasks in task_catalog.groupby("barcode", sort=True):
        if column is None:
            passing = None
        else:
            chosen = passes if barcodes is None else passes & (barcodes == str(barcode))
            passing = set(obs.index[chosen.to_numpy()].astype(str))
        jobs.append(
            (
                [task_path(row) for row in tasks.itertuples(index=False)],
                tasks["reference"].astype(str).tolist(),
                str(barcode),
                site_types,
                passing,
            )
        )
    if workers > 1 and len(jobs) > 1:
        from smftools.parallel_utils import configure_worker_threads

        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            initializer=configure_worker_threads,
            initargs=(1,),
        ) as pool:
            frames = list(pool.map(_tally_barcode, *zip(*jobs)))
    else:
        frames = [_tally_barcode(*job) for job in jobs]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame(columns=COUNT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def context_statistics(
    counts: pd.DataFrame, sequences: dict[str, str], *, flank: int, kmers: Iterable[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sites with contexts; offset enrichment and k-mer rates per barcode.

    Statistics are computed per (reference, site type, CpG flag) with the
    barcode as the group; references without a sequence are skipped.
    """
    from smftools.analysis.compute.site_context_bias import (
        kmer_rates,
        offset_enrichment,
        site_contexts,
        strand_of,
    )

    kmers = tuple(sorted(set(int(k) for k in kmers)))
    for k in kmers:
        if k < 1 or k % 2 == 0 or k > 2 * flank + 1:
            raise ValueError(f"k-mer size {k} must be odd and between 1 and {2 * flank + 1}")
    if counts.empty:
        return counts.assign(context=[], rate=[], cpg=[]), pd.DataFrame(), pd.DataFrame()
    known = counts["physical_reference"].map(lambda name: strand_of(str(name))[0] in sequences)
    for reference in sorted(set(counts.loc[~known, "physical_reference"])):
        logger.warning("Context QC: no sequence for %s; its sites are skipped", reference)
    sites = site_contexts(
        counts.loc[known].rename(columns={"barcode": "group"}), sequences, flank=flank
    )
    sites["cpg"] = sites["context"].str[flank + 1] == "G"
    enrichment, rates = [], []
    keys = ["physical_reference", "site_type", "cpg"]
    for (reference, site_type, cpg), frame in sites.groupby(keys, sort=True):
        tags = {"physical_reference": reference, "site_type": site_type, "cpg": cpg}
        enrichment.append(offset_enrichment(frame, flank=flank).assign(**tags))
        for k in kmers:
            rates.append(kmer_rates(frame, flank=flank, k=k).assign(k=k, **tags))
    sites = sites.rename(columns={"group": "barcode"})
    enrichment = (
        pd.concat(enrichment, ignore_index=True).rename(columns={"group": "barcode"})
        if enrichment
        else pd.DataFrame()
    )
    rates = (
        pd.concat(rates, ignore_index=True).rename(columns={"group": "barcode"})
        if rates
        else pd.DataFrame()
    )
    return sites, enrichment, rates


def _label(reference: str, site_type: str, cpg: bool) -> str:
    return f"{reference} {site_type.removesuffix('_site')} sites, {'CpG' if cpg else 'non-CpG'}"


def _short_names(barcodes) -> dict[str, str]:
    """Barcode names without their shared kit prefix (``SQK-NBD114-96_barcode33`` -> ``barcode33``)."""
    import os

    names = [str(name) for name in barcodes]
    prefix = os.path.commonprefix(names)
    prefix = prefix[: prefix.rfind("_") + 1] if "_" in prefix else ""
    return {name: name[len(prefix) :] or name for name in names}


def plot_context_qc(
    enrichment: pd.DataFrame,
    rates: pd.DataFrame,
    layout,
    *,
    min_calls: int = MIN_PLOT_CALLS,
) -> list[Path]:
    """Per reference, site type and CpG flag: logos, offset heatmaps and k-mer
    rates with one panel per barcode (barcodes under ``min_calls`` calls left out)."""
    from smftools.analysis.plot.site_context_bias import (
        plot_enrichment_logo,
        plot_kmer_rates,
        plot_offset_enrichment_heatmap,
    )
    from smftools.cli.stage_artifacts import register_plot_artifact

    from .partitioned_executor import _component

    root = Path(layout.categories[CONTEXT_QC_CATEGORY])
    written = []
    keys = ["physical_reference", "site_type", "cpg"]
    calls = rates[rates["k"] == rates["k"].min()].groupby(keys + ["barcode"])["observed"].sum()
    short = _short_names(enrichment["barcode"].unique())
    for (reference, site_type, cpg), frame in enrichment.groupby(keys, sort=True):
        enough = calls.loc[(reference, site_type, cpg)]
        enough = set(enough.index[enough >= min_calls])
        frame = frame[frame["barcode"].isin(enough)]
        if frame.empty:
            continue
        stem = (
            f"{_component(reference)}__{site_type.removesuffix('_site')}"
            f"__{'cpg' if cpg else 'non_cpg'}"
        )
        title = _label(reference, site_type, cpg)
        figures = [
            ("enrichment_logo", root / f"{stem}__enrichment_logo.png", plot_enrichment_logo),
            (
                "offset_enrichment",
                root / f"{stem}__offset_enrichment.png",
                plot_offset_enrichment_heatmap,
            ),
        ]
        for plot_type, path, draw in figures:
            draw(frame.assign(group=frame["barcode"].map(short)), path, ncols=4, title=title)
            written.append((plot_type, path, reference))
        selected = rates[
            (rates["physical_reference"] == reference)
            & (rates["site_type"] == site_type)
            & (rates["cpg"] == cpg)
            & (rates["k"] > 1)
            & rates["barcode"].isin(enough)
        ]
        for k, by_k in selected.groupby("k"):
            path = root / f"{stem}__kmer_rates_k{k}.png"
            plot_kmer_rates(by_k.assign(group=by_k["barcode"].map(short)), path, title=title)
            written.append(("kmer_rates", path, reference))
    for plot_type, path, reference in written:
        register_plot_artifact(
            layout,
            path,
            stage="preprocess",
            category=CONTEXT_QC_CATEGORY,
            plot_type=f"context_qc_{plot_type}",
            reference=str(reference),
        )
    return [path for _, path, _ in written]


def write_stage_context_qc(
    output_dir: str | Path,
    task_catalog: str | Path,
    obs: pd.DataFrame,
    uns,
    cfg,
    *,
    plot_layout=None,
    workers: int = 1,
) -> Path | None:
    """Tables (and, with ``plot_layout``, figures) under ``<output_dir>/context_qc``.

    Returns the directory, or ``None`` when the settings are off or the
    modality has no site types.
    """
    from smftools.tools.site_context_bias import sequences_from_uns

    from .partitioned_executor import PREPROCESS_STORE_SUBDIR

    if not bool(getattr(cfg, "stage_context_qc", True)):
        return None
    site_types = site_types_for(cfg)
    if not site_types:
        logger.info("Context QC: no site types for this modality; skipped")
        return None
    output_dir = Path(output_dir)
    flank = int(getattr(cfg, "stage_context_qc_flank", 3) or 3)
    kmers = tuple(getattr(cfg, "stage_context_qc_kmers", None) or (1, 3))
    catalog = pd.read_parquet(task_catalog)

    from .partitioned_executor import _component

    def task_path(row) -> Path:
        return (
            output_dir
            / PREPROCESS_STORE_SUBDIR
            / f"reference={_component(row.reference)}"
            / f"core={int(row.core_start):012d}-{int(row.core_end):012d}"
            / f"barcode={_component(row.barcode)}"
            / f"chunk={int(row.chunk_index):05d}"
        )

    counts = count_stage_sites(catalog, task_path, obs, site_types, workers=workers)
    target = output_dir / CONTEXT_QC_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(target / "site_counts.parquet", index=False)
    sites, enrichment, rates = context_statistics(
        counts, sequences_from_uns(uns), flank=flank, kmers=kmers
    )
    sites.to_parquet(target / "sites.parquet", index=False)
    enrichment.to_csv(target / "offset_enrichment.csv", index=False)
    rates.to_csv(target / "kmer_rates.csv", index=False)
    figures = []
    if plot_layout is not None and len(enrichment):
        figures = [str(path) for path in plot_context_qc(enrichment, rates, plot_layout)]
    passing = next((name for name in PASSING_COLUMNS if name in obs), None)
    (target / "run.json").write_text(
        json.dumps(
            {
                "site_types": list(site_types),
                "flank": flank,
                "kmers": list(kmers),
                "passing_column": passing,
                "barcodes": sorted(set(counts["barcode"].astype(str))),
                "sites": int(len(sites)),
                "figures": len(figures),
            },
            indent=2,
        )
    )
    return target
