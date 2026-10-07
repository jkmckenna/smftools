"""Sequence-context QC of HMM state calls, per emission variant (`SCQ-02`).

Two questions per barcode, reference and variant, at the HMM's own sites:

1. **Residual bias** -- does the accessible state still follow sequence?
   Accessible calls / observed sites per C-centred k-mer, relative to the
   barcode's overall accessible rate: flat if the HMM reads chromatin.
2. **Modification within accessible-called sites** -- modified / accessible
   per k-mer: the enzyme's preference with most of the chromatin effect
   removed, exportable as an `HCE-01` weight table (source ``accessible``).

Tallies are made per task while its decoded layers are at hand
(`tally_hmm_task`) and reduced once all tasks finish (`write_hmm_context_qc`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTEXT_QC_SUBDIR = "context_qc"
CONTEXT_QC_CATEGORY = "context_qc"
MIN_PLOT_CALLS = 10_000
ACCESSIBLE_SUFFIX = "_all_accessible_features"
TALLY_COLUMNS = [
    "barcode",
    "physical_reference",
    "model",
    "variant",
    "position",
    "observed",
    "modified",
    "accessible",
    "accessible_modified",
]
MEASURES = {
    # measure: (observed column, modified column, plot scale, x label)
    "residual_bias": ("observed", "accessible", "relative", "accessible calls"),
    "accessible_rate": ("accessible", "accessible_modified", "absolute", "modification"),
}


def _variant_name(spec) -> str:
    return spec.variant or "default"


def tally_hmm_task(adata, reference: str, barcode: str, core_mask, cfg, specs) -> pd.DataFrame:
    """Per site of one task's core: observed, modified, accessible-called and
    modified-while-accessible reads, for every single-channel model spec whose
    accessible layer is on ``adata``."""
    from .partitioned_hmm import _prepare_model_input

    positions = np.asarray(adata.var_names, dtype=np.int64)
    core = positions[np.asarray(core_mask, dtype=bool)]
    frames = []
    for spec in specs:
        layer = f"{spec.label}{ACCESSIBLE_SUFFIX}"
        if len(spec.signals) != 1 or layer not in adata.layers:
            continue
        values, coords, _ = _prepare_model_input(adata, reference, spec, spec.config(cfg))
        values = np.asarray(values, dtype=float)
        coords = np.asarray(coords, dtype=np.int64)
        inside = np.isin(coords, core)
        if not inside.any():
            continue
        values, coords = values[:, inside], coords[inside]
        columns = np.searchsorted(positions, coords)
        state = np.asarray(adata.layers[layer], dtype=float)[:, columns]
        observed = ~np.isnan(values)
        modified = observed & (np.nan_to_num(values, nan=0.0) >= 0.5)
        accessible = observed & (np.nan_to_num(state, nan=0.0) > 0)
        n_observed = observed.sum(axis=0)
        keep = n_observed > 0
        frames.append(
            pd.DataFrame(
                {
                    "barcode": str(barcode),
                    "physical_reference": str(reference),
                    "model": spec.base_label or spec.label,
                    "variant": _variant_name(spec),
                    "position": coords[keep],
                    "observed": n_observed[keep].astype(np.int64),
                    "modified": modified.sum(axis=0)[keep].astype(np.int64),
                    "accessible": accessible.sum(axis=0)[keep].astype(np.int64),
                    "accessible_modified": (modified & accessible)
                    .sum(axis=0)[keep]
                    .astype(np.int64),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TALLY_COLUMNS)


def write_task_partial(output_dir: Path, task_id: str, tallies: pd.DataFrame) -> str:
    """Save one task's tallies; returns the path relative to ``output_dir``."""
    if tallies.empty:
        return ""
    name = hashlib.sha1(str(task_id).encode()).hexdigest()[:20]
    path = Path(output_dir) / CONTEXT_QC_SUBDIR / "partials" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tallies.to_parquet(path, index=False)
    return path.relative_to(output_dir).as_posix()


def _measure_sites(sites: pd.DataFrame, measure: str) -> pd.DataFrame:
    observed, modified, _, _ = MEASURES[measure]
    frame = sites.assign(observed=sites[observed], modified=sites[modified])
    return frame[frame["observed"] > 0]


def context_tables(
    tallies: pd.DataFrame, sequences: dict[str, str], *, flank: int, kmers: Iterable[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sites with contexts, and k-mer rates per barcode for both measures.

    Rates are per (reference, model, variant, CpG flag, measure) with the
    barcode as the group; references without a sequence are skipped.
    """
    from smftools.analysis.compute.site_context_bias import kmer_rates, site_contexts, strand_of

    kmers = tuple(sorted({int(k) for k in kmers}))
    if tallies.empty:
        return tallies.assign(context=[], cpg=[]), pd.DataFrame()
    known = tallies["physical_reference"].map(lambda name: strand_of(str(name))[0] in sequences)
    for reference in sorted(set(tallies.loc[~known, "physical_reference"])):
        logger.warning("HMM context QC: no sequence for %s; its sites are skipped", reference)
    sites = site_contexts(
        tallies.loc[known].rename(columns={"barcode": "group"}), sequences, flank=flank
    )
    sites["cpg"] = sites["context"].str[flank + 1] == "G"
    rates = []
    keys = ["physical_reference", "model", "variant", "cpg"]
    for values, frame in sites.groupby(keys, sort=True):
        tags = dict(zip(keys, values, strict=True))
        for measure in MEASURES:
            subset = _measure_sites(frame, measure)
            if subset.empty:
                continue
            for k in kmers:
                rates.append(
                    kmer_rates(subset, flank=flank, k=k).assign(k=k, measure=measure, **tags)
                )
    sites = sites.drop(columns="rate").rename(columns={"group": "barcode"})
    rates = (
        pd.concat(rates, ignore_index=True).rename(columns={"group": "barcode"})
        if rates
        else pd.DataFrame()
    )
    return sites, rates


def accessible_weight_tables(sites: pd.DataFrame, *, flank: int, k: int) -> dict[str, pd.DataFrame]:
    """Per model and variant, an `HCE-01` weight table (group = barcode) from the
    modification rate within accessible-called sites, references pooled."""
    from smftools.analysis.compute.site_context_bias import kmer_rates, weight_table

    tables = {}
    if k < 3 or sites.empty:
        return tables
    for (model, variant), frame in sites.groupby(["model", "variant"], sort=True):
        subset = _measure_sites(frame.rename(columns={"barcode": "group"}), "accessible_rate")
        if subset.empty:
            continue
        rates = kmer_rates(subset, flank=flank, k=k)
        tables[f"{model}_{variant}"] = weight_table(rates, k=k, source="accessible")
    return tables


def plot_hmm_context_qc(rates: pd.DataFrame, layout, *, min_calls: int = MIN_PLOT_CALLS):
    """Per reference, model, barcode and measure: k-mer profiles with every
    variant overlaid (one colour each), non-CpG and CpG panels."""
    from smftools.analysis.plot.site_context_bias import plot_kmer_rate_series
    from smftools.cli.stage_artifacts import register_plot_artifact
    from smftools.preprocessing.stage_context_qc import _short_names

    from .partitioned_hmm import VARIANT_COLORS, _component

    root = Path(layout.categories[CONTEXT_QC_CATEGORY])
    shown = rates[rates["k"] == rates["k"].max()]
    if shown.empty or int(shown["k"].iloc[0]) < 3:
        return []
    calls = (
        rates[(rates["k"] == rates["k"].min()) & (rates["measure"] == "residual_bias")]
        .groupby(["physical_reference", "model", "barcode"])["observed"]
        .sum()
    )
    short = _short_names(shown["barcode"].unique())
    written = []
    keys = ["physical_reference", "model", "barcode", "measure"]
    for (reference, model, barcode, measure), frame in shown.groupby(keys, sort=True):
        if calls.get((reference, model, barcode), 0) < min_calls:
            continue
        variants = list(dict.fromkeys(sorted(frame["variant"], key=lambda v: v != "default")))
        frame = frame.assign(
            group=frame["variant"] + "|" + np.where(frame["cpg"], "CpG", "non-CpG")
        )
        panels = {
            panel: [
                {
                    "label": variant,
                    "groups": [f"{variant}|{panel}"],
                    "color": VARIANT_COLORS[index % len(VARIANT_COLORS)],
                }
                for index, variant in enumerate(variants)
                if f"{variant}|{panel}" in set(frame["group"])
            ]
            for panel in ("non-CpG", "CpG")
        }
        panels = {panel: series for panel, series in panels.items() if series}
        if not panels:
            continue
        _, _, scale, label = MEASURES[measure]
        path = root / (
            f"{_component(reference)}__{_component(model)}__{_component(short[barcode])}"
            f"__{measure}.png"
        )
        plot_kmer_rate_series(
            frame,
            path,
            panels=panels,
            scale=scale,
            title=f"{reference} {short[barcode]} [{model}]: {measure.replace('_', ' ')}"
            f" ({label}, {scale})",
            legend_title="HMM",
        )
        register_plot_artifact(
            layout,
            path,
            stage="hmm",
            category=CONTEXT_QC_CATEGORY,
            plot_type=f"hmm_context_qc_{measure}",
            reference=str(reference),
        )
        written.append(path)
    return written


def write_hmm_context_qc(output_dir: str | Path, records, uns, cfg, *, layout=None) -> Path | None:
    """Reduce the task partials into tables (and figures with ``layout``)
    under ``<output_dir>/context_qc``. ``None`` when off or nothing was tallied."""
    from smftools.analysis.compute.site_context_bias import write_weight_table
    from smftools.tools.site_context_bias import sequences_from_uns

    if not bool(getattr(cfg, "stage_context_qc", True)):
        return None
    output_dir = Path(output_dir)
    paths = [str(record.get("context_qc_partial") or "") for record in records]
    frames = [pd.read_parquet(output_dir / path) for path in paths if path]
    if not frames:
        return None
    tallies = (
        pd.concat(frames, ignore_index=True)
        .groupby(TALLY_COLUMNS[:5], as_index=False)[TALLY_COLUMNS[5:]]
        .sum()
    )
    flank = int(getattr(cfg, "stage_context_qc_flank", 3) or 3)
    kmers = tuple(getattr(cfg, "stage_context_qc_kmers", None) or (1, 3))
    target = output_dir / CONTEXT_QC_SUBDIR
    tallies.to_parquet(target / "site_counts.parquet", index=False)
    # The reduced counts replace the per-task partials.
    shutil.rmtree(target / "partials", ignore_errors=True)
    sites, rates = context_tables(tallies, sequences_from_uns(uns), flank=flank, kmers=kmers)
    sites.to_parquet(target / "sites.parquet", index=False)
    rates.to_csv(target / "kmer_rates.csv", index=False)
    weight_files = []
    for k in kmers:
        for name, table in accessible_weight_tables(sites, flank=flank, k=k).items():
            path = target / f"accessible_weights_{name}_k{k}.parquet"
            write_weight_table(table, path)
            weight_files.append(path.name)
    figures = []
    if layout is not None and len(rates):
        figures = plot_hmm_context_qc(rates, layout)
    (target / "run.json").write_text(
        json.dumps(
            {
                "flank": flank,
                "kmers": list(kmers),
                "variants": sorted(set(tallies["variant"])),
                "models": sorted(set(tallies["model"])),
                "barcodes": sorted(set(tallies["barcode"].astype(str))),
                "sites": int(len(sites)),
                "weight_tables": weight_files,
                "figures": len(figures),
            },
            indent=2,
        )
    )
    return target
