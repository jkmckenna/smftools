"""Barcode ligation / detection / tagging model for ligation kits (`BQC-02`).

`EGL-31` (``barcode_contamination_qc``) counts barcodes on a spike-in and
compares the two end calls. Raw end calls mix three things this module
separates, from the per-end evidence `BQC-01` puts on obs:

- **miscalls**: confident calls (edit distance <= ``CONFIDENT_EDIT_DISTANCE``)
  naming a kit barcode that is not in the pool -- only pool barcodes are in the
  tube -- give the per-end miscall rate;
- **ends without a barcode**: a read end with no sequence beyond the insert
  (``rear_clip_length`` <= ``NO_BARCODE_CLIP``) carries no barcode; one with
  ``BARCODE_CLIP`` bases carries one, called or not;
- **free-end tagging**: barcodes from other libraries on ends left free.

Chemistry. Native barcoding is two ligations: a barcode per sample, then,
after pooling, a sequencing adapter that only joins barcoded ends. A molecule
is sequenced at most once, from an adapter end. The barcode is part of the
molecule, so a rear barcode is read whether or not its adapter joined. In the
pool, a still-free end can pick up a carried-over barcode (probability lambda,
barcode b with probability pi_b). Rapid (transposase) and PCR barcoding attach
barcodes differently; for them this model is not fitted
(:func:`barcode_chemistry`).

Estimates, per run:

- **phi**, the foreign share of barcoded ends: confident mismatch reads (both
  ends barcoded and called in-pool, different barcodes) less in-pool miscalls.
  With psi = phi / (1 - phi), contamination relative to a library's own reads
  is, exactly, psi^2 pi_b^2 (N - N_b) / N_b for double-same reads (both ends
  free and tagged b) and psi pi_b / share_b for single reads -- neither needs
  the ligation efficiency, so genome-wide runs get them too;
- **pi_b**: barcodes on mismatch-read ends less their own-barcode half;
- **q**, native barcode ligation per end, per library: needs reads covering
  their whole reference (amplicons), where a free rear end cannot be a
  truncated read. Reads with / without a rear barcode are in ratio
  a (2 - alpha) / (2 (1 - a)), alpha the adapter efficiency per barcoded end
  (not yet measured, `BQC-01b`), so q is reported for alpha = 1 and alpha -> 0;
- **lambda** = psi q / (1 - q), when q is available;
- **detection** per end, and **ownership posteriors** for mismatch reads;
- **spike-in** (optional): an unbarcoded spike-in has q = 0, so its reads
  measure tagging directly -- if its reads cover the reference end to end and
  both ends ligate (checked: the share of reads starting at each end).

Mismatch reads must be in obs: with ``skip_unclassified`` they are dropped and
phi cannot be estimated (recorded in the summary). Nothing here filters or
corrects reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

from .barcode_contamination_qc import (
    CONTAMINATION_QC_SUBDIR,
    REFERENCE_COLUMN,
    UNASSIGNED,
    poisson_interval,
    spike_in_mask,
)

logger = get_logger(__name__)

TAGGING_BY_BARCODE_FILENAME = "tagging_by_barcode.parquet"
TAGGING_SUMMARY_FILENAME = "tagging_summary.json"
TAGGING_SCHEMA_VERSION = 1

#: A barcode call at or below this edit distance is treated as confident.
#: On a 1.3M-read native run, out-of-pool calls were >= 6 edits 99 % of the
#: time and own-barcode calls <= 4 edits 94 % of the time.
CONFIDENT_EDIT_DISTANCE = 4
#: Bases beyond the insert at or below which an end carries no barcode.
NO_BARCODE_CLIP = 20
#: Bases beyond the insert for an end carrying a barcode (+ adapter arm); more
#: is read-through into further sequence and is left out.
BARCODE_CLIP = (35, 130)
#: Alignment ends within this many bases of the reference ends count as full length.
SPAN_TOLERANCE = 15
#: Full-length reads needed before q is fitted.
MIN_FULL_LENGTH_READS = 1000
#: A barcode is in the pool when its confident double-same reads reach this
#: share of the median pool barcode's (used only when no pool is declared).
POOL_SHARE_OF_MEDIAN = 0.01
BOOTSTRAP_DRAWS = 500

_CHEMISTRY_PREFIXES = (
    (("SQK-NBD", "EXP-NBD", "SQK-NBK"), "native_ligation"),
    (("SQK-RBK", "SQK-RPB", "SQK-RAB"), "rapid"),
    (("SQK-PCB", "EXP-PBC", "SQK-PBK"), "pcr"),
)


def barcode_chemistry(kit: str | None) -> str:
    """Barcoding chemistry of a kit name: ``native_ligation``, ``rapid``,
    ``pcr`` or ``unknown``. Only ``native_ligation`` gets the model."""
    name = str(kit or "").strip().upper()
    for prefixes, chemistry in _CHEMISTRY_PREFIXES:
        if name.startswith(prefixes):
            return chemistry
    return "unknown"


def kit_barcode_count(kit: str | None) -> int | None:
    """Number of barcodes in a packaged kit (None for unknown or custom kits)."""
    from smftools.constants import BARCODE_KIT_ALIASES

    path = BARCODE_KIT_ALIASES.get(str(kit or "").strip())
    if path is None or not Path(path).is_file():
        return None
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    barcodes = data.get("barcodes")
    return len(barcodes) if isinstance(barcodes, dict) else None


def _calls(obs: pd.DataFrame, end: str) -> pd.Series:
    names = obs[f"barcode_{end}"].astype(str).str.strip()
    distance = pd.to_numeric(obs[f"barcode_{end}_edit_distance"], errors="coerce")
    confident = (distance <= CONFIDENT_EDIT_DISTANCE) & ~names.str.lower().isin(UNASSIGNED)
    return names.where(confident)


def _end_class(clip: pd.Series) -> pd.Series:
    clip = pd.to_numeric(clip, errors="coerce")
    return pd.Series(
        np.select(
            [clip <= NO_BARCODE_CLIP, clip.between(*BARCODE_CLIP), clip > BARCODE_CLIP[1]],
            ["none", "barcode", "long"],
            "short",
        ),
        index=clip.index,
    )


def end_evidence(obs: pd.DataFrame, *, spike_in_references: Iterable[str] = ()) -> pd.DataFrame:
    """Per read: end classes, confident calls, full length, spike-in.

    Raises ``KeyError`` naming the missing columns when obs predates `BQC-01`.
    """
    required = [
        "barcode_front",
        "barcode_rear",
        "barcode_front_edit_distance",
        "barcode_rear_edit_distance",
        "front_clip_length",
        "rear_clip_length",
    ]
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(f"obs lacks per-end barcode evidence (BQC-01): {missing}")
    evidence = pd.DataFrame(index=obs.index)
    evidence["front_end"] = _end_class(obs["front_clip_length"])
    evidence["rear_end"] = _end_class(obs["rear_clip_length"])
    evidence["front_call"] = _calls(obs, "front")
    evidence["rear_call"] = _calls(obs, "rear")
    if {"reference_start", "reference_end", "reference_length"} <= set(obs.columns):
        start = pd.to_numeric(obs["reference_start"], errors="coerce")
        end = pd.to_numeric(obs["reference_end"], errors="coerce")
        length = pd.to_numeric(obs["reference_length"], errors="coerce")
        evidence["full_length"] = (start <= SPAN_TOLERANCE) & (end >= length - SPAN_TOLERANCE)
    else:
        evidence["full_length"] = False
    if "has_supplementary_alignment" in obs.columns:
        evidence["supplementary"] = obs["has_supplementary_alignment"].fillna(False).astype(bool)
    else:
        evidence["supplementary"] = False
    references = [str(name) for name in spike_in_references if str(name)]
    if references and REFERENCE_COLUMN in obs.columns:
        evidence["spike"] = spike_in_mask(obs, references).to_numpy()
    else:
        evidence["spike"] = False
    direction = obs.get("Read_mapping_direction", obs.get("mapping_direction"))
    evidence["reverse"] = (
        direction.astype(str).str.lower().str.startswith("rev") if direction is not None else False
    )
    return evidence


def infer_pool(evidence: pd.DataFrame) -> list[str]:
    """Barcodes with a library: confident double-same reads >=
    ``POOL_SHARE_OF_MEDIAN`` of the median such barcode."""
    real = evidence[~evidence["spike"]]
    same = real[real["front_call"].notna() & (real["front_call"] == real["rear_call"])]
    counts = same["front_call"].value_counts()
    if counts.empty:
        return []
    top = counts[counts >= counts.max() * POOL_SHARE_OF_MEDIAN]
    floor = max(10.0, float(top.median()) * POOL_SHARE_OF_MEDIAN)
    return sorted(counts[counts >= floor].index.astype(str))


def _counts(evidence: pd.DataFrame, pool: set[str]) -> dict[str, Any]:
    real = evidence[~evidence["spike"] & ~evidence["supplementary"]]
    both = real[(real["front_end"] == "barcode") & (real["rear_end"] == "barcode")]
    ends = pd.concat([both["front_call"], both["rear_call"]]).dropna()
    in_pool = (both["front_call"].isin(pool)) & (both["rear_call"].isin(pool))
    paired = both[in_pool]
    same = paired["front_call"] == paired["rear_call"]
    return {
        "both_barcoded_reads": int(len(both)),
        "confident_ends": int(len(ends)),
        "confident_out_of_pool": int((~ends.isin(pool)).sum()),
        "double_same": int(same.sum()),
        "mismatch": int((~same).sum()),
        "front_detection": float(both["front_call"].notna().mean()) if len(both) else None,
        "rear_detection": float(both["rear_call"].notna().mean()) if len(both) else None,
        "_paired": paired,
        "_same": same,
    }


def _phi(counts: Mapping[str, Any], n_pool: int, kit_size: int | None) -> dict[str, Any]:
    confident, out = counts["confident_ends"], counts["confident_out_of_pool"]
    pairs = counts["mismatch"] + counts["double_same"]
    unused = (kit_size or 0) - n_pool
    if confident and unused > 0:
        out_rate = out / confident
        miscall = out_rate * (n_pool - 1) / unused
    else:
        out_rate = miscall = None
    mismatch_share = counts["mismatch"] / pairs if pairs else None
    if mismatch_share is None:
        phi = None
    else:
        phi = max(mismatch_share / 2 - (miscall or 0.0), 0.0)
    return {
        "out_of_pool_call_rate": out_rate,
        "in_pool_miscall_rate": miscall,
        "miscall_rate_identifiable": miscall is not None,
        "mismatch_share_of_confident_pairs": mismatch_share,
        "foreign_share_of_barcoded_ends_phi": phi,
    }


def _composition(counts: Mapping[str, Any], pool: list[str]) -> pd.DataFrame:
    paired, same = counts["_paired"], counts["_same"]
    library = paired.loc[same, "front_call"].value_counts().reindex(pool, fill_value=0)
    share = library / library.sum() if library.sum() else library * 0.0
    mismatch = paired[~same]
    ends = (
        pd.concat([mismatch["front_call"], mismatch["rear_call"]])
        .value_counts()
        .reindex(pool, fill_value=0)
    )
    foreign = (ends - len(mismatch) * share).clip(lower=0)
    pi = foreign / foreign.sum() if foreign.sum() else foreign * 0.0
    table = pd.DataFrame(
        {
            "library_double_same": library,
            "library_share": share,
            "mismatch_ends": ends,
            "foreign_ends": foreign,
            "pi": pi,
        }
    )
    table.index.name = "barcode"
    table["pi_over_share"] = table["pi"] / table["library_share"].replace(0, np.nan)
    return table


def _contamination(phi: float | None, table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if phi is None or phi >= 1:
        empty = pd.Series(np.nan, index=table.index)
        return empty, empty
    psi = phi / (1 - phi)
    n = table["library_double_same"]
    own = n.replace(0, np.nan)
    double = psi**2 * table["pi"] ** 2 * (n.sum() - n) / own
    single = psi * table["pi"] / table["library_share"].replace(0, np.nan)
    return double, single


def _bootstrap(
    counts: Mapping[str, Any],
    table: pd.DataFrame,
    n_pool: int,
    kit_size: int | None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    doubles, singles = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        draw = dict(counts)
        for key in ("confident_ends", "confident_out_of_pool", "double_same", "mismatch"):
            draw[key] = int(rng.poisson(counts[key]))
        draw["confident_ends"] = max(draw["confident_ends"], 1)
        phi = _phi(draw, n_pool, kit_size)["foreign_share_of_barcoded_ends_phi"]
        resampled = table.copy()
        foreign = rng.poisson(table["foreign_ends"].to_numpy() + 0.5).astype(float)
        resampled["pi"] = foreign / foreign.sum()
        resampled["library_double_same"] = rng.poisson(table["library_double_same"])
        library = resampled["library_double_same"]
        resampled["library_share"] = library / max(library.sum(), 1)
        double, single = _contamination(phi, resampled)
        doubles.append(double)
        singles.append(single)
    return pd.DataFrame(doubles), pd.DataFrame(singles)


def _ligation(evidence: pd.DataFrame, pool: list[str]) -> tuple[pd.DataFrame, int]:
    """Per library: reads with / without a rear barcode, among full-length
    reads with a confidently called front barcode, and q at both alpha bounds."""
    full = evidence[
        evidence["full_length"]
        & ~evidence["spike"]
        & ~evidence["supplementary"]
        & (evidence["front_end"] == "barcode")
        & evidence["front_call"].isin(pool)
    ]
    rear = full.groupby("front_call")["rear_end"]
    both = rear.apply(lambda x: int((x == "barcode").sum())).reindex(pool, fill_value=0)
    one = rear.apply(lambda x: int((x == "none").sum())).reindex(pool, fill_value=0)
    ratio = both / one.replace(0, np.nan)
    table = pd.DataFrame(
        {
            "rear_barcoded_reads": both,
            "rear_free_reads": one,
            "q_alpha_1": 2 * ratio / (1 + 2 * ratio),
            "q_alpha_0": ratio / (1 + ratio),
        }
    )
    return table, int(len(full))


def _ownership(
    counts: Mapping[str, Any], phi: float | None, miscall: float | None, table: pd.DataFrame
) -> dict[str, Any]:
    mismatch = counts["_paired"][~counts["_same"]]
    if phi is None or phi >= 1 or mismatch.empty:
        return {"mismatch_reads": int(len(mismatch))}
    psi = phi / (1 - phi)
    per_barcode = (miscall or 0.0) / max(len(table) - 1, 1)
    share, pi = table["library_share"], table["pi"]
    x, y = mismatch["front_call"], mismatch["rear_call"]
    sx, sy = share.reindex(x).to_numpy(), share.reindex(y).to_numpy()
    px, py = pi.reindex(x).to_numpy(), pi.reindex(y).to_numpy()
    front = sx * (psi * py + per_barcode)
    rear = sy * (psi * px + per_barcode)
    third = (1 - sx - sy) * psi**2 * px * py
    total = front + rear + third
    with np.errstate(invalid="ignore", divide="ignore"):
        posterior = np.column_stack([front, rear, third]) / total[:, None]
    best = np.nanmax(posterior, axis=1)
    winner = np.nanargmax(np.nan_to_num(posterior, nan=-1.0), axis=1)
    return {
        "mismatch_reads": int(len(mismatch)),
        "owner_posterior_ge_0_95": float(np.mean(best >= 0.95)),
        "front_owner_share": float(np.mean(winner == 0)),
        "rear_owner_share": float(np.mean(winner == 1)),
        "third_sample_share": float(np.mean(winner == 2)),
    }


def _spike_in(evidence: pd.DataFrame) -> dict[str, Any] | None:
    spike = evidence[evidence["spike"]]
    if spike.empty:
        return None
    calls = pd.concat(
        [
            spike.loc[spike["front_end"] == "barcode", "front_call"],
            spike.loc[spike["rear_end"] == "barcode", "rear_call"],
        ]
    ).dropna()
    full = spike[spike["full_length"] & (spike["front_end"] == "barcode")]
    result: dict[str, Any] = {
        "reads": int(len(spike)),
        "confident_end_calls": int(len(calls)),
        "full_length_reads": int(len(full)),
        "lambda_alpha_1": None,
        "lambda_alpha_0": None,
    }
    if len(full) < 20:
        result["both_ends_ligate"] = None
        result["lambda_unavailable_reason"] = (
            "too few spike-in reads cover the reference end to end (is the reference "
            "the whole spike-in molecule?)"
        )
        return result
    # Reads start at a ligated end; a spike-in whose reads all start at one
    # reference end measures that end only.
    reverse = float(full["reverse"].mean())
    both = int((full["rear_end"] == "barcode").sum())
    one = int((full["rear_end"] == "none").sum())
    result.update(
        {
            "share_starting_at_reference_end": reverse,
            "both_ends_ligate": bool(min(reverse, 1 - reverse) >= 0.1),
            "rear_barcoded_reads": both,
            "rear_free_reads": one,
        }
    )
    if not result["both_ends_ligate"]:
        result["lambda_unavailable_reason"] = (
            "spike-in reads start at one reference end only (one end does not ligate)"
        )
    elif one:
        ratio = both / one
        result["lambda_alpha_1"] = 2 * ratio / (1 + 2 * ratio)
        result["lambda_alpha_0"] = ratio / (1 + ratio)
    return result


def barcode_tagging_report(
    obs: pd.DataFrame,
    *,
    barcode_kit: str | None,
    spike_in_references: Iterable[str] = (),
    pool: Iterable[str] | None = None,
    kit_size: int | None = None,
    unclassified_dropped: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the model; return the per-barcode table and the summary."""
    chemistry = barcode_chemistry(barcode_kit)
    summary: dict[str, Any] = {
        "schema_version": TAGGING_SCHEMA_VERSION,
        "barcode_kit": barcode_kit,
        "chemistry": chemistry,
        "confident_edit_distance": CONFIDENT_EDIT_DISTANCE,
        "no_barcode_clip": NO_BARCODE_CLIP,
        "barcode_clip": list(BARCODE_CLIP),
    }
    if chemistry != "native_ligation":
        summary["skipped_reason"] = f"model is for native (two-step) ligation kits, not {chemistry}"
        return pd.DataFrame(), summary
    evidence = end_evidence(obs, spike_in_references=spike_in_references)
    members = sorted({str(b) for b in pool}) if pool else infer_pool(evidence)
    summary["pool"] = members
    summary["pool_source"] = "declared" if pool else "inferred from double-same reads"
    summary["kit_size"] = kit_size
    if not members:
        summary["skipped_reason"] = "no barcode has confident double-same reads"
        return pd.DataFrame(), summary
    member_set = set(members)
    counts = _counts(evidence, member_set)
    summary.update({k: v for k, v in counts.items() if not k.startswith("_")})
    summary.update(_phi(counts, len(members), kit_size))
    if unclassified_dropped:
        summary["phi_caveat"] = (
            "unclassified reads were dropped (skip_unclassified): mismatch reads are missing, phi is a floor"
        )
    table = _composition(counts, members)
    phi = summary["foreign_share_of_barcoded_ends_phi"]
    table["double_contamination"], table["single_contamination"] = _contamination(phi, table)
    doubles, singles = _bootstrap(counts, table, len(members), kit_size)
    for name, draws in (("double_contamination", doubles), ("single_contamination", singles)):
        table[f"{name}_low"] = draws.quantile(0.025).reindex(table.index)
        table[f"{name}_high"] = draws.quantile(0.975).reindex(table.index)
    ligation, full_reads = _ligation(evidence, members)
    table = table.join(ligation)
    summary["full_length_reads"] = full_reads
    summary["q_identifiable"] = full_reads >= MIN_FULL_LENGTH_READS
    if summary["q_identifiable"]:
        both = int(ligation["rear_barcoded_reads"].sum())
        one = int(ligation["rear_free_reads"].sum())
        ratio = both / one if one else np.inf
        q1 = 2 * ratio / (1 + 2 * ratio)
        q0 = ratio / (1 + ratio)
        summary["q_alpha_1"], summary["q_alpha_0"] = float(q1), float(q0)
        if phi is not None and phi < 1:
            psi = phi / (1 - phi)
            summary["lambda_alpha_1"] = float(psi * q1 / (1 - q1)) if q1 < 1 else None
            summary["lambda_alpha_0"] = float(psi * q0 / (1 - q0)) if q0 < 1 else None
    else:
        summary["q_unavailable_reason"] = (
            f"{full_reads} full-length reads (< {MIN_FULL_LENGTH_READS}): a read ending "
            "without a rear barcode may be truncated (genome-wide), so q and lambda are not "
            "identifiable; contamination does not need them"
        )
    real = evidence[~evidence["spike"]]
    summary["long_rear_share"] = float((real["rear_end"] == "long").mean()) if len(real) else None
    summary["empty_pool_barcodes"] = [
        str(b) for b in table.index[table["library_double_same"] == 0]
    ]
    summary["ownership"] = _ownership(counts, phi, summary["in_pool_miscall_rate"], table)
    summary["spike_in"] = _spike_in(evidence)
    for name in ("double_contamination", "single_contamination"):
        values = table[name].dropna()
        summary[f"{name}_median"] = float(values.median()) if len(values) else None
        summary[f"{name}_max"] = float(values.max()) if len(values) else None
        summary[f"{name}_max_barcode"] = str(values.idxmax()) if len(values) else None
    low, high = poisson_interval(counts["mismatch"])
    summary["mismatch_interval"] = [low, high]
    return table, summary


def write_barcode_tagging_qc(
    obs: pd.DataFrame,
    output_dir: str | Path,
    *,
    barcode_kit: str | None,
    spike_in_references: Iterable[str] = (),
    pool: Iterable[str] | None = None,
    kit_size: int | None = None,
    unclassified_dropped: bool = False,
) -> dict[str, Path]:
    """Write the per-barcode table and summary beside the contamination QC."""
    output_dir = Path(output_dir) / CONTAMINATION_QC_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    table, summary = barcode_tagging_report(
        obs,
        barcode_kit=barcode_kit,
        spike_in_references=spike_in_references,
        pool=pool,
        kit_size=kit_size,
        unclassified_dropped=unclassified_dropped,
    )
    table_path = output_dir / TAGGING_BY_BARCODE_FILENAME
    table.reset_index().to_parquet(table_path, index=False)
    summary_path = output_dir / TAGGING_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if "skipped_reason" in summary:
        logger.info("Barcode tagging model skipped: %s", summary["skipped_reason"])
    else:
        logger.info(
            "Barcode tagging: foreign share of barcoded ends %.2e; contamination of "
            "double-same reads median %.1e (max %.1e), of single reads median %.1e (max %.1e)",
            summary["foreign_share_of_barcoded_ends_phi"] or float("nan"),
            summary["double_contamination_median"] or float("nan"),
            summary["double_contamination_max"] or float("nan"),
            summary["single_contamination_median"] or float("nan"),
            summary["single_contamination_max"] or float("nan"),
        )
    return {"per_barcode": table_path, "summary": summary_path}
