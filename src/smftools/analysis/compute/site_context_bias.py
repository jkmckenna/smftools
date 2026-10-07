"""Sequence-context bias of modification sites (`SCB`): pure statistics.

The unit is a *site*: one position of a channel's site type on one physical
reference, within one group. Per site, ``observed`` counts molecules with a
call there (0 or 1) and ``modified`` those with a 1. Everything here is a
function of that site table and the reference sequences; reading molecules
is `smftools.tools.site_context_bias`.

Contexts are strand-oriented. Sequences are forward strand; a site on a
``*_bottom`` reference is the reverse complement of the forward window, so
every context reads 5'->3' on the modified strand with the site at the centre.
Windows running off a reference end are padded with ``N``.

Observed calls are the background: the context distribution of every
potential site of the type, weighted by how often it was measured. Enrichment
compares the modified-weighted distribution against it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

SITE_COLUMNS = ["group", "physical_reference", "position", "observed", "modified"]
BASES = ("A", "C", "G", "T", "N")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_STRAND_SUFFIXES = ("_top", "_bottom")


def strand_of(reference: str) -> tuple[str, str]:
    """``("6B6", "bottom")`` for ``6B6_bottom``; references without a suffix are top."""
    for suffix in _STRAND_SUFFIXES:
        if reference.endswith(suffix):
            return reference[: -len(suffix)], suffix[1:]
    return reference, "top"


def accumulate_site_calls(
    counts: dict,
    *,
    keys: Sequence[tuple[str, str]],
    calls: np.ndarray,
    observed: np.ndarray,
) -> dict:
    """Add one batch's calls to ``counts``.

    ``keys`` gives each row's ``(group, physical_reference)``; ``calls`` and
    ``observed`` are rows x positions. ``counts`` maps a key to its running
    ``(observed, modified)`` arrays over positions.
    """
    calls = np.asarray(calls)
    observed = np.asarray(observed, dtype=bool)
    modified = observed & (calls == 1)
    codes, unique = pd.factorize(pd.Series([f"{g}\x00{r}" for g, r in keys]))
    for code, label in enumerate(unique):
        rows = codes == code
        key = tuple(label.split("\x00"))
        add_observed = observed[rows].sum(axis=0, dtype=np.int64)
        add_modified = modified[rows].sum(axis=0, dtype=np.int64)
        if key in counts:
            counts[key][0] += add_observed
            counts[key][1] += add_modified
        else:
            counts[key] = [add_observed, add_modified]
    return counts


def site_table(counts: Mapping, positions: Sequence[int]) -> pd.DataFrame:
    """One row per (group, physical reference, position) with at least one call."""
    positions = np.asarray(positions, dtype=np.int64)
    frames = []
    for (group, reference), (observed, modified) in sorted(counts.items()):
        keep = np.asarray(observed) > 0
        frames.append(
            pd.DataFrame(
                {
                    "group": group,
                    "physical_reference": reference,
                    "position": positions[keep],
                    "observed": np.asarray(observed)[keep],
                    "modified": np.asarray(modified)[keep],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=SITE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def strand_window(sequence: str, strand: str, position: int, flank: int) -> str:
    """The ``2 * flank + 1`` window at ``position``, 5'->3' on the modified strand.

    ``sequence`` is forward strand; a bottom-strand window is reverse
    complemented. Positions past either end read ``N``.
    """
    sequence = sequence.upper()
    start, end = position - flank, position + flank + 1
    window = (
        "N" * max(0, -start)
        + sequence[max(0, start) : min(len(sequence), end)]
        + "N" * max(0, end - len(sequence))
    )
    return window.translate(_COMPLEMENT)[::-1] if strand == "bottom" else window


def site_contexts(
    sites: pd.DataFrame,
    sequences: Mapping[str, str],
    *,
    flank: int,
    sequence_for: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Add each site's strand-oriented ``2 * flank + 1`` context and its rate.

    ``sequences`` maps a reference (strand suffix removed) to its forward
    sequence. ``sequence_for`` optionally maps a physical reference to the
    sequence key to use instead -- e.g. the frame reference of a dataset whose
    molecules were placed in another reference's coordinates.
    """
    if flank < 0:
        raise ValueError("flank must be >= 0")
    contexts = []
    for reference, position in zip(sites["physical_reference"], sites["position"], strict=True):
        base, strand = strand_of(str(reference))
        key = (sequence_for or {}).get(str(reference), base)
        if key not in sequences:
            raise KeyError(f"no sequence for reference {key!r}")
        contexts.append(strand_window(sequences[key], strand, int(position), flank))
    out = sites.copy()
    out["context"] = contexts
    out["rate"] = out["modified"] / out["observed"]
    return out


def unambiguous(sites: pd.DataFrame, column: str = "context") -> pd.DataFrame:
    """Sites whose ``column`` is all A/C/G/T.

    Drops windows touching reference ``N`` (masked bases or reference-end
    padding) from modified and background calls alike: few sites carry them,
    so their enrichments are large and noisy and swamp the rest.
    """
    return sites.loc[sites[column].str.fullmatch("[ACGT]*")]


def offset_enrichment(
    sites: pd.DataFrame, *, flank: int, pseudocount: float = 0.5, drop_ambiguous: bool = True
) -> pd.DataFrame:
    """Per group, offset and base: modified- vs observed-weighted frequency.

    ``log2_enrichment = log2((m_b + c) / (M + 5c)) - log2((o_b + c) / (O + 5c))``
    with ``c`` the pseudocount over the five bases (``N`` included); NaN where
    a base has no observed calls at that offset. ``drop_ambiguous`` removes
    sites whose context contains a non-ACGT base first (`unambiguous`).
    """
    if drop_ambiguous:
        sites = unambiguous(sites)
    rows = []
    for group, frame in sites.groupby("group", sort=True):
        letters = np.array([list(context) for context in frame["context"]])
        observed = frame["observed"].to_numpy(dtype=float)
        modified = frame["modified"].to_numpy(dtype=float)
        total_observed, total_modified = observed.sum(), modified.sum()
        for index in range(2 * flank + 1):
            column = letters[:, index] if len(letters) else np.array([])
            for base in BASES:
                hit = column == base
                o, m = observed[hit].sum(), modified[hit].sum()
                mod_freq = (m + pseudocount) / (total_modified + pseudocount * len(BASES))
                obs_freq = (o + pseudocount) / (total_observed + pseudocount * len(BASES))
                rows.append(
                    {
                        "group": group,
                        "offset": index - flank,
                        "base": base,
                        "observed": o,
                        "modified": m,
                        "mod_freq": mod_freq,
                        "obs_freq": obs_freq,
                        # No observed calls: no information, not an enrichment
                        # the pseudocount would otherwise invent.
                        "log2_enrichment": (
                            float(np.log2(mod_freq / obs_freq)) if o > 0 else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def wilson_interval(successes, trials, *, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportions (elementwise)."""
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half, centre + half


def kmer_rates(
    sites: pd.DataFrame, *, flank: int, k: int, drop_ambiguous: bool = True
) -> pd.DataFrame:
    """Per group and centred ``k``-mer: calls, rate, Wilson interval, distinct sites.

    ``drop_ambiguous`` drops k-mers containing a non-ACGT base. Relative columns
    put groups with different overall activity on one scale:
    ``log2_relative_rate = log2(rate / overall_rate)``, ``overall_rate`` being
    the group's rate over every k-mer kept (its Wilson bounds scale alike).
    """
    if k < 1 or k % 2 == 0 or k > 2 * flank + 1:
        raise ValueError(f"k must be odd and between 1 and {2 * flank + 1}")
    half = k // 2
    frame = sites.assign(kmer=sites["context"].str[flank - half : flank + half + 1])
    if drop_ambiguous:
        frame = unambiguous(frame, "kmer")
    table = (
        frame.groupby(["group", "kmer"], sort=True)
        .agg(
            observed=("observed", "sum"),
            modified=("modified", "sum"),
            n_sites=("position", "size"),
        )
        .reset_index()
    )
    table["rate"] = table["modified"] / table["observed"]
    table["rate_low"], table["rate_high"] = wilson_interval(table["modified"], table["observed"])
    totals = table.groupby("group")[["modified", "observed"]].transform("sum")
    table["overall_rate"] = totals["modified"] / totals["observed"]
    with np.errstate(divide="ignore", invalid="ignore"):
        for column, source in (
            ("log2_relative_rate", "rate"),
            ("log2_relative_low", "rate_low"),
            ("log2_relative_high", "rate_high"),
        ):
            table[column] = np.log2(table[source] / table["overall_rate"])
    return table


def group_differences(enrichment: pd.DataFrame, *, reference_group: str) -> pd.DataFrame:
    """Each group's ``log2_enrichment`` minus the reference group's, per offset and base."""
    if reference_group not in set(enrichment["group"]):
        raise KeyError(f"reference group {reference_group!r} is not among the groups")
    reference = enrichment.loc[enrichment["group"] == reference_group].set_index(
        ["offset", "base"]
    )["log2_enrichment"]
    others = enrichment.loc[enrichment["group"] != reference_group]
    keyed = others.set_index(["offset", "base"])
    out = others[["group", "offset", "base"]].copy()
    out["log2_enrichment"] = keyed["log2_enrichment"].to_numpy()
    out["reference_log2_enrichment"] = reference.reindex(keyed.index).to_numpy()
    out["difference"] = out["log2_enrichment"] - out["reference_log2_enrichment"]
    out["reference_group"] = reference_group
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Context codes and weight tables for context-aware HMM emissions (`HCE-01`)
# ---------------------------------------------------------------------------

NOT_A_CONTEXT = -1  # not a C on the modified strand, or a window touching N
WEIGHT_COLUMNS = ["group", "k", "kmer", "weight", "n_sites", "observed", "source"]
# accessible: modification within HMM accessible-called sites (`SCQ-02`).
WEIGHT_SOURCES = frozenset({"naked_dna", "learned", "cells", "accessible"})


def context_kmers(k: int) -> list[str]:
    """Every centred ``k``-mer with C at its centre, in code order (4^(k-1) of them)."""
    from itertools import product

    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    half = k // 2
    return [
        "".join(left) + "C" + "".join(right)
        for left in product("ACGT", repeat=half)
        for right in product("ACGT", repeat=half)
    ]


def context_index(
    sequence: str, strand: str, positions, *, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Per position: its centred ``k``-mer code (`context_kmers` order) and a CpG flag.

    Windows read as `strand_window` (5'->3' on the modified strand). A position
    whose centre is not C on that strand, or whose window touches ``N``, gets
    `NOT_A_CONTEXT` -- never weighted. The CpG flag marks a G immediately 3'
    of the C on the modified strand.
    """
    code_of = {kmer: code for code, kmer in enumerate(context_kmers(k))}
    half = k // 2
    positions = np.asarray(positions, dtype=np.int64)
    codes = np.full(positions.size, NOT_A_CONTEXT, dtype=np.int64)
    cpg = np.zeros(positions.size, dtype=bool)
    for index, position in enumerate(positions):
        window = strand_window(sequence, strand, int(position), max(half, 1))
        centre = len(window) // 2
        if window[centre] != "C":
            continue
        cpg[index] = window[centre + 1] == "G"
        kmer = window[centre - half : centre + half + 1]
        codes[index] = code_of.get(kmer, NOT_A_CONTEXT)
    return codes, cpg


def weight_table(rates: pd.DataFrame, *, k: int, source: str) -> pd.DataFrame:
    """Relative k-mer weights (smoothed rate over the group's overall rate) from `kmer_rates`.

    ``source`` records where the rates came from: ``naked_dna`` (every site
    accessible: enzyme preference alone), ``learned`` (an HMM fit) or ``cells``
    (chromatin and methylation confound it -- not for correcting cells).
    """
    if source not in WEIGHT_SOURCES:
        raise ValueError(f"source must be one of {sorted(WEIGHT_SOURCES)}")
    kmers = set(context_kmers(k))
    frame = rates.loc[rates["kmer"].isin(kmers)]
    if "k" in frame:
        frame = frame.loc[frame["k"] == k]
    table = pd.DataFrame(
        {
            "group": frame["group"].astype(str).to_numpy(),
            "k": k,
            "kmer": frame["kmer"].to_numpy(),
            "weight": _smoothed_weights(frame),
            "n_sites": frame["n_sites"].to_numpy(),
            "observed": frame["observed"].to_numpy(),
            "source": source,
        }
    )
    return table.sort_values(["group", "kmer"]).reset_index(drop=True)


def _smoothed_weights(frame: pd.DataFrame, pseudocount: float = 0.5) -> np.ndarray:
    """``(modified + c) / (observed + 2c)`` over the group's overall rate.

    Smoothed so a k-mer never modified in the data is unlikely, not impossible
    (a weight of 0 would forbid modification there in the HMM).
    """
    modified = frame["modified"].to_numpy(dtype=float)
    observed = frame["observed"].to_numpy(dtype=float)
    rate = (modified + pseudocount) / (observed + 2 * pseudocount)
    return rate / frame["overall_rate"].to_numpy(dtype=float)


def write_weight_table(table: pd.DataFrame, path) -> None:
    _check_weight_table(table)
    path = str(path)
    if path.endswith(".parquet"):
        table[WEIGHT_COLUMNS].to_parquet(path, index=False)
    else:
        table[WEIGHT_COLUMNS].to_csv(path, index=False)


def read_weight_table(path) -> pd.DataFrame:
    path = str(path)
    table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    table["group"] = table["group"].astype(str)
    _check_weight_table(table)
    return table


def weights_for(table: pd.DataFrame, group: str, k: int) -> np.ndarray:
    """Weights in `context_kmers` code order for one group; 1 where a k-mer is absent."""
    frame = table.loc[(table["group"] == str(group)) & (table["k"] == k)]
    if frame.empty:
        raise KeyError(f"no k={k} weights for group {group!r}")
    by_kmer = dict(zip(frame["kmer"], frame["weight"], strict=True))
    return np.array([by_kmer.get(kmer, 1.0) for kmer in context_kmers(k)], dtype=float)


def _check_weight_table(table: pd.DataFrame) -> None:
    missing = [column for column in WEIGHT_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"weight table lacks columns {missing}")
    bad = set(table["source"]) - WEIGHT_SOURCES
    if bad:
        raise ValueError(f"unknown weight sources {sorted(bad)}")
    if (table["weight"] <= 0).any() or not np.isfinite(table["weight"]).all():
        raise ValueError("weights must be finite and positive")
