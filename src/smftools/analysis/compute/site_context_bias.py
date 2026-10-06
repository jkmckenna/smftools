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
        sequence = sequences[key].upper()
        start, end = int(position) - flank, int(position) + flank + 1
        window = (
            "N" * max(0, -start)
            + sequence[max(0, start) : min(len(sequence), end)]
            + "N" * max(0, end - len(sequence))
        )
        contexts.append(window.translate(_COMPLEMENT)[::-1] if strand == "bottom" else window)
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

    ``drop_ambiguous`` drops k-mers containing a non-ACGT base.
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
