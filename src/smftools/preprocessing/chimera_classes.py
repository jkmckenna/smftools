"""Composite chimera classification across independent detection methods.

Three columns describe chimerism, and they are not interchangeable:

| column | axis | method |
|---|---|---|
| `chimeric_variant_sites` | reference identity | a read carrying a segment of the *other allele* |
| `deaminase_PCR_chimera` | strand chemistry | per-read `C->T`/`G->A` totals plus a two-segment purity |
| `deaminase_segment_chimera` | strand chemistry | located segments from penalized change points (`EGL-19`) |

The first is orthogonal to the other two: a molecule can join two strands of one
allele, two alleles of one strand, or both. The second and third answer the same
question by different means and agree on only 3 of 18 molecules on the `251105`
pilot, which is why both are kept rather than one replacing the other.

`omit_chimeric_reads` consumes the union (decided 2026-08-20). The union is
materialized as its own column rather than computed at each call site so that a
surprising exclusion can be decomposed into which method fired, and so that a
missing method degrades in one place instead of five.
"""

from __future__ import annotations

import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

VARIANT_CHIMERA_COLUMN = "chimeric_variant_sites"
DEAMINASE_SCALAR_COLUMN = "deaminase_PCR_chimera"
DEAMINASE_SEGMENT_COLUMN = "deaminase_segment_chimera"
COMPOSITE_COLUMN = "is_chimeric_any"

CHIMERA_METHOD_COLUMNS = (
    VARIANT_CHIMERA_COLUMN,
    DEAMINASE_SCALAR_COLUMN,
    DEAMINASE_SEGMENT_COLUMN,
)


def _as_bool(series: pd.Series) -> pd.Series:
    """Interpret a chimera column as booleans, including string-valued ones.

    Reuses the plotting coercion rather than `astype(bool)`: these columns
    round-trip through parquet as categories of `"True"`/`"False"`, and
    `bool("False")` is `True` (`F13`).
    """
    from smftools.plotting.plotting_utils import coerce_bool_series

    return coerce_bool_series(series)


def append_composite_chimera_column(obs: pd.DataFrame) -> pd.DataFrame:
    """Add the union of whichever chimera methods produced a column.

    A method that did not run -- deamination bypassed (`EGL-25`), `direct`
    modality, variant reporting off -- contributes nothing rather than
    contributing `False` for every read. The distinction matters: treating an
    absent column as "not chimeric" would silently turn a disabled detector into
    a clean bill of health, which is the failure shape of `F13` and `F17`.
    """
    present = [column for column in CHIMERA_METHOD_COLUMNS if column in obs.columns]
    if not present:
        logger.info("No chimera method columns present; %s not added", COMPOSITE_COLUMN)
        return obs
    composite = pd.Series(False, index=obs.index)
    for column in present:
        composite |= _as_bool(obs[column])
    obs[COMPOSITE_COLUMN] = composite
    logger.info(
        "Composite chimera column from %s: %d of %d read(s) chimeric by at least one method",
        list(present),
        int(composite.sum()),
        len(composite),
    )
    return obs
