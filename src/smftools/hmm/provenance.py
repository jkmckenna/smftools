"""Which model produced a read's HMM layers (`DLS-02`).

Every HMM generation records, per read, the immutable model each decode task
applied (the derived read index, ``read_index/``), and per model how it was
fitted (``fit_catalog.parquet``). :func:`hmm_read_models` joins the two, so a
stored posterior -- or any layer derived from it -- can be traced to its model,
fit settings, emission group and efficiency offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

#: Fit-catalog columns carried onto each read, when the catalog has them.
FIT_COLUMNS = (
    "fit_id",
    "model_label",
    "model_architecture",
    "fit_config_hash",
    "emission_scope",
    "emission_group",
    "efficiency_offset",
)


def hmm_read_models(
    generation: str | Path,
    read_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per read and model of an HMM generation directory.

    Columns: ``read_id``, ``molecule_uid``, ``task_id``, ``reference``,
    ``barcode``, ``model_id``, ``model_checksum`` and the fit catalog's
    ``FIT_COLUMNS``. ``read_ids`` restricts the rows.
    """
    generation = Path(generation)
    shards = sorted((generation / "read_index").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"{generation} has no read_index/ (not an HMM generation?)")
    columns = [
        "read_id",
        "molecule_uid",
        "task_id",
        "reference",
        "barcode",
        "model_id",
        "model_checksum",
    ]
    wanted = None if read_ids is None else set(map(str, read_ids))
    parts = []
    for shard in shards:
        frame = pd.read_parquet(shard, columns=columns)
        if wanted is not None:
            frame = frame[frame["read_id"].astype(str).isin(wanted)]
        if len(frame):
            parts.append(frame)
    reads = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    catalog_path = generation / "fit_catalog.parquet"
    if not catalog_path.is_file():
        return reads
    catalog = pd.read_parquet(catalog_path)
    if "model_id" not in catalog.columns:
        return reads
    keep = ["model_id", *[c for c in FIT_COLUMNS if c in catalog.columns]]
    catalog = catalog[keep].dropna(subset=["model_id"]).drop_duplicates("model_id")
    return reads.merge(catalog, on="model_id", how="left")
