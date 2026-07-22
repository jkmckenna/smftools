"""Per-stage ``obs.parquet``: only the columns a stage newly produced, keyed by read_id.

Formalizes the ``obs`` analog from ``dev/experiment_storage_schema.md`` (Phase 3,
partial). This module covers writing/reading per-stage ``obs.parquet`` and joining
them back into a full view -- it deliberately does **not** touch ``spine.h5ad`` or its
role as the read path's obs source (that's the separate, larger "single
``experiment_spine.h5ad``, replacing per-stage ``spine.h5ad``" change, which touches
``project/registry.py``'s path semantics and every existing spine consumer -- not done
here). ``obs.parquet`` is written *alongside* the existing ``spine.h5ad``, purely
additive.

A stage's full obs view is every earlier stage's ``obs.parquet`` plus its own, joined
by ``read_id`` (inner join -- a later stage's molecule set is always a subset of an
earlier one's, per the schema's own model, so this converges to the last stage's row
set regardless of join order).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OBS_FILENAME = "obs.parquet"


def obs_parquet_path(stage_dir: str | Path, *, filename: str = OBS_FILENAME) -> Path:
    return Path(stage_dir) / filename


def write_stage_obs(
    stage_dir: str | Path,
    obs: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    filename: str = OBS_FILENAME,
) -> Path:
    """Write a stage's ``obs.parquet``: ``read_id`` (from ``obs``'s index) plus either
    ``columns`` (the stage's newly-produced columns only -- the normalized case) or
    every column in ``obs`` when ``columns`` is ``None`` (the raw-stage case: nothing
    to normalize away, since there's no earlier stage's obs to avoid re-storing).

    ``filename`` defaults to ``obs.parquet`` but can be overridden when a stage
    already owns that name for an unrelated artifact (e.g. preprocess's QC sidecar,
    which is a different, denormalized shape used as internal working state) -- see
    ``partitioned_executor.py``'s ``PREPROCESS_OBS_SIDECAR``.
    """
    selected = obs if columns is None else obs[list(columns)]
    frame = selected.copy()
    index_read_ids = frame.index.astype(str)
    if "read_id" in frame:
        if (
            not frame["read_id"]
            .astype(str)
            .reset_index(drop=True)
            .equals(pd.Series(index_read_ids, dtype=object))
        ):
            raise ValueError("obs read_id column must match its observation index")
    else:
        frame.insert(0, "read_id", index_read_ids)
    frame = frame.reset_index(drop=True)
    path = obs_parquet_path(stage_dir, filename=filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_stage_obs(stage_dir: str | Path, *, filename: str = OBS_FILENAME) -> pd.DataFrame:
    """Read one stage's ``obs.parquet`` back, indexed by ``read_id``."""
    frame = pd.read_parquet(obs_parquet_path(stage_dir, filename=filename))
    return frame.set_index("read_id", drop=False)


def read_joined_obs(stage_dirs: list[str | Path]) -> pd.DataFrame:
    """Join multiple stages' ``obs.parquet`` by ``read_id``, earliest first.

    Inner join at each step: a later stage's row set is always a subset of an
    earlier one's (post-QC/dedup filtering), so the result converges to the last
    stage's row set. If two stages' files somehow share a column name (shouldn't
    happen if each stage only ever writes its own newly-produced columns), the
    earlier stage's value wins -- last-write-wins would silently prefer whichever
    stage happened to be iterated last, which is a worse default for provenance.
    """
    frames = [read_stage_obs(stage_dir) for stage_dir in stage_dirs]
    if not frames:
        return pd.DataFrame()
    joined = frames[0]
    for frame in frames[1:]:
        overlap = [column for column in frame.columns if column in joined.columns]
        joined = joined.join(frame.drop(columns=overlap), how="inner")
    return joined
