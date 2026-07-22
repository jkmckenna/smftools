"""Portable physical chunk sizing independent of logical pipeline tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ZARR_CHUNK_BYTES = 16 * 1024**2
DEFAULT_PARQUET_ROW_GROUP_BYTES = 64 * 1024**2


def portable_matrix_chunks(
    shape: tuple[int, int],
    dtype: np.dtype | type,
    *,
    target_bytes: int = DEFAULT_ZARR_CHUNK_BYTES,
    max_rows: int = 2_048,
    max_columns: int = 8_192,
) -> tuple[int, int]:
    """Return bounded 2-D chunks independent of a logical task's full width."""
    n_rows, n_columns = map(int, shape)
    if n_rows <= 0 or n_columns <= 0:
        raise ValueError("chunked matrix dimensions must be positive")
    if target_bytes <= 0 or max_rows <= 0 or max_columns <= 0:
        raise ValueError("chunk limits must be positive")
    rows = min(n_rows, max_rows)
    bytes_per_value = max(1, np.dtype(dtype).itemsize)
    columns_for_budget = max(1, target_bytes // (rows * bytes_per_value))
    columns = min(n_columns, max_columns, columns_for_budget)
    return rows, columns


def portable_parquet_row_group_rows(
    frame: pd.DataFrame,
    *,
    target_bytes: int = DEFAULT_PARQUET_ROW_GROUP_BYTES,
    max_rows: int = 250_000,
) -> int:
    """Estimate a bounded Parquet row-group length from in-memory row width."""
    if target_bytes <= 0 or max_rows <= 0:
        raise ValueError("row-group limits must be positive")
    if frame.empty:
        return 1
    bytes_per_row = max(1, int(frame.memory_usage(index=False, deep=True).sum()) // len(frame))
    return max(1, min(len(frame), max_rows, target_bytes // bytes_per_row))
