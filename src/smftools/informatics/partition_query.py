"""Predicate-pruned Parquet indexes and bounded Zarr selection helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_QUERY_MEMORY_MB = 256


def _as_values(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return sorted({str(item) for item in value})


def _and(expressions):
    result = None
    for expression in expressions:
        if expression is not None:
            result = expression if result is None else result & expression
    return result


def query_molecule_index(
    index_path: str | Path,
    *,
    references=None,
    samples=None,
    barcodes=None,
    read_ids=None,
    molecule_uids=None,
    start: int | None = None,
    end: int | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Query the raw molecule dataset with PyArrow predicate pushdown."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(index_path), format="parquet")
    available = set(dataset.schema.names)
    requested = {
        "references": ("Reference_strand", _as_values(references)),
        "samples": ("Sample", _as_values(samples)),
        "barcodes": ("Barcode", _as_values(barcodes)),
        "read_ids": ("read_id", _as_values(read_ids)),
        "molecule_uids": ("molecule_uid", _as_values(molecule_uids)),
    }
    expressions = []
    for label, (field, values) in requested.items():
        if values is None:
            continue
        if field not in available:
            raise KeyError(f"molecule index cannot filter {label}: missing column {field!r}")
        expressions.append(ds.field(field).isin(values))
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    if start is not None:
        expressions.extend(
            [
                ds.field("reference_start") < int(end),
                ds.field("reference_end") > int(start),
            ]
        )
    selected_columns = None if columns is None else [name for name in columns if name in available]
    frame = dataset.to_table(filter=_and(expressions), columns=selected_columns).to_pandas()
    order = [column for column in ("canonical_row", "group_path", "group_row") if column in frame]
    return frame.sort_values(order, kind="stable").reset_index(drop=True) if order else frame


def query_derived_index(
    index_path: str | Path,
    *,
    references=None,
    barcodes=None,
    read_ids=None,
    molecule_uids=None,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Query one stage's read-to-task dataset before opening task Zarr stores."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(index_path), format="parquet")
    expressions = []
    for field, values in (
        ("reference", _as_values(references)),
        ("barcode", _as_values(barcodes)),
        ("read_id", _as_values(read_ids)),
        ("molecule_uid", _as_values(molecule_uids)),
    ):
        if values is not None:
            expressions.append(ds.field(field).isin(values))
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    if start is not None:
        expressions.extend(
            [
                ds.field("core_start") < int(end),
                ds.field("core_end") > int(start),
            ]
        )
    frame = dataset.to_table(filter=_and(expressions)).to_pandas()
    identity = [
        column
        for column in ("molecule_uid", "group_path", "group_row", "task_id")
        if column in frame
    ]
    if identity:
        frame = frame.drop_duplicates(identity)
    order = [
        column
        for column in ("reference", "core_start", "barcode", "group_path", "group_row")
        if column in frame
    ]
    return frame.sort_values(order, kind="stable").reset_index(drop=True) if order else frame


def query_batch_rows(
    n_positions: int,
    *,
    n_arrays: int = 1,
    memory_mb: int = DEFAULT_QUERY_MEMORY_MB,
) -> int:
    """Resolve a conservative read batch from consumer memory and projection width."""
    if memory_mb <= 0:
        raise ValueError("query memory_mb must be positive")
    # float32-equivalent bytes plus a 2x allowance for serialization/transient copies.
    bytes_per_row = max(1, int(n_positions)) * max(1, int(n_arrays)) * 4 * 2
    return max(1, int(memory_mb) * 1024**2 // bytes_per_row)


def _position_selector(var_names, start: int | None, end: int | None):
    if start is None:
        return slice(None)
    positions = np.asarray(var_names, dtype=np.int64)
    return np.flatnonzero((positions >= int(start)) & (positions < int(end)))


def read_zarr_subset(
    path: str | Path,
    *,
    row_indices: Iterable[int] | None = None,
    read_ids: Iterable[str] | None = None,
    start: int | None = None,
    end: int | None = None,
    layers: Iterable[str] | None = None,
    lazy: bool | None = None,
    memory_mb: int = DEFAULT_QUERY_MEMORY_MB,
):
    """Read bounded rows and positions, slicing lazy arrays before ``to_memory``."""
    import anndata as ad

    if row_indices is not None and read_ids is not None:
        raise ValueError("supply row_indices or read_ids, not both")
    requested_layers = None if layers is None else set(map(str, layers))

    def _project(obj, rows):
        projected = obj[rows, _position_selector(obj.var_names, start, end)]
        if requested_layers is None:
            return projected
        # Deleting keys from a lazy AnnData view initializes the view and can
        # compute every layer. Rebuild only the lightweight container metadata
        # around the requested lazy arrays so projection still precedes compute.
        obs = projected.obs.to_memory() if hasattr(projected.obs, "to_memory") else projected.obs
        var = projected.var.to_memory() if hasattr(projected.var, "to_memory") else projected.var
        return ad.AnnData(
            X=projected.X,
            obs=obs,
            var=var,
            layers={
                name: projected.layers[name]
                for name in requested_layers.intersection(projected.layers)
            },
            obsm=dict(projected.obsm),
            varm=dict(projected.varm),
            obsp=dict(projected.obsp),
            varp=dict(projected.varp),
            uns=dict(projected.uns),
        )

    use_lazy = lazy is not False
    if use_lazy:
        try:
            from anndata.experimental import read_lazy

            source = read_lazy(path)
            rows: np.ndarray | list[str] | slice = (
                np.asarray(list(row_indices), dtype=np.int64)
                if row_indices is not None
                else list(map(str, read_ids))
                if read_ids is not None
                else slice(None)
            )
            row_count = source.n_obs if isinstance(rows, slice) else len(rows)
            width = (
                len(_position_selector(source.var_names, start, end))
                if start is not None
                else source.n_vars
            )
            batch_size = query_batch_rows(
                width,
                n_arrays=1
                + (len(requested_layers) if requested_layers is not None else len(source.layers)),
                memory_mb=memory_mb,
            )
            if row_count == 0:
                return _project(source, np.asarray([], dtype=np.int64)).to_memory()
            parts = []
            for offset in range(0, row_count, batch_size):
                batch_rows = (
                    slice(offset, min(offset + batch_size, row_count))
                    if isinstance(rows, slice)
                    else rows[offset : offset + batch_size]
                )
                parts.append(_project(source, batch_rows).to_memory())
            if len(parts) == 1:
                return parts[0]
            return ad.concat(parts, join="outer", merge="first", uns_merge="first")
        except Exception:
            if lazy is True:
                raise

    from ..readwrite import safe_read_zarr

    source, _ = safe_read_zarr(path, verbose=False)
    rows = (
        np.asarray(list(row_indices), dtype=np.int64)
        if row_indices is not None
        else list(map(str, read_ids))
        if read_ids is not None
        else slice(None)
    )
    return _project(source, rows).copy()
