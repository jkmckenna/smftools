"""Per-molecule indexes for partitioned derived task artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .molecule_identity import (
    EXPERIMENT_UID_COLUMN,
    MOLECULE_UID_COLUMN,
    molecule_uid,
    validate_experiment_uid,
)
from .physical_layout import portable_parquet_row_group_rows

DERIVED_READ_INDEX_DIRNAME = "read_index"
DERIVED_READ_INDEX_SCHEMA_VERSION = 1


def prepare_derived_read_index(output_dir: str | Path) -> Path:
    """Create an empty stage index directory before bounded task writers run."""
    path = Path(output_dir) / DERIVED_READ_INDEX_DIRNAME
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_derived_read_index(
    output_dir: str | Path,
    *,
    stage: str,
    task,
    obs: pd.DataFrame,
    group_path: str | None,
    stage_schema_version: int,
    model_artifacts: Iterable[Mapping[str, object]] = (),
) -> Path:
    """Write one bounded read-to-task index shard in stored observation order."""
    output_dir = Path(output_dir)
    experiment_values = obs.get(EXPERIMENT_UID_COLUMN)
    if experiment_values is None or experiment_values.isna().any():
        raise ValueError(f"{stage} task {task.task_id!r} lacks experiment_uid identity")
    unique_experiments = experiment_values.astype(str).unique()
    if len(unique_experiments) != 1:
        raise ValueError(f"{stage} task {task.task_id!r} spans multiple experiments")
    experiment_uid = validate_experiment_uid(unique_experiments[0])
    read_ids = obs.get("read_id", pd.Series(obs.index.astype(str), index=obs.index)).astype(str)
    molecule_uids = obs.get(MOLECULE_UID_COLUMN)
    expected_molecule_uids = pd.Series(
        [molecule_uid(experiment_uid, read_id) for read_id in read_ids], index=obs.index
    )
    if molecule_uids is None:
        molecule_uids = expected_molecule_uids
    elif not molecule_uids.astype(str).equals(expected_molecule_uids):
        raise ValueError(f"{stage} task {task.task_id!r} has inconsistent molecule_uid values")

    artifacts = list(model_artifacts) or [None]
    rows: list[dict[str, object]] = []
    for group_row, (read_id, this_molecule_uid) in enumerate(
        zip(read_ids, molecule_uids.astype(str))
    ):
        for artifact in artifacts:
            rows.append(
                {
                    EXPERIMENT_UID_COLUMN: experiment_uid,
                    "read_id": read_id,
                    MOLECULE_UID_COLUMN: this_molecule_uid,
                    "stage": str(stage),
                    "task_id": str(task.task_id),
                    "reference": str(task.reference),
                    "core_start": int(task.core_start),
                    "core_end": int(task.core_end),
                    "load_start": int(task.load_start),
                    "load_end": int(task.load_end),
                    "barcode": str(task.barcode),
                    "chunk_index": int(task.chunk_index),
                    "group_path": group_path,
                    "group_row": group_row if group_path is not None else None,
                    "model_id": None if artifact is None else str(artifact.get("model_id")),
                    "model_checksum": (
                        None
                        if artifact is None
                        else str(
                            artifact.get("model_checksum", artifact.get("checkpoint_sha256", ""))
                        )
                    ),
                    "stage_schema_version": int(stage_schema_version),
                    "index_schema_version": DERIVED_READ_INDEX_SCHEMA_VERSION,
                }
            )

    digest = hashlib.sha256(str(task.task_id).encode("utf-8")).hexdigest()[:20]
    path = output_dir / DERIVED_READ_INDEX_DIRNAME / f"task-{digest}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in (
        EXPERIMENT_UID_COLUMN,
        "read_id",
        MOLECULE_UID_COLUMN,
        "stage",
        "task_id",
        "reference",
        "barcode",
        "group_path",
        "model_id",
        "model_checksum",
    ):
        frame[column] = frame[column].astype("string")
    frame["group_row"] = frame["group_row"].astype("Int64")
    temporary_path = path.with_suffix(".tmp.parquet")
    try:
        frame.to_parquet(
            temporary_path,
            index=False,
            row_group_size=portable_parquet_row_group_rows(frame),
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
