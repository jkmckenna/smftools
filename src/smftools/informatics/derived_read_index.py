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
    TEMPLATE_ID_COLUMN,
    molecule_uid,
    validate_experiment_uid,
)
from .physical_layout import portable_parquet_row_group_rows

DERIVED_READ_INDEX_DIRNAME = "read_index"
DERIVED_READ_INDEX_SCHEMA_VERSION = 1
LATENT_READ_INDEX_SCHEMA_VERSION = 3
_LATENT_MOLECULE_BUCKET_LENGTH = 2


def molecule_index_bucket(value: object) -> str:
    """Return the stable partition bucket for one molecule UID."""
    normalized = str(value)
    if len(normalized) < _LATENT_MOLECULE_BUCKET_LENGTH:
        raise ValueError(f"invalid molecule_uid for index partitioning: {value!r}")
    return f"b{normalized[:_LATENT_MOLECULE_BUCKET_LENGTH]}"


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
    template_ids = obs.get(TEMPLATE_ID_COLUMN, read_ids).astype(str)
    molecule_uids = obs.get(MOLECULE_UID_COLUMN)
    expected_molecule_uids = pd.Series(
        [molecule_uid(experiment_uid, template_id) for template_id in template_ids],
        index=obs.index,
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


def write_latent_read_index(
    output_dir: str | Path,
    *,
    obs: pd.DataFrame,
    record: Mapping[str, object],
    generation_id: str,
    group_path: str,
    reference_uid: str | None,
    stage_schema_version: int,
) -> list[Path]:
    """Write bounded molecule-to-latent-task shards in stored observation order.

    The dataset is partitioned by a stable prefix of ``molecule_uid``. A task
    writes at most one shard per occupied bucket, so partition pruning does not
    create one file per molecule.
    """
    analysis_core_id = str(record.get("analysis_core_id", ""))
    if not analysis_core_id:
        raise ValueError("latent task lacks analysis_core_id")
    experiment_values = obs.get(EXPERIMENT_UID_COLUMN)
    if experiment_values is None or experiment_values.isna().any():
        raise ValueError(f"latent task {analysis_core_id!r} lacks experiment_uid identity")
    unique_experiments = experiment_values.astype(str).unique()
    if len(unique_experiments) != 1:
        raise ValueError(f"latent task {analysis_core_id!r} spans multiple experiments")
    experiment_uid = validate_experiment_uid(unique_experiments[0])
    read_ids = obs.get("read_id", pd.Series(obs.index.astype(str), index=obs.index)).astype(str)
    template_ids = obs.get(TEMPLATE_ID_COLUMN, read_ids).astype(str)
    molecule_uids = obs.get(MOLECULE_UID_COLUMN)
    expected_molecule_uids = pd.Series(
        [molecule_uid(experiment_uid, template_id) for template_id in template_ids],
        index=obs.index,
    )
    if molecule_uids is None:
        molecule_uids = expected_molecule_uids
    elif not molecule_uids.astype(str).equals(expected_molecule_uids):
        raise ValueError(f"latent task {analysis_core_id!r} has inconsistent molecule_uid values")

    representation_keys = sorted(map(str, record.get("obsm_keys", ())))
    loading_keys = sorted(map(str, record.get("varm_keys", ())))
    label_keys = sorted(
        str(column) for column in record.get("obs_columns", ()) if str(column).startswith("leiden_")
    )
    task_checksum = str(record.get("group_sha256", ""))
    if not task_checksum:
        raise ValueError(f"latent task {analysis_core_id!r} lacks a task checksum")
    model_id = str(record.get("model_id", ""))
    model_checksum = str(record.get("model_checksum", ""))
    if not model_id or not model_checksum:
        raise ValueError(f"latent task {analysis_core_id!r} lacks model provenance")

    rows = []
    for group_row, (read_id, this_molecule_uid) in enumerate(
        zip(read_ids, molecule_uids.astype(str), strict=True)
    ):
        rows.append(
            {
                EXPERIMENT_UID_COLUMN: experiment_uid,
                "read_id": read_id,
                MOLECULE_UID_COLUMN: this_molecule_uid,
                "molecule_bucket": molecule_index_bucket(this_molecule_uid),
                "stage": "latent",
                "task_id": analysis_core_id,
                "reference": str(record["reference"]),
                "reference_uid": reference_uid,
                "analysis_core_id": analysis_core_id,
                "core_start": int(record["core_start"]),
                "core_end": int(record["core_end"]),
                "latent_generation_id": str(generation_id),
                "group_path": str(group_path),
                "group_row": group_row,
                "representation_keys": representation_keys,
                "loading_keys": loading_keys,
                "label_keys": label_keys,
                "task_checksum": task_checksum,
                "model_id": model_id,
                "model_checksum": model_checksum,
                "stage_schema_version": int(stage_schema_version),
                "index_schema_version": LATENT_READ_INDEX_SCHEMA_VERSION,
            }
        )

    frame = pd.DataFrame(rows)
    for column in (
        EXPERIMENT_UID_COLUMN,
        "read_id",
        MOLECULE_UID_COLUMN,
        "molecule_bucket",
        "stage",
        "task_id",
        "reference",
        "reference_uid",
        "analysis_core_id",
        "latent_generation_id",
        "group_path",
        "task_checksum",
        "model_id",
        "model_checksum",
    ):
        frame[column] = frame[column].astype("string")

    digest = hashlib.sha256(f"{generation_id}\0{analysis_core_id}".encode("utf-8")).hexdigest()[:20]
    paths = []
    index_root = Path(output_dir) / DERIVED_READ_INDEX_DIRNAME
    for bucket, bucket_frame in frame.groupby("molecule_bucket", sort=True):
        bucket_dir = index_root / f"molecule_bucket={bucket}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        path = bucket_dir / f"task-{digest}.parquet"
        temporary_path = path.with_suffix(".tmp.parquet")
        try:
            bucket_frame.drop(columns=["molecule_bucket"]).to_parquet(
                temporary_path,
                index=False,
                row_group_size=portable_parquet_row_group_rows(bucket_frame),
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        paths.append(path)
    return paths
