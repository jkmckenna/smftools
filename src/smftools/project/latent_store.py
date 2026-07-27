"""Scoped project access to experiment-local latent task artifacts.

Latent task owners have independent coordinate systems. This module therefore
returns and exports one result per owner; it never concatenates task ``obsm`` or
``varm`` arrays across experiments or analysis cores.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..informatics.artifact_paths import resolve_artifact_path
from ..informatics.derived_read_index import molecule_index_bucket

LATENT_EXPORT_SCHEMA_VERSION = 1
LATENT_EXPORT_FORMAT = "anndata-zarr-v3"
LATENT_EXPORT_CATALOG_COLUMNS = (
    "schema_version",
    "part_id",
    "artifact_format",
    "experiment",
    "experiment_uid",
    "reference",
    "reference_uid",
    "analysis_core_id",
    "core_start",
    "core_end",
    "latent_generation_id",
    "model_id",
    "model_checksum",
    "representations",
    "loadings",
    "labels",
    "source_checksum",
    "n_reads",
    "n_positions",
    "path",
)
_IDENTITY_COLUMNS = ("read_id", "experiment_uid", "molecule_uid")


@dataclass(frozen=True)
class LatentScope:
    """Immutable coordinate ownership and provenance for one latent task result."""

    experiment: str
    experiment_uid: str
    reference: str
    reference_uid: str | None
    analysis_core_id: str
    core_start: int
    core_end: int
    latent_generation_id: str
    group_path: str
    task_checksum: str
    model_id: str | None
    model_checksum: str | None
    representations: tuple[str, ...]
    loadings: tuple[str, ...]
    labels: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-safe ownership record."""
        record = asdict(self)
        for key in ("representations", "loadings", "labels"):
            record[key] = list(record[key])
        return record


@dataclass(frozen=True)
class LatentPart:
    """One projected task-local AnnData and its immutable coordinate scope."""

    adata: object
    scope: LatentScope


def _values(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(sorted({str(item) for item in value}))


def _list_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        if bool(pd.isna(value)):
            return ()
    except (TypeError, ValueError):
        pass
    return tuple(sorted(map(str, value)))


def _optional_string(value: object) -> str | None:
    if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
        return None
    normalized = str(value)
    return normalized or None


def _and(expressions):
    result = None
    for expression in expressions:
        result = expression if result is None else result & expression
    return result


def _mapped_loading(representation: str) -> str | None:
    mappings = (
        ("X_pca_", "PCs_"),
        ("X_nmf_", "H_nmf_"),
        ("X_cp_", "H_cp_"),
    )
    for prefix, loading_prefix in mappings:
        if representation.startswith(prefix):
            return loading_prefix + representation.removeprefix(prefix)
    return None


def _projected_fields(
    row: pd.Series,
    *,
    representations,
    labels,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    available_obsm = _list_value(row["representation_keys"])
    available_varm = _list_value(row["loading_keys"])
    available_labels = _list_value(row["label_keys"])
    requested_representations = _values(representations)
    requested_labels = _values(labels)

    if requested_representations is None:
        selected_obsm = available_obsm
        selected_varm = available_varm
    else:
        available = set(available_obsm).union(available_varm)
        missing = sorted(set(requested_representations).difference(available))
        if missing:
            raise KeyError(f"latent representations are unavailable: {missing}")
        selected_obsm = tuple(
            key for key in requested_representations if key in set(available_obsm)
        )
        selected_varm_set = {key for key in requested_representations if key in set(available_varm)}
        for key in selected_obsm:
            mapped = _mapped_loading(key)
            if mapped in available_varm:
                selected_varm_set.add(mapped)
        selected_varm = tuple(sorted(selected_varm_set))

    if requested_labels is None:
        selected_labels = available_labels
    else:
        missing = sorted(set(requested_labels).difference(available_labels))
        if missing:
            raise KeyError(f"latent labels are unavailable: {missing}")
        selected_labels = requested_labels
    return tuple(selected_obsm), tuple(selected_varm), tuple(selected_labels)


def _read_scoped_part(
    path: Path,
    rows: np.ndarray,
    *,
    representations: tuple[str, ...],
    loadings: tuple[str, ...],
    labels: tuple[str, ...],
    scope: LatentScope,
):
    """Slice rows and fields from lazy arrays before materializing one owner."""
    import anndata as ad

    source = ad.experimental.read_lazy(path)
    if rows.size and (rows.min() < 0 or rows.max() >= source.n_obs):
        raise RuntimeError(f"latent index group_row is out of bounds for {path}")
    projected = source[rows, :]
    missing_identity = sorted(set(_IDENTITY_COLUMNS).difference(projected.obs.columns))
    if missing_identity:
        raise RuntimeError(
            f"latent task store is missing identity columns {missing_identity}: {path}"
        )
    obs_columns = [
        column for column in (*_IDENTITY_COLUMNS, *labels) if column in projected.obs.columns
    ]
    obs = projected.obs[obs_columns]
    obs = obs.to_memory() if hasattr(obs, "to_memory") else obs
    var = projected.var.to_memory() if hasattr(projected.var, "to_memory") else projected.var
    selected = ad.AnnData(
        obs=obs,
        var=var,
        obsm={key: projected.obsm[key] for key in representations},
        varm={key: projected.varm[key] for key in loadings},
        uns={"latent_coordinate_scope": scope.as_dict()},
    )
    return selected.to_memory() if hasattr(selected, "to_memory") else selected


def _task_groups(
    catalog,
    *,
    canonical_reference,
    experiments,
    set_name,
    molecule_uids,
    analysis_core_ids,
):
    """Resolve index rows before opening any latent task store."""
    import pyarrow.dataset as ds

    selected_experiments = catalog._resolve_experiment_filter(experiments, set_name)
    wanted_molecules = _values(molecule_uids)
    wanted_cores = _values(analysis_core_ids)
    legacy_latent = []
    groups = []
    for entry in catalog.experiments():
        if selected_experiments is not None and entry["id"] not in selected_experiments:
            continue
        index_value = entry.get("catalogs", {}).get("latent_read_index")
        if index_value is None:
            if "latent" in entry.get("spines", {}):
                legacy_latent.append(entry["id"])
            continue
        index_path = Path(index_value)
        if not index_path.exists():
            raise FileNotFoundError(
                f"registered latent read index is missing for experiment {entry['id']!r}: "
                f"{index_path}"
            )
        dataset = ds.dataset(index_path, format="parquet", partitioning="hive")
        required = {
            "experiment_uid",
            "reference",
            "reference_uid",
            "analysis_core_id",
            "core_start",
            "core_end",
            "latent_generation_id",
            "molecule_uid",
            "molecule_bucket",
            "group_path",
            "group_row",
            "representation_keys",
            "loading_keys",
            "label_keys",
            "task_checksum",
            "model_id",
            "model_checksum",
        }
        missing = sorted(required.difference(dataset.schema.names))
        if missing:
            raise RuntimeError(
                f"latent index for experiment {entry['id']!r} requires migration; "
                f"missing columns: {missing}"
            )
        expressions = []
        if canonical_reference is not None:
            expressions.append(ds.field("reference_uid") == str(canonical_reference))
        if wanted_cores is not None:
            expressions.append(ds.field("analysis_core_id").isin(wanted_cores))
        if wanted_molecules is not None:
            expressions.extend(
                [
                    ds.field("molecule_uid").isin(wanted_molecules),
                    ds.field("molecule_bucket").isin(
                        sorted({molecule_index_bucket(value) for value in wanted_molecules})
                    ),
                ]
            )
        frame = dataset.to_table(filter=_and(expressions)).to_pandas()
        if frame.empty:
            continue
        if set(frame["experiment_uid"].astype(str)) != {str(entry["experiment_uid"])}:
            raise RuntimeError(f"latent index ownership conflicts with experiment {entry['id']!r}")
        owner_columns = (
            "analysis_core_id",
            "latent_generation_id",
            "group_path",
            "task_checksum",
        )
        for _, owner_rows in frame.groupby(list(owner_columns), sort=True, dropna=False):
            for column in (
                "experiment_uid",
                "reference",
                "reference_uid",
                "core_start",
                "core_end",
                "model_id",
                "model_checksum",
            ):
                if owner_rows[column].astype(str).nunique(dropna=False) != 1:
                    raise RuntimeError(
                        f"latent owner has conflicting {column} values for "
                        f"experiment {entry['id']!r}"
                    )
            for column in ("representation_keys", "loading_keys", "label_keys"):
                summaries = {_list_value(value) for value in owner_rows[column].tolist()}
                if len(summaries) != 1:
                    raise RuntimeError(
                        f"latent owner has conflicting {column} for experiment {entry['id']!r}"
                    )
            groups.append((entry, owner_rows.sort_values("group_row", kind="stable")))

    if legacy_latent:
        raise RuntimeError(
            "legacy monolithic latent artifacts do not have the task-owner index required "
            "for scoped project access. Re-run latent analysis in partitioned mode and "
            f"re-add these experiments: {sorted(legacy_latent)}"
        )
    if not groups:
        raise ValueError("no latent task owners match the requested project filters")
    return groups


def iter_latent_parts(
    catalog,
    *,
    canonical_reference=None,
    experiments=None,
    set_name=None,
    molecule_uids=None,
    analysis_core_ids=None,
    representations=None,
    labels=None,
):
    """Yield one projected latent result per immutable coordinate owner.

    Index filters are applied before task stores are opened. Requested rows,
    ``obsm``/``varm`` representations, and label columns are sliced from lazy
    arrays before they are materialized.
    """
    groups = _task_groups(
        catalog,
        canonical_reference=canonical_reference,
        experiments=experiments,
        set_name=set_name,
        molecule_uids=molecule_uids,
        analysis_core_ids=analysis_core_ids,
    )

    def _generate():
        for entry, rows in groups:
            first = rows.iloc[0]
            selected_obsm, selected_varm, selected_labels = _projected_fields(
                first,
                representations=representations,
                labels=labels,
            )
            scope = LatentScope(
                experiment=str(entry["id"]),
                experiment_uid=str(first["experiment_uid"]),
                reference=str(first["reference"]),
                reference_uid=_optional_string(first["reference_uid"]),
                analysis_core_id=str(first["analysis_core_id"]),
                core_start=int(first["core_start"]),
                core_end=int(first["core_end"]),
                latent_generation_id=str(first["latent_generation_id"]),
                group_path=str(first["group_path"]),
                task_checksum=str(first["task_checksum"]),
                model_id=_optional_string(first.get("model_id")),
                model_checksum=_optional_string(first.get("model_checksum")),
                representations=selected_obsm,
                loadings=selected_varm,
                labels=selected_labels,
            )
            run_root = Path(entry["path"]).resolve()
            group_path = resolve_artifact_path(scope.group_path, run_root)
            if group_path is None or not group_path.resolve().is_relative_to(run_root):
                raise RuntimeError(
                    f"latent group path is not portable for experiment {entry['id']!r}: "
                    f"{scope.group_path!r}"
                )
            if not group_path.is_dir():
                raise FileNotFoundError(
                    f"latent task store is missing for experiment {entry['id']!r}: {group_path}"
                )
            if rows["group_row"].isna().any() or rows["group_row"].duplicated().any():
                raise RuntimeError(
                    f"latent index has invalid group_row values for {scope.analysis_core_id!r}"
                )
            if rows["molecule_uid"].astype(str).duplicated().any():
                raise RuntimeError(
                    f"latent index has duplicate molecules for {scope.analysis_core_id!r}"
                )
            adata = _read_scoped_part(
                group_path,
                rows["group_row"].to_numpy(dtype=np.int64),
                representations=selected_obsm,
                loadings=selected_varm,
                labels=selected_labels,
                scope=scope,
            )
            expected_molecules = rows["molecule_uid"].astype(str).tolist()
            if "molecule_uid" in adata.obs:
                if adata.obs["molecule_uid"].astype(str).tolist() != expected_molecules:
                    raise RuntimeError(
                        f"latent task/index molecule mismatch for {scope.analysis_core_id!r}"
                    )
            yield LatentPart(adata=adata, scope=scope)

    return _generate()


def _part_id(part: LatentPart) -> str:
    molecules = (
        part.adata.obs["molecule_uid"].astype(str).tolist()
        if "molecule_uid" in part.adata.obs
        else part.adata.obs_names.astype(str).tolist()
    )
    identity = "\0".join(
        (
            part.scope.experiment_uid,
            part.scope.analysis_core_id,
            part.scope.latent_generation_id,
            part.scope.task_checksum,
            ",".join(part.scope.representations),
            ",".join(part.scope.loadings),
            ",".join(part.scope.labels),
            ",".join(molecules),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def _slug(value: object) -> str:
    from .set_store import slug

    return slug(str(value))


def _validate_export(catalog: pd.DataFrame, root: Path) -> None:
    missing = sorted(set(LATENT_EXPORT_CATALOG_COLUMNS).difference(catalog.columns))
    if missing:
        raise RuntimeError(f"latent export catalog is missing required columns: {missing}")
    if catalog.empty:
        raise RuntimeError("latent export produced no scoped artifacts")
    if catalog["part_id"].duplicated().any() or catalog["path"].duplicated().any():
        raise RuntimeError("latent export contains duplicate owners or paths")
    if (catalog["n_reads"] <= 0).any() or (catalog["n_positions"] < 0).any():
        raise RuntimeError("latent export contains invalid artifact dimensions")
    expected_paths = set()
    for value in catalog["path"]:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"latent export path is not portable: {value!r}")
        if not (root / relative).is_dir():
            raise RuntimeError(f"latent export artifact is missing: {value!r}")
        expected_paths.add(relative.as_posix())
    observed_paths = {
        path.relative_to(root).as_posix() for path in (root / "parts").rglob("*.zarr")
    }
    if observed_paths != expected_paths:
        raise RuntimeError("latent export catalog does not match scoped artifacts on disk")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_latent_parts(
    catalog,
    output_dir: str | Path,
    *,
    canonical_reference=None,
    experiments=None,
    set_name=None,
    molecule_uids=None,
    analysis_core_ids=None,
    representations=None,
    labels=None,
) -> Path:
    """Transactionally export one portable Zarr artifact per latent task owner."""
    from ..readwrite import atomic_write_json, safe_write_zarr

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"latent project export already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    rows = []
    try:
        for part in catalog.iter_latent_parts(
            canonical_reference=canonical_reference,
            experiments=experiments,
            set_name=set_name,
            molecule_uids=molecule_uids,
            analysis_core_ids=analysis_core_ids,
            representations=representations,
            labels=labels,
        ):
            part_id = _part_id(part)
            relative = (
                Path("parts")
                / f"experiment={_slug(part.scope.experiment)}"
                / (f"core={_slug(part.scope.analysis_core_id)}__part={part_id}.zarr")
            )
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            safe_write_zarr(
                part.adata,
                path,
                backup=False,
                verbose=False,
                zarr_format=3,
            )
            rows.append(
                {
                    "schema_version": LATENT_EXPORT_SCHEMA_VERSION,
                    "part_id": part_id,
                    "artifact_format": LATENT_EXPORT_FORMAT,
                    "experiment": part.scope.experiment,
                    "experiment_uid": part.scope.experiment_uid,
                    "reference": part.scope.reference,
                    "reference_uid": part.scope.reference_uid,
                    "analysis_core_id": part.scope.analysis_core_id,
                    "core_start": part.scope.core_start,
                    "core_end": part.scope.core_end,
                    "latent_generation_id": part.scope.latent_generation_id,
                    "model_id": part.scope.model_id,
                    "model_checksum": part.scope.model_checksum,
                    "representations": part.scope.representations,
                    "loadings": part.scope.loadings,
                    "labels": part.scope.labels,
                    "source_checksum": part.scope.task_checksum,
                    "n_reads": part.adata.n_obs,
                    "n_positions": part.adata.n_vars,
                    "path": relative.as_posix(),
                }
            )
        export_catalog = pd.DataFrame(rows, columns=LATENT_EXPORT_CATALOG_COLUMNS)
        export_catalog = export_catalog.sort_values(
            ["experiment", "analysis_core_id", "part_id"],
            ignore_index=True,
        )
        _validate_export(export_catalog, temporary)
        catalog_path = temporary / "catalog.parquet"
        export_catalog.to_parquet(catalog_path, index=False)
        atomic_write_json(
            temporary / "manifest.json",
            {
                "schema_version": LATENT_EXPORT_SCHEMA_VERSION,
                "status": "complete",
                "artifact_format": LATENT_EXPORT_FORMAT,
                "catalog": "catalog.parquet",
                "catalog_sha256": _sha256(catalog_path),
                "n_parts": len(export_catalog),
                "n_reads": int(export_catalog["n_reads"].sum()),
                "filters": {
                    "canonical_reference": canonical_reference,
                    "experiments": list(_values(experiments) or ()),
                    "set_name": set_name,
                    "molecule_uids": list(_values(molecule_uids) or ()),
                    "analysis_core_ids": list(_values(analysis_core_ids) or ()),
                    "representations": list(_values(representations) or ()),
                    "labels": list(_values(labels) or ()),
                },
            },
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir
