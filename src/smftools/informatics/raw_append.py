"""Assemble an append-only raw store without mutating selected generations."""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd

from smftools.constants import RAW_DIR

from ..readwrite import (
    normalize_uns_string_lists,
    safe_read_h5ad,
    safe_write_h5ad,
    uns_string_list,
)
from .input_manifest import (
    InputManifestTransition,
    ResolvedInputManifest,
    classify_input_manifest_transition,
    read_resolved_input_manifest,
)
from .molecule_identity import MOLECULE_UID_COLUMN, SEGMENT_UID_COLUMN
from .raw_generation import validate_raw_generation
from .raw_store import (
    BARCODE_INDEX_FILENAME,
    INTERVAL_CATALOG_FILENAME,
    MOLECULE_INDEX_DIRNAME,
    MOLECULES_FILENAME,
    SEGMENT_INDEX_DIRNAME,
    SEGMENTS_FILENAME,
    write_raw_pointer_indexes,
)
from .sidecar_manifest import register_sidecar, sidecar_manifest_path
from .stage_obs import write_stage_obs
from .storage_planner import plan_references


class RawAppendError(RuntimeError):
    """Raised when added raw artifacts cannot extend a selected generation."""


@dataclass(frozen=True)
class RawAppendAssembly:
    """Complete staged source artifacts ready for immutable publication."""

    root: Path
    sources: Mapping[str, Path]
    reused_molecules: int
    added_molecules: int
    reused_segments: int
    added_segments: int


@dataclass(frozen=True)
class RawAppendPlan:
    """Eligibility decision for an incremental raw-generation transition."""

    eligible: bool
    reason: str
    transition: InputManifestTransition


def _non_source_artifact_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(
        sorted(
            str(value)
            for value in values
            if not str(value).startswith(("input-manifest:", "source:"))
        )
    )


def plan_raw_append(
    previous_generation: str | Path,
    current_manifest: ResolvedInputManifest,
    *,
    run_root: str | Path,
    config_hash: str,
    input_artifact_ids: list[str] | tuple[str, ...],
) -> RawAppendPlan:
    """Classify a request and gate append on unchanged config/reference inputs."""
    previous_generation = Path(previous_generation)
    generation = validate_raw_generation(previous_generation, run_root=run_root)
    previous_manifest = read_resolved_input_manifest(
        previous_generation / "input_manifest" / "resolved_input_manifest.json"
    )
    transition = classify_input_manifest_transition(previous_manifest, current_manifest)
    if not transition.permits_incremental_append:
        return RawAppendPlan(False, f"source transition is {transition.kind.value}", transition)
    if str(generation.get("config_hash", "")) != str(config_hash):
        return RawAppendPlan(False, "raw configuration changed", transition)
    previous_non_source = _non_source_artifact_ids(generation.get("input_artifact_ids"))
    current_non_source = _non_source_artifact_ids(input_artifact_ids)
    if previous_non_source != current_non_source:
        return RawAppendPlan(False, "alignment reference or non-source input changed", transition)
    return RawAppendPlan(True, "pure source addition", transition)


def _safe_group_path(value: object) -> Path:
    relative = Path(str(value))
    if (
        not str(relative)
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "raw"
    ):
        raise RawAppendError(f"raw append contains an unsafe shard path: {value!r}")
    return relative


def _hardlink_tree(source: Path, destination: Path) -> None:
    """Hardlink immutable files into disposable assembly space."""
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
        else:
            raise RawAppendError(f"raw generation contains an unsupported artifact: {path}")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1_000_000):
        candidate = path.with_name(f"{path.stem}-append-{index:05d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RawAppendError(f"could not allocate a collision-safe shard path beside {path}")


def _rewrite_added_shards(
    frame: pd.DataFrame,
    *,
    source_root: Path,
    assembly_root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if "group_path" not in frame:
        raise RawAppendError("raw append catalog lacks group_path")
    mapping: dict[str, str] = {}
    for value in sorted(frame["group_path"].astype(str).unique()):
        relative = _safe_group_path(value)
        source = source_root / relative
        if not source.is_file():
            raise RawAppendError(f"added raw shard is missing: {source}")
        destination = _next_available_path(assembly_root / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        mapping[value] = destination.relative_to(assembly_root).as_posix()
    rewritten = frame.copy()
    rewritten["group_path"] = rewritten["group_path"].astype(str).map(mapping)
    return rewritten, mapping


def _reject_identity_collisions(
    previous: pd.DataFrame,
    added: pd.DataFrame,
    *,
    column: str,
    label: str,
) -> None:
    if column not in previous or column not in added:
        raise RawAppendError(f"raw append {label} catalogs lack {column}")
    collision = sorted(set(previous[column].astype(str)) & set(added[column].astype(str)))
    if collision:
        preview = ", ".join(map(repr, collision[:5]))
        raise RawAppendError(f"raw append has colliding {label} identities: {preview}")


def _combine_spines(
    previous_path: Path,
    added_path: Path,
    destination: Path,
    *,
    analysis_mode: str,
    load_cache_mode: str,
    max_full_matrix_gb: float,
    genome_tile_size: int,
    genome_tile_halo: int,
    transition: Mapping[str, Any],
    shard_mapping: Mapping[str, str],
) -> tuple[int, int]:
    previous, _ = safe_read_h5ad(previous_path, verbose=False)
    added, _ = safe_read_h5ad(added_path, verbose=False)
    previous_uid = str(previous.uns.get("experiment_uid", ""))
    if not previous_uid or previous_uid != str(added.uns.get("experiment_uid", "")):
        raise RawAppendError("raw append experiment identity does not match selected generation")
    previous_lengths = {
        str(key): int(value)
        for key, value in dict(previous.uns.get("reference_lengths", {})).items()
    }
    added_lengths = {
        str(key): int(value) for key, value in dict(added.uns.get("reference_lengths", {})).items()
    }
    if previous_lengths != added_lengths:
        raise RawAppendError("raw append reference lengths do not match selected generation")
    added_obs = added.obs.copy()
    for column in ("ragged_shard", "group_path"):
        if column in added_obs:
            rewritten = added_obs[column].astype(str).map(shard_mapping)
            if rewritten.isna().any():
                raise RawAppendError(f"added spine {column} references an unknown raw shard")
            added_obs[column] = rewritten
    duplicate_obs = sorted(set(previous.obs_names.astype(str)) & set(added.obs_names.astype(str)))
    if duplicate_obs:
        preview = ", ".join(map(repr, duplicate_obs[:5]))
        raise RawAppendError(f"raw append has colliding molecule rows: {preview}")
    # Column sets legitimately differ between generations: optional columns like
    # source_read_id only appear for namespaced sources, so the union with null
    # fill is the correct semantic ("this molecule has no source read id") rather
    # than schema drift to reject. Do not turn this into an equality check.
    obs = pd.concat([previous.obs, added_obs], axis=0, sort=False)
    if "canonical_row" in obs:
        obs["canonical_row"] = range(len(obs))
    obs_names = [str(value) for value in obs.index]
    obs = obs.reset_index(drop=True)
    obs.index = pd.Index(np.asarray(obs_names, dtype=str))
    spine = ad.AnnData(obs=obs)
    spine.uns = copy.deepcopy(dict(added.uns))
    previous_partitions = previous.uns.get("alignment_source_partitions", {})
    added_partitions = added.uns.get("alignment_source_partitions", {})
    if isinstance(previous_partitions, Mapping) or isinstance(added_partitions, Mapping):
        spine.uns["alignment_source_partitions"] = {
            **(dict(previous_partitions) if isinstance(previous_partitions, Mapping) else {}),
            **(dict(added_partitions) if isinstance(added_partitions, Mapping) else {}),
        }
    # signal_columns is discovered from the data rather than the reference, so an
    # append whose new sources lack a channel the prior generation observed would
    # otherwise narrow the combined list. Keep the prior order and append only
    # genuinely new channels.
    previous_signal = previous.uns.get("signal_columns")
    added_signal = added.uns.get("signal_columns")
    if previous_signal is not None or added_signal is not None:
        combined_signal = uns_string_list(previous_signal)
        known_signal = set(combined_signal)
        for column in uns_string_list(added_signal):
            if column not in known_signal:
                known_signal.add(column)
                combined_signal.append(column)
        spine.uns["signal_columns"] = combined_signal
    plans = plan_references(
        obs,
        previous_lengths,
        analysis_mode=analysis_mode,
        load_cache_mode=load_cache_mode,
        max_full_matrix_gb=max_full_matrix_gb,
        tile_size=genome_tile_size,
        tile_halo=genome_tile_halo,
    )
    spine.uns["reference_plans"] = {plan.reference: plan.to_dict() for plan in plans}
    spine.uns["raw_append_transition"] = dict(transition)
    normalize_uns_string_lists(spine)
    safe_write_h5ad(spine, destination, backup=False, verbose=False)
    return int(previous.n_obs), int(added.n_obs)


def assemble_raw_append(
    run_root: str | Path,
    previous_generation: str | Path,
    *,
    transition: Mapping[str, Any],
    analysis_mode: str = "auto",
    load_cache_mode: str = "auto",
    max_full_matrix_gb: float = 8.0,
    genome_tile_size: int = 10_000,
    genome_tile_halo: int = 1_000,
) -> RawAppendAssembly:
    """Combine a newly extracted source subset with one immutable generation.

    Prior generation files are hardlinked only into disposable assembly space;
    canonical working outputs and both immutable generations remain independent.
    Identity collisions fail before publication can advance ``current.json``.
    """
    run_root = Path(run_root)
    previous_generation = Path(previous_generation)
    validate_raw_generation(previous_generation, run_root=run_root)
    raw_root = run_root / RAW_DIR
    assembly = raw_root / ".append-staging" / uuid4().hex
    assembly.mkdir(parents=True)
    try:
        _hardlink_tree(previous_generation / "raw", assembly / "raw")
        added_catalog = pd.read_parquet(raw_root / INTERVAL_CATALOG_FILENAME)
        rewritten_catalog, shard_mapping = _rewrite_added_shards(
            added_catalog,
            source_root=raw_root,
            assembly_root=assembly,
        )
        previous_catalog = pd.read_parquet(previous_generation / INTERVAL_CATALOG_FILENAME)
        catalog = pd.concat([previous_catalog, rewritten_catalog], ignore_index=True, sort=False)
        catalog = catalog.sort_values(
            ["reference", "start_bin", "min_start", "group_path"], kind="stable"
        ).reset_index(drop=True)
        catalog.to_parquet(assembly / INTERVAL_CATALOG_FILENAME, index=False)

        previous_segments = pd.read_parquet(previous_generation / SEGMENTS_FILENAME)
        added_segments = pd.read_parquet(run_root / SEGMENTS_FILENAME).copy()
        added_segments["group_path"] = added_segments["group_path"].astype(str).map(shard_mapping)
        if added_segments["group_path"].isna().any():
            raise RawAppendError("added segment catalog references an unknown raw shard")
        _reject_identity_collisions(
            previous_segments,
            added_segments,
            column=SEGMENT_UID_COLUMN,
            label="segment",
        )
        segments = pd.concat([previous_segments, added_segments], ignore_index=True, sort=False)
        segments["canonical_row"] = range(len(segments))
        segments.to_parquet(assembly / SEGMENTS_FILENAME, index=False)

        previous_molecules = pd.read_parquet(previous_generation / MOLECULES_FILENAME)
        added_molecules = pd.read_parquet(run_root / MOLECULES_FILENAME)
        _reject_identity_collisions(
            previous_molecules,
            added_molecules,
            column=MOLECULE_UID_COLUMN,
            label="molecule",
        )
        molecules = pd.concat([previous_molecules, added_molecules], ignore_index=True, sort=False)
        if "canonical_row" in molecules:
            molecules["canonical_row"] = range(len(molecules))
        molecules.to_parquet(assembly / MOLECULES_FILENAME, index=False)
        molecule_index, segment_index = write_raw_pointer_indexes(segments, assembly)

        barcode_path: Path | None = None
        prior_barcode = previous_generation / BARCODE_INDEX_FILENAME
        added_barcode = raw_root / BARCODE_INDEX_FILENAME
        if prior_barcode.exists() or added_barcode.exists():
            frames: list[pd.DataFrame] = []
            if prior_barcode.exists():
                frames.append(pd.read_parquet(prior_barcode))
            if added_barcode.exists():
                frame = pd.read_parquet(added_barcode).copy()
                frame["group_path"] = frame["group_path"].astype(str).map(shard_mapping)
                if frame["group_path"].isna().any():
                    raise RawAppendError("added barcode index references an unknown raw shard")
                frames.append(frame)
            barcode_path = assembly / BARCODE_INDEX_FILENAME
            pd.concat(frames, ignore_index=True, sort=False).to_parquet(barcode_path, index=False)

        spine_path = assembly / "spine.h5ad"
        reused_molecules, added_molecule_count = _combine_spines(
            previous_generation / "spine.h5ad",
            raw_root / "spine.h5ad",
            spine_path,
            analysis_mode=analysis_mode,
            load_cache_mode=load_cache_mode,
            max_full_matrix_gb=max_full_matrix_gb,
            genome_tile_size=genome_tile_size,
            genome_tile_halo=genome_tile_halo,
            transition=transition,
            shard_mapping=shard_mapping,
        )
        combined_spine, _ = safe_read_h5ad(spine_path, verbose=False)
        combined_spine.obs_names = pd.Index(
            np.asarray([str(value) for value in combined_spine.obs_names], dtype=str)
        )
        combined_spine.uns["ragged_store"] = sorted(catalog["group_path"].astype(str).unique())
        normalize_uns_string_lists(combined_spine)
        safe_write_h5ad(combined_spine, spine_path, backup=False, verbose=False)
        obs_path = write_stage_obs(assembly, combined_spine.obs)

        manifest_path = sidecar_manifest_path(assembly)
        register_sidecar(manifest_path, "ragged_store", assembly / "raw")
        register_sidecar(manifest_path, "spine", spine_path)
        register_sidecar(manifest_path, "interval_catalog", assembly / INTERVAL_CATALOG_FILENAME)
        register_sidecar(manifest_path, "molecules", assembly / MOLECULES_FILENAME)
        register_sidecar(manifest_path, "molecule_index", molecule_index)
        register_sidecar(manifest_path, "segments", assembly / SEGMENTS_FILENAME)
        register_sidecar(manifest_path, "segment_index", segment_index)
        if barcode_path is not None:
            register_sidecar(manifest_path, "barcode_index", barcode_path)
        register_sidecar(manifest_path, "obs", obs_path)
        sources = {
            "spine": spine_path,
            "ragged_store": assembly / "raw",
            "interval_catalog": assembly / INTERVAL_CATALOG_FILENAME,
            "obs": obs_path,
            "molecules": assembly / MOLECULES_FILENAME,
            "molecule_index": molecule_index,
            "segments": assembly / SEGMENTS_FILENAME,
            "segment_index": segment_index,
            "sidecar_manifest": manifest_path,
        }
        if barcode_path is not None:
            sources["barcode_index"] = barcode_path
        return RawAppendAssembly(
            root=assembly,
            sources=sources,
            reused_molecules=reused_molecules,
            added_molecules=added_molecule_count,
            reused_segments=len(previous_segments),
            added_segments=len(added_segments),
        )
    except Exception:
        shutil.rmtree(assembly, ignore_errors=True)
        raise


def discard_raw_append_assembly(assembly: RawAppendAssembly | None) -> None:
    """Remove disposable append assembly files after publication or failure."""
    if assembly is not None:
        shutil.rmtree(assembly.root, ignore_errors=True)
