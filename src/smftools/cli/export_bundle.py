"""Experiment and project export bundles for lossless re-ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from smftools.logging_utils import get_logger

from ..informatics.export_bundle import (
    BUNDLE_IDENTITY_MAP_FILENAME,
    BUNDLE_INPUT_MANIFEST_FILENAME,
    artifact_record,
    write_bundle_manifest,
    write_canonical_input_manifest,
    write_identity_map,
)
from ..informatics.molecule_identity import EXPERIMENT_UID_COLUMN
from ..informatics.partition_read import load_spine, resolve_relative_path
from .export_fastq import _QC_PASS_COLUMNS, _obs_by_read_id, _resolve_group_by, _resolve_qc_obs

logger = get_logger(__name__)


@dataclass
class _ExportContext:
    experiment_id: str
    raw_spine_path: Path
    raw_spine: Any
    segments: pd.DataFrame
    selected: pd.DataFrame
    selection: dict[str, Any]
    modality: str
    trim_state: str


def _run_root(spine_path: Path) -> Path:
    if spine_path.parent.parent.name in {"generations", ".staging"}:
        return spine_path.parents[3]
    return spine_path.parents[1]


def _segments_for_spine(spine_path: Path, spine) -> pd.DataFrame:
    path = resolve_relative_path(spine.uns.get("segments_catalog"), _run_root(spine_path))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"raw segment catalog is unavailable for export: {path}")
    segments = pd.read_parquet(path)
    if "segment_read_id" not in segments:
        raise ValueError("raw segment catalog lacks segment_read_id")
    if "ragged_shard" not in segments and "group_path" in segments:
        segments["ragged_shard"] = segments["group_path"]
    if "ragged_row" not in segments and "group_row" in segments:
        segments["ragged_row"] = segments["group_row"]
    segments.index = segments["segment_read_id"].astype(str)
    return segments


def _selection_contract(source: str, accepted: set[str] | None) -> dict[str, Any]:
    token = source.split("@", 1)[0]
    return {
        "source": source,
        "unfiltered": accepted is None,
        "qc_selected": accepted is not None,
        "deduplicated": token in {"passes_dedup", "pp_dedup"},
        "selection_column": token if token in _QC_PASS_COLUMNS else None,
    }


def _trim_state(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    normalized = str(value or "unknown").lower()
    return normalized if normalized in {"true", "false", "unknown"} else "unknown"


def _context(
    *,
    experiment_id: str,
    raw_spine_path: Path,
    label_obs: pd.DataFrame,
    accepted: set[str] | None,
    source: str,
    group_by: str,
    modality: str,
    trim_state: str,
    project_namespaced: bool,
) -> _ExportContext:
    spine = load_spine(raw_spine_path)
    raw_obs = _obs_by_read_id(spine.obs)
    selected_molecules = raw_obs if accepted is None else raw_obs.loc[raw_obs.index.isin(accepted)]
    if selected_molecules.empty:
        raise ValueError(f"export selection contains no molecules for {experiment_id!r}")
    segments = _segments_for_spine(raw_spine_path, spine)
    molecule_uids = set(selected_molecules["molecule_uid"].astype(str))
    selected = segments.loc[segments["molecule_uid"].astype(str).isin(molecule_uids)].copy()
    label_by_molecule = pd.Series(
        label_obs[group_by].reindex(raw_obs.index).astype("string").to_numpy(),
        index=raw_obs["molecule_uid"].astype(str),
    )
    selected["export_group"] = selected["molecule_uid"].astype(str).map(label_by_molecule)
    selected["export_group"] = selected["export_group"].fillna("unknown").astype(str)
    if project_namespaced:
        selected["export_group"] = experiment_id + "__" + selected["export_group"]
    selected["experiment_id"] = experiment_id
    return _ExportContext(
        experiment_id=experiment_id,
        raw_spine_path=raw_spine_path,
        raw_spine=spine,
        segments=segments,
        selected=selected,
        selection=_selection_contract(source, accepted),
        modality=modality,
        trim_state=trim_state,
    )


def _identity_row(row: pd.Series, artifact_path: str) -> dict[str, Any]:
    mate = str(row.get("mate", "unpaired"))
    molecule = str(row["molecule_uid"])
    bundle_read_id = (
        f"{molecule}/{'1' if mate == 'R1' else '2'}" if mate in {"R1", "R2"} else molecule
    )
    return {
        "bundle_read_id": bundle_read_id,
        "bundle_template_id": molecule,
        "experiment_id": str(row.get("experiment_id", "")),
        "experiment_uid": str(row.get(EXPERIMENT_UID_COLUMN, "")),
        "namespace": str(row.get("namespace", "")),
        "source_read_id": str(row.get("source_read_id", row.get("template_id", ""))),
        "template_id": str(row.get("template_id", "")),
        "molecule_uid": molecule,
        "segment_id": str(row.get("segment_id", "single")),
        "segment_uid": str(row.get("segment_uid", "")),
        "mate": mate,
        "sample": str(row.get("sample", row.get("Sample", ""))),
        "barcode": str(row.get("barcode", row.get("Barcode", ""))),
        "artifact_path": artifact_path,
    }


def _write_fastq_bundle(
    contexts: list[_ExportContext], outdir: Path, *, gzip_output: bool, scope: str
) -> Path:
    from ..informatics.fastq_export import write_fastq_manifest, write_fastq_per_barcode

    outdir.mkdir(parents=True, exist_ok=True)
    overall: dict[str, dict[str, object]] = {}
    identities: list[dict[str, Any]] = []
    declarations: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for context in contexts:
        frame = context.selected.copy()
        mates = frame.get("mate", pd.Series("unpaired", index=frame.index)).astype(str)
        paired = mates.isin({"R1", "R2"})
        file_groups = frame["export_group"].astype(str)
        file_groups = file_groups.where(~paired, file_groups + "__" + mates)
        record_names = pd.Series(
            [
                f"{molecule}/{'1' if mate == 'R1' else '2'}"
                if mate in {"R1", "R2"}
                else str(molecule)
                for molecule, mate in zip(frame["molecule_uid"], mates, strict=True)
            ],
            index=frame.index,
        )
        written = write_fastq_per_barcode(
            frame,
            context.raw_spine_path.parent,
            outdir,
            group_labels=file_groups,
            record_names=record_names,
            gzip_output=gzip_output,
        )
        overlap = set(overall).intersection(written)
        if overlap:
            raise ValueError(f"export groups collide across experiments: {sorted(overlap)}")
        overall.update(written)
        for group, info in written.items():
            path = Path(info["path"])
            group_rows = frame.loc[file_groups == group]
            mate_values = set(group_rows.get("mate", pd.Series("unpaired", index=group_rows.index)))
            mate = next(iter(mate_values)) if len(mate_values) == 1 else "unpaired"
            base_group = group.removesuffix("__R1").removesuffix("__R2")
            namespace_values = set(
                group_rows.get("namespace", pd.Series("", index=group_rows.index)).astype(str)
            )
            namespace = (
                next(iter(namespace_values)) if len(namespace_values) == 1 else ""
            ) or context.experiment_id
            declarations.append(
                {
                    "path": path.relative_to(outdir).as_posix(),
                    "source_kind": "fastq",
                    "source_role": "reads",
                    "sample": base_group,
                    "barcode": base_group,
                    "read_group": base_group,
                    "pair_id": base_group if mate in {"R1", "R2"} else "",
                    "mate": mate,
                    "namespace": namespace,
                    "modification_capability": "sequence_only",
                    "trimmed": context.trim_state,
                }
            )
            artifacts.append(artifact_record(outdir, path, kind="fastq", n_reads=info["n_reads"]))
            identities.extend(
                _identity_row(row, path.relative_to(outdir).as_posix())
                for _, row in group_rows.iterrows()
            )

    write_fastq_manifest(outdir, overall)
    write_identity_map(outdir / BUNDLE_IDENTITY_MAP_FILENAME, identities)
    modality = contexts[0].modality if len({item.modality for item in contexts}) == 1 else "mixed"
    write_canonical_input_manifest(
        outdir / BUNDLE_INPUT_MANIFEST_FILENAME,
        declarations,
        alignment_mode="align",
        modality="",
    )
    selection = {
        "experiments": {item.experiment_id: item.selection for item in contexts},
        "unfiltered": all(item.selection["unfiltered"] for item in contexts),
    }
    generations = [
        {
            "experiment_id": item.experiment_id,
            "experiment_uid": str(item.raw_spine.uns.get(EXPERIMENT_UID_COLUMN, "")),
            "raw_generation_id": str(item.raw_spine.uns.get("raw_generation_id", "")),
            "modality": item.modality,
            "trimmed": item.trim_state,
        }
        for item in contexts
    ]
    write_bundle_manifest(
        outdir,
        bundle_kind="sequence_only",
        scope=scope,
        modality=modality,
        selection=selection,
        source_generations=generations,
        artifacts=artifacts,
        lost_capabilities=("bam_auxiliary_tags", "alignment", "mm_ml", "read_groups"),
    )
    return outdir


def _alignment_manifests(context: _ExportContext) -> list[tuple[str, str, Path]]:
    from ..informatics.alignment_manifest import read_alignment_manifest
    from ..informatics.experiment_manifest import resolve_artifact_record
    from ..informatics.sidecar_manifest import (
        _load_manifest,
        resolve_sidecar,
        sidecar_manifest_path,
    )

    run_root = _run_root(context.raw_spine_path)
    candidates: dict[str, Path] = {}
    sidecar = sidecar_manifest_path(context.raw_spine_path.parent)
    if sidecar.is_file():
        for key in _load_manifest(sidecar).get("sidecars", {}):
            if key.startswith("alignment_manifest") and (resolved := resolve_sidecar(sidecar, key)):
                candidates[key] = resolved
    generation_manifest = context.raw_spine_path.parent / "generation_manifest.json"
    if generation_manifest.is_file():
        payload = json.loads(generation_manifest.read_text(encoding="utf-8"))
        for key, record in payload.get("dependencies", {}).items():
            if key.startswith("sidecar:alignment_manifest"):
                resolved = resolve_artifact_record(run_root, record)
                if resolved is not None:
                    candidates[key.removeprefix("sidecar:")] = resolved
    partitions = context.raw_spine.uns.get("alignment_source_partitions", {})
    result = []
    for key, manifest_path in sorted(candidates.items()):
        manifest = read_alignment_manifest(manifest_path)
        bam = manifest_path.parent / manifest["artifacts"]["bam"]["path"]
        source_id = key.split(":", 1)[1] if ":" in key else ""
        namespace = str(partitions.get(source_id, {}).get("namespace", ""))
        result.append((source_id, namespace, bam))
    if not result and "bam_path" in context.raw_spine.obs:
        values = set(context.raw_spine.obs["bam_path"].dropna().astype(str))
        for value in sorted(values):
            bam = resolve_relative_path(value, run_root)
            if bam is not None and bam.is_file():
                result.append(("", "", bam))
    if not result:
        raise FileNotFoundError(f"no owned alignment is available for {context.experiment_id!r}")
    return result


def _write_bam_bundle(contexts: list[_ExportContext], outdir: Path, *, scope: str) -> Path:
    from ..informatics.bam_functions import _require_pysam

    pysam = _require_pysam()
    inputs_dir = outdir / "alignments"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    declarations = []
    artifacts = []
    identities = []
    for context in contexts:
        for index, (source_id, namespace, source_bam) in enumerate(_alignment_manifests(context)):
            selected = context.selected
            if namespace and "namespace" in selected:
                selected = selected.loc[selected["namespace"].astype(str) == namespace]
            source_names = selected.get("source_read_id")
            if source_names is None:
                source_names = selected["template_id"]
            else:
                source_names = source_names.fillna(selected["template_id"])
                source_names = source_names.astype(str).where(
                    source_names.astype(str) != "", selected["template_id"].astype(str)
                )
            qnames = set(source_names.astype(str))
            if not qnames:
                continue
            stem = f"{context.experiment_id}-{source_id[:12] or index}"
            destination = inputs_dir / f"{stem}.bam"
            observed: set[str] = set()
            missing_modification_tags: set[str] = set()
            with (
                pysam.AlignmentFile(str(source_bam), "rb", check_sq=False) as source,
                pysam.AlignmentFile(str(destination), "wb", header=source.header) as target,
            ):
                for read in source.fetch(until_eof=True):
                    if str(read.query_name) in qnames:
                        target.write(read)
                        observed.add(str(read.query_name))
                        if (
                            context.modality == "direct"
                            and not read.is_secondary
                            and not read.is_supplementary
                            and not (read.has_tag("MM") and read.has_tag("ML"))
                        ):
                            missing_modification_tags.add(str(read.query_name))
            missing = sorted(qnames.difference(observed))
            if missing:
                destination.unlink(missing_ok=True)
                raise ValueError(f"owned BAM is missing selected reads: {missing[:5]}")
            if missing_modification_tags:
                destination.unlink(missing_ok=True)
                raise ValueError(
                    "direct-modification BAM bundle cannot claim lossless MM/ML capability; "
                    f"selected reads lack tags: {sorted(missing_modification_tags)[:5]}"
                )
            pysam.index(str(destination))
            bai = Path(f"{destination}.bai")
            relative = destination.relative_to(outdir).as_posix()
            declarations.append(
                {
                    "path": relative,
                    "source_kind": "aligned_bam",
                    "source_role": "alignment",
                    "namespace": namespace or context.experiment_id,
                    "modification_capability": "mm_ml"
                    if context.modality == "direct"
                    else "conversion_sequence",
                    "trimmed": context.trim_state,
                }
            )
            artifacts.extend(
                [
                    artifact_record(outdir, destination, kind="bam"),
                    artifact_record(outdir, bai, kind="bai"),
                ]
            )
            identities.extend(_identity_row(row, relative) for _, row in selected.iterrows())
    if not declarations:
        raise ValueError("BAM bundle selection contains no alignment records")
    write_identity_map(outdir / BUNDLE_IDENTITY_MAP_FILENAME, identities)
    modality = contexts[0].modality if len({item.modality for item in contexts}) == 1 else "mixed"
    write_canonical_input_manifest(
        outdir / BUNDLE_INPUT_MANIFEST_FILENAME,
        declarations,
        alignment_mode="existing",
        modality="direct" if modality == "direct" else "conversion",
    )
    write_bundle_manifest(
        outdir,
        bundle_kind="lossless_bam",
        scope=scope,
        modality=modality,
        selection={"experiments": {item.experiment_id: item.selection for item in contexts}},
        source_generations=[
            {
                "experiment_id": item.experiment_id,
                "experiment_uid": str(item.raw_spine.uns.get(EXPERIMENT_UID_COLUMN, "")),
                "raw_generation_id": str(item.raw_spine.uns.get("raw_generation_id", "")),
                "modality": item.modality,
                "trimmed": item.trim_state,
            }
            for item in contexts
        ],
        artifacts=artifacts,
        lost_capabilities=(),
    )
    return outdir


def export_bundle_for_experiment(
    config_path: str,
    outdir: str | Path,
    *,
    bundle_format: str = "fastq",
    group_by: str | None = None,
    allow_unfiltered: bool = False,
    gzip_output: bool = True,
) -> Path:
    """Export one experiment as a portable FASTQ or BAM ingestion bundle."""
    from .helpers import get_adata_paths, load_experiment_config

    cfg = load_experiment_config(config_path)
    paths = get_adata_paths(cfg)
    if not paths.raw_spine or not paths.raw_spine.exists():
        raise FileNotFoundError(
            f"no raw ragged spine found at {paths.raw_spine}; "
            f"run `smftools experiment raw {config_path}` first"
        )
    label_obs, accepted, source = _resolve_qc_obs(paths, allow_unfiltered=allow_unfiltered)
    resolved_group = _resolve_group_by(label_obs, group_by, cfg)
    context = _context(
        experiment_id=str(getattr(cfg, "experiment_name", Path(config_path).stem)),
        raw_spine_path=Path(paths.raw_spine),
        label_obs=label_obs,
        accepted=accepted,
        source=source,
        group_by=resolved_group,
        modality=str(getattr(cfg, "smf_modality", "unknown")),
        trim_state=_trim_state(getattr(cfg, "trim", "unknown")),
        project_namespaced=False,
    )
    outdir = Path(outdir)
    if bundle_format == "fastq":
        return _write_fastq_bundle([context], outdir, gzip_output=gzip_output, scope="experiment")
    if bundle_format == "bam":
        return _write_bam_bundle([context], outdir, scope="experiment")
    raise ValueError("bundle_format must be 'fastq' or 'bam'")


def export_bundle_for_project(
    project_dir: str | Path,
    outdir: str | Path,
    *,
    bundle_format: str = "fastq",
    experiments: list[str] | None = None,
    allow_unfiltered: bool = False,
    gzip_output: bool = True,
) -> Path:
    """Export selected active project experiments as one portable bundle."""
    from ..project.registry import list_experiments

    entries = list_experiments(project_dir, active_only=True)
    if experiments is not None:
        wanted = {str(item) for item in experiments}
        entries = [entry for entry in entries if str(entry["id"]) in wanted]
    contexts = []
    for entry in entries:
        experiment_id = str(entry["id"])
        raw_path = Path(entry.get("spines", {}).get("raw", ""))
        preprocess_value = entry.get("spines", {}).get("preprocess")
        preprocess_path = Path(preprocess_value) if preprocess_value else None
        preprocess_available = preprocess_path is not None and preprocess_path.is_file()
        if not raw_path.is_file() or (not preprocess_available and not allow_unfiltered):
            logger.warning(
                "skipping %r: required raw/preprocess spine is unavailable", experiment_id
            )
            continue
        raw_spine = load_spine(raw_path)
        if preprocess_available:
            assert preprocess_path is not None
            label_obs = _obs_by_read_id(load_spine(preprocess_path).obs)
            accepted = None
            column = next((name for name in _QC_PASS_COLUMNS if name in label_obs), None)
            if column is not None:
                accepted = set(label_obs.index[label_obs[column].astype(bool)].astype(str))
            source = f"{column or 'preprocess_spine'}@{preprocess_path}"
        else:
            label_obs = _obs_by_read_id(raw_spine.obs)
            accepted = None
            source = f"raw_spine@{raw_path} (unfiltered)"
        group_by = "Sample" if "Sample" in label_obs else "Barcode"
        contexts.append(
            _context(
                experiment_id=experiment_id,
                raw_spine_path=raw_path,
                label_obs=label_obs,
                accepted=accepted,
                source=source,
                group_by=group_by,
                modality=str(raw_spine.uns.get("modality", "unknown")),
                trim_state="unknown",
                project_namespaced=True,
            )
        )
    if not contexts:
        raise ValueError("no project experiments are eligible for export")
    outdir = Path(outdir)
    if bundle_format == "fastq":
        return _write_fastq_bundle(contexts, outdir, gzip_output=gzip_output, scope="project")
    if bundle_format == "bam":
        return _write_bam_bundle(contexts, outdir, scope="project")
    raise ValueError("bundle_format must be 'fastq' or 'bam'")
