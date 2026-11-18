"""Project HMMs (`HPS-06`): fit once, store, decode every experiment with it.

``fit_project_model`` fits one model per reference group and strand on reads of
the selected samples of one registered experiment -- one HMM for an enzyme, its
transitions and emissions both fitted on those reads -- and stores the
checkpoints with a manifest under ``<project>/hmm_models/<name>/generations/<id>/``
(``current.json`` names the current one).

``apply_project_model`` decodes registered experiments with a stored model,
unchanged: the samples whose sample-sheet columns match the model's
``apply_samples``, every reference of a fitted strand (``6B6_enh_del_top`` by the
``top`` model). Each experiment's result is an HMM layer set,
``hmm_adata_outputs/layer_sets/project-<name>/`` (`HPS-07`), with the model's
HMM settings, so features are called the same way everywhere.

Spec (YAML or JSON)::

    name: cseda01_pe
    experiment: 260923_F1_hybrid_DAFseq_enzyme_panel   # registered experiment id
    samples: {enzyme: CseDa01, enzyme_prep: PureExpress, Enzyme_amount_uL: 2}
    apply_samples: {enzyme: CseDa01, enzyme_prep: PureExpress}  # default: samples
    reference_groups: {nkg2a: [6B6, 6BALB_cJ]}                  # pooled per strand
    exclude_references: [ctcf_mNanog_top, ctcf_mNanog_bottom]
    settings: {hmm_max_fit_reads: 2000}                         # config overrides
    sample_sheet:              # optional: one sheet for every experiment
      path: metadata/samples.csv          # relative to the project directory
      experiment_column: experiment_id    # rows of the experiment at hand
      barcode_column: barcode             # matched to the reads' Barcode
      match_on: barcode_number            # 11 == SQK-NBD114-24_barcode11

Without ``sample_sheet`` each experiment's own configured sample sheet is used.
An experiment's config is ``--config ID=PATH`` or else the
``experiment_config.csv`` in its run directory or the one beside it (a
modality-split run keeps its own inside), accepted only when its
``output_directory`` is the registered run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

MODELS_SUBDIR = "hmm_models"
LAYER_SET_PREFIX = "project-"
MANIFEST = "manifest.json"
TRAINING_READS = "training_reads.parquet"
SCHEMA_VERSION = 1
# HMM settings the apply side takes from the model (so features are called as
# when it was fitted); these few stay the decoding experiment's own.
APPLY_OWN_SETTINGS = frozenset({"hmm_device", "hmm_batch_size", "hmm_variants"})
EXTRA_SETTINGS = ("footprints", "accessible_patches", "cpg")


class ProjectModelError(ValueError):
    """A project-model request cannot be served."""


# ---- spec, registry and config -----------------------------------------------------------


def load_spec(path: str | Path) -> dict[str, Any]:
    """A model spec from YAML or JSON, validated."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        spec = json.loads(text)
    else:
        import yaml

        spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ProjectModelError(f"{path}: a model spec is a mapping")
    for key in ("name", "experiment", "samples", "reference_groups"):
        if not spec.get(key):
            raise ProjectModelError(f"{path}: model spec needs {key!r}")
    name = str(spec["name"])
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ProjectModelError(f"model name {name!r}: letters, digits, '_' and '-' only")
    for key in ("samples", "apply_samples", "reference_groups", "settings"):
        if spec.get(key) is not None and not isinstance(spec[key], dict):
            raise ProjectModelError(f"{path}: {key!r} must be a mapping")
    sheet = spec.get("sample_sheet")
    if sheet is not None and (not isinstance(sheet, dict) or not sheet.get("path")):
        raise ProjectModelError(f"{path}: 'sample_sheet' needs a 'path'")
    spec.setdefault("apply_samples", dict(spec["samples"]))
    spec.setdefault("exclude_references", [])
    spec.setdefault("settings", {})
    return spec


def spec_hash(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _registry_entry(project_dir: Path, experiment_id: str) -> tuple[dict, Path]:
    from .registry import _resolve_registry_path, load_registry

    registry = load_registry(project_dir)
    entry = registry.get("experiments", {}).get(experiment_id)
    if entry is None:
        raise ProjectModelError(f"experiment {experiment_id!r} is not registered in {project_dir}")
    return entry, _resolve_registry_path(entry["path"], project_dir).resolve()


def experiment_config(
    project_dir: str | Path, experiment_id: str, config_paths: Mapping[str, str] | None = None
):
    """(cfg, run_root) of a registered experiment."""
    from ..cli.helpers import load_experiment_config

    project_dir = Path(project_dir)
    _, run_root = _registry_entry(project_dir, experiment_id)
    explicit = (config_paths or {}).get(experiment_id)
    candidates = (
        [Path(explicit)]
        if explicit
        else [run_root / "experiment_config.csv", run_root.parent / "experiment_config.csv"]
    )
    refused = []
    for path in candidates:
        if not path.is_file():
            continue
        cfg = load_experiment_config(str(path))
        if Path(cfg.output_directory).resolve() == run_root:
            return cfg, run_root
        refused.append(f"{path} writes to {cfg.output_directory}")
    raise ProjectModelError(
        f"no config for {experiment_id} writes to {run_root}"
        + (f" ({'; '.join(refused)})" if refused else "")
        + f": pass --config {experiment_id}=PATH"
    )


def _spatial(cfg, run_root: Path) -> tuple[Path, str | None]:
    from ..cli.helpers import get_adata_paths
    from ..cli.hmm_layer_sets import _current
    from ..constants import SPATIAL_DIR

    source = getattr(get_adata_paths(cfg), "spatial_spine", None)
    if source is None or not Path(source).exists():
        raise ProjectModelError(f"{run_root}: no spatial spine to decode")
    generation, _ = _current(run_root / SPATIAL_DIR)
    return Path(source), generation


def _same(value: Any, wanted: Any) -> bool:
    """Sheet value equals the spec value, numerically when both are numbers."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    try:
        return float(value) == float(wanted)
    except (TypeError, ValueError):
        return str(value).strip() == str(wanted).strip()


def _barcode_key(value: Any, match_on: str) -> str | None:
    if match_on == "barcode_number":
        import re

        found = re.search(r"(\d+)\s*$", str(value))
        return str(int(found.group(1))) if found else None
    return str(value).strip()


def sample_attributes(
    project_dir: Path, spec: Mapping[str, Any], spine, cfg, experiment: str
) -> dict[str, dict[str, Any]]:
    """Barcode -> sample-sheet row for one experiment: the spec's project sheet
    when it names one, else the experiment's own configured sheet."""
    from ..constants import BARCODE
    from ..tools.partitioned_hmm import sample_sheet_attributes

    sheet_spec = spec.get("sample_sheet")
    if not sheet_spec:
        return sample_sheet_attributes(spine, cfg)
    path = Path(sheet_spec["path"])
    path = path if path.is_absolute() else Path(project_dir) / path
    sheet = pd.read_csv(path, dtype=str)
    experiment_column = sheet_spec.get("experiment_column")
    if experiment_column:
        sheet = sheet[sheet[experiment_column].astype(str) == str(experiment)]
    barcode_column = str(sheet_spec.get("barcode_column", BARCODE))
    match_on = str(sheet_spec.get("match_on", "exact"))
    rows: dict[str, dict[str, Any]] = {}
    for _, row in sheet.iterrows():
        key = _barcode_key(row[barcode_column], match_on)
        if key is None:
            continue
        if key in rows:
            raise ProjectModelError(f"{path}: {experiment} has two rows for barcode {key}")
        rows[key] = {k: v for k, v in row.items() if pd.notna(v)}
    attributes = {}
    for barcode in spine.obs[BARCODE].astype(str).unique():
        key = _barcode_key(barcode, match_on)
        if key in rows:
            attributes[str(barcode)] = rows[key]
    return attributes


def matching_barcodes(attributes: Mapping[str, Mapping[str, Any]], wanted: Mapping) -> set[str]:
    """Barcodes whose sample-sheet row has every ``wanted`` column value."""
    return {
        barcode
        for barcode, row in attributes.items()
        if all(_same(row.get(column), value) for column, value in wanted.items())
    }


# ---- fitting -------------------------------------------------------------------------------


def _fit_config(cfg, spec: Mapping[str, Any]):
    """The experiment's config set up for one model per reference group and strand,
    transitions and emissions fitted on the pooled selected reads."""
    fit_cfg = copy.copy(cfg)
    for key, value in (spec.get("settings") or {}).items():
        setattr(fit_cfg, key, value)
    fit_cfg.hmm_variants = {}
    fit_cfg._hmm_only_variant = None
    fit_cfg.hmm_reference_groups = {
        str(k): list(v) if not isinstance(v, str) else [v]
        for k, v in spec["reference_groups"].items()
    }
    fit_cfg.hmm_groupby = ["reference", "methbase"]
    fit_cfg.hmm_fit_strategy = "per_group"
    fit_cfg.hmm_emission_scope = "fit_group"
    fit_cfg.hmm_transition_scope = "fit_group"
    fit_cfg.hmm_distance_aware = False
    return fit_cfg


def _settings_snapshot(cfg) -> dict[str, Any]:
    """The HMM settings a decode reuses, JSON-safe."""
    values = {}
    for key in sorted(vars(cfg)):
        if (key.startswith("hmm_") or key in EXTRA_SETTINGS) and key not in APPLY_OWN_SETTINGS:
            values[key] = json.loads(json.dumps(getattr(cfg, key), default=str))
    return values


def _strand_group(reference: str) -> str:
    from ..hmm.fit_plan import _strand_of

    return str(_strand_of(str(reference))[1] or reference)


def fit_references(plan) -> tuple[list[str], str]:
    """A fit's references and their strand group. A fit pooled across references
    or tiled over windows (the planner's genome mode) names its references on its
    members, not in ``plan.reference``."""
    members = getattr(plan, "members", ()) or ()
    references = (
        sorted({str(member.reference) for member in members})
        if members
        else str(plan.reference).split("+")
    )
    strands = {_strand_group(reference) for reference in references}
    if len(strands) != 1:
        raise ProjectModelError(f"one fit spans strands {sorted(strands)}: {references}")
    return references, strands.pop()


def _container(project_dir: Path, name: str) -> Path:
    return Path(project_dir) / MODELS_SUBDIR / name


def current_model(project_dir: str | Path, name: str) -> tuple[str, dict, Path]:
    """(generation id, manifest, generation dir) of a stored model."""
    container = _container(Path(project_dir), name)
    pointer = container / "current.json"
    if not pointer.is_file():
        raise ProjectModelError(f"no project model {name!r} in {project_dir}")
    generation = json.loads(pointer.read_text(encoding="utf-8"))["generation_id"]
    directory = container / "generations" / generation
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    return generation, manifest, directory


def fit_project_model(
    project_dir: str | Path,
    spec_path: str | Path,
    *,
    config_paths: Mapping[str, str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit and store a project model; reuse the current one when its spec, the
    fitting settings and the training data generation are unchanged."""
    from ..hmm.model_artifacts import hmm_fit_config_hash
    from ..informatics.partition_read import load_spine
    from ..preprocessing.dispatch_plan import plan_preprocess_tasks
    from ..tools.partitioned_hmm import (
        _configured_model_specs,
        build_hmm_fit_plan,
        execute_hmm_fit_task,
        hmm_bytes_per_position,
    )

    project_dir = Path(project_dir)
    spec = load_spec(spec_path)
    name, experiment = str(spec["name"]), str(spec["experiment"])
    cfg, run_root = experiment_config(project_dir, experiment, config_paths)
    fit_cfg = _fit_config(cfg, spec)
    source, spatial_generation = _spatial(cfg, run_root)
    specs = _configured_model_specs(fit_cfg)
    if not specs:
        raise ProjectModelError(f"{experiment}: no HMM model for the configured signals")
    fingerprint = {
        "spec": spec_hash(spec),
        "fit_config_hashes": {s.label: hmm_fit_config_hash(s.config(fit_cfg)) for s in specs},
        "source_generation": spatial_generation,
    }
    container = _container(project_dir, name)
    if not force and (container / "current.json").is_file():
        generation, manifest, directory = current_model(project_dir, name)
        if manifest.get("fingerprint") == fingerprint:
            logger.info("Project model %s: %s is current; reused", name, generation)
            return {"name": name, "generation_id": generation, "reused": True, "path": directory}

    spine = load_spine(source)
    attributes = sample_attributes(project_dir, spec, spine, cfg, experiment)
    barcodes = matching_barcodes(attributes, spec["samples"])
    if not barcodes:
        raise ProjectModelError(f"{experiment}: no sample matches {spec['samples']}")
    excluded = {str(r) for r in spec.get("exclude_references") or []}
    filter_mask = next((c for c in ("passes_dedup", "passes_qc") if c in spine.obs), None)
    tasks = plan_preprocess_tasks(
        spine,
        target_task_memory_mb=int(getattr(fit_cfg, "target_task_memory_mb", 512)),
        partition_by_barcode=True,
        filter_mask=filter_mask,
        spine_path=source,
        bytes_per_position=hmm_bytes_per_position(fit_cfg, specs),
    )
    from ..hmm.fit_plan import reference_groups

    members = {member for group in reference_groups(fit_cfg).values() for member in group}

    def grouped(reference: str) -> bool:
        from ..hmm.fit_plan import _strand_of

        return reference in members or _strand_of(reference)[0] in members

    tasks = [
        t
        for t in tasks
        if str(t.barcode) in barcodes
        and str(t.reference) not in excluded
        and grouped(str(t.reference))
    ]
    if not tasks:
        raise ProjectModelError(f"{experiment}: the selected samples have no grouped reference")
    planning = build_hmm_fit_plan(tasks, specs, fit_cfg)

    generation = uuid.uuid4().hex
    staging = container / ".staging" / generation
    models_dir = staging / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    groups: dict[str, list[str]] = {}
    plans, training = [], []
    for plan in planning.base_plans:
        artifact = execute_hmm_fit_task(source, plan, fit_cfg, models_dir, None)
        references, strand = fit_references(plan)
        if bool(artifact.get("skipped", False)):
            raise ProjectModelError(
                f"{name}: fit of {plan.model_spec.label} on {plan.reference} failed: "
                f"{artifact.get('reason')}"
            )
        record = {k: v for k, v in artifact.items() if k not in ("fit_id", "parent_fit_id")}
        for key in ("checkpoint", "metadata"):  # as decode tasks record them
            path = Path(str(record[key]))
            if not path.parts or path.parts[0] != models_dir.name:
                record[key] = (Path(models_dir.name) / path).as_posix()
        artifacts.setdefault(strand, {})[plan.model_spec.label] = record
        groups[strand] = sorted(set(groups.get(strand, [])) | set(references))
        plans.append(
            {
                "fit_id": plan.fit_id,
                "strand_group": strand,
                "references": references,
                "label": plan.model_spec.label,
                "model_id": record.get("model_id"),
                "n_training_reads": len(plan.selected_read_ids),
            }
        )
        training += [
            {"read_id": str(r), "strand_group": strand, "label": plan.model_spec.label}
            for r in plan.selected_read_ids
        ]
    pd.DataFrame(training).to_parquet(staging / TRAINING_READS, index=False)
    from importlib.metadata import version

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "generation_id": generation,
        "spec": spec,
        "fingerprint": fingerprint,
        "training": {
            "experiment": experiment,
            "run_root": str(run_root),
            "spatial_generation": spatial_generation,
            "barcodes": sorted(barcodes),
            "sample_rows": {b: attributes[b] for b in sorted(barcodes)},
        },
        "reference_groups": groups,
        "artifacts": artifacts,
        "fits": plans,
        "settings": _settings_snapshot(fit_cfg),
        "smftools_version": version("smftools"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    final = container / "generations" / generation
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    shutil.rmtree(container / ".staging", ignore_errors=True)
    pointer = container / "current.json"
    tmp = pointer.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"generation_id": generation}))
    os.replace(tmp, pointer)
    logger.info("Project model %s: stored %s (%d model(s))", name, generation, len(plans))
    return {"name": name, "generation_id": generation, "reused": False, "path": final}


# ---- applying ------------------------------------------------------------------------------


def frozen_models(
    project_dir: str | Path, name: str, *, barcodes=None, training_reads: bool = False
):
    """The stored model as ``FrozenModels`` for a decode."""
    from ..hmm.project_models import FrozenModels

    generation, manifest, directory = current_model(project_dir, name)
    reads: frozenset[str] = frozenset()
    if training_reads and (directory / TRAINING_READS).is_file():
        reads = frozenset(pd.read_parquet(directory / TRAINING_READS)["read_id"].astype(str))
    return FrozenModels(
        name=name,
        generation_id=generation,
        models_dir=directory / "models",
        artifacts=manifest["artifacts"],
        reference_groups=manifest["reference_groups"],
        barcodes=None if barcodes is None else frozenset(map(str, barcodes)),
        exclude_references=frozenset(map(str, manifest["spec"].get("exclude_references") or [])),
        training_read_ids=reads,
    )


def apply_config(cfg, manifest: Mapping[str, Any]):
    """The decoding experiment's config with the model's HMM settings."""
    applied = copy.copy(cfg)
    for key, value in manifest.get("settings", {}).items():
        setattr(applied, key, value)
    applied.hmm_variants = {}
    applied._hmm_only_variant = None
    return applied


def apply_project_model(
    project_dir: str | Path,
    name: str,
    *,
    experiments: list[str] | None = None,
    config_paths: Mapping[str, str] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Decode registered experiments with a stored model, each into the layer set
    ``project-<name>``. Experiments with no matching sample are reported, not
    decoded."""
    from ..cli.hmm_layer_sets import publish_layer_set
    from ..informatics.partition_read import load_spine
    from .registry import load_registry

    project_dir = Path(project_dir)
    generation, manifest, _ = current_model(project_dir, name)
    if experiments is None:
        registry = load_registry(project_dir)
        experiments = sorted(
            k for k, v in registry.get("experiments", {}).items() if v.get("status") == "active"
        )
    results = []
    for experiment in experiments:
        try:
            cfg, run_root = experiment_config(project_dir, experiment, config_paths)
            source, spatial_generation = _spatial(cfg, run_root)
            attributes = sample_attributes(
                project_dir, manifest["spec"], load_spine(source), cfg, experiment
            )
        except (ProjectModelError, ValueError, FileNotFoundError) as exc:
            results.append({"experiment": experiment, "status": "skipped", "reason": str(exc)})
            continue
        barcodes = matching_barcodes(attributes, manifest["spec"]["apply_samples"])
        if not barcodes:
            results.append(
                {"experiment": experiment, "status": "skipped", "reason": "no matching sample"}
            )
            continue
        applied = apply_config(cfg, manifest)
        applied._hmm_frozen_models = frozen_models(
            project_dir,
            name,
            barcodes=barcodes,
            training_reads=experiment == manifest["training"]["experiment"],
        )
        published = publish_layer_set(
            run_root,
            source,
            applied,
            f"{LAYER_SET_PREFIX}{name}",
            source_generation=spatial_generation,
            force=force,
            project_model={
                "name": name,
                "generation_id": generation,
                "barcodes": sorted(barcodes),
            },
        )
        results.append({"experiment": experiment, "status": "decoded", **published})
    return results


def list_project_models(project_dir: str | Path) -> list[dict[str, Any]]:
    """Every stored model's current generation, training data and models."""
    root = Path(project_dir) / MODELS_SUBDIR
    rows = []
    for container in sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []:
        try:
            generation, manifest, _ = current_model(project_dir, container.name)
        except (ProjectModelError, FileNotFoundError, KeyError):
            continue
        rows.append(
            {
                "name": container.name,
                "generation_id": generation,
                "experiment": manifest["training"]["experiment"],
                "barcodes": ",".join(manifest["training"]["barcodes"]),
                "groups": ",".join(sorted(manifest["artifacts"])),
                "labels": ",".join(sorted({f["label"] for f in manifest["fits"]})),
                "created_at": manifest.get("created_at"),
            }
        )
    return rows
