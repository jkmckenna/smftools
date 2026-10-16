"""Where the bytes are, and what reclaims them (`SLC-11`).

One read-only report over a directory of run roots and projects:

- **experiments** (every directory with an ``experiment_manifest.json``):
  stage generations by disposition -- current, pinned, read by a live
  generation, prune candidate (nothing live reads it), unreadable -- from the
  `SLC-07` planner, plus the experiment's other bytes (inputs, BAMs, ...);
- **unmanaged directories**: directories beside an experiment root, in its run
  directory, that are not experiment roots themselves -- earlier output trees,
  test reruns. smftools never removes these (another project may read them);
- **ML workspaces** (``project_outputs/ml``, ``<experiment>/ml_outputs``):
  model bytes by architecture, models no run ever read, uncompressed
  payloads, already-retired models;
- **project analysis caches**: current vs stale / invalid bytes.

Projects are found under the scanned directory (a ``registry.json`` beside
``project_outputs``) or passed explicitly; their registries mark which
experiment roots are registered and protect the generations they name.
Nothing is written.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from smftools.constants import (
    ML_EXPERIMENT_OUTPUTS_DIR,
    ML_INDEX_DIR,
    ML_MODELS_DIR,
    ML_PROJECT_OUTPUTS_DIR,
    ML_RUNS_DIR,
)

USAGE_SCHEMA_VERSION = 1
EXPERIMENT_MANIFEST = "experiment_manifest.json"
PROJECT_REGISTRY = "registry.json"
PROJECT_OUTPUTS = "project_outputs"
_SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache"})


def tree_bytes(path: Path, *, skip_names: frozenset[str] = frozenset()) -> int:
    """Bytes of regular files under ``path`` (symlinks not followed)."""
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in skip_names]
        for name in files:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            if not os.path.islink(os.path.join(root, name)):
                total += stat.st_size
    return total


def _discover(directory: Path) -> tuple[list[Path], list[Path]]:
    """Experiment roots and project directories under ``directory``."""
    experiments, projects = [], []
    for root, dirs, files in os.walk(directory):
        here = Path(root)
        dirs[:] = sorted(
            name for name in dirs if not name.startswith(".") and name not in _SKIP_DIRS
        )
        if EXPERIMENT_MANIFEST in files:
            experiments.append(here)
            dirs[:] = []  # no experiment nests inside another
        elif PROJECT_REGISTRY in files and PROJECT_OUTPUTS in dirs:
            projects.append(here)
            dirs[:] = []
        elif "generations" in dirs:
            dirs.remove("generations")
    return experiments, projects


def _registered_roots(projects: Iterable[Path]) -> set[str]:
    """Real paths of the experiment roots the projects' registries point into."""
    roots: set[str] = set()
    for project in projects:
        registry = project / PROJECT_REGISTRY
        try:
            experiments = json.loads(registry.read_text(encoding="utf-8")).get("experiments", {})
        except (OSError, ValueError):
            continue
        for record in experiments.values():
            for value in (record.get("spines") or {}).values():
                path = (project / str(value)).resolve()
                for parent in (path, *path.parents):
                    if (parent / EXPERIMENT_MANIFEST).is_file():
                        roots.add(str(parent))
                        break
    return roots


@dataclass
class Bucket:
    count: int = 0
    bytes: int = 0

    def add(self, size: int | None) -> None:
        self.count += 1
        self.bytes += int(size or 0)

    def to_dict(self) -> dict[str, int]:
        return {"count": self.count, "bytes": self.bytes}


@dataclass
class ExperimentUsage:
    root: str
    registered: bool | None
    generations: dict[str, Bucket] = field(default_factory=dict)
    by_stage: dict[str, dict[str, Bucket]] = field(default_factory=dict)
    other_bytes: int = 0
    error: str | None = None

    @property
    def total_bytes(self) -> int:
        return self.other_bytes + sum(b.bytes for b in self.generations.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "registered": self.registered,
            "total_bytes": self.total_bytes,
            "other_bytes": self.other_bytes,
            "generations": {k: v.to_dict() for k, v in sorted(self.generations.items())},
            "by_stage": {
                stage: {k: v.to_dict() for k, v in sorted(buckets.items())}
                for stage, buckets in sorted(self.by_stage.items())
            },
            "error": self.error,
        }


# Planner dispositions -> report buckets
_BUCKETS = {
    "keep_current": "current",
    "keep_pinned": "pinned",
    "keep_referenced": "referenced",
    "prune_candidate": "prune_candidate",
    "keep_references_unknown": "references_unknown",
    "keep_unreadable": "unreadable",
}


def experiment_usage(
    root: Path, *, protect_from: Iterable[Path] = (), registered: bool | None = None
) -> ExperimentUsage:
    from smftools.informatics.generation_pruning import plan_experiment_generation_prune

    usage = ExperimentUsage(root=root.as_posix(), registered=registered)
    try:
        plan = plan_experiment_generation_prune(root, keep_last=0, protect_from=list(protect_from))
        for decision in plan.decisions:
            bucket = _BUCKETS.get(decision.disposition, decision.disposition)
            usage.generations.setdefault(bucket, Bucket()).add(decision.size_bytes)
            usage.by_stage.setdefault(decision.kind, {}).setdefault(bucket, Bucket()).add(
                decision.size_bytes
            )
    except Exception as exc:  # noqa: BLE001 -- reported, never fatal to the report
        usage.error = f"{type(exc).__name__}: {exc}"
    usage.other_bytes = tree_bytes(root, skip_names=frozenset({"generations"}))
    return usage


def _unmanaged(experiments: list[Path], projects: list[Path]) -> list[dict[str, Any]]:
    """Directories beside experiment roots that are neither experiment roots
    nor contain one (nor are projects)."""
    claimed = {str(path) for path in (*experiments, *projects)}
    run_dirs = sorted({path.parent for path in experiments})
    found = []
    for run_dir in run_dirs:
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir() or child.is_symlink() or child.name.startswith("."):
                continue
            if str(child) in claimed or any(c.startswith(str(child) + os.sep) for c in claimed):
                continue
            found.append({"path": child.as_posix(), "bytes": tree_bytes(child)})
    return found


def _is_ml_workspace(path: Path) -> bool:
    return (path / ML_MODELS_DIR).is_dir() and (path / ML_RUNS_DIR).is_dir()


def _payload_compressed(path: Path) -> bool | None:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(item.compress_type != zipfile.ZIP_STORED for item in archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return None


def ml_workspace_usage(workspace: Path) -> dict[str, Any]:
    """Model store accounting for one ML workspace."""
    read: set[str] = set()
    runs = 0
    for manifest in sorted((workspace / ML_RUNS_DIR).glob("*/run_manifest.json")):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runs += 1
        read.update(record.get("source_model_ids") or ())
    by_architecture: dict[str, Bucket] = {}
    never_read, uncompressed, total = Bucket(), Bucket(), Bucket()
    for manifest in sorted((workspace / ML_MODELS_DIR).glob("*/model_manifest.json")):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        size = tree_bytes(manifest.parent)
        name = (record.get("architecture") or {}).get("name", "unknown")
        by_architecture.setdefault(name, Bucket()).add(size)
        total.add(size)
        if record.get("model_id") not in read:
            never_read.add(size)
        artifact = record.get("artifact") or {}
        if str(artifact.get("media_type", "")).endswith("+zip"):
            payload = manifest.parent / str(artifact.get("relative_path", ""))
            if _payload_compressed(payload) is False:
                uncompressed.add(size)
    retired = Bucket()
    retired_path = workspace / ML_INDEX_DIR / "retired_models.json"
    if retired_path.is_file():
        try:
            for item in json.loads(retired_path.read_text(encoding="utf-8")).get("records", []):
                retired.add(item.get("size_bytes"))
        except (OSError, ValueError):
            pass
    return {
        "root": workspace.as_posix(),
        "runs": runs,
        "runs_bytes": tree_bytes(workspace / ML_RUNS_DIR),
        "models": total.to_dict(),
        "by_architecture": {
            name: bucket.to_dict()
            for name, bucket in sorted(by_architecture.items(), key=lambda kv: -kv[1].bytes)
        },
        "never_read": never_read.to_dict(),
        "uncompressed": uncompressed.to_dict(),
        "retired_already": retired.to_dict(),
    }


def _cache_usage(project: Path) -> dict[str, Any] | None:
    from smftools.project.analysis_inventory import analysis_cache_inventory

    try:
        entries = analysis_cache_inventory(project)["entries"]
    except Exception:  # noqa: BLE001
        return None
    by_status: Counter = Counter()
    for entry in entries:
        by_status[entry["status"]] += int(entry.get("size_bytes") or 0)
    return {"project": project.as_posix(), "bytes_by_status": dict(by_status)}


def _ml_workspaces(experiments: list[Path], projects: list[Path]) -> Iterator[Path]:
    for project in projects:
        candidate = project / PROJECT_OUTPUTS / ML_PROJECT_OUTPUTS_DIR
        if _is_ml_workspace(candidate):
            yield candidate
    for root in experiments:
        candidate = root / ML_EXPERIMENT_OUTPUTS_DIR
        if _is_ml_workspace(candidate):
            yield candidate


def storage_usage(directory: str | Path, *, projects: Iterable[str | Path] = ()) -> dict[str, Any]:
    """The full usage report for ``directory`` (read-only)."""
    directory = Path(directory).resolve()
    experiments, found_projects = _discover(directory)
    all_projects = sorted({*found_projects, *(Path(p).resolve() for p in projects)})
    registered = _registered_roots(all_projects) if all_projects else None
    experiment_reports = [
        experiment_usage(
            root,
            protect_from=all_projects,
            registered=None if registered is None else str(root.resolve()) in registered,
        )
        for root in experiments
    ]
    unmanaged = _unmanaged(experiments, found_projects)
    workspaces = [ml_workspace_usage(path) for path in _ml_workspaces(experiments, all_projects)]
    caches = [c for c in (_cache_usage(p) for p in all_projects) if c is not None]

    def total(bucket: str) -> int:
        return sum(e.generations.get(bucket, Bucket()).bytes for e in experiment_reports)

    unregistered = sum(e.total_bytes for e in experiment_reports if e.registered is False)
    reclaim = [
        {
            "bucket": "generations nothing live reads",
            "bytes": total("prune_candidate"),
            "how": "smftools experiment generations ROOT prune --keep-last N "
            "--protect-from PROJECT (plan; applying: SLC-08)",
        },
        {
            "bucket": "uncompressed model payloads",
            "bytes": sum(w["uncompressed"]["bytes"] for w in workspaces),
            "how": "compaction (SLC-02); recovers most of these bytes",
        },
        {
            "bucket": "models no run ever read",
            "bytes": sum(w["never_read"]["bytes"] for w in workspaces),
            "how": "smftools project ml-retention (superseded runs); unread models: SLC-04",
        },
        {
            "bucket": "unmanaged directories beside experiment roots",
            "bytes": sum(item["bytes"] for item in unmanaged),
            "how": "archive or remove by hand after checking no other project reads them",
        },
        {
            "bucket": "experiment roots no scanned project registers",
            "bytes": unregistered,
            "how": "check whether any project still reads them; archive",
        },
        {
            "bucket": "stale or invalid analysis caches",
            "bytes": sum(
                b
                for c in caches
                for status, b in c["bytes_by_status"].items()
                if status != "current"
            ),
            "how": "smftools project analyses list PROJECT --stale",
        },
    ]
    return {
        "schema_version": USAGE_SCHEMA_VERSION,
        "dry_run": True,
        "root": directory.as_posix(),
        "projects": [p.as_posix() for p in all_projects],
        "experiments": [e.to_dict() for e in experiment_reports],
        "unmanaged": unmanaged,
        "ml_workspaces": workspaces,
        "analysis_caches": caches,
        "reclaim": reclaim,
    }


def _human(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000
    return f"{size:.1f} TB"


def render_usage(report: dict[str, Any], *, top: int = 15) -> str:
    """A readable summary: totals per bucket, largest items, what reclaims them."""
    root = Path(report["root"])

    def rel(path: str) -> str:
        try:
            return Path(path).relative_to(root).as_posix() or "."
        except ValueError:
            return path

    lines = [f"Storage under {root} (read-only report)", ""]
    experiments = sorted(report["experiments"], key=lambda e: -e["total_bytes"])
    lines.append("EXPERIMENT ROOTS            total     current   referenced   prunable    other")
    for e in experiments[:top]:
        g = e["generations"]
        flag = "" if e["registered"] in (None, True) else "  [not registered]"
        lines.append(
            f"  {rel(e['root'])[:26]:<26} {_human(e['total_bytes']):>9} "
            f"{_human(g.get('current', {}).get('bytes', 0)):>10} "
            f"{_human(g.get('referenced', {}).get('bytes', 0) + g.get('pinned', {}).get('bytes', 0)):>11} "
            f"{_human(g.get('prune_candidate', {}).get('bytes', 0)):>10} "
            f"{_human(e['other_bytes']):>8}{flag}" + (f"  ERROR {e['error']}" if e["error"] else "")
        )
    if len(experiments) > top:
        lines.append(f"  ... {len(experiments) - top} more (--json for all)")
    if report["unmanaged"]:
        lines += ["", "UNMANAGED DIRECTORIES (beside experiment roots)"]
        for item in sorted(report["unmanaged"], key=lambda i: -i["bytes"])[:top]:
            lines.append(f"  {_human(item['bytes']):>9}  {rel(item['path'])}")
    for w in report["ml_workspaces"]:
        lines += [
            "",
            f"ML WORKSPACE {rel(w['root'])}: {w['models']['count']} models, "
            f"{_human(w['models']['bytes'])}; runs {_human(w['runs_bytes'])}",
        ]
        for name, bucket in list(w["by_architecture"].items())[:6]:
            lines.append(f"  {name:<28} {bucket['count']:>6} {_human(bucket['bytes']):>10}")
        lines.append(
            f"  never read by a run: {w['never_read']['count']} ({_human(w['never_read']['bytes'])}); "
            f"uncompressed: {w['uncompressed']['count']} ({_human(w['uncompressed']['bytes'])}); "
            f"retired already: {w['retired_already']['count']} "
            f"({_human(w['retired_already']['bytes'])})"
        )
    lines += ["", "RECLAIMABLE (by bucket; buckets overlap for ML models)"]
    for item in report["reclaim"]:
        if item["bytes"]:
            lines.append(f"  {_human(item['bytes']):>9}  {item['bucket']}: {item['how']}")
    return "\n".join(lines)
