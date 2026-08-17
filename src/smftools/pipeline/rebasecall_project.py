"""Fan one re-basecall request out across a project's experiments.

One request, one lineage per selected experiment. Each experiment resolves its
own chemistry and model bundle, because a project routinely spans flow cells and
kits: forcing one resolution across all of them would either block the whole
project on the odd one out or, worse, silently basecall an experiment with a
model chosen for a different chemistry.

Planning is read-only. Running publishes each experiment's lineage
independently and registers it *without* changing what the project resolves --
promotion stays explicit, and stays `SRB-08`'s.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..project.registry import ORIGINAL_LINEAGE, register_experiment_lineage
from .rebasecall_lineage import RebasecallLineageError
from .rebasecall_request import RebasecallRequest

REBASECALL_PROJECT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectRebasecallMember:
    """One project experiment's place in a fanned-out re-basecall request."""

    experiment_id: str
    experiment_uid: str | None
    run_root: Path
    config_path: Path | None
    status: str
    blockers: tuple[str, ...] = ()
    plan_id: str | None = None
    model: Mapping[str, Any] | None = None
    lineage_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_uid": self.experiment_uid,
            "run_root": str(self.run_root),
            "config_path": None if self.config_path is None else str(self.config_path),
            "status": self.status,
            "blockers": list(self.blockers),
            "plan_id": self.plan_id,
            "model": dict(self.model) if self.model else None,
            "lineage_id": self.lineage_id,
        }


@dataclass(frozen=True)
class ProjectRebasecallPlan:
    """A read-only, per-experiment view of what one request would do."""

    project_dir: Path
    request_id: str
    members: tuple[ProjectRebasecallMember, ...] = field(default_factory=tuple)
    schema_version: int = REBASECALL_PROJECT_SCHEMA_VERSION

    @property
    def ready(self) -> tuple[ProjectRebasecallMember, ...]:
        return tuple(member for member in self.members if member.status == "ready")

    @property
    def blocked(self) -> tuple[ProjectRebasecallMember, ...]:
        return tuple(member for member in self.members if member.status != "ready")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_dir": str(self.project_dir),
            "request_id": self.request_id,
            "member_count": len(self.members),
            "ready_count": len(self.ready),
            "blocked_count": len(self.blocked),
            "members": [member.to_dict() for member in self.members],
        }


def _experiment_config_path(run_root: Path) -> Path | None:
    candidate = run_root / "experiment_config.csv"
    return candidate if candidate.is_file() else None


def _selected_entries(
    project_dir: Path,
    *,
    experiments: Sequence[str] | None,
    set_name: str | None,
    catalog_opener: Callable[..., Any] | None,
) -> list[dict[str, Any]]:
    """Resolve which registered experiments this request covers.

    Selection always runs through the project catalog, so a fan-out sees exactly
    the experiments — and exactly the lineage — an ordinary project query would.
    """
    if catalog_opener is None:
        from ..project.catalog import ProjectCatalog

        catalog_opener = ProjectCatalog.open
    catalog = catalog_opener(project_dir)
    entries = catalog.experiments()
    if set_name is not None:
        from ..project.registry import resolve_set_membership

        members = set(resolve_set_membership(project_dir, set_name, catalog=catalog).resolved)
        entries = [entry for entry in entries if entry["id"] in members]
    if experiments is not None:
        wanted = {str(value) for value in experiments}
        missing = wanted.difference({entry["id"] for entry in entries})
        if missing:
            raise RebasecallLineageError(
                "project_experiment_unknown",
                f"project has no registered experiment(s): {sorted(missing)}",
            )
        entries = [entry for entry in entries if entry["id"] in wanted]
    if not entries:
        raise RebasecallLineageError(
            "project_selection_empty",
            "the request selects no registered experiments",
        )
    return entries


def plan_project_rebasecall(
    project_dir: str | Path,
    request: RebasecallRequest,
    *,
    experiments: Sequence[str] | None = None,
    set_name: str | None = None,
    planner: Callable[..., Any] | None = None,
    catalog_opener: Callable[..., Any] | None = None,
) -> ProjectRebasecallPlan:
    """Plan one lineage per selected experiment without writing anything.

    A blocked experiment does not block the others: each is reported with its
    own blockers so a project-wide request can be judged as a whole before any
    of it runs.
    """
    project_dir = Path(project_dir)
    if planner is None:
        from .rebasecall_plan import build_rebasecall_plan as planner
    entries = _selected_entries(
        project_dir,
        experiments=experiments,
        set_name=set_name,
        catalog_opener=catalog_opener,
    )

    members: list[ProjectRebasecallMember] = []
    for entry in sorted(entries, key=lambda item: str(item["id"])):
        run_root = Path(str(entry["path"]))
        config_path = _experiment_config_path(run_root)
        if config_path is None:
            members.append(
                ProjectRebasecallMember(
                    experiment_id=str(entry["id"]),
                    experiment_uid=entry.get("experiment_uid"),
                    run_root=run_root,
                    config_path=None,
                    status="blocked",
                    blockers=("experiment_config_missing",),
                )
            )
            continue
        try:
            from ..cli.helpers import load_experiment_config

            cfg = load_experiment_config(str(config_path))
            # Each experiment resolves its own chemistry and model bundle.
            plan = planner(cfg, request)
        except Exception as exc:  # planning must never take the project down
            members.append(
                ProjectRebasecallMember(
                    experiment_id=str(entry["id"]),
                    experiment_uid=entry.get("experiment_uid"),
                    run_root=run_root,
                    config_path=config_path,
                    status="blocked",
                    blockers=(f"{type(exc).__name__}: {exc}",),
                )
            )
            continue
        members.append(
            ProjectRebasecallMember(
                experiment_id=str(entry["id"]),
                experiment_uid=entry.get("experiment_uid"),
                run_root=run_root,
                config_path=config_path,
                status=plan.status,
                blockers=tuple(reason.code for reason in plan.blockers),
                plan_id=plan.plan_id,
                model=plan.model.to_dict(),
            )
        )
    return ProjectRebasecallPlan(
        project_dir=project_dir,
        request_id=request.request_id,
        members=tuple(members),
    )


def run_project_rebasecall(
    project_dir: str | Path,
    request: RebasecallRequest,
    rebasecall_root: str | Path,
    *,
    accepted_request_id: str,
    experiments: Sequence[str] | None = None,
    set_name: str | None = None,
    planner: Callable[..., Any] | None = None,
    catalog_opener: Callable[..., Any] | None = None,
    lineage_runner: Callable[..., Any] | None = None,
) -> ProjectRebasecallPlan:
    """Publish and register one lineage per ready experiment.

    Each experiment is independent: one failure is recorded against that member
    and the rest still run, because a project-wide re-basecall that aborted
    halfway would leave the user reasoning about which half happened.

    Registration never changes what the project resolves. A newly published
    lineage is queryable by name; making it the answer is explicit promotion.
    """
    project_dir = Path(project_dir)
    if accepted_request_id != request.request_id:
        raise RebasecallLineageError(
            "accepted_request_mismatch",
            "the supplied accepted request ID does not match the request",
        )
    lineage_runner = lineage_runner or _run_member_lineage

    planned = plan_project_rebasecall(
        project_dir,
        request,
        experiments=experiments,
        set_name=set_name,
        planner=planner,
        catalog_opener=catalog_opener,
    )

    completed: list[ProjectRebasecallMember] = []
    for member in planned.members:
        if member.status != "ready" or member.config_path is None:
            completed.append(member)
            continue
        try:
            result = lineage_runner(
                member=member,
                request=request,
                root=Path(rebasecall_root) / member.experiment_id,
            )
        except Exception as exc:
            completed.append(
                replace(
                    member,
                    status="failed",
                    blockers=(f"{type(exc).__name__}: {exc}",),
                )
            )
            continue
        lineage_id = str(result.lineage.lineage_id)
        register_experiment_lineage(
            project_dir,
            member.experiment_id,
            lineage_id,
            spines=_lineage_spines(result),
            metadata={
                "basecall_id": result.lineage.basecall_id,
                "request_id": request.request_id,
                "stage_generations": dict(result.lineage.stage_generations),
            },
        )
        completed.append(
            ProjectRebasecallMember(
                experiment_id=member.experiment_id,
                experiment_uid=member.experiment_uid,
                run_root=member.run_root,
                config_path=member.config_path,
                status="published",
                plan_id=member.plan_id,
                model=member.model,
                lineage_id=lineage_id,
            )
        )
    return ProjectRebasecallPlan(
        project_dir=project_dir,
        request_id=request.request_id,
        members=tuple(completed),
    )


def _run_member_lineage(
    *,
    member: ProjectRebasecallMember,
    request: RebasecallRequest,
    root: Path,
) -> Any:
    """Freeze, basecall, and publish one experiment's lineage.

    The per-experiment chain is exactly what `experiment rebasecall` does; the
    project layer only decides which experiments it runs for and where their
    artifacts land.
    """
    from ..cli.helpers import load_experiment_config
    from .rebasecall_basecall import prepare_rebasecall_basecall
    from .rebasecall_plan import build_rebasecall_plan
    from .rebasecall_run import run_lineage_raw_stage
    from .rebasecall_selection import freeze_rebasecall_selection

    assert member.config_path is not None
    cfg = load_experiment_config(str(member.config_path))
    plan = build_rebasecall_plan(cfg, request)
    if plan.plan_id != member.plan_id:
        raise RebasecallLineageError(
            "accepted_plan_mismatch",
            f"experiment {member.experiment_id!r} changed between planning and running",
        )
    selection = freeze_rebasecall_selection(
        plan,
        root / "selection-results",
        accepted_plan_id=plan.plan_id,
    )
    basecall = prepare_rebasecall_basecall(
        cfg,
        request,
        root / "selection-results",
        root / "basecalls",
        accepted_plan_id=plan.plan_id,
        signal_root=root / "signal-results" if request.signal.materialize else None,
    )
    return run_lineage_raw_stage(
        plan,
        selection,
        basecall,
        root / "rebasecall_outputs",
        accepted_plan_id=plan.plan_id,
        parent_config_path=member.config_path,
    )


_STAGE_OUTPUT_DIRS = {
    "raw": "raw_outputs",
    "preprocess": "preprocess_adata_outputs",
    "spatial": "spatial_adata_outputs",
    "hmm": "hmm_adata_outputs",
    "latent": "latent_adata_outputs",
}


def _lineage_spines(result: Any) -> dict[str, Path]:
    """Map a published lineage's stage generations to their spine paths."""
    run_root = Path(result.run_root)
    spines: dict[str, Path] = {}
    for stage, generation_id in result.lineage.stage_generations.items():
        directory = _STAGE_OUTPUT_DIRS.get(stage)
        if directory is None:
            continue
        spines[stage] = run_root / directory / "generations" / generation_id / "spine.h5ad"
    if not spines:
        raise RebasecallLineageError(
            "lineage_spines_unavailable",
            "the published lineage records no stage generation to register",
        )
    return spines


def format_project_rebasecall_plan(plan: ProjectRebasecallPlan) -> str:
    """Render a project fan-out plan for a terminal."""
    lines = [
        f"Project: {plan.project_dir}",
        f"Request ID: {plan.request_id}",
        f"Experiments: {len(plan.members)} ({len(plan.ready)} ready, {len(plan.blocked)} blocked)",
        "",
    ]
    for member in plan.members:
        model = (member.model or {}).get("simplex_model") or {}
        detail = model.get("name") if isinstance(model, Mapping) else None
        lines.append(
            f"- {member.experiment_id}: {member.status}"
            + (f" [{detail}]" if detail else "")
            + (f" -> lineage {member.lineage_id}" if member.lineage_id else "")
        )
        for blocker in member.blockers:
            lines.append(f"    blocked: {blocker}")
    lines.extend(
        (
            "",
            f"Registration never changes the active lineage; it stays {ORIGINAL_LINEAGE!r} "
            "until an explicit promotion.",
        )
    )
    return "\n".join(lines)
