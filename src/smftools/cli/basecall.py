"""`smftools basecall`: publish an immutable basecall generation from signal.

A top-level command, not nested under `experiment` or `project` -- per
`src/smftools/cli/AGENTS.md`, that choice is recorded here deliberately: the
config-free form (`BCS-10`) is scoped to neither an experiment nor a project.

Two invocation forms, one core (`BCS-05`, `BCS-10`): `basecall(config_path)`
loads a config and reuses its model parameters (`model`, `model_dir`,
`barcode_kit`, ...); `run_from_paths` takes them as arguments instead, for
basecalling a drawer of runs off an archive drive with no experiment config
in sight. Both call `basecall_core` with a `cfg`-shaped object and publish
the same `basecall_outputs/` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from smftools.constants import BAM_SUFFIX, BASECALL_DIR
from smftools.logging_utils import get_logger, mark_stage_outcome, stage_logging_lifecycle

logger = get_logger(__name__)


class BasecallInputError(ValueError):
    """Raised when `input_data_path` is not signal, so there is nothing to basecall."""


def basecall(config_path: str) -> dict[str, Any]:
    """Load a config and publish (or reuse) its basecall generation."""
    from .helpers import load_experiment_config

    cfg = load_experiment_config(config_path)
    return basecall_core(cfg)


@dataclass
class _ConfigFreeBasecallConfig:
    """A minimal `cfg`-shaped object for `basecall_core`, built from bare paths (`BCS-10`).

    Carries exactly the attributes `basecall_core` (and the
    `resolved_input_source_identities` it calls) reads -- not a real
    `ExperimentConfig`, which requires an alignment reference and experiment
    metadata basecalling itself never touches. `resolved_stage_config` falls
    back to ``vars(cfg)`` for any object without `to_dict`, so this dataclass
    hashes for idempotency the same way a real config would.
    """

    input_type: str
    input_data_path: Path
    input_files: list[Path]
    output_directory: Path
    model: str
    model_dir: Path
    smf_modality: str
    device: str
    barcode_kit: Optional[str] = None
    barcode_both_ends: bool = False
    emit_moves: bool = True
    trim: bool = False
    mod_list: list[str] = field(default_factory=list)
    bam_suffix: str = BAM_SUFFIX
    sequencer: str = "ont"
    max_basecall_reads: Optional[int] = None
    force_redo_load_adata: bool = False
    input_manifest_path: Optional[str] = None
    alignment_mode: str = "align"
    fastq_barcode_map: Optional[dict] = None
    fastq_auto_pairing: bool = True


def run_from_paths(
    *,
    input_path: str | Path,
    output_directory: str | Path,
    model: str,
    model_dir: str | Path,
    barcode_kit: str | None = None,
    modifications: str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Publish (or reuse) a basecall generation without an experiment config (`BCS-10`).

    `modifications` implies modified basecalling when given (comma-separated,
    e.g. ``"5mC_5hmC"``) and canonical basecalling when omitted -- the same
    `direct`/not-`direct` distinction a config's `smf_modality` drives, since
    a config-free invocation has no SMF modality of its own to consult.
    """
    from ..config.discover_input_files import discover_input_files

    input_path = Path(input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    found = discover_input_files(input_path)
    is_pod5, is_fast5 = found["input_is_pod5"], found["input_is_fast5"]
    if is_pod5 and is_fast5:
        raise BasecallInputError(
            f"{input_path} contains both POD5 and FAST5 files; point --input at one "
            "signal representation."
        )
    if is_pod5:
        input_type, input_files = "pod5", list(found["pod5_paths"])
    elif is_fast5:
        input_type, input_files = "fast5", list(found["fast5_paths"])
    else:
        raise BasecallInputError(f"{input_path} contains no POD5 or FAST5 files to basecall.")

    mod_list = (
        [item.strip() for item in modifications.split(",") if item.strip()] if modifications else []
    )

    cfg = _ConfigFreeBasecallConfig(
        input_type=input_type,
        input_data_path=input_path,
        input_files=input_files,
        output_directory=output_directory,
        model=model,
        model_dir=Path(model_dir),
        smf_modality="direct" if mod_list else "canonical",
        device=device,
        barcode_kit=barcode_kit,
        mod_list=mod_list,
    )
    return basecall_core(cfg)


def basecall_core(cfg) -> dict[str, Any]:
    """Publish an immutable basecall generation from `cfg`'s POD5/FAST5 input.

    Reuses whatever a matching prior run already committed -- both the
    `dorado-basecalling` intermediate (shared with `raw`'s own inline
    basecalling; see `informatics.basecall_execution`) and, if
    `basecall_outputs`'s current generation already matches this config, the
    generation itself, which is left untouched rather than republished.

    Raises:
        BasecallInputError: `cfg.input_type` is not `pod5`/`fast5` -- there is
            nothing to basecall. Reusing an existing representation of the
            reads is `BCS-01`-`04`'s source selection, resolved already by
            the time a config loads; this command's job starts only once
            selection has determined that basecalling is actually required.
    """
    from ..informatics.basecall_execution import run_dorado_basecall
    from ..informatics.basecall_generation import (
        publish_basecall_generation,
        resolve_current_basecall_generation,
    )
    from .helpers import basecall_input_artifact_ids, stage_config_hash

    input_type = str(getattr(cfg, "input_type", "") or "").lower()
    if input_type not in {"pod5", "fast5"}:
        raise BasecallInputError(
            f"basecall requires POD5 or FAST5 signal input; this config resolved to "
            f"{input_type or 'no'} input. If reads for the configured model already "
            "exist, source selection should have used them already (BCS-01-04); if "
            "you want a *different* basecall of an already-ingested experiment, that "
            "is a re-basecalling lineage -- see `smftools experiment rebasecall`."
        )
    if cfg.sequencer != "ont":
        raise BasecallInputError(
            f"basecall currently supports only ont sequencers, not {cfg.sequencer!r}."
        )

    output_directory = Path(cfg.output_directory)
    config_hash = stage_config_hash(cfg)

    existing = resolve_current_basecall_generation(output_directory / BASECALL_DIR)
    if existing is not None:
        _, manifest = existing
        if str(manifest.get("config_hash", "")) == config_hash:
            logger.info(
                "basecall generation %s already matches this config; nothing to do.",
                manifest.get("generation_id"),
            )
            return {"generation_id": manifest.get("generation_id"), "reused_generation": True}

    def before_run() -> None:
        from ..memory_guard import require_memory_headroom

        require_memory_headroom(
            cfg,
            operation_label=(
                "dorado modified basecalling"
                if cfg.smf_modality == "direct"
                else "dorado canonical basecalling"
            ),
            estimator="external_basecalling_peak",
        )

    execution = run_dorado_basecall(
        input_data_path=cfg.input_data_path,
        output_directory=output_directory,
        workspace_directory=output_directory / BASECALL_DIR,
        model=cfg.model,
        model_dir=cfg.model_dir,
        modality=str(cfg.smf_modality),
        barcode_kit=cfg.barcode_kit,
        barcode_both_ends=cfg.barcode_both_ends,
        device=cfg.device,
        emit_moves=cfg.emit_moves,
        trim=cfg.trim,
        mod_list=cfg.mod_list if cfg.smf_modality == "direct" else None,
        bam_suffix=cfg.bam_suffix,
        max_basecall_reads=getattr(cfg, "max_basecall_reads", None),
        force_redo=bool(getattr(cfg, "force_redo_load_adata", False)),
        before_run=before_run,
    )

    outputs = publish_basecall_generation(
        output_directory,
        bam_path=execution.bam_path,
        model=str(cfg.model),
        modality=str(cfg.smf_modality),
        config_hash=config_hash,
        input_artifact_ids=basecall_input_artifact_ids(cfg),
        dorado_version=execution.dorado_version,
        bam_suffix=cfg.bam_suffix,
        extra_manifest_fields={
            "barcode_kit": cfg.barcode_kit,
            "reused_intermediate": execution.reused,
        },
    )
    logger.info(
        "Published basecall generation %s (model=%s, reused_intermediate=%s)",
        outputs["generation_id"],
        cfg.model,
        execution.reused,
    )
    return {**outputs, "reused_generation": False}


@stage_logging_lifecycle
def basecall_stage(cfg) -> dict[str, Any]:
    """Run basecall as `full`'s stage ahead of raw (`BCS-06`).

    Skips cleanly -- not an error -- when ``cfg.input_type`` is not POD5/FAST5
    signal, so a FASTQ/BAM-input experiment's `full` run stays byte-identical
    to today and this stage reports ``skipped`` in `full_summary.json`.
    Standalone `smftools basecall` keeps its hard `BasecallInputError` for the
    same input: a user invoking it directly on non-signal input has likely
    made a mistake worth surfacing loudly, whereas most `full` runs simply
    have nothing for this stage to do.
    """
    from ..logging_utils import setup_stage_logging

    output_directory = Path(cfg.output_directory)
    setup_stage_logging(cfg, output_directory / BASECALL_DIR)
    try:
        result = basecall_core(cfg)
    except BasecallInputError as exc:
        mark_stage_outcome("skipped", reason=str(exc))
        return {"reused_generation": True, "generation_id": None, "skipped": True}
    if result.get("reused_generation"):
        mark_stage_outcome("skipped", reason="basecall generation already current")
    return result
