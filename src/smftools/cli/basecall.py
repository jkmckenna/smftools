"""`smftools basecall`: publish an immutable basecall generation from signal (`BCS-05`).

A top-level command, not nested under `experiment` or `project` -- per
`src/smftools/cli/AGENTS.md`, that choice is recorded here deliberately,
matching the plan's own reasoning: a config-free form (`BCS-10`, not yet
built) is scoped to neither an experiment nor a project.

Config-driven only for now. `run` reuses the experiment's own model
parameters (`model`, `model_dir`, `barcode_kit`, ...); the config-free
`--input/--output` invocation is `BCS-10`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smftools.constants import BASECALL_DIR
from smftools.logging_utils import get_logger, mark_stage_outcome, stage_logging_lifecycle

logger = get_logger(__name__)


class BasecallInputError(ValueError):
    """Raised when `input_data_path` is not signal, so there is nothing to basecall."""


def basecall(config_path: str) -> dict[str, Any]:
    """Load a config and publish (or reuse) its basecall generation."""
    from .helpers import load_experiment_config

    cfg = load_experiment_config(config_path)
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
    from .helpers import stage_config_hash

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

    from ..informatics.raw_intermediate_manifest import artifact_checksum

    outputs = publish_basecall_generation(
        output_directory,
        bam_path=execution.bam_path,
        model=str(cfg.model),
        modality=str(cfg.smf_modality),
        config_hash=config_hash,
        input_artifact_ids=[artifact_checksum(cfg.input_data_path)],
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
