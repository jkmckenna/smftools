"""Run dorado, with intermediate reuse: the basecall stage's execution step (`BCS-05`).

Mirrors the orchestration `cli/load_adata.py` uses for `raw`'s own inline
POD5 basecalling -- subsample, build an `IntermediateSpec`, dispatch
canonical vs. modified basecalling, commit the result -- so a standalone
`smftools basecall` run and `raw`'s inline call reuse the *same* cached
intermediate (`prepare_intermediate` keys reuse on operation + input
checksum + config, not on which caller asked for it -- both write under
``<output_directory>/raw_outputs/intermediates/dorado-basecalling/``,
`IntermediateSpec`'s own fixed location, regardless of which stage is
asking). Basecalling a run once, however it was invoked, is never redone.

Publishing the result as an immutable `basecall_outputs/` generation is
`informatics.basecall_generation`, not this module. Unifying this function's
call site with `raw`'s own inline one -- so `raw` also produces a real
generation instead of only an intermediate -- is `BCS-06`'s job, deferred
deliberately rather than refactored into an already-large, currently
untested-at-this-path production function as a side effect of this change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .basecalling import canoncall, modcall
from .raw_intermediate_manifest import (
    IntermediateSpec,
    artifact_checksum,
    commit_intermediate,
    committed_output,
    executable_version,
    prepare_intermediate,
)


@dataclass(frozen=True)
class BasecallExecutionResult:
    """What running (or reusing) dorado produced."""

    bam_path: Path
    dorado_version: Optional[str]
    reused: bool


def run_dorado_basecall(
    *,
    input_data_path: str | Path,
    output_directory: str | Path,
    workspace_directory: str | Path,
    model: str,
    model_dir: str | Path,
    modality: str,
    barcode_kit: Optional[str],
    barcode_both_ends: bool,
    device: str,
    emit_moves: bool,
    trim: bool,
    mod_list: Optional[Sequence[str]] = None,
    bam_suffix: str = ".bam",
    max_basecall_reads: Optional[int] = None,
    force_redo: bool = False,
    before_run: Optional[Callable[[], None]] = None,
) -> BasecallExecutionResult:
    """Basecall `input_data_path` with dorado, reusing a matching prior run.

    Args:
        input_data_path: POD5 (or FAST5) input to basecall.
        output_directory: The run's output directory -- `prepare_intermediate`
            stores and looks up the reusable intermediate under
            ``<output_directory>/raw_outputs/intermediates/``, the same fixed
            location regardless of which caller asks.
        workspace_directory: Scratch directory for an optional subsampled
            POD5 copy (`max_basecall_reads`).
        model, model_dir, modality, barcode_kit, barcode_both_ends, device,
        emit_moves, trim, mod_list, bam_suffix: The same basecalling
            parameters `ExperimentConfig` already carries.
        max_basecall_reads: Deliberately basecall only a random subsample,
            when set.
        force_redo: Skip the reuse check and basecall unconditionally.
        before_run: Called immediately before dorado actually runs -- never on
            the reuse path. Callers that need a memory-headroom preflight
            (which needs a resolved config this module deliberately does not
            depend on) supply it as a closure.

    Returns:
        BasecallExecutionResult: The BAM path, the installed Dorado version,
        and whether a prior compatible run was reused rather than rerun.
    """
    input_data_path = Path(input_data_path)
    workspace_directory = Path(workspace_directory)

    if max_basecall_reads is not None:
        from .pod5_functions import subsample_pod5_for_basecalling

        input_data_path = subsample_pod5_for_basecalling(
            input_data_path, max_basecall_reads, workspace_directory
        )

    dorado_kit_name = barcode_kit if barcode_kit != "custom" else None
    dorado_version = executable_version("dorado")
    basecall_spec = IntermediateSpec(
        operation="dorado-basecalling",
        input_artifacts=(("pod5-input", artifact_checksum(input_data_path)),),
        operation_config={
            "barcode_both_ends": bool(barcode_both_ends),
            "barcode_kit": dorado_kit_name,
            "device": str(device),
            "emit_moves": bool(emit_moves),
            "model": str(model),
            "modifications": list(mod_list or []) if modality == "direct" else [],
            "modality": str(modality),
            "trim": bool(trim),
        },
        tool_versions={"dorado": dorado_version},
    )
    basecall_workspace = prepare_intermediate(
        output_directory, basecall_spec, force_redo=force_redo
    )

    if basecall_workspace.reusable:
        unaligned_output = committed_output(basecall_workspace, "bam")
        if unaligned_output is None:
            raise RuntimeError("Validated basecalling commit has no BAM output.")
        return BasecallExecutionResult(
            bam_path=unaligned_output, dorado_version=dorado_version, reused=True
        )

    if before_run is not None:
        before_run()

    bam = basecall_workspace.root / "basecalls"
    unaligned_output = bam.with_suffix(bam_suffix)
    if modality != "direct":
        canoncall(
            str(model_dir),
            model,
            str(input_data_path),
            dorado_kit_name,
            str(bam),
            bam_suffix,
            barcode_both_ends,
            trim,
            device,
            emit_moves,
        )
    else:
        modcall(
            str(model_dir),
            model,
            str(input_data_path),
            dorado_kit_name,
            list(mod_list or []),
            str(bam),
            bam_suffix,
            barcode_both_ends,
            trim,
            device,
            emit_moves,
        )
    commit_intermediate(basecall_workspace, {"bam": unaligned_output})
    return BasecallExecutionResult(
        bam_path=unaligned_output, dorado_version=dorado_version, reused=False
    )
