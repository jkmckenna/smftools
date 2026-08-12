from __future__ import annotations

import subprocess
from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def _run_dorado_basecaller(command: list[str], output: str | Path) -> None:
    """Run one dorado basecalling command, streaming stdout into a BAM.

    Dorado writes the BAM to stdout and diagnostics to stderr, and it exits
    non-zero without writing anything when it cannot resolve a model or read
    its inputs. stderr is inherited so long-running progress output stays
    visible and unbuffered rather than accumulating in memory.

    Args:
        command: The fully resolved dorado argument vector.
        output: Path of the BAM file to receive dorado's stdout.

    Raises:
        RuntimeError: If dorado exits non-zero, or exits zero without writing
            any output. Either case leaves no usable BAM, so the empty file is
            removed to keep it from being mistaken for a real basecall.
    """
    output = Path(output)
    logger.info("Running dorado basecalling: %s", " ".join(command))
    logger.info("Writing dorado basecalls to %s", output)
    with output.open("wb") as handle:
        completed = subprocess.run(command, stdout=handle)
    if completed.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"dorado basecalling failed (exit {completed.returncode}). "
            "See the dorado output above for the reported cause."
        )
    if output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "dorado basecalling reported success but produced an empty BAM. "
            f"Verify that {command[-1]} contains readable signal data."
        )


def canoncall(
    model_dir,
    model,
    pod5_dir,
    barcode_kit,
    bam,
    bam_suffix,
    barcode_both_ends=False,
    trim=False,
    device="auto",
    emit_moves=False,
):
    """
    Wrapper function for dorado canonical base calling.

    Parameters:
        model_dir (str): a string representing the file path to the dorado basecalling model directory.
        model (str): a string representing the the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        barcode_kit (str): A string reppresenting the barcoding kit used in the experiment. Needed for demultiplexing
        bam (str): File path to the BAM file to output.
        bam_suffix (str): The suffix to use for the BAM file.
        barcode_both_ends (bool): Whether to require a barcode detection on both ends for demultiplexing.
        trim (bool): Whether to trim barcodes, adapters, and primers from read ends.
        device (str): The device to use. 'auto' is default, which can detect device to use. Can also specify metal, cpu, cuda.
        emit_moves (bool): Whether to emit move tables (mv tag) for signal-to-base alignment.

    Returns:
        None
            Outputs a BAM file holding the canonical base calls output by the dorado basecaller.

    Raises:
        RuntimeError: If dorado fails or produces no basecalls.
    """
    output = bam + bam_suffix
    command = [
        "dorado",
        "basecaller",
        "--models-directory",
        model_dir,
        "--device",
        device,
        "--batchsize",
        "0",
    ]
    if barcode_kit:
        command += ["--kit-name", barcode_kit]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    if emit_moves:
        command.append("--emit-moves")
    command += [model, pod5_dir]
    _run_dorado_basecaller(command, output)


def modcall(
    model_dir,
    model,
    pod5_dir,
    barcode_kit,
    mod_list,
    bam,
    bam_suffix,
    barcode_both_ends=False,
    trim=False,
    device="auto",
    emit_moves=False,
):
    """
    Wrapper function for dorado modified base calling.

    Parameters:
        model_dir (str): a string representing the file path to the dorado basecalling model directory.
        model (str): a string representing the the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        barcode_kit (str): A string representing the barcoding kit used in the experiment.
        mod_list (list): A list of modification types to use in the analysis.
        bam (str): File path to the BAM file to output.
        bam_suffix (str): The suffix to use for the BAM file.
        barcode_both_ends (bool): Whether to require a barcode detection on both ends for demultiplexing.
        trim (bool): Whether to trim barcodes, adapters, and primers from read ends
        device (str): Device to use for basecalling. auto, metal, cpu, cuda.
        emit_moves (bool): Whether to emit move tables (mv tag) for signal-to-base alignment.

    Returns:
        None
            Outputs a BAM file holding the modified base calls output by the dorado basecaller.

    Raises:
        RuntimeError: If dorado fails or produces no basecalls.
    """
    output = bam + bam_suffix
    command = [
        "dorado",
        "basecaller",
        "--models-directory",
        model_dir,
        "--modified-bases",
    ]
    command += mod_list

    if barcode_kit:
        command += ["--kit-name", barcode_kit]
    command += ["--device", device, "--batchsize", "0"]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    if emit_moves:
        command.append("--emit-moves")
    command += [model, pod5_dir]
    _run_dorado_basecaller(command, output)
