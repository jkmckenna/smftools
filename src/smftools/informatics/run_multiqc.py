from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def run_multiqc(input_dir: str | Path, output_dir: str | Path) -> None:
    """Run MultiQC on a directory and save the report to the output directory.

    Args:
        input_dir: Path to the directory containing QC reports (e.g., FastQC, Samtools outputs).
        output_dir: Path to the directory where MultiQC reports should be saved.
    """
    import subprocess

    from ..readwrite import make_dirs

    # Ensure the output directory exists
    make_dirs(output_dir)

    input_dir = str(input_dir)
    output_dir = str(output_dir)

    # Construct MultiQC command
    command = ["multiqc", input_dir, "-o", output_dir]

    logger.info(f"Running MultiQC on '{input_dir}' and saving results to '{output_dir}'...")

    # Run MultiQC. This step is best-effort and should not fail the load workflow.
    try:
        cp = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.stdout:
            logger.debug("MultiQC stdout:\n%s", cp.stdout.strip())
        logger.info(f"MultiQC report generated successfully in: {output_dir}")
    except FileNotFoundError:
        logger.warning("MultiQC executable not found; skipping MultiQC report generation.")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        if stderr:
            logger.warning(
                "MultiQC failed (non-fatal, skipping report). Exit code %s. Stderr tail:\n%s",
                e.returncode,
                "\n".join(stderr.splitlines()[-20:]),
            )
        elif stdout:
            logger.warning(
                "MultiQC failed (non-fatal, skipping report). Exit code %s. Stdout tail:\n%s",
                e.returncode,
                "\n".join(stdout.splitlines()[-20:]),
            )
        else:
            logger.warning(
                "MultiQC failed (non-fatal, skipping report). Exit code %s.",
                e.returncode,
            )
