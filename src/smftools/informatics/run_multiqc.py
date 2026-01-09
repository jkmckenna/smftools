from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def run_multiqc(input_dir, output_dir):
    """
    Runs MultiQC on a given directory and saves the report to the specified output directory.

    Parameters:
    - input_dir (str): Path to the directory containing QC reports (e.g., FastQC, Samtools, bcftools outputs).
    - output_dir (str): Path to the directory where MultiQC reports should be saved.

    Returns:
    - None: The function executes MultiQC and prints the status.
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

    # Run MultiQC
    try:
        subprocess.run(command, check=True)
        logger.info(f"MultiQC report generated successfully in: {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running MultiQC: {e}")
