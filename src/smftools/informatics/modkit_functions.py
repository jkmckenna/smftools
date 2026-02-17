from __future__ import annotations

import subprocess

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def extract_mods(
    thresholds,
    mod_tsv_dir,
    split_dir,
    bam_suffix,
    skip_unclassified=True,
    modkit_summary=False,
    threads=None,
    single_bam=None,
):
    """
    Takes all of the aligned, sorted, split modified BAM files and runs Nanopore Modkit Extract to load the modification data into zipped TSV files

    Parameters:
        thresholds (list): A list of thresholds to use for marking each basecalled base as passing or failing on canonical and modification call status.
        mod_tsv_dir (str): A string representing the file path to the directory to hold the modkit extract outputs.
        split_dir (str): A string representing the file path to the directory containing the converted aligned_sorted_split BAM files.
        bam_suffix (str): The suffix to use for the BAM file.
        skip_unclassified (bool): Whether to skip unclassified bam file for modkit extract command
        modkit_summary (bool): Whether to run and display modkit summary
        threads (int): Number of threads to use
        single_bam (Path | None): When set, use this single BAM instead of iterating split_dir.

    Returns:
        None
        Runs modkit extract on input aligned_sorted_split modified BAM files to output zipped TSVs containing modification calls.

    """
    from pathlib import Path

    filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold = thresholds
    if single_bam is not None:
        bam_files = [Path(single_bam)]
    else:
        bam_files = sorted(
            p for p in split_dir.iterdir() if bam_suffix in p.name and ".bai" not in p.name
        )
        if skip_unclassified:
            bam_files = [p for p in bam_files if "unclassified" not in p.name]
    logger.info(f"Running modkit extract for the following bam files: {bam_files}")

    if threads:
        threads = str(threads)
    else:
        pass

    for input_file in bam_files:
        logger.debug(input_file)
        # Construct the output TSV file path
        output_tsv = mod_tsv_dir / (input_file.stem + "_extract.tsv")
        output_tsv_gz = output_tsv.parent / (output_tsv.name + ".gz")
        if output_tsv_gz.exists():
            logger.debug(f"{output_tsv_gz} already exists, skipping modkit extract")
        else:
            logger.info(f"Extracting modification data from {input_file}")
            if modkit_summary:
                # Run modkit summary
                subprocess.run(["modkit", "summary", str(input_file)])
            else:
                pass
            # Run modkit extract
            if threads:
                extract_command = [
                    "modkit",
                    "extract",
                    "calls",
                    "--mapped-only",
                    "--filter-threshold",
                    f"{filter_threshold}",
                    "--mod-thresholds",
                    f"m:{m5C_threshold}",
                    "--mod-thresholds",
                    f"a:{m6A_threshold}",
                    "--mod-thresholds",
                    f"h:{hm5C_threshold}",
                    "-t",
                    threads,
                    str(input_file),
                    str(output_tsv),
                ]
            else:
                extract_command = [
                    "modkit",
                    "extract",
                    "calls",
                    "--mapped-only",
                    "--filter-threshold",
                    f"{filter_threshold}",
                    "--mod-thresholds",
                    f"m:{m5C_threshold}",
                    "--mod-thresholds",
                    f"a:{m6A_threshold}",
                    "--mod-thresholds",
                    f"h:{hm5C_threshold}",
                    str(input_file),
                    str(output_tsv),
                ]
            subprocess.run(extract_command)
            # Zip the output TSV
            logger.info(f"zipping {output_tsv}")
            if threads:
                zip_command = ["pigz", "-f", "-p", threads, str(output_tsv)]
            else:
                zip_command = ["pigz", "-f", str(output_tsv)]
            subprocess.run(zip_command, check=True)
    return


def make_modbed(aligned_sorted_output, thresholds, mod_bed_dir):
    """
    Generating position methylation summaries for each barcoded sample starting from the overall BAM file that was direct output of dorado aligner.
    Parameters:
        aligned_sorted_output (str): A string representing the file path to the aligned_sorted non-split BAM file.

    Returns:
        None
    """
    import subprocess

    filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold = thresholds
    command = [
        "modkit",
        "pileup",
        str(aligned_sorted_output),
        str(mod_bed_dir),
        "--partition-tag",
        "BC",
        "--only-tabs",
        "--filter-threshold",
        f"{filter_threshold}",
        "--mod-thresholds",
        f"m:{m5C_threshold}",
        "--mod-thresholds",
        f"a:{m6A_threshold}",
        "--mod-thresholds",
        f"h:{hm5C_threshold}",
    ]
    subprocess.run(command)


def modQC(aligned_sorted_output, thresholds):
    """
    Output the percentile of bases falling at a call threshold (threshold is a probability between 0-1) for the overall BAM file.
    It is generally good to look at these parameters on positive and negative controls.

    Parameters:
        aligned_sorted_output (str): A string representing the file path of the aligned_sorted non-split BAM file output by the dorado aligned.
        thresholds (list): A list of floats to pass for call thresholds.

    Returns:
        None
    """
    import subprocess

    filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold = thresholds
    subprocess.run(["modkit", "sample-probs", str(aligned_sorted_output)])
    command = [
        "modkit",
        "summary",
        str(aligned_sorted_output),
        "--filter-threshold",
        f"{filter_threshold}",
        "--mod-thresholds",
        f"m:{m5C_threshold}",
        "--mod-thresholds",
        f"a:{m6A_threshold}",
        "--mod-thresholds",
        f"h:{hm5C_threshold}",
    ]
    subprocess.run(command)
