from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

from ..config import LoadExperimentConfig
from ..informatics.basecalling import canoncall, modcall
from ..readwrite import make_dirs

logger = get_logger(__name__)

p5 = require("pod5", extra="ont", purpose="POD5 IO")


def basecall_pod5s(config_path: str | Path) -> None:
    """Basecall POD5 inputs using a configuration file.

    Args:
        config_path: Path to the basecall configuration file.
    """
    # Default params
    bam_suffix = ".bam"  # If different, change from here.

    # Load experiment config parameters into global variables
    experiment_config = LoadExperimentConfig(config_path)
    var_dict = experiment_config.var_dict

    # These below variables will point to default_value if they are empty in the experiment_config.csv or if the variable is fully omitted from the csv.
    default_value = None

    # General config variable init
    input_data_path = Path(
        var_dict.get("input_data_path", default_value)
    )  # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = Path(
        var_dict.get("output_directory", default_value)
    )  # Path to the output directory to make for the analysis. Necessary.
    model = var_dict.get("model", default_value)  # needed for dorado basecaller
    model_dir = Path(var_dict.get("model_dir", default_value))  # model directory
    barcode_kit = var_dict.get("barcode_kit", default_value)  # needed for dorado basecaller
    barcode_both_ends = var_dict.get("barcode_both_ends", default_value)  # dorado demultiplexing
    trim = var_dict.get("trim", default_value)  # dorado adapter and barcode removal
    device = var_dict.get("device", "auto")

    # Modified basecalling specific variable init
    filter_threshold = var_dict.get("filter_threshold", default_value)
    m6A_threshold = var_dict.get("m6A_threshold", default_value)
    m5C_threshold = var_dict.get("m5C_threshold", default_value)
    hm5C_threshold = var_dict.get("hm5C_threshold", default_value)
    thresholds = [filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold]
    mod_list = var_dict.get("mod_list", default_value)

    # Make initial output directory
    make_dirs([output_directory])

    # Get the input filetype
    if input_data_path.is_file():
        input_data_filetype = input_data_path.suffixes[0]
        input_is_pod5 = input_data_filetype in [".pod5", ".p5"]
        input_is_fast5 = input_data_filetype in [".fast5", ".f5"]

    elif input_data_path.is_dir():
        # Get the file names in the input data dir
        input_files = input_data_path.iterdir()
        input_is_pod5 = sum([True for file in input_files if ".pod5" in file or ".p5" in file])
        input_is_fast5 = sum([True for file in input_files if ".fast5" in file or ".f5" in file])

    # If the input files are not pod5 files, and they are fast5 files, convert the files to a pod5 file before proceeding.
    if input_is_fast5 and not input_is_pod5:
        # take the input directory of fast5 files and write out a single pod5 file into the output directory.
        output_pod5 = output_directory / "FAST5s_to_POD5.pod5"
        logger.info(
            f"Input directory contains fast5 files, converting them and concatenating into a single pod5 file in the {output_pod5}"
        )
        fast5_to_pod5(input_data_path, output_pod5)
        # Reassign the pod5_dir variable to point to the new pod5 file.
        input_data_path = output_pod5

    model_basename = model.name
    model_basename = model_basename.replace(".", "_")

    if mod_list:
        mod_string = "_".join(mod_list)
        bam = output_directory / f"{model_basename}_{mod_string}_calls"
        modcall(
            model,
            input_data_path,
            barcode_kit,
            mod_list,
            bam,
            bam_suffix,
            barcode_both_ends,
            trim,
            device,
        )
    else:
        bam = output_directory / f"{model_basename}_canonical_basecalls"
        canoncall(
            model, input_data_path, barcode_kit, bam, bam_suffix, barcode_both_ends, trim, device
        )


def fast5_to_pod5(
    fast5_dir: str | Path | Iterable[str | Path],
    output_pod5: str | Path = "FAST5s_to_POD5.pod5",
) -> None:
    """Convert FAST5 inputs into a single POD5 file.

    Args:
        fast5_dir: FAST5 file path, directory, or iterable of file paths to convert.
        output_pod5: Output POD5 file path.

    Raises:
        FileNotFoundError: If no FAST5 files are found or the input path is invalid.
    """

    output_pod5 = str(output_pod5)  # ensure string

    # 1) If user gives a list of FAST5 files
    if isinstance(fast5_dir, (list, tuple)):
        fast5_paths = [str(Path(f)) for f in fast5_dir]
        cmd = ["pod5", "convert", "fast5", *fast5_paths, "--output", output_pod5]
        subprocess.run(cmd, check=True)
        return

    # Ensure Path object
    p = Path(fast5_dir)

    # 2) If user gives a single file
    if p.is_file():
        cmd = ["pod5", "convert", "fast5", str(p), "--output", output_pod5]
        subprocess.run(cmd, check=True)
        return

    # 3) If user gives a directory → collect FAST5s
    if p.is_dir():
        fast5_paths = sorted(str(f) for f in p.glob("*.fast5"))
        if not fast5_paths:
            raise FileNotFoundError(f"No FAST5 files found in {p}")

        cmd = ["pod5", "convert", "fast5", *fast5_paths, "--output", output_pod5]
        subprocess.run(cmd, check=True)
        return

    raise FileNotFoundError(f"Input path invalid: {fast5_dir}")


def subsample_pod5(
    pod5_path: str | Path,
    read_name_path: str | int,
    output_directory: str | Path,
) -> None:
    """Write a subsampled POD5 containing selected reads.

    Args:
        pod5_path: POD5 file path or directory of POD5 files to subsample.
        read_name_path: Path to a text file of read names (one per line) or an integer
            specifying a random subset size.
        output_directory: Directory to write the subsampled POD5 file.
    """

    if os.path.isdir(pod5_path):
        pod5_path_is_dir = True
        input_pod5_base = "input_pod5s.pod5"
        files = os.listdir(pod5_path)
        pod5_files = [os.path.join(pod5_path, file) for file in files if ".pod5" in file]
        pod5_files.sort()
        logger.info(f"Found input pod5s: {pod5_files}")

    elif os.path.exists(pod5_path):
        pod5_path_is_dir = False
        input_pod5_base = os.path.basename(pod5_path)

    else:
        logger.error("pod5_path passed does not exist")
        return None

    if type(read_name_path) is str:
        input_read_name_base = os.path.basename(read_name_path)
        output_base = (
            input_pod5_base.split(".pod5")[0]
            + "_"
            + input_read_name_base.split(".txt")[0]
            + "_subsampled.pod5"
        )

        # extract read names into a list of strings
        with open(read_name_path, "r") as file:
            read_names = [line.strip() for line in file]

        logger.info(f"Looking for read_ids: {read_names}")
        read_records = []

        if pod5_path_is_dir:
            for input_pod5 in pod5_files:
                with p5.Reader(input_pod5) as reader:
                    try:
                        for read_record in reader.reads(selection=read_names, missing_ok=True):
                            read_records.append(read_record.to_read())
                            logger.info(f"Found read in {input_pod5}: {read_record.read_id}")
                    except Exception:
                        logger.warning("Skipping pod5, could not find reads")
        else:
            with p5.Reader(pod5_path) as reader:
                try:
                    for read_record in reader.reads(selection=read_names):
                        read_records.append(read_record.to_read())
                        logger.info(f"Found read in {input_pod5}: {read_record}")
                except Exception:
                    logger.warning("Could not find reads")

    elif type(read_name_path) is int:
        import random

        output_base = (
            input_pod5_base.split(".pod5")[0] + f"_{read_name_path}_randomly_subsampled.pod5"
        )
        all_read_records = []

        if pod5_path_is_dir:
            # Shuffle the list of input pod5 paths
            random.shuffle(pod5_files)
            for input_pod5 in pod5_files:
                # iterate over the input pod5s
                logger.info(f"Opening pod5 file {input_pod5}")
                with p5.Reader(pod5_path) as reader:
                    for read_record in reader.reads():
                        all_read_records.append(read_record.to_read())
                # When enough reads are in all_read_records, stop accumulating reads.
                if len(all_read_records) >= read_name_path:
                    break

            if read_name_path <= len(all_read_records):
                read_records = random.sample(all_read_records, read_name_path)
            else:
                logger.info(
                    "Trying to sample more reads than are contained in the input pod5s, taking all reads"
                )
                read_records = all_read_records

        else:
            with p5.Reader(pod5_path) as reader:
                for read_record in reader.reads():
                    # get all read records from the input pod5
                    all_read_records.append(read_record.to_read())
            if read_name_path <= len(all_read_records):
                # if the subsampling amount is less than the record amount in the file, randomly subsample the reads
                read_records = random.sample(all_read_records, read_name_path)
            else:
                logger.info(
                    "Trying to sample more reads than are contained in the input pod5s, taking all reads"
                )
                read_records = all_read_records

    output_pod5 = os.path.join(output_directory, output_base)

    # Write the subsampled POD5
    with p5.Writer(output_pod5) as writer:
        writer.add_reads(read_records)
