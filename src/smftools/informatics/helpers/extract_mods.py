## extract_mods

def extract_mods(thresholds, mod_tsv_dir, split_dir, bam_suffix, skip_unclassified=True, modkit_summary=False, threads=None):
    """
    Takes all of the aligned, sorted, split modified BAM files and runs Nanopore Modkit Extract to load the modification data into zipped TSV files

    Parameters:
        thresholds (list): A list of thresholds to use for marking each basecalled base as passing or failing on canonical and modification call status.
        mod_tsv_dir (str): A string representing the file path to the directory to hold the modkit extract outputs.
        split_dit (str): A string representing the file path to the directory containing the converted aligned_sorted_split BAM files.
        bam_suffix (str): The suffix to use for the BAM file.
        skip_unclassified (bool): Whether to skip unclassified bam file for modkit extract command
        modkit_summary (bool): Whether to run and display modkit summary
        threads (int): Number of threads to use

    Returns:
        None
        Runs modkit extract on input aligned_sorted_split modified BAM files to output zipped TSVs containing modification calls.

    """
    import os
    import subprocess
    import glob
    import zipfile
    
    os.chdir(mod_tsv_dir)
    filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold = thresholds
    bam_files = glob.glob(os.path.join(split_dir, f"*{bam_suffix}"))

    if threads:
        threads = str(threads)
    else:
        pass

    for input_file in bam_files:
        print(input_file)
        # Extract the file basename
        file_name = os.path.basename(input_file)
        if skip_unclassified and "unclassified" in file_name:
            print("Skipping modkit extract on unclassified reads")
        else:
            # Construct the output TSV file path
            output_tsv_temp = os.path.join(mod_tsv_dir, file_name)
            output_tsv = output_tsv_temp.replace(bam_suffix, "") + "_extract.tsv"
            if os.path.exists(f"{output_tsv}.gz"):
                print(f"{output_tsv}.gz already exists, skipping modkit extract")
            else:
                print(f"Extracting modification data from {input_file}")
                if modkit_summary:
                    # Run modkit summary
                    subprocess.run(["modkit", "summary", input_file])
                else:
                    pass
                # Run modkit extract
                if threads:
                    extract_command = [
                        "modkit", "extract",
                        "calls", "--mapped-only",
                        "--filter-threshold", f'{filter_threshold}',
                        "--mod-thresholds", f"m:{m5C_threshold}",
                        "--mod-thresholds", f"a:{m6A_threshold}",
                        "--mod-thresholds", f"h:{hm5C_threshold}",
                        "-t", threads,
                        input_file, output_tsv
                        ]
                else:
                    extract_command = [
                        "modkit", "extract",
                        "calls", "--mapped-only",
                        "--filter-threshold", f'{filter_threshold}',
                        "--mod-thresholds", f"m:{m5C_threshold}",
                        "--mod-thresholds", f"a:{m6A_threshold}",
                        "--mod-thresholds", f"h:{hm5C_threshold}",
                        input_file, output_tsv
                        ]                    
                subprocess.run(extract_command)
                # Zip the output TSV
                print(f'zipping {output_tsv}')
                if threads:
                    zip_command = ["pigz", "-f", "-p", threads, output_tsv]
                else:
                    zip_command = ["pigz", "-f", output_tsv]
                subprocess.run(zip_command, check=True)