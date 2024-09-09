## make_modbed

# Direct SMF
def make_modbed(aligned_sorted_output, thresholds, mod_bed_dir):
    """
    Generating position methylation summaries for each barcoded sample starting from the overall BAM file that was direct output of dorado aligner.
    Parameters:
        aligned_sorted_output (str): A string representing the file path to the aligned_sorted non-split BAM file.
    
    Returns:
        None
    """
    import os
    import subprocess
    
    os.chdir(mod_bed_dir)
    filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold = thresholds
    command = [
        "modkit", "pileup", aligned_sorted_output, mod_bed_dir,
        "--partition-tag", "BC",
        "--only-tabs",
        "--filter-threshold", f'{filter_threshold}',
        "--mod-thresholds", f"m:{m5C_threshold}",
        "--mod-thresholds", f"a:{m6A_threshold}",
        "--mod-thresholds", f"h:{hm5C_threshold}"
    ]
    subprocess.run(command)