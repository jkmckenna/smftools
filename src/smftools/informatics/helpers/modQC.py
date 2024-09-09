## modQC

# Direct SMF
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
    subprocess.run(["modkit", "sample-probs", aligned_sorted_output])
    command = [
        "modkit", "summary", aligned_sorted_output,
        "--filter-threshold", f"{filter_threshold}",
        "--mod-thresholds", f"m:{m5C_threshold}",
        "--mod-thresholds", f"a:{m6A_threshold}",
        "--mod-thresholds", f"h:{hm5C_threshold}"
    ]
    subprocess.run(command)