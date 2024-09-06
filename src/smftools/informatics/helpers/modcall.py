## modcall

# Direct methylation specific
def modcall(model, pod5_dir, barcode_kit, mod_list, bam, bam_suffix):
    """
    Wrapper function for dorado modified base calling.

    Parameters:
        model (str): a string representing the file path to the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        barcode_kit (str): A string representing the barcoding kit used in the experiment.
        mod_list (list): A list of modification types to use in the analysis.
        bam (str): File path to the BAM file to output.
        bam_suffix (str): The suffix to use for the BAM file.
    
    Returns:
        None
            Outputs a BAM file holding the modified base calls output by the dorado basecaller.
    """
    import subprocess
    output = bam + bam_suffix
    command = [
    "dorado", "basecaller", model, pod5_dir, "--kit-name", barcode_kit, "-Y",
    "--modified-bases", ",".join(mod_list)]  # Join MOD_LIST elements with commas
    with open(output, "w") as outfile:
        subprocess.run(command, stdout=outfile)