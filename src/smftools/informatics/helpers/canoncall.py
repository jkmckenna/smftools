## canoncall

# Conversion SMF specific
def canoncall(model, pod5_dir, barcode_kit, bam, bam_suffix):
    """
    Wrapper function for dorado canonical base calling.

    Parameters:
        model (str): a string representing the file path to the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        barcode_kit (str): A string reppresenting the barcoding kit used in the experiment.
        bam (str): File path to the BAM file to output.
        bam_suffix (str): The suffix to use for the BAM file.
    
    Returns:
        None
            Outputs a BAM file holding the canonical base calls output by the dorado basecaller.
    """
    import subprocess
    output = bam + bam_suffix
    command = ["dorado", "basecaller", model, pod5_dir, "--kit-name", barcode_kit, "-Y"]
    with open(output, "w") as outfile:
        subprocess.run(command, stdout=outfile)