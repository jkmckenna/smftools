## canoncall

# Conversion SMF specific
def canoncall(model_dir, model, pod5_dir, barcode_kit, bam, bam_suffix, barcode_both_ends=True, trim=False, device='auto'):
    """
    Wrapper function for dorado canonical base calling.

    Parameters:
        model_dir (str): a string representing the file path to the dorado basecalling model directory.
        model (str): a string representing the the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        barcode_kit (str): A string reppresenting the barcoding kit used in the experiment.
        bam (str): File path to the BAM file to output.
        bam_suffix (str): The suffix to use for the BAM file.
        barcode_both_ends (bool): Whether to require a barcode detection on both ends for demultiplexing.
        trim (bool): Whether to trim barcodes, adapters, and primers from read ends.
        device (str): The device to use. 'auto' is default, which can detect device to use. Can also specify metal, cpu, cuda.
    
    Returns:
        None
            Outputs a BAM file holding the canonical base calls output by the dorado basecaller.
    """
    import subprocess
    output = bam + bam_suffix
    command = ["dorado", "basecaller", "--models-directory", model_dir, "--kit-name", barcode_kit, "--device", device, "--batchsize", "0"]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    command += [model, pod5_dir]
    command_string = " ".join(command)
    print(f"Running {command_string}\n to generate {output}")
    with open(output, "w") as outfile:
        subprocess.run(command, stdout=outfile)