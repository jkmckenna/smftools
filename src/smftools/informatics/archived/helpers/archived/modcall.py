## modcall

# Direct methylation specific
def modcall(model_dir, model, pod5_dir, barcode_kit, mod_list, bam, bam_suffix, barcode_both_ends=True, trim=False, device='auto'):
    """
    Wrapper function for dorado modified base calling.

    Parameters:
        model_dir (str): a string representing the file path to the dorado basecalling model directory.
        model (str): a string representing the the dorado basecalling model.
        pod5_dir (str): a string representing the file path to the experiment directory containing the POD5 files.
        barcode_kit (str): A string representing the barcoding kit used in the experiment.
        mod_list (list): A list of modification types to use in the analysis.
        bam (str): File path to the BAM file to output.
        bam_suffix (str): The suffix to use for the BAM file.
        barcode_both_ends (bool): Whether to require a barcode detection on both ends for demultiplexing.
        trim (bool): Whether to trim barcodes, adapters, and primers from read ends
        device (str): Device to use for basecalling. auto, metal, cpu, cuda.
    
    Returns:
        None
            Outputs a BAM file holding the modified base calls output by the dorado basecaller.
    """
    import subprocess
    output = bam + bam_suffix
    command = ["dorado", "basecaller", "--models-directory", model_dir, "--kit-name", barcode_kit, "--modified-bases"]
    command += mod_list
    command += ["--device", device, "--batchsize", "0"]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    command += [model, pod5_dir]
    print(f'Running: {" ".join(command)}')
    with open(output, "w") as outfile:
        subprocess.run(command, stdout=outfile)