## canoncall
import subprocess

# Conversion SMF specific
def canoncall(model, pod5_dir, barcode_kit, bam, bam_suffix):
    """
    Wrapper function for dorado canonical base calling.
    """
    output = bam + bam_suffix
    command = ["dorado", "basecaller", model, pod5_dir, "--kit-name", barcode_kit, "-Y"]
    with open(output, "w") as outfile:
        subprocess.run(command, stdout=outfile)