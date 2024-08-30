## modcall
import subprocess

# Direct methylation specific
def modcall(model, pod5_dir, barcode_kit, mod_list, bam, bam_suffix):
    """
    Wrapper function for dorado modified base calling.
    """
    output = bam + bam_suffix
    command = [
    "dorado", "basecaller", model, pod5_dir, "--kit-name", barcode_kit, "-Y",
    "--modified-bases", ",".join(mod_list)]  # Join MOD_LIST elements with commas
    with open(output, "w") as outfile:
        subprocess.run(command, stdout=outfile)