## extract_mods
import os
import subprocess
import glob
import zipfile

def extract_mods(thresholds, mod_tsv_dir, split_dir, bam_suffix):
    """
    Takes all of the aligned, sorted, split modified BAM files and runs Nanopore Modkit Extract to load the modification data into zipped TSV files
    """
    os.chdir(mod_tsv_dir)
    filter_threshold, m6A_threshold, m5C_threshold, hm5C_threshold = thresholds
    bam_files = glob.glob(os.path.join(split_dir, f"*{bam_suffix}"))
    for input_file in bam_files:
        print(input_file)
        # Extract the file basename
        file_name = os.path.basename(input_file)
        # Construct the output TSV file path
        output_tsv_temp = os.path.join(mod_tsv_dir, file_name)
        output_tsv = output_tsv_temp.replace(bam_suffix, "") + "_extract.tsv"
        # Run modkit summary
        subprocess.run(["modkit", "summary", input_file])
        # Run modkit extract
        subprocess.run([
            "modkit", "extract",
            "--filter-threshold", filter_threshold,
            "--mod-thresholds", f"m:{m5C_threshold}",
            "--mod-thresholds", f"a:{m6A_threshold}",
            "--mod-thresholds", f"h:{hm5C_threshold}",
            input_file, "null",
            "--read-calls", output_tsv
        ])
        # Zip the output TSV
        print(f'zipping {output_tsv}')
        with zipfile.ZipFile(f"{output_tsv}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_tsv, os.path.basename(output_tsv))
        # Remove the non-zipped TSV
        print(f'removing {output_tsv}')
        os.remove(output_tsv)