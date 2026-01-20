from __future__ import annotations

## demux_and_index_BAM

def demux_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix, barcode_kit, barcode_both_ends, trim, fasta, make_bigwigs, threads):
    """
    A wrapper function for splitting BAMS and indexing them.
    Parameters:
        aligned_sorted_BAM (str): A string representing the file path of the aligned_sorted BAM file.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        bam_suffix (str): A suffix to add to the bam file.
        barcode_kit (str): Name of barcoding kit.
        barcode_both_ends (bool): Whether to require both ends to be barcoded.
        trim (bool): Whether to trim off barcodes after demultiplexing.
        fasta (str): File path to the reference genome to align to.
        make_bigwigs (bool): Whether to make bigwigs
        threads (int): Number of threads to use.
    
    Returns:
        bam_files (list): List of split BAM file path strings
            Splits an input BAM file on barcode value and makes a BAM index file.
    """
    from ...readwrite import make_dirs
    import os
    import subprocess
    import glob
    
    input_bam = aligned_sorted_BAM.with_suffix(bam_suffix)
    command = ["dorado", "demux", "--kit-name", barcode_kit]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    if threads:
        command += ["-t", str(threads)]
    else:
        pass
    command += ["--emit-summary", "--sort-bam", "--output-dir", str(split_dir)]
    command.append(str(input_bam))
    command_string = ' '.join(command)
    print(f"Running: {command_string}")
    subprocess.run(command)

    bam_files = sorted(
        p for p in split_dir.glob(f"*{bam_suffix}")
        if p.is_file() and p.suffix == bam_suffix and "unclassified" not in p.name
    )

    if not bam_files:
        raise FileNotFoundError(f"No BAM files found in {split_dir} with suffix {bam_suffix}")
    
    return bam_files