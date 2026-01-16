from __future__ import annotations

## split_and_index_BAM

def split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix):
    """
    A wrapper function for splitting BAMS and indexing them.
    Parameters:
        aligned_sorted_BAM (str): A string representing the file path of the aligned_sorted BAM file.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        bam_suffix (str): A suffix to add to the bam file.
    
    Returns:
        None
            Splits an input BAM file on barcode value and makes a BAM index file.
    """
    from ...readwrite import date_string, make_dirs
    from pathlib import Path
    import os
    import pysam
    import glob
    from .separate_bam_by_bc import separate_bam_by_bc

    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    file_prefix = date_string()
    separate_bam_by_bc(aligned_sorted_output, file_prefix, bam_suffix, split_dir)
    # Make a BAM index file for the BAMs in that directory
    bam_pattern = '*' + bam_suffix
    bam_files = glob.glob(split_dir / bam_pattern)
    bam_files = [str(bam) for bam in bam_files if '.bai' not in str(bam)]
    for input_file in bam_files:
        pysam.index(input_file)

    return bam_files