## split_and_index_BAM
from .. import readwrite
import os
import subprocess
import glob
from .separate_bam_by_bc import separate_bam_by_bc

def split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix):
    """
    A wrapper function for splitting BAMS and indexing them
    """
    os.chdir(split_dir)
    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    file_prefix = readwrite.datestring()
    separate_bam_by_bc(aligned_sorted_output, file_prefix)
    # Make a BAM index file for the BAMs in that directory
    bam_pattern = '*' + bam_suffix
    bam_files = glob.glob(os.path.join(split_dir, bam_pattern))
    for input_file in bam_files:
        subprocess.run(["samtools", "index", input_file])
        print(f"Indexed {input_file}")