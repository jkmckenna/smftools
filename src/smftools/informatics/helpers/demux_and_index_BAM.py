## demux_and_index_BAM
from concurrent.futures import ProcessPoolExecutor


def process_bam(input_file, plotting_dir, bed_dir, fasta, make_bigwigs):
    """Helper function to process a BAM file in parallel."""
    from .aligned_BAM_to_bed import aligned_BAM_to_bed
    from .extract_readnames_from_BAM import extract_readnames_from_BAM

    aligned_BAM_to_bed(input_file, plotting_dir, bed_dir, fasta, make_bigwigs)
    extract_readnames_from_BAM(input_file)

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
    from .. import readwrite
    import os
    import subprocess
    import glob
    from .make_dirs import make_dirs
    
    input_bam = aligned_sorted_BAM + bam_suffix
    command = ["dorado", "demux", "--kit-name", barcode_kit]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    if threads:
        command += ["-t", str(threads)]
    else:
        pass
    command += ["--emit-summary", "--sort-bam", "--output-dir", split_dir]
    command.append(input_bam)
    command_string = ' '.join(command)
    print(f"Running: {command_string}")
    subprocess.run(command)

    # Make a BAM index file for the BAMs in that directory
    bam_pattern = '*' + bam_suffix
    bam_files = glob.glob(os.path.join(split_dir, bam_pattern))
    bam_files = [bam for bam in bam_files if '.bai' not in bam and 'unclassified' not in bam]
    bam_files.sort()

    if not bam_files:
        raise FileNotFoundError(f"No BAM files found in {split_dir} with suffix {bam_suffix}")

    # Create required directories
    plotting_dir = os.path.join(split_dir, 'demultiplexed_bed_histograms')
    bed_dir = os.path.join(split_dir, 'demultiplexed_read_alignment_coordinates')
    make_dirs([plotting_dir, bed_dir])

    # Process BAM files in parallel
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_bam, bam, plotting_dir, bed_dir, fasta, make_bigwigs) for bam in bam_files]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    return bam_files