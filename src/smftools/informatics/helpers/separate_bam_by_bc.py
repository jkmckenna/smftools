## separate_bam_by_bc

# General
def separate_bam_by_bc(input_bam, output_prefix, bam_suffix, split_dir):
    """
    Separates an input BAM file on the BC SAM tag values.

    Parameters:
        input_bam (str): File path to the BAM file to split.
        output_prefix (str): A prefix to append to the output BAM.
        bam_suffix (str): A suffix to add to the bam file.
        split_dir (str): String indicating path to directory to split BAMs into
    
    Returns:
        None
            Writes out split BAM files.
    """
    import pysam
    import os

    bam_base = os.path.basename(input_bam)
    bam_base_minus_suffix = bam_base.split(bam_suffix)[0]

    # Open the input BAM file for reading
    with pysam.AlignmentFile(input_bam, "rb") as bam:
        # Create a dictionary to store output BAM files
        output_files = {}
        # Iterate over each read in the BAM file
        for read in bam:
            try:
                # Get the barcode tag value
                bc_tag = read.get_tag("BC", with_value_type=True)[0].split('barcode')[1]
                # Open the output BAM file corresponding to the barcode
                if bc_tag not in output_files:
                    output_path = os.path.join(split_dir, f"{output_prefix}_{bam_base_minus_suffix}_{bc_tag}{bam_suffix}")
                    output_files[bc_tag] = pysam.AlignmentFile(output_path, "wb", header=bam.header)
                # Write the read to the corresponding output BAM file
                output_files[bc_tag].write(read)
            except KeyError:
                 print(f"BC tag not present for read: {read.query_name}")
    # Close all output BAM files
    for output_file in output_files.values():
        output_file.close()