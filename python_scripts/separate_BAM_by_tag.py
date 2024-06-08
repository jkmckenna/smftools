import pysam
import argparse

def separate_bam_by_bc(input_bam, output_prefix):
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
                    output_files[bc_tag] = pysam.AlignmentFile(f"{output_prefix}_{bc_tag}.bam", "wb", header=bam.header)
                # Write the read to the corresponding output BAM file
                output_files[bc_tag].write(read)
            except KeyError:
                 print(f"BC tag not present for read: {read.query_name}")
    # Close all output BAM files
    for output_file in output_files.values():
        output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse BAM file for modified bases.")
    parser.add_argument("input_bam", help="Path to the input BAM file to split by barcode.")
    parser.add_argument("output_prefix", help="Absolute file path string to append to the prefix of the output BAM.")
    args = parser.parse_args()
    separate_bam_by_bc(args.input_bam, args.output_prefix)
