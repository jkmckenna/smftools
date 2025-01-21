import pysam
import sys

def extract_reads(bam_file_path, num_reads=10):
    # Open the BAM file
    bam_file = pysam.AlignmentFile(bam_file_path, "rb")
    
    # Iterate through the first 'num_reads' reads and print the sequences
    count = 0
    for read in bam_file:
        print(f"Read {count + 1}: {read.query_sequence}")
        count += 1
        if count >= num_reads:
            break
    
    # Close the BAM file
    bam_file.close()

if __name__ == "__main__":
    # Ensure a BAM file path is provided as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python extract_reads.py <path_to_bam_file>")
        sys.exit(1)

    # Get the BAM file path from command line arguments
    bam_file_path = sys.argv[1]
    
    # Call the function to extract the first 10 reads
    extract_reads(bam_file_path)