## count_aligned_reads

# General
def count_aligned_reads(bam_file):
    """
    Counts the number of aligned reads in a bam file that map to each reference record.
    
    Parameters:
        bam_file (str): A string representing the path to an aligned BAM file.
    
    Returns:
       aligned_reads_count (int): The total number or reads aligned in the BAM.
       unaligned_reads_count (int): The total number of reads not aligned in the BAM.
       record_counts (dict): A dictionary keyed by reference record instance that points toa tuple containing the total reads mapped to the record and the fraction of mapped reads which map to the record.

    """
    from .. import readwrite
    import pysam
    from tqdm import tqdm
    from collections import defaultdict

    print('{0}: Counting aligned reads in BAM > {1}'.format(readwrite.time_string(), bam_file))
    aligned_reads_count = 0
    unaligned_reads_count = 0
    # Make a dictionary, keyed by the reference_name of reference chromosome that points to an integer number of read counts mapped to the chromosome, as well as the proportion of mapped reads in that chromosome
    record_counts = defaultdict(int)

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        total_reads = bam.mapped + bam.unmapped
        # Iterate over reads to get the total mapped read counts and the reads that map to each reference
        for read in tqdm(bam, desc='Counting aligned reads in BAM', total=total_reads):
            if read.is_unmapped:
                unaligned_reads_count += 1
            else:
                aligned_reads_count += 1
                record_counts[read.reference_name] += 1  # Automatically increments if key exists, adds if not

        # reformat the dictionary to contain read counts mapped to the reference, as well as the proportion of mapped reads in reference
        for reference in record_counts:
            proportion_mapped_reads_in_record = record_counts[reference] / aligned_reads_count
            record_counts[reference] = (record_counts[reference], proportion_mapped_reads_in_record)

    return aligned_reads_count, unaligned_reads_count, dict(record_counts)