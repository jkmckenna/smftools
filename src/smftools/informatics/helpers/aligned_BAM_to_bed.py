# aligned_BAM_to_bed

def aligned_BAM_to_bed(aligned_BAM, plotting_dir, bed_dir, fasta):
    """
    Takes an aligned BAM as input and writes a bed file of reads as output.
    Bed columns are: Record name, start position, end position, read length, read name

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract to a BED file.
        plotting_dir (str): Path to write out read alignment length and coverage histograms
        bed_dir (str): Path to write out read alignment coordinates
        fasta (str): File path to the reference genome to align to.

    Returns:
        None

    """
    import subprocess
    import os
    from .bed_to_bigwig import bed_to_bigwig
    from .plot_read_length_and_coverage_histograms import plot_read_length_and_coverage_histograms

    bed_output_basename = os.path.basename(aligned_BAM).split('.bam')[0] + '_bed.bed'
    bed_output = os.path.join(bed_dir, bed_output_basename)

    samtools_view = subprocess.Popen(["samtools", "view", aligned_BAM], stdout=subprocess.PIPE) 
    with open(bed_output, "w") as output_file:
        awk_process = subprocess.Popen(["awk", '{print $3 "\t" $4 "\t" $4+length($10)-1 "\t" length($10)-1 "\t" $1}'], stdin=samtools_view.stdout, stdout=output_file)    
    samtools_view.stdout.close()
    awk_process.wait()
    samtools_view.wait()

    def split_bed(bed, delete_input=True):
        """
        Reads in a BED file and splits it into two separate BED files based on alignment status.

        Parameters:
            bed (str): Path to the input BED file.
            delete_input (bool): Whether to delete the input bed file

        Returns:
            aligned (str): Path to the aligned bed file
        """
        unaligned = bed.split('.bed')[0] + '_unaligned.bed'
        aligned = bed.split('.bed')[0] + '_aligned.bed'
        
        with open(bed, 'r') as infile, \
            open(unaligned, 'w') as unaligned_outfile, \
            open(aligned, 'w') as aligned_outfile:
            
            for line in infile:
                fields = line.strip().split('\t')
                
                if fields[0] == '*':
                    unaligned_outfile.write(line)
                else:
                    aligned_outfile.write(line)

        if delete_input:
            os.remove(bed)
        
        return aligned
    
    aligned_bed = split_bed(bed_output)

    # Write out basic plots of reference coverage and read lengths
    plot_read_length_and_coverage_histograms(aligned_bed, plotting_dir)

    # Make a bedgraph and bigwig for the aligned reads
    bed_to_bigwig(fasta, aligned_bed)



