# aligned_BAM_to_bed

def aligned_BAM_to_bed(aligned_BAM, plotting_dir):
    """
    Takes an aligned BAM as input and writes a bed file of reads as output.
    Bed columns are: Record name, start position, end position, read length, read name

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract to a BED file.

    Returns:
        None

    """
    import subprocess
    from .plot_read_length_and_coverage_histograms import plot_read_length_and_coverage_histograms
    bed_output = aligned_BAM.split('.bam')[0] + '_bed.bed'
    samtools_view = subprocess.Popen(["samtools", "view", aligned_BAM], stdout=subprocess.PIPE) 
    with open(bed_output, "w") as output_file:
        awk_process = subprocess.Popen(["awk", '{print $3 "\t" $4 "\t" $4+length($10)-1 "\t" length($10)-1 "\t" $1}'], stdin=samtools_view.stdout, stdout=output_file)    
    samtools_view.stdout.close()
    awk_process.wait()
    samtools_view.wait()

    def split_bed(bed):
        """
        Reads in a BED file and splits it into two separate BED files based on alignment status.

        Parameters:
            bed (str): Path to the input BED file.

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
        
        return aligned
    
    aligned_bed = split_bed(bed_output)
    plot_read_length_and_coverage_histograms(aligned_bed, plotting_dir)