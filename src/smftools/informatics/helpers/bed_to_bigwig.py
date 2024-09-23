# bed_to_bigwig

def bed_to_bigwig(fasta, bed):
    """
    Takes a bed file of reads and makes a bedgraph plus a bigwig

    Parameters:
        fasta (str): File path to the reference genome to align to.
        bed (str): File path to the input bed.
    Returns:
        None
    """
    import os
    import subprocess

    bed_basename = os.path.basename(bed)
    parent_dir = os.path.dirname(bed)
    bed_basename_minus_suffix = bed_basename.split('.bed')[0]
    fasta_basename = os.path.basename(fasta)
    fasta_dir = os.path.dirname(fasta)
    fasta_basename_minus_suffix = fasta_basename.split('.fa')[0]
    chrom_basename = fasta_basename_minus_suffix + '.chrom.sizes'
    chrom_path = os.path.join(fasta_dir, chrom_basename)
    bedgraph_basename = bed_basename_minus_suffix + '_bedgraph.bedgraph'
    bedgraph_output = os.path.join(parent_dir, bedgraph_basename)
    bigwig_basename = bed_basename_minus_suffix + '_bigwig.bw'
    bigwig_output = os.path.join(parent_dir, bigwig_basename)

    # Make the bedgraph
    with open(bedgraph_output, 'w') as outfile:
        # Command as a list
        command = ["bedtools", "genomecov", "-i", bed, "-g", chrom_path, "-bg"]
        print(f'Making bedgraph from {bed_basename}')
        subprocess.run(command, stdout=outfile)

    # Make the bigwig
    command = ["bedGraphToBigWig", bedgraph_output, chrom_path, bigwig_output]
    print(f'Making bigwig from {bedgraph_basename}')
    subprocess.run(command)