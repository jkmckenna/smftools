# subsample_fasta_from_bed

def subsample_fasta_from_bed(input_FASTA, input_bed, output_directory, output_FASTA):
    """
    Take a genome-wide FASTA file and a bed file containing coordinate windows of interest. Outputs a subsampled FASTA.

    Parameters:
        input_FASTA (str): String representing the path to the input FASTA file.
        input_bed (str): String representing the path to the input BED file.
        output_directory (str): String representing the path to the output directory for the new FASTA file.
        output_FASTA (str): Name of the output FASTA.
    
    Returns:
        None
    """
    from pyfaidx import Fasta
    import os

    # Load the FASTA file using pyfaidx
    fasta = Fasta(input_FASTA)

    output_FASTA_path = os.path.join(output_directory, output_FASTA)
    
    # Open the BED file
    with open(input_bed, 'r') as bed, open(output_FASTA_path, 'w') as out_fasta:
        for line in bed:
            # Each line in BED file contains: chrom, start, end (and possibly more columns)
            fields = line.strip().split()
            n_fields = len(fields)
            chrom = fields[0]
            start = int(fields[1])  # BED is 0-based
            end = int(fields[2])    # BED is 0-based and end is exclusive
            if n_fields > 3:
                description = " ".join(fields[3:])
            
            # Check if the chromosome exists in the FASTA file
            if chrom in fasta:
                # pyfaidx is 1-based, so convert coordinates accordingly
                sequence = fasta[chrom][start:end].seq
                # Write the sequence to the output FASTA file
                if n_fields > 3:
                    out_fasta.write(f">{chrom}:{start}-{end}    {description}\n")
                else:
                    out_fasta.write(f">{chrom}:{start}-{end}\n")
                out_fasta.write(f"{sequence}\n")
            else:
                print(f"Warning: {chrom} not found in the FASTA file")
