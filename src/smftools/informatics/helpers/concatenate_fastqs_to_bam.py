# concatenate_fastqs_to_bam

def concatenate_fastqs_to_bam(fastq_files, output_bam, barcode_tag='BC', gzip_suffix='.gz'):
    """
    Concatenate multiple demultiplexed FASTQ (.fastq or .fq) files into an unaligned BAM and add the FASTQ barcode suffix to the BC tag.

    Parameters:
        fastq_files (list): List of paths to demultiplexed FASTQ files.
        output_bam (str): Path to the output BAM file.
        barcode_tag (str): The SAM tag for storing the barcode (default: 'BC').
        gzip_suffix (str): Suffix to use for input gzip files (Defaul: '.gz')

    Returns:
        None
    """
    import os
    import pysam
    import gzip
    from Bio import SeqIO
    from tqdm import tqdm

    n_fastqs = len(fastq_files)

    with pysam.AlignmentFile(output_bam, "wb", header={"HD": {"VN": "1.0"}, "SQ": []}) as bam_out:
        for fastq_file in tqdm(fastq_files, desc="Processing FASTQ files"):
            # Extract barcode from the FASTQ filename (handles .fq, .fastq, .fq.gz, and .fastq.gz extensions)
            base_name = os.path.basename(fastq_file)
            if n_fastqs > 1:
                if base_name.endswith('.fastq.gz'):
                    barcode = base_name.split('_')[-1].replace(f'.fastq{gzip_suffix}', '')
                elif base_name.endswith('.fq.gz'):
                    barcode = base_name.split('_')[-1].replace(f'.fq{gzip_suffix}', '')
                elif base_name.endswith('.fastq'):
                    barcode = base_name.split('_')[-1].replace('.fastq', '')
                elif base_name.endswith('.fq'):
                    barcode = base_name.split('_')[-1].replace('.fq', '')
                else:
                    raise ValueError(f"Unexpected file extension for {fastq_file}. Only .fq, .fastq, .fq{gzip_suffix}, and .fastq{gzip_suffix} are supported.")
            else:
                barcode = 'barcode0'

            # Read the FASTQ file (handle gzipped and non-gzipped files)
            open_func = gzip.open if fastq_file.endswith(gzip_suffix) else open
            with open_func(fastq_file, 'rt') as fq_in:
                for record in SeqIO.parse(fq_in, 'fastq'):
                    # Create an unaligned BAM entry for each FASTQ record
                    aln = pysam.AlignedSegment()
                    aln.query_name = record.id
                    aln.query_sequence = str(record.seq)
                    aln.flag = 4  # Unmapped
                    aln.query_qualities = pysam.qualitystring_to_array(record.letter_annotations["phred_quality"])
                    # Add the barcode to the BC tag
                    aln.set_tag(barcode_tag, barcode)
                    # Write to BAM file
                    bam_out.write(aln)
