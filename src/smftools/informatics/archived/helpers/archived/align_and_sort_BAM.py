from pathlib import Path
import os
import subprocess
from typing import List, Optional, Union
import pysam

def _bam_to_fastq_with_pysam(bam_path: Union[str, Path], fastq_path: Union[str, Path]) -> None:
    """
    Minimal BAM->FASTQ using pysam. Writes unmapped or unaligned reads as-is.
    """
    bam_path = str(bam_path)
    fastq_path = str(fastq_path)
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam, open(fastq_path, "w") as fq:
        for r in bam.fetch(until_eof=True):
            # Skip secondary/supplementary if you want (optional):
            # if r.is_secondary or r.is_supplementary: continue
            name = r.query_name
            seq = r.query_sequence or ""
            qual = r.qual or ""
            fq.write(f"@{name}\n{seq}\n+\n{qual}\n")

def _sort_bam_with_pysam(in_bam: Union[str, Path], out_bam: Union[str, Path], threads: Optional[int] = None) -> None:
    in_bam, out_bam = str(in_bam), str(out_bam)
    args = []
    if threads:
        args += ["-@", str(threads)]
    args += ["-o", out_bam, in_bam]
    pysam.sort(*args)

def _index_bam_with_pysam(bam_path: Union[str, Path], threads: Optional[int] = None) -> None:
    bam_path = str(bam_path)
    # pysam.index supports samtools-style args
    if threads:
        pysam.index("-@", str(threads), bam_path)
    else:
        pysam.index(bam_path)

def align_and_sort_BAM(fasta, 
                       input, 
                       bam_suffix='.bam', 
                       output_directory='aligned_outputs', 
                       make_bigwigs=False, 
                       threads=None, 
                       aligner='minimap2',
                       aligner_args=['-a', '-x', 'map-ont', '--MD', '-Y', '-y', '-N', '5', '--secondary=no']):
    """
    A wrapper for running dorado aligner and samtools functions
    
    Parameters:
        fasta (str): File path to the reference genome to align to.
        input (str): File path to the basecalled file to align. Works for .bam and .fastq files
        bam_suffix (str): The suffix to use for the BAM file.
        output_directory (str): A file path to the directory to output all the analyses.
        make_bigwigs (bool): Whether to make bigwigs
        threads (int): Number of additional threads to use
        aligner (str): Aligner to use. minimap2 and dorado options
        aligner_args (list): list of optional parameters to use for the alignment

    Returns:
        None
            The function writes out files for: 1) An aligned BAM, 2) and aligned_sorted BAM, 3) an index file for the aligned_sorted BAM, 4) A bed file for the aligned_sorted BAM, 5) A text file containing read names in the aligned_sorted BAM
    """
    input_basename = input.name
    input_suffix = input.suffix
    input_as_fastq = input.with_name(input.stem + '.fastq')

    output_path_minus_suffix = output_directory / input.stem
    
    aligned_BAM = output_path_minus_suffix.with_name(output_path_minus_suffix.stem + "_aligned")
    aligned_output = aligned_BAM.with_suffix(bam_suffix)
    aligned_sorted_BAM =aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(bam_suffix)

    if threads:
        threads = str(threads)
    else:
        pass
    
    if aligner == 'minimap2':
        print(f"Converting BAM to FASTQ: {input}")
        _bam_to_fastq_with_pysam(input, input_as_fastq)
        # bam_to_fastq_command = ['samtools', 'fastq', input]
        # subprocess.run(bam_to_fastq_command, stdout=open(input_as_fastq, "w"))
        print(f"Aligning FASTQ to Reference: {input_as_fastq}")
        if threads:
            minimap_command = ['minimap2'] + aligner_args + ['-t', threads, str(fasta), str(input_as_fastq)]
        else:
            minimap_command = ['minimap2'] + aligner_args + [str(fasta), str(input_as_fastq)]
        subprocess.run(minimap_command, stdout=open(aligned_output, "w"))
        os.remove(input_as_fastq)

    elif aligner == 'dorado':
        # Run dorado aligner
        print(f"Aligning BAM to Reference: {input}")
        if threads:
            alignment_command = ["dorado", "aligner", "-t", threads] + aligner_args + [str(fasta), str(input)]
        else:
            alignment_command = ["dorado", "aligner"] + aligner_args + [str(fasta), str(input)]
        subprocess.run(alignment_command, stdout=open(aligned_output, "wb"))

    else:
        print(f'Aligner not recognized: {aligner}. Choose from minimap2 and dorado')
        return
    
    # --- Sort & Index with pysam ---
    print(f"[pysam] Sorting: {aligned_output} -> {aligned_sorted_output}")
    _sort_bam_with_pysam(aligned_output, aligned_sorted_output, threads=threads)

    print(f"[pysam] Indexing: {aligned_sorted_output}")
    _index_bam_with_pysam(aligned_sorted_output, threads=threads)

    # Sort the BAM on positional coordinates
    # print(f"Sorting BAM: {aligned_output}")
    # if threads:
    #     sort_command = ["samtools", "sort", "-@", threads, "-o", aligned_sorted_output, aligned_output]
    # else:
    #     sort_command = ["samtools", "sort", "-o", aligned_sorted_output, aligned_output]
    # subprocess.run(sort_command)

    # # Create a BAM index file
    # print(f"Indexing BAM: {aligned_sorted_output}")
    # if threads:
    #     index_command = ["samtools", "index", "-@", threads, aligned_sorted_output]
    # else:
    #     index_command = ["samtools", "index", aligned_sorted_output]
    # subprocess.run(index_command)