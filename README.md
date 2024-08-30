# smftools
A tool for processing raw sequencing data for single molecule footprinting experiments at single genomic loci.

## Dependencies
The following tools need to be installed and configured:
1) [Dorado](https://github.com/nanoporetech/dorado) -> For standard/modified basecalling and alignment. Can be attained by downloading and configuring nanopore MinKnow software.
2) [Samtools](https://github.com/samtools/samtools) -> For working with SAM/BAM files
3) [Minimap2](https://github.com/lh3/minimap2) -> The aligner used by Dorado
4) [Modkit](https://github.com/nanoporetech/modkit) -> Extracting summary statistics and read level methylation calls from modified BAM files
