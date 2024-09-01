# smftools
A Python tool for processing raw sequencing data derived from single molecule footprinting experiments into [anndata](https://anndata.readthedocs.io/en/latest/) objects. Additional functionality for preprocessing, analysis, and visualization. 

## Philosophy
While most genomic data structures handle low-coverage data (<100X) along large references, smftools prioritizes high-coverage data (scalable to at least 1 million X coverage) of a few genomic loci at a time. This enables efficient data storage, rapid data operations, hierarchical metadata handling, seamless integration with various machine-learning packages, and ease of visualization. Furthermore, functionality is modularized, enabling analysis sessions to be saved, reloaded, and easily shared with collaborators.

## Dependencies
The following CLI tools need to be installed and configured before using smftools:
1) [Dorado](https://github.com/nanoporetech/dorado) -> For standard/modified basecalling and alignment. Can be attained by downloading and configuring nanopore MinKnow software.
2) [Samtools](https://github.com/samtools/samtools) -> For working with SAM/BAM files
3) [Minimap2](https://github.com/lh3/minimap2) -> The aligner used by Dorado
4) [Modkit](https://github.com/nanoporetech/modkit) -> Extracting summary statistics and read level methylation calls from modified BAM files

## Announcements
### 08/30/24 - The pre-alpha phase package ([smftools-0.1.0](https://pypi.org/project/smftools/)) is installable through pypi!
Currently, this package (smftools-0.1.0) is going through rapid improvement (dependency handling accross Linux and Mac OS, testing, documentation, debugging) and is still too early in development for standard use. The underlying functionality was originally developed as a collection of scripts for single molecule footprinting (SMF) experiments in our lab, but is being packaged/developed to facilitate the expansion of SMF to any lab that is interested in performing these styles of experiments/analyses. The alpha-phase package is expected to be available within a couple months, so stay tuned!
