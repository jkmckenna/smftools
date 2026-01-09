[![PyPI](https://img.shields.io/pypi/v/smftools.svg)](https://pypi.org/project/smftools)
[![Docs](https://readthedocs.org/projects/smftools/badge/?version=latest)](https://smftools.readthedocs.io/en/latest/?badge=latest)

# smftools
A Python tool for processing raw sequencing data derived from single molecule footprinting experiments into [anndata](https://anndata.readthedocs.io/en/latest/) objects. Additional functionality for preprocessing, spatial analyses, and HMM based feature annotation.

## Philosophy
While genomic data structures (SAM/BAM) were built to handle low-coverage data (<1000X) along large references, smftools prioritizes high-coverage data (scalable to >1,000,000X coverage) of a few genomic loci at a time. This enables efficient data storage, rapid data operations, hierarchical metadata handling, seamless integration with various machine-learning packages, and ease of visualization. Furthermore, functionality is modularized, enabling analysis sessions to be saved, reloaded, and easily shared with collaborators. Analyses are centered around the [anndata](https://anndata.readthedocs.io/en/latest/) object, and are heavily inspired by the work conducted within the single-cell genomics community.

## Dependencies
The following CLI tools need to be installed and configured before using the informatics (smftools.inform) module of smftools, which is used by the smftools load CLI command:
1) [Dorado](https://github.com/nanoporetech/dorado) -> Basecalling, alignment, demultiplexing. Required for Nanopore SMF experiments, but not Illumina SMF experiments.
2) [Minimap2](https://github.com/lh3/minimap2) -> Aligner if not using dorado. Support for other aligners could eventually be added if needed.
3) [Modkit](https://github.com/nanoporetech/modkit) -> Extracting read level methylation metrics from the MM/ML tags in BAM files. Only required for direct modification detection SMF protocols.
