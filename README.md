[![PyPI](https://img.shields.io/pypi/v/smftools.svg)](https://pypi.org/project/smftools)
[![Docs](https://readthedocs.org/projects/smftools/badge/?version=latest)](https://smftools.readthedocs.io/en/latest/?badge=latest)

# smftools
A Python tool for processing raw sequencing data derived from single molecule footprinting experiments into [anndata](https://anndata.readthedocs.io/en/latest/) objects. Additional functionality for preprocessing, spatial analyses, and HMM based feature annotation.

## Philosophy
While genomic data structures (SAM/BAM) were built to handle low-coverage data (<1000X) along large references, smftools prioritizes high-coverage data (scalable to >1,000,000X coverage) of a few genomic loci at a time. This enables efficient data storage, rapid data operations, hierarchical metadata handling, seamless integration with various machine-learning packages, and ease of visualization. Furthermore, functionality is modularized, enabling analysis sessions to be saved, reloaded, and easily shared with collaborators. Analyses are centered around the [anndata](https://anndata.readthedocs.io/en/latest/) object, and are heavily inspired by the work conducted within the single-cell genomics community.

## Dependencies
The following CLI tools need to be installed and configured before using the informatics (smftools.inform) module of smftools:
1) [Dorado](https://github.com/nanoporetech/dorado) -> Basecalling, alignment, demultiplexing.
2) [Minimap2](https://github.com/lh3/minimap2) -> Alignment if not using dorado.
3) [Modkit](https://github.com/nanoporetech/modkit) -> Extracting read level methylation metrics from modified BAM files. Only required for direct modification detection (ie methylation).

## Main Commands
### smftools load: Processes raw Nanopore/Illumina data from SMF experiments into an AnnData object.
![](docs/source/_static/smftools_informatics_diagram.png)
### smftools preprocess: Appends QC metrics to the AnnData object and performs filtering.
![](docs/source/_static/smftools_preprocessing_diagram.png)
### smftools spatial: Appends spatial analyses to the AnnData object.
- Currently Includes: Position X Position correlation matrices, clustering, dimensionality reduction, spatial autocorrelation. 
### smftools hmm: Fits a basic HMM to each sample and appends HMM feature layers
- Main outputs wills be stored in adata.layers
### smftools batch <command>: Performs batch processing on a csv of config file pathes for any of the above commands.
- Nice when analyzing multiple experiments
### smftools concatenate: Concatenates a list or directory of anndata objects.
- Mainly used for combining multiple experiments into a single anndata object.
