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

## Announcements

### 12/02/25 - Version 0.2.3 is available through PyPI
Version 0.2.3 provides the core smftools functionality through several command line commands (load, preprocess, spatial, hmm).

### 11/05/25 - Version 0.2.1 is available through PyPI
Version 0.2.1 makes the core workflow (smftools load) a command line tool that takes in an experiment_config.csv file for input/output and parameter management.

### 05/29/25 - Version 0.1.6 is available through PyPI.
Informatics, preprocessing, tools, plotting modules have core functionality that is approaching stability on MacOS(Intel/Silicon) and Linux(Ubuntu). I will work on improving documentation/tutorials shortly. The base PyTorch/Scikit-Learn ML-infrastructure is going through some organizational changes to work with PyTorch Lightning, Hydra, and WanDB to facilitate organizational scaling, multi-device usage, and logging.

### 10/01/24 - More recent versions are being updated frequently. Installation from source over PyPI is recommended!

### 09/09/24 - The version 0.1.1 package ([smftools-0.1.1](https://pypi.org/project/smftools/)) is installable through pypi!
The informatics module has been bumped to alpha-phase status. This module can deal with POD5s and unaligned BAMS from nanopore conversion and direct SMF experiments, as well as FASTQs from Illumina conversion SMF experiments. Primary output from this module is an AnnData object containing all relevant SMF data, which is compatible with all downstream smftools modules. The other modules are still in pre-alpha phase. Preprocessing, Tools, and Plotting modules should be promoted to alpha-phase within the next month or so.

### 08/30/24 - The version 0.1.0 package ([smftools-0.1.0](https://pypi.org/project/smftools/)) is installable through pypi!
Currently, this package (smftools-0.1.0) is going through rapid improvement (dependency handling accross Linux and Mac OS, testing, documentation, debugging) and is still too early in development for widespread use. The underlying functionality was originally developed as a collection of scripts for single molecule footprinting (SMF) experiments in our lab, but is being packaged/developed to facilitate the expansion of SMF to any lab that is interested in performing these styles of experiments/analyses. The alpha-phase package is expected to be available within a couple months, so stay tuned!
