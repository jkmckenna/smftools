# smftools
A tool for processing raw sequencing data for single molecule footprinting experiments at single genomic loci. Outputs an [anndata](https://github.com/scverse/anndata) object to be used in downstream analyses provided by [smfplot](https://github.com/jkmckenna/smfplot).

**Current biological questions include, but are not limited to:**
1) Site specific transcription factor binding kinetics, both in vitro and in permeabilized nuclei.
2) Locus specific nucleosome remodeling kinetics in permeabilized nuclei.
3) Thermodynamic modeling of transcriptional states as a function of transcription factor binding occupancy in permeabilized nuclei.
4) Long range cis-regulatory interactions throughout differentiation and during maintainance of cell state in permeabilized nuclei.
5) How enzymatic activities of transcriptional coactivators/corepressors kinetically alter chromatin microstates to regulate transcription in permeabilized nuclei.

## Table of Contents
- [User Guide](https://github.com/jkmckenna/smftools/edit/main/README.md#user-guide)
  - [Intended usage](https://github.com/jkmckenna/smftools/edit/main/README.md#intended-usage)
  - [Data output](https://github.com/jkmckenna/smftools/edit/main/README.md#data-output)
  - [Dependencies](https://github.com/jkmckenna/smftools/edit/main/README.md#dependencies)
  - [Basic Usage](https://github.com/jkmckenna/smftools/edit/main/README.md#basic-usage)
    - [General dataflow](https://github.com/jkmckenna/smftools/edit/main/README.md#general-dataflow)
  - [Performance Notes](https://github.com/jkmckenna/smftools/edit/main/README.md#performance-notes)
  - [In the works](https://github.com/jkmckenna/smftools/edit/main/README.md#in-the-works)

## User Guide
In the current format, this repository contains a collection of shell and python scripts configured to handle both standard and modified base call data derived from [Oxford Nanopore sequencers](https://nanoporetech.com/platform/technology) for the purpose of high-coverage, long-read, single-locus, single molecule footprinting (SMF) assays. The standard basecall workflow will be made compatible with any converted amplicon SMF experiment that has FASTQ files and will not be limited to Nanopore SMF experiments.

For the direct methylation calling protocol, these scripts collectively take raw nanopore POD5 data, perform high accuracy modified base calls for 5mC and 6mA, aligns the modified base called reads to a reference genome, extracts read-level positional methylation data, binarizes the methylation states, and packages the data into an [anndata](https://github.com/scverse/anndata) object. To run this workflow, open the shell script smftools_native.sh and fill out the user defined parameters. Next, execute the shell script in an environment that contains the dependencies listed above.

A standard base calling workflow is also provided for data derived from samples that were selectively deaminated, converted (C->T or A->G), and PCR amplified to detect methylation sites. This workflow takes raw nanopore POD5 data, performs high accuracy canonical base calling, creates all possible conversion possibilities for a reference FASTA, aligns the basecalled reads to the converted reference set, extracts the read-level positional methylation data based on conversion-state, binarizes the methylation states, and packages the data into an anndata object. To run this workflow, open the shell script smftools_converted.sh and fill out the user defined parameters. Next, execute the shell script in an environment that contains the dependencies listed above.

## Intended usage 
This infrastructure was developed to handle single molecule footprinting (SMF) data generated from adenine-based (Hia5-mediated m6A modification) and cytosine-based (M.CviPI-mediated 5mC modification of GpC cytosines) SMF workflows. This workflow works both with direct modification calls, as well as with canonical base calls derived from converted-base strategies (A->G conversion and C->T conversion). The direct detection and conversion based SMF approaches have a different workflow when processing the POD5 -> Anndata. However, the resulting anndata object dervied from the two workflows are essentially the same layout, but contain slightly different observation-level metadata. These anndata objects are directly compatible with the plotting package and notebooks provided with [smfplot](https://github.com/jkmckenna/smfplot).

## Data output
The core data structure output by this processing workflow is the [anndata](https://github.com/scverse/anndata) object. This object handles complex data nicely and is being actively developed and maintained by a solid community of developers in the single-cell genomics field. We have abstracted the usage of [scanpy](https://github.com/scverse/scanpy) and anndata objects to work with a Read X Position matrix analagous to the typical Cell X Gene matrix used in single cell RNA sequencing analyses.

## Dependencies
To run this workflow, the following tools need to be installed and configured:
1) [Dorado](https://github.com/nanoporetech/dorado) -> For standard/modified basecalling and alignment
2) [Samtools](https://github.com/samtools/samtools) -> For working with SAM/BAM files
3) [Minimap2](https://github.com/lh3/minimap2) -> The aligner used by Dorado
4) [Modkit](https://github.com/nanoporetech/modkit) -> Extracting summary statistics and read level methylation calls from modified BAM files

## Basic Usage
### General Dataflow:
**Input requirements ->** 
1) POD5 file of raw nanopore data.
2) FASTA of reference locus.
  
**Modified Base Calling Workflow**
1) Dorado basecaller takes an input POD5 file and performs high accuracy modified base calling to generate a modified BAM file.
2) Dorado aligner takes the modified BAM file and a reference FASTA to create an aligned BAM.
3) Samtools takes the aligned BAM and sorts it by positional index.
4) separate_BAM_by_tag.py script splits the BAM into individual BAM files based on BC tag.
5) Modkit extract takes the directory of split BAMs and generates read level modification metadata. This is output as a zipped TSV.
6) Modkit_extract_to_anndata.py script takes the directory containing the zipped TSV files, binarizes the data, and outputs the overall binarized data from the experiment into a gzipped anndata object (.h5ad.gz file)

**Converted Amplicon Workflow**
1) Dorado basecaller takes an input POD5 and performs high accuracy standard base calling to generate demultiplexed FASTQ files.
2) Generate_converted_FASTA.py converts an input reference FASTA to a FASTA containing all possible conversion types for the original loci of interst.
3) Minimap2 aligner takes the demultiplexed FASTQ files and the converted reference FASTA to create an aligned BAM.
4) Samtools takes the aligned BAM and sorts it by positional index.
5) Converted_BAM_to_anndata.py takes the converted BAM, determines which modifications and strands are present, extracts the base identities at all informative positions, binarizes the methylation state data, and packages all of the data into a gzipped anndata object (.h5ad.gz file)

## Performance Notes
Overall, an SMF experiment that takes a full Nanopore MinION will take a couple days to process on a standard laptop computer, but can be completed in several hours on a desktop computer with 256Gb RAM. I anticipate on making steps 2 and 3 listed below more efficient in time through using [dask](https://github.com/dask/dask)

1) The initial high accuracy basecalling step can take overnight on a standard laptop computer, but this should only take an hour or two if it's run on a high performance computer with at least 256Gb RAM.
2) The Modkit extract step also takes overnight on a standard laptop computer, but this should only take an hour or two if it's run on a high performance computer with at least 256Gb RAM.
3) Binarization of methylation data and packaging it into the final anndata object likewise has similar computational requirements as the above steps. You can specify the batch size that they would like to process at a time for the input TSV files generated by Modkit extract.


## In the works
**1) Building a CLI application for processing POD5 -> anndata:**
I am currently reformatting the repository to consolidate all of the existing code into a format that will be wrapped into a CLI application. I am doing this using the python [click](https://github.com/pallets/click) package. This CLI application will have functionality to process any locus defined SMF experiment, both for direct methylation calls derived from Nanopore sequencers and for converted base calls derived from any sequencing platform. 

**2) Additional functionality:**
Code is being developed for unique molecular identifier (UMI) based SMF experiments to enable consensus calling for higher accuracy, as well as for a more quantitative assessment of the relative abundances of microstates in a population (helps reduce PCR biases). This addtional functionality will be provided by the [UMIC-seq](https://github.com/fhlab/UMIC-seq) tool developed by the Hollfelder Lab.

**3) Optimizing performance through parallel computing :**
I anticipate on adding [dask](https://github.com/dask/dask) functionality to allow users to define how they would like to distribute the computational tasks. This will make massive differences in processing time when using a high-performace computing cluster or a local desktop computer with high memory capacities. This will improve computation time for the Modkit extract step, the binarization of the methylation state data, and the consolidation of the data into the final anndata object.
