[![PyPI](https://img.shields.io/pypi/v/smftools.svg)](https://pypi.org/project/smftools)
[![Docs](https://readthedocs.org/projects/smftools/badge/?version=latest)](https://smftools.readthedocs.io/en/latest/?badge=latest)

# smftools
A Python tool for automated processing of raw sequencing data derived from single molecule footprinting experiments into [zarr](https://github.com/zarr-developers/zarr-python) and [parquet](https://github.com/apache/parquet-format/) data formats. Experimental data is organized into projects using [DuckDB](https://github.com/duckdb/duckdb) (optional; falls back to a pandas/pyarrow union without it) to facilitate growth of data collections for scientific projects. An additional analysis subpackage provides functionality that can be imported for custom analyses and interactive analysis sessions.

## Philosophy
While genomic data structures (SAM/BAM) were built to store read alignment data and basic read metadata along large references, integration of downstream analyses is not feasible using this format alone. Smftools integrates experimental analyses across file formats, linking raw sequencing data files, BAM alignment files, and downstream analyses into modern storage formats such as zarr for arrays and parquet for tables. This enables efficient partitioned data storage, rapid and parallel data operations, hierarchical metadata handling, and seamless integration with machine-learning workflows. Furthermore, functionality is modularized into multiple processing stages, enabling analysis to restart from convenient checkpoints without having to rerun the full workflow. Collections of experiments are managed under smftools projects, which indexes individual experiments and combine them for continuously growing scientific projects.

## Installation
SMFtools requires Python 3.11 or newer. The default installation supports
`smftools experiment full` from a basecalled BAM, including the portable pysam
BAM backend, preprocessing, spatial analysis, HMM analysis, and plotting:

```bash
pip install smftools

# Or install the current development checkout.
git clone https://github.com/jkmckenna/smftools.git
cd smftools
pip install -e .
```

Install only the optional capabilities a run needs, for example
`pip install -e ".[ont,project]"`:

- `ont` -> POD5 input and Nanopore signal I/O.
- `umi` -> edit-distance-based UMI and barcode processing.
- `genome-io` -> pybedtools and pyBigWig genome-format backends.
- `project` -> DuckDB catalogs and lazy xarray-backed project reads.
- `analysis` -> downstream clustering, UMAP, tensor, graph, and XGBoost analyses.
- `ml-extended` -> Captum, Lightning, SHAP, Weights & Biases, and related ML tools.
- `qc` -> MultiQC report generation.
- `all` -> every optional runtime capability.

Older fine-grained extras remain compatibility aliases. In particular, `torch`,
`plotting`, and `pysam` are now redundant because those dependencies are part of
the default workflow install, and `all_2` is an alias for `all`.

Canonical contributor installs use dependency groups so test, lint, and docs
tools are never installed by a normal runtime install:

```bash
python -m pip install -e ".[all]"
python -m pip install --group dev --group docs
```

See the [installation guide](https://smftools.readthedocs.io/en/latest/installation.html)
for profile details and external command-line requirements.

## Command-line interface
smftools exposes two top-level command groups (`smftools --help` for the full list):

**`smftools experiment <config_path>`** -> pipeline stages for a single experiment:
- `raw` -> Prepare BAM artifacts and write the ragged raw store.
- `load` -> Optionally pre-build the dense zarr cache from raw artifacts.
- `preprocess` -> QC, filtering, and read-level preprocessing.
- `spatial` -> Spatial signal analysis.
- `hmm` -> HMM feature annotation and plotting.
- `full` -> Composed workflow: raw, preprocess, spatial, hmm.
- `batch` -> Run any single stage across many experiments listed in a CSV/TSV/TXT file.
- `concatenate`, `export-fastq`, `plot-current` -> supporting utilities.

**`smftools project <project_dir>`** -> registering and querying across experiments:
- `init` -> Initialize a project directory + registry.
- `add` / `remove` -> Register or deactivate an experiment in the project.
- `list` -> List registered experiments and harmonized references.
- `materialize` -> Pool a reference across matching experiments into one AnnData.
- `sample-store-list` -> List cataloged per-sample-store partitions.
- `export-fastq` -> Write one FASTQ per barcode of QC-passed reads, across every registered experiment.

Full documentation for each command and its options is at [smftools.readthedocs.io](https://smftools.readthedocs.io/).

## Dependencies
The following CLI tools need to be installed and configured before using the informatics (smftools.inform) module of smftools, which is used by the `smftools experiment raw` CLI command:
1) [Dorado](https://github.com/nanoporetech/dorado) -> Basecalling, alignment, demultiplexing. Required for Nanopore SMF experiments, but not Illumina SMF experiments.
2) [Minimap2](https://github.com/lh3/minimap2) -> Aligner if not using dorado. Support for other aligners could eventually be added if needed.
3) [Modkit](https://github.com/nanoporetech/modkit) -> Extracting read level methylation metrics from the MM/ML tags in BAM files. Only required for direct modification detection SMF protocols.

## License
MIT -- see [LICENSE](LICENSE).
