# Installation

SMFtools requires Python 3.11 or newer. Create an isolated environment and
upgrade pip before installing:

```shell
conda create -n smftools python=3.11
conda activate smftools
python -m pip install --upgrade pip
python -m pip install smftools
```

The default install supports `smftools experiment full` from a basecalled BAM.
It includes the portable pysam BAM backend and the dependencies used by
preprocessing, spatial analysis, HMM analysis, and pipeline plotting.

## Optional feature profiles

Install profiles individually or combine them in one command, such as
`python -m pip install "smftools[ont,project]"`.

| Profile | Additional capability |
| --- | --- |
| `ont` | POD5 input and Nanopore signal I/O |
| `umi` | Edit-distance-based UMI and barcode processing |
| `genome-io` | pybedtools and pyBigWig genome-format backends |
| `project` | DuckDB catalogs and lazy xarray-backed project reads |
| `analysis` | Clustering, UMAP, tensor, graph, and XGBoost analyses |
| `ml-extended` | Captum, Lightning, SHAP, Weights & Biases, and related ML tools |
| `qc` | MultiQC report generation |
| `all` | Every optional runtime capability |

For example, a POD5-starting experiment that also uses UMI processing can use:

```shell
python -m pip install "smftools[ont,umi]"
```

The `genome-io` profile installs Python packages for working with BED and
BigWig data. Some pybedtools operations also require a separately installed
`bedtools` executable.

## External command-line tools

External tools depend on the experiment's input and configured backends:

- Dorado performs Nanopore basecalling and can perform alignment and
  demultiplexing. It is not needed when the configured workflow starts from a
  suitable basecalled BAM and uses another aligner/backend.
- Minimap2 can perform alignment when Dorado alignment is not used.
- Modkit extracts modification probabilities from MM/ML BAM tags for direct
  modification-detection protocols. The pysam modification backend is included
  in the default Python install and can be selected instead.
- samtools, bedtools, and BedGraphToBigWig are optional CLI backends. pysam is
  included by default; pybedtools and pyBigWig are available through
  `smftools[genome-io]`.

Make any selected executable available on `PATH`. For example:

```shell
dorado --version
minimap2 --version
```

## Development installation

Clone the repository and install it in editable mode. Runtime feature extras
remain package extras, while the canonical test, lint, and documentation setup
uses local dependency groups and is never installed by a normal runtime install:

```shell
git clone https://github.com/jkmckenna/smftools.git
cd smftools
python -m venv venv-smftools
source venv-smftools/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
python -m pip install --group dev --group docs
```

To use the environment as a Jupyter kernel, install Jupyter tooling separately:

```shell
python -m pip install ipykernel jupyter
python -m ipykernel install --user --name=venv-smftools --display-name "Python (smftools)"
```

## Compatibility aliases

Existing install commands continue to work during this migration. The previous
fine-grained extras (`cluster`, `misc`, `plotting`, `pybedtools`, `pybigwig`,
`pysam`, `ml-base`, `xgboost`, `umap`, `torch`, `lazy`, and `catalog`) remain as
aliases. The old `dev` and `docs` extras also remain available while contributor
automation moves to dependency groups. The `torch`, `plotting`, and `pysam`
extras are now redundant because those dependencies are installed by default.
`all_2` is retained as a deprecated alias for `all`; new documentation and
automation should use `all`.
