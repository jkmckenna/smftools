# Command line tutorials

## Quick start

Most CLI workflows start with an experiment configuration CSV that points to your data, FASTA, and
output directory. Once the configuration is ready, you can run commands like:

```shell
smftools load /path/to/experiment_config.csv
smftools preprocess /path/to/experiment_config.csv
smftools spatial /path/to/experiment_config.csv
smftools hmm /path/to/experiment_config.csv
```

Each command will create (or reuse) stage-specific AnnData files in the output directory. Later
commands reuse results from earlier stages unless you explicitly force a redo via configuration
flags.

## What each command does

### `smftools load`

The load command builds the raw AnnData object from your raw sequencing data. It:

- Handles input formats (fast5/pod5/fastq/bam).
- Performs basecalling, alignment, demultiplexing, and BAM QC.
- Optionally generates BED/bigWig outputs for alignment summaries.
- Constructs the raw AnnData object (Single molecules x Positional coordinates).
- Adds basic read-level QC annotations.
- Writes the raw AnnData to the canonical output path and runs MultiQC.
- Optionally deletes intermediate BAMs, H5ADs, and TSVs.

### `smftools preprocess`

The preprocess command performs QC, binarization, filtering, and duplicate detection. It:

- Loads sample sheet metadata (if provided).
- Generates read length/quality QC plots and filters reads on these metrics.
- Binarizes direct-modification calls based on thresholds (hard or fit thresholds).
- Cleans NaNs in adata.layers.
- Computes positional coverage and base-context annotations.
- Calculates read modification statistics and QC plots.
- Filters reads based on modification thresholds.
- Adds base-context binary layers.
- Flags duplicate reads and performs complexity analyses (conversion/deamination workflows).
- Writes preprocessed and deduplicated AnnData outputs.

### `smftools spatial`

The spatial command runs downstream spatial analyses on the preprocessed data. It:

- Optionally loads sample sheet metadata.
- Optionally inverts and reindexes the data along the reference axis.
- Generates clustermaps for preprocessed (and deduplicated) AnnData.
- Runs PCA/UMAP/Leiden clustering.
- Computes spatial autocorrelation, rolling metrics, and grid summaries.
- Generates positionwise correlation matrices (non-direct modalities).
- Writes the spatial AnnData output.

### `smftools hmm`

The hmm command adds HMM-based feature annotation and summary plots. It:

- Ensures preprocessing and spatial analyses are up to date.
- Fits or reuses HMM models for configured feature sets.
- Annotates AnnData with HMM-derived layers and merged intervals.
- Calls HMM feature peaks and writes peak-calling outputs.
- Generates clustermaps, rolling traces, and fragment size plots for HMM layers.
- Writes the HMM AnnData output.

## Batch processing

Use the batch command to run a single task across multiple experiments.

```shell
smftools batch preprocess /path/to/config_paths.csv
```

The batch command accepts:

- **CSV/TSV** tables with a column of config paths (default column name: `config_path`).
- **TXT** files with one config path per line.

You can override the column name or delimiter if needed:

```shell
smftools batch spatial /path/to/configs.tsv --column my_config --sep $'\t'
```

Each path is validated; missing configs are skipped with a message, while valid configs run the
requested task in sequence.