# Command line tutorials

## Quick start

Most CLI workflows start with an experiment configuration CSV that points to your data, FASTA, and
output directory. Once the configuration is ready, you can run commands such as:

```shell
smftools load /path/to/experiment_config.csv
smftools preprocess /path/to/experiment_config.csv
smftools full /path/to/experiment_config.csv
smftools batch full /path/to/config_paths.csv
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
- adata.X contains binarized modification data (conversion/deaminase), or modification probabilitiesc (native).
- Adds basic read-level QC annotations (Read start, end, length, mean quality).
- Adds layers encoding read DNA sequences, base quality scores, base mismatches.
- Maintains BAM Tags/Flags in adata.obs.
- Writes the raw AnnData to the canonical output path and runs MultiQC.
- Optionally deletes intermediate BAMs, H5ADs, and TSVs.

### `smftools preprocess`

The preprocess command performs QC, binarization, filtering, and duplicate detection. It:

- Requires an Anndata created by smftools load.
- Loads sample sheet metadata (if provided).
- Generates read length/quality QC plots and filters reads on these metrics.
- Binarizes direct-modification calls based on thresholds (hard or fit thresholds).
- Cleans NaNs from adata.X and stores in adata.layers (nan0_0minus1, nan_half).
- Computes positional coverage and base-context annotations (GpC, CpG, ambiguous, other C, any C).
- Calculates read modification statistics and QC plots.
- Filters reads based on modification thresholds.
- Adds base-context binary modification layers.
- Optionally inverts and reindexes the data along the var (positions) axis.
- Flags duplicate reads based on nearest neighbor hamming distance of overlapping valid sites (Conversion/deamination).
- Performs complexity analyses using duplicate read clusters and Lander/Waterman fits (conversion/deamination workflows).
- Visualizes read span masks and base quality scores with clustermaps.
- Writes preprocessed (duplicates flagged, but kept) and preprocessed/deduplicated AnnData outputs.

### `smftools variant`

The variant command focuses on DNA sequence variation analyses. It:

- Requires at least a preprocessed AnnData object.
- Calculates position level variation frequencies per reference/sample.
- Generates z-scores for variant occurance given read level Q-scores and assuming uniform Palt transitions.
- Visualizes read DNA sequence encodings and mismatch encodings.

### `smftools chimeric`

The chimeric command is meant to find putative PCR chimeras. It:

- Requires at least a preprocessed AnnData object.
- Performs sliding window nearest neighbor hamming distance analysis per read.
- Visualizes the windowed nearest neighbor hamming distances per read.
- Assembles maximum spanning intervals of 0-hamming distance neighbors per read within the reference/sample.
- In progress.

### `smftools spatial`

The spatial command runs downstream spatial analyses on the preprocessed data. It:

- Requires at least a preprocessed AnnData object.
- Optionally loads sample sheet metadata.
- Optionally inverts and reindexes the data along the positions axis.
- Generates clustermaps for preprocessed (and deduplicated) AnnData.
- Computes spatial autocorrelation, rolling metrics, and grid summaries.
- Generates positionwise correlation matrices.
- Writes the spatial AnnData output.

### `smftools hmm`

The hmm command adds HMM-based feature annotation and summary plots. It:

- Requires at least a preprocessed AnnData object.
- Fits or reuses HMM models for configured feature sets.
- Annotates AnnData with HMM-derived feature layers (State layers and probability layers)
- Calls HMM feature peaks and writes peak-calling outputs.
- Generates clustermaps, bulk feature traces, and fragment size distribution plots for HMM layers.
- Writes the HMM AnnData output.

### `smftools latent`

The latent command constructs latent representations of the data. It:

- Requires at least a preprocessed AnnData object.
- Runs various dimensionality reduction and graph construction modalities:
    - Principle component analysis (PCA)
    - K-nearest neighbor (KNN)
    - Uniform manifold approximation and projection (UMAP)
    - Non-negative matrix factorization (NMF)
    - Canonical polyadic decomposition (PARAFAC)

### `smftools full`

The full command is a workflow wrapper. It runs the following sequentially:

- Load / preprocess / variant / chimeric / spatial / hmm / latent.


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