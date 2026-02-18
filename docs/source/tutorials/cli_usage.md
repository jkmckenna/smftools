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
- Maintains BAM tags/flags in adata.obs (UMI and barcode annotations loaded from Parquet sidecars).
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

General AnnData structures added by `preprocess`:

- `obs`
- sample-sheet metadata columns (when `sample_sheet_path` is provided) mapped by `sample_sheet_mapping_column`.
- read-level QC/ratio/modification summary fields used for filtering and plotting (for example read length/quality/mapping and fraction-modified metrics).
- optional UMI preprocessing fields when `use_umi=True`, including validity and clustering annotations (for example `U1_pass`, `U2_pass`, `U1_cluster`, `U2_cluster`, `RX_cluster`).
- optional UMI bipartite graph annotations and dominance metrics (for example edge-count/dominant-pair style fields) plus group-level stats in `uns`.
- duplicate-detection fields for non-direct modalities (for example merged cluster IDs/sizes and duplicate flags).
- `var`
- per-reference position masks and site-context columns used downstream (for example `position_in_<reference>` and `<reference>_<site_type>_site`).
- optional reindex columns from `reindex_references_adata` (suffix controlled by `reindexed_var_suffix`).
- `layers`
- binarized signal layer for direct modality (name controlled by `output_binary_layer_name`).
- NaN-cleaning strategy layers (for example `fill_nans_closest`, `nan0_0minus1`, `nan1_12`, `nan_minus_1`, `nan_half`).
- base-context/site-type derived binary layers used for downstream analyses and duplicate detection.
- `obsm`
- base-context level arrays written by context appending steps for downstream plotting/analysis.
- `uns`
- preprocessing stage metadata/flags plus auxiliary analysis outputs (for example UMI bipartite summaries and complexity-analysis summaries/fit outputs).

### `smftools variant`

The variant command focuses on DNA sequence variation analyses. It:

- Requires at least a preprocessed AnnData object.
- Calculates position level variation frequencies per reference/sample.
- Generates z-scores for variant occurance given read level Q-scores and assuming uniform Palt transitions.
- Visualizes read DNA sequence encodings and mismatch encodings.

General AnnData structures added by `variant`:

- `layers`
- `"{seq1_col}__{seq2_col}_variant_call"`: per-position variant call state (`1=seq1`, `2=seq2`, `0=unknown/no-coverage`, `-1=non-informative/non-mismatch`).
- `"{seq1_col}__{seq2_col}_variant_segments"`: segmented track per read span (`0=outside span`, `1=seq1 segment`, `2=seq2 segment`, `3=transition zone`).
- `var`
- `"{prefix}_seq1_acceptable_bases"` and `"{prefix}_seq2_acceptable_bases"`: accepted base sets used for variant matching at informative sites.
- `"{prefix}_informative_site"`: boolean mask of informative mismatch positions.
- `obs`
- `"{prefix}_breakpoint_count"` and `"{prefix}_is_chimeric"`: per-read breakpoint summary.
- `"{prefix}_variant_breakpoints"` and `variant_breakpoints`: list of inferred breakpoint positions per read.
- `chimeric_variant_sites` and `chimeric_variant_sites_type`: mismatch-segment chimera flags and categorical type labels.
- `"{prefix}_variant_segment_cigar"` and `variant_segment_cigar`: run-length string using `S` (self) and `X` (other).
- `"{prefix}_variant_self_base_count"` / `variant_self_base_count`: count of self-classified bases per read span.
- `"{prefix}_variant_other_base_count"` / `variant_other_base_count`: count of other-classified bases per read span.
- `uns`
- workflow completion flags (e.g., `append_variant_call_layer_performed`, `append_variant_segment_layer_performed`) and prior mismatch/substitution metadata used by variant calling.

### `smftools chimeric`

The chimeric command is meant to find putative PCR chimeras. It:

- Requires at least a preprocessed AnnData object.
- Performs sliding window nearest neighbor hamming distance analysis per read.
- Visualizes the windowed nearest neighbor hamming distances per read.
- Assembles maximum spanning intervals of 0-hamming distance neighbors per read within the reference/sample.

General AnnData structures added by `chimeric`:

- `obsm`
- `cfg.rolling_nn_obsm_key`: per-read rolling nearest-neighbor hamming distance tracks.
- `layers`
- `zero_hamming_distance_spans`: within-sample/reference top span mask derived from zero-distance partners.
- `cross_sample_zero_hamming_distance_spans`: top span mask from cross-sample pooling.
- `delta_zero_hamming_distance_spans`: clipped difference (`within - cross`) used for delta-based chimera evidence.
- `obs`
- `chimeric_by_mod_hamming_distance`: boolean flag based on longest positive delta span threshold.
- per-read top-segment tuple lists under keys like `"{rolling_nn_obsm_key}__top_segments"` (when top-segment extraction is enabled).
- `uns`
- rolling-distance and zero-pair metadata keyed by `rolling_nn_obsm_key`, including maps such as:
- `"{rolling_nn_obsm_key}_zero_pairs_map"` and `"{rolling_nn_obsm_key}_reference_map"`.
- optional stored segment records (`...__zero_hamming_segments`) and plotting metadata (`..._starts`, `..._window`, `..._step`, etc.), depending on cleanup settings in config.

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
