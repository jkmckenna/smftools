# Command line tutorials

## Quick start

Experiment-scoped commands (everything that takes a single experiment config) live under
`smftools experiment`. Project-scoped commands (everything that spans multiple registered
experiments) live under `smftools project`. Anything outside those two groups (e.g.
`subsample-pod5`) is a standalone utility with no experiment/project config of its own.

Most CLI workflows start with an experiment configuration CSV that points to your data, FASTA, and
output directory. Once the configuration is ready, you can run commands such as:

```shell
smftools experiment raw /path/to/experiment_config.csv
# Optional: pre-build the dense reference-grid cache.
smftools experiment load /path/to/experiment_config.csv
smftools experiment preprocess /path/to/experiment_config.csv
smftools experiment full /path/to/experiment_config.csv
smftools experiment batch raw /path/to/config_paths.csv
```

Each command creates or reuses stage-specific artifacts in the output directory. Later commands
reuse results from earlier stages unless you explicitly force a redo via configuration flags.

## What each command does

### `smftools experiment raw`

The raw command prepares sequencing inputs and writes the read-relative source of truth. It:

- Handles input formats (fast5/pod5/fastq/bam).
- Performs basecalling, alignment, demultiplexing, and BAM QC.
- Optionally generates BED/bigWig outputs for alignment summaries.
- Extracts one parquet row per physical read with CIGAR and query-coordinate signal arrays.
- Stores sequence, base-quality, mismatch, and modification values as Arrow list columns.
- Writes a thin molecule-index `spine.h5ad` containing identity and artifact pointers.
- Adds read-level QC, UMI, and barcode metadata without constructing padded dense matrices,
  including per-read `max_insertion_length`/`max_deletion_length` (longest internal indel from the
  alignment CIGAR), later consumed by `preprocess`'s CIGAR-indel filter.
- For `smf_modality: deaminase`, derives per-read C->T/G->A strand-switch metrics
  (`ct_event_count`, `ga_event_count`, `strand_segment_purity`, `strand_switch_position`) from the
  alignment CIGAR and writes a reference x barcode PCR-chimera-rate QC heatmap (`.png` + `.csv`)
  unless `bypass_raw_chimera_rate_plot` is set.
- Runs MultiQC while preserving the aligned BAM as a source artifact.

All artifacts produced by this command are grouped under `raw_outputs/`.

### `smftools experiment load`

The load command is optional in v2. It runs `raw` when needed, densifies one reference at a time,
and persists a barcode-ordered zarr cache. Downstream accessors use this cache when present and
otherwise densify the requested slice directly from ragged parquet.
The cache, catalog, and dense-index spine are stored under `load_adata_outputs/`.

### `smftools experiment preprocess`

The preprocess command performs QC, binarization, filtering, and duplicate detection. It:

- Requires an Anndata created by `smftools experiment load`.
- Loads sample sheet metadata (if provided).
- Generates read length/quality QC plots and filters reads on these metrics.
- Filters reads whose longest internal insertion/deletion (from the alignment CIGAR) exceeds
  `max_internal_insertion_length`/`max_internal_deletion_length` (default 10bp each; set either to
  `null` to disable, or `bypass_filter_reads_on_cigar_indels: True` to skip the step).
- Binarizes direct-modification calls based on thresholds (hard or fit thresholds).
- Cleans NaNs from adata.X and stores in adata.layers (nan0_0minus1, nan_half).
- Computes positional coverage and base-context annotations (GpC, CpG, ambiguous, other C, any C).
- Calculates read modification statistics and QC plots.
- Filters reads based on modification thresholds.
- Adds base-context binary modification layers.
- For `smf_modality: deaminase`, labels reads whose deamination signature switches from a C->T
  span to a G->A span partway through the read (evidence of a PCR chimera) in
  `obs["deaminase_PCR_chimera"]`. Reads are labeled, not removed. Controlled by
  `deaminase_chimera_min_events_per_span`, `deaminase_chimera_min_segment_purity`,
  `deaminase_chimera_max_single_strand_fraction`, and `bypass_label_deaminase_pcr_chimeras`.
- Optionally inverts and reindexes the data along the var (positions) axis.
- Flags duplicate reads based on nearest neighbor hamming distance of overlapping valid sites (Conversion/deamination).
- Performs complexity analyses using duplicate read clusters and Lander/Waterman fits (conversion/deamination workflows).
- Visualizes read span masks and base quality scores with clustermaps.
- Writes preprocessed (duplicates flagged, but kept) and preprocessed/deduplicated AnnData outputs.

General AnnData structures added by `preprocess`:

- `obs`
- sample-sheet metadata columns (when `sample_sheet_path` is provided) mapped by `sample_sheet_mapping_column`.
- read-level QC/ratio/modification summary fields used for filtering and plotting (for example read length/quality/mapping and fraction-modified metrics).
- optional UMI preprocessing fields when `use_umi=True`, including validity and clustering annotations (for example `U1_valid`, `U2_valid`, `U1_cluster`, `U2_cluster`, `RX_cluster`).
- optional UMI bipartite graph annotations and dominance metrics (for example edge-count/dominant-pair style fields) plus group-level stats in `uns`.
- duplicate-detection fields for non-direct modalities (for example merged cluster IDs/sizes and duplicate flags).
- `deaminase_PCR_chimera` (deaminase modality only): boolean flag for reads whose deamination signature switches strand partway through (see above).
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

### `smftools experiment variant`

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

### `smftools experiment chimeric`

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

### `smftools experiment spatial`

The spatial command runs downstream spatial analyses on the preprocessed data. It:

- Requires at least a preprocessed AnnData object.
- Automatically uses the partitioned preprocessing spine when one is available.
- Filters tasks to `passes_dedup` (or `passes_qc` when deduplication is unavailable).
- Dispatches non-empty reference/window/barcode chunks under the configured memory target.
- Writes linked Parquet metrics, barcode autocorrelation, position-profile sidecars, and a thin
  spatial spine without copying preprocessing matrices.
- Generates full-reference barcode clustermaps and position matrices for every locus-mode
  reference strand.
- In genome mode, `spatial_regions_bed` selects the 0-based, half-open regions used for dense
  clustermap and position-matrix products; native chromosome names apply to both strands.
- `alignment_regions_bed`, `analysis_regions_bed`, and `plot_regions_bed` are separate
  original-FASTA coordinate scopes. The raw stage normalizes configured BED3-BED6 files into
  versioned catalogs and publishes `reference_interval_map.parquet`. Partition-aware compute
  stages inherit the analysis catalog and share its authoritative cores. `spatial_regions_bed`
  retains its existing spatial-only behavior when no analysis catalog is configured.
- Plot intervals may span any number of adjacent analysis cores. They are stitched in coordinate
  order without halo duplication, and each plot-catalog row links to a JSON source manifest.
- Saves per-read ACF arrays and lag-pair counts in `read_metrics.zarr` `obsm` entries inside each
  general spatial task-store partition.
- Runs direct-signal Lomb-Scargle per read and site type, saving the normalized periodogram,
  nucleosome-repeat-length peak (`ls_nrl_bp`), peak power, raw peak power, SNR, FWHM, and scoring
  status in the same partition's `obsm` and `obs` entries.
- Stores lag and frequency/period coordinates in partition `uns`, with shared stage axis Parquets
  for catalog-level access.
- Plots barcode-stratified peak metrics and aligned bulk/barcode-mean Lomb-Scargle periodograms in
  the spatial `periodicity` category when `spatial_plot_read_lomb_scargle` is enabled.
- Plots per-read ACF and periodogram clustermaps under category-specific `read_clustermaps`
  directories when `spatial_plot_read_metric_clustermaps` is enabled.
- Optionally loads sample sheet metadata.
- Optionally inverts and reindexes the data along the positions axis.
- Generates clustermaps for preprocessed (and deduplicated) AnnData.
- Computes spatial autocorrelation, rolling metrics, and grid summaries.
- Generates positionwise correlation matrices.
- Writes the spatial AnnData output.

The optional dense-only analyses in the remaining bullets apply to
`spatial_execution_mode: legacy`. Use `auto` (default), `partitioned`, or `legacy` to choose the
execution path explicitly.

### `smftools experiment hmm`

The hmm command adds HMM-based feature annotation and summary plots. It:

- Requires at least a preprocessed AnnData object.
- Fits or reuses HMM models for configured feature sets.
- Annotates AnnData with HMM-derived feature layers (State layers and probability layers)
- Calls HMM feature peaks and writes peak-calling outputs.
- Generates clustermaps, bulk feature traces, and fragment size distribution plots for HMM layers.
- Writes the HMM AnnData output.

### `smftools experiment latent`

The latent command constructs latent representations of the data. It:

- Requires at least a preprocessed AnnData object.
- In `auto` mode, prefers partitioned HMM, spatial, and preprocessing spines in that order.
- Fits independent coordinate systems per reference locus or genome core. Coordinates and
  components from different units are not directly comparable.
- Bounds model fitting with `latent_max_fit_reads`; PCA, UMAP, and NMF project remaining reads in
  chunks, while CP runs only when the complete unit fits that bound.
- Runs various dimensionality reduction and graph construction modalities:
    - Principle component analysis (PCA)
    - K-nearest neighbor (KNN)
    - Uniform manifold approximation and projection (UMAP)
    - Non-negative matrix factorization (NMF)
    - Canonical polyadic decomposition (PARAFAC)
- Writes a task catalog, per-unit Zarr outputs, plot catalog, and thin latent spine under
  `latent_adata_outputs`.

`latent_execution_mode` accepts `auto`, `partitioned`, or `legacy`. The latent command remains a
standalone stage and is not run by `smftools experiment full`.

Migration note: existing configs require no new rows because all latent settings have defaults.
Set `latent_execution_mode,legacy` explicitly to retain monolithic output selection when both
legacy AnnData files and partitioned spines are present.

### `smftools experiment full`

The full command is a workflow wrapper. It runs the following sequentially:

- `raw`
- `preprocess`
- `spatial`
- `hmm`

Each stage uses its normal output discovery and force-redo settings. With `hmm_execution_mode:
auto`, a partitioned spatial or preprocessing spine dispatches bounded HMM tasks by reference,
genomic core/halo, barcode, and read chunk. HMM layers are stored in task Zarr groups and linked by
a thin HMM spine rather than materializing the full experiment.

### `smftools project`

The project command group manages a lightweight cross-experiment registry (`init`/`add`/`remove`/
`list`/`materialize`/`sample-store-list`). A project never copies or merges experiment data -- it
keeps pointers to each experiment's run directory plus a table harmonizing reference names across
experiments by sequence identity, so the same locus can be addressed by one canonical name even if
experiments called it something different.

- `project init PROJECT_DIR` creates the registry (`registry.json`) and a `sets/` directory for
  named experiment sets, plus starter working directories (`project_scripts/`, `project_outputs/`)
  and docs (`README.md`, `AGENTS.md`, `CLAUDE.md`, `PLAN.md`, `project.yaml`) -- none of which
  smftools reads back, they just give the project directory a useful starting point. `--name`
  sets the name used in the scaffolded docs (default: the directory name). Safe to re-run: never
  overwrites anything that already exists.
- `project add PROJECT_DIR EXPERIMENT_DIR` registers an experiment by pointer. `EXPERIMENT_DIR` may
  be the run's top-level output directory or one stage directory inside it (e.g. `raw_outputs/`) --
  either way, every pipeline stage spine found (`raw`, `preprocess`, `spatial`, `hmm`, ...) is
  recorded, not just one. Reads the raw spine's `uns` metadata (modality, sequence-hash reference
  identities) -- no matrices are opened. Reports any reference-name conflicts detected against
  already-registered experiments. `EXPERIMENT_DIR` may also be a single legacy monolithic
  `.h5ad`/`.h5ad.gz` file predating the partitioned-store pipeline -- pass `--stage` to name which
  pipeline stage it represents (guessed from the filename if omitted); repeated calls with the same
  `--id` for different legacy stage files accumulate onto one registry entry instead of replacing
  it, and the source file is only ever read, never modified (reference identity is computed on the
  fly instead of being cached back into it). See
  [Registering legacy (pre-partitioned-store) runs](directory_organization.md#registering-legacy-pre-partitioned-store-runs).
- `project list PROJECT_DIR` lists registered experiments (including which stages each has
  reached) and the harmonized reference table.
- `project materialize PROJECT_DIR CANONICAL_REFERENCE -o OUTPUT.h5ad.gz` resolves the canonical
  reference back to each matching experiment's own reference name(s), materializes each
  experiment's slice independently, and concatenates them (adding an `obs["experiment"]` column) --
  there is never a global merge across experiments. Each experiment's spine is picked
  independently: `--stage` requests a specific pipeline stage (skipping experiments that haven't
  reached it); the default falls back through the most-derived stage available per experiment
  (HMM > spatial > preprocess > raw), since a later stage's spine already carries forward
  everything earlier stages produced. `--read-metrics` additionally attaches spatial's per-read
  outputs (autocorrelation, Lomb-Scargle) where available. Supports `--set`/`--modality` filters
  and `--start`/`--end` genomic windows. Results are cached under `project_outputs/sets/` keyed by
  the query's *resolved* composition (which experiments/stages/spines it currently resolves to) --
  a repeat of the same query is a cache read, and registering or re-registering an experiment
  automatically invalidates any cache whose resolved membership that changes. `--force-recompute`
  skips a cache hit outright (still refreshes the cache afterward).
- `project remove PROJECT_DIR EXPERIMENT_ID` marks an experiment inactive (soft delete; the
  registry is append-only).
- `project sample-store-list PROJECT_DIR [--experiment-id ID]` lists the per-sample store's
  cataloged `(Reference_strand, sample)` partitions -- populated automatically by `project add`.
  Modern (partitioned-store) experiments get a `pointer` entry (no data copied, resolved through
  the registry at read time); legacy monolithic experiments get a `cache` entry (their molecules
  are cached once at registration, since they have no lazy read path otherwise).

`smftools project export-fastq ...` builds on the same registry.

### `smftools experiment export-fastq` / `smftools project export-fastq`

Writes one FASTQ (`.fastq.gz` by default) per barcode of QC-passed reads, for a single experiment
or an entire project -- available under both hierarchies, sharing the same underlying export logic.
Sequence and quality are read directly from the raw ragged store, so no BAM re-parsing is needed.

```shell
smftools experiment export-fastq /path/to/experiment_config.csv --outdir /path/to/fastq_outdir
smftools project export-fastq /path/to/project_dir --outdir /path/to/fastq_outdir
```

- `smftools experiment export-fastq` resolves the QC-passed read set from the most complete
  preprocessing artifact available, in priority order: the partitioned preprocess spine's
  `passes_dedup` (falling back to `passes_qc` then `passes_read_qc`), then the legacy deduplicated
  AnnData's read set, then the legacy QC-filtered AnnData's read set. Raises unless
  `--allow-unfiltered` is passed if none of these are available.
- `--group-by` (experiment command only) overrides the grouping obs column (default:
  `sample_name_col_for_plotting`, falling back to `Sample` then `Barcode`).
- `smftools project export-fastq` namespaces output filenames as
  `<experiment_id>__<barcode>.fastq.gz` and only includes experiments that have run partitioned
  preprocessing (others are skipped with a warning unless `--allow-unfiltered`); `--experiments`
  restricts to an explicit comma-separated id list.
- `--no-gzip` writes plain `.fastq`. A `fastq_manifest.csv` (barcode, read count, path) is written
  alongside the FASTQs.


## Batch processing

Use the batch command to run a single task across multiple experiments.

```shell
smftools experiment batch preprocess /path/to/config_paths.csv
```

The batch command accepts:

- **CSV/TSV** tables with a column of config paths (default column name: `config_path`).
- **TXT** files with one config path per line.

You can override the column name or delimiter if needed:

```shell
smftools experiment batch spatial /path/to/configs.tsv --column my_config --sep $'\t'
```

Each path is validated; missing configs are skipped with a message, while valid configs run the
requested task in sequence.
