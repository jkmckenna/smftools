# Experiment configuration CSV

smftools uses an experiment configuration CSV to define paths, modality settings, and workflow
options. You can start from the repository template (`experiment_config.csv`) and fill in your
experiment-specific values. The configuration CSV can override any parameter within the default.yaml
and modality specific config .yamls found within the config subpackage of smftools.

## CSV format

The configuration CSV is a table with the following columns:

| Column | Description |
| --- | --- |
| `variable` | Configuration key name (used by smftools). |
| `value` | Your value for this key. |
| `help` | Short description of the key. |
| `options` | Expected values (when applicable). |
| `type` | Expected value type (`str`, `int`, `float`, `list`). |

A shortened example looks like:

```text
variable,value,help,options,type
smf_modality,conversion,Modality of SMF. Can either be conversion or direct.,"conversion, direct",str
input_data_path,/path_to_POD5_directory,Path to directory/file containing input sequencing data,,str
fasta,/path_to_fasta.fasta,Path to initial FASTA file,,str
output_directory,/outputs,Directory to act as root for all analysis outputs,,str
experiment_id,my_experiment,Canonical experiment identity,,str
experiment_name,my_experiment,Compatibility alias; must equal experiment_id,,str
```

## Common fields

Below are some of the most commonly edited fields and how they affect the CLI workflows:

- `smf_modality`: Defines whether the data is `conversion`, `direct` or `deaminase`, which determines
  preprocessing and HMM feature handling.
- `input_data_path`: Location of raw input data (fast5/pod5/fastq/bam).
- `input_manifest_path`: Optional schema-1 CSV declaring the exact input files and metadata, or a
  schema-1 `bundle_manifest.json` produced by `experiment/project export-bundle`. Configure this
  or `input_data_path`, never both. Relative paths and bundle checksums are resolved from the
  manifest's directory.
- `alignment_mode`: Alignment policy. `align` is the default and preserves existing behavior,
  including realigning a supplied BAM. `existing` validates and owns one aligned BAM without
  changing its alignment placements.
- `fasta`: Reference FASTA for alignment and positional context.
- `alignment_regions_bed`: Optional original-FASTA BED file that restricts the alignment
  reference universe.
- `analysis_regions_bed`: Optional original-FASTA BED file defining shared downstream analysis
  scope. Preprocess, spatial, HMM, latent, and shared stage inputs inherit its normalized catalog.
- `plot_regions_bed`: Optional original-FASTA BED file defining presentation-only intervals. The
  catalog is published independently of compute scope and downstream plots stitch every completed
  analysis core that overlaps each interval.
- `plot_allow_unanalyzed_gaps`: Defaults to `False`, causing plot generation to fail when a
  requested interval is not fully covered by completed cores. Set it to `True` to retain and label
  those positions as `NaN`.
- `plot_subsample_seed`: Non-negative seed for deterministic per-barcode plot subsampling.
- `output_directory`: Root output folder for all generated AnnData files and plots.
- `experiment_id`: Canonical human-readable experiment identity. Set this explicitly; it is
  persisted separately from the opaque, immutable `experiment_uid`.
- `experiment_name`: Compatibility alias used for output filenames. Existing configs that set only
  `experiment_name` continue to work and promote it to `experiment_id`, but when both fields are
  present they must match. If neither is set, smftools derives the id from `output_directory` and
  emits a warning; the former date-based `YYMMDD_SMF_experiment` default is no longer used.
- `model_dir` / `model`: Dorado basecalling model configuration (nanopore runs).
- `demux_backend`: Demultiplexing backend (`dorado` or `smftools`).
- `barcode_kit`: Barcode kit name. Required for `dorado`; for `smftools`, use either a known alias or
  `custom` plus `custom_barcode_yaml`.
- `custom_barcode_yaml`: Barcode reference YAML path used when `demux_backend=smftools` and
  `barcode_kit=custom`.
- `use_umi` / `umi_yaml`: Optional UMI extraction controls. `umi_yaml` can define flanking-aware UMI
  extraction.
- `mapping_threshold`: Minimum mapping proportion per reference required for downstream steps.
- `mod_list`: Modification calls to use for direct-modality workflows.
- `conversion_types`: Target modification types for conversion workflows.

## Input contract and migration

Input discovery fails before creating an output directory or invoking an external tool when the
source is ambiguous or unsupported:

- Directories must contain one recognized input kind. Mixed POD5, FAST5, FASTQ, BAM, SAM, CRAM,
  or H5AD collections are rejected with per-kind counts instead of silently selecting one kind.
- Implicit BAM/CRAM directories remain rejected. Supply one alignment file or an explicit input
  manifest whose rows declare compatible BAM/CRAM source partitions and namespaces.
- SAM input remains unsupported. CRAM is supported only with `alignment_mode: existing` and must
  carry reference MD5 values matching the exact prepared FASTA.
- `aligner` must resolve to `dorado`, `minimap2`, `bwa-mem2`, or `bowtie2`. Common minimap2,
  BWA-MEM2, and Bowtie2 aliases normalize to those canonical names.
- Direct-modification experiments cannot use FASTQ because sequence-only FASTQ cannot retain MM/ML
  modification probabilities. Use raw signal or a modification-tagged BAM.
- Generated alignments use a structured adapter registry. smftools requires Dorado 0.7.0 or newer,
  minimap2 2.24.0 or newer, BWA-MEM2 2.2.1 or newer, Bowtie2/Bowtie2-build 2.4.0 or newer, and
  samtools 1.10.0 or newer when the external samtools backend is selected. Executables are probed
  and adapter capabilities are checked before alignment staging.
- The minimap2 BAM-to-FASTQ route is sequence-only and is therefore rejected for `direct`
  experiments because it would discard MM/ML tags. Use Dorado for a tag-preserving generated
  alignment, or use `alignment_mode: existing` for an authoritative aligned, tagged BAM.
- BWA-MEM2 and Bowtie2 always consume canonical sequence-only FASTQ staging, so direct MM/ML
  workflows and `align_from_bam: true` are rejected for those adapters.

Existing configurations that omit `alignment_mode` continue to use `align`. Set
`alignment_mode: existing` for aligned BAM/CRAM input that must be validated and ingested without
realignment. Explicit manifests may declare multiple compatible source partitions. smftools checks
readability, primary records, sequence/quality/CIGAR availability,
paired flags, exact prepared-reference `@SQ` names/lengths/order, and direct-modification MM/ML
tags. It then normalizes each source independently into an immutable owned BAM and index; source
alignments and their indexes remain untouched. Namespace-scoped template collisions are checked
before extraction, and the partition streams are never concatenated into one mandatory BAM. The flags
`input_already_demuxed`, `skip_bam_split`, and `align_from_bam` retain their existing meanings and
do not select existing-alignment ingestion.

Existing mode does not probe or invoke the configured aligner. The alignment manifest records
available BAM `@PG` provenance, or `unknown` when it is absent. Valid paired existing alignments
are retained with distinct R1/R2 segment identities and their proper-pair, singleton/discordant,
mate-reference, mate-position, orientation, and template-length metadata. Malformed paired flags
or missing mapped-mate coordinates fail with a distinct validation error. Singleton and
discordant records are reported but are not filtered.

Generated mode records the selected adapter, probed version, argument vector with path-independent
placeholders, declared capabilities, sort/index backend, and semantic reference identity in the
same schema-1 alignment manifest used by existing mode. Dorado and minimap2 currently build their
reference indexes in memory, so the identity is recorded for compatibility and restart decisions
rather than pointing to a persistent index artifact. BWA-MEM2 and Bowtie2 publish atomic,
content-addressed native indexes beneath the owned alignment workspace; the reference checksum,
adapter version, builder version, exact index files, and normalized index/alignment argv are part of
provenance. Minimap2, BWA-MEM2, and Bowtie2 accept synchronized paired FASTQ inputs through two
ordered mate streams; Dorado remains a single-BAM adapter. Paired FASTQ names
may use `/1`/`/2`, `_R1`/`_R2`, or CASAVA mate comments. Mismatched names, conflicting mate
annotations, and unequal record counts fail normalization. Mixed paired and unpaired FASTQ
collections require separate runs.

External conversion workflows must align against the exact transformed reference records that
smftools will validate. The public Python helper publishes that content-identified FASTA and its
manifest before alignment:

```python
from smftools.informatics.alignment_validation import prepare_alignment_reference_bundle

prepared_fasta, bundle_manifest = prepare_alignment_reference_bundle(
    "reference.fasta",
    "prepared-reference",
    modality="conversion",
    conversion_types=["unconverted", "5mC"],
    strands=["top", "bottom"],
)
```

### Canonical input manifest schema 1

An input manifest is a CSV with one source file per row. `path` is the only required column.
Supported optional columns are `source_kind`, `source_role`, `sample`, `barcode`, `read_group`,
`pair_id`, `mate`, `namespace`, `modification_capability`, and `trimmed`. A previously resolved
manifest may also include `source_id`, `sha256`, `size_bytes`, and `inferred_fields`; smftools
verifies declared identity fields against the current bytes.

For example:

```text
path,sample,barcode
reads/tumor_S1_L001_R1_001.fastq.gz,tumor,barcode01
reads/tumor_S1_L001_R2_001.fastq.gz,tumor,barcode01
```

Common Illumina/CASAVA and `R1`/`R2` filenames are paired automatically when
`fastq_auto_pairing` is enabled. Explicit CSV metadata takes precedence over filename inference.
Every pair must contain exactly one R1 and one R2 with compatible sample, barcode, read-group,
namespace, and source-role metadata.

Before external tools run, smftools streams each source through SHA-256 and rejects missing,
duplicate, unreadable, or concurrently modified inputs. Absolute source locations are excluded
from the canonical identity, so relocating an unchanged relative-path manifest and its files does
not change its digest. The raw task publishes `resolved_input_manifest.csv`,
`resolved_input_manifest.json`, and `input_resolution_report.json` under
`raw_outputs/input_manifest/`. A task-local SQLite cache avoids rehashing files whose complete
filesystem signature is unchanged; the content digest remains authoritative. Raw-stage records
created before these three artifacts existed are treated as incomplete and rebuilt on the next
raw request.

### Barcode and sample identity migration

Raw ingestion publishes one versioned barcode/sample identity sidecar for every input route. The
authority order is explicit manifest metadata, validated BAM `BC`/`RG`/`SM` metadata, configured
sequence classification, then legacy filename inference. Each raw molecule records the selected
barcode and sample together with its source, confidence, classification status, and any conflicting
lower-priority evidence. The paired JSON report records classified, unclassified, unknown, and
conflicting counts and fractions.

Filename inference remains compatible for legacy inputs, but now emits a warning when it supplies
the selected barcode. Add `barcode`, `sample`, `read_group`, and, when combining experiments,
`namespace` to the input manifest to make identity explicit. Mate tokens such as `R1` and `R2` are
removed before legacy inference and are never treated as barcodes. Already-demultiplexed runs may
keep `skip_bam_split: true`; barcode/sample metadata no longer depends on split-BAM filenames.

## Variant QC and migration

`variant_analysis_mode` accepts `auto`, `off`, `report`, or `filter`. Existing
configurations do not require migration: `auto` continues to select reporting
when both `references_to_align_for_variant_annotation` members are configured,
and `report` never removes a read.

See [](semantic_variant_workflows.md) for the reference-set and calling
contract, durable QC masks and cohorts, immutable generation lifecycle, and
standalone-variant migration boundary.

`filter` is intentionally opt-in and has no implicit biological thresholds. It
requires all of the following:

| Setting | Meaning |
| --- | --- |
| `variant_qc_min_callable_sites` | Minimum raw callable informative-site observations |
| `variant_qc_min_callable_fraction` | Minimum callable fraction of the reference-set informative sites, in `(0, 1]` |
| `variant_qc_min_calls_per_state` | Minimum raw calls supporting each state of a breakpoint |
| `variant_qc_disallowed_event_classes` | Non-empty list containing `breakpoint` and/or `ambiguous_reference_assignment` |

Filtering uses raw call counts rather than interpolated variant-segment lengths.
Insufficient or unavailable evidence remains diagnostic and passes variant QC.
A fully discordant read without a breakpoint is classified as
`ambiguous_reference_assignment`; it passes unless that class is explicitly
disallowed. Per-read indels are outside this initial variant-QC contract.
Duplicate detection considers all reads passing nonvariant QC and prefers a
variant-QC-pass member as the cluster keeper.

`smftools experiment variant` is retained only as a deprecated compatibility
alias. It requires `report` or `filter` mode and requests the same authoritative
preprocess generation as `smftools experiment preprocess`; standalone variant
H5AD existence and legacy `*_performed` flags do not establish compatibility.
The alias cannot upgrade a legacy deduplicated H5AD because filtered-out rows
are unavailable.

## Genome region scopes and migration

The three region fields are independent. Each accepts BED3 through BED6 using original FASTA,
0-based, half-open coordinates. smftools validates reference names and bounds, preserves optional
name, score, and strand fields, and writes versioned catalogs under `region_catalogs/`. Overlapping
and adjacent records remain separate and receive deterministic annotations and stable region IDs.
The raw stage also writes `reference_interval_map.parquet`, which maps reduced, conversion-state,
and stored strand references back to original FASTA coordinates.

Catalog normalization is deterministic:

| Input condition | Behavior |
| --- | --- |
| Blank lines, comments, `track`, or `browser` lines | Ignored |
| Overlapping records | Preserved separately and marked with `overlaps_previous` |
| Exactly adjacent records | Preserved separately and marked with `adjacent_previous` |
| BED name | Optional, but non-empty names must be unique within one catalog |
| BED score | Optional `.` or a finite number from 0 through 1000 |
| BED strand | Optional `.`, `+`, or `-`; it does not reverse coordinates |
| Empty analysis or plot BED | Published as a typed zero-row catalog |
| Empty alignment BED | Rejected because it would create an empty alignment reference |
| Invalid/missing reference, bounds, interval, name, score, or strand | Rejected with file and row context |

Records are sorted by original reference and coordinates without merging. Region IDs are derived
from normalized record content and therefore do not change when source rows are reordered. The
source-file SHA-256 remains available in Parquet metadata even for a zero-row catalog.

Analysis planning maps the catalog through `reference_interval_map.parquet`, unions overlapping
records, and splits the union on portable storage-tile boundaries. Every stage uses the same
non-overlapping authoritative cores and source region IDs. Stage-specific halos may extend loaded
context beyond a core, but only core positions are published. Changing `plot_regions_bed` does not
change this compute plan.

Plot generation maps each presentation interval back into stored coordinates, assembles adjacent
authoritative cores without repeating halo positions, and aligns rows by stable molecule identity.
Reads are selected from the derived index before arrays are loaded. Each registered stitched plot
links to a JSON source manifest containing the contributing task and artifact IDs, requested
layers, model IDs when applicable, and deterministic selection provenance.

`fasta_regions_of_interest` is a deprecated alias for `alignment_regions_bed`. Existing configs
continue to work with a warning. If both are supplied, they must identify the same path.
`spatial_regions_bed` remains a legacy spatial-only setting: it is not promoted to analysis or
plotting scope. Migrate it only when pipeline-wide analysis scope is actually intended.

## Resource limits

Resource settings are requests and ceilings, not guarantees that the requested capacity exists.
At command start, smftools resolves one resource envelope from the configuration and the local
machine or job allocation:

- `threads`: Requested CPU-worker ceiling. The resolved value cannot exceed the logical CPU count,
  process affinity, Linux cgroup CPU quota, or a recognized scheduler allocation (Slurm, PBS, SGE,
  or LSF).
- `max_memory_percent`: Maximum workflow memory as a percentage of physical RAM. It must be greater
  than zero and no more than 100.
- `max_memory_gb`: Optional fixed workflow-memory ceiling. When both memory settings are present,
  the more restrictive one applies.
- `memory_reserve_gb`: Memory retained outside the workflow after startup system, cgroup, and
  scheduler headroom are detected. The default is 1 GiB.
- `target_task_memory_mb`: Positive per-task planning estimate used to limit concurrent workers.
- `latent_max_fit_reads`: User ceiling for reads used to fit each partitioned latent unit. The
  effective count is the smaller of this value and the estimator's live-memory-safe count, but
  never below `latent_min_reads`.
- `latent_transform_chunk_reads`: User ceiling for each out-of-sample latent transform chunk. The
  executor recalculates its effective count from live headroom before the transform sequence.
- `latent_cp_memory_policy`: `skip` (default) records a structured reason and continues other
  enabled representations when complete-unit CP cannot fit; `fail` raises a resource error.
- `latent_plot_max_reads`: Positive plot-only ceiling for deterministic lazy row materialization.
  It does not reduce or change stored model outputs.
- `spatial_position_matrix_max_width`: Hard position-count limit for a dense position-by-position
  spatial product. The default is 5,000 positions.
- `spatial_position_matrix_max_mb`: Hard estimated-memory limit for all position matrices retained
  for one spatial plot region. The default is 1,024 MiB. This limit is checked together with the
  live workflow ceiling before matrix allocation.

Existing configurations do not require migration: omitted settings inherit their defaults,
including `latent_cp_memory_policy=skip` and `latent_plot_max_reads=10000`. CPU
utilization and the number of threads currently active elsewhere on a shared machine are
intentionally not used as hard limits because they are transient. Currently available memory is
included in the startup envelope. The resolved values and enforcement mode are written to stage
and performance logs. Linux reports whether a cgroup-v2 cap was activated; macOS and Windows
report worker-watchdog capability explicitly. Performance logging samples the complete process
tree independently of that enforcement mechanism, so sequential work and every supported OS emit
current/peak RSS and cumulative OS read/write byte counters as well as pool/task progress.
Partitioned latent generations additionally write `resource_plan.json` and task-catalog fields
containing estimator version, resource-envelope ID, requested and effective read counts, limiting
operation, CP skip reason, predicted peak, and measured process-tree peak.

## Tips

- Keep paths absolute whenever possible to avoid ambiguity.
- Lists are written in bracketed form, e.g. `[5mC]` or `[5mC_5hmC]`.
- If you update the CSV, re-run the CLI command pointing at the updated file.

## Read annotations

smftools annotates reads during `load_adata` and stores the results in `adata.obs`. Standard BAM
tags (e.g. `NM`, `MD`, `MM`, `ML`) are read directly from BAM files. UMI and barcode annotations
are computed in parallel and written to Parquet sidecar files alongside the aligned BAM, then loaded
into `adata.obs` from those sidecars. The aligned BAM itself is not modified.

**UMI annotations** (written to `.umi_tags.parquet`)

- `U1`: Orientation-corrected UMI for the *left* reference end of the mapped fragment (forward reads: US, reverse reads: UE).
- `U2`: Orientation-corrected UMI for the *right* reference end of the mapped fragment (forward reads: UE, reverse reads: US).
- `US`: Positional UMI from read start (delimited `UMI_seq;slot;flank_seq`).
- `UE`: Positional UMI from read end (delimited `UMI_seq;slot;flank_seq`).
- `RX`: Combined UMI string (`U1-U2`, or `U1`/`U2` if only one is present).
- `FC`: Flank context of the U1/U2 pair (e.g. `top-bottom`).

When `threads` is set, UMI extraction is parallelized across multiple CPU cores.

**Barcode annotations (smftools demux backend)** (written to `.barcode_tags.parquet`)

- `BC`: Assigned barcode name, or `unclassified`.
- `BM`: Match type (`both`, `read_start_only`, `read_end_only`, `mismatch`, `unclassified`).
- `B1`: Edit distance for the read-start barcode match.
- `B2`: Edit distance for the read-end barcode match.
- `B3`: Extracted barcode sequence from the read start (forward orientation).
- `B4`: Extracted barcode sequence from the read end (reverse-complemented to forward orientation).
- `B5`: Barcode name matched at the read start (corresponds to `B1`/`B3`).
- `B6`: Barcode name matched at the read end (corresponds to `B2`/`B4`).

When `threads` is set, barcode extraction is parallelized across multiple CPU cores.
Demultiplexing (splitting reads into per-barcode BAMs) uses the sidecar `BC` assignments.
Only primary alignments are included in split BAMs and sidecar files.

**Barcode annotations (dorado demux backend)**

- `BC`: Assigned barcode name (read from BAM tag).
- `bi`: Dorado barcode info array (if present; expanded into columns during load).

Notes:
- `BE`/`BF` are not used by smftools.

## Coordinate systems: reindexing and inversion

**Data is always stored in the reference coordinate frame.** Reindexing and
inversion define an alternative coordinate system for *display*; they never
change what is stored, what is analysed, or the order of any array. A run
configured with offsets produces byte-identical stores to one without.

### The transform

```
displayed = sign * (position + offset)        sign = -1 when inverted
```

Offset is applied first, then the sign, which keeps the anchor fixed under
inversion. With `reindexing_offsets: {my_ref: -3052}`, reference position 3052
becomes 0, and inverting mirrors the axis about that anchor:

| reference position | 944 | 2000 | 3052 | 3795 |
|---|---|---|---|---|
| offset only | -2108 | -1052 | **0** | 743 |
| offset + invert | 2108 | 1052 | **0** | -743 |

Position-dependent plots sort columns ascending by the displayed coordinate.
Because an inverted reference's values are already negated, ascending display
order is descending genomic order, so the rendered axis reverses without the
underlying array ever being reordered.

### Parameters

| parameter | meaning |
|---|---|
| `reindexing_offsets` | `{reference: int}`. Added to each position before the sign. Typically places a feature of interest at 0. |
| `reindexing_invert` | `True`/`False`, or `{reference: bool}` for per-reference control. Negates the displayed coordinate. |
| `reindexed_var_suffix` | Name suffix for the generated var column (default `reindexed`). Plots consume it; nothing else reads coordinates through it. |

Both are optional and independent: offsets without inversion re-zero the axis,
inversion without offsets mirrors about the reference origin.

### `invert_adata` is superseded — do not use it

`invert_adata` physically flipped `X` and every layer along the column axis. It
is a legacy CLI parameter, **is not honoured by the partitioned pipeline**, and
setting it has no effect on a modern run. It was superseded by
`reindexing_invert` in 2.12.0.

The distinction is the point: `invert_adata` changed the data, so an inverted
store and an uninverted store were different artifacts and could not be
compared. `reindexing_invert` changes only the projection, so the same store
serves every orientation and two runs differing only in display settings remain
directly comparable.

If you have a config carrying `invert_adata: True`, replace it with
`reindexing_invert: True`. Note that doing so changes the declared
configuration, so the next run will recompute rather than reuse — which is
correct, since the previous run's plots were not inverted at all.
