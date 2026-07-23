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
experiment_name,,An experiment name for the final h5ad file,,str
```

## Common fields

Below are some of the most commonly edited fields and how they affect the CLI workflows:

- `smf_modality`: Defines whether the data is `conversion`, `direct` or `deaminase`, which determines
  preprocessing and HMM feature handling.
- `input_data_path`: Location of raw input data (fast5/pod5/fastq/bam).
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
- `experiment_name`: Base name used for output AnnData files.
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
- `spatial_position_matrix_max_width`: Hard position-count limit for a dense position-by-position
  spatial product. The default is 5,000 positions.
- `spatial_position_matrix_max_mb`: Hard estimated-memory limit for all position matrices retained
  for one spatial plot region. The default is 1,024 MiB. This limit is checked together with the
  live workflow ceiling before matrix allocation.

Existing configurations do not require migration: omitted settings inherit their defaults. CPU
utilization and the number of threads currently active elsewhere on a shared machine are
intentionally not used as hard limits because they are transient. Currently available memory is
included in the startup envelope. The resolved values and enforcement mode are written to stage
and performance logs. Linux reports whether a cgroup-v2 cap was activated; macOS and Windows
report worker-watchdog capability explicitly.

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
