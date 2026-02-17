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
- `fasta_regions_of_interest`: Optional BED file to subset the FASTA.
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
