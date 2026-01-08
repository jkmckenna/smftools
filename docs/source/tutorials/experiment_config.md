# Experiment configuration CSV

smftools uses an experiment configuration CSV to define paths, modality settings, and workflow
options. You can start from the repository template (`experiment_config.csv`) and fill in your
experiment-specific values.

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

```csv
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
- `barcode_kit`: Demultiplexing configuration for barcoded nanopore experiments.
- `mapping_threshold`: Minimum mapping proportion per reference required for downstream steps.
- `mod_list`: Modification calls to use for direct-modality workflows.
- `conversion_types`: Target modification types for conversion workflows.

## Tips

- Keep paths absolute whenever possible to avoid ambiguity.
- Lists are written in bracketed form, e.g. `[5mC]` or `[5mC_5hmC]`.
- If you update the CSV, re-run the CLI command pointing at the updated file.