from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Final, Mapping, Tuple

## Operating-system metadata ##
# Files the OS or a file browser drops into directories without anyone asking.
# Merely opening a published generation's folder in macOS Finder creates a
# `.DS_Store` inside it; a directory checksum that counted such a file would
# report a validated, immutable artifact as corrupt for a reason having nothing
# to do with its contents. Content hashing therefore skips them.
OS_METADATA_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        ".DS_Store",  # macOS Finder
        "Thumbs.db",  # Windows Explorer
        "desktop.ini",  # Windows Explorer
        ".directory",  # KDE Dolphin
    }
)
# macOS resource-fork sidecars on non-HFS volumes, and Spotlight/trash state.
OS_METADATA_PREFIXES: Final[tuple[str, ...]] = ("._",)
OS_METADATA_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {".Spotlight-V100", ".Trashes", ".fseventsd", "__MACOSX"}
)


def is_os_metadata(path: Path) -> bool:
    """Return whether a path is operating-system metadata rather than content.

    Used by every directory content hash. Keeping one predicate means an
    artifact's digest cannot depend on which platform last browsed it.
    """
    if path.name in OS_METADATA_FILENAMES:
        return True
    if path.name.startswith(OS_METADATA_PREFIXES):
        return True
    return any(part in OS_METADATA_DIRECTORIES for part in path.parts)


## Helpers ##
def _deep_freeze(obj: Any) -> Any:
    """Recursively freeze common containers. Use for constant exports."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(_deep_freeze(v) for v in obj)
    return obj  # ints/strs/tuples (already immutable)


## Constants ##
BAM_SUFFIX: Final[str] = ".bam"
BARCODE_BOTH_ENDS: Final[bool] = False
SEMANTIC_GRAPH_DEFINITION_VERSION: Final[int] = 1
SEMANTIC_NODE_RESULT_SCHEMA_VERSION: Final[int] = 1
SEMANTIC_PLAN_SCHEMA_VERSION: Final[int] = 1
VARIANT_REFERENCE_SET_SCHEMA_VERSION: Final[int] = 1
VARIANT_INFORMATIVE_SITE_SCHEMA_VERSION: Final[int] = 1
VARIANT_CALL_SCHEMA_VERSION: Final[int] = 1
VARIANT_EVIDENCE_TASK_SCHEMA_VERSION: Final[int] = 1
VARIANT_EVIDENCE_INDEX_SCHEMA_VERSION: Final[int] = 1
VARIANT_EVIDENCE_GENERATION_SCHEMA_VERSION: Final[int] = 1
VARIANT_QC_METRICS_SCHEMA_VERSION: Final[int] = 1

# Barcode kit aliases for smftools demux backend
# Maps kit alias names to paths of barcode YAML files (relative to package data or absolute)
# Users can add custom aliases here or use "custom" with custom_barcode_yaml path
_private_barcode_kit_aliases: Dict[str, str] = {
    "SQK-NBD114-24": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-NBD114-24.yaml"
    ),
    "SQK-NBD114.24": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-NBD114-24.yaml"
    ),
    "SQK-NBD114-96": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-NBD114-96.yaml"
    ),
    "SQK-NBD114.96": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-NBD114-96.yaml"
    ),
    "SQK-RBK114-24": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-RBK114-24.yaml"
    ),
    "SQK-RBK114.24": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-RBK114-24.yaml"
    ),
    "SQK-RBK114-96": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-RBK114-96.yaml"
    ),
    "SQK-RBK114.96": str(
        Path(__file__).resolve().parent / "config" / "barcodes" / "SQK-RBK114-96.yaml"
    ),
}
BARCODE_KIT_ALIASES: Final[Mapping[str, str]] = _deep_freeze(_private_barcode_kit_aliases)

# UMI kit aliases for smftools UMI extraction
# Maps kit alias names to paths of UMI YAML config files
# Users can add custom aliases here or use "custom" with umi_yaml path
_private_umi_kit_aliases: Dict[str, str] = {
    "dual-nextera-12": str(
        Path(__file__).resolve().parent / "config" / "umis" / "dual-nextera-12.yaml"
    ),
}
UMI_KIT_ALIASES: Final[Mapping[str, str]] = _deep_freeze(_private_umi_kit_aliases)

REF_COL: Final[str] = "Reference_strand"
SAMPLE_COL: Final[str] = "Experiment_name_and_barcode"
SAMPLE: Final[str] = "Sample"
INFORMATICS_OUTPUTS_DIR: Final[str] = "informatics_outputs"
SPLIT_DIR: Final[str] = "demultiplexed_BAMs"
BAM_OUTPUTS_DIR: Final[str] = "bam_outputs"
MODKIT_OUTPUTS_DIR: Final[str] = "modkit_outputs"
FASTA_OUTPUTS_DIR: Final[str] = "fasta_outputs"
BED_OUTPUTS_DIR: Final[str] = "bed_outputs"
REGION_CATALOG_DIRNAME: Final[str] = "region_catalogs"
REFERENCE_INTERVAL_MAP_FILENAME: Final[str] = "reference_interval_map.parquet"
REGION_CATALOG_FILENAMES: Final[Mapping[str, str]] = _deep_freeze(
    {
        "alignment": "alignment_regions.parquet",
        "analysis": "analysis_regions.parquet",
        "plot": "plot_regions.parquet",
    }
)
H5_DIR: Final[str] = "h5ads"
DEMUX_TYPE: Final[str] = "demux_type"
BARCODE: Final[str] = "Barcode"
REFERENCE: Final[str] = "Reference"
REFERENCE_STRAND: Final[str] = "Reference_strand"
REFERENCE_DATASET_STRAND: Final[str] = "Reference_dataset_strand"
STRAND: Final[str] = "Strand"
DATASET: Final[str] = "Dataset"
READ_MISMATCH_TREND: Final[str] = "Read_mismatch_trend"
READ_MAPPING_DIRECTION: Final[str] = "Read_mapping_direction"
SEQUENCE_INTEGER_ENCODING: Final[str] = "sequence_integer_encoding"
SEQUENCE_INTEGER_DECODING: Final[str] = "sequence_integer_decoding"
MISMATCH_INTEGER_ENCODING: Final[str] = "mismatch_integer_encoding"
BASE_QUALITY_SCORES: Final[str] = "base_quality_scores"
READ_SPAN_MASK: Final[str] = "read_span_mask"
COVERED_BASE_MASK: Final[str] = "covered_base_mask"
MATE_COVERAGE_COUNT: Final[str] = "mate_coverage_count"
OVERLAP_CONFLICT_MASK: Final[str] = "overlap_conflict_mask"

LOAD_DIR: Final[str] = "load_adata_outputs"
BASECALL_DIR: Final[str] = "basecall_outputs"
RAW_DIR: Final[str] = "raw_outputs"
PREPROCESS_DIR: Final[str] = "preprocess_adata_outputs"
SPATIAL_DIR: Final[str] = "spatial_adata_outputs"
HMM_DIR: Final[str] = "hmm_adata_outputs"
LATENT_DIR: Final[str] = "latent_adata_outputs"
VARIANT_DIR: Final[str] = "variant_adata_outputs"
CHIMERIC_DIR: Final[str] = "chimeric_adata_outputs"
ML_EXPERIMENT_OUTPUTS_DIR: Final[str] = "ml_outputs"
ML_PROJECT_OUTPUTS_DIR: Final[str] = "ml"
ML_DATASETS_DIR: Final[str] = "datasets"
ML_RUNS_DIR: Final[str] = "runs"
ML_MODELS_DIR: Final[str] = "models"
ML_INDEX_DIR: Final[str] = "index"

_private_partitioned_stage_required_artifacts = {
    "raw": (
        "spine",
        "ragged_store",
        "interval_catalog",
        "molecules",
        "molecule_index",
        "segments",
        "segment_index",
        "reference_interval_map",
        "manifest",
        "input_manifest_csv",
        "input_manifest_json",
        "input_resolution_report",
        "generation_spine",
        "generation_manifest",
        "generation",
        "current",
    ),
    "preprocess": (
        "spine",
        "generation_spine",
        "store",
        "task_catalog",
        "read_index",
        "catalog",
        "var",
        "obs",
        "stage_obs",
        "plot_catalog",
        "manifest",
        "generation_manifest",
        "generation",
        "current",
    ),
    "spatial": (
        "spine",
        "generation_spine",
        "task_catalog",
        "read_index",
        "metrics",
        "autocorrelation",
        "task_store",
        "region_catalog",
        "plot_catalog",
        "manifest",
        "generation_manifest",
        "generation",
        "current",
    ),
    "hmm": (
        "spine",
        "generation_spine",
        "task_catalog",
        "read_index",
        "store",
        "models",
        "plot_catalog",
        "manifest",
        "generation_manifest",
        "generation",
        "current",
    ),
    "latent": (
        "spine",
        "generation_spine",
        "task_catalog",
        "read_index",
        "store",
        "plot_catalog",
        "manifest",
        "generation_manifest",
        "resource_plan",
        "generation",
        "current",
        "models",
    ),
}
PARTITIONED_STAGE_REQUIRED_ARTIFACTS: Final[Mapping[str, tuple[str, ...]]] = _deep_freeze(
    _private_partitioned_stage_required_artifacts
)

_private_partitioned_stage_nonempty_directories = {
    "raw": ("ragged_store",),
    "preprocess": ("store",),
    "spatial": ("task_store",),
    "hmm": ("store", "models"),
    "latent": ("store", "models"),
}
PARTITIONED_STAGE_NONEMPTY_DIRECTORIES: Final[Mapping[str, tuple[str, ...]]] = _deep_freeze(
    _private_partitioned_stage_nonempty_directories
)

LOGGING_DIR: Final[str] = "logs"

TRIM: Final[bool] = False

_private_conversions = ["unconverted"]
CONVERSIONS: Final[list[str]] = _deep_freeze(_private_conversions)

_private_mod_list = ("5mC_5hmC", "6mA")
MOD_LIST: Final[tuple[str, ...]] = _deep_freeze(_private_mod_list)

_private_mod_map: Dict[str, str] = {"6mA": "6mA", "5mC_5hmC": "5mC"}
MOD_MAP: Final[Mapping[str, str]] = _deep_freeze(_private_mod_map)

_private_strands = ("bottom", "top")
STRANDS: Final[tuple[str, ...]] = _deep_freeze(_private_strands)

MODKIT_EXTRACT_TSV_COLUMN_CHROM: Final[str] = "chrom"
MODKIT_EXTRACT_TSV_COLUMN_REF_POSITION: Final[str] = "ref_position"
MODKIT_EXTRACT_TSV_COLUMN_MODIFIED_PRIMARY_BASE: Final[str] = "modified_primary_base"
MODKIT_EXTRACT_TSV_COLUMN_REF_STRAND: Final[str] = "ref_strand"
MODKIT_EXTRACT_TSV_COLUMN_READ_ID: Final[str] = "read_id"
MODKIT_EXTRACT_TSV_COLUMN_CALL_CODE: Final[str] = "call_code"
MODKIT_EXTRACT_TSV_COLUMN_CALL_PROB: Final[str] = "call_prob"

MODKIT_EXTRACT_MODIFIED_BASE_A: Final[str] = "A"
MODKIT_EXTRACT_MODIFIED_BASE_C: Final[str] = "C"
MODKIT_EXTRACT_REF_STRAND_PLUS: Final[str] = "+"
MODKIT_EXTRACT_REF_STRAND_MINUS: Final[str] = "-"

_private_modkit_extract_call_code_modified = ("a", "h", "m")
MODKIT_EXTRACT_CALL_CODE_MODIFIED: Final[tuple[str, ...]] = _deep_freeze(
    _private_modkit_extract_call_code_modified
)
_private_modkit_extract_call_code_canonical = ("-",)
MODKIT_EXTRACT_CALL_CODE_CANONICAL: Final[tuple[str, ...]] = _deep_freeze(
    _private_modkit_extract_call_code_canonical
)

MODKIT_EXTRACT_SEQUENCE_BASES: Final[tuple[str, ...]] = _deep_freeze(("A", "C", "G", "T", "N"))
MODKIT_EXTRACT_SEQUENCE_PADDING_BASE: Final[str] = "PAD"
_private_modkit_extract_base_to_int: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
    "PAD": 5,
}
MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT: Final[Mapping[str, int]] = _deep_freeze(
    _private_modkit_extract_base_to_int
)
_private_modkit_extract_int_to_base: Dict[int, str] = {
    value: key for key, value in _private_modkit_extract_base_to_int.items()
}
MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE: Final[Mapping[int, str]] = _deep_freeze(
    _private_modkit_extract_int_to_base
)

# Base substitution a deamination/conversion chemistry writes, keyed by
# (modification type, strand of the molecule that was converted).
#
# Single source of truth. `informatics.fasta_functions._convert_FASTA_record`
# builds converted references from it, and `preprocessing.variant_reference`
# uses it to decide which bases a read could legitimately show at a reference
# position -- the two must never disagree, or a site excluded from variant
# calling would still be converted in the reference it is called against.
_private_conversion_base_substitutions: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("5mC", "top"): ("C", "T"),
    ("5mC", "bottom"): ("G", "A"),
    ("6mA", "top"): ("A", "G"),
    ("6mA", "bottom"): ("T", "C"),
}
CONVERSION_BASE_SUBSTITUTIONS: Final[Mapping[Tuple[str, str], Tuple[str, str]]] = _deep_freeze(
    _private_conversion_base_substitutions
)


#: Default cap on reads rendered into a single clustermap panel.
#:
#: Mirrors ``ExperimentConfig.clustermap_max_reads_per_plot`` and the
#: ``default.yaml`` entry. Call sites read the knob off ``cfg`` and fall back to
#: this when the attribute is absent; keeping the fallback in one place stops
#: the three copies drifting apart, which is how the CLI silently stayed on the
#: old value once before (``F18``).
DEFAULT_CLUSTERMAP_MAX_READS_PER_PLOT: Final[int] = 10000
