# experiment_config.py
from __future__ import annotations

import ast
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple, Union

from smftools.constants import (
    BAM_SUFFIX,
    BARCODE_BOTH_ENDS,
    CONVERSIONS,
    LOAD_DIR,
    MOD_LIST,
    MOD_MAP,
    REF_COL,
    SAMPLE_COL,
    SPLIT_DIR,
    STRANDS,
    TRIM,
)

from .discover_input_files import discover_input_files

# Optional dependency for YAML handling
try:
    import yaml
except Exception:
    yaml = None

import numpy as np
import pandas as pd


# -------------------------
# Utility parsing functions
# -------------------------
def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    try:
        return float(s) != 0.0
    except Exception:
        return False


def _parse_list(v: Any) -> List:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return []
    # try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # try python literal eval
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (list, tuple)):
            return list(lit)
    except Exception:
        pass
    # fallback comma separated
    s2 = s.strip("[]() ")
    parts = [p.strip() for p in s2.split(",") if p.strip() != ""]
    return parts


def _parse_numeric(v: Any, fallback: Any = None) -> Any:
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return fallback
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return fallback


def _try_json_or_literal(s: Any) -> Any:
    """Try parse JSON or python literal; otherwise return original string."""
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    s0 = s.strip()
    if s0 == "":
        return None
    # try json
    try:
        return json.loads(s0)
    except Exception:
        pass
    # try python literal
    try:
        return ast.literal_eval(s0)
    except Exception:
        pass
    return s


def resolve_aligner_args(
    merged: dict,
    default_by_aligner: Optional[Dict[str, List[str]]] = None,
    aligner_synonyms: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Resolve merged['aligner_args'] into a concrete list for the chosen aligner and sequencer.

    Behavior (search order):
      1. If aligner_args is a dict, try keys in this order (case-insensitive):
          a) "<aligner>@<sequencer>" (top-level combined key)
          b) aligner -> (if dict) sequencer (nested) -> 'default' fallback
          c) aligner -> (if list) use that list
          d) top-level 'default' key in aligner_args dict
      2. If aligner_args is a list -> return it (applies to any aligner/sequencer).
      3. If aligner_args is a string -> try parse JSON/literal or return single-element list.
      4. Otherwise fall back to builtin defaults per aligner.
    """
    # builtin defaults (aligner -> args)
    builtin_defaults = {
        "minimap2": ["-a", "-x", "map-ont", "--MD", "-Y", "-y", "-N", "5", "--secondary=no"],
        "dorado": ["--mm2-opts", "-N", "5"],
    }
    if default_by_aligner is None:
        default_by_aligner = builtin_defaults

    # synonyms mapping
    synonyms = {"mm2": "minimap2", "minimap": "minimap2", "minimap-2": "minimap2"}
    if aligner_synonyms:
        synonyms.update(aligner_synonyms)

    # canonicalize requested aligner and sequencer
    raw_aligner = merged.get("aligner", "minimap2") or "minimap2"
    raw_sequencer = merged.get("sequencer", None)  # e.g. 'ont', 'pacbio', 'illumina'
    key_align = str(raw_aligner).strip().lower()
    key_seq = None if raw_sequencer is None else str(raw_sequencer).strip().lower()
    if key_align in synonyms:
        key_align = synonyms[key_align]

    raw = merged.get("aligner_args", None)

    # helper to coerce a candidate to list[str]
    def _coerce_to_list(val):
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
        if isinstance(val, str):
            parsed = _try_json_or_literal(val)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
            return [str(parsed)]
        if val is None:
            return None
        return [str(val)]

    # If dict, do layered lookups
    if isinstance(raw, dict):
        # case-insensitive dict
        top_map = {str(k).lower(): v for k, v in raw.items()}

        # 1) try combined top-level key "aligner@sequencer"
        if key_seq:
            combined_key = f"{key_align}@{key_seq}"
            if combined_key in top_map:
                res = _coerce_to_list(top_map[combined_key])
                if res:
                    return res

        # 2) try aligner key
        if key_align in top_map:
            val = top_map[key_align]
            # if nested dict: try sequencer key then 'default'
            if isinstance(val, dict):
                submap = {str(k).lower(): v for k, v in val.items()}
                if key_seq and key_seq in submap:
                    res = _coerce_to_list(submap[key_seq])
                    if res:
                        return res
                if "default" in submap:
                    res = _coerce_to_list(submap["default"])
                    if res:
                        return res
                # nothing matched inside aligner->dict; fall back to top-level aligner (no sequencer)
            else:
                # aligner maps to list/str: use it
                res = _coerce_to_list(val)
                if res:
                    return res

        # 3) try top-level 'default' key inside aligner_args mapping
        if "default" in top_map:
            res = _coerce_to_list(top_map["default"])
            if res:
                return res

        # 4) last top-level attempt: any key equal to aligner synonyms etc (already handled)
        # fallthrough to builtin
    # If user provided a concrete list -> use it
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]

    # If scalar string, attempt to parse
    if isinstance(raw, str):
        parsed = _try_json_or_literal(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]

    # Nothing found -> fallback builtin default
    return list(default_by_aligner.get(key_align, []))


# HMM default params and helper functions
def normalize_hmm_feature_sets(raw: Any) -> Dict[str, dict]:
    """
    Normalize user-provided `hmm_feature_sets` into canonical structure:
      { group_name: {"features": {label: (lo, hi), ...}, "state": "<Modified|Non-Modified>"} }
    Accepts dict, JSON/string, None. Returns {} for empty input.
    """
    if raw is None:
        return {}
    parsed = raw
    if isinstance(raw, str):
        parsed = _try_json_or_literal(raw)
    if not isinstance(parsed, dict):
        return {}

    def _coerce_bound(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().lower()
        if s in ("inf", "infty", "infinite"):
            return np.inf
        if s in ("none", ""):
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _coerce_feature_map(feats):
        out = {}
        if not isinstance(feats, dict):
            return out
        for fname, rng in feats.items():
            if rng is None:
                out[fname] = (0.0, np.inf)
                continue
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                lo = _coerce_bound(rng[0]) or 0.0
                hi = _coerce_bound(rng[1])
                if hi is None:
                    hi = np.inf
                out[fname] = (float(lo), float(hi) if not np.isinf(hi) else np.inf)
            else:
                # scalar -> treat as upper bound
                val = _coerce_bound(rng)
                out[fname] = (0.0, float(val) if val is not None else np.inf)
        return out

    canonical = {}
    for grp, info in parsed.items():
        if not isinstance(info, dict):
            feats = _coerce_feature_map(info)
            canonical[grp] = {"features": feats, "state": "Modified"}
            continue
        feats = _coerce_feature_map(info.get("features", info.get("ranges", {})))
        state = info.get("state", info.get("label", "Modified"))
        canonical[grp] = {"features": feats, "state": state}
    return canonical


def normalize_peak_feature_configs(raw: Any) -> Dict[str, dict]:
    """
    Normalize user-provided `hmm_peak_feature_configs` into:
      {
        layer_name: {
          "min_distance": int,
          "peak_width": int,
          "peak_prominence": float,
          "peak_threshold": float,
          "rolling_window": int,
        },
        ...
      }

    Accepts dict, JSON/string, None. Returns {} for empty input.
    """
    if raw is None:
        return {}

    parsed = raw
    if isinstance(raw, str):
        parsed = _try_json_or_literal(raw)
    if not isinstance(parsed, dict):
        return {}

    defaults = {
        "min_distance": 200,
        "peak_width": 200,
        "peak_prominence": 0.2,
        "peak_threshold": 0.8,
        "rolling_window": 1,
    }

    out: Dict[str, dict] = {}
    for layer, conf in parsed.items():
        if conf is None:
            conf = {}
        if not isinstance(conf, dict):
            # allow shorthand like 300 -> interpreted as peak_width
            conf = {"peak_width": conf}

        full = defaults.copy()
        full.update(conf)
        out[str(layer)] = {
            "min_distance": int(full["min_distance"]),
            "peak_width": int(full["peak_width"]),
            "peak_prominence": float(full["peak_prominence"]),
            "peak_threshold": float(full["peak_threshold"]),
            "rolling_window": int(full["rolling_window"]),
        }
    return out


# -------------------------
# LoadExperimentConfig
# -------------------------
class LoadExperimentConfig:
    """
    Load an experiment CSV (or DataFrame / file-like) into a typed var_dict.

    CSV expected columns: 'variable', 'value', optional 'type'.
    If 'type' missing, the loader will infer type.

    Example
    -------
    loader = LoadExperimentConfig("experiment_config.csv")
    var_dict = loader.var_dict
    """

    def __init__(self, experiment_config: Union[str, Path, IO, pd.DataFrame]):
        self.source = experiment_config
        self.df = self._load_df(experiment_config)
        self.var_dict = self._parse_df(self.df)

    @staticmethod
    def _load_df(source: Union[str, Path, IO, pd.DataFrame]) -> pd.DataFrame:
        """Load a pandas DataFrame from path, file-like, or accept if already DataFrame."""
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            if isinstance(source, (str, Path)):
                p = Path(source)
                if not p.exists():
                    raise FileNotFoundError(f"Config file not found: {source}")
                df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
            else:
                # file-like
                df = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[""])
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        if "variable" not in df.columns:
            raise ValueError("Config CSV must contain a 'variable' column.")
        if "value" not in df.columns:
            df["value"] = ""
        if "type" not in df.columns:
            df["type"] = ""
        return df

    @staticmethod
    def _parse_value_as_type(value_str: Optional[str], dtype_hint: Optional[str]) -> Any:
        """
        Parse a single value string into a Python object guided by dtype_hint (or infer).
        Supports int, float, bool, list, JSON, Python literal, or string.
        """
        if value_str is None:
            return None
        v = str(value_str).strip()
        if v == "" or v.lower() == "none":
            return None

        hint = "" if dtype_hint is None else str(dtype_hint).strip().lower()

        def parse_bool(s: str):
            s2 = s.strip().lower()
            if s2 in ("1", "true", "t", "yes", "y", "on"):
                return True
            if s2 in ("0", "false", "f", "no", "n", "off"):
                return False
            raise ValueError(f"Cannot parse boolean from '{s}'")

        def parse_list_like(s: str):
            # try JSON first
            try:
                val = json.loads(s)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
            # try python literal
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple)):
                    return list(val)
            except Exception:
                pass
            # fallback split
            parts = [p.strip() for p in s.strip("()[] ").split(",") if p.strip() != ""]
            return parts

        if hint in ("int", "integer"):
            return int(v)
        if hint in ("float", "double"):
            return float(v)
        if hint in ("bool", "boolean"):
            return parse_bool(v)
        if hint in ("list", "array"):
            return parse_list_like(v)
        if hint in ("string", "str"):
            return v

        # infer
        try:
            return int(v)
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            pass
        try:
            return parse_bool(v)
        except Exception:
            pass
        try:
            j = json.loads(v)
            return j
        except Exception:
            pass
        try:
            lit = ast.literal_eval(v)
            return lit
        except Exception:
            pass
        if ("," in v) and (not any(ch in v for ch in "{}[]()")):
            return [p.strip() for p in v.split(",") if p.strip() != ""]
        return v

    def _parse_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for idx, row in df.iterrows():
            name = str(row["variable"]).strip()
            if name == "":
                continue
            raw_val = row.get("value", "")
            raw_type = row.get("type", "")
            if pd.isna(raw_val) or str(raw_val).strip() == "":
                raw_val = None
            try:
                parsed_val = self._parse_value_as_type(raw_val, raw_type)
            except Exception as e:
                warnings.warn(
                    f"Failed to parse config variable '{name}' (row {idx}): {e}. Storing raw value."
                )
                parsed_val = None if raw_val is None else raw_val
            if name in parsed:
                warnings.warn(
                    f"Duplicate config variable '{name}' encountered (row {idx}). Overwriting previous value."
                )
            parsed[name] = parsed_val
        return parsed

    def to_dataframe(self) -> pd.DataFrame:
        """Return parsed config as a pandas DataFrame (variable, value)."""
        rows = []
        for k, v in self.var_dict.items():
            rows.append({"variable": k, "value": v})
        return pd.DataFrame(rows)


# -------------------------
# deep merge & defaults loader (with inheritance)
# -------------------------
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts: returns new dict = a merged with b, where b overrides.
    If both values are dicts -> merge recursively; else b replaces a.
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_defaults_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML required to load YAML defaults (pip install pyyaml).")
        return yaml.safe_load(text) or {}
    elif suffix == ".json":
        return json.loads(text or "{}")
    else:
        # try json then yaml if available
        try:
            return json.loads(text)
        except Exception:
            if yaml is not None:
                return yaml.safe_load(text) or {}
            raise RuntimeError(f"Unknown defaults file type for {path}; provide JSON or YAML.")


def load_defaults_with_inheritance(
    defaults_dir: Union[str, Path],
    modality: Optional[str],
    *,
    default_basename: str = "default",
    allowed_exts: Tuple[str, ...] = (".yaml", ".yml", ".json"),
    debug: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Strict loader: only loads default + modality + any explicit 'extends' chain.

    - defaults_dir: directory containing defaults files.
    - modality: name of modality (e.g. "GpC"). We look for <modality>.<ext> in defaults_dir.
    - default_basename: name of fallback default file (without extension).
    - allowed_exts: allowed extensions to try.
    - debug: if True, prints what was loaded.

    Returns (merged_defaults_dict, load_order_list) where load_order_list are resolved file paths read.
    """
    pdir = Path(defaults_dir) if defaults_dir is not None else None
    if pdir is None or not pdir.exists():
        return {}, []

    # Resolve a "name" to a file in defaults_dir.
    # Only treat `name` as an explicit path if it contains a path separator or is absolute.
    def resolve_name_to_path(name: str) -> Optional[Path]:
        n = str(name).strip()
        if n == "":
            return None
        cand = Path(n)
        # If user provided a path-like string (contains slash/backslash or absolute), allow it
        if cand.is_absolute() or ("/" in n) or ("\\" in n):
            if cand.exists() and cand.suffix.lower() in allowed_exts:
                return cand.resolve()
            return None
        # Otherwise only look inside defaults_dir for name + ext (do NOT treat bare name as arbitrary file)
        for ext in allowed_exts:
            p = pdir / f"{n}{ext}"
            if p.exists():
                return p.resolve()
        return None

    visited = set()
    load_order: List[str] = []

    def _rec_load(name_or_path: Union[str, Path]) -> Dict[str, Any]:
        # Resolve to a file path (strict)
        if isinstance(name_or_path, Path):
            p = name_or_path
        else:
            p = resolve_name_to_path(str(name_or_path))
        if p is None:
            if debug:
                print(f"[defaults loader] resolve failed for '{name_or_path}'")
            return {}
        p = Path(p).resolve()
        p_str = str(p)
        if p_str in visited:
            if debug:
                print(f"[defaults loader] already visited {p_str} (skipping to avoid cycle)")
            return {}
        visited.add(p_str)

        data = _load_defaults_file(p)  # reuse your existing helper
        if not isinstance(data, dict):
            if debug:
                print(f"[defaults loader] file {p_str} did not produce a dict -> ignoring")
            data = {}

        # Extract any extends/inherits keys (string or list). They reference other named default files.
        bases = []
        for key in ("extends", "inherits", "base"):
            if key in data:
                b = data.pop(key)
                if isinstance(b, (list, tuple)):
                    bases = list(b)
                elif isinstance(b, str):
                    bases = [b]
                break

        merged = {}
        # Load bases first (in order); bases are resolved relative to defaults_dir unless given as path
        for base_name in bases:
            base_defaults = _rec_load(base_name)
            merged = deep_merge(merged, base_defaults)

        # Then merge this file's data (this file overrides its bases)
        merged = deep_merge(merged, data)
        load_order.append(p_str)
        if debug:
            print(f"[defaults loader] loaded {p_str}")
        return merged

    merged_defaults = {}
    # Load default.* first if present
    def_path = resolve_name_to_path(default_basename)
    if def_path is not None:
        merged_defaults = deep_merge(merged_defaults, _rec_load(def_path))

    # Load modality.* if present (modality overrides default)
    if modality:
        mod_path = resolve_name_to_path(modality)
        if mod_path is not None:
            merged_defaults = deep_merge(merged_defaults, _rec_load(mod_path))
        else:
            if debug:
                print(f"[defaults loader] no modality file found for '{modality}' in {pdir}")

    if debug:
        print("[defaults loader] final load order:", load_order)
    return merged_defaults, load_order


# -------------------------
# ExperimentConfig dataclass
# -------------------------
@dataclass
class ExperimentConfig:
    # Compute
    threads: Optional[int] = None
    device: str = "auto"

    # General I/O
    input_data_path: Optional[str] = None
    output_directory: Optional[str] = None
    emit_log_file: Optional[bool] = True
    log_level: Optional[str] = "INFO"
    fasta: Optional[str] = None
    bam_suffix: str = BAM_SUFFIX
    recursive_input_search: bool = True
    input_type: Optional[str] = None
    input_files: Optional[List[Path]] = None
    split_dir: str = SPLIT_DIR
    split_path: Optional[str] = None
    strands: List[str] = field(default_factory=lambda: STRANDS)
    conversions: List[str] = field(default_factory=lambda: CONVERSIONS)
    fasta_regions_of_interest: Optional[str] = None
    sample_sheet_path: Optional[str] = None
    sample_sheet_mapping_column: Optional[str] = "Experiment_name_and_barcode"
    experiment_name: Optional[str] = None
    input_already_demuxed: bool = False
    summary_file: Optional[Path] = None

    # FASTQ input specific
    fastq_barcode_map: Optional[Dict[str, str]] = None
    fastq_auto_pairing: bool = True

    # Remove intermediate file options
    delete_intermediate_bams: bool = False
    delete_intermediate_tsvs: bool = True

    # Conversion/Deamination file handling
    delete_intermediate_hdfs: bool = True

    # Direct SMF specific params for initial AnnData loading
    batch_size: int = 4
    skip_unclassified: bool = True
    delete_batch_hdfs: bool = True

    # Sequencing modality and general experiment params
    smf_modality: Optional[str] = None
    sequencer: Optional[str] = None

    # Enzyme / mod targets
    mod_target_bases: List[str] = field(default_factory=lambda: ["GpC", "CpG"])
    enzyme_target_bases: List[str] = field(default_factory=lambda: ["GpC"])

    # Conversion/deamination
    conversion_types: List[str] = field(default_factory=lambda: ["5mC"])

    # Nanopore specific for basecalling and demultiplexing
    model_dir: Optional[str] = None
    barcode_kit: Optional[str] = None
    model: str = "hac"
    barcode_both_ends: bool = BARCODE_BOTH_ENDS
    trim: bool = TRIM
    # General basecalling params
    filter_threshold: float = 0.8
    # Modified basecalling specific params
    m6A_threshold: float = 0.7
    m5C_threshold: float = 0.7
    hm5C_threshold: float = 0.7
    thresholds: List[float] = field(default_factory=list)
    mod_list: List[str] = field(
        default_factory=lambda: list(MOD_LIST)
    )  # Dorado modified basecalling codes
    mod_map: Dict[str, str] = field(
        default_factory=lambda: dict(MOD_MAP)
    )  # Map from dorado modified basecalling codes to codes used in modkit_extract_to_adata function

    # Alignment params
    mapping_threshold: float = 0.01  # Min threshold for fraction of reads in a sample mapping to a reference in order to include the reference in the anndata
    align_from_bam: bool = (
        False  # Whether minimap2 should align from a bam file as input. If False, aligns from FASTQ
    )
    aligner: str = "dorado"
    aligner_args: Optional[List[str]] = None
    make_bigwigs: bool = False
    make_beds: bool = False
    annotate_secondary_supplementary: bool = True
    samtools_backend: str = "auto"
    bedtools_backend: str = "auto"
    bigwig_backend: str = "auto"

    # Anndata structure
    reference_column: Optional[str] = REF_COL
    sample_column: Optional[str] = SAMPLE_COL

    # General Plotting
    sample_name_col_for_plotting: Optional[str] = "Barcode"
    rows_per_qc_histogram_grid: int = 12
    clustermap_demux_types_to_plot: List[str] = field(
        default_factory=lambda: ["single", "double", "already"]
    )

    # Preprocessing - Read length and quality filter params
    read_coord_filter: Optional[Sequence[float]] = field(default_factory=lambda: [None, None])
    read_len_filter_thresholds: Optional[Sequence[float]] = field(
        default_factory=lambda: [100, None]
    )
    read_len_to_ref_ratio_filter_thresholds: Optional[Sequence[float]] = field(
        default_factory=lambda: [0.4, 1.5]
    )
    read_quality_filter_thresholds: Optional[Sequence[float]] = field(
        default_factory=lambda: [15, None]
    )
    read_mapping_quality_filter_thresholds: Optional[Sequence[float]] = field(
        default_factory=lambda: [None, None]
    )

    # Preprocessing - Optional reindexing params
    reindexing_offsets: Dict[str, int] = field(default_factory=dict)
    reindexed_var_suffix: Optional[str] = "reindexed"

    # Preprocessing - Direct mod detection binarization params
    fit_position_methylation_thresholds: Optional[bool] = (
        False  # Whether to use Youden J-stat to determine position by positions thresholds for modification binarization.
    )
    binarize_on_fixed_methlyation_threshold: Optional[float] = (
        0.7  # The threshold used to binarize the anndata using a fixed value if fitting parameter above is False.
    )
    positive_control_sample_methylation_fitting: Optional[str] = (
        None  # A positive control Sample_name to use for fully modified template data
    )
    negative_control_sample_methylation_fitting: Optional[str] = (
        None  # A negative control Sample_name to use for fully unmodified template data
    )
    infer_on_percentile_sample_methylation_fitting: Optional[int] = (
        10  # If a positive/negative control are not provided and fitting the data is requested, use the indicated percentile windows from the top and bottom of the dataset.
    )
    inference_variable_sample_methylation_fitting: Optional[str] = (
        "Raw_modification_signal"  # The obs column value used for the percentile metric above.
    )
    fit_j_threshold: Optional[float] = (
        0.5  # The J-statistic threhold to use for determining which positions pass qc for mod detection thresholding
    )
    output_binary_layer_name: Optional[str] = "binarized_methylation"

    # Preprocessing - Read modification filter params
    read_mod_filtering_gpc_thresholds: List[float] = field(default_factory=lambda: [0.025, 0.975])
    read_mod_filtering_cpg_thresholds: List[float] = field(default_factory=lambda: [0.00, 1])
    read_mod_filtering_c_thresholds: List[float] = field(default_factory=lambda: [0.025, 0.975])
    read_mod_filtering_a_thresholds: List[float] = field(default_factory=lambda: [0.025, 0.975])
    read_mod_filtering_use_other_c_as_background: bool = True
    min_valid_fraction_positions_in_read_vs_ref: float = 0.2

    # Preprocessing - plotting params
    obs_to_plot_pp_qc: List[str] = field(
        default_factory=lambda: [
            "read_length",
            "mapped_length",
            "read_quality",
            "mapping_quality",
            "mapped_length_to_reference_length_ratio",
            "mapped_length_to_read_length_ratio",
            "Raw_modification_signal",
        ]
    )

    # Preprocessing - Duplicate detection params
    duplicate_detection_site_types: List[str] = field(
        default_factory=lambda: ["GpC", "CpG", "ambiguous_GpC_CpG"]
    )
    duplicate_detection_demux_types_to_use: List[str] = field(
        default_factory=lambda: ["single", "double", "already"]
    )
    duplicate_detection_distance_threshold: float = 0.07
    hamming_vs_metric_keys: List[str] = field(default_factory=lambda: ["Fraction_C_site_modified"])
    duplicate_detection_keep_best_metric: str = "read_quality"
    duplicate_detection_window_size_for_hamming_neighbors: int = 50
    duplicate_detection_min_overlapping_positions: int = 20
    duplicate_detection_do_hierarchical: bool = True
    duplicate_detection_hierarchical_linkage: str = "average"
    duplicate_detection_do_pca: bool = False

    # Preprocessing - Position QC
    position_max_nan_threshold: float = 0.1
    mismatch_frequency_range: Sequence[float] = field(default_factory=lambda: [0.05, 0.95])
    mismatch_frequency_layer: str = "mismatch_integer_encoding"
    mismatch_frequency_read_span_layer: str = "read_span_mask"
    mismatch_base_frequency_exclude_mod_sites: bool = False

    # Spatial Analysis - Clustermap params
    layer_for_clustermap_plotting: Optional[str] = "nan0_0minus1"
    clustermap_cmap_c: Optional[str] = "coolwarm"
    clustermap_cmap_gpc: Optional[str] = "coolwarm"
    clustermap_cmap_cpg: Optional[str] = "coolwarm"
    clustermap_cmap_a: Optional[str] = "coolwarm"
    spatial_clustermap_sortby: Optional[str] = "gpc"
    rolling_nn_layer: Optional[str] = "nan0_0minus1"
    rolling_nn_plot_layer: Optional[str] = "nan0_0minus1"
    rolling_nn_plot_layers: List[str] = field(
        default_factory=lambda: ["nan0_0minus1", "nan0_0minus1"]
    )
    rolling_nn_window: int = 10
    rolling_nn_step: int = 1
    rolling_nn_min_overlap: int = 8
    rolling_nn_return_fraction: bool = True
    rolling_nn_obsm_key: str = "rolling_nn_dist"
    rolling_nn_site_types: Optional[List[str]] = None
    rolling_nn_write_zero_pairs_csvs: bool = True
    rolling_nn_zero_pairs_uns_key: Optional[str] = None
    rolling_nn_zero_pairs_segments_key: Optional[str] = None
    rolling_nn_zero_pairs_layer_key: Optional[str] = None
    rolling_nn_zero_pairs_refine: bool = True
    rolling_nn_zero_pairs_max_nan_run: Optional[int] = None
    rolling_nn_zero_pairs_merge_gap: int = 0
    rolling_nn_zero_pairs_max_segments_per_read: Optional[int] = None
    rolling_nn_zero_pairs_max_overlap: Optional[int] = None
    rolling_nn_zero_pairs_layer_overlap_mode: str = "binary"
    rolling_nn_zero_pairs_layer_overlap_value: Optional[int] = None
    rolling_nn_zero_pairs_keep_uns: bool = True
    rolling_nn_zero_pairs_segments_keep_uns: bool = True
    rolling_nn_zero_pairs_top_segments_per_read: Optional[int] = None
    rolling_nn_zero_pairs_top_segments_max_overlap: Optional[int] = None
    rolling_nn_zero_pairs_top_segments_min_span: Optional[float] = None
    rolling_nn_zero_pairs_top_segments_write_csvs: bool = True
    rolling_nn_zero_pairs_segment_histogram_bins: int = 30

    # Cross-sample rolling NN analysis
    cross_sample_analysis: bool = False
    cross_sample_grouping_col: Optional[str] = None
    cross_sample_random_seed: int = 42

    # Spatial Analysis - UMAP/Leiden params
    layer_for_umap_plotting: Optional[str] = "nan_half"
    umap_layers_to_plot: List[str] = field(
        default_factory=lambda: ["mapped_length", "Raw_modification_signal"]
    )

    # Spatial Analysis - Spatial Autocorrelation params
    autocorr_normalization_method: str = "pearson"
    rows_per_qc_autocorr_grid: int = 12
    autocorr_rolling_window_size: int = 25
    autocorr_max_lag: int = 800
    autocorr_site_types: List[str] = field(default_factory=lambda: ["GpC", "CpG", "C"])

    # Spatial Analysis - Correlation Matrix params
    correlation_matrix_types: List[str] = field(
        default_factory=lambda: ["pearson", "binary_covariance"]
    )
    correlation_matrix_cmaps: List[str] = field(default_factory=lambda: ["seismic", "viridis"])
    correlation_matrix_site_types: List[str] = field(default_factory=lambda: ["GpC_site"])

    # HMM params
    hmm_n_states: int = 2
    hmm_init_emission_probs: List[list] = field(default_factory=lambda: [[0.8, 0.2], [0.2, 0.8]])
    hmm_init_transition_probs: List[list] = field(default_factory=lambda: [[0.9, 0.1], [0.1, 0.9]])
    hmm_init_start_probs: List[float] = field(default_factory=lambda: [0.5, 0.5])
    hmm_eps: float = 1e-8
    hmm_dtype: str = "float64"
    hmm_annotation_threshold: float = 0.5
    hmm_batch_size: int = 1024
    hmm_use_viterbi: bool = False
    hmm_device: Optional[str] = None
    hmm_methbases: Optional[List[str]] = (
        None  # if None, HMM.annotate_adata will fall back to mod_target_bases
    )
    # HMM fitting/application strategy
    hmm_fit_strategy: str = "per_group"  # "per_group" | "shared_transitions"
    hmm_shared_scope: List[str] = field(default_factory=lambda: ["reference", "methbase"])
    hmm_groupby: List[str] = field(default_factory=lambda: ["sample", "reference", "methbase"])
    # Shared-transitions adaptation behavior
    hmm_adapt_emissions: bool = True
    hmm_adapt_startprobs: bool = True
    hmm_emission_adapt_iters: int = 5
    hmm_emission_adapt_tol: float = 1e-4
    footprints: Optional[bool] = True
    accessible_patches: Optional[bool] = True
    cpg: Optional[bool] = False
    hmm_feature_sets: Dict[str, Any] = field(default_factory=dict)
    hmm_feature_colormaps: Dict[str, Any] = field(default_factory=dict)
    hmm_merge_layer_features: Optional[List[Tuple]] = field(default_factory=lambda: [(None, 60)])
    clustermap_cmap_hmm: Optional[str] = "coolwarm"
    hmm_clustermap_feature_layers: List[str] = field(
        default_factory=lambda: ["all_accessible_features"]
    )
    hmm_clustermap_length_layers: List[str] = field(
        default_factory=lambda: ["all_accessible_features"]
    )
    hmm_clustermap_sortby: Optional[str] = "hmm"
    hmm_peak_feature_configs: Dict[str, Any] = field(default_factory=dict)

    # Pipeline control flow - load adata
    force_redo_load_adata: bool = False

    # Pipeline control flow - preprocessing and QC
    force_redo_preprocessing: bool = False
    force_reload_sample_sheet: bool = True
    bypass_add_read_length_and_mapping_qc: bool = False
    force_redo_add_read_length_and_mapping_qc: bool = False
    bypass_clean_nan: bool = False
    force_redo_clean_nan: bool = False
    bypass_append_base_context: bool = False
    force_redo_append_base_context: bool = False
    invert_adata: bool = False
    bypass_append_binary_layer_by_base_context: bool = False
    force_redo_append_binary_layer_by_base_context: bool = False
    bypass_append_mismatch_frequency_sites: bool = False
    force_redo_append_mismatch_frequency_sites: bool = False
    bypass_calculate_read_modification_stats: bool = False
    force_redo_calculate_read_modification_stats: bool = False
    bypass_filter_reads_on_modification_thresholds: bool = False
    force_redo_filter_reads_on_modification_thresholds: bool = False
    bypass_flag_duplicate_reads: bool = False
    force_redo_flag_duplicate_reads: bool = False
    bypass_complexity_analysis: bool = False
    force_redo_complexity_analysis: bool = False

    # Pipeline control flow - Spatial Analyses
    force_redo_spatial_analyses: bool = False
    bypass_basic_clustermaps: bool = False
    force_redo_basic_clustermaps: bool = False
    bypass_basic_umap: bool = False
    force_redo_basic_umap: bool = False
    bypass_spatial_autocorr_calculations: bool = False
    force_redo_spatial_autocorr_calculations: bool = False
    bypass_spatial_autocorr_plotting: bool = False
    force_redo_spatial_autocorr_plotting: bool = False
    bypass_matrix_corr_calculations: bool = False
    force_redo_matrix_corr_calculations: bool = False
    bypass_matrix_corr_plotting: bool = False
    force_redo_matrix_corr_plotting: bool = False

    # Pipeline control flow - HMM Analyses
    bypass_hmm_fit: bool = False
    force_redo_hmm_fit: bool = False
    bypass_hmm_apply: bool = False
    force_redo_hmm_apply: bool = False

    # metadata
    config_source: Optional[str] = None

    # -------------------------
    # Construction helpers
    # -------------------------
    @classmethod
    def from_var_dict(
        cls,
        var_dict: Optional[Dict[str, Any]],
        date_str: Optional[str] = None,
        config_source: Optional[str] = None,
        defaults_dir: Optional[Union[str, Path]] = None,
        defaults_map: Optional[Dict[str, Dict[str, Any]]] = None,
        merge_with_defaults: bool = True,
        override_with_csv: bool = True,
        allow_csv_extends: bool = True,
        allow_null_override: bool = False,
    ) -> Tuple["ExperimentConfig", Dict[str, Any]]:
        """
        Create ExperimentConfig from a raw var_dict (as produced by LoadExperimentConfig).
        Returns (instance, report) where report contains modality/defaults/merged info.

        merge_with_defaults: load defaults from defaults_dir or defaults_map.
        override_with_csv: CSV values override defaults; if False defaults take precedence.
        allow_csv_extends: allow the CSV to include 'extends' to pull in extra defaults files.
        allow_null_override: if False, CSV keys with value None will NOT override defaults (keeps defaults).
        """
        var_dict = var_dict or {}

        # 1) normalize incoming values
        normalized: Dict[str, Any] = {}
        for k, v in var_dict.items():
            if v is None:
                normalized[k] = None
                continue
            if isinstance(v, str):
                s = v.strip()
                if s == "" or s.lower() == "none":
                    normalized[k] = None
                else:
                    normalized[k] = _try_json_or_literal(s)
            else:
                normalized[k] = v

        modality = normalized.get("smf_modality")
        if isinstance(modality, (list, tuple)) and len(modality) > 0:
            modality = modality[0]

        defaults_loaded = {}
        defaults_source_chain: List[str] = []
        if merge_with_defaults:
            if defaults_map and modality in defaults_map:
                defaults_loaded = dict(defaults_map[modality] or {})
                defaults_source_chain = [f"defaults_map['{modality}']"]
            elif defaults_dir is not None:
                defaults_loaded, defaults_source_chain = load_defaults_with_inheritance(
                    defaults_dir, modality
                )

        # If CSV asks to extend defaults, load those and merge
        merged = dict(defaults_loaded or {})

        if allow_csv_extends:
            extends = normalized.get("extends") or normalized.get("inherits")
            if extends:
                if isinstance(extends, str):
                    ext_list = [extends]
                elif isinstance(extends, (list, tuple)):
                    ext_list = list(extends)
                else:
                    ext_list = []
                for ext in ext_list:
                    ext_defaults, ext_sources = (
                        load_defaults_with_inheritance(defaults_dir, ext)
                        if defaults_dir
                        else ({}, [])
                    )
                    merged = deep_merge(merged, ext_defaults)
                    for s in ext_sources:
                        if s not in defaults_source_chain:
                            defaults_source_chain.append(s)

        # Now overlay CSV values
        # Prepare csv_effective depending on allow_null_override
        csv_effective = {}
        for k, v in normalized.items():
            if k in ("extends", "inherits"):
                continue
            if v is None and not allow_null_override:
                # skip: keep default
                continue
            csv_effective[k] = v

        if override_with_csv:
            merged = deep_merge(merged, csv_effective)
        else:
            # defaults take precedence: only set keys missing in merged
            for k, v in csv_effective.items():
                if k not in merged:
                    merged[k] = v

        # experiment_name default
        if merged.get("experiment_name") is None and date_str:
            merged["experiment_name"] = f"{date_str}_SMF_experiment"

        # Input file types and path handling
        input_data_path = Path(merged["input_data_path"])

        # Detect the input filetype
        if input_data_path.is_file():
            suffix = input_data_path.suffix.lower()
            suffixes = [
                s.lower() for s in input_data_path.suffixes
            ]  # handles multi-part extensions

            # recognize multi-suffix cases like .fastq.gz or .fq.gz
            if any(s in [".pod5", ".p5"] for s in suffixes):
                input_type = "pod5"
                input_files = [Path(input_data_path)]
            elif any(s in [".fast5", ".f5"] for s in suffixes):
                input_type = "fast5"
                input_files = [Path(input_data_path)]
            elif any(s in [".fastq", ".fq"] for s in suffixes):
                input_type = "fastq"
                input_files = [Path(input_data_path)]
            elif any(s in [".bam"] for s in suffixes):
                input_type = "bam"
                input_files = [Path(input_data_path)]
            elif any(s in [".h5ad", ".h5"] for s in suffixes):
                input_type = "h5ad"
                input_files = [Path(input_data_path)]
            else:
                print("Error detecting input file type")

        elif input_data_path.is_dir():
            found = discover_input_files(
                input_data_path,
                bam_suffix=merged.get("bam_suffix", BAM_SUFFIX),
                recursive=merged["recursive_input_search"],
            )

            if found["input_is_pod5"]:
                input_type = "pod5"
                input_files = found["pod5_paths"]
            elif found["input_is_fast5"]:
                input_type = "fast5"
                input_files = found["fast5_paths"]
            elif found["input_is_fastq"]:
                input_type = "fastq"
                input_files = found["fastq_paths"]
            elif found["input_is_bam"]:
                input_type = "bam"
                input_files = found["bam_paths"]
            elif found["input_is_h5ad"]:
                input_type = "h5ad"
                input_files = found["h5ad_paths"]

            print(
                f"Found {found['all_files_searched']} files; "
                f"fastq={len(found['fastq_paths'])}, "
                f"bam={len(found['bam_paths'])}, "
                f"pod5={len(found['pod5_paths'])}, "
                f"fast5={len(found['fast5_paths'])}, "
                f"h5ad={len(found['h5ad_paths'])}"
            )

        # summary file output path
        output_dir = Path(merged["output_directory"])
        summary_file_basename = merged["experiment_name"] + "_output_summary.csv"
        summary_file = output_dir / summary_file_basename

        # Demultiplexing output path
        split_dir = merged.get("split_dir", SPLIT_DIR)
        split_path = output_dir / LOAD_DIR / split_dir

        # final normalization
        if "strands" in merged:
            merged["strands"] = _parse_list(merged["strands"])
        if "conversions" in merged:
            merged["conversions"] = _parse_list(merged["conversions"])
        if "mod_target_bases" in merged:
            merged["mod_target_bases"] = _parse_list(merged["mod_target_bases"])
        if "conversion_types" in merged:
            merged["conversion_types"] = _parse_list(merged["conversion_types"])

        merged["filter_threshold"] = float(_parse_numeric(merged.get("filter_threshold", 0.8), 0.8))
        merged["m6A_threshold"] = float(_parse_numeric(merged.get("m6A_threshold", 0.7), 0.7))
        merged["m5C_threshold"] = float(_parse_numeric(merged.get("m5C_threshold", 0.7), 0.7))
        merged["hm5C_threshold"] = float(_parse_numeric(merged.get("hm5C_threshold", 0.7), 0.7))
        merged["thresholds"] = [
            merged["filter_threshold"],
            merged["m6A_threshold"],
            merged["m5C_threshold"],
            merged["hm5C_threshold"],
        ]

        for bkey in (
            "barcode_both_ends",
            "trim",
            "input_already_demuxed",
            "make_bigwigs",
            "skip_unclassified",
            "delete_batch_hdfs",
        ):
            if bkey in merged:
                merged[bkey] = _parse_bool(merged[bkey])

        if "batch_size" in merged:
            merged["batch_size"] = int(_parse_numeric(merged.get("batch_size", 4), 4))
        if "threads" in merged:
            tval = _parse_numeric(merged.get("threads", None), None)
            merged["threads"] = None if tval is None else int(tval)

        if "aligner_args" in merged and merged.get("aligner_args") is None:
            merged.pop("aligner_args", None)

        # --- Resolve aligner_args into concrete list for the chosen aligner ---
        merged["aligner_args"] = resolve_aligner_args(merged)

        if "mod_list" in merged:
            merged["mod_list"] = _parse_list(merged.get("mod_list"))

        # Preprocessing args
        obs_to_plot_pp_qc = _parse_list(merged.get("obs_to_plot_pp_qc", None))

        # HMM feature set handling
        if "hmm_feature_sets" in merged:
            merged["hmm_feature_sets"] = normalize_hmm_feature_sets(merged["hmm_feature_sets"])
        else:
            # allow older names (footprint_ranges, accessible_ranges, cpg_ranges) — optional:
            maybe_fs = {}
            if "footprint_ranges" in merged or "hmm_footprint_ranges" in merged:
                maybe_fs["footprint"] = {
                    "features": merged.get("hmm_footprint_ranges", merged.get("footprint_ranges")),
                    "state": merged.get("hmm_footprint_state", "Non-Modified"),
                }
            if "accessible_ranges" in merged or "hmm_accessible_ranges" in merged:
                maybe_fs["accessible"] = {
                    "features": merged.get(
                        "hmm_accessible_ranges", merged.get("accessible_ranges")
                    ),
                    "state": merged.get("hmm_accessible_state", "Modified"),
                }
            if "cpg_ranges" in merged or "hmm_cpg_ranges" in merged:
                maybe_fs["cpg"] = {
                    "features": merged.get("hmm_cpg_ranges", merged.get("cpg_ranges")),
                    "state": merged.get("hmm_cpg_state", "Modified"),
                }
            if maybe_fs:
                merged.setdefault("hmm_feature_sets", {})
                for k, v in maybe_fs.items():
                    merged["hmm_feature_sets"].setdefault(k, v)

            # final normalization will be done below
            # (do not set local hmm_feature_sets here — do it once below)
            pass

        # Final normalization of hmm_feature_sets and canonical local variables
        merged["hmm_feature_sets"] = normalize_hmm_feature_sets(merged.get("hmm_feature_sets", {}))
        hmm_feature_sets = merged.get("hmm_feature_sets", {})
        hmm_feature_colormaps = merged.get("hmm_feature_colormaps", {})
        if not isinstance(hmm_feature_colormaps, dict):
            hmm_feature_colormaps = {}
        hmm_annotation_threshold = merged.get("hmm_annotation_threshold", 0.5)
        hmm_batch_size = int(merged.get("hmm_batch_size", 1024))
        hmm_use_viterbi = bool(merged.get("hmm_use_viterbi", False))
        hmm_device = merged.get("hmm_device", None)
        hmm_methbases = _parse_list(merged.get("hmm_methbases", None))
        if not hmm_methbases:  # None or []
            hmm_methbases = _parse_list(merged.get("mod_target_bases", None))
        if not hmm_methbases:
            hmm_methbases = ["C"]
        hmm_methbases = list(hmm_methbases)
        hmm_merge_layer_features = _parse_list(merged.get("hmm_merge_layer_features", None))
        hmm_clustermap_feature_layers = _parse_list(
            merged.get("hmm_clustermap_feature_layers", "all_accessible_features")
        )
        hmm_clustermap_length_layers = _parse_list(
            merged.get("hmm_clustermap_length_layers", hmm_clustermap_feature_layers)
        )

        hmm_fit_strategy = str(merged.get("hmm_fit_strategy", "per_group")).strip()
        hmm_shared_scope = _parse_list(merged.get("hmm_shared_scope", ["reference", "methbase"]))
        hmm_groupby = _parse_list(merged.get("hmm_groupby", ["sample", "reference", "methbase"]))

        hmm_adapt_emissions = _parse_bool(merged.get("hmm_adapt_emissions", True))
        hmm_adapt_startprobs = _parse_bool(merged.get("hmm_adapt_startprobs", True))
        hmm_emission_adapt_iters = int(_parse_numeric(merged.get("hmm_emission_adapt_iters", 5), 5))
        hmm_emission_adapt_tol = float(
            _parse_numeric(merged.get("hmm_emission_adapt_tol", 1e-4), 1e-4)
        )

        # HMM peak feature configs (for call_hmm_peaks)
        merged["hmm_peak_feature_configs"] = normalize_peak_feature_configs(
            merged.get("hmm_peak_feature_configs", {})
        )
        hmm_peak_feature_configs = merged.get("hmm_peak_feature_configs", {})

        # instantiate dataclass
        instance = cls(
            annotate_secondary_supplementary=merged.get("annotate_secondary_supplementary", True),
            smf_modality=merged.get("smf_modality"),
            input_data_path=input_data_path,
            recursive_input_search=merged.get("recursive_input_search"),
            input_type=input_type,
            input_files=input_files,
            output_directory=output_dir,
            summary_file=summary_file,
            fasta=merged.get("fasta"),
            sequencer=merged.get("sequencer"),
            model_dir=merged.get("model_dir"),
            barcode_kit=merged.get("barcode_kit"),
            fastq_barcode_map=merged.get("fastq_barcode_map"),
            fastq_auto_pairing=merged.get("fastq_auto_pairing"),
            bam_suffix=merged.get("bam_suffix", BAM_SUFFIX),
            split_dir=split_dir,
            split_path=split_path,
            strands=merged.get("strands", STRANDS),
            conversions=merged.get("conversions", CONVERSIONS),
            fasta_regions_of_interest=merged.get("fasta_regions_of_interest"),
            mapping_threshold=float(merged.get("mapping_threshold", 0.01)),
            experiment_name=merged.get("experiment_name"),
            model=merged.get("model", "hac"),
            barcode_both_ends=merged.get("barcode_both_ends", BARCODE_BOTH_ENDS),
            trim=merged.get("trim", TRIM),
            input_already_demuxed=merged.get("input_already_demuxed", False),
            threads=merged.get("threads"),
            emit_log_file=merged.get("emit_log_file", True),
            log_level=merged.get("log_level", "INFO"),
            sample_sheet_path=merged.get("sample_sheet_path"),
            sample_sheet_mapping_column=merged.get("sample_sheet_mapping_column"),
            delete_intermediate_bams=merged.get("delete_intermediate_bams", False),
            delete_intermediate_tsvs=merged.get("delete_intermediate_tsvs", True),
            align_from_bam=merged.get("align_from_bam", False),
            aligner=merged.get("aligner", "minimap2"),
            aligner_args=merged.get("aligner_args", None),
            device=merged.get("device", "auto"),
            make_bigwigs=merged.get("make_bigwigs", False),
            make_beds=merged.get("make_beds", False),
            samtools_backend=merged.get("samtools_backend", "auto"),
            bedtools_backend=merged.get("bedtools_backend", "auto"),
            bigwig_backend=merged.get("bigwig_backend", "auto"),
            delete_intermediate_hdfs=merged.get("delete_intermediate_hdfs", True),
            mod_target_bases=merged.get("mod_target_bases", ["GpC", "CpG"]),
            enzyme_target_bases=merged.get("enzyme_target_bases", ["GpC"]),
            conversion_types=merged.get("conversions", ["unconverted"])
            + merged.get("conversion_types", ["5mC"]),
            filter_threshold=merged.get("filter_threshold", 0.8),
            m6A_threshold=merged.get("m6A_threshold", 0.7),
            m5C_threshold=merged.get("m5C_threshold", 0.7),
            hm5C_threshold=merged.get("hm5C_threshold", 0.7),
            thresholds=merged.get("thresholds", []),
            mod_list=merged.get("mod_list", list(MOD_LIST)),
            mod_map=merged.get("mod_map", list(MOD_MAP)),
            batch_size=merged.get("batch_size", 4),
            skip_unclassified=merged.get("skip_unclassified", True),
            delete_batch_hdfs=merged.get("delete_batch_hdfs", True),
            reference_column=merged.get("reference_column", REF_COL),
            sample_column=merged.get("sample_column", SAMPLE_COL),
            sample_name_col_for_plotting=merged.get("sample_name_col_for_plotting", "Barcode"),
            obs_to_plot_pp_qc=obs_to_plot_pp_qc,
            fit_position_methylation_thresholds=merged.get(
                "fit_position_methylation_thresholds", False
            ),
            binarize_on_fixed_methlyation_threshold=merged.get(
                "binarize_on_fixed_methlyation_threshold", 0.7
            ),
            positive_control_sample_methylation_fitting=merged.get(
                "positive_control_sample_methylation_fitting", None
            ),
            negative_control_sample_methylation_fitting=merged.get(
                "negative_control_sample_methylation_fitting", None
            ),
            infer_on_percentile_sample_methylation_fitting=merged.get(
                "infer_on_percentile_sample_methylation_fitting", 10
            ),
            inference_variable_sample_methylation_fitting=merged.get(
                "inference_variable_sample_methylation_fitting", "Raw_modification_signal"
            ),
            fit_j_threshold=merged.get("fit_j_threshold", 0.5),
            output_binary_layer_name=merged.get(
                "output_binary_layer_name", "binarized_methylation"
            ),
            reindexing_offsets=merged.get("reindexing_offsets", {None: None}),
            reindexed_var_suffix=merged.get("reindexed_var_suffix", "reindexed"),
            clustermap_demux_types_to_plot=merged.get(
                "clustermap_demux_types_to_plot", ["single", "double", "already"]
            ),
            layer_for_clustermap_plotting=merged.get(
                "layer_for_clustermap_plotting", "nan0_0minus1"
            ),
            clustermap_cmap_c=merged.get("clustermap_cmap_c", "coolwarm"),
            clustermap_cmap_gpc=merged.get("clustermap_cmap_gpc", "coolwarm"),
            clustermap_cmap_cpg=merged.get("clustermap_cmap_cpg", "coolwarm"),
            clustermap_cmap_a=merged.get("clustermap_cmap_a", "coolwarm"),
            spatial_clustermap_sortby=merged.get("spatial_clustermap_sortby", "gpc"),
            rolling_nn_layer=merged.get("rolling_nn_layer", "nan0_0minus1"),
            rolling_nn_plot_layer=merged.get("rolling_nn_plot_layer", "nan0_0minus1"),
            rolling_nn_plot_layers=merged.get(
                "rolling_nn_plot_layers", ["nan0_0minus1", "nan0_0minus1"]
            ),
            rolling_nn_window=merged.get("rolling_nn_window", 15),
            rolling_nn_step=merged.get("rolling_nn_step", 2),
            rolling_nn_min_overlap=merged.get("rolling_nn_min_overlap", 10),
            rolling_nn_return_fraction=merged.get("rolling_nn_return_fraction", True),
            rolling_nn_obsm_key=merged.get("rolling_nn_obsm_key", "rolling_nn_dist"),
            rolling_nn_site_types=merged.get("rolling_nn_site_types", None),
            rolling_nn_write_zero_pairs_csvs=merged.get("rolling_nn_write_zero_pairs_csvs", True),
            rolling_nn_zero_pairs_uns_key=merged.get("rolling_nn_zero_pairs_uns_key", None),
            rolling_nn_zero_pairs_segments_key=merged.get(
                "rolling_nn_zero_pairs_segments_key", None
            ),
            rolling_nn_zero_pairs_layer_key=merged.get("rolling_nn_zero_pairs_layer_key", None),
            rolling_nn_zero_pairs_refine=merged.get("rolling_nn_zero_pairs_refine", True),
            rolling_nn_zero_pairs_max_nan_run=merged.get("rolling_nn_zero_pairs_max_nan_run", None),
            rolling_nn_zero_pairs_merge_gap=merged.get("rolling_nn_zero_pairs_merge_gap", 0),
            rolling_nn_zero_pairs_max_segments_per_read=merged.get(
                "rolling_nn_zero_pairs_max_segments_per_read", None
            ),
            rolling_nn_zero_pairs_max_overlap=merged.get("rolling_nn_zero_pairs_max_overlap", None),
            rolling_nn_zero_pairs_layer_overlap_mode=merged.get(
                "rolling_nn_zero_pairs_layer_overlap_mode", "binary"
            ),
            rolling_nn_zero_pairs_layer_overlap_value=merged.get(
                "rolling_nn_zero_pairs_layer_overlap_value", None
            ),
            rolling_nn_zero_pairs_keep_uns=merged.get("rolling_nn_zero_pairs_keep_uns", True),
            rolling_nn_zero_pairs_segments_keep_uns=merged.get(
                "rolling_nn_zero_pairs_segments_keep_uns", True
            ),
            rolling_nn_zero_pairs_top_segments_per_read=merged.get(
                "rolling_nn_zero_pairs_top_segments_per_read", None
            ),
            rolling_nn_zero_pairs_top_segments_max_overlap=merged.get(
                "rolling_nn_zero_pairs_top_segments_max_overlap", None
            ),
            rolling_nn_zero_pairs_top_segments_min_span=merged.get(
                "rolling_nn_zero_pairs_top_segments_min_span", None
            ),
            rolling_nn_zero_pairs_top_segments_write_csvs=merged.get(
                "rolling_nn_zero_pairs_top_segments_write_csvs", True
            ),
            rolling_nn_zero_pairs_segment_histogram_bins=merged.get(
                "rolling_nn_zero_pairs_segment_histogram_bins", 30
            ),
            cross_sample_analysis=merged.get("cross_sample_analysis", False),
            cross_sample_grouping_col=merged.get("cross_sample_grouping_col", None),
            cross_sample_random_seed=merged.get("cross_sample_random_seed", 42),
            layer_for_umap_plotting=merged.get("layer_for_umap_plotting", "nan_half"),
            umap_layers_to_plot=merged.get(
                "umap_layers_to_plot", ["mapped_length", "Raw_modification_signal"]
            ),
            rows_per_qc_histogram_grid=merged.get("rows_per_qc_histogram_grid", 12),
            rows_per_qc_autocorr_grid=merged.get("rows_per_qc_autocorr_grid", 12),
            autocorr_normalization_method=merged.get("autocorr_normalization_method", "pearson"),
            autocorr_rolling_window_size=merged.get("autocorr_rolling_window_size", 25),
            autocorr_max_lag=merged.get("autocorr_max_lag", 800),
            autocorr_site_types=merged.get("autocorr_site_types", ["GpC", "CpG", "C"]),
            hmm_n_states=merged.get("hmm_n_states", 2),
            hmm_init_emission_probs=merged.get("hmm_init_emission_probs", [[0.8, 0.2], [0.2, 0.8]]),
            hmm_init_transition_probs=merged.get(
                "hmm_init_transition_probs", [[0.9, 0.1], [0.1, 0.9]]
            ),
            hmm_init_start_probs=merged.get("hmm_init_start_probs", [0.5, 0.5]),
            hmm_eps=merged.get("hmm_eps", 1e-8),
            hmm_fit_strategy=hmm_fit_strategy,
            hmm_shared_scope=hmm_shared_scope,
            hmm_groupby=hmm_groupby,
            hmm_adapt_emissions=hmm_adapt_emissions,
            hmm_adapt_startprobs=hmm_adapt_startprobs,
            hmm_emission_adapt_iters=hmm_emission_adapt_iters,
            hmm_emission_adapt_tol=hmm_emission_adapt_tol,
            hmm_dtype=merged.get("hmm_dtype", "float64"),
            hmm_feature_sets=hmm_feature_sets,
            hmm_feature_colormaps=hmm_feature_colormaps,
            hmm_annotation_threshold=hmm_annotation_threshold,
            hmm_batch_size=hmm_batch_size,
            hmm_use_viterbi=hmm_use_viterbi,
            hmm_methbases=hmm_methbases,
            hmm_device=hmm_device,
            hmm_merge_layer_features=hmm_merge_layer_features,
            clustermap_cmap_hmm=merged.get("clustermap_cmap_hmm", "coolwarm"),
            hmm_clustermap_feature_layers=hmm_clustermap_feature_layers,
            hmm_clustermap_length_layers=hmm_clustermap_length_layers,
            hmm_clustermap_sortby=merged.get("hmm_clustermap_sortby", "hmm"),
            hmm_peak_feature_configs=hmm_peak_feature_configs,
            footprints=merged.get("footprints", None),
            accessible_patches=merged.get("accessible_patches", None),
            cpg=merged.get("cpg", None),
            read_coord_filter=merged.get("read_coord_filter", [None, None]),
            read_len_filter_thresholds=merged.get("read_len_filter_thresholds", [100, None]),
            read_len_to_ref_ratio_filter_thresholds=merged.get(
                "read_len_to_ref_ratio_filter_thresholds", [0.3, None]
            ),
            read_quality_filter_thresholds=merged.get("read_quality_filter_thresholds", [15, None]),
            read_mapping_quality_filter_thresholds=merged.get(
                "read_mapping_quality_filter_thresholds", [None, None]
            ),
            read_mod_filtering_gpc_thresholds=merged.get(
                "read_mod_filtering_gpc_thresholds", [0.025, 0.975]
            ),
            read_mod_filtering_cpg_thresholds=merged.get(
                "read_mod_filtering_cpg_thresholds", [0.0, 1.0]
            ),
            read_mod_filtering_c_thresholds=merged.get(
                "read_mod_filtering_c_thresholds", [0.025, 0.975]
            ),
            read_mod_filtering_a_thresholds=merged.get(
                "read_mod_filtering_a_thresholds", [0.025, 0.975]
            ),
            read_mod_filtering_use_other_c_as_background=merged.get(
                "read_mod_filtering_use_other_c_as_background", True
            ),
            min_valid_fraction_positions_in_read_vs_ref=merged.get(
                "min_valid_fraction_positions_in_read_vs_ref", 0.2
            ),
            duplicate_detection_site_types=merged.get(
                "duplicate_detection_site_types", ["GpC", "CpG", "ambiguous_GpC_CpG"]
            ),
            duplicate_detection_demux_types_to_use=merged.get(
                "duplicate_detection_demux_types_to_use", ["single", "double", "already"]
            ),
            duplicate_detection_distance_threshold=merged.get(
                "duplicate_detection_distance_threshold", 0.07
            ),
            duplicate_detection_keep_best_metric=merged.get(
                "duplicate_detection_keep_best_metric", "read_quality"
            ),
            duplicate_detection_window_size_for_hamming_neighbors=merged.get(
                "duplicate_detection_window_size_for_hamming_neighbors", 50
            ),
            duplicate_detection_min_overlapping_positions=merged.get(
                "duplicate_detection_min_overlapping_positions", 20
            ),
            duplicate_detection_do_hierarchical=merged.get(
                "duplicate_detection_do_hierarchical", True
            ),
            duplicate_detection_hierarchical_linkage=merged.get(
                "duplicate_detection_hierarchical_linkage", "average"
            ),
            duplicate_detection_do_pca=merged.get("duplicate_detection_do_pca", False),
            position_max_nan_threshold=merged.get("position_max_nan_threshold", 0.1),
            correlation_matrix_types=merged.get(
                "correlation_matrix_types", ["pearson", "binary_covariance"]
            ),
            correlation_matrix_cmaps=merged.get("correlation_matrix_cmaps", ["seismic", "viridis"]),
            correlation_matrix_site_types=merged.get("correlation_matrix_site_types", ["GpC_site"]),
            hamming_vs_metric_keys=merged.get(
                "hamming_vs_metric_keys", ["Fraction_C_site_modified"]
            ),
            force_redo_load_adata=merged.get("force_redo_load_adata", False),
            force_redo_preprocessing=merged.get("force_redo_preprocessing", False),
            force_reload_sample_sheet=merged.get("force_reload_sample_sheet", True),
            bypass_add_read_length_and_mapping_qc=merged.get(
                "bypass_add_read_length_and_mapping_qc", False
            ),
            force_redo_add_read_length_and_mapping_qc=merged.get(
                "force_redo_add_read_length_and_mapping_qc", False
            ),
            bypass_clean_nan=merged.get("bypass_clean_nan", False),
            force_redo_clean_nan=merged.get("force_redo_clean_nan", False),
            bypass_append_base_context=merged.get("bypass_append_base_context", False),
            force_redo_append_base_context=merged.get("force_redo_append_base_context", False),
            invert_adata=merged.get("invert_adata", False),
            bypass_append_binary_layer_by_base_context=merged.get(
                "bypass_append_binary_layer_by_base_context", False
            ),
            force_redo_append_binary_layer_by_base_context=merged.get(
                "force_redo_append_binary_layer_by_base_context", False
            ),
            bypass_calculate_read_modification_stats=merged.get(
                "bypass_calculate_read_modification_stats", False
            ),
            force_redo_calculate_read_modification_stats=merged.get(
                "force_redo_calculate_read_modification_stats", False
            ),
            bypass_filter_reads_on_modification_thresholds=merged.get(
                "bypass_filter_reads_on_modification_thresholds", False
            ),
            force_redo_filter_reads_on_modification_thresholds=merged.get(
                "force_redo_filter_reads_on_modification_thresholds", False
            ),
            bypass_flag_duplicate_reads=merged.get("bypass_flag_duplicate_reads", False),
            force_redo_flag_duplicate_reads=merged.get("force_redo_flag_duplicate_reads", False),
            bypass_complexity_analysis=merged.get("bypass_complexity_analysis", False),
            force_redo_complexity_analysis=merged.get("force_redo_complexity_analysis", False),
            force_redo_spatial_analyses=merged.get("force_redo_spatial_analyses", False),
            bypass_basic_clustermaps=merged.get("bypass_basic_clustermaps", False),
            force_redo_basic_clustermaps=merged.get("force_redo_basic_clustermaps", False),
            bypass_basic_umap=merged.get("bypass_basic_umap", False),
            force_redo_basic_umap=merged.get("force_redo_basic_umap", False),
            bypass_spatial_autocorr_calculations=merged.get(
                "bypass_spatial_autocorr_calculations", False
            ),
            force_redo_spatial_autocorr_calculations=merged.get(
                "force_redo_spatial_autocorr_calculations", False
            ),
            bypass_spatial_autocorr_plotting=merged.get("bypass_spatial_autocorr_plotting", False),
            force_redo_spatial_autocorr_plotting=merged.get(
                "force_redo_spatial_autocorr_plotting", False
            ),
            bypass_matrix_corr_calculations=merged.get("bypass_matrix_corr_calculations", False),
            force_redo_matrix_corr_calculations=merged.get(
                "force_redo_matrix_corr_calculations", False
            ),
            bypass_matrix_corr_plotting=merged.get("bypass_matrix_corr_plotting", False),
            force_redo_matrix_corr_plotting=merged.get("force_redo_matrix_corr_plotting", False),
            bypass_hmm_fit=merged.get("bypass_hmm_fit", False),
            force_redo_hmm_fit=merged.get("force_redo_hmm_fit", False),
            bypass_hmm_apply=merged.get("bypass_hmm_apply", False),
            force_redo_hmm_apply=merged.get("force_redo_hmm_apply", False),
            config_source=config_source or "<var_dict>",
        )

        report = {
            "modality": modality,
            "defaults_source_chain": defaults_source_chain,
            "defaults_loaded": defaults_loaded,
            "csv_normalized": normalized,
            "merged": merged,
        }
        return instance, report

    # convenience: load from CSV via LoadExperimentConfig
    @classmethod
    def from_csv(
        cls,
        csv_input: Union[str, Path, IO, pd.DataFrame],
        date_str: Optional[str] = None,
        config_source: Optional[str] = None,
        defaults_dir: Optional[Union[str, Path]] = None,
        defaults_map: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple["ExperimentConfig", Dict[str, Any]]:
        """
        Load CSV using LoadExperimentConfig (or accept DataFrame) and build ExperimentConfig.
        Additional kwargs passed to from_var_dict().
        """
        loader = (
            LoadExperimentConfig(csv_input)
            if not isinstance(csv_input, pd.DataFrame)
            else LoadExperimentConfig(pd.DataFrame(csv_input))
        )
        var_dict = loader.var_dict
        return cls.from_var_dict(
            var_dict,
            date_str=date_str,
            config_source=config_source,
            defaults_dir=defaults_dir,
            defaults_map=defaults_map,
            **kwargs,
        )

    # -------------------------
    # validation & serialization
    # -------------------------
    @staticmethod
    def _validate_hmm_features_structure(hfs: dict) -> List[str]:
        errs = []
        if not isinstance(hfs, dict):
            errs.append("hmm_feature_sets must be a mapping if provided.")
            return errs
        for g, info in hfs.items():
            if not isinstance(info, dict):
                errs.append(
                    f"hmm_feature_sets['{g}'] must be a mapping with 'features' and 'state'."
                )
                continue
            feats = info.get("features")
            if not isinstance(feats, dict) or len(feats) == 0:
                errs.append(f"hmm_feature_sets['{g}'] must include non-empty 'features' mapping.")
                continue
            for fname, rng in feats.items():
                try:
                    lo, hi = float(rng[0]), float(rng[1])
                    if lo < 0 or hi <= lo:
                        errs.append(
                            f"Feature range for {g}:{fname} must satisfy 0 <= lo < hi; got {rng}."
                        )
                except Exception:
                    errs.append(f"Feature range for {g}:{fname} is invalid: {rng}")
        return errs

    def validate(self, require_paths: bool = True, raise_on_error: bool = True) -> List[str]:
        """
        Validate the config. If require_paths True, check paths (input_data_path, fasta) exist;
        attempt to create output_directory if missing.
        Returns a list of error messages (empty if none). Raises ValueError if raise_on_error True.
        """
        errors: List[str] = []
        if not self.input_data_path:
            errors.append("input_data_path is required but missing.")
        if not self.output_directory:
            errors.append("output_directory is required but missing.")
        if not self.fasta:
            errors.append("fasta (reference FASTA) is required but missing.")

        if require_paths:
            if self.input_data_path and not Path(self.input_data_path).exists():
                errors.append(f"input_data_path does not exist: {self.input_data_path}")
            if self.fasta and not Path(self.fasta).exists():
                errors.append(f"fasta does not exist: {self.fasta}")
            outp = Path(self.output_directory) if self.output_directory else None
            if outp and not outp.exists():
                try:
                    outp.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Could not create output_directory {self.output_directory}: {e}")

        if not (0.0 <= float(self.mapping_threshold) <= 1.0):
            errors.append("mapping_threshold must be in [0,1].")
        for t in (
            self.filter_threshold,
            self.m6A_threshold,
            self.m5C_threshold,
            self.hm5C_threshold,
        ):
            if not (0.0 <= float(t) <= 1.0):
                errors.append(f"threshold value {t} must be in [0,1].")

        if raise_on_error and errors:
            raise ValueError("ExperimentConfig validation failed:\n  " + "\n  ".join(errors))

        errs = _validate_hmm_features_structure(self.hmm_feature_sets)
        errors.extend(errs)

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Dump config to YAML (string if path None) or save to file at path.
        If pyyaml is not installed, fallback to JSON for file write.
        """
        data = self.to_dict()
        if path is None:
            if yaml is None:
                return json.dumps(data, indent=2)
            return yaml.safe_dump(data, sort_keys=False)
        else:
            p = Path(path)
            if yaml is None:
                p.write_text(json.dumps(data, indent=2), encoding="utf8")
            else:
                p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf8")
            return str(p)

    def save(self, path: Union[str, Path]) -> str:
        return self.to_yaml(path)

    def __repr__(self) -> str:
        return f"<ExperimentConfig modality={self.smf_modality} experiment_name={self.experiment_name} source={self.config_source}>"
