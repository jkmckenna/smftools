# experiment_config.py
from __future__ import annotations
import ast
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, IO

# Optional dependency for YAML handling
try:
    import yaml
except Exception:
    yaml = None

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

# add this helper somewhere near your other utilities (e.g. below _try_json_or_literal)

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
        "minimap2": ['-a', '-x', 'map-ont', '--MD', '-Y', '-y', '-N', '5', '--secondary=no'],
        "dorado": ['--mm2-opts', '-N', '5'],
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
        if 'variable' not in df.columns:
            raise ValueError("Config CSV must contain a 'variable' column.")
        if 'value' not in df.columns:
            df['value'] = ''
        if 'type' not in df.columns:
            df['type'] = ''
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

        hint = (dtype_hint or "").strip().lower()

        def parse_bool(s: str):
            s2 = s.strip().lower()
            if s2 in ('1', 'true', 't', 'yes', 'y', 'on'):
                return True
            if s2 in ('0', 'false', 'f', 'no', 'n', 'off'):
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
            parts = [p.strip() for p in s.strip("()[] ").split(',') if p.strip() != ""]
            return parts

        if hint in ('int', 'integer'):
            return int(v)
        if hint in ('float', 'double'):
            return float(v)
        if hint in ('bool', 'boolean'):
            return parse_bool(v)
        if hint in ('list', 'array'):
            return parse_list_like(v)
        if hint in ('string', 'str'):
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
        if (',' in v) and (not any(ch in v for ch in '{}[]()')):
            return [p.strip() for p in v.split(',') if p.strip() != ""]
        return v

    def _parse_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for idx, row in df.iterrows():
            name = str(row['variable']).strip()
            if name == "":
                continue
            raw_val = row.get('value', "")
            raw_type = row.get('type', "")
            if pd.isna(raw_val) or str(raw_val).strip() == "":
                raw_val = None
            try:
                parsed_val = self._parse_value_as_type(raw_val, raw_type)
            except Exception as e:
                warnings.warn(f"Failed to parse config variable '{name}' (row {idx}): {e}. Storing raw value.")
                parsed_val = None if raw_val is None else raw_val
            if name in parsed:
                warnings.warn(f"Duplicate config variable '{name}' encountered (row {idx}). Overwriting previous value.")
            parsed[name] = parsed_val
        return parsed

    def to_dataframe(self) -> pd.DataFrame:
        """Return parsed config as a pandas DataFrame (variable, value)."""
        rows = []
        for k, v in self.var_dict.items():
            rows.append({'variable': k, 'value': v})
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
    # Core (may be None, then validate will catch if required)
    smf_modality: Optional[str] = None
    input_data_path: Optional[str] = None
    output_directory: Optional[str] = None
    fasta: Optional[str] = None
    model_dir: Optional[str] = None
    barcode_kit: Optional[str] = None
    sequencer: Optional[str] = None

    # Defaults / optional
    bam_suffix: str = ".bam"
    recursive_input_search: bool = True
    split_dir: str = "demultiplexed_BAMs"
    strands: List[str] = field(default_factory=lambda: ["bottom", "top"])
    conversions: List[str] = field(default_factory=lambda: ["unconverted"])
    fasta_regions_of_interest: Optional[str] = None
    mapping_threshold: float = 0.01
    experiment_name: Optional[str] = None
    model: str = "hac"
    barcode_both_ends: bool = False
    trim: bool = False
    input_already_demuxed: bool = False
    threads: Optional[int] = None
    sample_sheet_path: Optional[str] = None
    sample_sheet_mapping_column: Optional[str] = 'Barcode'
    aligner: str = "minimap2"
    aligner_args: Optional[List[str]] = None
    device: str = "auto"
    make_bigwigs: bool = False

    # FASTQ input specific
    fastq_barcode_map: Optional[Dict[str, str]] = None
    fastq_auto_pairing: bool = True

    # Enzyme / mod targets
    mod_target_bases: List[str] = field(default_factory=lambda: ["GpC", "CpG"])
    enzyme_target_bases: List[str] = field(default_factory=lambda: ["GpC"])

    # Conversion/deamination
    conversion_types: List[str] = field(default_factory=lambda: ["5mC"])

    # Direct methylation thresholds
    filter_threshold: float = 0.8
    m6A_threshold: float = 0.7
    m5C_threshold: float = 0.7
    hm5C_threshold: float = 0.7
    thresholds: List[float] = field(default_factory=list)
    mod_list: List[str] = field(default_factory=lambda: ["5mC_5hmC", "6mA"])
    batch_size: int = 4
    skip_unclassified: bool = True
    delete_batch_hdfs: bool = True

    # Anndata structure
    reference_column: Optional[str] = 'Reference_strand'

    # Plotting
    sample_name_col_for_plotting: Optional[str] = 'Barcode'
    layer_for_clustermap_plotting: Optional[str] = 'nan0_0minus1'
    layer_for_umap_plotting: Optional[str] = 'nan_half'
    rows_per_qc_histogram_grid: int = 12
    rows_per_qc_autocorr_grid: int = 12
    autocorr_rolling_window_size: int = 25
    autocorr_max_lag: int = 500
    autocorr_site_types: List[str] = field(default_factory=lambda: ['GpC', 'CpG', 'any_C'])

    # QC
    duplicate_detection_site_types: List[str] = field(default_factory=lambda: ['GpC', 'CpG', 'ambiguous_GpC_CpG'])
    duplicate_detection_distance_threshold: float = 0.12

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
                defaults_loaded, defaults_source_chain = load_defaults_with_inheritance(defaults_dir, modality)

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
                    ext_defaults, ext_sources = (load_defaults_with_inheritance(defaults_dir, ext) if defaults_dir else ({}, []))
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

        for bkey in ("barcode_both_ends", "trim", "input_already_demuxed", "make_bigwigs", "skip_unclassified", "delete_batch_hdfs"):
            if bkey in merged:
                merged[bkey] = _parse_bool(merged[bkey])

        if "batch_size" in merged:
            merged["batch_size"] = int(_parse_numeric(merged.get("batch_size", 4), 4))
        if "threads" in merged:
            tval = _parse_numeric(merged.get("threads", None), None)
            merged["threads"] = None if tval is None else int(tval)

        # aligner_args default handling
        if merged.get("aligner_args") is None:
            if str(merged.get("aligner", "minimap2")).lower() == "minimap2":
                merged["aligner_args"] = ['-a', '-x', 'map-ont', '--MD', '-Y', '-y', '-N', '5', '--secondary=no']
            else:
                merged["aligner_args"] = ['--mm2-opts', '-N 5']
                
        # --- Resolve aligner_args into concrete list for the chosen aligner ---
        merged['aligner_args'] = resolve_aligner_args(merged)

        if "mod_list" in merged:
            merged["mod_list"] = _parse_list(merged.get("mod_list"))

        # instantiate dataclass
        instance = cls(
            smf_modality = merged.get("smf_modality"),
            input_data_path = merged.get("input_data_path"),
            recursive_input_search = merged.get("recursive_input_search"),
            output_directory = merged.get("output_directory"),
            fasta = merged.get("fasta"),
            sequencer = merged.get("sequencer"),
            model_dir = merged.get("model_dir"),
            barcode_kit = merged.get("barcode_kit"),
            fastq_barcode_map = merged.get("fastq_barcode_map"),
            fastq_auto_pairing = merged.get("fastq_auto_pairing"),
            bam_suffix = merged.get("bam_suffix", ".bam"),
            split_dir = merged.get("split_dir", "demultiplexed_BAMs"),
            strands = merged.get("strands", ["bottom","top"]),
            conversions = merged.get("conversions", ["unconverted"]),
            fasta_regions_of_interest = merged.get("fasta_regions_of_interest"),
            mapping_threshold = float(merged.get("mapping_threshold", 0.01)),
            experiment_name = merged.get("experiment_name"),
            model = merged.get("model", "hac"),
            barcode_both_ends = merged.get("barcode_both_ends", False),
            trim = merged.get("trim", False),
            input_already_demuxed = merged.get("input_already_demuxed", False),
            threads = merged.get("threads"),
            sample_sheet_path = merged.get("sample_sheet_path"),
            sample_sheet_mapping_column = merged.get("sample_sheet_mapping_column"),
            aligner = merged.get("aligner", "minimap2"),
            aligner_args = merged.get("aligner_args", None),
            device = merged.get("device", "auto"),
            make_bigwigs = merged.get("make_bigwigs", False),
            mod_target_bases = merged.get("mod_target_bases", ["GpC","CpG"]),
            enzyme_target_bases = merged.get("mod_target_bases", ["GpC"]), 
            conversion_types = merged.get("conversion_types", ["5mC"]),
            filter_threshold = merged.get("filter_threshold", 0.8),
            m6A_threshold = merged.get("m6A_threshold", 0.7),
            m5C_threshold = merged.get("m5C_threshold", 0.7),
            hm5C_threshold = merged.get("hm5C_threshold", 0.7),
            thresholds = merged.get("thresholds", []),
            mod_list = merged.get("mod_list", ["5mC_5hmC","6mA"]),
            batch_size = merged.get("batch_size", 4),
            skip_unclassified = merged.get("skip_unclassified", True),
            delete_batch_hdfs = merged.get("delete_batch_hdfs", True),
            reference_column = merged.get("reference_column", 'Reference_strand'),
            sample_name_col_for_plotting = merged.get("sample_name_col_for_plotting", 'Barcode'),
            layer_for_clustermap_plotting = merged.get("layer_for_clustermap_plotting", 'nan0_0minus1'), 
            layer_for_umap_plotting = merged.get("layer_for_umap_plotting", 'nan_half'),
            rows_per_qc_histogram_grid = merged.get("rows_per_qc_histogram_grid", 12),
            rows_per_qc_autocorr_grid = merged.get("rows_per_qc_autocorr_grid", 12),
            autocorr_rolling_window_size = merged.get("autocorr_rolling_window_size", 25),
            autocorr_max_lag = merged.get("autocorr_max_lag", 500), 
            autocorr_site_types = merged.get("autocorr_site_types", ['GpC', 'CpG', 'any_C']),
            duplicate_detection_site_types = merged.get("duplicate_detection_site_types", ['GpC', 'CpG', 'ambiguous_GpC_CpG']),
            duplicate_detection_distance_threshold = merged.get("duplicate_detection_distance_threshold", 0.12),
            config_source = config_source or "<var_dict>",
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
        loader = LoadExperimentConfig(csv_input) if not isinstance(csv_input, pd.DataFrame) else LoadExperimentConfig(pd.DataFrame(csv_input))
        var_dict = loader.var_dict
        return cls.from_var_dict(var_dict, date_str=date_str, config_source=config_source, defaults_dir=defaults_dir, defaults_map=defaults_map, **kwargs)

    # -------------------------
    # validation & serialization
    # -------------------------
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
        for t in (self.filter_threshold, self.m6A_threshold, self.m5C_threshold, self.hm5C_threshold):
            if not (0.0 <= float(t) <= 1.0):
                errors.append(f"threshold value {t} must be in [0,1].")

        if raise_on_error and errors:
            raise ValueError("ExperimentConfig validation failed:\n  " + "\n  ".join(errors))
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
