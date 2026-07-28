"""Preprocess integration for reporting-only partitioned variant evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..informatics.partition_read import load_spine, resolve_relative_path
from .variant_reference import VariantReferenceSet, variant_reference_set_from_legacy

VARIANT_REPORTING_SUBDIR = "variant"

_SUMMARY_RENAMES = {
    "variant_reference_set_id": "variant_reference_set_id",
    "evidence_status": "variant_evidence_status",
    "informative_site_count": "variant_informative_site_count",
    "callable_site_count": "variant_callable_site_count",
    "no_call_count": "variant_no_call_count",
    "member_1_call_count": "variant_member_1_call_count",
    "member_2_call_count": "variant_member_2_call_count",
    "breakpoint_count": "variant_breakpoint_count",
    "has_breakpoint": "variant_has_breakpoint",
    "has_other_reference_segment": "chimeric_variant_sites",
    "other_reference_segment_type": "chimeric_variant_sites_type",
    "self_base_count": "variant_self_base_count",
    "other_base_count": "variant_other_base_count",
    "segment_cigar": "variant_segment_cigar",
}


def variant_reporting_enabled(cfg: Any) -> bool:
    """Return whether normalized configuration requests reporting."""
    return str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "report"


def _reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def resolve_variant_reference_set(
    spine_path: str | Path,
    cfg: Any,
) -> VariantReferenceSet:
    """Resolve legacy configured members from raw-spine canonical sequences."""
    spine = load_spine(spine_path, verbose=False)
    references = dict(spine.uns.get("References", {}) or {})
    sequence_sources: dict[str, str] = {}
    for key, raw_sequence in references.items():
        key = str(key)
        if not key.endswith("_FASTA_sequence"):
            continue
        reference = key.removesuffix("_FASTA_sequence")
        sequence = str(raw_sequence).upper()
        sequence_sources[f"{reference}_top_strand_FASTA_base"] = sequence
        sequence_sources[f"{reference}_bottom_strand_FASTA_base"] = _reverse_complement(sequence)
    reference_set = variant_reference_set_from_legacy(
        getattr(cfg, "references_to_align_for_variant_annotation", [None, None]),
        sequence_sources,
    )
    if reference_set is None:
        raise ValueError("variant reporting requires references_to_align_for_variant_annotation")
    return reference_set


def append_variant_reporting_annotations(
    obs_path: str | Path,
    variant_obs_path: str | Path,
) -> Path:
    """Merge evidence summaries and reporting-only QC masks into preprocess obs."""
    import pyarrow.dataset as arrow_dataset

    obs_path = Path(obs_path)
    obs = pd.read_parquet(obs_path)
    evidence = (
        arrow_dataset.dataset(Path(variant_obs_path), format="parquet").to_table().to_pandas()
    )
    if evidence["read_id"].astype(str).duplicated().any():
        raise ValueError("variant reporting produced duplicate read summaries")
    available = {
        source: target for source, target in _SUMMARY_RENAMES.items() if source in evidence
    }
    summary = evidence[["read_id", *available]].rename(columns=available)
    obs = obs.merge(summary, on="read_id", how="left", validate="one_to_one")

    # Reporting never removes a molecule. Keep the pre-existing QC result as an
    # explicit independently typed channel, then compose the unchanged result.
    obs["passes_nonvariant_qc"] = obs["passes_qc"].astype(bool)
    obs["passes_variant_qc"] = pd.Series(True, index=obs.index, dtype=bool)
    obs["variant_qc_reason"] = ""
    obs["nonvariant_qc_reason"] = ""
    obs.loc[~obs["passes_read_qc"].astype(bool), "nonvariant_qc_reason"] = "failed_read_qc"
    modification_failed = ~obs["passes_modification_qc"].astype(bool)
    both_failed = modification_failed & (obs["nonvariant_qc_reason"] != "")
    obs.loc[modification_failed & ~both_failed, "nonvariant_qc_reason"] = "failed_modification_qc"
    obs.loc[both_failed, "nonvariant_qc_reason"] += ";failed_modification_qc"
    obs["passes_qc"] = obs["passes_nonvariant_qc"] & obs["passes_variant_qc"]
    obs.to_parquet(obs_path, index=False)
    return obs_path


def query_preprocess_variant_evidence(
    spine_path: str | Path,
    *,
    variant_reference_set_ids: Iterable[str] | None = None,
    molecule_uids: Iterable[str] | None = None,
    experiment_uids: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Query authoritative variant sidecars discovered from any downstream spine."""
    from .partitioned_variant import query_partitioned_variant_evidence

    spine_path = Path(spine_path)
    spine = load_spine(spine_path, verbose=False)
    run_root = (
        spine_path.parent.parent.parent.parent
        if spine_path.parent.parent.name in {"generations", ".staging"}
        else spine_path.parent.parent
    )
    manifest = resolve_relative_path(
        spine.uns.get("preprocess_variant_generation_manifest"),
        run_root,
    )
    if manifest is None or not manifest.is_file():
        raise ValueError("spine does not publish preprocess variant evidence")
    return query_partitioned_variant_evidence(
        manifest.parent,
        variant_reference_set_ids=variant_reference_set_ids,
        molecule_uids=molecule_uids,
        experiment_uids=experiment_uids,
    )
