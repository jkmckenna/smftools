"""Preprocess integration for partitioned variant evidence and strict QC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..informatics.partition_read import load_spine, resolve_relative_path
from .variant_reference import (
    VariantReferenceSet,
    calculate_variant_informative_sites,
    variant_reference_set_from_legacy,
)

VARIANT_REPORTING_SUBDIR = "variant"

_SUMMARY_RENAMES = {
    "variant_reference_set_id": "variant_reference_set_id",
    "aligned_member_index": "variant_aligned_member_index",
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

VARIANT_QC_BREAKPOINT = "breakpoint"
VARIANT_QC_AMBIGUOUS_REFERENCE = "ambiguous_reference_assignment"
VARIANT_QC_SELF_CONSISTENT = "self_consistent"
VARIANT_QC_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
VARIANT_QC_EVIDENCE_UNAVAILABLE = "evidence_unavailable"


def variant_reporting_enabled(cfg: Any) -> bool:
    """Return whether normalized configuration requests variant evidence."""
    return str(getattr(cfg, "variant_analysis_mode", "off")).lower() in {
        "report",
        "filter",
    }


def _variant_qc_classes(obs: pd.DataFrame, cfg: Any) -> pd.Series:
    """Classify strict QC events from raw evidence counts, never segment lengths."""
    minimum_callable = int(getattr(cfg, "variant_qc_min_callable_sites", None) or 1)
    minimum_fraction = float(getattr(cfg, "variant_qc_min_callable_fraction", None) or 0.0)
    minimum_state_calls = int(getattr(cfg, "variant_qc_min_calls_per_state", None) or 1)
    classes = pd.Series(
        VARIANT_QC_EVIDENCE_UNAVAILABLE,
        index=obs.index,
        dtype="object",
    )
    complete = obs["variant_evidence_status"].fillna("").astype(str).eq("complete")
    informative = pd.to_numeric(obs["variant_informative_site_count"], errors="coerce").fillna(0)
    callable_sites = pd.to_numeric(obs["variant_callable_site_count"], errors="coerce").fillna(0)
    callable_fraction = callable_sites.div(informative.where(informative > 0))
    obs["variant_callable_fraction"] = callable_fraction.astype(float)
    sufficient = (
        complete & (callable_sites >= minimum_callable) & (callable_fraction >= minimum_fraction)
    )
    classes.loc[complete & ~sufficient] = VARIANT_QC_INSUFFICIENT_EVIDENCE

    first_calls = pd.to_numeric(obs["variant_member_1_call_count"], errors="coerce").fillna(0)
    second_calls = pd.to_numeric(obs["variant_member_2_call_count"], errors="coerce").fillna(0)
    aligned_member = pd.to_numeric(obs["variant_aligned_member_index"], errors="coerce")
    self_calls = first_calls.where(aligned_member.eq(0), second_calls)
    other_calls = second_calls.where(aligned_member.eq(0), first_calls)
    valid_member = aligned_member.isin([0, 1])
    classes.loc[complete & ~valid_member] = VARIANT_QC_EVIDENCE_UNAVAILABLE

    breakpoint = obs["variant_has_breakpoint"].fillna(False).astype(bool)
    supported_breakpoint = (
        sufficient
        & valid_member
        & breakpoint
        & (self_calls >= minimum_state_calls)
        & (other_calls >= minimum_state_calls)
    )
    classes.loc[supported_breakpoint] = VARIANT_QC_BREAKPOINT
    unsupported_breakpoint = sufficient & valid_member & breakpoint & ~supported_breakpoint
    classes.loc[unsupported_breakpoint] = VARIANT_QC_INSUFFICIENT_EVIDENCE

    without_breakpoint = sufficient & valid_member & ~breakpoint
    ambiguous = without_breakpoint & self_calls.eq(0) & (other_calls >= minimum_state_calls)
    classes.loc[ambiguous] = VARIANT_QC_AMBIGUOUS_REFERENCE
    self_consistent = without_breakpoint & other_calls.eq(0) & self_calls.gt(0)
    classes.loc[self_consistent] = VARIANT_QC_SELF_CONSISTENT
    classes.loc[without_breakpoint & ~(ambiguous | self_consistent)] = (
        VARIANT_QC_INSUFFICIENT_EVIDENCE
    )
    return classes


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


def variant_candidate_positions_by_reference(
    spine_path: str | Path,
    cfg: Any,
) -> dict[str, set[int]]:
    """Reference positions that carry allele identity, keyed by reference strand.

    Computed from the *unconverted* catalog, so it is the full candidate set
    before any chemistry excludes part of it, and -- crucially -- derived from
    the references alone with no read data. That is what keeps the deamination
    and variant lanes acyclic: deamination evidence excludes these positions,
    then variant calling consumes the resulting segments, and nothing flows
    backwards (`EGL-20a`).

    Without the exclusion the two lanes explain each other. At a C/T informative
    site a genuine reference difference is indistinguishable from a `C->T`
    deamination event; on the `241213` pilot **20 of 22** informative sites
    involve a C or G, so allele identity would masquerade as chemistry and
    inflate the evidence a chimera call rests on.
    """
    reference_set = resolve_variant_reference_set(spine_path, cfg)
    catalog = calculate_variant_informative_sites(reference_set)
    by_reference: dict[str, set[int]] = {}
    for member_index, member in enumerate(reference_set.members):
        source_id = str(member.source_id or "")
        reference_strand = source_id.removesuffix("_strand_FASTA_base")
        if not reference_strand:
            continue
        positions = {int(site.member_positions[member_index]) for site in catalog.informative_sites}
        by_reference[reference_strand] = positions
    return by_reference


def append_variant_reporting_annotations(
    obs_path: str | Path,
    variant_obs_path: str | Path,
    cfg: Any,
) -> Path:
    """Merge evidence summaries and compose report/filter variant QC masks."""
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

    # Keep the pre-existing QC result as an explicit independent channel before
    # composing the selected variant policy.
    obs["passes_nonvariant_qc"] = obs["passes_qc"].astype(bool)
    obs["variant_qc_class"] = _variant_qc_classes(obs, cfg)
    filter_mode = str(getattr(cfg, "variant_analysis_mode", "off")).lower() == "filter"
    disallowed = {str(value) for value in getattr(cfg, "variant_qc_disallowed_event_classes", [])}
    failed_variant = (
        obs["variant_qc_class"].isin(disallowed)
        if filter_mode
        else pd.Series(False, index=obs.index)
    )
    obs["passes_variant_qc"] = (~failed_variant).astype(bool)
    obs["variant_qc_reason"] = ""
    obs.loc[failed_variant, "variant_qc_reason"] = "disallowed_" + obs.loc[
        failed_variant, "variant_qc_class"
    ].astype(str)
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
