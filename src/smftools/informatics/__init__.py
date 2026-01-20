from __future__ import annotations

from importlib import import_module

_LAZY_ATTRS = {
    "_bed_to_bigwig": "smftools.informatics.bed_functions",
    "_plot_bed_histograms": "smftools.informatics.bed_functions",
    "add_demux_type_annotation": "smftools.informatics.h5ad_functions",
    "add_read_length_and_mapping_qc": "smftools.informatics.h5ad_functions",
    "align_and_sort_BAM": "smftools.informatics.bam_functions",
    "bam_qc": "smftools.informatics.bam_functions",
    "basecall_pod5s": "smftools.informatics.pod5_functions",
    "canoncall": "smftools.informatics.basecalling",
    "concatenate_fastqs_to_bam": "smftools.informatics.bam_functions",
    "converted_BAM_to_adata": "smftools.informatics.converted_BAM_to_adata",
    "count_aligned_reads": "smftools.informatics.bam_functions",
    "demux_and_index_BAM": "smftools.informatics.bam_functions",
    "extract_base_identities": "smftools.informatics.bam_functions",
    "extract_mods": "smftools.informatics.modkit_functions",
    "extract_read_features_from_bam": "smftools.informatics.bam_functions",
    "extract_read_lengths_from_bed": "smftools.informatics.bed_functions",
    "extract_readnames_from_bam": "smftools.informatics.bam_functions",
    "fast5_to_pod5": "smftools.informatics.pod5_functions",
    "find_conversion_sites": "smftools.informatics.fasta_functions",
    "generate_converted_FASTA": "smftools.informatics.fasta_functions",
    "get_chromosome_lengths": "smftools.informatics.fasta_functions",
    "get_native_references": "smftools.informatics.fasta_functions",
    "index_fasta": "smftools.informatics.fasta_functions",
    "make_modbed": "smftools.informatics.modkit_functions",
    "modQC": "smftools.informatics.modkit_functions",
    "modcall": "smftools.informatics.basecalling",
    "modkit_extract_to_adata": "smftools.informatics.modkit_extract_to_adata",
    "ohe_batching": "smftools.informatics.ohe",
    "ohe_layers_decode": "smftools.informatics.ohe",
    "one_hot_decode": "smftools.informatics.ohe",
    "one_hot_encode": "smftools.informatics.ohe",
    "run_multiqc": "smftools.informatics.run_multiqc",
    "separate_bam_by_bc": "smftools.informatics.bam_functions",
    "split_and_index_BAM": "smftools.informatics.bam_functions",
    "subsample_fasta_from_bed": "smftools.informatics.fasta_functions",
    "subsample_pod5": "smftools.informatics.pod5_functions",
    "aligned_BAM_to_bed": "smftools.informatics.bed_functions",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "basecall_pod5s",
    "converted_BAM_to_adata",
    "subsample_fasta_from_bed",
    "subsample_pod5",
    "fast5_to_pod5",
    "run_multiqc",
]
