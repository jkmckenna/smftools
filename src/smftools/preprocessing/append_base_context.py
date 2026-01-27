from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def append_base_context(
    adata: "ad.AnnData",
    ref_column: str = "Reference_strand",
    use_consensus: bool = False,
    native: bool = False,
    mod_target_bases: list[str] = ["GpC", "CpG"],
    bypass: bool = False,
    force_redo: bool = False,
    uns_flag: str = "append_base_context_performed",
) -> None:
    """Append base context annotations to ``adata``.

    Args:
        adata: AnnData object.
        ref_column: Obs column used to stratify references.
        use_consensus: Whether to use consensus sequences rather than FASTA references.
        native: If ``True``, use native SMF assumptions; otherwise use conversion assumptions.
        mod_target_bases: Base contexts that may be modified.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
    """
    import numpy as np

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        # QC already performed; nothing to do
        return

    logger.info("Adding base context based on reference FASTA sequence for sample")
    references = adata.obs[ref_column].cat.categories
    site_types = []

    if any(base in mod_target_bases for base in ["GpC", "CpG", "C"]):
        site_types += ["GpC_site", "CpG_site", "ambiguous_GpC_CpG_site", "other_C_site", "C_site"]

    if "A" in mod_target_bases:
        site_types += ["A_site"]

    for ref in references:
        # Assess if the strand is the top or bottom strand converted
        if "top" in ref:
            strand = "top"
        elif "bottom" in ref:
            strand = "bottom"

        if native:
            basename = ref.split(f"_{strand}")[0]
            if use_consensus:
                sequence = adata.uns[f"{basename}_consensus_sequence"]
            else:
                # This sequence is the unconverted FASTA sequence of the original input FASTA for the locus
                sequence = adata.uns[f"{basename}_FASTA_sequence"]
        else:
            basename = ref.split(f"_{strand}")[0]
            if use_consensus:
                sequence = adata.uns[f"{basename}_consensus_sequence"]
            else:
                # This sequence is the unconverted FASTA sequence of the original input FASTA for the locus
                sequence = adata.uns[f"{basename}_FASTA_sequence"]

        # Init a dict keyed by reference site type that points to a bool of whether the position is that site type.
        boolean_dict = {}
        for site_type in site_types:
            boolean_dict[f"{ref}_{site_type}"] = np.full(len(sequence), False, dtype=bool)

        if any(base in mod_target_bases for base in ["GpC", "CpG", "C"]):
            if strand == "top":
                # Iterate through the sequence and apply the criteria
                for i in range(1, len(sequence) - 1):
                    if sequence[i] == "C":
                        boolean_dict[f"{ref}_C_site"][i] = True
                        if sequence[i - 1] == "G" and sequence[i + 1] != "G":
                            boolean_dict[f"{ref}_GpC_site"][i] = True
                        elif sequence[i - 1] == "G" and sequence[i + 1] == "G":
                            boolean_dict[f"{ref}_ambiguous_GpC_CpG_site"][i] = True
                        elif sequence[i - 1] != "G" and sequence[i + 1] == "G":
                            boolean_dict[f"{ref}_CpG_site"][i] = True
                        elif sequence[i - 1] != "G" and sequence[i + 1] != "G":
                            boolean_dict[f"{ref}_other_C_site"][i] = True
            elif strand == "bottom":
                # Iterate through the sequence and apply the criteria
                for i in range(1, len(sequence) - 1):
                    if sequence[i] == "G":
                        boolean_dict[f"{ref}_C_site"][i] = True
                        if sequence[i + 1] == "C" and sequence[i - 1] != "C":
                            boolean_dict[f"{ref}_GpC_site"][i] = True
                        elif sequence[i - 1] == "C" and sequence[i + 1] == "C":
                            boolean_dict[f"{ref}_ambiguous_GpC_CpG_site"][i] = True
                        elif sequence[i - 1] == "C" and sequence[i + 1] != "C":
                            boolean_dict[f"{ref}_CpG_site"][i] = True
                        elif sequence[i - 1] != "C" and sequence[i + 1] != "C":
                            boolean_dict[f"{ref}_other_C_site"][i] = True
            else:
                logger.error(
                    "Top or bottom strand of conversion could not be determined. Ensure this value is in the Reference name."
                )

        if "A" in mod_target_bases:
            if strand == "top":
                # Iterate through the sequence and apply the criteria
                for i in range(1, len(sequence) - 1):
                    if sequence[i] == "A":
                        boolean_dict[f"{ref}_A_site"][i] = True
            elif strand == "bottom":
                # Iterate through the sequence and apply the criteria
                for i in range(1, len(sequence) - 1):
                    if sequence[i] == "T":
                        boolean_dict[f"{ref}_A_site"][i] = True
            else:
                logger.error(
                    "Top or bottom strand of conversion could not be determined. Ensure this value is in the Reference name."
                )

        for site_type in site_types:
            # Site context annotations for each reference
            adata.var[f"{ref}_{site_type}"] = boolean_dict[f"{ref}_{site_type}"].astype(bool)
            # Restrict the site type labels to only be in positions that occur at a high enough frequency in the dataset
            if adata.uns.get("calculate_coverage_performed", False):
                adata.var[f"{ref}_{site_type}_valid_coverage"] = (
                    (adata.var[f"{ref}_{site_type}"]) & (adata.var[f"position_in_{ref}"])
                )
            #     if native:
            #         adata.obsm[f"{ref}_{site_type}_valid_coverage"] = adata[
            #             :, adata.var[f"{ref}_{site_type}_valid_coverage"]
            #         ].layers["binarized_methylation"]
            #     else:
            #         adata.obsm[f"{ref}_{site_type}_valid_coverage"] = adata[
            #             :, adata.var[f"{ref}_{site_type}_valid_coverage"]
            #         ].X
            # else:
            #     pass

            # if native:
            #     adata.obsm[f"{ref}_{site_type}"] = adata[:, adata.var[f"{ref}_{site_type}"]].layers[
            #         "binarized_methylation"
            #     ]
            # else:
            #     adata.obsm[f"{ref}_{site_type}"] = adata[:, adata.var[f"{ref}_{site_type}"]].X

    # mark as done
    adata.uns[uns_flag] = True

    return None
