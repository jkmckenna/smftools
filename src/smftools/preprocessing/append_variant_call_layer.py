from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def append_variant_call_layer(
    adata: "ad.AnnData",
    seq1_column: str,
    seq2_column: str,
    seq1_converted_column: str | None = None,
    seq2_converted_column: str | None = None,
    sequence_layer: str = "sequence_integer_encoding",
    read_span_layer: str = "read_span_mask",
    reference_col: str = "Reference_strand",
    output_prefix: str | None = None,
    uns_flag: str = "append_variant_call_layer_performed",
    force_redo: bool = False,
    bypass: bool = False,
) -> None:
    """Append a layer recording per-read, per-position variant calls at reference mismatch sites.

    Uses the substitution map from ``append_sequence_mismatch_annotations`` to
    correctly handle coordinate shifts caused by indels between references.
    For each substitution, reads aligned to ref1 are checked at ref1's var index,
    and reads aligned to ref2 are checked at ref2's var index.

    For conversion SMF, reads are mapped to *converted* references while the
    alignment that identifies mismatch positions uses *unconverted* sequences.
    When ``seq1_converted_column`` / ``seq2_converted_column`` are provided, each
    reference gets a **set** of acceptable bases at each mismatch position
    (unconverted + converted), since not every base converts in every read.
    A position is informative only if the two acceptable-base sets are disjoint.
    A read base matching either the unconverted or converted form of a reference
    counts as a match for that reference.

    Values in the output layer:
      1 = matches seq1 base(s)
      2 = matches seq2 base(s)
      0 = unknown (N, PAD, no coverage, or matches neither)
     -1 = not a mismatch position (or not informative after conversion)

    Args:
        adata: AnnData object.
        seq1_column: Column in ``adata.var`` with the first reference base per position (unconverted).
        seq2_column: Column in ``adata.var`` with the second reference base per position (unconverted).
        seq1_converted_column: Optional column in ``adata.var`` with the converted seq1 bases.
            When provided, both unconverted and converted bases are accepted as ref1 matches.
        seq2_converted_column: Optional column in ``adata.var`` with the converted seq2 bases.
        sequence_layer: Layer containing integer-encoded actual read bases.
        read_span_layer: Layer containing read span masks.
        reference_col: Obs column defining which reference each read is aligned to.
        output_prefix: Prefix for the output layer name. Defaults to ``{seq1_column}__{seq2_column}``.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        bypass: Whether to skip processing.
    """
    if bypass:
        return

    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        return

    if sequence_layer not in adata.layers:
        logger.debug("Sequence layer '%s' not found; skipping variant call layer.", sequence_layer)
        return

    output_prefix = output_prefix or f"{seq1_column}__{seq2_column}"
    layer_name = f"{output_prefix}_variant_call"

    # Get the substitution map from alignment annotations
    sub_map_key = f"{output_prefix}_substitution_map"
    sub_map = adata.uns.get(sub_map_key)
    if sub_map is None or (hasattr(sub_map, "__len__") and len(sub_map) == 0):
        logger.warning(
            "Substitution map '%s' not found or empty; skipping variant call layer.",
            sub_map_key,
        )
        return

    import pandas as pd

    if isinstance(sub_map, pd.DataFrame):
        vi1_arr = sub_map["seq1_var_idx"].values
        vi2_arr = sub_map["seq2_var_idx"].values
        b1_arr = sub_map["seq1_base"].values
        b2_arr = sub_map["seq2_base"].values
    else:
        vi1_arr = np.asarray(sub_map.get("seq1_var_idx", []))
        vi2_arr = np.asarray(sub_map.get("seq2_var_idx", []))
        b1_arr = np.asarray(sub_map.get("seq1_base", []))
        b2_arr = np.asarray(sub_map.get("seq2_base", []))
    n_subs = len(vi1_arr)
    if n_subs == 0:
        logger.warning("Substitution map is empty; skipping variant call layer.")
        return

    mismatch_map = adata.uns.get("mismatch_integer_encoding_map", {})
    if not mismatch_map:
        logger.debug("Mismatch encoding map not found; skipping variant call layer.")
        return

    n_value = int(mismatch_map.get("N", MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"]))
    pad_value = int(mismatch_map.get("PAD", MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"]))
    uninformative = {n_value, pad_value}

    # Build base -> int lookup
    base_to_int: dict[str, int] = {}
    for base, value in mismatch_map.items():
        if base not in {"N", "PAD"} and isinstance(value, (int, np.integer)):
            base_to_int[base.upper()] = int(value)

    # Reverse lookup: int -> base letter (for storing readable annotations in var)
    int_to_base: dict[int, str] = {v: k for k, v in base_to_int.items()}

    n_obs, n_vars = adata.shape
    result = np.full((n_obs, n_vars), -1, dtype=np.int8)

    # Per-position var annotations
    ref1_acceptable_bases = [""] * n_vars
    ref2_acceptable_bases = [""] * n_vars
    is_informative = np.zeros(n_vars, dtype=bool)

    seq_matrix = np.asarray(adata.layers[sequence_layer])
    has_span = read_span_layer in adata.layers
    if has_span:
        span_matrix = np.asarray(adata.layers[read_span_layer])

    # Determine which reference each read belongs to
    ref_labels = adata.obs[reference_col].values
    ref_categories = adata.obs[reference_col].cat.categories

    # Map each reference category to seq1 or seq2.
    # Column names like "6B6_top_strand_FASTA_base" have stem "6B6_top" matching ref categories.
    suffix = "_strand_FASTA_base"
    seq1_stem = seq1_column[: -len(suffix)] if seq1_column.endswith(suffix) else seq1_column
    seq2_stem = seq2_column[: -len(suffix)] if seq2_column.endswith(suffix) else seq2_column
    ref_to_seq: dict[str, int] = {}  # ref_category -> 1 or 2
    for ref in ref_categories:
        if ref == seq1_stem:
            ref_to_seq[ref] = 1
        elif ref == seq2_stem:
            ref_to_seq[ref] = 2
        else:
            logger.debug(
                "Reference '%s' does not match seq1 stem '%s' or seq2 stem '%s'.",
                ref,
                seq1_stem,
                seq2_stem,
            )
    logger.info("Reference-to-sequence mapping: %s", ref_to_seq)

    # Build per-reference acceptable base sets.
    # For conversion SMF, a read base can match either the unconverted or converted
    # form of a reference. A substitution is informative only when the two sets are disjoint.
    use_converted = bool(seq1_converted_column and seq2_converted_column)
    if use_converted:
        if seq1_converted_column not in adata.var:
            logger.warning(
                "Converted column '%s' not in adata.var; falling back to unconverted.",
                seq1_converted_column,
            )
            use_converted = False
        elif seq2_converted_column not in adata.var:
            logger.warning(
                "Converted column '%s' not in adata.var; falling back to unconverted.",
                seq2_converted_column,
            )
            use_converted = False
        else:
            conv1_bases = adata.var[seq1_converted_column].values
            conv2_bases = adata.var[seq2_converted_column].values
            logger.info(
                "Using converted columns for variant calling: '%s', '%s'.",
                seq1_converted_column,
                seq2_converted_column,
            )

    logger.info("Processing %d substitutions for variant calling.", n_subs)

    n_informative = 0
    n_collapsed = 0
    for i in range(n_subs):
        vi1 = int(vi1_arr[i])
        vi2 = int(vi2_arr[i])

        # Unconverted bases (always available from substitution map)
        ub1 = base_to_int.get(str(b1_arr[i]).upper())
        ub2 = base_to_int.get(str(b2_arr[i]).upper())
        if ub1 is None or ub2 is None:
            continue

        # Build sets of acceptable integer-encoded bases for each reference
        ref1_ints: set[int] = {ub1}
        ref2_ints: set[int] = {ub2}
        if use_converted:
            cb1 = base_to_int.get(str(conv1_bases[vi1]).upper())
            cb2 = base_to_int.get(str(conv2_bases[vi2]).upper())
            if cb1 is not None:
                ref1_ints.add(cb1)
            if cb2 is not None:
                ref2_ints.add(cb2)

        # Store acceptable bases at the primary var index for this substitution
        ref1_bases_str = ",".join(sorted(int_to_base.get(v, "?") for v in ref1_ints))
        ref2_bases_str = ",".join(sorted(int_to_base.get(v, "?") for v in ref2_ints))
        ref1_acceptable_bases[vi1] = ref1_bases_str
        ref2_acceptable_bases[vi2] = ref2_bases_str

        # Position is informative only if the acceptable base sets are disjoint
        if ref1_ints & ref2_ints:
            n_collapsed += 1
            continue
        n_informative += 1
        is_informative[vi1] = True
        if vi2 != vi1:
            is_informative[vi2] = True

        # Pre-compute numpy arrays for fast membership testing
        ref1_arr = np.array(list(ref1_ints), dtype=seq_matrix.dtype)
        ref2_arr = np.array(list(ref2_ints), dtype=seq_matrix.dtype)

        # For each reference, use that reference's var index from the substitution map.
        # Reads aligned to seq1's reference use vi1; reads aligned to seq2's reference use vi2.
        for ref in ref_categories:
            seq_id = ref_to_seq.get(ref)
            if seq_id is None:
                continue
            var_idx = vi1 if seq_id == 1 else vi2

            ref_mask = ref_labels == ref

            read_bases = seq_matrix[ref_mask, var_idx]
            if has_span:
                covered = span_matrix[ref_mask, var_idx] > 0
            else:
                covered = np.ones(ref_mask.sum(), dtype=bool)

            calls = np.zeros(ref_mask.sum(), dtype=np.int8)
            calls[np.isin(read_bases, ref1_arr) & covered] = 1
            calls[np.isin(read_bases, ref2_arr) & covered] = 2
            calls[~covered | np.isin(read_bases, list(uninformative))] = 0

            result[ref_mask, var_idx] = calls

    logger.info(
        "Variant calling complete: %d informative, %d collapsed (overlapping base sets).",
        n_informative,
        n_collapsed,
    )

    adata.var[f"{output_prefix}_seq1_acceptable_bases"] = pd.Categorical(ref1_acceptable_bases)
    adata.var[f"{output_prefix}_seq2_acceptable_bases"] = pd.Categorical(ref2_acceptable_bases)
    adata.var[f"{output_prefix}_informative_site"] = is_informative

    adata.layers[layer_name] = result

    adata.uns[uns_flag] = True
    logger.info("Added variant call layer '%s'.", layer_name)


def append_variant_segment_layer(
    adata: "ad.AnnData",
    seq1_column: str,
    seq2_column: str,
    variant_call_layer: str | None = None,
    read_span_layer: str = "read_span_mask",
    reference_col: str = "Reference_strand",
    output_prefix: str | None = None,
    uns_flag: str = "append_variant_segment_layer_performed",
    force_redo: bool = False,
    bypass: bool = False,
) -> None:
    """Segment each read span into contiguous seq1/seq2 regions based on variant calls.

    Uses the per-position variant calls (1=seq1, 2=seq2) at informative mismatch
    sites to segment each read into contiguous regions.  At boundaries where the
    class switches, a putative breakpoint is placed at the midpoint between the
    two flanking mismatch positions.

    Values in the output layer:
      0 = outside read span (no coverage)
      1 = seq1 segment
      2 = seq2 segment
      3 = transition zone between different-class segments

    Per-read outputs in ``adata.obs``:
      - ``{output_prefix}_breakpoint_count``: number of class transitions.
      - ``{output_prefix}_variant_breakpoints``: list of putative breakpoint positions
        (midpoint between flanking informative mismatch sites).
      - ``variant_breakpoints``: alias of ``{output_prefix}_variant_breakpoints``.
      - ``{output_prefix}_variant_segment_cigar`` / ``variant_segment_cigar``:
        run-length string with ``S`` (self) and ``X`` (other).
      - ``{output_prefix}_variant_self_base_count`` / ``variant_self_base_count``:
        number of span bases labeled as self.
      - ``{output_prefix}_variant_other_base_count`` / ``variant_other_base_count``:
        number of span bases labeled as other.

    Args:
        adata: AnnData object.
        seq1_column: Column in ``adata.var`` with the first reference base.
        seq2_column: Column in ``adata.var`` with the second reference base.
        variant_call_layer: Layer with per-position variant calls.  Auto-derived if None.
        read_span_layer: Layer containing read span masks.
        reference_col: Obs column defining which reference each read is aligned to.
        output_prefix: Prefix for output layer/obs names.  Defaults to ``{seq1_column}__{seq2_column}``.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        bypass: Whether to skip processing.
    """
    if bypass:
        return

    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        return

    import pandas as pd

    output_prefix = output_prefix or f"{seq1_column}__{seq2_column}"
    if variant_call_layer is None:
        variant_call_layer = f"{output_prefix}_variant_call"

    if variant_call_layer not in adata.layers:
        logger.warning(
            "Variant call layer '%s' not found; skipping segment layer.", variant_call_layer
        )
        return

    has_span = read_span_layer in adata.layers
    if not has_span:
        logger.warning("Read span layer '%s' not found; skipping segment layer.", read_span_layer)
        return

    call_matrix = np.asarray(adata.layers[variant_call_layer])
    span_matrix = np.asarray(adata.layers[read_span_layer])
    n_obs, n_vars = adata.shape

    segment_layer = np.zeros((n_obs, n_vars), dtype=np.int8)
    breakpoint_counts = np.zeros(n_obs, dtype=np.int32)
    breakpoint_positions: list[list[float | int]] = [[] for _ in range(n_obs)]

    for i in range(n_obs):
        span_row = span_matrix[i]
        call_row = call_matrix[i]

        # Find read span boundaries
        covered = np.where(span_row > 0)[0]
        if len(covered) == 0:
            continue
        span_start = int(covered[0])
        span_end = int(covered[-1])

        # Collect informative positions (call == 1 or 2) within span
        informative_mask = (call_row == 1) | (call_row == 2)
        informative_positions = np.where(informative_mask)[0]
        # Restrict to within span
        informative_positions = informative_positions[
            (informative_positions >= span_start) & (informative_positions <= span_end)
        ]

        if len(informative_positions) == 0:
            # No informative sites — leave as 0 (no segment info)
            continue

        # Sort by position (should already be sorted)
        informative_positions = np.sort(informative_positions)
        classes = call_row[informative_positions]  # 1 or 2

        n_bp = 0
        row_breakpoints: list[float | int] = []
        # Walk through consecutive informative positions and fill segments
        prev_pos = informative_positions[0]
        prev_cls = int(classes[0])

        # Extend first class leftward to span start
        segment_layer[i, span_start:prev_pos] = prev_cls

        for k in range(1, len(informative_positions)):
            cur_pos = informative_positions[k]
            cur_cls = int(classes[k])

            if cur_cls == prev_cls:
                # Same class — fill from prev_pos to cur_pos
                segment_layer[i, prev_pos:cur_pos] = prev_cls
            else:
                # Class transition — fill gap between informative sites with transition value
                segment_layer[i, prev_pos] = prev_cls
                segment_layer[i, prev_pos + 1 : cur_pos] = 3
                n_bp += 1
                midpoint = (int(prev_pos) + int(cur_pos)) / 2.0
                row_breakpoints.append(int(midpoint) if midpoint.is_integer() else float(midpoint))

            prev_pos = cur_pos
            prev_cls = cur_cls

        # Fill the last informative position itself
        segment_layer[i, prev_pos] = prev_cls
        # Extend last class rightward to span end (inclusive)
        segment_layer[i, prev_pos : span_end + 1] = prev_cls
        # But re-mark breakpoints that may have been overwritten — they weren't,
        # since we only extend from prev_pos forward and breakpoints are before prev_pos.

        breakpoint_counts[i] = n_bp
        breakpoint_positions[i] = row_breakpoints

    layer_name = f"{output_prefix}_variant_segments"
    adata.layers[layer_name] = segment_layer

    adata.obs[f"{output_prefix}_breakpoint_count"] = breakpoint_counts
    adata.obs[f"{output_prefix}_variant_breakpoints"] = pd.Series(
        breakpoint_positions,
        index=adata.obs.index,
        dtype=object,
    )
    adata.obs["variant_breakpoints"] = pd.Series(
        breakpoint_positions,
        index=adata.obs.index,
        dtype=object,
    )
    adata.obs[f"{output_prefix}_is_chimeric"] = breakpoint_counts > 0

    # Per-read chimeric flags from mismatch segments relative to each read's own reference.
    # A mismatch segment is a contiguous run where a seq1-aligned read is labeled as seq2,
    # or vice versa, within the read span.
    ref_labels = adata.obs[reference_col].values
    ref_categories = adata.obs[reference_col].cat.categories
    suffix = "_strand_FASTA_base"
    seq1_stem = seq1_column[: -len(suffix)] if seq1_column.endswith(suffix) else seq1_column
    seq2_stem = seq2_column[: -len(suffix)] if seq2_column.endswith(suffix) else seq2_column

    ref_to_seq: dict[str, int] = {}
    for ref in ref_categories:
        if ref == seq1_stem:
            ref_to_seq[ref] = 1
        elif ref == seq2_stem:
            ref_to_seq[ref] = 2

    chimeric_flags = np.zeros(n_obs, dtype=bool)
    chimeric_types: list[str] = ["no_segment_mismatch"] * n_obs
    self_base_counts = np.zeros(n_obs, dtype=np.int32)
    other_base_counts = np.zeros(n_obs, dtype=np.int32)
    segment_cigars: list[str] = [""] * n_obs

    for i in range(n_obs):
        covered = np.where(span_matrix[i] > 0)[0]
        if len(covered) == 0:
            continue

        span_start = int(covered[0])
        span_end = int(covered[-1])
        in_span = segment_layer[i, span_start : span_end + 1]

        seq_id = ref_to_seq.get(ref_labels[i])
        if seq_id is None:
            continue

        mismatch_value = 2 if seq_id == 1 else 1
        self_value = seq_id
        mismatch_mask = in_span == mismatch_value

        self_base_counts[i] = int(np.sum(in_span == self_value))
        other_base_counts[i] = int(np.sum(in_span == mismatch_value))

        # Build an S/X run-length string from classified span positions.
        # Values 0 and 3 are treated as run boundaries and excluded from run lengths.
        runs: list[str] = []
        run_symbol: str | None = None
        run_len = 0
        for val in in_span:
            sym: str | None
            if val == self_value:
                sym = "S"
            elif val == mismatch_value:
                sym = "X"
            else:
                sym = None
            if sym is None:
                if run_symbol is not None and run_len > 0:
                    runs.append(f"{run_len}{run_symbol}")
                    run_symbol = None
                    run_len = 0
                continue
            if sym == run_symbol:
                run_len += 1
            else:
                if run_symbol is not None and run_len > 0:
                    runs.append(f"{run_len}{run_symbol}")
                run_symbol = sym
                run_len = 1
        if run_symbol is not None and run_len > 0:
            runs.append(f"{run_len}{run_symbol}")
        segment_cigars[i] = "".join(runs)

        if not np.any(mismatch_mask):
            continue

        starts = np.where(mismatch_mask & ~np.r_[False, mismatch_mask[:-1]])[0]
        ends = np.where(mismatch_mask & ~np.r_[mismatch_mask[1:], False])[0]
        n_segments = len(starts)
        chimeric_flags[i] = True

        if n_segments >= 2:
            chimeric_types[i] = "multi_segment_mismatch"
        else:
            start = int(starts[0])
            end = int(ends[0])
            if start == 0:
                chimeric_types[i] = "left_segment_mismatch"
            elif end == (len(in_span) - 1):
                chimeric_types[i] = "right_segment_mismatch"
            else:
                chimeric_types[i] = "middle_segment_mismatch"

    adata.obs["chimeric_variant_sites"] = chimeric_flags
    adata.obs["chimeric_variant_sites_type"] = pd.Categorical(
        chimeric_types,
        categories=[
            "no_segment_mismatch",
            "left_segment_mismatch",
            "right_segment_mismatch",
            "middle_segment_mismatch",
            "multi_segment_mismatch",
        ],
    )
    adata.obs[f"{output_prefix}_variant_segment_cigar"] = segment_cigars
    adata.obs["variant_segment_cigar"] = segment_cigars
    adata.obs[f"{output_prefix}_variant_self_base_count"] = self_base_counts
    adata.obs["variant_self_base_count"] = self_base_counts
    adata.obs[f"{output_prefix}_variant_other_base_count"] = other_base_counts
    adata.obs["variant_other_base_count"] = other_base_counts

    n_chimeric = int(np.sum(breakpoint_counts > 0))
    logger.info(
        "Variant segmentation complete: %d reads with breakpoints out of %d total.",
        n_chimeric,
        n_obs,
    )

    adata.uns[uns_flag] = True
    logger.info("Added variant segment layer '%s'.", layer_name)
