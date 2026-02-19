"""POD5 signal plotting utilities for nanopore reads."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

logger = get_logger(__name__)


def _get_reference_coordinates(adata) -> np.ndarray:
    """Return reference coordinates from adata.var, preferring Original_var_names.

    If the adata was reindexed, ``adata.var["Original_var_names"]`` holds the
    original reference coordinates; otherwise ``adata.var_names`` is used.
    """
    if "Original_var_names" in adata.var.columns:
        coords = np.asarray(adata.var["Original_var_names"])
        try:
            return coords.astype(int)
        except (ValueError, TypeError):
            pass
    return np.asarray(adata.var_names).astype(int)


def plot_read_current_traces(
    adata,
    read_ids: list[str],
    reference_start: Optional[int] = None,
    reference_end: Optional[int] = None,
    var_start: Optional[int] = None,
    var_end: Optional[int] = None,
    use_moves: bool = True,
    figsize: tuple = (15, 4),
    title: Optional[str] = None,
    pod5_dir: str | Path | None = None,
    save_path: str | Path | None = None,
):
    """Plot raw current signal from POD5 files for specified reads.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with obs['fn'] or obs['pod5_origin'] (POD5 filenames)
        and uns['bam_paths'] (experiment-specific BAM file paths).
    read_ids : list of str
        List of read IDs to plot.
    reference_start : int, optional
        Start position in reference coordinates (0-based). If provided with
        reference_end, only plot signal for this genomic region.
    reference_end : int, optional
        End position in reference coordinates (0-based, exclusive).
    var_start : int, optional
        Start var index (0-based) into adata.var. The corresponding reference
        coordinate is looked up from ``adata.var["Original_var_names"]`` (if
        the adata was reindexed) or ``adata.var_names``. Takes precedence
        over ``reference_start`` / ``reference_end`` when both are provided.
    var_end : int, optional
        End var index (0-based, exclusive).
    use_moves : bool, default True
        If True and move table (mv tag) is available, use it for precise
        signal-to-base mapping. If False or mv tag unavailable, use proportional
        estimation.
    figsize : tuple, default (15, 4)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, uses read ID.
    pod5_dir : str or Path, optional
        Directory containing POD5 files. When provided, the basename from
        obs['fn'] / obs['pod5_origin'] is joined with this directory to form
        the absolute path.
    save_path : str or Path, optional
        If provided, save the figure to this path as PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    Notes
    -----
    Requires POD5 files to be accessible at the paths stored in obs['fn']
    or obs['pod5_origin']. For precise signal mapping, requires BAM files
    with move tables (mv tag) generated using dorado's --emit-moves flag.

    Examples
    --------
    >>> plot_read_current_traces(adata, ["read_001", "read_002"],
    ...                          reference_start=1000, reference_end=1500)
    >>> # Or use var indices to plot signal for a span of adata columns:
    >>> plot_read_current_traces(adata, ["read_001"], var_start=0, var_end=100)
    """
    pysam = require("pysam", extra="bam", purpose="reading BAM files")
    pod5 = require("pod5", extra="ont", purpose="reading POD5 files")

    # Resolve var indices to reference coordinates
    if var_start is not None and var_end is not None:
        ref_coords = _get_reference_coordinates(adata)
        reference_start = int(ref_coords[var_start])
        reference_end = int(ref_coords[min(var_end, len(ref_coords)) - 1]) + 1
        logger.info(
            f"Resolved var span [{var_start}:{var_end}] to reference "
            f"coordinates [{reference_start}:{reference_end})"
        )

    # Check required data
    has_fn = "fn" in adata.obs.columns
    has_pod5_origin = "pod5_origin" in adata.obs.columns
    if not has_fn and not has_pod5_origin:
        raise ValueError("adata.obs must contain 'fn' or 'pod5_origin' column with POD5 filenames")
    if "bam_paths" not in adata.uns:
        raise ValueError(
            "adata.uns must contain 'bam_paths' dict with experiment-specific BAM paths"
        )
    pod5_col = "fn" if has_fn else "pod5_origin"

    n_reads = len(read_ids)
    fig, axes = plt.subplots(n_reads, 1, figsize=figsize, squeeze=False)

    for idx, read_id in enumerate(read_ids):
        ax = axes[idx, 0]

        # Get read information from adata
        if read_id not in adata.obs_names:
            logger.warning(f"Read {read_id} not found in adata, skipping")
            ax.text(
                0.5,
                0.5,
                f"Read {read_id} not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        read_obs = adata.obs.loc[read_id]
        pod5_filename = read_obs.get(pod5_col)
        experiment_name = read_obs.get("Experiment_name")

        if pd.isna(pod5_filename):
            logger.warning(f"No POD5 filename for read {read_id}, skipping")
            ax.text(
                0.5,
                0.5,
                f"No POD5 file for {read_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Determine BAM file path
        bam_path = None
        if experiment_name:
            # Try aligned first, then unaligned
            aligned_key = f"{experiment_name}_aligned"
            unaligned_key = f"{experiment_name}_unaligned"
            bam_path = adata.uns["bam_paths"].get(aligned_key) or adata.uns["bam_paths"].get(
                unaligned_key
            )

        if not bam_path:
            logger.warning(
                f"No BAM path found for experiment {experiment_name}, skipping {read_id}"
            )
            ax.text(
                0.5,
                0.5,
                f"No BAM path for {read_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        try:
            # Resolve POD5 path: if pod5_dir provided, join with basename
            if pod5_dir is not None:
                pod5_filename = str(Path(pod5_dir) / Path(pod5_filename).name)

            # Extract signal from POD5
            signal = _extract_signal_from_pod5(pod5_filename, read_id)
            if signal is None:
                ax.text(
                    0.5,
                    0.5,
                    f"Could not extract signal for {read_id}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Get alignment info and optionally map to reference coordinates
            if reference_start is not None and reference_end is not None:
                result = _map_reference_to_signal(
                    bam_path, read_id, reference_start, reference_end, signal, use_moves
                )
                if result is None:
                    ax.text(
                        0.5,
                        0.5,
                        f"Could not map region for {read_id}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue
                signal, mapping_method = result
            else:
                # Full signal — check strand and flip for reverse reads
                is_rev = _check_is_reverse(bam_path, read_id)
                if is_rev is True:
                    signal = signal[::-1]
                strand = "reverse" if is_rev else "forward"
                mapping_method = f"full signal, {strand}"

            # Plot signal
            ax.plot(signal, linewidth=0.5)
            ax.set_ylabel("Current (pA)")
            ax.set_xlabel("Signal index")

            # Build title with mapping method
            method_label = f"[{mapping_method}]"
            if title:
                base = title if n_reads == 1 else f"{title} - {read_id}"
            else:
                base = read_id
            ax.set_title(f"{base} {method_label}")

        except Exception as e:
            logger.error(f"Error plotting read {read_id}: {e}")
            ax.text(
                0.5, 0.5, f"Error: {str(e)[:50]}", ha="center", va="center", transform=ax.transAxes
            )

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved current trace plot to {save_path}")
    return fig


def _extract_signal_from_pod5(pod5_path: str, read_id: str) -> Optional[np.ndarray]:
    """Extract raw signal from POD5 file for a specific read.

    Parameters
    ----------
    pod5_path : str
        Path to POD5 file.
    read_id : str
        Read ID to extract.

    Returns
    -------
    signal : np.ndarray or None
        Raw current signal, or None if not found.
    """
    pod5 = require("pod5", extra="ont", purpose="reading POD5 files")

    pod5_path = Path(pod5_path)
    if not pod5_path.exists():
        logger.error(f"POD5 file not found: {pod5_path}")
        return None

    try:
        with pod5.Reader(pod5_path) as reader:
            for read in reader.reads():
                if str(read.read_id) == read_id:
                    return read.signal
        logger.warning(f"Read {read_id} not found in {pod5_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading POD5 file {pod5_path}: {e}")
        return None


def _check_is_reverse(bam_path: str, read_id: str) -> Optional[bool]:
    """Check whether a read is mapped to the reverse strand.

    Returns True (reverse), False (forward), or None (not found / unmapped).
    """
    pysam = require("pysam", extra="bam", purpose="reading BAM files")
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.query_name == read_id:
                    if read.is_unmapped:
                        return None
                    return read.is_reverse
    except Exception as e:
        logger.error(f"Error checking strand for {read_id}: {e}")
    return None


def _map_reference_to_signal(
    bam_path: str,
    read_id: str,
    ref_start: int,
    ref_end: int,
    full_signal: np.ndarray,
    use_moves: bool = True,
) -> Optional[np.ndarray]:
    """Map reference coordinates to signal indices.

    Parameters
    ----------
    bam_path : str
        Path to BAM file.
    read_id : str
        Read ID.
    ref_start : int
        Reference start position (0-based).
    ref_end : int
        Reference end position (0-based, exclusive).
    full_signal : np.ndarray
        Full signal array.
    use_moves : bool, default True
        Whether to use move table if available.

    Returns
    -------
    result : tuple[np.ndarray, str] or None
        Tuple of (signal_region, method) where method is ``"moves"`` or
        ``"proportional"``, or *None* if mapping failed.
    """
    pysam = require("pysam", extra="bam", purpose="reading BAM files")

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.query_name == read_id:
                    # Check if read is aligned
                    if read.is_unmapped:
                        logger.warning(f"Read {read_id} is unmapped, cannot map to reference")
                        return None

                    is_reverse = read.is_reverse
                    strand = "reverse" if is_reverse else "forward"

                    # Get move table if available and requested
                    moves = None
                    if use_moves and read.has_tag("mv"):
                        moves = np.array(read.get_tag("mv"), dtype=np.uint8)

                    # Map coordinates
                    if moves is not None:
                        sig = _map_with_moves(
                            read, ref_start, ref_end, full_signal, moves, is_reverse
                        )
                        return (sig, f"moves, {strand}") if sig is not None else None
                    else:
                        sig = _map_proportional(read, ref_start, ref_end, full_signal, is_reverse)
                        return (sig, f"proportional, {strand}") if sig is not None else None

        logger.warning(f"Read {read_id} not found in BAM file")
        return None
    except Exception as e:
        logger.error(f"Error mapping coordinates for {read_id}: {e}")
        return None


def _map_with_moves(
    read,
    ref_start: int,
    ref_end: int,
    signal: np.ndarray,
    moves: np.ndarray,
    is_reverse: bool = False,
) -> Optional[np.ndarray]:
    """Map reference to signal using move table.

    For reverse-strand reads, ``get_aligned_pairs()`` returns query positions
    in the BAM orientation (reverse-complemented).  The move table, however,
    indexes bases in the *original basecall order* (same direction as the raw
    signal).  We therefore flip query positions for reverse reads before
    indexing into the move table.
    """
    query_length = read.query_length
    query_start = None
    query_end = None

    for qp, rp in read.get_aligned_pairs():
        if rp is not None:
            if rp >= ref_start and query_start is None:
                query_start = qp
            if rp >= ref_end:
                query_end = qp
                break

    if query_start is None or query_end is None:
        logger.warning("Could not map reference region to query")
        return None

    # Convert BAM query positions to signal-order (original basecall) positions
    if is_reverse:
        # BAM qpos 0 = last base in original signal order
        orig_start = query_length - 1 - query_end
        orig_end = query_length - 1 - query_start
    else:
        orig_start = query_start
        orig_end = query_end

    # Use move table to map original-order bases to signal indices
    signal_start = int(np.sum(moves[:orig_start]))
    signal_end = int(np.sum(moves[:orig_end]))

    return signal[signal_start:signal_end]


def _map_proportional(
    read,
    ref_start: int,
    ref_end: int,
    signal: np.ndarray,
    is_reverse: bool = False,
) -> Optional[np.ndarray]:
    """Map reference to signal using proportional estimation.

    For reverse-strand reads, BAM query positions are flipped back to the
    original basecall order before computing proportional signal indices.
    """
    query_length = read.query_length
    query_start = None
    query_end = None

    for qp, rp in read.get_aligned_pairs():
        if rp is not None:
            if rp >= ref_start and query_start is None:
                query_start = qp
            if rp >= ref_end:
                query_end = qp
                break

    if query_start is None or query_end is None:
        logger.warning("Could not map reference region to query")
        return None

    # Convert BAM query positions to signal-order positions for reverse reads
    if is_reverse:
        orig_start = query_length - 1 - query_end
        orig_end = query_length - 1 - query_start
    else:
        orig_start = query_start
        orig_end = query_end

    # Proportional mapping
    signal_length = len(signal)
    stride = signal_length / query_length

    signal_start = int(orig_start * stride)
    signal_end = int(orig_end * stride)

    return signal[signal_start:signal_end]
