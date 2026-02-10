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


def plot_read_current_traces(
    adata,
    read_ids: list[str],
    reference_start: Optional[int] = None,
    reference_end: Optional[int] = None,
    use_moves: bool = True,
    figsize: tuple = (15, 4),
    title: Optional[str] = None,
):
    """Plot raw current signal from POD5 files for specified reads.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with obs['fn'] (POD5 filenames) and uns['bam_paths']
        (experiment-specific BAM file paths).
    read_ids : list of str
        List of read IDs to plot.
    reference_start : int, optional
        Start position in reference coordinates (0-based). If provided with
        reference_end, only plot signal for this genomic region.
    reference_end : int, optional
        End position in reference coordinates (0-based, exclusive).
    use_moves : bool, default True
        If True and move table (mv tag) is available, use it for precise
        signal-to-base mapping. If False or mv tag unavailable, use proportional
        estimation.
    figsize : tuple, default (15, 4)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, uses read ID.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    Notes
    -----
    Requires POD5 files to be accessible at the paths stored in obs['fn'].
    For precise signal mapping, requires BAM files with move tables (mv tag)
    generated using dorado's --emit-moves flag.

    Examples
    --------
    >>> plot_read_current_traces(adata, ["read_001", "read_002"],
    ...                          reference_start=1000, reference_end=1500)
    """
    pysam = require("pysam", extra="bam", purpose="reading BAM files")
    pod5 = require("pod5", extra="ont", purpose="reading POD5 files")

    # Check required data
    if "fn" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'fn' column with POD5 filenames")
    if "bam_paths" not in adata.uns:
        raise ValueError(
            "adata.uns must contain 'bam_paths' dict with experiment-specific BAM paths"
        )

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
        pod5_filename = read_obs.get("fn")
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
                signal_region = _map_reference_to_signal(
                    bam_path, read_id, reference_start, reference_end, signal, use_moves
                )
                if signal_region is None:
                    ax.text(
                        0.5,
                        0.5,
                        f"Could not map region for {read_id}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue
                signal = signal_region

            # Plot signal
            ax.plot(signal, linewidth=0.5)
            ax.set_ylabel("Current (pA)")
            ax.set_xlabel("Signal index")

            if title:
                ax.set_title(title if n_reads == 1 else f"{title} - {read_id}")
            else:
                ax.set_title(read_id)

        except Exception as e:
            logger.error(f"Error plotting read {read_id}: {e}")
            ax.text(
                0.5, 0.5, f"Error: {str(e)[:50]}", ha="center", va="center", transform=ax.transAxes
            )

    plt.tight_layout()
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
                if read.read_id == read_id:
                    return read.signal
        logger.warning(f"Read {read_id} not found in {pod5_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading POD5 file {pod5_path}: {e}")
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
    signal_region : np.ndarray or None
        Signal for the specified reference region, or None if mapping failed.
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

                    # Get move table if available and requested
                    moves = None
                    if use_moves and read.has_tag("mv"):
                        moves = np.array(read.get_tag("mv"), dtype=np.uint8)

                    # Map coordinates
                    if moves is not None:
                        return _map_with_moves(read, ref_start, ref_end, full_signal, moves)
                    else:
                        return _map_proportional(read, ref_start, ref_end, full_signal)

        logger.warning(f"Read {read_id} not found in BAM file")
        return None
    except Exception as e:
        logger.error(f"Error mapping coordinates for {read_id}: {e}")
        return None


def _map_with_moves(
    read, ref_start: int, ref_end: int, signal: np.ndarray, moves: np.ndarray
) -> Optional[np.ndarray]:
    """Map reference to signal using move table."""
    # Convert reference coordinates to query coordinates
    ref_pos = read.reference_start
    query_pos = 0
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

    # Use move table to map query bases to signal
    signal_start = np.sum(moves[:query_start])
    signal_end = np.sum(moves[:query_end])

    return signal[signal_start:signal_end]


def _map_proportional(
    read, ref_start: int, ref_end: int, signal: np.ndarray
) -> Optional[np.ndarray]:
    """Map reference to signal using proportional estimation."""
    # Get query positions for reference region
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

    # Proportional mapping
    query_length = read.query_length
    signal_length = len(signal)
    stride = signal_length / query_length

    signal_start = int(query_start * stride)
    signal_end = int(query_end * stride)

    return signal[signal_start:signal_end]
