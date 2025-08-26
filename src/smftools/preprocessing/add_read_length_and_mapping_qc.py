import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, List, Dict, Union

def add_read_length_and_mapping_qc(
    adata,
    bam_files: Optional[List[str]] = None,
    read_metrics: Optional[Dict[str, Union[list, tuple]]] = None,
    uns_flag: str = "read_lenth_and_mapping_qc_performed",
    extract_read_features_from_bam_callable = None,
    bypass: bool = False,
    force_redo: bool = True
):
    """
    Populate adata.obs with read/mapping QC columns.

    Parameters
    ----------
    adata
        AnnData to annotate (modified in-place).
    bam_files
        Optional list of BAM files to extract metrics from. Ignored if read_metrics supplied.
    read_metrics
        Optional dict mapping obs_name -> [read_length, read_quality, reference_length, mapped_length, mapping_quality]
        If provided, this will be used directly and bam_files will be ignored.
    uns_flag
        key in final_adata.uns used to record that QC was performed (kept the name with original misspelling).
    extract_read_features_from_bam_callable
        Optional callable(bam_path) -> dict mapping read_name -> list/tuple of metrics.
        If not provided and bam_files is given, function will attempt to call `extract_read_features_from_bam`
        from the global namespace (your existing helper).
    Returns
    -------
    None (mutates final_adata in-place)
    """

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        # QC already performed; nothing to do
        return

    # Build read_metrics dict either from provided arg or by extracting from bam files
    if read_metrics is None:
        read_metrics = {}
        if bam_files:
            extractor = extract_read_features_from_bam_callable or globals().get("extract_read_features_from_bam")
            if extractor is None:
                raise ValueError("No `read_metrics` provided and `extract_read_features_from_bam` not found.")
            for bam in bam_files:
                bam_read_metrics = extractor(bam)
                if not isinstance(bam_read_metrics, dict):
                    raise ValueError(f"extract_read_features_from_bam returned non-dict for {bam}")
                read_metrics.update(bam_read_metrics)
        else:
            # nothing to do
            read_metrics = {}

    # Convert read_metrics dict -> DataFrame (rows = read id)
    # Values may be lists/tuples or scalars; prefer lists/tuples with 5 entries.
    if len(read_metrics) == 0:
        # fill with NaNs
        n = adata.n_obs
        adata.obs['read_length'] = np.full(n, np.nan)
        adata.obs['mapped_length'] = np.full(n, np.nan)
        adata.obs['reference_length'] = np.full(n, np.nan)
        adata.obs['read_quality'] = np.full(n, np.nan)
        adata.obs['mapping_quality'] = np.full(n, np.nan)
    else:
        # Build DF robustly
        # Convert values to lists where possible, else to [val, val, val...]
        max_cols = 5
        rows = {}
        for k, v in read_metrics.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals = list(v)
            else:
                # scalar -> replicate into 5 columns to preserve original behavior
                vals = [v] * max_cols
            # Ensure length >= 5
            if len(vals) < max_cols:
                vals = vals + [np.nan] * (max_cols - len(vals))
            rows[k] = vals[:max_cols]

        df = pd.DataFrame.from_dict(rows, orient='index', columns=[
            'read_length', 'read_quality', 'reference_length', 'mapped_length', 'mapping_quality'
        ])

        # Reindex to final_adata.obs_names so order matches adata
        # If obs_names are not present as keys in df, the results will be NaN
        df_reindexed = df.reindex(adata.obs_names).astype(float)

        adata.obs['read_length'] = df_reindexed['read_length'].values
        adata.obs['mapped_length'] = df_reindexed['mapped_length'].values
        adata.obs['reference_length'] = df_reindexed['reference_length'].values
        adata.obs['read_quality'] = df_reindexed['read_quality'].values
        adata.obs['mapping_quality'] = df_reindexed['mapping_quality'].values

    # Compute ratio columns safely (avoid divide-by-zero and preserve NaN)
    # read_length_to_reference_length_ratio
    rl = pd.to_numeric(adata.obs['read_length'], errors='coerce').to_numpy(dtype=float)
    ref_len = pd.to_numeric(adata.obs['reference_length'], errors='coerce').to_numpy(dtype=float)
    mapped_len = pd.to_numeric(adata.obs['mapped_length'], errors='coerce').to_numpy(dtype=float)

    # safe divisions: use np.where to avoid warnings and replace inf with nan
    with np.errstate(divide='ignore', invalid='ignore'):
        rl_to_ref = np.where((ref_len != 0) & np.isfinite(ref_len), rl / ref_len, np.nan)
        mapped_to_ref = np.where((ref_len != 0) & np.isfinite(ref_len), mapped_len / ref_len, np.nan)
        mapped_to_read = np.where((rl != 0) & np.isfinite(rl), mapped_len / rl, np.nan)

    adata.obs['read_length_to_reference_length_ratio'] = rl_to_ref
    adata.obs['mapped_length_to_reference_length_ratio'] = mapped_to_ref
    adata.obs['mapped_length_to_read_length_ratio'] = mapped_to_read

    # Add read level raw modification signal: sum over X rows
    X = adata.X
    if sp.issparse(X):
        # sum returns (n_obs, 1) sparse matrix; convert to 1d array
        raw_sig = np.asarray(X.sum(axis=1)).ravel()
    else:
        raw_sig = np.asarray(X.sum(axis=1)).ravel()

    adata.obs['Raw_modification_signal'] = raw_sig

    # mark as done
    adata.uns[uns_flag] = True

    return None