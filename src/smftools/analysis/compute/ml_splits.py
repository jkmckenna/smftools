"""
Matrix/table-level split helpers for machine-learning evaluation.

These helpers operate on metadata tables and split indices only. They do not
touch AnnData or perform any file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_disjoint_groups(
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    metadata_df: pd.DataFrame,
    group_col: str,
) -> bool:
    """
    Return True when the train and test splits do not share any holdout groups.
    """
    train_groups = set(metadata_df.iloc[list(train_idx)][group_col].dropna().tolist())
    test_groups = set(metadata_df.iloc[list(test_idx)][group_col].dropna().tolist())
    return train_groups.isdisjoint(test_groups)


def summarize_split(
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    metadata_df: pd.DataFrame,
    label_col: str,
    extra_group_cols: list[str] | None = None,
) -> dict:
    """
    Summarize a train/test split from row metadata.

    For binary numeric labels in ``{0, 1}``, also emits ``n_pos_*`` and
    ``n_neg_*`` convenience counts.
    """
    train_df = metadata_df.iloc[list(train_idx)]
    test_df = metadata_df.iloc[list(test_idx)]

    out = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

    labels = pd.to_numeric(metadata_df[label_col], errors="coerce")
    finite_labels = labels[np.isfinite(labels)]
    is_binary_01 = (
        set(pd.unique(finite_labels.astype(int))) <= {0, 1} if len(finite_labels) else False
    )
    if is_binary_01:
        train_y = pd.to_numeric(train_df[label_col], errors="coerce").to_numpy(dtype=float)
        test_y = pd.to_numeric(test_df[label_col], errors="coerce").to_numpy(dtype=float)
        out["n_pos_train"] = int(np.nansum(train_y == 1))
        out["n_neg_train"] = int(np.nansum(train_y == 0))
        out["n_pos_test"] = int(np.nansum(test_y == 1))
        out["n_neg_test"] = int(np.nansum(test_y == 0))

    if extra_group_cols:
        for col in extra_group_cols:
            train_vals = sorted(pd.unique(train_df[col].dropna()).tolist())
            test_vals = sorted(pd.unique(test_df[col].dropna()).tolist())
            out[f"train_{col}s"] = ",".join(map(str, train_vals))
            out[f"test_{col}s"] = ",".join(map(str, test_vals))

    return out


def build_leave_one_group_out_splits(
    metadata_df: pd.DataFrame,
    group_col: str,
    label_col: str,
    sort_groups: bool = True,
) -> list[dict]:
    """
    Build leave-one-group-out folds from a metadata table.

    Each fold holds out one unique ``group_col`` value as test and uses all
    remaining rows as train. Folds that do not contain both classes in both
    train and test are omitted.
    """
    groups = pd.Series(metadata_df[group_col]).dropna().unique().tolist()
    if sort_groups:
        groups = sorted(groups)

    y_all = metadata_df[label_col].to_numpy()
    folds: list[dict] = []

    for fold_id, heldout_group in enumerate(groups):
        test_idx = np.flatnonzero(metadata_df[group_col].to_numpy() == heldout_group)
        train_idx = np.flatnonzero(metadata_df[group_col].to_numpy() != heldout_group)
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if len(np.unique(y_all[train_idx])) < 2 or len(np.unique(y_all[test_idx])) < 2:
            continue
        if not validate_disjoint_groups(train_idx, test_idx, metadata_df, group_col):
            raise ValueError(f"Group leakage detected for held-out group {heldout_group!r}")
        folds.append(
            {
                "fold_id": int(fold_id),
                "heldout_group": heldout_group,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

    return folds
