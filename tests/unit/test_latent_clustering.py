"""Per-embedding KNN + Leiden for the latent stage (`EGL-28a`).

The defect this replaces is subtle because it produces plausible output: Leiden
was computed once from UMAP's internal graph and reused for every strategy, so
a PCA embedding and an NMF embedding of the same molecules carried *identical*
labels. Nothing errors; the comparison between representations is simply
vacuous. These tests pin the independence that fixes it, and the provenance
that keeps a transferred label from reading as a fitted one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.tools.latent_clustering import (
    LABEL_SOURCE_FIT,
    LABEL_SOURCE_TRANSFERRED,
    build_knn_connectivities,
    cluster_all_embeddings,
    cluster_embedding,
    embedding_keys,
    parse_embedding_key,
    resolve_parameter,
    transfer_labels,
)

pytestmark = pytest.mark.unit


def _cfg(**overrides):
    base = dict(
        latent_knn_neighbors=10,
        # 0.3 recovers the fixture's ground truth; 1.0 over-partitions a tight
        # blob into five clusters. That sensitivity is the lane's premise, not
        # a defect -- see `test_resolution_choice_changes_the_answer`.
        latent_leiden_resolution=0.3,
        latent_knn_metric="euclidean",
        latent_knn_neighbors_by_strategy={},
        latent_leiden_resolution_by_strategy={},
        latent_knn_metric_by_strategy={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _two_blob_adata(n_per_blob=40, separation=20.0, seed=0):
    """Two well-separated blobs, embedded twice with *different* structure.

    `X_pca_s` separates the blobs; `X_nmf_s` deliberately does not. Any scheme
    that borrows one embedding's labels for the other cannot tell them apart.
    """
    import anndata as ad

    rng = np.random.default_rng(seed)
    n = 2 * n_per_blob
    separated = np.vstack(
        [rng.normal(0, 1, size=(n_per_blob, 2)), rng.normal(separation, 1, size=(n_per_blob, 2))]
    )
    mixed = rng.normal(0, 1, size=(n, 2))
    adata = ad.AnnData(
        X=np.zeros((n, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"m{index:04d}" for index in range(n)]),
    )
    adata.obsm["X_pca_s"] = separated.astype(np.float32)
    adata.obsm["X_nmf_s"] = mixed.astype(np.float32)
    return adata


# --- key handling ------------------------------------------------------------


def test_embedding_keys_are_listed_stably():
    adata = _two_blob_adata()
    assert embedding_keys(adata) == ["X_nmf_s", "X_pca_s"]


def test_key_parsing_splits_strategy_from_suffix():
    assert parse_embedding_key("X_cp_full_ohe_sequence_N_masked") == (
        "cp",
        "full_ohe_sequence_N_masked",
    )
    assert parse_embedding_key("X_umap_shared_valid_mod_sites") == (
        "umap",
        "shared_valid_mod_sites",
    )


# --- parameter resolution ----------------------------------------------------


def test_per_strategy_override_wins_over_the_shared_knob():
    """The right resolution for a 10-component PCA and a 2-component UMAP differ."""
    cfg = _cfg(latent_leiden_resolution_by_strategy={"pca": 0.5})
    assert resolve_parameter(cfg, "pca", "leiden_resolution", 0.1) == 0.5
    assert resolve_parameter(cfg, "umap", "leiden_resolution", 0.1) == 0.3


def test_shared_knob_applies_when_no_override_exists():
    """Existing configs must keep behaving exactly as they do now."""
    assert resolve_parameter(_cfg(), "cp", "knn_neighbors", 15) == 10


# --- graph construction ------------------------------------------------------


def test_knn_graph_is_square_and_symmetric():
    points = np.random.default_rng(0).normal(size=(30, 2))
    graph = build_knn_connectivities(points, 5)
    assert graph.shape == (30, 30)
    assert (graph != graph.T).nnz == 0, "neighbourhood must be mutual for undirected Leiden"


def test_neighbor_count_is_clamped_to_the_population():
    """A k above the population is an error in sklearn, not a saturated graph."""
    points = np.random.default_rng(0).normal(size=(4, 2))
    assert build_knn_connectivities(points, 100).shape == (4, 4)


# --- label transfer ----------------------------------------------------------


def test_transfer_is_a_vote_not_a_single_nearest_neighbour():
    """1-NN cannot express that a molecule sits between two clusters.

    Here the query is nearest to a lone "b" but sits inside a crowd of "a"s;
    a vote reports "a" with a confidence below 1, which is the signal that
    makes a suspicious heatmap block checkable.
    """
    fit_points = np.array([[0.0], [0.1], [0.2], [0.3], [0.55]])
    fit_labels = np.array(["a", "a", "a", "a", "b"])
    labels, confidence = transfer_labels(np.array([[0.5]]), fit_points, fit_labels, n_neighbors=5)
    assert labels[0] == "a"
    assert 0.0 < confidence[0] < 1.0


def test_transfer_confidence_is_the_agreeing_fraction():
    fit_points = np.array([[0.0], [1.0], [2.0], [3.0]])
    fit_labels = np.array(["a", "a", "a", "b"])
    _labels, confidence = transfer_labels(np.array([[1.0]]), fit_points, fit_labels, n_neighbors=4)
    assert confidence[0] == pytest.approx(0.75)


def test_transfer_without_any_fit_points_is_unassigned_not_a_guess():
    labels, confidence = transfer_labels(
        np.zeros((3, 1)), np.empty((0, 1)), np.array([]), n_neighbors=3
    )
    assert set(labels) == {"unassigned"}
    assert not confidence.any()


# --- the crux: independence --------------------------------------------------


def test_each_embedding_is_clustered_in_its_own_coordinates():
    """The defect being fixed: one clustering wearing several names.

    `X_pca_s` separates two blobs, `X_nmf_s` does not. If labels were borrowed
    across embeddings these would agree; they must not.
    """
    adata = _two_blob_adata()
    fit_indices = np.arange(adata.n_obs)
    cluster_all_embeddings(adata, fit_indices=fit_indices, cfg=_cfg())

    pca_labels = adata.obs["leiden_pca_s"].astype(str).to_numpy()
    truth = np.array(["blob0"] * 40 + ["blob1"] * 40)
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(pca_labels, truth) > 0.9, "the separating embedding must find them"
    assert "leiden_nmf_s" in adata.obs
    assert (
        adjusted_rand_score(pca_labels, adata.obs["leiden_nmf_s"].astype(str).to_numpy()) < 0.5
    ), "the non-separating embedding must not inherit the separation"


def test_fit_rows_are_marked_fit_with_full_confidence():
    adata = _two_blob_adata()
    cluster_embedding(adata, "X_pca_s", fit_indices=np.arange(adata.n_obs), cfg=_cfg())

    assert set(adata.obs["leiden_pca_s_label_source"]) == {LABEL_SOURCE_FIT}
    assert (adata.obs["leiden_pca_s_label_confidence"] == 1.0).all()


def test_unfit_rows_are_marked_transferred():
    """A clustermap must be able to distinguish fitted from proximity-assigned."""
    adata = _two_blob_adata()
    fit_indices = np.concatenate([np.arange(30), np.arange(40, 70)])
    record = cluster_embedding(adata, "X_pca_s", fit_indices=fit_indices, cfg=_cfg())

    source = adata.obs["leiden_pca_s_label_source"].astype(str).to_numpy()
    assert record["transferred_read_count"] == 20
    assert (source[fit_indices] == LABEL_SOURCE_FIT).all()
    unfit = np.setdiff1d(np.arange(adata.n_obs), fit_indices)
    assert (source[unfit] == LABEL_SOURCE_TRANSFERRED).all()


def test_transferred_labels_do_not_cross_the_gap():
    """Transfer must be correct, not merely marked as transferred.

    Asserting each blob ends up a *single* cluster would be too strong: the fit
    subset can legitimately split one blob at a given resolution. The property
    that must hold regardless is that no molecule is assigned a label belonging
    exclusively to the far blob -- a transfer that jumps a 20-sigma gap is
    wrong however the fit clustering came out.
    """
    adata = _two_blob_adata()
    fit_indices = np.concatenate([np.arange(30), np.arange(40, 70)])
    cluster_embedding(adata, "X_pca_s", fit_indices=fit_indices, cfg=_cfg())

    labels = adata.obs["leiden_pca_s"].astype(str).to_numpy()
    near_fit = set(labels[np.arange(30)])
    far_fit = set(labels[np.arange(40, 70)])
    assert near_fit.isdisjoint(far_fit), "the blobs must not share a fitted cluster"

    transferred_near = set(labels[np.arange(30, 40)])
    transferred_far = set(labels[np.arange(70, 80)])
    assert transferred_near <= near_fit
    assert transferred_far <= far_fit


def test_parameters_are_recorded_for_the_run():
    adata = _two_blob_adata()
    record = cluster_embedding(adata, "X_pca_s", fit_indices=np.arange(adata.n_obs), cfg=_cfg())

    assert record["strategy"] == "pca"
    assert record["n_neighbors"] == 10
    assert adata.uns["leiden_pca_s_params"]["resolution"] == 0.3


def test_embedding_too_small_to_cluster_is_skipped():
    import anndata as ad

    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata.obsm["X_pca_s"] = np.zeros((2, 2), dtype=np.float32)
    assert cluster_embedding(adata, "X_pca_s", fit_indices=np.arange(2), cfg=_cfg()) is None
    assert "leiden_pca_s" not in adata.obs


def test_config_exposes_the_per_strategy_knobs():
    from smftools.config.experiment_config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.latent_cluster_embeddings is True
    assert cfg.latent_knn_neighbors_by_strategy == {}
    # Resolution ships populated -- measured per strategy against a 4-10
    # cluster target; see `test_latent_resolution_defaults`. Neighbours and
    # metric ship empty because no measurement justified splitting them.
    assert set(cfg.latent_leiden_resolution_by_strategy) == {"pca", "umap", "nmf", "cp"}


def test_resolution_choice_changes_the_answer():
    """Why the per-strategy knobs exist, pinned rather than left as folklore.

    One resolution cannot serve every embedding: on two well-separated blobs,
    0.3 recovers them exactly and 1.0 splits them into five. On the real DAF
    unit the same spread appears across strategies at a single resolution --
    PCA collapsed to one cluster while a CP embedding produced 39.
    """
    from sklearn.metrics import adjusted_rand_score

    truth = np.array(["blob0"] * 40 + ["blob1"] * 40)
    scores = {}
    for resolution in (0.3, 1.0):
        adata = _two_blob_adata()
        cluster_embedding(
            adata,
            "X_pca_s",
            fit_indices=np.arange(adata.n_obs),
            cfg=_cfg(latent_leiden_resolution=resolution),
        )
        scores[resolution] = adjusted_rand_score(
            adata.obs["leiden_pca_s"].astype(str).to_numpy(), truth
        )

    assert scores[0.3] > 0.99
    assert scores[1.0] < scores[0.3]


def test_cluster_records_are_a_mapping_not_a_list():
    """`uns` is written to zarr, which cannot serialize a list of dicts.

    Storing the records as a list published fine in memory and failed only at
    write time, so the shape is pinned here rather than left to the executor
    tests to catch.
    """
    adata = _two_blob_adata()
    records = cluster_all_embeddings(adata, fit_indices=np.arange(adata.n_obs), cfg=_cfg())
    mapping = {str(record["embedding"]): dict(record) for record in records}

    assert set(mapping) == {"X_pca_s", "X_nmf_s"}
    assert all(isinstance(value, dict) for value in mapping.values())
