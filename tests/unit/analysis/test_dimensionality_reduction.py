import numpy as np
import pytest

from smftools.analysis.compute.dimensionality_reduction import (
    cluster_from_pca,
    run_pipeline,
    umap_from_pca,
)


def _make_clustered_features(n_reads=60, n_features=20, seed=0):
    """Two well-separated Gaussian blobs in feature space -- reliably produces >=2
    Leiden clusters and a stable PCA/UMAP fit."""
    rng = np.random.default_rng(seed)
    half = n_reads // 2
    blob_a = rng.normal(loc=0.0, scale=0.5, size=(half, n_features))
    blob_b = rng.normal(loc=8.0, scale=0.5, size=(n_reads - half, n_features))
    return np.vstack([blob_a, blob_b])


@pytest.mark.unit
def test_run_pipeline_returns_fitted_models():
    feat = _make_clustered_features()
    result = run_pipeline(feat, min_reads=5)

    assert result is not None
    X_pca, X_umap, clusters, explained_variance_ratio, pca_model, umap_model = result
    assert X_pca.shape[0] == feat.shape[0]
    assert X_umap.shape == (feat.shape[0], 2)
    assert clusters.shape[0] == feat.shape[0]
    assert explained_variance_ratio.shape[0] == X_pca.shape[1]

    # Fitted models are real, usable transformers, not just placeholders.
    assert hasattr(pca_model, "transform")
    assert hasattr(umap_model, "transform")


@pytest.mark.unit
def test_run_pipeline_pca_model_transforms_new_data_into_same_space():
    feat = _make_clustered_features()
    X_pca, _, _, _, pca_model, _ = run_pipeline(feat, min_reads=5)

    # Transforming the original features again through the fitted model must
    # reproduce the original PCA coordinates (deterministic, no refit).
    replayed = pca_model.transform(feat)
    assert np.allclose(replayed, X_pca)


@pytest.mark.unit
def test_run_pipeline_umap_model_transforms_new_pca_points():
    feat = _make_clustered_features()
    X_pca, X_umap, _, _, pca_model, umap_model = run_pipeline(feat, min_reads=5)

    new_point_pca = pca_model.transform(feat[:3])
    embedded = umap_model.transform(new_point_pca)
    assert embedded.shape == (3, 2)


@pytest.mark.unit
def test_run_pipeline_returns_none_below_min_reads():
    feat = _make_clustered_features(n_reads=4)
    assert run_pipeline(feat, min_reads=10) is None


@pytest.mark.unit
def test_umap_from_pca_returns_embedding_and_model():
    feat = _make_clustered_features()
    X_pca, _, _, _, pca_model, _ = run_pipeline(feat, min_reads=5)

    X_umap, model = umap_from_pca(X_pca)
    assert X_umap.shape == (X_pca.shape[0], 2)
    assert hasattr(model, "transform")


@pytest.mark.unit
def test_cluster_from_pca_still_returns_plain_label_array():
    feat = _make_clustered_features()
    X_pca, _, _, _, _, _ = run_pipeline(feat, min_reads=5)

    clusters = cluster_from_pca(X_pca)
    assert clusters.shape == (X_pca.shape[0],)
    assert clusters.dtype.kind in "iu"
