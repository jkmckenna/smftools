import numpy as np
import pandas as pd
import pytest

from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.project.embedding_store import (
    EmbeddingCompositionError,
    embedding_dir,
    fit_or_extend_embedding,
)
from smftools.project.registry import add_experiment, init_project

SEQUENCE = "ACGTACGTACGT"
NPOS = 12


def _make_clustered_raw_experiment(out_dir, *, reference_strand, uid, n_blob_a=15, n_blob_b=15, seed=0):
    """Two well-separated per-position signal populations -- reliably PCA/Leiden
    separable, matching the pattern in tests/unit/analysis/test_dimensionality_reduction.py."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_blob_a + n_blob_b):
        center = 0.1 if i < n_blob_a else 0.9
        signal = np.clip(rng.normal(center, 0.03, NPOS), 0.0, 1.0).tolist()
        rows.append(
            {
                "read_id": f"{reference_strand}_r{i}",
                "reference": reference_strand.rsplit("_", 1)[0],
                "Reference_strand": reference_strand,
                "sample": "bc01",
                "barcode": "bc01",
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": f"{NPOS}M",
                "aligned_length": NPOS,
                "sequence": [i % 4 for _ in range(NPOS)],
                "quality": [30] * NPOS,
                "mismatch": [4] * NPOS,
                "modification_signal": signal,
            }
        )
    write_raw_store(
        pd.DataFrame(rows),
        out_dir,
        reference_lengths={reference_strand: NPOS},
        extra_uns={
            "reference_uids": {reference_strand: uid},
            "modality": "direct",
            "experiment": out_dir.name,
        },
    )
    return out_dir


def _make_project(tmp_path, *, n_blob_a=15, n_blob_b=15, reference_strand="geneA_top", seed=0):
    uid = reference_uid(SEQUENCE, NPOS)
    _make_clustered_raw_experiment(
        tmp_path / "expA",
        reference_strand=reference_strand,
        uid=uid,
        n_blob_a=n_blob_a,
        n_blob_b=n_blob_b,
        seed=seed,
    )
    proj = tmp_path / "project"
    init_project(proj)
    add_experiment(proj, tmp_path / "expA")
    return proj, uid


def test_fit_or_extend_embedding_full_fit(tmp_path):
    proj, uid = _make_project(tmp_path)

    result = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    assert len(result["obs_names"]) == 30
    assert result["X_pca"].shape[0] == 30
    assert result["X_umap"].shape == (30, 2)
    assert result["clusters"].shape == (30,)
    assert len(set(result["clusters"].tolist())) >= 2
    assert result["meta"]["fit_kind"] == "full"

    directory = embedding_dir(proj, uid, min_reads=5, n_neighbors=5)
    for filename in ("pca_model.pkl", "umap_model.pkl", "pca_space.npy", "coords.npy", "clusters.npy", "meta.json"):
        assert (directory / filename).exists()


def test_fit_or_extend_embedding_second_call_is_a_cache_hit(tmp_path):
    proj, uid = _make_project(tmp_path)
    first = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    second = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    assert second["obs_names"] == first["obs_names"]
    assert np.array_equal(second["X_pca"], first["X_pca"])
    assert np.array_equal(second["X_umap"], first["X_umap"])


def test_fit_or_extend_embedding_extends_on_growth_without_moving_existing_points(tmp_path):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10, seed=1)
    first = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    assert len(first["obs_names"]) == 20

    # Register a second experiment sharing the same canonical reference -- the set grows.
    uid2 = reference_uid(SEQUENCE, NPOS)
    _make_clustered_raw_experiment(
        tmp_path / "expB", reference_strand="geneB_top", uid=uid2, n_blob_a=5, n_blob_b=5, seed=2
    )
    add_experiment(proj, tmp_path / "expB")

    extended = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    assert len(extended["obs_names"]) == 30
    assert extended["meta"]["fit_kind"] == "extended"
    assert extended["meta"]["n_new_reads"] == 10

    # Existing points' coordinates are untouched by extension.
    old_index = {name: i for i, name in enumerate(first["obs_names"])}
    new_index = {name: i for i, name in enumerate(extended["obs_names"])}
    for name in first["obs_names"]:
        assert np.allclose(
            extended["X_pca"][new_index[name]], first["X_pca"][old_index[name]]
        )
        assert np.allclose(
            extended["X_umap"][new_index[name]], first["X_umap"][old_index[name]]
        )

    # Same fitted models are reused, not refit.
    assert extended["pca_model"] is not None
    assert extended["meta"]["fit_at"] == first["meta"]["fit_at"]


def test_fit_or_extend_embedding_raises_without_force_recompute_when_reads_disappear(tmp_path):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10, seed=1)
    fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    uid2 = reference_uid(SEQUENCE, NPOS)
    _make_clustered_raw_experiment(
        tmp_path / "expB", reference_strand="geneB_top", uid=uid2, n_blob_a=5, n_blob_b=5, seed=2
    )
    add_experiment(proj, tmp_path / "expB")

    # Filtering to only expB excludes every previously-embedded read from expA.
    with pytest.raises(EmbeddingCompositionError):
        fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5, experiments=["expB"])


def test_fit_or_extend_embedding_force_recompute_archives_previous_version(tmp_path):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10, seed=1)
    fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    uid2 = reference_uid(SEQUENCE, NPOS)
    _make_clustered_raw_experiment(
        tmp_path / "expB", reference_strand="geneB_top", uid=uid2, n_blob_a=5, n_blob_b=5, seed=2
    )
    add_experiment(proj, tmp_path / "expB")

    refit = fit_or_extend_embedding(
        proj, uid, min_reads=5, n_neighbors=5, experiments=["expB"], force_recompute=True
    )

    assert refit["meta"]["fit_kind"] == "full"
    assert len(refit["obs_names"]) == 10

    directory = embedding_dir(proj, uid, min_reads=5, n_neighbors=5)
    versions_dir = directory / "versions"
    assert versions_dir.is_dir()
    archived = list(versions_dir.iterdir())
    assert len(archived) == 1
    assert (archived[0] / "meta.json").exists()


def test_embedding_dir_is_cheap_and_does_not_create_anything(tmp_path):
    proj, uid = _make_project(tmp_path)
    directory = embedding_dir(proj, uid)
    assert not directory.exists()
