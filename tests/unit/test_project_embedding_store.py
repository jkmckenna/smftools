import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.project.embedding_store import (
    CURRENT_FILENAME,
    EmbeddingCompatibilityError,
    EmbeddingCompositionError,
    EmbeddingTrustError,
    embedding_dir,
    fit_or_extend_embedding,
    read_embedding,
)
from smftools.project.registry import add_experiment, add_set, init_project, remove_experiment

SEQUENCE = "ACGTACGTACGT"
NPOS = 12


def _make_clustered_raw_experiment(
    out_dir, *, reference_strand, uid, n_blob_a=15, n_blob_b=15, seed=0
):
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
    pointer = json.loads((directory / CURRENT_FILENAME).read_text())
    generation = directory / pointer["generation_path"]
    assert generation.is_dir()
    assert pointer["generation_id"] == result["meta"]["generation_id"]
    assert result["meta"]["source"]["feature_input_digest"]
    assert result["meta"]["source"]["ordered_molecule_membership_digest"]
    assert result["meta"]["dependencies"]["scikit-learn"]
    source_member = result["meta"]["source"]["members"][0]
    assert source_member["stage"] == "raw"
    assert source_member["spine_sha256"]
    assert not Path(source_member["spine_path"]).is_absolute()
    for filename in ("pca_model.pkl", "umap_model.pkl", "pca_space.npy", "coords.npy"):
        assert (generation / filename).exists()


def test_exact_cache_hit_does_not_unpickle_models(tmp_path, monkeypatch):
    proj, uid = _make_project(tmp_path)
    first = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    def refuse_pickle(*args, **kwargs):
        raise AssertionError("exact cache hit must not unpickle models")

    monkeypatch.setattr("smftools.project.embedding_store.pickle.load", refuse_pickle)
    second = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    assert second["obs_names"] == first["obs_names"]
    assert np.array_equal(second["X_pca"], first["X_pca"])
    assert np.array_equal(second["X_umap"], first["X_umap"])
    assert second["pca_model"] is None
    assert second["umap_model"] is None


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

    with pytest.raises(EmbeddingTrustError, match="trust_local_models=True"):
        fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    extended = fit_or_extend_embedding(
        proj,
        uid,
        min_reads=5,
        n_neighbors=5,
        trust_local_models=True,
    )

    assert len(extended["obs_names"]) == 30
    assert extended["meta"]["fit_kind"] == "extended"
    assert extended["meta"]["n_new_reads"] == 10

    # Existing points' coordinates are untouched by extension.
    old_index = {name: i for i, name in enumerate(first["obs_names"])}
    new_index = {name: i for i, name in enumerate(extended["obs_names"])}
    for name in first["obs_names"]:
        assert np.allclose(extended["X_pca"][new_index[name]], first["X_pca"][old_index[name]])
        assert np.allclose(extended["X_umap"][new_index[name]], first["X_umap"][old_index[name]])

    # Same fitted models are reused, not refit.
    assert extended["pca_model"] is not None
    assert extended["meta"]["fit_at"] == first["meta"]["fit_at"]
    assert extended["meta"]["prior_generation_id"] == first["meta"]["generation_id"]
    directory = embedding_dir(proj, uid, min_reads=5, n_neighbors=5)
    assert len(list((directory / "generations").iterdir())) == 2


def test_removal_requires_force_recompute(tmp_path):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10, seed=1)
    fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    uid2 = reference_uid(SEQUENCE, NPOS)
    _make_clustered_raw_experiment(
        tmp_path / "expB", reference_strand="geneB_top", uid=uid2, n_blob_a=5, n_blob_b=5, seed=2
    )
    add_experiment(proj, tmp_path / "expB")
    fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5, trust_local_models=True)
    remove_experiment(proj, "expA")

    with pytest.raises(EmbeddingCompositionError, match="absent"):
        fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)


def test_force_recompute_preserves_previous_generation(tmp_path):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10, seed=1)
    first = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)

    refit = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5, force_recompute=True)

    assert refit["meta"]["fit_kind"] == "full"
    assert refit["meta"]["prior_generation_id"] == first["meta"]["generation_id"]
    directory = embedding_dir(proj, uid, min_reads=5, n_neighbors=5)
    generations = list((directory / "generations").iterdir())
    assert len(generations) == 2
    assert (directory / "generations" / first["meta"]["generation_id"]).is_dir()


def test_embedding_dir_is_cheap_and_does_not_create_anything(tmp_path):
    proj, uid = _make_project(tmp_path)
    directory = embedding_dir(proj, uid)
    assert not directory.exists()


def test_embedding_definition_has_no_semantic_selection_collisions(tmp_path):
    proj, uid = _make_project(tmp_path)
    add_set(proj, "selected", experiments=["expA"])
    paths = {
        embedding_dir(proj, uid),
        embedding_dir(proj, uid, set_name="selected"),
        embedding_dir(proj, uid, experiments=["expA"]),
        embedding_dir(proj, uid, modality="direct"),
        embedding_dir(proj, uid, stage="raw"),
        embedding_dir(proj, "different-reference"),
    }
    assert len(paths) == 6


def test_changed_existing_features_require_force_recompute(tmp_path, monkeypatch):
    proj, uid = _make_project(tmp_path)
    fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    from smftools.project import embedding_store

    original = embedding_store._make_features

    def changed_features(*args, **kwargs):
        features, names = original(*args, **kwargs)
        features[0, 0] += 0.25
        return features, names

    monkeypatch.setattr(embedding_store, "_make_features", changed_features)
    with pytest.raises(EmbeddingCompositionError, match="changed feature values"):
        fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)


def test_dependency_incompatibility_blocks_extension(tmp_path, monkeypatch):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10)
    fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    _make_clustered_raw_experiment(
        tmp_path / "expB",
        reference_strand="geneB_top",
        uid=uid,
        n_blob_a=5,
        n_blob_b=5,
        seed=2,
    )
    add_experiment(proj, tmp_path / "expB")
    from smftools.project import embedding_store

    current = embedding_store._dependencies()
    monkeypatch.setattr(
        embedding_store,
        "_dependencies",
        lambda: {**current, "scikit-learn": "incompatible"},
    )
    with pytest.raises(EmbeddingCompatibilityError, match="dependencies changed"):
        fit_or_extend_embedding(
            proj,
            uid,
            min_reads=5,
            n_neighbors=5,
            trust_local_models=True,
        )


def test_interrupted_initial_fit_publishes_no_current_generation(tmp_path, monkeypatch):
    proj, uid = _make_project(tmp_path)
    from smftools.project import embedding_store

    original_write = embedding_store.atomic_write_json

    def fail_current(path, *args, **kwargs):
        if path.name == CURRENT_FILENAME:
            raise RuntimeError("injected initial publication failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(embedding_store, "atomic_write_json", fail_current)
    with pytest.raises(RuntimeError, match="injected"):
        fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    directory = embedding_dir(proj, uid, min_reads=5, n_neighbors=5)
    assert not (directory / CURRENT_FILENAME).exists()
    assert not list((directory / "generations").glob("*"))


def test_interrupted_extension_keeps_prior_generation_current(tmp_path, monkeypatch):
    proj, uid = _make_project(tmp_path, n_blob_a=10, n_blob_b=10)
    first = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    _make_clustered_raw_experiment(
        tmp_path / "expB",
        reference_strand="geneB_top",
        uid=uid,
        n_blob_a=5,
        n_blob_b=5,
        seed=2,
    )
    add_experiment(proj, tmp_path / "expB")
    from smftools.project import embedding_store

    original_write = embedding_store.atomic_write_json

    def fail_current(path, *args, **kwargs):
        if path.name == CURRENT_FILENAME:
            raise RuntimeError("injected extension failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(embedding_store, "atomic_write_json", fail_current)
    with pytest.raises(RuntimeError, match="injected"):
        fit_or_extend_embedding(
            proj,
            uid,
            min_reads=5,
            n_neighbors=5,
            trust_local_models=True,
        )
    directory = embedding_dir(proj, uid, min_reads=5, n_neighbors=5)
    pointer = json.loads((directory / CURRENT_FILENAME).read_text())
    assert pointer["generation_id"] == first["meta"]["generation_id"]
    assert len(list((directory / "generations").iterdir())) == 1


def test_relocated_project_embedding_reads_relative_current_generation(tmp_path):
    proj, uid = _make_project(tmp_path)
    first = fit_or_extend_embedding(proj, uid, min_reads=5, n_neighbors=5)
    relocated = tmp_path / "relocated-project"
    shutil.copytree(proj, relocated)

    moved = read_embedding(relocated, uid, min_reads=5, n_neighbors=5)

    assert moved["meta"]["generation_id"] == first["meta"]["generation_id"]
    assert np.array_equal(moved["X_umap"], first["X_umap"])
    assert moved["pca_model"] is None
