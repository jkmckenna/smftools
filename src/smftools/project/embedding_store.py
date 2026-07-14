"""Set-level embedding store: PCA -> UMAP -> Leiden over a materialized set, with the
fitted PCA/UMAP models persisted so a set that *grows* gets its new molecules
transformed into the existing embedding by default (cheap; existing points' coordinates
never move) instead of a full refit. Leiden clustering has no equivalent incremental
transform (a KNN-graph community recompute over a bigger graph can shift boundaries for
every point, not just new ones), so new points instead get a nearest-neighbor cluster
assignment against the fixed embedding.

Deliberately keyed independently of ``set_store``'s composition-hash base cache: that
cache recomputes automatically on *any* resolved-membership change (by design, for
correctness), but an embedding needs to survive across membership growth in order to be
extendable at all -- so it lives at ``<set>/embeddings/<embedding_hash>/``, hashed only
over the parameters that define the embedding itself (feature choice, window, Leiden
resolution, ...), never over resolved membership.

Wraps ``smftools.analysis.compute.dimensionality_reduction`` directly (Tier 2, reused
as-is -- ``run_pipeline``/``umap_from_pca`` were extended to also return their fitted
PCA/UMAP models specifically to support this). An explicit full refit
(``force_recompute=True``) archives the previous embedding under ``versions/<timestamp>/``
first rather than silently overwriting it, since analyses/figures built on stable
coordinates shouldn't shift underneath them without it being an explicit, visible act.

Final piece of Phase 4, ``dev/project_sample_and_set_stores.md``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .set_store import materialize_set, set_label, sets_root

EMBEDDINGS_DIRNAME = "embeddings"
VERSIONS_DIRNAME = "versions"
PCA_MODEL_FILENAME = "pca_model.pkl"
UMAP_MODEL_FILENAME = "umap_model.pkl"
PCA_SPACE_FILENAME = "pca_space.npy"
COORDS_FILENAME = "coords.npy"
CLUSTERS_FILENAME = "clusters.npy"
OBS_NAMES_FILENAME = "obs_names.json"
META_FILENAME = "meta.json"


class EmbeddingCompositionError(ValueError):
    """Raised when the current set no longer contains every previously-embedded read.

    Growth (new reads added) is handled automatically; anything else (a read
    disappeared -- re-registration, removal, ...) needs an explicit
    ``force_recompute=True`` rather than silently reinterpreting the embedding.
    """


def _embedding_definition(
    *, layer, start, end, feature_kind, leiden_resolution, n_neighbors, min_reads, random_state
) -> dict:
    return {
        "layer": layer,
        "start": start,
        "end": end,
        "feature_kind": feature_kind,
        "leiden_resolution": leiden_resolution,
        "n_neighbors": n_neighbors,
        "min_reads": min_reads,
        "random_state": random_state,
    }


def _definition_hash(definition: dict) -> str:
    encoded = json.dumps(definition, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def embedding_dir(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    feature_kind: str = "raw",
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    min_reads: int = 10,
    random_state: int = 42,
) -> Path:
    """Return the directory a :func:`fit_or_extend_embedding` call with these exact
    embedding-defining parameters would use -- cheap, does not touch anything."""
    definition = _embedding_definition(
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    label = set_label(set_name, canonical_reference)
    return sets_root(project_dir) / label / EMBEDDINGS_DIRNAME / _definition_hash(definition)


def _make_features(adata, *, feature_kind: str, layer, start, end):
    from ..analysis.compute.dimensionality_reduction import (
        coverage_filter,
        make_features_acf,
        make_features_raw,
    )

    positions = np.asarray(adata.var_names, dtype=np.int64)
    if start is not None or end is not None:
        window = np.ones(positions.shape[0], dtype=bool)
        if start is not None:
            window &= positions >= int(start)
        if end is not None:
            window &= positions < int(end)
        adata = adata[:, window]
        positions = positions[window]

    matrix_source = adata.layers[layer] if layer is not None else adata.X
    mat = np.asarray(matrix_source, dtype=np.float64)
    obs_names = list(adata.obs_names)

    mat, positions, obs_names, _ = coverage_filter(mat, positions, obs_names)
    if feature_kind == "acf":
        feat, valid = make_features_acf(mat, positions)
        obs_names = [name for name, keep in zip(obs_names, valid) if keep]
    elif feature_kind == "raw":
        feat = make_features_raw(mat)
    else:
        raise ValueError(f"feature_kind must be 'raw' or 'acf', got {feature_kind!r}")
    return feat, obs_names


def _read_artifacts(directory: Path) -> dict:
    with (directory / PCA_MODEL_FILENAME).open("rb") as handle:
        pca_model = pickle.load(handle)
    with (directory / UMAP_MODEL_FILENAME).open("rb") as handle:
        umap_model = pickle.load(handle)
    return {
        "X_pca": np.load(directory / PCA_SPACE_FILENAME),
        "X_umap": np.load(directory / COORDS_FILENAME),
        "clusters": np.load(directory / CLUSTERS_FILENAME),
        "obs_names": json.loads((directory / OBS_NAMES_FILENAME).read_text()),
        "pca_model": pca_model,
        "umap_model": umap_model,
        "meta": json.loads((directory / META_FILENAME).read_text()),
    }


def _write_artifacts(directory: Path, *, X_pca, X_umap, clusters, obs_names, pca_model, umap_model, meta):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / PCA_SPACE_FILENAME, X_pca)
    np.save(directory / COORDS_FILENAME, X_umap)
    np.save(directory / CLUSTERS_FILENAME, clusters)
    (directory / OBS_NAMES_FILENAME).write_text(json.dumps(list(obs_names)))
    with (directory / PCA_MODEL_FILENAME).open("wb") as handle:
        pickle.dump(pca_model, handle)
    with (directory / UMAP_MODEL_FILENAME).open("wb") as handle:
        pickle.dump(umap_model, handle)
    (directory / META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))


def _archive_existing(directory: Path) -> None:
    """Move an existing embedding's artifacts under versions/<timestamp>/ before a
    full refit overwrites them -- so a stable coordinate space anyone built figures
    against never just silently disappears."""
    artifact_names = [
        PCA_MODEL_FILENAME,
        UMAP_MODEL_FILENAME,
        PCA_SPACE_FILENAME,
        COORDS_FILENAME,
        CLUSTERS_FILENAME,
        OBS_NAMES_FILENAME,
        META_FILENAME,
    ]
    if not any((directory / name).exists() for name in artifact_names):
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive_dir = directory / VERSIONS_DIRNAME / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in artifact_names:
        path = directory / name
        if path.exists():
            shutil.move(str(path), str(archive_dir / name))


def _assign_nearest_cluster(new_points: np.ndarray, reference_points: np.ndarray, reference_clusters: np.ndarray) -> np.ndarray:
    """Nearest-neighbor cluster assignment for points added to a fixed embedding.

    Not a Leiden recompute -- Leiden has no clean incremental equivalent (a bigger
    graph can reshuffle every community, not just the new points), so this simply
    inherits the label of each new point's single nearest already-embedded neighbor
    in PCA space.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(reference_points)
    _, indices = nn.kneighbors(new_points)
    return reference_clusters[indices[:, 0]]


def fit_or_extend_embedding(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    feature_kind: str = "raw",
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    min_reads: int = 10,
    random_state: int = 42,
    force_recompute: bool = False,
) -> dict:
    """Fit, extend, or refit a set's PCA/UMAP/Leiden embedding.

    - **No existing embedding for this definition, or ``force_recompute=True``**: a
      full fit via ``analysis.compute.dimensionality_reduction.run_pipeline``. A
      ``force_recompute`` first archives whatever was there under
      ``versions/<timestamp>/``.
    - **Existing embedding, and every previously-embedded read is still present in
      the current materialization (pure growth)**: only the *new* reads' features are
      ``.transform()``ed through the persisted PCA/UMAP models -- existing
      coordinates are untouched -- and each new point gets a cluster label by
      nearest-neighbor lookup against the existing embedding, not a Leiden recompute.
    - **Existing embedding, but a previously-embedded read is now missing** (the set
      shrank or changed in some way other than pure growth): raises
      :class:`EmbeddingCompositionError` unless ``force_recompute=True``, rather than
      silently reinterpreting what the embedding means.

    Returns a dict: ``{"X_pca", "X_umap", "clusters", "obs_names", "pca_model",
    "umap_model", "meta"}`` -- the same shape :func:`analysis.compute.
    dimensionality_reduction.run_pipeline` produces, plus the read-id ordering and a
    small provenance record.
    """
    adata = materialize_set(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    feat, obs_names = _make_features(adata, feature_kind=feature_kind, layer=layer, start=start, end=end)

    directory = embedding_dir(
        project_dir,
        canonical_reference,
        set_name=set_name,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
    )
    existing = None
    if not force_recompute and (directory / META_FILENAME).exists():
        existing = _read_artifacts(directory)
        previous_names = existing["obs_names"]
        if set(previous_names) - set(obs_names):
            raise EmbeddingCompositionError(
                f"{len(set(previous_names) - set(obs_names))} previously-embedded read(s) are "
                "no longer present in this set's current materialization -- pass "
                "force_recompute=True to refit from scratch (the previous embedding "
                "is archived, not lost)."
            )

    if existing is not None and set(obs_names) == set(existing["obs_names"]):
        return existing

    if existing is None:
        result = _fit_from_scratch(
            feat,
            obs_names,
            leiden_resolution=leiden_resolution,
            min_reads=min_reads,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        explained_variance_ratio = result.pop("explained_variance_ratio")
        meta = {
            "canonical_reference": canonical_reference,
            "set_name": set_name,
            "definition": _embedding_definition(
                layer=layer,
                start=start,
                end=end,
                feature_kind=feature_kind,
                leiden_resolution=leiden_resolution,
                n_neighbors=n_neighbors,
                min_reads=min_reads,
                random_state=random_state,
            ),
            "n_reads": len(obs_names),
            "fit_kind": "full",
            "fit_at": datetime.now(timezone.utc).isoformat(),
            "explained_variance_ratio": explained_variance_ratio,
        }
        if force_recompute:
            _archive_existing(directory)
        _write_artifacts(directory, obs_names=obs_names, meta=meta, **result)
        return {**result, "obs_names": obs_names, "meta": meta, "explained_variance_ratio": explained_variance_ratio}

    # Pure growth: transform the new reads only.
    obs_index = {name: i for i, name in enumerate(obs_names)}
    new_names = [name for name in obs_names if name not in set(existing["obs_names"])]
    new_rows = np.asarray([obs_index[name] for name in new_names], dtype=np.int64)
    new_feat = feat[new_rows]

    new_X_pca = existing["pca_model"].transform(new_feat)
    new_X_umap = existing["umap_model"].transform(new_X_pca)
    new_clusters = _assign_nearest_cluster(new_X_pca, existing["X_pca"], existing["clusters"])

    combined_obs_names = existing["obs_names"] + new_names
    combined_X_pca = np.vstack([existing["X_pca"], new_X_pca])
    combined_X_umap = np.vstack([existing["X_umap"], new_X_umap])
    combined_clusters = np.concatenate([existing["clusters"], new_clusters])

    meta = {
        **existing["meta"],
        "n_reads": len(combined_obs_names),
        "fit_kind": "extended",
        "extended_at": datetime.now(timezone.utc).isoformat(),
        "n_new_reads": len(new_names),
    }
    _write_artifacts(
        directory,
        X_pca=combined_X_pca,
        X_umap=combined_X_umap,
        clusters=combined_clusters,
        obs_names=combined_obs_names,
        pca_model=existing["pca_model"],
        umap_model=existing["umap_model"],
        meta=meta,
    )
    return {
        "X_pca": combined_X_pca,
        "X_umap": combined_X_umap,
        "clusters": combined_clusters,
        "obs_names": combined_obs_names,
        "pca_model": existing["pca_model"],
        "umap_model": existing["umap_model"],
        "meta": meta,
    }


def _fit_from_scratch(feat, obs_names, *, leiden_resolution, min_reads, n_neighbors, random_state) -> dict:
    from ..analysis.compute.dimensionality_reduction import run_pipeline

    result = run_pipeline(
        feat,
        leiden_resolution=leiden_resolution,
        min_reads=min_reads,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    if result is None:
        raise ValueError(f"fewer than min_reads={min_reads} reads survived feature preparation")
    X_pca, X_umap, clusters, explained_variance_ratio, pca_model, umap_model = result
    return {
        "X_pca": X_pca,
        "X_umap": X_umap,
        "clusters": clusters,
        "pca_model": pca_model,
        "umap_model": umap_model,
        "explained_variance_ratio": explained_variance_ratio.tolist(),
    }
