import numpy as np
import pandas as pd
import pytest

import smftools.latent_model_artifacts as model_artifacts
from smftools.informatics.molecule_identity import (
    EXPERIMENT_UID_COLUMN,
    MOLECULE_UID_COLUMN,
)
from smftools.latent_model_artifacts import (
    LATENT_MODEL_STATE,
    LatentModelArtifactError,
    deterministic_fit_membership,
    latent_model_id,
    latent_model_key,
    load_latent_model_state,
    mask_identity,
    write_latent_model_artifact,
)


def _obs(order):
    return pd.DataFrame(
        {
            EXPERIMENT_UID_COLUMN: ["experiment-uid"] * len(order),
            MOLECULE_UID_COLUMN: [f"molecule-{value}" for value in order],
        },
        index=[f"read-{value}" for value in order],
    )


def _key(fit_molecules, *, forced_revision=None):
    return latent_model_key(
        source_identity={"generation_id": "source-a", "checksum": "abc"},
        analysis_core_id="core-a",
        representation_specs=[
            {
                "suffix": "signal",
                "types": ["pca", "umap", "nmf"],
                "feature_mask_identity": mask_identity(np.array([True, False, True])),
            }
        ],
        algorithm_parameters={"n_pcs": 2, "random_state": 7},
        fit_molecule_uids=fit_molecules,
        forced_fit_revision=forced_revision,
    )


def test_fit_membership_is_invariant_to_input_order():
    forward = _obs([0, 1, 2, 3, 4])
    reverse = _obs([4, 3, 2, 1, 0])

    first = deterministic_fit_membership(
        forward,
        forward.index.tolist(),
        limit=3,
        random_state=7,
        coordinate_owner="core-a",
    )
    second = deterministic_fit_membership(
        reverse,
        reverse.index.tolist(),
        limit=3,
        random_state=7,
        coordinate_owner="core-a",
    )

    assert first == second


def test_same_semantics_produce_same_model_id():
    fit_molecules = ["molecule-a", "molecule-b"]

    assert latent_model_id(_key(fit_molecules)) == latent_model_id(_key(fit_molecules))
    assert latent_model_id(_key(fit_molecules)) != latent_model_id(
        _key(fit_molecules, forced_revision="forced-a")
    )
    changed_source = _key(fit_molecules)
    changed_source["source"] = {"generation_id": "source-b", "checksum": "def"}
    changed_config = _key(fit_molecules)
    changed_config["algorithm_parameters"] = {"n_pcs": 3, "random_state": 7}
    changed_mask = _key(fit_molecules)
    changed_mask["representations"][0]["feature_mask_identity"] = mask_identity(
        np.array([False, True, True])
    )
    baseline = latent_model_id(_key(fit_molecules))
    assert {
        latent_model_id(changed_source),
        latent_model_id(changed_config),
        latent_model_id(changed_mask),
    }.isdisjoint({baseline})


def test_model_artifact_rejects_tamper_and_requires_explicit_trust(tmp_path):
    key = _key(["molecule-a", "molecule-b"])
    artifact = write_latent_model_artifact(
        tmp_path,
        key=key,
        state={
            "portable": np.arange(4),
            "cp_factors": {"X_cp_signal": np.arange(6).reshape(3, 2)},
        },
        fit_molecule_uids=["molecule-a", "molecule-b"],
        cp_provenance=[
            {
                "representation": "X_cp_signal",
                "incremental_transform_supported": False,
            }
        ],
    )

    with pytest.raises(LatentModelArtifactError, match="explicit trust"):
        load_latent_model_state(
            artifact.path,
            expected_model_id=artifact.model_id,
            expected_model_checksum=artifact.model_checksum,
            trusted_local=False,
        )

    state, loaded = load_latent_model_state(
        artifact.path,
        expected_model_id=artifact.model_id,
        expected_model_checksum=artifact.model_checksum,
        trusted_local=True,
    )
    assert np.array_equal(state["portable"], np.arange(4))
    assert np.array_equal(state["cp_factors"]["X_cp_signal"], np.arange(6).reshape(3, 2))
    assert loaded.manifest["cp_provenance"][0]["incremental_transform_supported"] is False

    state_path = artifact.path / LATENT_MODEL_STATE
    state_path.write_bytes(state_path.read_bytes() + b"tampered")
    with pytest.raises(LatentModelArtifactError, match="checksum"):
        load_latent_model_state(
            artifact.path,
            expected_model_id=artifact.model_id,
            expected_model_checksum=artifact.model_checksum,
            trusted_local=True,
        )


def test_model_artifact_rejects_dependency_version_mismatch(tmp_path, monkeypatch):
    artifact = write_latent_model_artifact(
        tmp_path,
        key=_key(["molecule-a"]),
        state={},
        fit_molecule_uids=["molecule-a"],
        cp_provenance=[],
    )
    current = model_artifacts.dependency_versions()
    incompatible = {**current, "numpy": f"{current['numpy']}-incompatible"}
    monkeypatch.setattr(model_artifacts, "dependency_versions", lambda: incompatible)

    with pytest.raises(LatentModelArtifactError, match="dependency versions"):
        load_latent_model_state(
            artifact.path,
            expected_model_id=artifact.model_id,
            expected_model_checksum=artifact.model_checksum,
            trusted_local=True,
        )
