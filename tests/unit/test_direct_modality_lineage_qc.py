"""A direct-modality descendant recomputes modification QC rather than inheriting it.

`SRB-M3` is the finding that direct-modification re-basecalling must resolve
compatible simplex and modification models and must not reuse QC derived from
the old MM/ML probabilities. Model-pair resolution has its own coverage in
``tests/unit/informatics/test_dorado_model.py``; this is the other half, and the
one that is easy to get wrong quietly: the descendant's modification QC has to
come from the descendant's own signal.

The two generations here carry deliberately opposite modification signal, so a
descendant that inherited the parent's QC would pass where it must fail.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.preprocess_generation import (
    publish_preprocess_generation,
    resolve_current_preprocess_generation,
)
from tests.unit.test_partitioned_preprocess_executor import _cfg, _deaminase_cfg, _frame

pytestmark = pytest.mark.unit

_LINEAGE_PROVENANCE = {
    "lineage_id": "a" * 64,
    "origin_experiment_uid": "uid-a",
    "parent_raw_generation_id": "parent-raw",
    "parent_preprocess_generation_id": None,
    "selection_id": "b" * 64,
    "source_resolution_digest": None,
    "basecall_id": "c" * 64,
    "generation_kind": "selected_cohort",
    "identity_map": None,
}


def _direct_cfg():
    """Direct modality with a fixed binarization threshold and QC that bites."""
    cfg = _cfg()
    cfg.smf_modality = "direct"
    cfg.fit_position_methylation_thresholds = False
    cfg.binarize_on_fixed_methlyation_threshold = 0.5
    # Without an explicit threshold the modification filter has nothing to
    # decide, and the test could not tell recomputation from inheritance.
    cfg.read_mod_filtering_cpg_thresholds = [0.5, None]
    cfg.read_mod_filtering_gpc_thresholds = None
    return cfg


def _frame_with_signal(signals):
    """The shared direct-modality frame with its modification signal replaced."""
    frame = _frame().copy()
    frame["modification_signal"] = [list(values) for values in signals]
    return frame


def _publish_raw(frame, run_root, generation_id):
    directory = run_root / "raw_outputs" / "generations" / generation_id
    directory.mkdir(parents=True, exist_ok=True)
    return write_raw_store(
        frame,
        directory,
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )


def test_a_direct_descendant_recomputes_modification_qc_from_its_own_signal(tmp_path):
    run_root = tmp_path / "run"
    cfg = _direct_cfg()
    preprocess_dir = run_root / "preprocess_adata_outputs"

    # The parent's reads are heavily modified; the descendant's are not.
    parent = _publish_raw(
        _frame_with_signal([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        run_root,
        "parent-raw",
    )
    descendant = _publish_raw(
        _frame_with_signal([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        run_root,
        "descendant-raw",
    )

    parent_outputs = publish_preprocess_generation(parent["spine"], cfg, preprocess_dir)
    descendant_outputs = publish_preprocess_generation(
        descendant["spine"],
        cfg,
        preprocess_dir,
        lineage_provenance=dict(_LINEAGE_PROVENANCE),
        select_current=False,
    )

    parent_obs = pd.read_parquet(parent_outputs["obs"]).set_index("read_id")
    descendant_obs = pd.read_parquet(descendant_outputs["obs"]).set_index("read_id")

    # The descendant's modification metrics came from the descendant's signal.
    assert not np.allclose(
        parent_obs["Raw_modification_signal"].to_numpy(dtype=float),
        descendant_obs["Raw_modification_signal"].to_numpy(dtype=float),
    )
    assert descendant_obs["Raw_modification_signal"].max() < (
        parent_obs["Raw_modification_signal"].min()
    )
    # ... and so did its QC verdict, which is the half that inheritance would
    # silently get wrong.
    assert parent_obs["passes_modification_qc"].any()
    assert not descendant_obs["passes_modification_qc"].any()


def test_a_direct_descendant_publishes_beside_the_parent_and_records_its_lineage(tmp_path):
    run_root = tmp_path / "run"
    cfg = _direct_cfg()
    preprocess_dir = run_root / "preprocess_adata_outputs"
    parent = _publish_raw(
        _frame_with_signal([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        run_root,
        "parent-raw",
    )
    descendant = _publish_raw(
        _frame_with_signal([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        run_root,
        "descendant-raw",
    )

    publish_preprocess_generation(parent["spine"], cfg, preprocess_dir)
    parent_current, parent_manifest = resolve_current_preprocess_generation(preprocess_dir)
    descendant_outputs = publish_preprocess_generation(
        descendant["spine"],
        cfg,
        preprocess_dir,
        lineage_provenance=dict(_LINEAGE_PROVENANCE),
        select_current=False,
    )

    still_current, still_manifest = resolve_current_preprocess_generation(preprocess_dir)
    descendant_dir = descendant_outputs["generation"]
    manifest = json.loads((descendant_dir / "generation_manifest.json").read_text(encoding="utf-8"))

    assert descendant_dir != parent_current
    assert still_current == parent_current
    assert still_manifest["generation_id"] == parent_manifest["generation_id"]
    assert manifest["lineage"] == _LINEAGE_PROVENANCE
    # The parent's own generation records no lineage, so absence stays meaningful.
    parent_manifest_payload = json.loads(
        (parent_current / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert parent_manifest_payload["lineage"] is None


@pytest.mark.parametrize(
    "cfg_factory",
    [
        pytest.param(_direct_cfg, id="direct"),
        pytest.param(_cfg, id="conversion"),
        pytest.param(_deaminase_cfg, id="deaminase"),
    ],
)
def test_every_modality_publishes_a_descendant_beside_its_parent(tmp_path, cfg_factory):
    """The lineage publication contract must hold for each modality, not just one.

    Modalities differ in how they compute QC, so a descendant that published
    correctly for one could still take the selector or skip work in another.
    """
    run_root = tmp_path / "run"
    cfg = cfg_factory()
    preprocess_dir = run_root / "preprocess_adata_outputs"
    parent = _publish_raw(
        _frame_with_signal([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        run_root,
        "parent-raw",
    )
    descendant = _publish_raw(
        _frame_with_signal([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        run_root,
        "descendant-raw",
    )

    publish_preprocess_generation(parent["spine"], cfg, preprocess_dir)
    parent_current, parent_manifest = resolve_current_preprocess_generation(preprocess_dir)
    descendant_outputs = publish_preprocess_generation(
        descendant["spine"],
        cfg,
        preprocess_dir,
        lineage_provenance=dict(_LINEAGE_PROVENANCE),
        select_current=False,
    )

    still_current, still_manifest = resolve_current_preprocess_generation(preprocess_dir)
    manifest = json.loads(
        (descendant_outputs["generation"] / "generation_manifest.json").read_text(encoding="utf-8")
    )

    assert descendant_outputs["generation"] != parent_current
    assert still_current == parent_current
    assert still_manifest["generation_id"] == parent_manifest["generation_id"]
    assert manifest["lineage"] == _LINEAGE_PROVENANCE
    # The descendant's obs came from the descendant's own raw generation.
    descendant_obs = pd.read_parquet(descendant_outputs["obs"])
    assert set(descendant_obs["read_id"]) == {"read1", "read2"}
