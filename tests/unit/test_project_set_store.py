import pandas as pd
import pytest

from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.project.registry import add_experiment, init_project
from smftools.project.set_store import iter_set_parts

SEQUENCE = "ACGTACGTACGT"


def _make_raw_experiment(out_dir, *, reference_strand, uid, npos=12, n=4, sample="bc01"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "read_id": f"{reference_strand}_r{i}",
            "reference": reference_strand.rsplit("_", 1)[0],
            "Reference_strand": reference_strand,
            "sample": sample,
            "barcode": sample,
            "strand": "top",
            "mapping_direction": "fwd",
            "reference_start": 0,
            "cigar": f"{npos}M",
            "aligned_length": npos,
            "sequence": [i % 4 for _ in range(npos)],
            "quality": [30] * npos,
            "mismatch": [4] * npos,
            "modification_signal": [float(i % 2)] * npos,
        }
        for i in range(n)
    ]
    write_raw_store(
        pd.DataFrame(rows),
        out_dir,
        reference_lengths={reference_strand: npos},
        extra_uns={
            "reference_uids": {reference_strand: uid},
            "modality": "direct",
            "experiment": out_dir.name,
        },
    )
    return out_dir


def _make_project(tmp_path, n_expA=4, n_expB=3):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=n_expA)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=n_expB)
    proj = tmp_path / "project"
    init_project(proj)
    add_experiment(proj, tmp_path / "expA")
    add_experiment(proj, tmp_path / "expB")
    return proj, uid


def test_iter_set_parts_yields_one_projected_part_per_experiment(tmp_path):
    proj, uid = _make_project(tmp_path)
    parts = list(iter_set_parts(proj, uid))

    assert len(parts) == 2  # one per member experiment
    by_exp = {p.obs["experiment"].iloc[0]: p for p in parts}
    assert set(by_exp) == {"expA", "expB"}
    assert by_exp["expA"].n_obs == 4
    assert by_exp["expB"].n_obs == 3
    for part in parts:
        # normalize_part contract: single experiment, stamped stage, stripped var,
        # unnamed obs index (so parts concat cleanly).
        assert part.obs["experiment"].nunique() == 1
        assert "project_stage" in part.uns
        assert part.var.shape[1] == 0
        assert part.obs.index.name is None


def test_iter_set_parts_projects_layers(tmp_path):
    proj, uid = _make_project(tmp_path)
    # Only X, no layers.
    x_only = list(iter_set_parts(proj, uid, layers=[]))
    assert all(len(p.layers) == 0 for p in x_only)
    # A named layer subset.
    one_layer = list(iter_set_parts(proj, uid, layers=["sequence_integer_encoding"]))
    assert all(set(p.layers) == {"sequence_integer_encoding"} for p in one_layer)


def test_iter_set_parts_is_lazy_but_resolves_membership_eagerly(tmp_path):
    proj, uid = _make_project(tmp_path)
    # Unmatched reference raises at call time (membership resolved eagerly), not on
    # first iteration.
    with pytest.raises(ValueError):
        iter_set_parts(proj, "does_not_exist")

    # A valid call returns a generator that hasn't materialized anything yet.
    import types

    gen = iter_set_parts(proj, uid)
    assert isinstance(gen, types.GeneratorType)
    assert sum(1 for _ in gen) == 2


def test_iter_set_parts_reflects_new_registration_without_any_cache(tmp_path):
    proj, uid = _make_project(tmp_path)
    assert len(list(iter_set_parts(proj, uid))) == 2

    _make_raw_experiment(tmp_path / "expC", reference_strand="geneC_top", uid=uid, n=2)
    add_experiment(proj, tmp_path / "expC")

    # No cache to invalidate -- membership is resolved fresh each call.
    parts = list(iter_set_parts(proj, uid))
    assert len(parts) == 3
    assert {p.obs["experiment"].iloc[0] for p in parts} == {"expA", "expB", "expC"}
    # And the set store writes nothing to disk.
    assert not (proj / "project_outputs" / "sets").exists()
