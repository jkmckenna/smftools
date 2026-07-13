import pandas as pd
import pytest

from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.project import registry as reg
from smftools.project.catalog import ProjectCatalog, project_adata

SEQUENCE = "ACGTACGTACGT"  # length 12


def _make_raw_experiment(out_dir, *, reference_strand, uid, npos=12, n=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "read_id": f"{reference_strand}_r{i}",
            "reference": reference_strand.rsplit("_", 1)[0],
            "Reference_strand": reference_strand,
            "sample": "bc01",
            "barcode": "bc01",
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


@pytest.fixture
def project_with_two_experiments(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    # same sequence (same uid), DIFFERENT reference names across experiments
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    reg.init_project(proj)
    reg.add_experiment(proj, tmp_path / "expA")
    reg.add_experiment(proj, tmp_path / "expB")
    return proj, uid


def test_catalog_harmonizes_different_names(project_with_two_experiments):
    proj, uid = project_with_two_experiments
    cat = ProjectCatalog.open(proj)
    refs = cat.references()
    # differently-named references with identical sequence collapse to one canonical
    assert set(refs["canonical_reference"]) == {uid}
    assert set(refs["reference_strand"]) == {"geneA_top", "geneB_top"}
    assert set(cat.select(canonical_reference=uid)["experiment"]) == {"expA", "expB"}


def test_project_adata_concats_across_experiments(project_with_two_experiments):
    proj, uid = project_with_two_experiments
    combined = project_adata(proj, uid)
    assert combined.n_obs == 7  # 4 + 3 reads
    assert combined.n_vars == 12
    assert set(combined.obs["experiment"]) == {"expA", "expB"}
    # the base layers materialized from ragged
    assert "sequence_integer_encoding" in combined.layers


def test_duckdb_query_over_references(project_with_two_experiments):
    proj, _uid = project_with_two_experiments
    cat = ProjectCatalog.open(proj)
    result = cat.query("SELECT DISTINCT experiment FROM refs WHERE modality = 'direct'")
    assert set(result["experiment"]) == {"expA", "expB"}


def test_saved_query_set_selection(project_with_two_experiments):
    proj, uid = project_with_two_experiments
    reg.add_set(proj, "direct_only", query="modality = 'direct'")
    cat = ProjectCatalog.open(proj)
    sel = cat.select(canonical_reference=uid, set_name="direct_only")
    assert set(sel["experiment"]) == {"expA", "expB"}


def test_select_no_match_is_empty(project_with_two_experiments):
    proj, _uid = project_with_two_experiments
    assert ProjectCatalog.open(proj).select(canonical_reference="nonexistent").empty


def test_project_adata_stage_selection(tmp_path):
    """Experiments at different pipeline stages: auto picks the most-derived
    stage per experiment; an explicit stage skips (not crashes on) experiments
    that haven't reached it yet."""
    import anndata as ad

    from smftools.informatics.partition_read import relative_uns_path
    from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

    uid = reference_uid(SEQUENCE, 12)
    # Real nested layout (run_dir/raw_outputs/, run_dir/preprocess_adata_outputs/,
    # ...), not the flat single-dir shorthand `_make_raw_experiment` normally
    # uses, since this test needs sibling stage dirs under one run root.
    run_dir = tmp_path / "expA_run"
    _make_raw_experiment(run_dir / "raw_outputs", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB_run" / "raw_outputs", reference_strand="geneB_top", uid=uid, n=3)

    # Give expA a "preprocess" stage too: a spine.copy() with one marker obs
    # column added and source_base_dir pointing back at the raw stage (so
    # materialize() can still reach the ragged shards), matching the real shape
    # of a preprocess-stage spine (see preprocessing.partitioned_executor).
    raw_spine, _ = safe_read_h5ad(run_dir / "raw_outputs" / "spine.h5ad")
    preprocess_spine: ad.AnnData = raw_spine.copy()
    preprocess_spine.obs["passes_qc"] = True
    preprocess_dir = run_dir / "preprocess_adata_outputs"
    preprocess_dir.mkdir()
    preprocess_spine.uns["source_base_dir"] = relative_uns_path(run_dir / "raw_outputs", run_dir)
    safe_write_h5ad(preprocess_spine, preprocess_dir / "spine.h5ad", backup=False, verbose=False)

    proj = tmp_path / "project"
    reg.init_project(proj)
    exp_a_id, entry_a = reg.add_experiment(
        proj, tmp_path / "expA_run" / "raw_outputs", experiment_id="expA"
    )
    exp_b_id, entry_b = reg.add_experiment(
        proj, tmp_path / "expB_run" / "raw_outputs", experiment_id="expB"
    )
    assert set(entry_a["spines"]) == {"raw", "preprocess"}
    assert set(entry_b["spines"]) == {"raw"}

    # Auto: expA materializes from its preprocess spine (has passes_qc), expB
    # from its raw spine (no preprocess stage) -- both included, no crash.
    auto = project_adata(proj, uid)
    assert set(auto.obs["experiment"]) == {exp_a_id, exp_b_id}
    a_rows = auto.obs["experiment"] == exp_a_id
    assert bool(auto.obs.loc[a_rows, "passes_qc"].all())

    # Explicit stage="preprocess": only expA has reached it; expB is skipped
    # with a warning rather than raising.
    preprocess_only = project_adata(proj, uid, stage="preprocess")
    assert set(preprocess_only.obs["experiment"]) == {exp_a_id}

    # Explicit stage="raw": both experiments have it, neither has passes_qc.
    raw_only = project_adata(proj, uid, stage="raw")
    assert set(raw_only.obs["experiment"]) == {exp_a_id, exp_b_id}
    assert "passes_qc" not in raw_only.obs
