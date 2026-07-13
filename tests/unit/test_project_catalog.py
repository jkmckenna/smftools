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
