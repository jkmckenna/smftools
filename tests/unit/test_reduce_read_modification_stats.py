import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.partitioned_executor import reduce_read_modification_stats
from smftools.readwrite import safe_write_zarr


def _task_store(path, *, read_ids, total, modified, raw_signal):
    # "GpC_site" (not "GpC") is the real site_type token this codebase uses --
    # it's derived from var column names like "refA_GpC_site" via
    # column.removeprefix(f"{reference}_") in _core_skeleton, so it already
    # carries the "_site" suffix.
    n = len(read_ids)
    adata = ad.AnnData(X=np.zeros((n, 1)), obs=pd.DataFrame(index=read_ids))
    adata.obs["Total_GpC_site_in_read_partial"] = total
    adata.obs["Modified_GpC_site_count_partial"] = modified
    adata.obs["Raw_modification_signal_partial"] = raw_signal
    safe_write_zarr(adata, path, verbose=False)
    return path


def test_reduce_read_modification_stats_totals_are_per_reference_not_per_read(tmp_path):
    # Regression test for a real ~30+ minute stall on real 1.3M-read data: the
    # old implementation recomputed var_frame["reference"] == reference (a full
    # scan) once per *read* instead of once per *unique reference*. This
    # fixture is built so a reference-filtering bug -- summing another
    # reference's site column into this one's total -- changes the answer,
    # not just the runtime: refA's site column is True for refB's rows too
    # (and vice versa), so a leak would inflate both totals.
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    _task_store(
        store_dir / "refA.zarr",
        read_ids=["read1", "read2"],
        total=[2, 3],
        modified=[1, 3],
        raw_signal=[1.0, 2.0],
    )
    _task_store(
        store_dir / "refB.zarr",
        read_ids=["read3"],
        total=[1],
        modified=[0],
        raw_signal=[0.5],
    )
    catalog = pd.DataFrame(
        [
            {"reference": "refA", "group_path": "refA.zarr"},
            {"reference": "refB", "group_path": "refB.zarr"},
        ]
    )
    catalog_path = store_dir / "catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)

    # refA rows (0-2): refA_GpC_site has 3 True -- correct refA total is 3.
    # refB rows (3-4) also have refA_GpC_site = True; if the reference filter
    # leaked, refA's total would incorrectly become 5.
    # refB rows (3-4): refB_GpC_site has 1 True -- correct refB total is 1.
    # refA rows (0-2) also have refB_GpC_site = True; a leak would make it 4.
    var_frame = pd.DataFrame(
        {
            "reference": ["refA", "refA", "refA", "refB", "refB"],
            "refA_GpC_site": [True, True, True, True, True],
            "refB_GpC_site": [True, True, True, True, False],
        }
    )
    var_path = tmp_path / "var.parquet"
    var_frame.to_parquet(var_path, index=False)

    obs_path = tmp_path / "obs.parquet"
    pd.DataFrame({"read_id": ["read1", "read2", "read3"]}).to_parquet(obs_path, index=False)

    reduce_read_modification_stats(catalog_path, var_path, obs_path)

    result = pd.read_parquet(obs_path).set_index("read_id")

    # Fractions are per-read, unaffected by the totals bug -- sanity check.
    assert result.loc["read1", "Fraction_GpC_site_modified"] == 0.5
    assert result.loc["read2", "Fraction_GpC_site_modified"] == 1.0
    assert result.loc["read3", "Fraction_GpC_site_modified"] == 0.0

    # The actual regression check: correct, non-leaked per-reference totals.
    assert result.loc["read1", "Total_GpC_site_in_reference"] == 3
    assert result.loc["read2", "Total_GpC_site_in_reference"] == 3
    assert result.loc["read3", "Total_GpC_site_in_reference"] == 1

    assert result.loc["read1", "Valid_GpC_site_in_read_vs_reference"] == 2 / 3
    assert result.loc["read2", "Valid_GpC_site_in_read_vs_reference"] == 1.0
    assert result.loc["read3", "Valid_GpC_site_in_read_vs_reference"] == 1.0


def test_reduce_read_modification_stats_totals_computed_once_per_unique_reference(
    tmp_path, monkeypatch
):
    # Directly guards against the O(n_reads) redundancy regressing: with 3
    # reads sharing one reference, the var_frame equality scan must run once
    # (per unique reference), not 3 times.
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    _task_store(
        store_dir / "refA.zarr",
        read_ids=["read1", "read2", "read3"],
        total=[1, 1, 1],
        modified=[1, 0, 1],
        raw_signal=[1.0, 1.0, 1.0],
    )
    catalog = pd.DataFrame([{"reference": "refA", "group_path": "refA.zarr"}])
    catalog_path = store_dir / "catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)

    var_frame = pd.DataFrame({"reference": ["refA"], "refA_GpC_site": [True]})
    var_path = tmp_path / "var.parquet"
    var_frame.to_parquet(var_path, index=False)

    obs_path = tmp_path / "obs.parquet"
    pd.DataFrame({"read_id": ["read1", "read2", "read3"]}).to_parquet(obs_path, index=False)

    call_count = 0
    real_eq = pd.Series.__eq__

    def counting_eq(self, other):
        nonlocal call_count
        if self.name == "reference":
            call_count += 1
        return real_eq(self, other)

    monkeypatch.setattr(pd.Series, "__eq__", counting_eq, raising=True)
    reduce_read_modification_stats(catalog_path, var_path, obs_path)

    assert call_count == 1, (
        f"var_frame['reference'] == reference was compared {call_count} times "
        "for 3 reads sharing 1 reference; expected exactly 1 (once per unique "
        "reference, not once per read)"
    )
