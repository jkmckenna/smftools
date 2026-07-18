import anndata as ad
import numpy as np
import pandas as pd

from smftools.informatics.partition_store import (
    build_spine,
    write_catalog,
    write_experiment_store,
    write_partitioned_store,
)
from smftools.informatics.sidecar_manifest import resolve_sidecar
from smftools.readwrite import safe_read_h5ad, safe_read_zarr

LAYERS = ("sequence_integer_encoding", "base_quality_scores", "read_span_mask")


def _synth(n_refs=3, n_samples=2, per=40, n_pos=30, seed=1):
    rng = np.random.default_rng(seed)
    n = n_refs * n_samples * per
    x = rng.integers(0, 2, (n, n_pos)).astype(np.float32)
    x[rng.random((n, n_pos)) < 0.6] = np.nan
    layers = {
        "sequence_integer_encoding": rng.integers(0, 6, (n, n_pos), dtype=np.int8),
        "base_quality_scores": rng.integers(-1, 60, (n, n_pos), dtype=np.int8),
        "read_span_mask": rng.integers(0, 2, (n, n_pos), dtype=np.int8),
    }
    refs = np.repeat([f"ref{r}_top" for r in range(n_refs)], n_samples * per)
    samples = np.tile(np.repeat([f"barcode{s:02d}" for s in range(n_samples)], per), n_refs)
    obs = pd.DataFrame(
        {
            "Reference_strand": pd.Categorical(refs),
            "Sample": pd.Categorical(samples),
            "Barcode": pd.Categorical(samples),
            "Strand": pd.Categorical(["top"] * n),
            "Dataset": pd.Categorical(["unconverted"] * n),
            "Read_mapping_direction": pd.Categorical(rng.choice(["fwd", "rev"], n)),
        },
        index=[f"read{i:05d}" for i in range(n)],
    )
    a = ad.AnnData(X=x, obs=obs, layers=layers)
    a.var_names = [str(p) for p in range(n_pos)]
    a.uns["sequence_integer_encoding_map"] = {"A": 0, "C": 1, "G": 2, "T": 3}
    a.uns["References"] = {"ref0_top_FASTA_sequence": "ACGT"}
    return a


def test_write_experiment_store_layout_and_catalog(tmp_path):
    a = _synth()
    out = write_experiment_store(
        a, tmp_path, experiment="exp1", modality="conversion", bam_path="/x/aligned.bam"
    )
    for key in ("store", "spine", "catalog", "manifest"):
        assert out[key].exists(), f"{key} missing"

    cat = pd.read_parquet(out["catalog"])
    # one row per (reference, sample) partition
    assert len(cat) == a.obs.groupby(["Reference_strand", "Sample"], observed=True).ngroups
    assert cat["n_reads"].sum() == a.n_obs
    assert set(cat["experiment"]) == {"exp1"}
    assert set(cat["modality"]) == {"conversion"}
    # every group_path points at a real partition dir
    for gp in cat["group_path"]:
        assert (tmp_path / gp).is_dir()

    # manifest registration resolves
    assert resolve_sidecar(out["manifest"], "spine") is not None
    assert resolve_sidecar(out["manifest"], "catalog") is not None


def test_spine_is_thin_with_pointers(tmp_path):
    a = _synth()
    out = write_experiment_store(a, tmp_path, experiment="exp1", bam_path="/x/aligned.bam")
    spine, _ = safe_read_h5ad(out["spine"])
    assert spine.n_obs == a.n_obs
    assert spine.n_vars == 0  # no X/var: thin index only
    assert len(spine.layers) == 0
    for col in ("partition", "partition_row", "Reference_strand", "Sample", "bam_path"):
        assert col in spine.obs.columns
    assert set(map(str, spine.obs["bam_path"])) == {"/x/aligned.bam"}


def test_partition_roundtrip_matches_source(tmp_path):
    a = _synth()
    partitions = write_partitioned_store(a, tmp_path)
    # pick one partition and reconstruct it
    p = partitions[0]
    part, _ = safe_read_zarr(tmp_path / p.group_path)

    mask = (a.obs["Reference_strand"].astype(str) == p.reference).values & (
        a.obs["Sample"].astype(str) == p.sample
    ).values
    expect = a[mask]
    assert list(part.obs_names) == list(expect.obs_names) == p.read_ids
    assert np.array_equal(np.asarray(part.X), np.asarray(expect.X), equal_nan=True)
    for layer in LAYERS:
        assert np.array_equal(np.asarray(part.layers[layer]), np.asarray(expect.layers[layer]))


def test_partition_row_pointer_is_correct(tmp_path):
    a = _synth()
    partitions = write_partitioned_store(a, tmp_path)
    spine = build_spine(a, partitions)
    # for a sampled read, partition_row must index the same read within its partition
    read_id = str(a.obs_names[123])
    rec = spine.obs.loc[read_id]
    part, _ = safe_read_zarr(tmp_path / rec["partition"])
    assert str(part.obs_names[int(rec["partition_row"])]) == read_id


def test_write_catalog_columns(tmp_path):
    a = _synth(n_refs=2, n_samples=1, per=10)
    partitions = write_partitioned_store(a, tmp_path)
    cat_path = write_catalog(tmp_path / "catalog.parquet", partitions, experiment="e", modality="m")
    df = pd.read_parquet(cat_path)
    assert list(df.columns) == [
        "experiment",
        "partition_id",
        "reference",
        "sample",
        "n_reads",
        "n_positions",
        "group_path",
        "modality",
        "config_hash",
        "created_at",
    ]
    assert len(df) == len(partitions)
