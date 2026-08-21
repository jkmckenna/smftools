"""Per-read mismatch clustermaps during preprocess (`EGL-26`).

Unlike `EGL-17`/`EGL-21`, this lane rasterizes nothing -- ``materialize``
already emits both layers. What can silently go wrong here is therefore not
arithmetic but *gating*: plotting a reference whose mod-site columns are
missing would show conversion chemistry as if it were sequence error, and
naming the output after the plotted layer would file mismatch panels under a
name that says "sequence".
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.partitioned_mismatch_plots import (
    DEFAULT_DEMUX_TYPES,
    _mod_site_columns,
    generate_mismatch_clustermaps,
)

pytestmark = pytest.mark.unit


def _cfg(**overrides):
    base = dict(
        mod_target_bases=["GpC", "CpG"],
        threads=1,
        clustermap_max_reads_per_plot=10000,
        plot_subsample_seed=0,
        sample_name_col_for_plotting="Sample",
        reindexed_var_suffix="reindexed",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- the mod-site requirement ------------------------------------------------


def test_required_columns_are_the_named_bases():
    assert _mod_site_columns("6B6_top", ["GpC", "CpG"]) == [
        "6B6_top_GpC_site",
        "6B6_top_CpG_site",
    ]


def test_ambiguous_column_is_not_required():
    """The renderer folds it in when present but treats it as optional.

    Requiring it here would skip references the renderer can handle perfectly
    well, turning an optional refinement into a hard gate.
    """
    assert "6B6_top_ambiguous_GpC_CpG_site" not in _mod_site_columns("6B6_top", ["GpC", "CpG"])


def test_no_mod_target_bases_skips_entirely():
    """With nothing to exclude the panel would be chemistry, not error."""
    assert generate_mismatch_clustermaps(None, None, None, None, cfg=_cfg(mod_target_bases=[])) == []


# --- gating, with the store stubbed out --------------------------------------


class _Plan:
    reference = "ref_top"
    start = 0
    end = 20
    gaps = ()

    def source_manifest(self):
        return {"reference": self.reference, "start": self.start, "end": self.end}


def _adata(var_columns: dict[str, np.ndarray]):
    import anndata as ad

    n_obs, n_vars = 4, 20
    adata = ad.AnnData(
        X=np.zeros((n_obs, n_vars), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "Sample": ["s1"] * n_obs,
                "Reference_strand": ["ref_top"] * n_obs,
            },
            index=[f"r{index}" for index in range(n_obs)],
        ),
        var=pd.DataFrame(var_columns, index=[str(position) for position in range(n_vars)]),
    )
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(n_obs, n_vars)).astype(np.int8)
    adata.layers["sequence_integer_encoding"] = codes
    adata.layers["mismatch_integer_encoding"] = codes.copy()
    adata.layers["read_span_mask"] = np.ones((n_obs, n_vars), dtype=np.int8)
    return adata


def _patch_store(monkeypatch, adata):
    import smftools.informatics.partition_read as partition_read
    import smftools.informatics.plot_region_stitching as stitching

    spine = SimpleNamespace(obs=pd.DataFrame({"passes_dedup": [True] * 4}))
    selection = SimpleNamespace(
        read_ids=("r0", "r1", "r2", "r3"), seed=0, selection_sha256="x", molecule_uids=()
    )
    monkeypatch.setattr(partition_read, "load_spine", lambda *a, **k: spine)
    monkeypatch.setattr(partition_read, "materialize", lambda *a, **k: adata)
    monkeypatch.setattr(stitching, "resolve_plot_region_plans", lambda *a, **k: [_Plan()])
    monkeypatch.setattr(stitching, "select_plot_reads", lambda *a, **k: selection)
    monkeypatch.setattr(stitching, "mask_unanalyzed_gaps", lambda *a, **k: None)


def test_reference_without_mod_site_columns_is_skipped(monkeypatch, tmp_path):
    """Skipping is the honest outcome; rendering anyway is worse than no plot.

    Without the mask every converted C reads as a mismatch, so the panel would
    show the chemistry the experiment applied rather than sequence error.
    """
    _patch_store(monkeypatch, _adata({"position_in_ref_top": np.ones(20, dtype=bool)}))
    layout = SimpleNamespace(categories={"mismatch_clustermaps": tmp_path})

    assert generate_mismatch_clustermaps(None, layout, None, None, cfg=_cfg()) == []
    assert not list(tmp_path.glob("*.png")), "no panel may be written for an unmaskable reference"


def test_reference_with_mod_site_columns_renders(monkeypatch, tmp_path):
    mask = np.zeros(20, dtype=bool)
    mask[::4] = True
    _patch_store(
        monkeypatch,
        _adata(
            {
                "position_in_ref_top": np.ones(20, dtype=bool),
                "ref_top_GpC_site": mask,
                "ref_top_CpG_site": np.zeros(20, dtype=bool),
            }
        ),
    )
    layout = SimpleNamespace(categories={"mismatch_clustermaps": tmp_path})
    registered: list[dict] = []
    import smftools.cli.stage_artifacts as stage_artifacts

    monkeypatch.setattr(
        stage_artifacts, "register_plot_artifact", lambda *a, **k: registered.append(k)
    )
    monkeypatch.setattr(stage_artifacts, "write_plot_source_manifest", lambda *a, **k: "manifest")

    results = generate_mismatch_clustermaps(None, layout, None, None, cfg=_cfg())

    assert results, "a maskable reference must produce a panel"
    written = list(tmp_path.glob("*.png"))
    assert written, "the panel must reach disk"
    # `EGL-17` shipped 30 plots that contributed zero catalog rows and were
    # invisible to anything discovering plots through the catalog.
    assert registered, "every panel must be registered in the plot catalog"
    assert registered[0]["category"] == "mismatch_clustermaps"


def test_panels_are_named_for_mismatch_not_the_plotted_layer(monkeypatch, tmp_path):
    """The renderer names files after the plotted layer by default.

    That would file `..._sequence_integer_encoding.png` under a category called
    `mismatch_clustermaps` -- the old CLI disambiguated the two variants by
    directory, but here the category *is* the directory.
    """
    _patch_store(
        monkeypatch,
        _adata(
            {
                "position_in_ref_top": np.ones(20, dtype=bool),
                "ref_top_GpC_site": np.zeros(20, dtype=bool),
                "ref_top_CpG_site": np.zeros(20, dtype=bool),
            }
        ),
    )
    layout = SimpleNamespace(categories={"mismatch_clustermaps": tmp_path})
    import smftools.cli.stage_artifacts as stage_artifacts

    monkeypatch.setattr(stage_artifacts, "register_plot_artifact", lambda *a, **k: None)
    monkeypatch.setattr(stage_artifacts, "write_plot_source_manifest", lambda *a, **k: "manifest")

    generate_mismatch_clustermaps(None, layout, None, None, cfg=_cfg())

    names = [path.name for path in tmp_path.glob("*.png")]
    assert names
    assert all("mismatch_no_mod_sites" in name for name in names)
    assert not any("sequence_integer_encoding" in name for name in names)


# --- wiring ------------------------------------------------------------------


def test_demux_fallback_keeps_reads_rather_than_dropping_them():
    """An empty sequence would filter out every read, not none of them.

    The renderer keeps reads whose `demux_type` is *in* the set, so falling
    back to `()` inverts the intent of the default.
    """
    assert DEFAULT_DEMUX_TYPES == ("single", "double", "already")


def test_category_is_registered_for_preprocess():
    from smftools.cli.stage_artifacts import STAGE_PLOT_CATEGORIES

    assert "mismatch_clustermaps" in STAGE_PLOT_CATEGORIES["preprocess"]


def test_config_knob_defaults_on_and_parses():
    from smftools.config.experiment_config import ExperimentConfig

    assert ExperimentConfig().plot_mismatch_clustermaps is True
    cfg, _ = ExperimentConfig.from_var_dict({"plot_mismatch_clustermaps": "FALSE"})
    assert cfg.plot_mismatch_clustermaps is False
