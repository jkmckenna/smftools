"""The ligation / tagging model recovers what it is simulated with (`BQC-02`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.barcode_tagging_model import (
    TAGGING_SUMMARY_FILENAME,
    barcode_chemistry,
    barcode_tagging_report,
    kit_barcode_count,
    write_barcode_tagging_qc,
)

pytestmark = pytest.mark.unit

KIT = [f"NB{i:02d}" for i in range(1, 25)]
POOL = KIT[:8]
LENGTH = 1000


def _simulate(
    *,
    q=0.8,
    lam=0.01,
    detection=0.9,
    miscall=0.002,
    molecules=20_000,
    full_length=True,
    spike_molecules=0,
    spike_one_sided=False,
    seed=0,
):
    """Reads from the two-step ligation model (adapter efficiency 1).

    Returns obs and, per read, whether it is a foreign molecule (both native
    ends free) -- the truth single-read contamination is checked against.
    """
    rng = np.random.default_rng(seed)
    owner = np.repeat(np.arange(len(POOL)), molecules)
    n = len(owner)

    def ends(native_q):
        native = rng.random(n_rows) < native_q
        tagged = ~native & (rng.random(n_rows) < lam)
        foreign = rng.integers(0, len(POOL), n_rows)
        return native, tagged, foreign

    rows = []
    for spike in (False, True):
        n_rows = spike_molecules if spike else n
        if not n_rows:
            continue
        native_q = 0.0 if spike else q
        own = np.full(n_rows, -1) if spike else owner
        a_native, a_tag, a_foreign = ends(native_q)
        b_native, b_tag, b_foreign = ends(native_q)
        if spike and spike_one_sided:
            b_native[:] = False
            b_tag[:] = False
        a_code = np.where(a_native, own, np.where(a_tag, a_foreign, -1))
        b_code = np.where(b_native, own, np.where(b_tag, b_foreign, -1))
        sequenced = (a_code >= 0) | (b_code >= 0)
        # Read from a barcoded end; with both barcoded, either end first.
        a_first = (a_code >= 0) & ((b_code < 0) | (rng.random(n_rows) < 0.5))
        front = np.where(a_first, a_code, b_code)[sequenced]
        rear = np.where(a_first, b_code, a_code)[sequenced]
        reverse = ~a_first[sequenced]
        foreign_molecule = ~(a_native | b_native)[sequenced]
        rows.append((spike, front, rear, reverse, foreign_molecule))

    frames = []
    for spike, front, rear, reverse, foreign_molecule in rows:
        m = len(front)
        frame = pd.DataFrame(index=range(m))
        for end, code in (("front", front), ("rear", rear)):
            present = code >= 0
            confident = present & (rng.random(m) < detection)
            names = np.array(POOL)[np.clip(code, 0, None)]
            wrong = present & (rng.random(m) < miscall)
            names = np.where(wrong, np.array(KIT)[rng.integers(0, len(KIT), m)], names)
            frame[f"barcode_{end}"] = np.where(present, names, "")
            frame[f"barcode_{end}_edit_distance"] = np.where(
                confident | wrong, 2.0, np.where(present, 9.0, np.nan)
            )
            frame[f"{end}_clip_length"] = np.where(present, 70 if end == "front" else 50, 0)
        frame["reference_start"] = 0 if full_length else rng.integers(100, 400, m)
        frame["reference_end"] = LENGTH if full_length else rng.integers(500, 900, m)
        frame["reference_length"] = LENGTH
        frame["Reference_strand"] = "spike_top" if spike else "amp_top"
        frame["Read_mapping_direction"] = np.where(reverse, "rev", "fwd")
        frame["has_supplementary_alignment"] = False
        frame["truth_foreign"] = foreign_molecule & ~spike
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _report(obs, **kwargs):
    kwargs.setdefault("pool", POOL)
    kwargs.setdefault("kit_size", len(KIT))
    return barcode_tagging_report(obs, barcode_kit="SQK-NBD114-24", **kwargs)


@pytest.mark.parametrize(
    "kit, chemistry",
    [
        ("SQK-NBD114-24", "native_ligation"),
        ("sqk-nbd114.96", "native_ligation"),
        ("SQK-RBK114-24", "rapid"),
        ("SQK-PCB114-24", "pcr"),
        (None, "unknown"),
    ],
)
def test_chemistry_from_kit_name(kit, chemistry):
    assert barcode_chemistry(kit) == chemistry


def test_kit_size_is_read_from_the_packaged_kit():
    assert kit_barcode_count("SQK-NBD114-24") == 24
    assert kit_barcode_count("custom") is None


def test_other_chemistries_are_not_fitted():
    obs = _simulate(molecules=200)
    table, summary = barcode_tagging_report(obs, barcode_kit="SQK-RBK114-24")
    assert table.empty and "rapid" in summary["skipped_reason"]


def test_obs_without_per_end_evidence_raises():
    obs = _simulate(molecules=200).drop(columns=["rear_clip_length"])
    with pytest.raises(KeyError, match="BQC-01"):
        _report(obs)


def test_recovers_ligation_tagging_and_miscalls():
    obs = _simulate()
    table, summary = _report(obs)
    assert summary["q_identifiable"]
    assert summary["q_alpha_1"] == pytest.approx(0.8, abs=0.01)
    assert summary["lambda_alpha_1"] == pytest.approx(0.01, rel=0.25)
    # 16 of the 24 kit barcodes are unused: out-of-pool calls are 2/3 of miscalls.
    assert summary["out_of_pool_call_rate"] == pytest.approx(0.002 * 16 / 24, rel=0.25)
    assert summary["front_detection"] == pytest.approx(0.9 + 0.1 * 0.002, abs=0.01)
    assert set(table.index) == set(POOL)
    assert table["q_alpha_1"].between(0.77, 0.83).all()


def test_single_contamination_matches_the_simulated_truth():
    obs = _simulate(molecules=40_000)
    table, _summary = _report(obs)
    called = obs["barcode_front_edit_distance"] <= 4
    single = called & (obs["rear_clip_length"] == 0) & obs["Reference_strand"].eq("amp_top")
    truth = obs.loc[single, "truth_foreign"].groupby(obs.loc[single, "barcode_front"]).mean()
    model = table["single_contamination"]
    assert model.median() == pytest.approx(truth.reindex(POOL).median(), rel=0.35)
    # Double-same reads need both free ends tagged by the same barcode.
    assert (table["double_contamination"] < model).all()


def test_pool_is_inferred_when_not_declared():
    obs = _simulate()
    _table, summary = barcode_tagging_report(obs, barcode_kit="SQK-NBD114-24", kit_size=24)
    assert summary["pool"] == POOL
    assert summary["pool_source"].startswith("inferred")


def test_genome_wide_reads_keep_contamination_but_not_q():
    obs = _simulate(full_length=False)
    table, summary = _report(obs)
    assert not summary["q_identifiable"]
    assert "q_alpha_1" not in summary and "truncated" in summary["q_unavailable_reason"]
    assert table["single_contamination"].notna().all()


def test_spike_in_measures_tagging_directly():
    obs = _simulate(lam=0.2, spike_molecules=40_000)
    _table, summary = _report(obs, spike_in_references=["spike"])
    spike = summary["spike_in"]
    assert spike["both_ends_ligate"]
    assert spike["lambda_alpha_1"] == pytest.approx(0.2, rel=0.1)


def test_one_sided_spike_in_is_flagged_not_used():
    obs = _simulate(lam=0.2, spike_molecules=40_000, spike_one_sided=True)
    _table, summary = _report(obs, spike_in_references=["spike"])
    spike = summary["spike_in"]
    assert spike["both_ends_ligate"] is False
    assert spike["lambda_alpha_1"] is None
    assert "one reference end" in spike["lambda_unavailable_reason"]


def test_writes_beside_the_contamination_qc(tmp_path):
    paths = write_barcode_tagging_qc(
        _simulate(molecules=2_000),
        tmp_path,
        barcode_kit="SQK-NBD114-24",
        pool=POOL,
        kit_size=24,
        unclassified_dropped=True,
    )
    assert paths["summary"].name == TAGGING_SUMMARY_FILENAME
    assert paths["summary"].parent.name == "barcode_contamination_qc"
    assert "phi_caveat" in paths["summary"].read_text()
    assert len(pd.read_parquet(paths["per_barcode"])) == len(POOL)
