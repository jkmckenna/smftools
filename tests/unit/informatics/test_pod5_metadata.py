import types

import pandas as pd

from smftools.informatics import pod5_functions
from smftools.informatics.raw_store import write_raw_store
from smftools.readwrite import safe_read_h5ad


class _FakeReader:
    def __init__(self, reads):
        self._reads = reads

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reads(self):
        return iter(self._reads)


def _fake_read(read_id, channel, num_samples, current):
    ns = types.SimpleNamespace
    return ns(
        read_id=read_id,
        read_number=7,
        pore=ns(channel=channel, well=2, pore_type="R10"),
        num_samples=num_samples,
        median_before=210.5,
        open_pore_level=225.0,
        calibration=ns(offset=-4.0, scale=0.18),
        predicted_scaling=ns(scale=1.1, shift=0.2),
        tracked_scaling=ns(scale=1.0, shift=0.0),
        start_sample=1000,
        num_minknow_events=12,
        end_reason=ns(reason=ns(name="signal_positive"), forced=False),
        run_info=ns(
            sample_rate=5000,
            acquisition_id="acq123",
            flow_cell_id="FC1",
            sequencing_kit="SQK-LSK114",
            experiment_name="exp",
            sample_id="sampleA",
        ),
        signal_pa=current,
    )


def _install_fake_pod5(monkeypatch, reads):
    monkeypatch.setattr(pod5_functions, "p5", types.SimpleNamespace(Reader=lambda _p: _FakeReader(reads)))


def test_extract_pod5_metadata_scalar_columns(tmp_path, monkeypatch):
    pod5_file = tmp_path / "reads.pod5"
    pod5_file.touch()
    _install_fake_pod5(monkeypatch, [_fake_read("readA", 5, 4000, [1.0, 2.0]), _fake_read("readB", 9, 5000, [3.0])])

    frame = pod5_functions.extract_pod5_read_metadata(pod5_file, verbose=False)

    assert list(frame.index) == ["readA", "readB"]
    assert frame.loc["readA", "pod5_origin"] == "reads.pod5"
    assert frame.loc["readA", "pod5_channel"] == 5
    assert frame.loc["readA", "pod5_pore_type"] == "R10"
    assert frame.loc["readB", "pod5_num_samples"] == 5000
    assert frame.loc["readA", "pod5_end_reason"] == "signal_positive"
    assert frame.loc["readA", "pod5_sequencing_kit"] == "SQK-LSK114"
    assert frame.loc["readA", "pod5_sample_rate"] == 5000
    assert "pod5_current_pa" not in frame.columns  # excluded by default


def test_extract_pod5_metadata_target_filter_and_current(tmp_path, monkeypatch):
    pod5_file = tmp_path / "reads.pod5"
    pod5_file.touch()
    _install_fake_pod5(monkeypatch, [_fake_read("readA", 5, 4000, [1.0, 2.0]), _fake_read("readB", 9, 5000, [3.0])])

    frame = pod5_functions.extract_pod5_read_metadata(
        pod5_file, target_ids=["readB"], include_current=True, verbose=False
    )

    assert list(frame.index) == ["readB"]
    assert frame.loc["readB", "pod5_current_pa"] == [3.0]


def _ragged_frame_with_pod5() -> pd.DataFrame:
    rows = []
    for offset, (read_id, ref, length) in enumerate(
        [("r1", "ref1", 4), ("r2", "ref1", 4)]
    ):
        rows.append(
            {
                "read_id": read_id,
                "reference": ref,
                "Reference_strand": f"{ref}_top",
                "sample": "bc01",
                "barcode": "bc01",
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": f"{length}M",
                "aligned_length": length,
                "sequence": [offset % 4] * length,
                "quality": [30] * length,
                "mismatch": [4] * length,
                "modification_signal": [float(offset)] * length,
                # POD5 metadata: scalar columns + a ragged current trace
                "pod5_origin": "reads.pod5",
                "pod5_channel": 5 + offset,
                "pod5_median_before": 200.0 + offset,
                "pod5_current_pa": [1.0, 2.0, 3.0] * length,
            }
        )
    return pd.DataFrame(rows)


def test_pod5_scalar_metadata_lands_on_spine_but_current_stays_in_parquet(tmp_path):
    paths = write_raw_store(
        _ragged_frame_with_pod5(),
        tmp_path,
        reference_lengths={"ref1_top": 4},
        shard_size=10,
    )
    spine, _ = safe_read_h5ad(paths["spine"])
    # scalar pod5_* columns are carried onto the thin spine
    assert "pod5_origin" in spine.obs.columns
    assert "pod5_channel" in spine.obs.columns
    assert "pod5_median_before" in spine.obs.columns
    assert set(map(str, spine.obs["pod5_origin"])) == {"reads.pod5"}
    # the ragged current trace is NOT promoted to the spine
    assert "pod5_current_pa" not in spine.obs.columns
    # but it is retained in the parquet shard
    shard = pd.read_parquet(paths["ragged_store"][0])
    assert "pod5_current_pa" in shard.columns
