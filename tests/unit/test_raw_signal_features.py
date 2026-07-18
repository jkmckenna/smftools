import types

import numpy as np
import pandas as pd

from smftools.cli import raw_adata
from smftools.informatics import pod5_functions


def _frame():
    return pd.DataFrame(
        [
            {
                "read_id": "r1",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "reference_start": 0,
                "cigar": "3M",
                "aligned_length": 3,
                "mapping_direction": "fwd",
                "sequence": [0, 1, 2],
                "quality": [30, 31, 32],
                "mismatch": [4, 4, 4],
                "modification_signal": [0.0, 0.0, 0.0],
            },
            {
                "read_id": "r2",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "reference_start": 0,
                "cigar": "3M",
                "aligned_length": 3,
                "mapping_direction": "rev",
                "sequence": [2, 1, 0],
                "quality": [30, 31, 32],
                "mismatch": [4, 4, 4],
                "modification_signal": [0.0, 0.0, 0.0],
            },
        ]
    )


def test_attach_signal_features_maps_moves_and_orients(tmp_path, monkeypatch):
    pod5_file = tmp_path / "reads.pod5"
    pod5_file.touch()

    # stride=2, flags [1,1,1] -> 3 bases, one 2-sample block each.
    move_tables = {"r1": ([2, 1, 1, 1], 0), "r2": ([2, 1, 1, 1], 0)}
    monkeypatch.setattr(raw_adata, "_read_move_tables", lambda bam, ids, **kw: move_tables)

    signals = {
        "r1": np.array([10, 10, 20, 20, 30, 30], dtype=float),
        "r2": np.array([100, 100, 200, 200, 300, 300], dtype=float),
    }
    monkeypatch.setattr(
        pod5_functions,
        "iter_pod5_signals",
        lambda path, read_ids=None, **kw: iter([(rid, signals[rid]) for rid in read_ids]),
    )

    cfg = types.SimpleNamespace(
        input_type="pod5",
        input_data_path=str(pod5_file),
        extract_signal_features=True,
        samtools_backend="auto",
    )
    out = raw_adata._attach_signal_features(_frame(), cfg=cfg, aligned_bam=tmp_path / "aln.bam")
    out = out.set_index("read_id")

    # forward read: per-base means in query order
    assert out.loc["r1", "current_mean"] == [10.0, 20.0, 30.0]
    assert out.loc["r1", "dwell"] == [2.0, 2.0, 2.0]
    # reverse read: features flipped into BAM query order
    assert out.loc["r2", "current_mean"] == [300.0, 200.0, 100.0]


def test_attach_signal_features_skips_without_moves(tmp_path, monkeypatch):
    pod5_file = tmp_path / "reads.pod5"
    pod5_file.touch()
    monkeypatch.setattr(raw_adata, "_read_move_tables", lambda bam, ids, **kw: {})
    cfg = types.SimpleNamespace(
        input_type="pod5",
        input_data_path=str(pod5_file),
        extract_signal_features=True,
        samtools_backend="auto",
    )
    out = raw_adata._attach_signal_features(_frame(), cfg=cfg, aligned_bam=tmp_path / "aln.bam")
    assert "current_mean" not in out.columns  # gracefully skipped


def test_attach_signal_features_disabled_by_config(tmp_path):
    cfg = types.SimpleNamespace(
        input_type="pod5", input_data_path=str(tmp_path), extract_signal_features=False
    )
    out = raw_adata._attach_signal_features(_frame(), cfg=cfg, aligned_bam=tmp_path / "aln.bam")
    assert "current_mean" not in out.columns
