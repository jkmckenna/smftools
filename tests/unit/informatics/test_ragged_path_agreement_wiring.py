"""Barcode agreement must be reported on the ragged path (`EGL-29b` wiring fix).

The reporting was originally wired beside the *dense* AnnData attach in
`load_adata`. The ragged/raw path returns before reaching it, so the feature
never ran for the partitioned pipeline -- which is the path every current run
takes. The logic was fine; it was attached to dead code.

These pin the wiring rather than the logic, because the logic already has its
own tests and the wiring is what was wrong.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.cli import raw_adata
from smftools.informatics.demux_agreement import AGREEMENT_COLUMN

pytestmark = pytest.mark.unit


def test_the_ragged_attach_calls_the_agreement_report():
    """A direct check that the call exists on the path that actually runs."""
    source = inspect.getsource(raw_adata._attach_obs_metadata)
    assert "report_barcode_agreement" in source


def test_agreement_runs_where_the_sidecar_is_attached(tmp_path, caplog):
    """End to end through `_attach_obs_metadata` with a real sidecar."""
    from smftools.informatics.barcode_sidecar import BARCODE_IDENTITY_COLUMNS

    reads = [f"read{index}" for index in range(6)]
    sidecar = pd.DataFrame(
        {
            **{column: "" for column in BARCODE_IDENTITY_COLUMNS},
            "read_name": reads,
            "barcode": ["bc01"] * 6,
            "BC": ["bc01"] * 4 + ["bc02"] * 2,
        }
    )
    sidecar["identity_schema_version"] = 1
    path = tmp_path / "identity.parquet"
    sidecar.to_parquet(path, index=False)

    frame = pd.DataFrame({"read_id": reads, "cigar": ["10M"] * 6})

    with caplog.at_level("INFO", logger="smftools.informatics.demux_agreement"):
        result = raw_adata._attach_obs_metadata(
            frame,
            cfg=SimpleNamespace(
                experiment_name="exp", conversion_types=[], smf_modality="deaminase"
            ),
            bam_path=tmp_path / "absent.bam",
            barcode_sidecar=path,
            umi_sidecar=None,
            metrics={},
        )

    assert AGREEMENT_COLUMN in result.columns
    assert any("Barcode agreement" in record.message for record in caplog.records)


def test_disagreements_are_visible_on_the_ragged_path(tmp_path, caplog):
    """The point of the wiring: a wrong kit or assignment must surface here."""
    from smftools.informatics.barcode_sidecar import BARCODE_IDENTITY_COLUMNS

    reads = [f"read{index}" for index in range(10)]
    sidecar = pd.DataFrame(
        {
            **{column: "" for column in BARCODE_IDENTITY_COLUMNS},
            "read_name": reads,
            "barcode": ["bc01"] * 10,
            "BC": ["bc02"] * 10,
        }
    )
    sidecar["identity_schema_version"] = 1
    path = tmp_path / "identity.parquet"
    sidecar.to_parquet(path, index=False)

    frame = pd.DataFrame({"read_id": reads, "cigar": ["10M"] * 10})

    with caplog.at_level("WARNING", logger="smftools.informatics.demux_agreement"):
        raw_adata._attach_obs_metadata(
            frame,
            cfg=SimpleNamespace(
                experiment_name="exp", conversion_types=[], smf_modality="deaminase"
            ),
            bam_path=tmp_path / "absent.bam",
            barcode_sidecar=path,
            umi_sidecar=None,
            metrics={},
        )

    assert any("disagree" in record.message for record in caplog.records)


def test_no_sidecar_is_not_an_error(tmp_path):
    frame = pd.DataFrame({"read_id": ["r1"], "cigar": ["10M"]})
    result = raw_adata._attach_obs_metadata(
        frame,
        cfg=SimpleNamespace(experiment_name="exp", conversion_types=[], smf_modality="deaminase"),
        bam_path=tmp_path / "absent.bam",
        barcode_sidecar=None,
        umi_sidecar=None,
        metrics={},
    )
    assert AGREEMENT_COLUMN not in result.columns
