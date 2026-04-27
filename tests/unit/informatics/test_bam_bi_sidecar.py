from array import array

import numpy as np
import pandas as pd

from smftools.informatics import bam_functions


def test_derive_bm_from_bi_to_sidecar_serializes_array_tag(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self, query_name, tags):
            self.query_name = query_name
            self.is_secondary = False
            self.is_supplementary = False
            self._tags = tags

        def has_tag(self, tag):
            return tag in self._tags

        def get_tag(self, tag):
            return self._tags[tag]

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(
                [
                    FakeRead(
                        "read_with_bi",
                        {
                            "BC": "barcode01",
                            "bi": array("f", [1.0, 32.0, 45.0, 0.8, 3170.0, 39.0, 0.9]),
                        },
                    ),
                    FakeRead("read_no_bi", {"BC": "barcode02"}),
                    FakeRead("read_unclassified", {}),
                ]
            )

    monkeypatch.setattr(
        bam_functions,
        "_require_pysam",
        lambda: type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")
    sidecar_path = tmp_path / "sample.barcode_tags.parquet"

    bam_functions.derive_bm_from_bi_to_sidecar(bam_path, sidecar_path, threshold=0.65)

    df = pd.read_parquet(sidecar_path).set_index("read_name")
    assert set(df.index) == {"read_with_bi", "read_no_bi"}
    assert df.loc["read_with_bi", "BM"] == "both"
    assert df.loc["read_no_bi", "BM"] == "unknown"

    bi_values = np.asarray(df.loc["read_with_bi", "bi"], dtype=float)
    np.testing.assert_allclose(bi_values, [1.0, 32.0, 45.0, 0.8, 3170.0, 39.0, 0.9])
