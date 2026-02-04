"""Tests for UMI preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from smftools.preprocessing.preprocess_umi_annotations import (
    _cluster_umis_by_edit_distance,
    _contains_n,
    _edit_distance,
    _has_homopolymer_run,
    _homopolymer_fraction,
    _unique_base_count,
    preprocess_umi_annotations,
    validate_umi,
)


class TestUmiValidationHelpers:
    """Tests for UMI validation helper functions."""

    def test_homopolymer_fraction_all_same(self):
        assert _homopolymer_fraction("AAAA") == 1.0

    def test_homopolymer_fraction_half(self):
        assert _homopolymer_fraction("AATT") == 0.5

    def test_homopolymer_fraction_empty(self):
        assert _homopolymer_fraction("") == 1.0

    def test_unique_base_count_all_same(self):
        assert _unique_base_count("AAAA") == 1

    def test_unique_base_count_all_different(self):
        assert _unique_base_count("ACGT") == 4

    def test_unique_base_count_two_bases(self):
        assert _unique_base_count("ATAT") == 2

    def test_unique_base_count_with_n(self):
        # N is not counted as a valid base
        assert _unique_base_count("ACGN") == 3

    def test_contains_n_true(self):
        assert _contains_n("ACGN") is True

    def test_contains_n_false(self):
        assert _contains_n("ACGT") is False

    def test_contains_n_lowercase(self):
        assert _contains_n("acgn") is True

    def test_has_homopolymer_run_true(self):
        assert _has_homopolymer_run("ACGTAAAAA", max_run=4) is True

    def test_has_homopolymer_run_false(self):
        assert _has_homopolymer_run("ACGTAAAA", max_run=4) is False

    def test_has_homopolymer_run_exactly_at_threshold(self):
        assert _has_homopolymer_run("ACGTAAAA", max_run=3) is True


class TestEditDistance:
    """Tests for edit distance calculation."""

    def test_edit_distance_identical(self):
        assert _edit_distance("ACGT", "ACGT") == 0

    def test_edit_distance_one_substitution(self):
        assert _edit_distance("ACGT", "ACTT") == 1

    def test_edit_distance_one_insertion(self):
        assert _edit_distance("ACGT", "ACGTT") == 1

    def test_edit_distance_one_deletion(self):
        assert _edit_distance("ACGT", "ACT") == 1

    def test_edit_distance_empty_strings(self):
        assert _edit_distance("", "") == 0
        assert _edit_distance("ACGT", "") == 4
        assert _edit_distance("", "ACGT") == 4

    def test_edit_distance_with_max_dist(self):
        # Distance is 2, but max_dist is 1
        dist = _edit_distance("ACGT", "AATT", max_dist=1)
        assert dist > 1


class TestValidateUmi:
    """Tests for UMI validation."""

    def test_validate_umi_valid(self):
        is_valid, reason = validate_umi("ACGTACGT")
        assert is_valid is True
        assert reason == "valid"

    def test_validate_umi_none(self):
        is_valid, reason = validate_umi(None)
        assert is_valid is False
        assert reason == "missing"

    def test_validate_umi_empty(self):
        is_valid, reason = validate_umi("")
        assert is_valid is False
        assert reason == "empty"

    def test_validate_umi_contains_n(self):
        is_valid, reason = validate_umi("ACGNACGT")
        assert is_valid is False
        assert reason == "contains_N"

    def test_validate_umi_high_homopolymer(self):
        # 6/8 = 75% is > 70% default threshold
        is_valid, reason = validate_umi("AAAAAACT")
        assert is_valid is False
        assert reason == "high_homopolymer_fraction"

    def test_validate_umi_low_complexity(self):
        # Only 1 unique base - but homopolymer fraction check happens first
        is_valid, reason = validate_umi("AAAA", min_unique_bases=2, max_homopolymer_fraction=1.1)
        assert is_valid is False
        assert reason == "low_complexity"

    def test_validate_umi_wrong_length(self):
        is_valid, reason = validate_umi("ACGT", expected_length=8)
        assert is_valid is False
        assert "wrong_length" in reason

    def test_validate_umi_homopolymer_run(self):
        # ACAAAAAT has 5 A's which exceeds max_homopolymer_run=4
        # But we need to set max_homopolymer_fraction high enough to not fail that check first
        is_valid, reason = validate_umi("ACAAAAAT", max_homopolymer_run=4, max_homopolymer_fraction=0.8)
        assert is_valid is False
        assert "homopolymer_run" in reason


class TestClusterUmisByEditDistance:
    """Tests for UMI clustering."""

    def test_cluster_empty_list(self):
        result = _cluster_umis_by_edit_distance([])
        assert result == {}

    def test_cluster_single_umi(self):
        result = _cluster_umis_by_edit_distance(["ACGT"])
        assert result == {"ACGT": "ACGT"}

    def test_cluster_identical_umis(self):
        result = _cluster_umis_by_edit_distance(["ACGT", "ACGT", "ACGT"])
        assert result == {"ACGT": "ACGT"}

    def test_cluster_one_edit_apart_directional(self):
        # High-count UMI should absorb low-count
        umis = ["ACGT", "ACGT", "ACGT", "ACTT"]  # ACGT appears 3x, ACTT 1x
        result = _cluster_umis_by_edit_distance(umis, max_edit_distance=1, directional=True)
        # ACTT should be mapped to ACGT
        assert result["ACGT"] == "ACGT"
        assert result["ACTT"] == "ACGT"

    def test_cluster_too_far_apart(self):
        # These are 2 edits apart
        umis = ["ACGT", "AATT"]
        result = _cluster_umis_by_edit_distance(umis, max_edit_distance=1, directional=True)
        assert result["ACGT"] == "ACGT"
        assert result["AATT"] == "AATT"

    def test_cluster_non_directional(self):
        umis = ["ACGT", "ACGT", "ACTT"]
        result = _cluster_umis_by_edit_distance(umis, max_edit_distance=1, directional=False)
        # Both should map to the same cluster
        assert result["ACGT"] == result["ACTT"]


class TestPreprocessUmiAnnotations:
    """Tests for the main preprocessing function."""

    @pytest.fixture
    def simple_adata(self):
        """Create a simple AnnData object with UMI columns."""
        import anndata as ad

        n_obs = 10
        n_vars = 5

        obs = pd.DataFrame(
            {
                "U1": [
                    "ACGTACGT",
                    "ACGTACGT",
                    "ACGTACTT",  # 1 edit from ACGTACGT
                    "TGCATGCA",
                    "TGCATGCA",
                    "AAAAAAAA",  # homopolymer - should be invalid
                    None,  # missing
                    "ACGNACGT",  # contains N - should be invalid (same length as others)
                    "ACGT",  # wrong length (4 vs inferred 8)
                    "ATATATAT",  # low complexity (only 2 bases)
                ],
                "U2": [
                    "GGGGCCCC",
                    "GGGGCCCC",
                    "GGGGCCCC",
                    "TTTTAAAA",
                    "TTTTAAAA",
                    "CCCCGGGG",
                    "CCCCGGGG",
                    "CCCCGGGG",
                    "CCCCGGGG",
                    "CCCCGGGG",
                ],
                "Barcode": ["S1"] * 5 + ["S2"] * 5,
                "Reference_strand": ["ref1"] * 10,
            },
            index=[f"read_{i}" for i in range(n_obs)],
        )

        X = np.random.rand(n_obs, n_vars)

        return ad.AnnData(X=X, obs=obs)

    def test_preprocess_basic(self, simple_adata):
        """Test basic preprocessing workflow."""
        result = preprocess_umi_annotations(
            simple_adata,
            umi_cols=["U1", "U2"],
            max_homopolymer_fraction=0.7,
            min_unique_bases=3,
            max_homopolymer_run=4,
            cluster_max_edit_distance=1,
        )

        # Check that new columns were created
        assert "U1_valid" in result.obs.columns
        assert "U1_invalid_reason" in result.obs.columns
        assert "U1_cluster" in result.obs.columns
        assert "U1_cluster_size" in result.obs.columns
        assert "U2_valid" in result.obs.columns
        assert "RX_cluster" in result.obs.columns
        assert "umi_cluster_key" in result.obs.columns

        # Check validation results
        # Index 0, 1, 2 should have valid U1 (ACGTACGT, ACGTACGT, ACGTACTT)
        assert result.obs.loc["read_0", "U1_valid"] == True
        assert result.obs.loc["read_1", "U1_valid"] == True
        assert result.obs.loc["read_2", "U1_valid"] == True

        # Index 5 should be invalid (homopolymer AAAAAAAA)
        assert result.obs.loc["read_5", "U1_valid"] == False
        assert "homopolymer" in result.obs.loc["read_5", "U1_invalid_reason"]

        # Index 6 should be invalid (missing)
        assert result.obs.loc["read_6", "U1_valid"] == False
        assert result.obs.loc["read_6", "U1_invalid_reason"] == "missing"

        # Index 7 should be invalid (contains N)
        assert result.obs.loc["read_7", "U1_valid"] == False
        assert result.obs.loc["read_7", "U1_invalid_reason"] == "contains_N"

        # Check that uns flag was set
        assert result.uns.get("preprocess_umi_annotations_performed") == True

    def test_preprocess_clustering(self, simple_adata):
        """Test that clustering groups similar UMIs."""
        result = preprocess_umi_annotations(
            simple_adata,
            umi_cols=["U1"],
            cluster_max_edit_distance=1,
            min_unique_bases=2,  # Allow ATATATAT to pass
        )

        # ACGTACGT and ACGTACTT are 1 edit apart, should cluster together
        cluster_0 = result.obs.loc["read_0", "U1_cluster"]
        cluster_2 = result.obs.loc["read_2", "U1_cluster"]

        # They should have the same cluster representative
        assert cluster_0 == cluster_2

        # TGCATGCA should be in a different cluster
        cluster_3 = result.obs.loc["read_3", "U1_cluster"]
        assert cluster_3 != cluster_0

    def test_preprocess_bypass(self, simple_adata):
        """Test that bypass=True skips processing."""
        result = preprocess_umi_annotations(simple_adata, bypass=True)

        # No new columns should be added
        assert "U1_valid" not in result.obs.columns

    def test_preprocess_force_redo(self, simple_adata):
        """Test that force_redo reruns processing."""
        # First run
        result = preprocess_umi_annotations(simple_adata, umi_cols=["U1"])
        assert result.uns.get("preprocess_umi_annotations_performed") == True

        # Second run without force_redo should skip
        result.obs["U1_valid"] = False  # Modify to check if it gets recomputed
        result = preprocess_umi_annotations(result, umi_cols=["U1"], force_redo=False)
        assert result.obs.loc["read_0", "U1_valid"] == False  # Should not be recomputed

        # With force_redo should recompute
        result = preprocess_umi_annotations(result, umi_cols=["U1"], force_redo=True)
        assert result.obs.loc["read_0", "U1_valid"] == True  # Should be recomputed

    def test_preprocess_missing_columns(self, simple_adata):
        """Test handling of missing UMI columns."""
        result = preprocess_umi_annotations(
            simple_adata,
            umi_cols=["U1", "NONEXISTENT"],
        )

        # Should process U1 but not crash on missing column
        assert "U1_valid" in result.obs.columns
        assert "NONEXISTENT_valid" not in result.obs.columns

    def test_preprocess_stats_stored(self, simple_adata):
        """Test that preprocessing stats are stored in uns."""
        result = preprocess_umi_annotations(simple_adata, umi_cols=["U1", "U2"])

        assert "umi_preprocessing_stats" in result.uns
        stats = result.uns["umi_preprocessing_stats"]

        assert "validation_stats" in stats
        assert "total_clusters" in stats
        assert "params" in stats
        assert stats["params"]["umi_cols"] == ["U1", "U2"]
