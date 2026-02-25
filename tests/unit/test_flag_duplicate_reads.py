"""Unit tests for flag_duplicate_reads logical correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import anndata as ad

from smftools.preprocessing.flag_duplicate_reads import flag_duplicate_reads


@pytest.fixture(autouse=True)
def _suppress_plots(monkeypatch):
    """Patch out all matplotlib plotting inside flag_duplicate_reads.

    The plotting helpers are integration concerns; these tests only care about
    the duplicate-detection logic and should not trigger any rendering.
    """
    import smftools.preprocessing.flag_duplicate_reads as _mod

    monkeypatch.setattr(_mod, "plot_histogram_pages", lambda *a, **kw: None)
    monkeypatch.setattr(_mod, "plot_hamming_vs_metric_pages", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_COL = "Sample"
REF_COL = "Reference_strand"
QUALITY_COL = "read_quality"


def _make_adata(
    X: np.ndarray,
    samples: list,
    refs: list,
    quality: list,
    ref_name: str = "refA",
) -> ad.AnnData:
    """Build a minimal AnnData suitable for flag_duplicate_reads.

    Parameters
    ----------
    X:
        (n_reads, n_positions) float array — the methylation matrix.
    samples:
        Per-read sample label (length n_reads).
    refs:
        Per-read reference label (length n_reads).
    quality:
        Per-read quality score used as keep_best_metric.
    ref_name:
        The reference name used to name var columns for masking.
    """
    n_reads, n_pos = X.shape
    obs = pd.DataFrame(
        {
            SAMPLE_COL: pd.Categorical(samples),
            REF_COL: pd.Categorical(refs),
            QUALITY_COL: quality,
        },
        index=[f"read_{i}" for i in range(n_reads)],
    )
    var = pd.DataFrame(index=[f"pos_{i}" for i in range(n_pos)])
    # var columns used by the var_filters_sets masking logic
    var[f"{ref_name}_site"] = True
    var[f"position_in_{ref_name}"] = True
    return ad.AnnData(X=X.astype(float), obs=obs, var=var)


def _var_filters(ref_name: str = "refA") -> list:
    """Return a var_filters_sets that selects all positions for ref_name."""
    return [[f"{ref_name}_site", f"position_in_{ref_name}"]]


# ---------------------------------------------------------------------------
# Bug 1 — cluster ID collision in final keeper enforcement
# ---------------------------------------------------------------------------


class TestClusterIdCollision:
    """
    Cluster IDs are assigned independently per (sample, ref) group, both
    starting from 0.  After ad.concat the final keeper enforcement must not
    group reads from different (sample, ref) combinations under the same
    cluster ID bucket.

    Scenario
    --------
    Two groups:

      sampleA / refA:  A_keeper (quality=0.8) and A_dup (quality=0.7)
                       — near-duplicates, Hamming = 1/20 = 0.05 < threshold
      sampleB / refA:  B_keeper (quality=0.5) and B_dup (quality=0.9)
                       — near-duplicates, Hamming = 1/20 = 0.05 < threshold

    Both groups produce local cluster_id = 0 for their duplicate pair.

    Because A_keeper is close to A_dup in lexicographic order its
    fwd_hamming_to_next < threshold, so the is_duplicate_distance flag marks
    A_keeper as a duplicate.  The keeper enforcement is supposed to rescue it.

    WITHOUT the fix: the groupby mixes all 4 reads under cluster_id=0 and
    picks B_dup (quality=0.9, the global best) as "keeper" — A_keeper is never
    rescued and stays is_duplicate=True (false positive).

    WITH the fix: groupby keys include (cluster_id, sample, ref).  Each group
    is processed independently:
      - sampleA cluster 0 → A_keeper rescued
      - sampleB cluster 0 → B_dup rescued
    """

    N_POS = 20
    THRESHOLD = 0.07  # 1/20 = 0.05 < 0.07
    REF = "refA"

    def _build(self):
        # sampleA: A_keeper all-zeros, A_dup one diff at last position
        A_keeper = [0.0] * self.N_POS
        A_dup = [0.0] * (self.N_POS - 1) + [1.0]
        # sampleB: B_keeper all-ones, B_dup one diff at last position
        B_keeper = [1.0] * self.N_POS
        B_dup = [1.0] * (self.N_POS - 1) + [0.0]

        X = np.array([A_keeper, A_dup, B_keeper, B_dup])
        samples = ["sampleA", "sampleA", "sampleB", "sampleB"]
        refs = [self.REF] * 4
        # A_keeper best in sampleA, B_dup best in sampleB (but has higher
        # global quality than A_keeper to expose the collision bug)
        quality = [0.8, 0.7, 0.5, 0.9]
        return _make_adata(X, samples, refs, quality, ref_name=self.REF)

    def test_each_group_keeper_is_not_duplicate(self):
        adata = self._build()
        adata_unique, adata_full = flag_duplicate_reads(
            adata,
            var_filters_sets=_var_filters(self.REF),
            distance_threshold=self.THRESHOLD,
            obs_reference_col=REF_COL,
            sample_col=SAMPLE_COL,
            keep_best_metric=QUALITY_COL,
            keep_best_higher=True,
            window_size=50,
            min_overlap_positions=5,
            do_hierarchical=False,
            output_directory=None,
        )
        obs = adata_full.obs

        # A_keeper (read_0) must NOT be a duplicate — it is the best in sampleA
        assert not obs.loc["read_0", "is_duplicate"], (
            "A_keeper (read_0, quality=0.8) was incorrectly marked as a "
            "duplicate — cluster ID collision between sampleA and sampleB."
        )

        # A_dup (read_1) must be a duplicate
        assert obs.loc["read_1", "is_duplicate"], "A_dup (read_1) should be a duplicate."

        # B_dup (read_3, quality=0.9) is the best in sampleB — should be keeper
        assert not obs.loc["read_3", "is_duplicate"], (
            "B_dup (read_3, quality=0.9) should be the keeper for sampleB."
        )

        # B_keeper (read_2, quality=0.5) is the worse read in sampleB — duplicate
        assert obs.loc["read_2", "is_duplicate"], "B_keeper (read_2) should be a duplicate."

    def test_unique_adata_contains_one_keeper_per_group(self):
        adata = self._build()
        adata_unique, _ = flag_duplicate_reads(
            adata,
            var_filters_sets=_var_filters(self.REF),
            distance_threshold=self.THRESHOLD,
            obs_reference_col=REF_COL,
            sample_col=SAMPLE_COL,
            keep_best_metric=QUALITY_COL,
            keep_best_higher=True,
            window_size=50,
            min_overlap_positions=5,
            do_hierarchical=False,
            output_directory=None,
        )
        assert adata_unique.n_obs == 2, (
            f"Expected 2 unique reads (one keeper per group), got {adata_unique.n_obs}."
        )
        assert "read_0" in adata_unique.obs.index, "A_keeper (read_0) missing from unique set."
        assert "read_3" in adata_unique.obs.index, "B_dup (read_3) missing from unique set."


# ---------------------------------------------------------------------------
# Bug 2 — reverse pass excludes forward-matched reads, leaving rev_hamming_to_prev unpopulated
# ---------------------------------------------------------------------------


class TestReversePassAnnotations:
    """
    Reads matched in the forward pass were previously excluded from the reverse
    pass entirely.  This meant their rev_hamming_to_prev distance was never
    recorded (left as NaN), even though the reverse pass would have computed
    it correctly.

    The reverse pass does NOT find geometrically new duplicate pairs — ascending
    and descending sort orders are mirrors, so the window comparisons cover
    identical sets of read pairs.  What the full reverse pass adds is complete
    annotation of rev_hamming_to_prev for ALL reads, including those that
    participated in forward pairs.

    Scenario (20 positions, threshold=0.07)
    ----------------------------------------
    Two reads in one (sample, ref) group:

      R_A = [0]*20              quality=0.9  (keeper)
      R_D = [0]*19 + [1]        quality=0.3  (duplicate)

    Hamming(R_A, R_D) = 1/20 = 0.05 < 0.07 → matched in forward pass.
    Both added to involved_in_fwd → excluded from buggy reverse pass.

    Ascending lex: R_A(0), R_D(1).
    Forward records: R_A.fwd_hamming_to_next = 0.05.

    Descending: R_D(0), R_A(1).
    With the full reverse pass (fix): R_A.rev_hamming_to_prev = 0.05.
    With the buggy reduced reverse pass: both excluded → rev pass skipped
    entirely → R_A.rev_hamming_to_prev = NaN.
    """

    N_POS = 20
    THRESHOLD = 0.07
    REF = "refA"

    def _build(self):
        R_A = [0.0] * self.N_POS
        R_D = [0.0] * (self.N_POS - 1) + [1.0]
        X = np.array([R_A, R_D])
        samples = ["sampleA"] * 2
        refs = [self.REF] * 2
        quality = [0.9, 0.3]
        return _make_adata(X, samples, refs, quality, ref_name=self.REF)

    def _run(self, adata):
        return flag_duplicate_reads(
            adata,
            var_filters_sets=_var_filters(self.REF),
            distance_threshold=self.THRESHOLD,
            obs_reference_col=REF_COL,
            sample_col=SAMPLE_COL,
            keep_best_metric=QUALITY_COL,
            keep_best_higher=True,
            window_size=50,
            min_overlap_positions=5,
            do_hierarchical=False,
            output_directory=None,
        )

    def test_rev_hamming_to_prev_populated_for_forward_matched_reads(self):
        """rev_hamming_to_prev must be set for reads that were in a forward pair.

        With the buggy code, both reads are excluded from the reverse pass
        because they appeared in a forward pair, leaving rev_hamming_to_prev=NaN.
        With the fix, the reverse pass runs on all reads and correctly records
        the distance.
        """
        _, adata_full = self._run(self._build())
        obs = adata_full.obs

        rev_val = obs.loc["read_0", "rev_hamming_to_prev"]
        expected = 1.0 / self.N_POS  # = 0.05

        assert not np.isnan(rev_val), (
            "rev_hamming_to_prev for read_0 (R_A) should be populated by the "
            "full reverse pass, not left as NaN."
        )
        assert np.isclose(rev_val, expected, atol=1e-6), (
            f"Expected rev_hamming_to_prev ≈ {expected:.4f}, got {rev_val:.4f}."
        )

    def test_forward_matched_read_correctly_deduplicated(self):
        """Basic sanity: the lower-quality read is a duplicate, keeper is not."""
        _, adata_full = self._run(self._build())
        obs = adata_full.obs

        assert not obs.loc["read_0", "is_duplicate"], "R_A (quality=0.9) should be the keeper."
        assert obs.loc["read_1", "is_duplicate"], "R_D (quality=0.3) should be a duplicate."


# ---------------------------------------------------------------------------
# Parallel equivalence — n_jobs > 1 must give identical results to n_jobs = 1
# ---------------------------------------------------------------------------


class TestParallelEquivalence:
    """Verify that n_jobs=2 produces identical results to n_jobs=1.

    Uses a 3-group AnnData (3 samples × 1 ref) so that the parallel executor
    has multiple independent units of work to dispatch.
    """

    N_POS = 20
    THRESHOLD = 0.07
    REF = "refA"

    def _build(self):
        """Build a 3-group AnnData with one near-duplicate pair per group."""
        rows = []
        samples = []
        refs = []
        quality = []
        for i, sample_name in enumerate(["sampleA", "sampleB", "sampleC"]):
            base_val = float(i % 2)
            keeper = [base_val] * self.N_POS
            dup = keeper[:]
            dup[-1] = 1.0 - base_val  # flip last position
            rows.append(keeper)
            rows.append(dup)
            samples += [sample_name, sample_name]
            refs += [self.REF, self.REF]
            quality += [0.9, 0.3]

        X = np.array(rows)
        return _make_adata(X, samples, refs, quality, ref_name=self.REF)

    def _run(self, adata, n_jobs):
        return flag_duplicate_reads(
            adata,
            var_filters_sets=_var_filters(self.REF),
            distance_threshold=self.THRESHOLD,
            obs_reference_col=REF_COL,
            sample_col=SAMPLE_COL,
            keep_best_metric=QUALITY_COL,
            keep_best_higher=True,
            window_size=50,
            min_overlap_positions=5,
            do_hierarchical=False,
            output_directory=None,
            n_jobs=n_jobs,
        )

    def test_parallel_matches_serial(self):
        """n_jobs=2 must produce the same per-read annotations as n_jobs=1."""
        adata1 = self._build()
        adata2 = self._build()
        _, adata_full_1 = self._run(adata1, n_jobs=1)
        _, adata_full_2 = self._run(adata2, n_jobs=2)

        obs1 = adata_full_1.obs.sort_index()
        obs2 = adata_full_2.obs.sort_index()

        dup_cols = [
            "sequence__is_duplicate",
            "sequence__merged_cluster_id",
            "sequence__cluster_size",
            "sequence__lex_is_keeper",
            "sequence__lex_is_duplicate",
            "is_duplicate",
        ]
        for col in dup_cols:
            if col in obs1.columns:
                pd.testing.assert_series_equal(
                    obs1[col].reset_index(drop=True),
                    obs2[col].reset_index(drop=True),
                    check_names=False,
                    obj=f"column '{col}'",
                )
