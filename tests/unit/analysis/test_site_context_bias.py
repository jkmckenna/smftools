"""SCB-01: site-context bias statistics (pure functions)."""

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.site_context_bias import (
    accumulate_site_calls,
    group_differences,
    kmer_rates,
    offset_enrichment,
    site_contexts,
    site_table,
    strand_of,
    unambiguous,
    wilson_interval,
)

#            0123456789
SEQUENCE = "AACTGGCAAT"


def _sites(rows):
    return pd.DataFrame(
        rows, columns=["group", "physical_reference", "position", "observed", "modified"]
    )


def test_strand_of():
    assert strand_of("6B6_bottom") == ("6B6", "bottom")
    assert strand_of("6B6_enh_del_top") == ("6B6_enh_del", "top")
    assert strand_of("locus") == ("locus", "top")


def test_contexts_are_strand_oriented_and_padded():
    sites = _sites(
        [
            ("g", "ref_top", 2, 4, 1),  # C at 2: forward window
            ("g", "ref_bottom", 5, 4, 1),  # G at 5: reverse complement
            ("g", "ref_top", 0, 1, 0),  # off the left end
            ("g", "ref_top", 9, 1, 0),  # off the right end
        ]
    )
    out = site_contexts(sites, {"ref": SEQUENCE}, flank=2)
    assert list(out["context"]) == ["AACTG", "TGCCA", "NNAAC", "AATNN"]
    assert out["context"].str[2].tolist()[:2] == ["C", "C"]  # the site is central, as C
    assert out["rate"].tolist()[0] == 0.25


def test_sequence_for_reads_another_reference():
    sites = _sites([("g", "del_top", 2, 1, 1)])
    out = site_contexts(sites, {"frame": SEQUENCE}, flank=1, sequence_for={"del_top": "frame"})
    assert out["context"].tolist() == ["ACT"]
    with pytest.raises(KeyError, match="no sequence"):
        site_contexts(sites, {"frame": SEQUENCE}, flank=1)


def test_accumulate_and_site_table():
    counts: dict = {}
    calls = np.array([[1, 0, np.nan], [1, 1, np.nan]], dtype=float)
    observed = ~np.isnan(calls)
    for _ in range(2):  # two identical batches add up
        accumulate_site_calls(
            counts, keys=[("a", "ref_top"), ("b", "ref_top")], calls=calls, observed=observed
        )
    table = site_table(counts, positions=[10, 11, 12])
    assert table.to_dict("records") == [
        {
            "group": "a",
            "physical_reference": "ref_top",
            "position": 10,
            "observed": 2,
            "modified": 2,
        },
        {
            "group": "a",
            "physical_reference": "ref_top",
            "position": 11,
            "observed": 2,
            "modified": 0,
        },
        {
            "group": "b",
            "physical_reference": "ref_top",
            "position": 10,
            "observed": 2,
            "modified": 2,
        },
        {
            "group": "b",
            "physical_reference": "ref_top",
            "position": 11,
            "observed": 2,
            "modified": 2,
        },
    ]


def _planted(group="g", rate_with_t=1.0, rate_without=0.0):
    """Sites whose context has T at +1 are modified at one rate, the rest at another."""
    contexts = ["ACT", "GCT", "TCT", "ACA", "GCG", "TCC", "CCA", "ACG"]
    rows = []
    for index, context in enumerate(contexts):
        rate = rate_with_t if context[2] == "T" else rate_without
        rows.append(
            {
                "group": group,
                "physical_reference": "ref_top",
                "position": index,
                "observed": 100,
                "modified": int(round(100 * rate)),
                "context": context,
            }
        )
    return pd.DataFrame(rows)


def test_enrichment_finds_the_planted_base():
    table = offset_enrichment(_planted(), flank=1)
    plus_one = table[table["offset"] == 1].set_index("base")["log2_enrichment"]
    assert plus_one["T"] > 1
    assert (plus_one.drop(["T", "N"]) < 0).all()
    assert np.isnan(plus_one["N"])  # never observed: no information
    centre = table[table["offset"] == 0].set_index("base")["log2_enrichment"]
    assert abs(centre["C"]) < 0.1  # every site is a C: no information at the centre


def test_kmer_rates_and_intervals():
    rates = kmer_rates(_planted(rate_with_t=0.8, rate_without=0.1), flank=1, k=3)
    act = rates.set_index("kmer").loc["ACT"]
    assert act["rate"] == pytest.approx(0.8) and act["n_sites"] == 1
    assert (rates["rate_low"] <= rates["rate"]).all() and (
        rates["rate"] <= rates["rate_high"]
    ).all()
    centre = kmer_rates(_planted(), flank=1, k=1).set_index("kmer")
    assert centre.loc["C", "n_sites"] == 8
    with pytest.raises(ValueError):
        kmer_rates(_planted(), flank=1, k=2)
    with pytest.raises(ValueError):
        kmer_rates(_planted(), flank=1, k=5)


def test_wilson_interval_known_values():
    low, high = wilson_interval([5], [10])
    assert low[0] == pytest.approx(0.2366, abs=1e-3)
    assert high[0] == pytest.approx(0.7634, abs=1e-3)


def test_group_differences_against_a_reference_group():
    sites = pd.concat(
        [_planted("enzyme_a", 1.0, 0.0), _planted("enzyme_b", 0.5, 0.5)], ignore_index=True
    )
    table = offset_enrichment(sites, flank=1)
    diff = group_differences(table, reference_group="enzyme_b")
    assert set(diff["group"]) == {"enzyme_a"}
    t_plus_one = diff[(diff["offset"] == 1) & (diff["base"] == "T")]["difference"].iloc[0]
    assert t_plus_one > 1  # enzyme_a prefers +1 T; enzyme_b is unbiased
    with pytest.raises(KeyError):
        group_differences(table, reference_group="missing")


def test_ambiguous_contexts_are_dropped_by_default():
    sites = pd.concat(
        [
            _planted(),
            pd.DataFrame(
                [
                    {
                        "group": "g",
                        "physical_reference": "ref_top",
                        "position": 99,
                        "observed": 100,
                        "modified": 100,
                        "context": "NCT",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    assert len(unambiguous(sites)) == len(sites) - 1
    dropped = offset_enrichment(sites, flank=1)
    assert dropped.loc[dropped["base"] == "N", "log2_enrichment"].isna().all()
    pd.testing.assert_frame_equal(dropped, offset_enrichment(_planted(), flank=1))
    kept = offset_enrichment(sites, flank=1, drop_ambiguous=False)
    assert kept.loc[(kept["base"] == "N") & (kept["offset"] == -1), "log2_enrichment"].notna().all()
    assert "NCT" not in set(kmer_rates(sites, flank=1, k=3)["kmer"])
    assert "NCT" in set(kmer_rates(sites, flank=1, k=3, drop_ambiguous=False)["kmer"])
    assert "C" in set(kmer_rates(sites, flank=1, k=1)["kmer"])  # the centre alone is unambiguous


def test_kmer_relative_rates():
    rates = kmer_rates(_planted(rate_with_t=0.8, rate_without=0.1), flank=1, k=3)
    overall = rates["modified"].sum() / rates["observed"].sum()
    assert np.allclose(rates["overall_rate"], overall)
    act = rates.set_index("kmer").loc["ACT"]
    assert act["log2_relative_rate"] == pytest.approx(np.log2(0.8 / overall))
    assert act["log2_relative_low"] < act["log2_relative_rate"] < act["log2_relative_high"]
    # The centre base alone is every site: relative rate 0.
    centre = kmer_rates(_planted(), flank=1, k=1)
    assert centre["log2_relative_rate"].iloc[0] == pytest.approx(0.0)


def test_context_index_codes_and_cpg():
    from smftools.analysis.compute.site_context_bias import (
        NOT_A_CONTEXT,
        context_index,
        context_kmers,
    )

    kmers = context_kmers(3)
    assert len(kmers) == 16 and all(kmer[1] == "C" for kmer in kmers)
    #            0123456789
    sequence = "AACGGCAAAC"
    codes, cpg = context_index(sequence, "top", range(10), k=3)
    assert kmers[codes[2]] == "ACG" and cpg[2]  # C at 2, G at +1: CpG
    assert kmers[codes[5]] == "GCA" and not cpg[5]
    assert codes[0] == NOT_A_CONTEXT  # not a C
    assert codes[9] == NOT_A_CONTEXT  # C at the end: window touches N
    # Bottom strand: a forward G is a C on the modified strand.
    codes, cpg = context_index(sequence, "bottom", [3, 4], k=3)
    assert kmers[codes[0]] == "CCG" and cpg[0]  # forward 2-4 "CGG", reverse complement
    assert kmers[codes[1]] == "GCC" and not cpg[1]
    with pytest.raises(ValueError):
        context_kmers(2)


def test_context_codes_match_site_contexts():
    from smftools.analysis.compute.site_context_bias import context_index, context_kmers

    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACGT"), 200))
    for strand in ("top", "bottom"):
        positions = np.arange(1, 199)
        codes, _ = context_index(sequence, strand, positions, k=3)
        sites = _sites([("g", f"ref_{strand}", int(p), 1, 0) for p in positions])
        contexts = site_contexts(sites, {"ref": sequence}, flank=1)["context"].to_numpy()
        for code, context in zip(codes, contexts, strict=True):
            if context[1] == "C":
                assert context_kmers(3)[code] == context
            else:
                assert code == -1


def test_weight_table_round_trip(tmp_path):
    from smftools.analysis.compute.site_context_bias import (
        context_kmers,
        read_weight_table,
        weight_table,
        weights_for,
        write_weight_table,
    )

    rates = kmer_rates(_planted(rate_with_t=0.8, rate_without=0.2), flank=1, k=3)
    table = weight_table(rates, k=3, source="naked_dna")
    assert set(table["kmer"]) <= set(context_kmers(3)) and (table["source"] == "naked_dna").all()
    act = table.set_index("kmer").loc["ACT", "weight"]
    overall = rates["modified"].sum() / rates["observed"].sum()
    assert act == pytest.approx((80 + 0.5) / (100 + 1) / overall)
    never = weight_table(kmer_rates(_planted(1.0, 0.0), flank=1, k=3), k=3, source="cells")
    assert (never["weight"] > 0).all()  # never modified: unlikely, not impossible
    for suffix in ("parquet", "csv"):
        path = tmp_path / f"w.{suffix}"
        write_weight_table(table, path)
        pd.testing.assert_frame_equal(read_weight_table(path), table, check_dtype=False)
    weights = weights_for(table, "g", 3)
    assert weights.shape == (16,) and weights[context_kmers(3).index("ACT")] == pytest.approx(act)
    assert weights[context_kmers(3).index("TCG")] == 1.0  # absent: neutral
    with pytest.raises(KeyError):
        weights_for(table, "other", 3)
    with pytest.raises(ValueError, match="source"):
        weight_table(rates, k=3, source="guess")
    with pytest.raises(ValueError, match="positive"):
        write_weight_table(table.assign(weight=0.0), tmp_path / "bad.csv")
