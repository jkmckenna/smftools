"""Motif files and the built-in PWM scanner (`MOT-01`)."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from smftools.analysis.compute.motifs import (
    encode,
    log_odds,
    parse_motif_id,
    read_motifs,
    reverse_complement,
    scan,
    score_model,
    strand_models,
)

pytestmark = pytest.mark.unit

MEME = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.3 C 0.2 G 0.2 T 0.3

MOTIF M1:GATA:GATA alt1

letter-probability matrix: alength= 4 w= 4 nsites= 20 E= 0
  0.676859  0.103120  0.104093  0.115927
  0.023893  0.035573  0.009097  0.931437
  0.934550  0.021763  0.019600  0.024087
  0.050000  0.050000  0.850000  0.050000

MOTIF M2

letter-probability matrix: alength= 4 w= 3 nsites= 10 E= 0
  0.1 0.7 0.1 0.1
  0.1 0.1 0.7 0.1
  0.25 0.25 0.25 0.25
"""


@pytest.fixture
def motif_file(tmp_path):
    path = tmp_path / "motifs.meme"
    path.write_text(MEME)
    return read_motifs(path)


def test_parser_keeps_probabilities_exactly(motif_file):
    first, second = motif_file.motifs
    assert first.motif_id == "M1:GATA:GATA" and first.alt_id == "alt1"
    assert first.probabilities[0, 0] == pytest.approx(0.676859 / 1.0, rel=1e-4)
    assert first.width == 4 and second.width == 3 and second.nsites == 10
    np.testing.assert_allclose(motif_file.background, [0.3, 0.2, 0.2, 0.3])
    assert (first.name, first.family) == ("GATA", "GATA")
    assert parse_motif_id("plain") == ("plain", "")


def test_parser_rejects_bad_files(tmp_path):
    empty = tmp_path / "empty.meme"
    empty.write_text("MEME version 4\n")
    with pytest.raises(ValueError, match="no motifs"):
        read_motifs(empty)
    duplicate = tmp_path / "dup.meme"
    duplicate.write_text(MEME.replace("MOTIF M2", "MOTIF M1:GATA:GATA"))
    with pytest.raises(ValueError, match="not unique"):
        read_motifs(duplicate)
    with pytest.raises(FileNotFoundError):
        read_motifs(tmp_path / "missing.meme")


def test_scores_equal_a_hand_computation(motif_file):
    motif = motif_file.motifs[0]
    background = np.full(4, 0.25)
    hits = scan({"r": "CCCAGTAGCCC"}, motif_file, max_pvalue=1.0, motif_ids=[motif.motif_id])
    row = hits[(hits.start == 3) & (hits.motif_strand == "+")].iloc[0]
    matrix = log_odds(motif, background)
    expected = sum(matrix[k, "ACGT".index(base)] for k, base in enumerate("AGTA"))
    assert row.score == pytest.approx(expected)
    assert row.matched_sequence == "AGTA" and row.end - row.start == 4


@pytest.mark.parametrize("background", [np.full(4, 0.25), np.array([0.3, 0.2, 0.2, 0.3])])
def test_pvalues_equal_brute_force_enumeration(motif_file, background):
    motif = motif_file.motifs[0]
    model = score_model(log_odds(motif, background), background)
    words = np.array(list(itertools.product(range(4), repeat=motif.width)))
    weights = np.prod(background[words], axis=1)
    integer = model.integer[np.arange(motif.width), words].sum(axis=1)
    for value in np.unique(integer):
        expected = weights[integer >= value].sum()
        assert model.pvalues(np.array([value]))[0] == pytest.approx(expected, rel=1e-9)


def test_minus_strand_hits_mirror_a_forward_scan_of_the_reverse_complement(motif_file):
    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACGT"), 300))
    forward = scan({"r": sequence}, motif_file, max_pvalue=0.05)
    reverse = scan({"r": reverse_complement(sequence)}, motif_file, max_pvalue=0.05)
    minus = forward[forward.motif_strand == "-"]
    plus_on_rc = reverse[reverse.motif_strand == "+"]
    mirrored = set(
        zip(plus_on_rc.motif_id, len(sequence) - plus_on_rc.end, plus_on_rc.matched_sequence)
    )
    assert set(zip(minus.motif_id, minus.start, minus.matched_sequence)) == mirrored


def test_minus_strand_pvalues_use_the_reverse_complement_distribution(motif_file):
    # A strand-asymmetric background: the reverse complement scores differently.
    background = np.array([0.4, 0.3, 0.2, 0.1])
    motif = motif_file.motifs[0]
    models = strand_models(motif, background)
    assert not np.allclose(models["+"].tail, models["-"].tail[: len(models["+"].tail)])
    minus = models["-"]
    words = np.array(list(itertools.product(range(4), repeat=motif.width)))
    weights = np.prod(background[words], axis=1)
    integer = minus.integer[np.arange(motif.width), words].sum(axis=1)
    for value in np.unique(integer)[::7]:
        assert minus.pvalues(np.array([value]))[0] == pytest.approx(
            weights[integer >= value].sum(), rel=1e-9
        )


def test_windows_with_n_are_skipped_and_case_is_ignored(motif_file):
    upper = scan({"r": "AGTAGTA"}, motif_file, max_pvalue=1.0)
    lower = scan({"r": "agtagta"}, motif_file, max_pvalue=1.0)
    pd.testing.assert_frame_equal(upper, lower)
    with_n = scan({"r": "AGTNAGTA"}, motif_file, max_pvalue=1.0)
    for row in with_n.itertuples():
        assert "N" not in "AGTNAGTA"[row.start : row.end]
    assert encode("ACGTN").tolist() == [0, 1, 2, 3, -1]


def test_threshold_is_monotone(motif_file):
    rng = np.random.default_rng(1)
    sequence = "".join(rng.choice(list("ACGT"), 500))
    counts = [len(scan({"r": sequence}, motif_file, max_pvalue=p)) for p in (1e-3, 1e-2, 1e-1)]
    assert counts == sorted(counts) and counts[-1] > counts[0]
    hits = scan({"r": sequence}, motif_file, max_pvalue=1e-2)
    assert (hits.pvalue <= 1e-2).all()


def test_scan_references_caches_by_content(tmp_path):
    from smftools.tools.motifs import fasta_sequences, scan_references

    meme = tmp_path / "m.meme"
    meme.write_text(MEME)
    fasta = tmp_path / "refs.fa"
    rng = np.random.default_rng(2)
    fasta.write_text(">ref1\n" + "".join(rng.choice(list("ACGT"), 400)) + "\n>ref2\nACGTAGTACGT\n")
    sequences = fasta_sequences(fasta)
    out = tmp_path / "scan"
    hits, record = scan_references(meme, sequences, out, max_pvalue=0.01)
    assert not record["reused"] and (out / "motif_hits.parquet").exists()
    assert set(hits["engine"]) <= {"builtin"} and record["references"] == {"ref1": 400, "ref2": 11}
    _, record = scan_references(meme, sequences, out, max_pvalue=0.01)
    assert record["reused"]
    _, record = scan_references(meme, sequences, out, max_pvalue=0.02)
    assert not record["reused"]  # settings changed
    meme.write_text(MEME.replace("0.850000", "0.849000"))
    _, record = scan_references(meme, sequences, out, max_pvalue=0.02)
    assert not record["reused"]  # motif file changed
    only, _ = scan_references(meme, sequences, tmp_path / "one", references=["ref2"])
    assert set(only["reference"]) <= {"ref2"}
    with pytest.raises(KeyError, match="references not found"):
        scan_references(meme, sequences, tmp_path / "x", references=["nope"])


def test_motifs_scan_command(tmp_path):
    from click.testing import CliRunner

    from smftools.cli_entry import cli

    meme = tmp_path / "m.meme"
    meme.write_text(MEME)
    fasta = tmp_path / "refs.fa"
    fasta.write_text(">ref1\nCCCAGTAGCCCAGTAGCCC\n")
    out = tmp_path / "scan"
    args = ["motifs", "scan", "--motifs", str(meme), "--fasta", str(fasta), "-o", str(out)]
    result = CliRunner().invoke(cli, [*args, "--max-pvalue", "0.05"])
    assert result.exit_code == 0, result.output
    assert "scanned:" in result.output
    record = json.loads((out / "run.json").read_text())
    assert record["key"]["max_pvalue"] == 0.05
    result = CliRunner().invoke(cli, args + ["--experiment-dir", str(tmp_path)])
    assert result.exit_code != 0 and "exactly one" in result.output


# --- MOT-02: the FIMO engine ---------------------------------------------------

FIMO_TEXT = """motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence
M1:GATA:GATA\talt1\tref1\t4\t7\t+\t10.5\t1.2e-05\t\tAGTA
M2\t\tref1\t10\t12\t-\t3.1\t4e-03\t\tcga
"""


def test_fimo_text_parses_to_the_hit_table():
    from smftools.analysis.compute.motifs import HIT_COLUMNS
    from smftools.tools.motifs import parse_fimo_text

    hits = parse_fimo_text(FIMO_TEXT)
    assert list(hits.columns) == HIT_COLUMNS
    first, second = hits.itertuples(index=False)
    assert (first.start, first.end, first.motif_strand) == (
        3,
        7,
        "+",
    )  # 1-based closed -> 0-based half-open
    assert (first.motif_name, first.family, first.motif_alt_id) == ("GATA", "GATA", "alt1")
    assert second.matched_sequence == "CGA" and second.motif_alt_id == ""
    assert parse_fimo_text("motif_id\tmotif_alt_id\n").empty


def test_missing_fimo_is_a_clear_error(tmp_path):
    from smftools.tools.motifs import find_fimo, scan_references

    with pytest.raises(FileNotFoundError, match="--engine builtin"):
        find_fimo(tmp_path / "no_fimo_here")
    meme = tmp_path / "m.meme"
    meme.write_text(MEME)
    with pytest.raises(FileNotFoundError):
        scan_references(meme, {"r": "ACGT"}, tmp_path / "o", engine="fimo", fimo=tmp_path / "nope")
    with pytest.raises(ValueError, match="engine"):
        scan_references(meme, {"r": "ACGT"}, tmp_path / "o", engine="homer")


def test_sequence_background_is_strand_symmetric(motif_file):
    from smftools.analysis.compute.motifs import resolve_background

    frequencies = resolve_background("sequence", motif_file, {"r": "AAAAAAAACG"})
    assert frequencies[0] == pytest.approx(frequencies[3])
    assert frequencies[1] == pytest.approx(frequencies[2])


@pytest.mark.skipif(
    __import__("shutil").which("fimo") is None, reason="FIMO (MEME suite) not installed"
)
@pytest.mark.parametrize("background", ["uniform", "motif", "sequence"])
def test_fimo_and_builtin_engines_agree(tmp_path, background):
    from smftools.tools.motifs import scan_references

    meme = tmp_path / "m.meme"
    meme.write_text(MEME)
    rng = np.random.default_rng(3)
    sequences = {"r1": "".join(rng.choice(list("ACGT"), 3000)), "r2": "ACGTAGTANAGTA"}
    kwargs = dict(max_pvalue=0.01, background=background)
    builtin, _ = scan_references(meme, sequences, tmp_path / "b", **kwargs)
    fimo, record = scan_references(meme, sequences, tmp_path / "f", engine="fimo", **kwargs)
    assert record["key"]["engine"] == "fimo" and record["key"]["fimo_version"]
    keys = ["motif_id", "reference", "start", "motif_strand"]
    merged = builtin.merge(fimo, on=keys, how="outer", suffixes=("_b", "_f"), indicator=True)
    shared = merged[merged["_merge"] == "both"]
    assert len(shared) >= 0.97 * max(len(builtin), len(fimo))
    assert np.abs(shared.score_b - shared.score_f).max() < 0.1
    assert (shared.matched_sequence_b == shared.matched_sequence_f).all()
    # Hits found by only one engine sit at the threshold.
    edge = merged[merged["_merge"] != "both"]
    assert (edge[["pvalue_b", "pvalue_f"]].max(axis=1) > 0.005).all()
