"""Motif files and PWM scanning (`MOT-01`).

Motifs come from a user-supplied file -- nothing here assumes one. The MEME
minimal format is read exactly (Biopython's parser rounds letter
probabilities to integer counts). Scanning is vectorized log-odds scoring of
every window on both strands, with p-values from the exact background
distribution of the integer-scaled score (FIMO's method).

Coordinates are 0-based, half-open, on the forward strand of the scanned
sequence. A minus-strand hit is the reverse-complement matrix matching the
forward window ``[start, end)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

BASES = "ACGT"
BACKGROUNDS = ("uniform", "motif", "sequence")
DEFAULT_PSEUDOCOUNT = 0.1  # FIMO's --motif-pseudo
SCORE_BINS = 10_000  # integer resolution of the score distribution
HIT_COLUMNS = [
    "motif_id",
    "motif_alt_id",
    "motif_name",
    "family",
    "reference",
    "start",
    "end",
    "motif_strand",
    "score",
    "pvalue",
    "matched_sequence",
]


@dataclass(frozen=True)
class Motif:
    """One position-probability matrix (width x A, C, G, T)."""

    motif_id: str
    alt_id: str
    probabilities: np.ndarray
    nsites: float = 20.0

    @property
    def width(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def name(self) -> str:
        return parse_motif_id(self.motif_id)[0]

    @property
    def family(self) -> str:
        return parse_motif_id(self.motif_id)[1]


def parse_motif_id(motif_id: str) -> tuple[str, str]:
    """``(name, family)`` from IDs such as ``AC0395:SOX:Sox`` (archetype style);
    otherwise the ID itself and no family."""
    parts = str(motif_id).split(":")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return str(motif_id), ""


@dataclass(frozen=True)
class MotifFile:
    path: Path
    sha256: str
    motifs: tuple[Motif, ...]
    background: np.ndarray  # the file's own background (uniform when absent)


def read_motifs(path: str | Path) -> MotifFile:
    """Read a MEME (minimal) motif file with exact letter probabilities."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"motif file not found: {path}")
    text = path.read_text()
    motifs: list[Motif] = []
    background = np.full(4, 0.25)
    lines = text.splitlines()
    index = 0
    motif_id = alt_id = None
    while index < len(lines):
        line = lines[index].strip()
        parts = line.split()
        if line.startswith("ALPHABET") and "ACGT" not in line.replace(" ", ""):
            raise ValueError(f"{path}: only the DNA alphabet (ACGT) is supported")
        if line.startswith("Background letter frequencies"):
            values = lines[index + 1].split()
            freqs = dict(zip(values[0::2], values[1::2]))
            if set(BASES) <= set(freqs):
                background = np.array([float(freqs[b]) for b in BASES])
                background = background / background.sum()
        elif line.startswith("MOTIF"):
            motif_id = parts[1] if len(parts) > 1 else f"motif{len(motifs) + 1}"
            alt_id = parts[2] if len(parts) > 2 else ""
        elif line.startswith("letter-probability matrix"):
            if motif_id is None:
                raise ValueError(f"{path}: a matrix before any MOTIF line")
            settings = dict(zip(parts[2::2], parts[3::2]))
            width = int(settings["w="])
            nsites = float(settings.get("nsites=", 20) or 20)
            rows = []
            while len(rows) < width:
                index += 1
                if index >= len(lines):
                    raise ValueError(f"{path}: motif {motif_id} ends early")
                values = lines[index].split()
                if values:
                    rows.append([float(value) for value in values[:4]])
            matrix = np.asarray(rows, dtype=float)
            if matrix.shape != (width, 4) or np.any(matrix < 0):
                raise ValueError(f"{path}: motif {motif_id} has a malformed matrix")
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            motifs.append(Motif(motif_id, alt_id or "", matrix, nsites))
            motif_id = alt_id = None
        index += 1
    if not motifs:
        raise ValueError(f"{path}: no motifs found (MEME minimal format expected)")
    if len({motif.motif_id for motif in motifs}) != len(motifs):
        raise ValueError(f"{path}: motif IDs are not unique")
    sha = hashlib.sha256(text.encode()).hexdigest()
    return MotifFile(path, sha, tuple(motifs), background)


def resolve_background(
    kind: str, motif_file: MotifFile, sequences: Mapping[str, str]
) -> np.ndarray:
    """Background base frequencies: ``uniform``, the motif file's, or the
    scanned sequences' (ACGT only)."""
    if kind == "uniform":
        return np.full(4, 0.25)
    if kind == "motif":
        return motif_file.background.copy()
    if kind == "sequence":
        joined = "".join(sequences.values()).upper()
        counts = np.array([joined.count(b) for b in BASES], dtype=float)
        if counts.sum() == 0:
            raise ValueError("no ACGT bases to estimate a background from")
        return (counts + 1) / (counts.sum() + 4)
    raise ValueError(f"background must be one of {BACKGROUNDS}")


def log_odds(
    motif: Motif, background: np.ndarray, pseudocount: float = DEFAULT_PSEUDOCOUNT
) -> np.ndarray:
    """log2 odds per position and base; the pseudocount is spread by background
    over the motif's ``nsites`` counts (as FIMO)."""
    probabilities = (motif.probabilities * motif.nsites + pseudocount * background) / (
        motif.nsites + pseudocount
    )
    return np.log2(probabilities / background)


@dataclass(frozen=True)
class ScoreModel:
    """A matrix scaled to integers, and P(score >= s) under the background."""

    matrix: np.ndarray  # width x 4, log2 odds
    integer: np.ndarray  # width x 4, scaled and shifted to >= 0
    scale: float
    offset: float  # sum of per-row minima of ``matrix``
    tail: np.ndarray  # tail[k] = P(integer score >= k)

    def integer_scores(self, rows: np.ndarray) -> np.ndarray:
        return np.round((rows - self.offset) * self.scale).astype(np.int64)

    def pvalues(self, integer_scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(integer_scores, 0, len(self.tail) - 1)
        return self.tail[clipped]

    def threshold(self, pvalue: float) -> int:
        """The smallest integer score with P <= ``pvalue`` (``len(tail)`` if none)."""
        passing = np.flatnonzero(self.tail <= pvalue)
        return int(passing[0]) if passing.size else len(self.tail)


def score_model(matrix: np.ndarray, background: np.ndarray, bins: int = SCORE_BINS) -> ScoreModel:
    """Integer-scale ``matrix`` and compute its background score distribution
    by dynamic programming over positions."""
    minima = matrix.min(axis=1)
    span = float((matrix.max(axis=1) - minima).sum())
    scale = (bins - 1) / span if span > 0 else 1.0
    integer = np.round((matrix - minima[:, None]) * scale).astype(np.int64)
    distribution = np.zeros(int(integer.max(axis=1).sum()) + 1)
    distribution[0] = 1.0
    for row in integer:
        updated = np.zeros_like(distribution)
        for base in range(4):
            shift = int(row[base])
            updated[shift:] += background[base] * distribution[: len(distribution) - shift]
        distribution = updated
    tail = np.cumsum(distribution[::-1])[::-1]
    return ScoreModel(matrix, integer, scale, float(minima.sum()), tail)


def encode(sequence: str) -> np.ndarray:
    """Base codes 0-3 (A, C, G, T; case ignored), -1 for anything else."""
    lookup = np.full(256, -1, dtype=np.int8)
    for code, base in enumerate(BASES):
        lookup[ord(base)] = code
        lookup[ord(base.lower())] = code
    return lookup[np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)]


def _window_scores(codes: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per window start: the summed ``table[k, base]`` and whether every base is ACGT."""
    width = table.shape[0]
    n = codes.size - width + 1
    if n <= 0:
        return np.zeros(0, dtype=table.dtype), np.zeros(0, dtype=bool)
    valid_base = codes >= 0
    safe = np.where(valid_base, codes, 0)
    total = np.zeros(n, dtype=table.dtype)
    for k in range(width):
        total += table[k][safe[k : k + n]]
    # A window is valid when it holds no non-ACGT base.
    bad = np.concatenate([[0], np.cumsum(~valid_base)])
    valid = (bad[width:] - bad[:n]) == 0
    return total, valid


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgtNn", "TGCAtgcaNn"))[::-1]


def strand_models(
    motif: Motif, background: np.ndarray, pseudocount: float = DEFAULT_PSEUDOCOUNT
) -> dict[str, ScoreModel]:
    """Score models of the motif (``+``) and its reverse complement (``-``).

    Each has its own background distribution: with unequal A/T and C/G
    frequencies the reverse complement scores differently.
    """
    matrix = log_odds(motif, background, pseudocount)
    return {
        "+": score_model(matrix, background),
        "-": score_model(np.ascontiguousarray(matrix[::-1, ::-1]), background),
    }


def scan_sequence(
    sequence: str,
    motif: Motif,
    models: Mapping[str, ScoreModel],
    *,
    max_pvalue: float,
    reference: str = "",
) -> pd.DataFrame:
    """Hits of one motif on both strands of one sequence at p <= ``max_pvalue``."""
    codes = encode(sequence)
    frames = []
    for strand, model in models.items():
        integer_scores, valid = _window_scores(codes, model.integer)
        keep = np.flatnonzero(valid & (integer_scores >= model.threshold(max_pvalue)))
        if not keep.size:
            continue
        scores, _ = _window_scores(codes, model.matrix)
        starts = keep
        words = [sequence[s : s + motif.width] for s in starts]
        if strand == "-":
            words = [reverse_complement(word) for word in words]
        frames.append(
            pd.DataFrame(
                {
                    "motif_id": motif.motif_id,
                    "motif_alt_id": motif.alt_id,
                    "motif_name": motif.name,
                    "family": motif.family,
                    "reference": reference,
                    "start": starts.astype(np.int64),
                    "end": (starts + motif.width).astype(np.int64),
                    "motif_strand": strand,
                    "score": scores[keep],
                    "pvalue": model.pvalues(integer_scores[keep]),
                    "matched_sequence": [word.upper() for word in words],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=HIT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def scan(
    sequences: Mapping[str, str],
    motif_file: MotifFile,
    *,
    max_pvalue: float = 1e-4,
    background: str | Sequence[float] = "uniform",
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    motif_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Every motif (or ``motif_ids``) on both strands of every sequence.

    ``sequences`` maps a reference name to its forward sequence. Windows with
    a non-ACGT base are skipped; case is ignored.
    """
    if not 0 < max_pvalue <= 1:
        raise ValueError("max_pvalue must be in (0, 1]")
    if isinstance(background, str):
        frequencies = resolve_background(background, motif_file, sequences)
    else:
        frequencies = np.asarray(background, dtype=float)
        frequencies = frequencies / frequencies.sum()
    wanted = set(motif_ids) if motif_ids is not None else None
    if wanted is not None:
        unknown = wanted - {motif.motif_id for motif in motif_file.motifs}
        if unknown:
            raise KeyError(f"motifs not in {motif_file.path.name}: {sorted(unknown)[:5]}")
    frames = []
    for motif in motif_file.motifs:
        if wanted is not None and motif.motif_id not in wanted:
            continue
        models = strand_models(motif, frequencies, pseudocount)
        for reference, sequence in sequences.items():
            frames.append(
                scan_sequence(sequence, motif, models, max_pvalue=max_pvalue, reference=reference)
            )
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame(columns=HIT_COLUMNS)
    hits = pd.concat(frames, ignore_index=True)
    return hits.sort_values(["reference", "start", "motif_id", "motif_strand"]).reset_index(
        drop=True
    )
