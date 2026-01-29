from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AlignmentMismatch:
    """Record a mismatch or gap between two aligned sequences."""

    event: Literal["substitution", "insertion", "deletion"]
    seq1_pos: int | None
    seq1_base: str | None
    seq2_pos: int | None
    seq2_base: str | None


def align_sequences_with_mismatches(
    seq1: str,
    seq2: str,
    match_score: int = 1,
    mismatch_score: int = -1,
    gap_score: int = -1,
    ignore_n: bool = True,
) -> tuple[str, str, list[AlignmentMismatch]]:
    """Globally align two sequences and return mismatch positions/bases.

    The alignment uses a simple Needleman-Wunsch dynamic programming approach
    with configurable scores. Mismatch reporting is based on the aligned
    sequences and returns 0-based coordinates in each sequence. Gap events
    are represented with a ``None`` coordinate for the gapped sequence.

    Args:
        seq1: First sequence (treated as reference for positions).
        seq2: Second sequence.
        match_score: Score for matching bases.
        mismatch_score: Score for mismatching bases.
        gap_score: Score for introducing a gap.
        ignore_n: Whether to ignore mismatches involving the base ``N``.

    Returns:
        Tuple of (aligned_seq1, aligned_seq2, mismatches).
    """
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    n, m = len(seq1), len(seq2)

    scores = [[0] * (m + 1) for _ in range(n + 1)]
    traceback = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + gap_score
        traceback[i][0] = "up"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + gap_score
        traceback[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_score = scores[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score
            )
            up_score = scores[i - 1][j] + gap_score
            left_score = scores[i][j - 1] + gap_score

            best_score = max(diag_score, up_score, left_score)
            scores[i][j] = best_score
            if best_score == diag_score:
                traceback[i][j] = "diag"
            elif best_score == up_score:
                traceback[i][j] = "up"
            else:
                traceback[i][j] = "left"

    aligned1: list[str] = []
    aligned2: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = traceback[i][j]
        if direction == "diag":
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif direction == "up":
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned_seq1 = "".join(reversed(aligned1))
    aligned_seq2 = "".join(reversed(aligned2))

    mismatches: list[AlignmentMismatch] = []
    seq1_index = 0
    seq2_index = 0
    for base1, base2 in zip(aligned_seq1, aligned_seq2, strict=True):
        if base1 == "-":
            if not (ignore_n and base2 == "N"):
                mismatches.append(
                    AlignmentMismatch(
                        event="insertion",
                        seq1_pos=None,
                        seq1_base=None,
                        seq2_pos=seq2_index,
                        seq2_base=base2,
                    )
                )
            seq2_index += 1
            continue
        if base2 == "-":
            if not (ignore_n and base1 == "N"):
                mismatches.append(
                    AlignmentMismatch(
                        event="deletion",
                        seq1_pos=seq1_index,
                        seq1_base=base1,
                        seq2_pos=None,
                        seq2_base=None,
                    )
                )
            seq1_index += 1
            continue
        if base1 != base2 and not (ignore_n and "N" in (base1, base2)):
            mismatches.append(
                AlignmentMismatch(
                    event="substitution",
                    seq1_pos=seq1_index,
                    seq1_base=base1,
                    seq2_pos=seq2_index,
                    seq2_base=base2,
                )
            )
        seq1_index += 1
        seq2_index += 1

    return aligned_seq1, aligned_seq2, mismatches
