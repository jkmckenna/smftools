"""Motif scans of an experiment's, a project's or a FASTA's references (`MOT-01`).

The motif file is always the caller's: nothing here assumes one. Hits are
written once per (motif file, sequences, settings) and reused while those are
unchanged -- the cache key holds the motif file's and each sequence's content
hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

HITS_FILE = "motif_hits.parquet"
RUN_FILE = "run.json"


def fasta_sequences(path: str | Path) -> dict[str, str]:
    """Forward sequences by record ID from a FASTA file."""
    from Bio import SeqIO

    sequences = {record.id: str(record.seq) for record in SeqIO.parse(str(path), "fasta")}
    if not sequences:
        raise ValueError(f"no sequences in {path}")
    return sequences


def experiment_sequences(experiment_dir: str | Path) -> dict[str, str]:
    """Reference sequences recorded in an experiment's spines (strand suffix removed)."""
    from .site_context_bias import reference_sequences

    experiment_dir = Path(experiment_dir)
    candidates = [experiment_dir / "experiment_spine_outputs" / "spine.h5ad"] + [
        experiment_dir / stage / "spine.h5ad"
        for stage in ("preprocess_adata_outputs", "raw_outputs", "hmm_adata_outputs")
    ]
    spines = [path for path in candidates if path.is_file()]
    if not spines:
        raise FileNotFoundError(f"no spine with reference sequences under {experiment_dir}")
    sequences = reference_sequences(spines)
    if not sequences:
        raise ValueError(f"the spines under {experiment_dir} record no reference sequences")
    return sequences


def project_sequences(project_dir: str | Path, experiments: Iterable[str] | None = None) -> dict:
    """Reference sequences across a project's registered experiments; a name
    recorded with two different sequences is an error."""
    from smftools.project.registry import load_registry

    project_dir = Path(project_dir)
    registry = load_registry(project_dir).get("experiments", {})
    chosen = list(experiments) if experiments else sorted(registry)
    unknown = sorted(set(chosen) - set(registry))
    if unknown:
        raise KeyError(f"unknown experiment(s): {unknown}")
    sequences: dict[str, str] = {}
    for experiment_id in chosen:
        found = experiment_sequences((project_dir / registry[experiment_id]["path"]).resolve())
        for name, sequence in found.items():
            if name in sequences and sequences[name].upper() != sequence.upper():
                raise ValueError(f"reference {name!r} differs between experiments")
            sequences.setdefault(name, sequence)
    return sequences


def _sequence_digest(sequences: dict[str, str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(sequence.upper().encode()).hexdigest()
        for name, sequence in sorted(sequences.items())
    }


def scan_references(
    motifs_path: str | Path,
    sequences: dict[str, str],
    output_dir: str | Path,
    *,
    max_pvalue: float = 1e-4,
    background: str = "uniform",
    pseudocount: float = 0.1,
    motif_ids: Sequence[str] | None = None,
    references: Sequence[str] | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Scan ``sequences`` (or the named ``references``) and write the hit table
    to ``output_dir``; reuse it when the motif file, sequences and settings are
    unchanged. Returns ``(hits, run record)``."""
    from smftools import __version__
    from smftools.analysis.compute.motifs import read_motifs, scan

    if references:
        missing = sorted(set(references) - set(sequences))
        if missing:
            raise KeyError(f"references not found: {missing}; available: {sorted(sequences)}")
        sequences = {name: sequences[name] for name in references}
    motif_file = read_motifs(motifs_path)
    key = {
        "motif_file_sha256": motif_file.sha256,
        "sequences_sha256": _sequence_digest(sequences),
        "max_pvalue": float(max_pvalue),
        "background": background,
        "pseudocount": float(pseudocount),
        "motif_ids": sorted(motif_ids) if motif_ids else None,
        "engine": "builtin",
    }
    output_dir = Path(output_dir)
    run_path = output_dir / RUN_FILE
    if not refresh and run_path.is_file() and (output_dir / HITS_FILE).is_file():
        record = json.loads(run_path.read_text())
        if record.get("key") == key:
            return pd.read_parquet(output_dir / HITS_FILE), {**record, "reused": True}
    hits = scan(
        sequences,
        motif_file,
        max_pvalue=max_pvalue,
        background=background,
        pseudocount=pseudocount,
        motif_ids=motif_ids,
    )
    hits = hits.assign(engine="builtin", motif_file_sha256=motif_file.sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    hits.to_parquet(output_dir / HITS_FILE, index=False)
    record = {
        "key": key,
        "motif_file": str(Path(motifs_path).resolve()),
        "motifs": len(motif_file.motifs) if not motif_ids else len(motif_ids),
        "references": {name: len(sequence) for name, sequence in sorted(sequences.items())},
        "hits": int(len(hits)),
        "smftools_version": __version__,
    }
    run_path.write_text(json.dumps(record, indent=2))
    return hits, {**record, "reused": False}
