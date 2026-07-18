"""Set store: streamed, partition-native access to a project-level cross-experiment
selection.

A "set" is not a stored artifact -- it is a *query* (a canonical reference plus optional
set-name/modality/experiment filters, resolved against the registry). ``iter_set_parts``
streams that selection one projected experiment-slice at a time, so peak memory scales
with a single member's projected slice, never the whole pool. There is no ``base.h5ad``
and no set-level cache: membership is resolved fresh each call, so a set is never stale.

This replaces the earlier design (see ``dev/project_sample_and_set_stores.md``, "Set
store" + "Proposed redesign") that concatenated every member into one monolithic
``base.h5ad``. That approach held every member plus the pooled result in memory at once
(~56 GB on a real 11-experiment project) and produced >200 GB unwritable output from the
outer-join upcasting ~25 int layers to float at full locus. The two things that made it
tractable -- and that this module bakes in -- are **streaming** (never all members
resident) and **projection** (materialize only the ``layers`` / genomic window a caller
actually needs, not all layers at full locus).

Concatenating the stream into one in-memory AnnData is still available as an explicit,
size-guarded opt-in: ``project.catalog.project_adata``.

``slug``/``sets_root``/``set_label`` remain here -- shared with ``embedding_store`` for
the ``sets/<label>/embeddings/`` directory.
"""

from __future__ import annotations

import re
from pathlib import Path

SETS_OUTPUT_DIRNAME = "sets"


def slug(value: str) -> str:
    """Filesystem-safe slug, shared by every project_outputs/ set directory name."""
    slugged = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return slugged or "x"


def sets_root(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_outputs" / SETS_OUTPUT_DIRNAME


def set_label(set_name: str | None, canonical_reference: str) -> str:
    """The directory name a set's artifacts (e.g. embeddings) live under: the set name
    if given, else the canonical reference -- shared by ``set_store`` and
    ``embedding_store`` so an embedding directory and its set always agree on which set
    they belong to."""
    return slug(set_name) if set_name else slug(canonical_reference)


def resolve_set_members(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name=None,
    modality=None,
    experiments=None,
    stage: str | None = None,
) -> list[dict]:
    """Resolve which ``(experiment, stage, spine_path, reference_strands)`` a set query
    selects, right now -- cheap (registry + in-memory alias table, no matrices)."""
    from .catalog import ProjectCatalog
    from .catalog import resolve_set_members as _resolve

    catalog = ProjectCatalog.open(project_dir)
    return _resolve(
        catalog,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )


def normalize_part(sub, experiment: str, stage: str):
    """Apply the per-part normalization every cross-experiment consumer needs.

    - stamps ``obs["experiment"]`` and ``uns["project_stage"]``,
    - clears the obs index name (differently-named indices across spines make
      ``ad.concat`` emit a reserved ``"_index"`` column),
    - strips ``var`` to the position axis only (a legacy HMM stage carries ~1200
      per-reference var columns whose outer-union overflows HDF5's per-attribute
      limit -- see the design doc).
    """
    sub.obs["experiment"] = experiment
    sub.uns["project_stage"] = stage
    sub.obs.index.name = None
    sub.var = sub.var.iloc[:, :0]
    return sub


def iter_set_parts(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    layers=None,
    start: int | None = None,
    end: int | None = None,
    read_metrics: bool = False,
    lazy: bool = False,
):
    """Stream one projected AnnData slice per member experiment of a set.

    Yields each matching experiment's reads at ``canonical_reference`` (resolved to that
    experiment's own reference-strand name(s)), materialized with the requested
    ``layers``/``start``/``end`` **projection** so no slice bigger than one member's
    projected reads is ever built. Each part is normalized (see :func:`normalize_part`)
    so parts can be concatenated or feature-extracted directly.

    Pass ``layers=[]`` for X only (no layers), ``layers=[names]`` for a subset, or
    ``layers=None`` for all layers (the memory-heavy default -- prefer a subset).
    ``start``/``end`` restrict to a genomic window. Streaming + projection are what keep
    this bounded; see the module docstring.

    Membership is resolved eagerly (so an empty selection raises here, not on first
    iteration); slices are materialized lazily as the generator is consumed.

    Raises:
        ValueError: if no experiment reference matches ``canonical_reference``.
    """
    members = resolve_set_members(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    if not members:
        raise ValueError(
            f"no experiment references match canonical_reference={canonical_reference!r}"
        )

    def _gen():
        from ..informatics.partition_read import materialize

        for member in members:
            sub = materialize(
                member["spine_path"],
                references=member["reference_strands"],
                start=start,
                end=end,
                layers=layers,
                read_metrics=read_metrics,
                lazy=lazy,
            )
            yield normalize_part(sub, member["experiment"], member["stage"])

    return _gen()
