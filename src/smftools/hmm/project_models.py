"""Frozen project HMMs (`HPS-06`): decode an experiment with stored models.

A project model is fitted once (``smftools.project.hmm_models.fit_project_model``)
and stored with its checkpoints and a manifest. Applying it decodes another
experiment's spatial spine with those models unchanged: no fit, no adaptation.
This module is the stage-side half: which stored model decodes which task, and
the catalogs that record it.

Reference mapping. A model is stored per *reference group* (for example ``top``:
``6B6_top`` + ``6BALB_cJ_top``). A task's reference uses the group that lists it
or, failing that, the group whose name is the reference's ``_<suffix>`` (so a
``6B6_enh_del_top`` read is decoded by the ``top`` model). The decode itself
recomputes sequence contexts for the task's own reference (``_annotate_task``),
so a model fitted on one reference decodes another of the same locus.

Samples. ``barcodes`` limits decoding to the samples the model is for (one
enzyme); other tasks are left out, and the stage keeps only decoded reads on its
spine, so a read of another enzyme is absent rather than read as all-zero.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

PROJECT_FIT_KIND = "PROJECT"


@dataclass(frozen=True)
class FrozenModels:
    """Stored project models to decode with, carried on a config as
    ``_hmm_frozen_models``.

    ``artifacts``: reference group -> model label -> portable artifact record
    (``checkpoint`` / ``metadata`` relative to ``models_dir``). ``reference_groups``:
    group -> the references it was fitted on. ``barcodes``: the samples to decode
    (None: every sample). ``exclude_references``: never decoded.
    """

    name: str
    generation_id: str
    models_dir: Path
    artifacts: Mapping[str, Mapping[str, Mapping[str, Any]]]
    reference_groups: Mapping[str, Sequence[str]]
    barcodes: frozenset[str] | None = None
    exclude_references: frozenset[str] = frozenset()
    training_read_ids: frozenset[str] = field(default_factory=frozenset)

    def group_for(self, reference: str) -> str | None:
        """The reference group whose model decodes ``reference`` (None: no model)."""
        reference = str(reference)
        if reference in self.exclude_references:
            return None
        for group, members in self.reference_groups.items():
            if reference in {str(m) for m in members}:
                return group
        matches = [group for group in self.artifacts if reference.endswith(f"_{group}")]
        if len(matches) > 1:
            raise ValueError(
                f"reference {reference!r} matches several model groups by suffix: {matches}"
            )
        return matches[0] if matches else None


def _copy_artifact_files(frozen: FrozenModels, models_dir: Path) -> None:
    """Copy every stored checkpoint and metadata file into the stage's own
    model store, at the same relative path, so the generation is self-contained."""
    for by_label in frozen.artifacts.values():
        for artifact in by_label.values():
            for key in ("checkpoint", "metadata"):
                relative = Path(str(artifact[key]))
                if relative.parts and relative.parts[0] == models_dir.name:
                    relative = Path(*relative.parts[1:])
                source = Path(frozen.models_dir) / relative
                target = models_dir / relative
                if not source.is_file():
                    raise FileNotFoundError(f"project model {frozen.name}: missing {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)


def frozen_apply_plan(
    tasks: Sequence[Any],
    frozen: FrozenModels,
    models_dir: str | Path,
    fit_catalog_path: str | Path,
    fit_selection_path: str | Path,
) -> tuple[list[Any], dict[str, dict[str, dict[str, Any]]]]:
    """The tasks a project model decodes and each one's model per label.

    Writes the fit catalog (one row per stored model, ``fit_kind`` PROJECT) and
    the fit selection (the model's training reads, so a decode of the training
    experiment marks them as ``hmm_fit_read``).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    _copy_artifact_files(frozen, models_dir)
    kept, assignments, skipped = [], {}, {}
    for task in tasks:
        if frozen.barcodes is not None and str(task.barcode) not in frozen.barcodes:
            skipped["sample"] = skipped.get("sample", 0) + 1
            continue
        group = frozen.group_for(str(task.reference))
        if group is None or group not in frozen.artifacts:
            skipped["reference"] = skipped.get("reference", 0) + 1
            continue
        kept.append(task)
        assignments[str(task.task_id)] = {
            label: dict(artifact) for label, artifact in frozen.artifacts[group].items()
        }
    if not kept:
        raise RuntimeError(
            f"project model {frozen.name} decodes no task here (left out: {skipped or 'none'})"
        )
    if skipped:
        logger.info("Project model %s: tasks left out %s", frozen.name, skipped)
    rows = [
        {
            "fit_id": str(artifact.get("model_id")),
            "fit_kind": PROJECT_FIT_KIND,
            "fit_state": "complete",
            "project_model": frozen.name,
            "project_model_generation": frozen.generation_id,
            "reference_group": group,
            "model_label": label,
            "model_id": str(artifact.get("model_id")),
            "checkpoint_sha256": str(artifact.get("checkpoint_sha256", "")),
            "checkpoint": str(artifact.get("checkpoint")),
        }
        for group, by_label in sorted(frozen.artifacts.items())
        for label, artifact in sorted(by_label.items())
    ]
    pd.DataFrame(rows).to_parquet(fit_catalog_path, index=False)
    pd.DataFrame({"read_id": sorted(frozen.training_read_ids)}, dtype=str).to_parquet(
        fit_selection_path, index=False
    )
    return kept, assignments
