"""Sequence-context QC for finished stages, without re-running them (`SCQ-03`).

Runs made before `SCQ` have no ``context_qc`` outputs. This writes them from a
stage's current generation: the same tables the stage writes itself
(`SCQ-01` preprocess, `SCQ-02` HMM), under ``<generation>/context_qc/``.

A published generation is not modified otherwise. Its ``plots/`` tree and
sidecar manifest are covered by the generation's checksums (a preprocess
generation is re-validated whenever it is reused), so backfilled figures go
to ``<generation>/context_qc/plots/context_qc/`` with their own plot catalog
rather than into ``plots/``.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

logger = logging.getLogger(__name__)

STAGES = ("preprocess", "hmm")
STAGE_DIRS = {"preprocess": "preprocess_adata_outputs", "hmm": "hmm_adata_outputs"}


def experiment_config(experiment_dir: str | Path, config_path: str | Path | None = None):
    """The experiment's resolved config: ``config_path`` if given, else the one
    recorded in ``experiment_manifest.json``."""
    if config_path is not None:
        from smftools.cli.helpers import load_experiment_config

        return load_experiment_config(str(config_path))
    manifest = Path(experiment_dir) / "experiment_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found; pass a config file")
    config = json.loads(manifest.read_text()).get("config")
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{manifest} records no config; pass a config file")
    return SimpleNamespace(**config)


def current_generation(experiment_dir: str | Path, stage: str) -> Path | None:
    from smftools.informatics.generation import resolve_current_generation

    current = resolve_current_generation(Path(experiment_dir) / STAGE_DIRS[stage])
    return None if current is None else Path(current[0])


def _layout(generation: Path, stage: str):
    from smftools.cli.stage_artifacts import prepare_stage_plot_layout

    return prepare_stage_plot_layout(
        generation / "context_qc", stage=stage, categories=("context_qc",)
    )


def _hmm_task_partial(spine_path: str, generation: str, record: dict, cfg) -> str:
    """Re-materialize one HMM task (its reads, its core) and tally it."""
    import anndata as ad
    import numpy as np
    import zarr
    from threadpoolctl import threadpool_limits

    from smftools.informatics.partition_read import materialize

    from .hmm_context_qc import tally_hmm_task, write_task_partial
    from .partitioned_hmm import _configured_model_specs

    generation = Path(generation)
    store = zarr.open_group(str(generation / record["group_path"]), mode="r")
    read_ids = ad.io.read_elem(store["obs"]).index.astype(str).tolist()
    if not read_ids:
        return ""
    with threadpool_limits(limits=1):
        adata = materialize(
            spine_path,
            references=record["reference"],
            read_ids=read_ids,
            start=int(record["core_start"]),
            end=int(record["core_end"]),
        )
        tallies = tally_hmm_task(
            adata,
            record["reference"],
            record["barcode"],
            np.ones(adata.n_vars, dtype=bool),
            cfg,
            _configured_model_specs(cfg),
        )
    return write_task_partial(generation, record["task_id"], tallies)


def backfill_context_qc(
    experiment_dir: str | Path,
    stage: str,
    cfg,
    *,
    workers: int = 1,
    refresh: bool = False,
    figures: bool = True,
) -> dict:
    """Write one stage's context-QC outputs from its current generation.

    Returns a status record: ``written``, ``exists`` (outputs present and no
    ``refresh``), ``no_generation`` or ``empty`` (nothing to tally).
    """
    from smftools.informatics.partition_read import load_spine

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    generation = current_generation(experiment_dir, stage)
    if generation is None:
        return {"stage": stage, "status": "no_generation"}
    target = generation / "context_qc"
    record = {"stage": stage, "generation": str(generation)}
    if (target / "run.json").exists() and not refresh:
        return {**record, "status": "exists"}
    shutil.rmtree(target, ignore_errors=True)
    # Asked for explicitly: the setting that gates it inside the stage is moot.
    cfg = copy.copy(cfg)
    cfg.stage_context_qc = True
    spine_path = generation / "spine.h5ad"
    spine = load_spine(spine_path, verbose=False)
    layout = _layout(generation, stage) if figures else None
    if stage == "preprocess":
        from smftools.preprocessing.stage_context_qc import write_stage_context_qc

        written = write_stage_context_qc(
            generation,
            generation / "task_catalog.parquet",
            spine.obs,
            spine.uns,
            cfg,
            plot_layout=layout,
            workers=workers,
        )
    else:
        from .hmm_context_qc import write_hmm_context_qc

        records = pd.read_parquet(generation / "task_catalog.parquet").to_dict("records")
        jobs = [(str(spine_path), str(generation), item, cfg) for item in records]
        if workers > 1 and len(jobs) > 1:
            from smftools.parallel_utils import configure_worker_threads

            with ProcessPoolExecutor(
                max_workers=min(workers, len(jobs)),
                initializer=configure_worker_threads,
                initargs=(1,),
            ) as pool:
                partials = list(pool.map(_hmm_task_partial, *zip(*jobs)))
        else:
            partials = [_hmm_task_partial(*job) for job in jobs]
        for item, partial in zip(records, partials, strict=True):
            item["context_qc_partial"] = partial
        written = write_hmm_context_qc(generation, records, spine.uns, cfg, layout=layout)
    if written is None:
        if layout is not None:
            shutil.rmtree(target, ignore_errors=True)
        return {**record, "status": "empty"}
    run = json.loads((written / "run.json").read_text())
    run["backfilled"] = True
    (written / "run.json").write_text(json.dumps(run, indent=2))
    return {**record, "status": "written", "figures": run.get("figures", 0)}
