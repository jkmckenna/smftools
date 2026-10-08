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

STAGES = ("preprocess", "hmm", "hmm-fractions")
FRACTIONS_SUBDIR = "molecule_fractions"
STAGE_DIRS = {
    "preprocess": "preprocess_adata_outputs",
    "hmm": "hmm_adata_outputs",
    "hmm-fractions": "hmm_adata_outputs",
}


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


def _hmm_task_molecules(spine_path: str, generation: str, record: dict, cfg) -> str:
    """Re-materialize one HMM task and compute its reads' fractions (`HCE-10`)."""
    import hashlib

    import anndata as ad
    import numpy as np
    import zarr
    from threadpoolctl import threadpool_limits

    from smftools.informatics.partition_read import materialize

    from .partitioned_hmm import (
        FRACTION_LAYER_SUFFIXES,
        _configured_model_specs,
        molecule_fractions,
        molecule_site_fractions,
    )

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
        core = np.ones(adata.n_vars, dtype=bool)
        layers = [name for name in adata.layers if str(name).endswith(FRACTION_LAYER_SUFFIXES)]
        columns = {
            **molecule_fractions(adata, layers, core),
            **molecule_site_fractions(
                adata, record["reference"], core, cfg, _configured_model_specs(cfg)
            ),
        }
    frame = pd.DataFrame(columns, index=adata.obs_names.astype(str))
    frame.index.name = "read_id"
    frame = frame.assign(
        task_id=str(record["task_id"]),
        barcode=str(record["barcode"]),
        reference=str(record["reference"]),
        core_start=int(record["core_start"]),
        core_end=int(record["core_end"]),
    )
    name = hashlib.sha1(str(record["task_id"]).encode()).hexdigest()[:20]
    path = generation / FRACTIONS_SUBDIR / "partials" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path.relative_to(generation).as_posix()


def _run_tasks(function, jobs, workers: int) -> list:
    if workers > 1 and len(jobs) > 1:
        from smftools.parallel_utils import configure_worker_threads

        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            initializer=configure_worker_threads,
            initargs=(1,),
        ) as pool:
            return list(pool.map(function, *zip(*jobs)))
    return [function(*job) for job in jobs]


def backfill_molecule_fractions(
    experiment_dir: str | Path,
    cfg,
    *,
    workers: int = 1,
    refresh: bool = False,
    figures: bool = True,
) -> dict:
    """Per-read HMM and raw fractions of a finished HMM generation (`HCE-10`).

    What the HMM stage now stores per read (`HCE-06`, `HCE-08`, `HCE-09`):
    ``<layer>_fraction`` over the read span, ``<layer>_site_fraction`` at the
    model's observed sites and ``<model>_site_modified_fraction``. Written as
    one table, ``<generation>/molecule_fractions/molecule_fractions.parquet``,
    with the per-molecule violin and HMM-vs-raw scatter figures beside it
    (``plots/features/``). The generation's own read table is not changed.
    """
    from smftools.cli.stage_artifacts import prepare_stage_plot_layout

    from .partitioned_hmm import (
        _configured_model_specs,
        _plot_hmm_vs_raw_scatter,
        _plot_molecule_fractions,
    )

    generation = current_generation(experiment_dir, "hmm-fractions")
    if generation is None:
        return {"stage": "hmm-fractions", "status": "no_generation"}
    target = generation / FRACTIONS_SUBDIR
    record = {"stage": "hmm-fractions", "generation": str(generation)}
    if (target / "run.json").exists() and not refresh:
        return {**record, "status": "exists"}
    shutil.rmtree(target, ignore_errors=True)
    records = pd.read_parquet(generation / "task_catalog.parquet").to_dict("records")
    spine_path = str(generation / "spine.h5ad")
    partials = _run_tasks(
        _hmm_task_molecules,
        [(spine_path, str(generation), item, cfg) for item in records],
        workers,
    )
    frames = [pd.read_parquet(generation / path) for path in partials if path]
    if not frames:
        shutil.rmtree(target, ignore_errors=True)
        return {**record, "status": "empty"}
    table = pd.concat(frames)
    table.to_parquet(target / "molecule_fractions.parquet")
    shutil.rmtree(target / "partials", ignore_errors=True)
    written = []
    if figures:
        layout = prepare_stage_plot_layout(target, stage="hmm", categories=("features",))
        by_task = {task_id: frame for task_id, frame in table.groupby("task_id", sort=False)}
        group_task = {str(item["group_path"]): str(item["task_id"]) for item in records}

        def reader(path, columns):
            task_id = group_task.get(Path(path).relative_to(generation).as_posix())
            frame = by_task.get(task_id, pd.DataFrame())
            return frame[[column for column in columns if column in frame]]

        specs = _configured_model_specs(cfg)
        _plot_molecule_fractions(records, generation, layout, specs=specs, obs_reader=reader)
        _plot_hmm_vs_raw_scatter(records, generation, layout, specs=specs, obs_reader=reader)
        written = sorted(p.name for p in layout.categories["features"].glob("*.png"))
    (target / "run.json").write_text(
        json.dumps(
            {
                "backfilled": True,
                "reads": int(len(table)),
                "columns": [c for c in table.columns if c.endswith("fraction")],
                "figures": len(written),
            },
            indent=2,
        )
    )
    return {**record, "status": "written", "figures": len(written)}


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
    if stage == "hmm-fractions":
        return backfill_molecule_fractions(
            experiment_dir, cfg, workers=workers, refresh=refresh, figures=figures
        )
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
        partials = _run_tasks(_hmm_task_partial, jobs, workers)
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
