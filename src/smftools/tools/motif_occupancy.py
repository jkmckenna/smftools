"""Per-molecule motif occupancy for a plan dataset (`MOT-04`).

The plan's dataset supplies the molecules and channels; the caller says which
channels play which role -- class channels per state (``tf_bound``,
``medium_bound``, ``nucleosome``, ``accessible``; a state may combine several
channels, e.g. putative-nucleosome and large-bound layers) and one ``sites``
channel of raw calls (observed sites decide whether a read is informative at
a motif). Batches stream over worker processes by whole blocks (as `RPG`);
states come back in the dataset's molecule order.

Outputs: ``states.npz`` (reads x instances, codes into ``STATES``, with the
molecule UIDs), ``instances.parquet`` (the motif instances scored),
``reads.parquet`` (molecule identity and every grouping), ``occupancy.parquet``
(per grouping, group and instance: counts and fractions per state).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from smftools.analysis.compute.motif_occupancy import (
    CLASS_STATES,
    STATES,
    OccupancyRules,
    classify,
    group_occupancy,
    resolve_instances,
)

STATES_FILE = "states.npz"
INSTANCES_FILE = "instances.parquet"
READS_FILE = "reads.parquet"
OCCUPANCY_FILE = "occupancy.parquet"
KEY_FILE = "occupancy_key.json"
ALL = "all"


def _groupings(group_by) -> list[str]:
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(dict.fromkeys(str(column) for column in group_by))


def _shard(document, dataset_name, scope, groupings, roles, sites, hits, rules, worker, workers):
    """One worker's blocks; plain data back (bound datasets do not pickle)."""
    from smftools.machine_learning.orchestration import bind_ml_dataset
    from smftools.machine_learning.plan import parse_ml_plan

    bound = bind_ml_dataset(parse_ml_plan(document), dataset_name, group_by=groupings, **scope)
    plan = bound.dataset.plan
    declared = [item.name for item in plan.dataset.input_schema.channels]
    coordinates = np.asarray(plan.coordinates)
    instances, table = resolve_instances(hits, coordinates, rules.flank)
    site_index = declared.index(sites)
    role_index = {state: [declared.index(c) for c in channels] for state, channels in roles.items()}
    batches = (
        bound.iter_batches(worker_id=worker, num_workers=workers)
        if workers > 1
        else bound.iter_batches()
    )
    uids, blocks = [], []
    for batch in batches:
        classes = {}
        for state, index in role_index.items():
            observed = batch.observed_mask[:, :, index]
            values = np.where(observed, np.nan_to_num(batch.values[:, :, index], nan=0.0), 0.0)
            classes[state] = (
                (values > 0).any(axis=2).astype(np.float32),
                observed.any(axis=2),
            )
        blocks.append(classify(classes, batch.observed_mask[:, :, site_index], instances, rules))
        uids.extend(batch.molecule_uids)
    states = (
        np.concatenate(blocks, axis=0) if blocks else np.zeros((0, len(instances)), dtype=np.int8)
    )
    return uids, states, bound.identity, str(plan.dataset.input_schema.reference), table


def _instances_digest(table: pd.DataFrame) -> str:
    columns = [c for c in ("motif_id", "reference", "start", "end", "motif_strand") if c in table]
    text = table[columns].astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def compute_occupancy(
    plan,
    dataset_name: str,
    hits: pd.DataFrame,
    *,
    roles: Mapping[str, Sequence[str]],
    sites: str,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    group_by: str | Sequence[str] | None = None,
    rules: OccupancyRules = OccupancyRules(),
    workers: int = 1,
) -> dict:
    """States per read and instance, the instances scored and the read table."""
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    roles = {state: list(channels) for state, channels in roles.items() if channels}
    unknown_states = sorted(set(roles) - set(CLASS_STATES))
    if unknown_states:
        raise ValueError(f"unknown states {unknown_states}; states are {list(CLASS_STATES)}")
    if not roles:
        raise ValueError(f"no class channels; give channels for some of {list(CLASS_STATES)}")
    used = [c for channels in roles.values() for c in channels] + [sites]
    unknown = sorted(set(used) - set(declared))
    if unknown:
        raise KeyError(f"channels {unknown} not in dataset channels {declared}")
    groupings = _groupings(group_by)
    scope: dict[str, Any] = {"project_dir": project_dir, "experiment_dir": experiment_dir}
    document = plan.to_dict()
    jobs = [
        (document, dataset_name, scope, groupings, roles, sites, hits, rules, worker, workers)
        for worker in range(workers)
    ]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        from smftools.parallel_utils import configure_worker_threads

        with ProcessPoolExecutor(
            max_workers=workers, initializer=configure_worker_threads, initargs=(1,)
        ) as pool:
            shards = list(pool.map(_shard, *zip(*jobs)))
    else:
        shards = [_shard(*jobs[0])]
    identity = shards[0][2]
    frame = shards[0][3]
    uids = [uid for shard in shards for uid in shard[0]]
    states = np.concatenate([shard[1] for shard in shards], axis=0)
    order = {uid: rank for rank, uid in enumerate(identity["molecule_uid"])}
    sort = np.argsort([order[uid] for uid in uids], kind="stable")
    uids = [uids[i] for i in sort]
    states = states[sort]
    columns = [c for c in ("read_id", "experiment_id", "physical_reference") if c in identity]
    reads = identity.set_index("molecule_uid").loc[uids, columns + groupings].reset_index()
    for column in groupings:
        reads[column] = reads[column].astype(str)
    instances = shards[0][4]
    return {
        "states": states,
        "molecule_uids": uids,
        "instances": instances,
        "reads": reads,
        "frame_reference": frame,
        "groupings": groupings,
    }


def frame_name(dataset) -> str:
    """A dataset spec's frame: its coordinate frame's reference, else its first reference."""
    frame = getattr(dataset, "coordinate_frame", None)
    return str(getattr(frame, "reference", None) or dataset.references[0])


def occupancy_tables(result: dict) -> pd.DataFrame:
    """Per grouping, group and instance (``group_occupancy``)."""
    frames = []
    for grouping in result["groupings"] or [ALL]:
        labels = (
            result["reads"][grouping].to_numpy()
            if grouping != ALL
            else np.full(len(result["reads"]), ALL)
        )
        frames.append(
            group_occupancy(result["states"], labels, result["instances"]).assign(grouping=grouping)
        )
    table = pd.concat(frames, ignore_index=True)
    leading = ["grouping", "group", "instance"]
    return table[[*leading, *[c for c in table.columns if c not in leading]]]


def read_states(output_dir: str | Path) -> dict:
    """A saved run: states, molecule UIDs, instances, reads."""
    output_dir = Path(output_dir)
    with np.load(output_dir / STATES_FILE, allow_pickle=False) as saved:
        states, uids = saved["states"], saved["molecule_uids"].tolist()
    record = json.loads((output_dir / KEY_FILE).read_text())
    return {
        "states": states,
        "molecule_uids": uids,
        "instances": pd.read_parquet(output_dir / INSTANCES_FILE),
        "reads": pd.read_parquet(output_dir / READS_FILE),
        "frame_reference": record["frame_reference"],
        "groupings": record["groupings"],
    }


def run_motif_occupancy(
    plan,
    dataset_name: str,
    output_dir: str | Path,
    motif_hits,
    *,
    roles: Mapping[str, Sequence[str]],
    sites: str,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    group_by: str | Sequence[str] | None = None,
    motif_reference: str | None = None,
    max_pvalue: float | None = None,
    families: Sequence[str] | None = None,
    motifs: Sequence[str] | None = None,
    rules: OccupancyRules = OccupancyRules(),
    workers: int = 1,
    refresh: bool = False,
) -> dict:
    """Compute (or reuse) per-read states; write tables and ``run.json``.

    Reused when the plan, dataset, referenced files, channel roles, rules,
    groupings and the motif instances scored all match.
    """
    from smftools import __version__
    from smftools.analysis.compute.motif_tracks import filter_hits
    from smftools.tools.analysis_cache import cache_key
    from smftools.tools.motif_tracks import read_hits

    output_dir = Path(output_dir)
    hits = read_hits(motif_hits)
    frame = frame_name(plan.datasets[dataset_name])
    reference = motif_reference or frame
    if not (hits["reference"].astype(str) == reference).any():
        available = sorted(hits["reference"].astype(str).unique())
        raise KeyError(
            f"no motif instances on reference {reference!r} (the dataset frame); "
            f"the table has {available} -- pass motif_reference"
        )
    selected = filter_hits(
        hits, reference=reference, max_pvalue=max_pvalue, families=families, motifs=motifs
    )
    roles = {state: sorted(channels) for state, channels in roles.items() if channels}
    key = cache_key(
        plan,
        dataset_name,
        base_dir=project_dir or experiment_dir,
        parameters={
            "roles": roles,
            "sites": sites,
            "group_by": _groupings(group_by),
            "rules": rules.record(),
            "instances": _instances_digest(selected),
        },
    )
    key_path = output_dir / KEY_FILE
    reused = (
        not refresh
        and key_path.is_file()
        and (output_dir / STATES_FILE).is_file()
        and json.loads(key_path.read_text()).get("key") == key
    )
    if reused:
        result = read_states(output_dir)
    else:
        result = compute_occupancy(
            plan,
            dataset_name,
            selected,
            roles=roles,
            sites=sites,
            project_dir=project_dir,
            experiment_dir=experiment_dir,
            group_by=group_by,
            rules=rules,
            workers=workers,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / STATES_FILE,
            states=result["states"],
            molecule_uids=np.asarray(result["molecule_uids"], dtype=str),
        )
        result["instances"].to_parquet(output_dir / INSTANCES_FILE, index=False)
        result["reads"].to_parquet(output_dir / READS_FILE, index=False)
        key_path.write_text(
            json.dumps(
                {
                    "key": key,
                    "frame_reference": result["frame_reference"],
                    "groupings": result["groupings"],
                    "states": list(STATES),
                },
                default=str,
            )
        )
    occupancy = occupancy_tables(result)
    occupancy.to_parquet(output_dir / OCCUPANCY_FILE, index=False)
    states = result["states"]
    record = {
        "key": key,
        "states_reused": reused,
        "frame_reference": result["frame_reference"],
        "motif_reference": reference,
        "molecules": int(states.shape[0]),
        "instances": int(states.shape[1]),
        "instances_outside_dataset": int(len(selected) - states.shape[1]),
        "state_counts": {state: int((states == code).sum()) for code, state in enumerate(STATES)},
        "roles": roles,
        "sites": sites,
        "rules": rules.record(),
        "groups": {
            grouping: sorted(result["reads"][grouping].unique()) for grouping in result["groupings"]
        },
        "smftools_version": __version__,
    }
    (output_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return record
