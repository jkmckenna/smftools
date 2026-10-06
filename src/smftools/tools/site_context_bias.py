"""Count a channel's modified and observed calls per site (`SCB-01`).

Molecules come from a datasets-only ML plan through `bind_ml_dataset`
(`MLX-10`): its channel names the layer and site context, ``filters`` the
QC/dedup flags, a label table or identity column the groups, ``positions``
the window. Calls stream batch by batch -- split over worker processes by
whole blocks (`MLX-11`) -- into running per-site sums, so memory scales with
sites x groups, not molecules. The statistics are
`smftools.analysis.compute.site_context_bias`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smftools.analysis.compute.site_context_bias import accumulate_site_calls, site_table

ALL = "all"


@dataclass
class SiteCounts:
    """Observed and modified calls per (group, physical reference, position)."""

    sites: pd.DataFrame
    channel: str
    frame_reference: str
    # physical reference -> key of the sequence its positions index (the frame
    # reference: positions are always in the dataset's frame coordinates).
    sequence_for: dict[str, str] = field(default_factory=dict)
    # Stage spines of the experiments read: where reference sequences live.
    spines: list[str] = field(default_factory=list)


def _bind(plan, dataset_name: str, scope: dict, group_by: str | None):
    from smftools.machine_learning.orchestration import bind_ml_dataset

    return bind_ml_dataset(plan, dataset_name, group_by=[group_by] if group_by else [], **scope)


def _count_shard(document, dataset_name, scope, group_by, channel, worker_id, workers):
    """One worker's blocks: counts, positions, and the frame/physical references.

    Takes a plan *document* and returns plain data, so it runs in a worker
    process (parsed plans and bound datasets do not pickle).
    """
    from smftools.machine_learning.plan import parse_ml_plan

    bound = _bind(parse_ml_plan(document), dataset_name, scope, group_by)
    channels = [item.name for item in bound.dataset.plan.dataset.input_schema.channels]
    index = channels.index(channel)
    identity = bound.identity.set_index("molecule_uid")
    group = identity[group_by].astype(str) if group_by else pd.Series(ALL, index=identity.index)
    physical = identity["physical_reference"].astype(str)
    batches = (
        bound.iter_batches(worker_id=worker_id, num_workers=workers)
        if workers > 1
        else bound.iter_batches()
    )
    counts: dict = {}
    for batch in batches:
        uids = list(batch.molecule_uids)
        keys = list(zip(group.loc[uids], physical.loc[uids], strict=True))
        accumulate_site_calls(
            counts,
            keys=keys,
            calls=batch.values[:, :, index],
            observed=batch.observed_mask[:, :, index],
        )
    frame = str(bound.dataset.plan.dataset.input_schema.reference)
    spines = sorted(
        {
            str(path)
            for source in bound.dataset.plan.sources.values()
            for path in source.stage_spines.values()
        }
    )
    return counts, np.asarray(bound.dataset.plan.coordinates), frame, sorted(set(physical)), spines


def count_site_calls(
    plan,
    dataset_name: str,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    group_by: str | None = None,
    channel: str | None = None,
    workers: int = 1,
) -> SiteCounts:
    """Stream a plan dataset's calls on ``channel`` into per-site counts.

    ``group_by`` names an identity column (a label-table column, ``Barcode``,
    ``experiment_id``, ...); without it every molecule is group ``"all"``.
    ``channel`` defaults to the dataset's first. ``workers`` > 1 splits the
    read over processes by whole blocks; the counts are identical.
    """
    scope: dict[str, Any] = {"project_dir": project_dir, "experiment_dir": experiment_dir}
    declared = [item.name for item in plan.datasets[dataset_name].channels]
    channel = channel or declared[0]
    if channel not in declared:
        raise KeyError(f"channel {channel!r} not in dataset channels {declared}")
    document = plan.to_dict()
    jobs = [(document, dataset_name, scope, group_by, channel, w, workers) for w in range(workers)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            shards = list(pool.map(_count_shard, *zip(*jobs)))
    else:
        shards = [_count_shard(*jobs[0])]
    counts: dict = {}
    for shard, *_ in shards:
        for key, (observed, modified) in shard.items():
            if key in counts:
                counts[key][0] += observed
                counts[key][1] += modified
            else:
                counts[key] = [observed, modified]
    _, positions, frame, _, spines = shards[0]
    physicals = sorted({name for _, _, _, names, _ in shards for name in names})
    # Positions are in the dataset's single frame reference's coordinates
    # (its own reference, or the frame molecules were mapped into, `MLX-03`),
    # so every context is read from the frame reference's sequence.
    return SiteCounts(
        sites=site_table(counts, positions),
        channel=channel,
        frame_reference=frame,
        sequence_for=dict.fromkeys(physicals, frame),
        spines=spines,
    )


def reference_sequences(spine_paths) -> dict[str, str]:
    """Forward sequences by reference name, from spines' ``References``.

    Stored sequences are padded with ``N`` to the longest reference; each is cut
    back to its recorded length (either strand's), or, without one, stripped
    of trailing ``N`` -- padding must not read as bases in a context window.
    """
    from smftools.analysis.compute.site_context_bias import strand_of
    from smftools.informatics.partition_read import REFERENCE_LENGTHS_KEY, load_spine

    sequences: dict[str, str] = {}
    for path in spine_paths:
        spine = load_spine(path, verbose=False)
        lengths: dict[str, int] = {}
        for reference, length in dict(spine.uns.get(REFERENCE_LENGTHS_KEY, {}) or {}).items():
            lengths.setdefault(strand_of(str(reference))[0], int(length))
        references = dict(spine.uns.get("References", {}) or {})
        for key, value in {**references, **dict(spine.uns)}.items():
            if not (isinstance(key, str) and key.endswith("_FASTA_sequence")):
                continue
            if not isinstance(value, str):
                continue
            name = key[: -len("_FASTA_sequence")]
            sequence = value[: lengths[name]] if name in lengths else value.rstrip("Nn")
            sequences.setdefault(name, sequence)
    return sequences


COUNTS_FILE = "site_counts.parquet"
COUNTS_META = "site_counts.json"


def _counts_key(plan, dataset_name: str, channel: str | None, group_by: str | None) -> dict:
    return {
        "plan_hash": plan.plan_hash,
        "dataset": dataset_name,
        "channel": channel,
        "group_by": group_by,
    }


def load_or_count(
    plan,
    dataset_name: str,
    output_dir: str | Path,
    *,
    refresh: bool = False,
    **kwargs,
) -> tuple[SiteCounts, bool]:
    """Counts from ``output_dir`` when they were made from the same plan, dataset,
    channel and grouping; otherwise count and save them. Returns ``(counts, reused)``.
    """
    import json

    output_dir = Path(output_dir)
    key = _counts_key(plan, dataset_name, kwargs.get("channel"), kwargs.get("group_by"))
    table, meta = output_dir / COUNTS_FILE, output_dir / COUNTS_META
    if not refresh and table.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get("key") == key:
            return (
                SiteCounts(
                    sites=pd.read_parquet(table),
                    channel=saved["channel"],
                    frame_reference=saved["frame_reference"],
                    sequence_for=saved["sequence_for"],
                    spines=saved["spines"],
                ),
                True,
            )
    counts = count_site_calls(plan, dataset_name, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts.sites.to_parquet(table, index=False)
    meta.write_text(
        json.dumps(
            {
                "key": key,
                "channel": counts.channel,
                "frame_reference": counts.frame_reference,
                "sequence_for": counts.sequence_for,
                "spines": counts.spines,
            },
            indent=2,
        )
    )
    return counts, False


def run_context_bias(
    plan,
    dataset_name: str,
    output_dir: str | Path,
    *,
    project_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    group_by: str | None = None,
    channel: str | None = None,
    flank: int = 3,
    kmers: tuple[int, ...] = (1, 3),
    reference_group: str | None = None,
    drop_ambiguous: bool = True,
    workers: int = 1,
    refresh: bool = False,
    figures: bool = True,
) -> dict:
    """Count (or reuse counts), then write site, enrichment, k-mer and difference
    tables and figures to ``output_dir``. Returns the ``run.json`` record."""
    import json

    from smftools.analysis.compute.site_context_bias import (
        group_differences,
        kmer_rates,
        offset_enrichment,
        site_contexts,
    )

    output_dir = Path(output_dir)
    kmers = tuple(sorted(set(kmers)))
    for k in kmers:
        if k < 1 or k % 2 == 0 or k > 2 * flank + 1:
            raise ValueError(f"k-mer size {k} must be odd and between 1 and {2 * flank + 1}")
    counts, reused = load_or_count(
        plan,
        dataset_name,
        output_dir,
        refresh=refresh,
        project_dir=project_dir,
        experiment_dir=experiment_dir,
        group_by=group_by,
        channel=channel,
        workers=workers,
    )
    groups = list(dict.fromkeys(counts.sites["group"]))
    if reference_group is not None and reference_group not in groups:
        raise KeyError(f"reference group {reference_group!r} not among groups {groups}")

    sites = site_contexts(
        counts.sites,
        reference_sequences(counts.spines),
        flank=flank,
        sequence_for=counts.sequence_for,
    )
    sites.to_parquet(output_dir / "sites.parquet", index=False)
    enrichment = offset_enrichment(sites, flank=flank, drop_ambiguous=drop_ambiguous)
    enrichment.to_csv(output_dir / "offset_enrichment.csv", index=False)
    rates = pd.concat(
        [
            kmer_rates(sites, flank=flank, k=k, drop_ambiguous=drop_ambiguous).assign(k=k)
            for k in kmers
        ],
        ignore_index=True,
    )
    rates.to_csv(output_dir / "kmer_rates.csv", index=False)
    differences = None
    if reference_group is not None and len(groups) > 1:
        differences = group_differences(enrichment, reference_group=reference_group)
        differences.to_csv(output_dir / "group_differences.csv", index=False)

    written = []
    if figures:
        from smftools.analysis.plot.site_context_bias import (
            plot_enrichment_logo,
            plot_group_differences,
            plot_kmer_rates,
            plot_offset_enrichment_heatmap,
        )

        plot_offset_enrichment_heatmap(enrichment, output_dir / "offset_enrichment.png")
        plot_enrichment_logo(enrichment, output_dir / "enrichment_logo.png")
        written += ["offset_enrichment.png", "enrichment_logo.png"]
        for k in kmers:
            if k == 1:
                continue  # the centre base alone: one row, nothing to compare
            name = f"kmer_rates_k{k}.png"
            plot_kmer_rates(rates.loc[rates["k"] == k], output_dir / name)
            written.append(name)
        if differences is not None:
            plot_group_differences(differences, output_dir / "group_differences.png")
            written.append("group_differences.png")

    from smftools import __version__

    record = {
        **_counts_key(plan, dataset_name, counts.channel, group_by),
        "counts_reused": reused,
        "frame_reference": counts.frame_reference,
        "flank": flank,
        "kmers": list(kmers),
        "reference_group": reference_group,
        "drop_ambiguous": drop_ambiguous,
        "groups": groups,
        "sites": int(len(counts.sites)),
        "figures": written,
        "smftools_version": __version__,
    }
    (output_dir / "run.json").write_text(json.dumps(record, indent=2))
    return record
