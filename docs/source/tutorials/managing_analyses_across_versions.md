# Managing analyses across smftools versions

An smftools upgrade can change an algorithm, the semantic graph, or neither.
Before recomputing an experiment or project, use the read-only planning and
inventory commands to distinguish code-driven invalidation from changed inputs
or configuration. Existing generations and analysis caches remain available
until you explicitly manage their retention.

## Inspect experiment impact

Run the experiment planner under the installed smftools version and request its
upgrade-impact view:

```shell
smftools experiment plan experiment_config.csv --target full --upgrade-impact
```

This command does not execute a stage or modify the experiment. It groups the
existing semantic plan by compatibility state and separates direct triggers
from downstream effects:

- `compatible` nodes can be reused.
- `stale_algorithm`, `stale_config`, and `stale_input` nodes are direct
  recomputation triggers.
- `dependent_recompute` nodes are invalidated by an upstream trigger named in
  the report.
- `blocked_missing_input` nodes cannot run until their required input is
  restored or supplied.

Although the option is named `--upgrade-impact`, it faithfully reports every
reason already found by the semantic planner. A configuration or input change
therefore remains visible alongside an algorithm-version change.

Experiment cost estimates are observations, not throughput predictions. The
report sums valid nonnegative `elapsed_seconds` values from prior completed
stages. When only some affected nodes have historical timings, the estimate is
marked partial and the unknown nodes are listed. No timing history means the
cost is reported as unknown.

Use `--json` for the stable, schema-versioned representation:

```shell
smftools experiment plan experiment_config.csv \
  --target full --upgrade-impact --json > experiment-upgrade-impact.json
```

## Inspect project impact

Project planning uses the same compatibility states. Supply the target and
canonical reference required by the ordinary project plan:

```shell
smftools project plan PROJECT_DIR embedding REFERENCE_UID --upgrade-impact
```

Project products are task-local, and their persisted cache definitions contain
more detail than the coarse plan request. The impact report therefore leaves
historical project cost unknown instead of guessing from artifact size or
treating an arbitrary cache as compatible.

This is separate from cache inventory. To inspect periodicity and embedding
caches already retained on disk, run:

```shell
smftools project analyses list PROJECT_DIR --stale
```

The inventory classifies definitions as `current`, `stale`, or `invalid` by
comparing their stored algorithm and semantic-graph versions with the installed
ones. It reads metadata and file sizes only; it does not load result tables or
unpickle estimator models. Add `--json` when the result will be archived or
consumed by automation.

## What changes during recomputation

Periodicity and embedding definitions carry independently bumpable algorithm
versions. A periodicity-only implementation change creates a new periodicity
cache key without invalidating embeddings, while a shared semantic-graph change
creates new keys for both. Old cache directories are neither rewritten nor
deleted automatically.

Generation-aware experiment stages follow the same preservation principle. A
recomputed stage publishes a new immutable generation and advances
`current.json`; the prior generation stays addressable. Pin any generation that
must be retained for a publication or external record:

```shell
smftools experiment generations EXPERIMENT_ID pin STAGE GENERATION_ID \
  --reason "paper figure 3"
```

Generation pruning remains a dry-run policy planner. It protects current,
pinned, recent, unreadable, and policy-retained generations, and deletion is not
available while byte-level reproducibility is not authoritative.

## Recommended upgrade workflow

1. Record or pin the generations that support published results.
2. Run experiment and project plans with `--upgrade-impact` under the version
   you intend to use.
3. Inventory project caches with `project analyses list --stale`.
4. Restore any missing inputs, then rerun only the requested targets.
5. Re-run the plans and inventories to confirm the new results are compatible.
6. Keep historical generations and caches until their retention requirements
   and reproducibility have been reviewed explicitly.

For the on-disk generation and retention layout, see
[](directory_organization.md#immutable-generations-and-retention). For the full
planner contract and compatibility states, see
[](semantic_variant_workflows.md).
