# Selective re-basecalling and processing lineages

Re-basecalling an experiment with a newer model produces a *new* result. It does
not correct the old one. smftools therefore publishes it as a **processing
lineage**: a complete, immutable descendant of the experiment's earlier
artifacts, held beside the original rather than replacing it. Nothing you have
already cited changes until you explicitly promote the descendant, and even then
the original stays queryable.

## The distinction that matters most

Two requests that both "re-basecall the experiment" can answer very different
scientific questions. Choosing the wrong one is the easiest way to draw a
conclusion your data does not support.

| Selection mode | What is basecalled | What the result supports |
| --- | --- | --- |
| `all-signal` | Every read in the authoritative POD5 sources | Full-signal reanalysis. Comparable to the original as a whole. |
| `all-parent-molecules` | Every molecule the parent raw generation resolved | Reanalysis of the parent's universe. Excludes signal the parent never ingested. |
| `qc` | Molecules passing a predicate over one immutable preprocess generation | **Old-QC-selected** reanalysis. Scope-biased by construction. |
| `ids` | An explicit molecule/read list | Selected cohort. Scope-biased by construction. |

`all-signal` and `all-parent-molecules` are deliberately not aliases, and the
plan shows you the count difference before anything runs.

### Why old-QC selection is scope-biased

A `qc` request selects reads using QC computed from the **old** basecalls. Those
reads were, by definition, the ones the old model handled well. Re-basecalling
only them and comparing pass rates measures how the new model performs on the
old model's easy cases — not how it performs on your data.

This is a legitimate and often useful request: it is the cheapest way to ask
whether a specific cohort's calls improve. It is not a measurement of overall
basecaller quality, and a selected lineage must never be described as equivalent
to the full universe. The published transition report makes the scope explicit,
and the basecall generation records its kind (`full_source`, `parent_universe`,
or `selected_cohort`) so the artifact itself says which question it answers.

## Selection and QC are separate facts

No QC result is carried forward. The descendant lineage recomputes QC and dedup
against the new calls, and the two are compared rather than conflated:

```shell
smftools experiment rebasecall plan experiment_config.csv request.yaml --json
```

Every published lineage carries a `qc_transition.parquet` with **one row per
selected origin molecule**, whether or not it survived. Each row records the
origin identity, its POD5 read and source, the basecall outputs and new
molecules it produced, the recomputed QC and dedup flags, and a terminal status:

- `no_signal` — the selected read never resolved to source signal.
- `no_call` — the basecaller produced no output for it.
- `dropped_in_raw` — the new calls produced no descendant molecule.
- `failed_qc` — the descendant molecule did not pass recomputed QC or dedup.
- `duplicate` — it was marked a duplicate.
- `passed` — it passed recomputed QC and dedup.
- `qc_not_run` — the lineage stopped before preprocess, so QC was not recomputed.

Because every selected molecule appears exactly once, the counts reconcile: the
row count equals the frozen selection's, and the published summary can be
recomputed from the table alone. That property is checked, not assumed.

## Model pinning

A request may ask for a floating model such as `hac@latest`. What gets published
is never floating. smftools resolves the alias against the installed Dorado's
model catalog and the POD5 run conditions, then records the exact simplex and
modification model names, their content checksums as a model-bundle digest, the
Dorado version, and the normalized invocation.

Reuse keys on that resolved bundle, not the request string. Two requests both
spelling `hac@latest` over the same reads publish **different** basecalls once
the installed models differ — which is the entire point of resolving before
recording. The output BAM's header is checked against the resolved bundle after
execution, so a basecall produced by an unexpected model fails rather than
publishes.

## Direct modification calling

For the direct modality the adapter resolves one compatible simplex model and
the requested modification models together. Old MM/ML probabilities are not
reused, and neither is any QC derived from them: modification QC is recomputed
against the new calls like everything else. If a requested modification variant
has no compatible model for the resolved simplex, the plan blocks rather than
silently basecalling without it.

`min_qscore` defaults to zero at the basecaller boundary so smftools owns the
publication QC decision. A nonzero basecaller filter is allowed as explicit
scientific configuration, and appears as pre-ingestion loss in the transition
report.

## Source retention and disk

Re-basecalling needs the original POD5 signal. Source resolution is
checksum-first: the recorded path is used only if its bytes still match, and you
may supply an explicit relocation map when sources have moved. There is no
basename matching and no filesystem search.

Two ways to keep a lineage reproducible:

1. **Retain the source POD5s.** Cheapest in new storage, but the lineage cannot
   vouch for files it does not own — validation reports it as not replayable.
2. **Materialize filtered signal.** Set `signal.materialize: true` and the run
   publishes content-addressed filtered POD5s per source, holding exactly the
   selected reads. Self-contained and checksum-validated, so the lineage stays
   replayable even after the originals are gone.

Rough sizing: filtered signal costs roughly the selected fraction of the source
POD5 tree, the new BAM is comparable to the original basecall's, and each
descendant stage generation costs about what the parent's did. A selected-cohort
lineage over a small fraction of reads is correspondingly small; an `all-signal`
lineage is not.

## Promotion, rollback, and validation

Publication and selection are separate. A published lineage is registered and
queryable by name immediately, while the project keeps answering with whatever
it answered with before:

```shell
smftools project rebasecall plan PROJECT request.yaml --json
```

Making a lineage the answer is explicit promotion, and promotion validates
first. The check revalidates the lineage manifest, every stage generation it
names, the transition report's reconciliation, and the basecall's bytes. An
incomplete lineage cannot be activated, and a lineage that cannot be verified is
refused rather than trusted.

Rollback needs no separate machinery: promoting a prior complete lineage —
including `original` — is the same operation, and the lineage you moved away
from remains registered and queryable.

Two lineages of one experiment are the same biology processed twice. A query
resolves exactly one per experiment, and a selection that would include two
fails before materialization rather than double-counting every molecule.

## Publication checklist

Before citing a re-basecalled result:

1. **Name the question.** Is this full-signal reanalysis or an old-QC-selected
   cohort? The basecall's `generation_kind` answers it; your methods section
   should too.
2. **Validate the lineage**, with `require_replayable` if it is the reproducible
   record for a paper rather than an intermediate result.
3. **Check the transition report reconciles** and read its terminal-status
   counts, not just the survivors.
4. **Record the resolved model identity** — full model names, bundle digest, and
   Dorado version — rather than the floating alias you requested.
5. **Pin the generations** you cite, so retention policy cannot reclaim them.
   See [](directory_organization.md#immutable-generations-and-retention).
6. **Decide the source-retention story**: retained POD5s or materialized
   filtered signal. State which.
7. **Promote deliberately**, and only after the above. Prior lineages survive
   promotion, so this is reversible.

## Migration from existing runs

Nothing needs rewriting to use this. An experiment with no lineages reads as
holding a single `original` lineage, and a project registry written before
lineages existed resolves exactly as it did.

Two things to know when re-basecalling older runs:

- The authoritative input manifest must name POD5 sources. A run ingested from
  BAM has no signal to re-basecall, and the plan blocks saying so rather than
  guessing.
- A descendant is published beside the parent without advancing the stage's
  `current.json`, so ordinary readers keep resolving the parent until you
  promote. Existing scripts need no changes.

For the on-disk generation layout, see
[](directory_organization.md#immutable-generations-and-retention). For managing
results across smftools upgrades, see
[](managing_analyses_across_versions.md).
