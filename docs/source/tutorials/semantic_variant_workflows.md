# Semantic planning and integrated variant analysis

This guide describes how smftools decides whether work is reusable and how
variant evidence participates in preprocessing. Use it when planning a restart,
interpreting a compatibility decision, migrating a standalone variant workflow,
or packaging an experiment for an external workflow engine.

## Semantic graph and execution work

The semantic graph records scientific products: their configuration identity,
algorithm and schema versions, input identities, dependencies, output channels,
and validated artifacts. It answers whether an existing result means the same
thing as the requested result.

The execution work graph records how one incompatible semantic node is
calculated: partitions, tiles, task catalogs, reducers, plots, and resource
limits. A workflow engine may schedule those tasks differently without changing
the semantic result. Conversely, changing a scientific input or algorithm
invalidates a semantic node even when the same execution layout could be reused.

The coarse experiment graph is:

```text
raw -> preprocess -> spatial -> hmm -> latent
```

`full` targets the last enabled stage. The deprecated `variant` target resolves
to `preprocess`; variant analysis is not a separate stage. Within an immutable
preprocess generation, finer semantic nodes distinguish partition-derived
features, the variant reference set, variant evidence, reducers, variant cohort
metrics, and plots. This permits a plot or metric change to reuse compatible
evidence rather than rerunning every partition.

Project planning is separate from experiment execution. It freezes a selected
set of experiment generations and tracks membership, genomic features, and
variant-reporting identities independently. A membership or feature change
invalidates materialization, sample analysis, and shared embeddings that consume
that channel. A reporting-only variant change is recorded without needlessly
changing consumers that do not use variant reporting.

## Inspect a plan before running

Experiment planning is deterministic and read-only:

```shell
smftools experiment plan experiment.csv --target full
smftools experiment plan experiment.csv --target preprocess --json
```

Project planning is also read-only:

```shell
smftools project plan PROJECT_DIR materialization CANONICAL_REFERENCE --json
smftools project plan PROJECT_DIR embedding CANONICAL_REFERENCE --json
```

Each requested node has one compatibility state:

| State | Meaning |
| --- | --- |
| `compatible` | The validated result has matching scientific identity and can be reused |
| `missing` | No prior result exists for this node |
| `stale_config` | An output-affecting configuration value changed |
| `stale_algorithm` | The implementation or output schema changed |
| `stale_input` | A source artifact or consumed channel changed |
| `invalid_artifact` | A recorded output is absent, corrupt, unsafe, or fails validation |
| `dependent_recompute` | An upstream dependency must be recomputed |
| `blocked_missing_input` | A required external or upstream input is unavailable |

Planning never repairs or publishes data. Running the requested target executes
only the incompatible portion allowed by its dependency plan.

`force_redo_*` settings request targeted recomputation. They are intentionally
excluded from scientific configuration hashes: setting a force flag changes the
decision to recompute, not the meaning of the result. Coarse stage flags include
`force_redo_load_adata`, `force_redo_preprocessing`,
`force_redo_flag_duplicate_reads`, `force_redo_spatial_analyses`, the HMM force
flags, and `force_redo_latent_analyses`. Preprocess maps its more specific force
flags to the affected task, reducer, or plot nodes.

## Immutable preprocess generations

Preprocess writes a fresh generation beneath
`preprocess_adata_outputs/generations/`. It validates the task catalog, stores,
indexes, spine, manifests, checksums, schemas, and relative pointers before an
atomic `current.json` update makes the generation readable. Readers validate
that pointer and generation rather than choosing a directory by modification
time.

If generation construction, validation, publication, or canonical-spine update
fails, smftools restores the previous current pointer and removes the failed
replacement. Previous complete generations are retained. There is currently no
supported CLI for selecting an arbitrary retained generation; do not hand-edit
`current.json`. To reproduce an earlier scientific result, run its preserved
configuration and inputs so smftools can validate and publish the requested
state through the normal lifecycle.

Pointers within a generation are relative to the experiment run root. Moving
the complete run directory therefore preserves validation. Moving individual
generation files or copying only `current.json` does not.

## Variant evidence ownership

The partitioned raw stage owns aligned molecule identity, sequence observations,
and reference sequence sources. Preprocess owns the derived reference contract,
informative-site catalog, per-molecule evidence, QC annotations, cohort metrics,
and plots. Bare read IDs are not project-global identifiers; experiment and
molecule identities prevent collisions when two experiments contain the same
read name.

`references_to_align_for_variant_annotation` currently names exactly two
distinct reference members. Missing members, a partial pair, or an alias that
matches multiple raw reference sources is rejected. The resulting
`variant_reference_set_id` is based on sequence, orientation, accepted bases,
alignment scoring, conversion semantics, and versioned calling policies rather
than filesystem paths or display labels, so it is stable after relocation.
Multiple reference sets may coexist in stored evidence even though one legacy
configuration entry initially defines a two-member set.

Informative sites are substitution positions at which the accepted bases of the
two members are disjoint. Per-read indels are excluded from the initial calling
contract. At an informative position, a read receives the unique matching member
call or a no-call. No-call positions do not support either state. Ordered changes
between callable member states produce breakpoint annotations; a transition may
cross a storage-tile boundary because evidence is reduced across the complete
authoritative read span.

## Report and filter modes

`variant_analysis_mode` has four values:

| Mode | Behavior |
| --- | --- |
| `auto` | Uses `report` when both legacy reference members are configured; otherwise `off` |
| `off` | Does not request integrated variant evidence |
| `report` | Computes evidence, annotations, metrics, and plots without removing reads |
| `filter` | Computes the same products and applies only the explicitly configured policy |

Variant classification uses raw callable counts and has five durable classes:
`self_consistent`, `breakpoint`, `ambiguous_reference_assignment`,
`insufficient_evidence`, and `evidence_unavailable`. A fully discordant read
without a breakpoint is `ambiguous_reference_assignment`. Missing evidence,
zero callable sites, or an under-supported event remains diagnostic and passes
variant QC.

Preprocess preserves independent masks and reasons:

- `passes_nonvariant_qc` and `nonvariant_qc_reason` record read and modification
  QC before variant policy.
- `variant_qc_class`, `passes_variant_qc`, and `variant_qc_reason` record the
  variant decision.
- `passes_qc` is their conjunction.
- `is_duplicate` records duplicate selection, and `passes_dedup` is final QC
  combined with the keeper decision. Duplicate clustering considers all reads
  that pass nonvariant QC and prefers a variant-pass member as keeper.

In `filter` mode, thresholds for callable sites, callable fraction, and calls per
state determine whether evidence is sufficient. Only event classes explicitly
listed in `variant_qc_disallowed_event_classes` fail variant QC. See
[](experiment_config.md#variant-qc-and-migration) for the required settings.

## Named metric cohorts

Variant QC metrics are generation-scoped and emitted for these fixed cohorts:

| Cohort | Membership |
| --- | --- |
| `all_aligned` | Every aligned molecule in the reference set |
| `pre_dedup_nonvariant_qc` | Molecules passing nonvariant QC |
| `post_dedup_nonvariant_qc` | Nonvariant-QC molecules excluding duplicate reads |
| `pre_dedup_final_qc` | Molecules passing combined nonvariant and variant QC |
| `post_dedup_final_qc` | Final-QC molecules retained by duplicate selection |

Metrics include every cohort member when measuring noncallable rates. Callable
event denominators require complete evidence and at least one callable
informative site. Results are grouped overall and, when available, by reference,
sample, and reference/sample. The Parquet metrics and JSON/TSV summaries carry
their source generation and reference-set identities.

## Migration and downstream behavior

`smftools experiment variant` remains a deprecated compatibility alias. It
requests the authoritative preprocess generation in `report` or `filter` mode
and does not create or trust a standalone variant stage. Historical standalone
variant H5AD files are retained-row snapshots: they can be read by the legacy
reader, but file existence and legacy `*_performed` flags do not establish
semantic compatibility.

A legacy raw or preprocess H5AD cannot reconstruct complete all-molecule
evidence when earlier filtering or deduplication removed rows. Restore or
regenerate the partitioned raw source, then rerun preprocess. Current spatial,
HMM, latent, project materialization, and project embedding products consume
validated stage generations and channel identities; an upstream membership or
feature change invalidates only the downstream products that consume it.

## External workflows and containers

Workflow engines should use `smftools experiment run`, inspect
`workflow_result.json`, and finish with `smftools experiment validate`. The
result schema records the semantic plan, terminal outcome (`success`,
`compatible_skip`, or `failed`), generation/result identities, relative
artifacts, checksums, schemas, timings, resource decisions, and structured
failure details. `software_versions.json` records smftools, Python, selected
external tools and models, and production-container identity when supplied.

The supported production CPU/BAM image runs the same contract under Docker and
an Apptainer-compatible arbitrary UID. It contains the installed smftools wheel,
Python runtime dependencies, `minimap2`, `samtools`, `/bin/bash`, and `ps`; it
does not copy host files unless they are explicitly bind-mounted at runtime.
Build, mount, provenance, included-tool, and limitation details are in
[](../containers.md).
