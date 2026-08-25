# Selective POD5 re-basecalling and processing-lineage audit

> **Repository state reviewed:** `0e742db` — recorded in this document.
> **190 commits on `main` since.** An audit describes the code at a moment; it
> goes stale rather than completing. Re-verify any specific claim before relying on it.

**Audit date:** 2026-08-14

**Repository:** `smftools`

**Repository state reviewed:** `0e742db` on `chore/start-2.21.0-dev`

**Scope:** current POD5/input-manifest provenance, Dorado basecalling, molecule
identity, raw and preprocess generations, QC/dedup masks, semantic downstream
invalidation, project registration, and the feasibility of re-basecalling all
or selected molecules from an established experiment/project.

**Out of scope:** implementation, automatic deletion of historical artifacts,
and changing scientific QC defaults.

## Executive assessment

The requested workflow is technically feasible and fits the existing
architecture well, but it is not currently a supported end-to-end operation.

Both required low-level selection mechanisms already exist upstream:

- Dorado accepts a newline-delimited read-ID file through `--read-ids` and
  basecalls only those POD5 reads.
- `pod5 filter` can materialize selected read IDs from one or more POD5 inputs
  into a new POD5 artifact.

The repository also has most of the necessary lineage foundation:

- content-addressed input manifests;
- immutable raw generations;
- immutable preprocess generations;
- stable experiment, molecule, and segment identities;
- retained QC and deduplication masks instead of row deletion;
- semantic downstream invalidation; and
- project registration by artifact pointer.

What is missing is an authority that joins those pieces. There is no command
that freezes a QC-defined molecule cohort from one preprocess generation,
resolves those molecules back to original POD5 UUIDs, runs a fully pinned
Dorado model, publishes a descendant raw generation, executes new downstream
stages, and registers the result without replacing the established project's
active processing lineage.

The recommended design is an immutable **processing lineage**. A lineage owns
one frozen selection, its source-signal resolution, the exact resolved Dorado
and modification models, a descendant raw generation, its downstream stage
results, and a terminal validation report. It is staged and published as a
unit. Project registration may expose multiple lineages for one biological
experiment, but exactly one is active by default; switching the publication
candidate is an explicit promotion operation.

## The scientific distinction that must remain visible

Three requests that sound like “re-basecall the project” have different
meanings:

| Scope | Signal reads presented to Dorado | Scientific interpretation |
| --- | --- | --- |
| `all-signal` | Every UUID in the authoritative POD5 sources | Full reprocessing; can discover reads that the old model failed to call, align, or pass QC |
| `all-parent-molecules` | POD5 parents represented in a selected old raw generation | Reprocess the old database universe; cannot recover signal reads absent from that generation |
| `selected-parent-molecules` | POD5 parents selected from a frozen old raw/preprocess generation | Derived publication cohort; selection is conditioned on the old basecalls and old QC |

Only `all-signal` supports an unbiased comparison of old and new basecaller QC
yield. A selected re-basecall is still useful when the publication cohort was
defined independently or the intent is to avoid spending compute on excluded
molecules, but it cannot discover a read that failed under the older model and
would pass under the newer model.

Every mode must recompute read, mapping, modification, variant, and duplicate
outputs that depend on the new sequence/alignment. A parent QC pass is an input
selection fact, not permission to copy the old QC result onto the new call.
The lineage report should at least distinguish:

- selected by the parent predicate;
- resolved to source POD5 signal;
- emitted by Dorado;
- ingested into the new raw generation;
- passing each newly computed QC mask;
- passing new deduplication; and
- missing or rejected at each boundary, with reasons.

## Current repository capabilities

### Source and signal provenance

`informatics/input_manifest.py` publishes a schema-1 resolved input manifest
with a SHA-256, byte size, source kind, source role, namespace, and source ID for
each POD5. Absolute paths are excluded from semantic identity but retained as
resolution hints. The raw generation copies this manifest into its immutable
artifact set.

For POD5-derived raw stores, `cli/raw_adata.py::_attach_pod5_metadata` adds
scalar `pod5_*` fields to matched reads. `pod5_origin` is currently only a
basename, while the input manifest owns the stronger per-file source identity.

The present source model therefore proves which POD5 bytes were used, but it
does not provide a relocation-aware archive/catalog that can find those same
bytes a year later. Re-basecalling must validate the original manifest path or
accept an explicit checksum-to-path relocation mapping. It must never accept a
same-named POD5 without checking its source SHA-256.

### Selective POD5 access

`informatics/pod5_functions.py` already reads selected UUIDs through the POD5
Python API, and its random `max_basecall_reads` helper can write a smaller POD5.
That helper is not a suitable publication contract: it samples by count rather
than accepting a frozen molecule selection, does not publish a selection
manifest, and is not connected to descendant generation provenance.

Current Oxford Nanopore interfaces provide two appropriate implementation
routes:

1. pass the frozen UUID file directly to `dorado basecaller --read-ids`; or
2. use `pod5 filter --ids` when a self-contained selected-signal artifact is
   requested.

Direct `--read-ids` should be the default compute path because it avoids
rewriting signal. Optional per-source filtered POD5 artifacts are valuable for
a publication archive or portable replay bundle.

### Basecalling provenance

`informatics/basecalling.py` builds canonical and modified Dorado commands,
but does not accept an exact read-ID selection. The intermediate compatibility
record includes the POD5 checksum, requested model string, selected options,
and Dorado executable version.

That is insufficient for a floating model selector such as `hac` or
`hac@latest`. Dorado documents that a variant selector resolves the latest
model compatible with the POD5 sequencing condition. Reproducibility therefore
requires the resulting full model name (and modification model names) from the
BAM read-group metadata, plus model artifact checksums where locally retained.
A request may use `@latest`; a published result may not use the floating alias
as its only model identity.

Model resolution is chemistry-specific. A project spanning different flow-cell
or kit conditions may resolve one request such as `hac@latest` to different
full model names per experiment. The project plan must show that fan-out before
execution and reject unsupported legacy conditions rather than silently using
one model everywhere.

### Molecule and source-read identity

The raw store publishes `experiment_uid`, `molecule_uid`, template/read ID, and
segment identity. Namespaced multi-source alignments retain `source_read_id`,
which is preferable to stripping a namespace heuristically.

Dorado read splitting is the principal identity complication. It is enabled by
default; split children have new UUIDs and carry their source POD5 parent in the
`pi` BAM tag. The current default BAM-tag extraction does not promote `pi` to
the raw spine. Consequently, an older database row can name a split child that
does not exist in the POD5 file.

Resolution must use the following ordered evidence and fail on ambiguity:

1. a future explicit `pod5_read_id`/`parent_read_id` field;
2. `source_read_id` or bare read ID when that UUID exists in the indexed POD5;
3. the retained source/aligned BAM's `pi` tag for split children; then
4. an unresolved result that blocks execution unless the user explicitly
   removes those rows from the request.

No implementation should silently treat missing selected UUIDs as acceptable.
If read splitting remains enabled in the new basecall, the lineage needs a
one-parent-to-many-output identity table. If identity-preserving mode disables
splitting, that scientific change must be part of basecall configuration and
provenance rather than an invisible implementation detail.

### QC selection

Partitioned preprocessing retains all rows and publishes explicit masks. The
current chain includes:

- `passes_read_qc`;
- `passes_modification_qc`;
- `passes_nonvariant_qc` and `passes_variant_qc` when variant analysis applies;
- `passes_qc` for combined QC;
- `is_duplicate`; and
- `passes_dedup`.

The FASTQ/export-bundle path already resolves the first available one of
`passes_dedup`, `passes_qc`, or `passes_read_qc`. That precedence is useful for
a convenience preset but not sufficient for a requested publication cohort.
A re-basecall request needs a versioned, explicit predicate against one exact
preprocess generation and must freeze the resulting molecule IDs before any
external tool runs.

Group-dependent analyses require additional care. Re-running duplicate
clustering or model fitting on a subset can produce different answers than
running the same analyses on the full signal universe. The lineage must record
its cohort boundary and must not claim full-experiment equivalence.

### Generation and project behavior

Raw and preprocess already publish immutable generations with atomic current
pointers. A force-redo raw run, however, advances the experiment's one raw
current pointer. Spatial, HMM, and latent results are downstream of the new raw
channel, but the repository does not currently publish a whole multi-stage
branch that preserves an old and new publication candidate side by side.

The project registry likewise records one discovered spine per stage for an
experiment. Registering a second run with the same `experiment_uid` under a new
experiment ID is deliberately rejected to prevent double-counting. This is the
correct invariant; re-basecalled results are revisions of one experiment, not
independent biological replicates.

A project-level solution therefore needs a lineage map beneath each experiment
entry, an explicit active lineage, and selection rules that prohibit accidental
pooling of two lineages from the same experiment. Cross-lineage comparisons may
join on origin identity, but ordinary materialization chooses one lineage per
experiment.

## External capability verification

The proposal relies on current primary documentation:

- [Dorado basecall overview](https://software-docs.nanoporetech.com/dorado/latest/basecaller/basecall_overview/)
  documents `--read-ids` for newline-delimited selective basecalling.
- [Dorado model selection](https://software-docs.nanoporetech.com/dorado/latest/models/selection/)
  documents full names, `@latest`, chemistry-aware variant selection, and
  modified-model compatibility.
- [Dorado FAQ](https://software-docs.nanoporetech.com/dorado/latest/troubleshooting/faq/)
  documents resolved basecall/modified model names in BAM `@RG` metadata and
  the supported-condition boundary.
- [Dorado read splitting](https://software-docs.nanoporetech.com/dorado/latest/basecaller/read_splitting/)
  documents default splitting, `--disable-read-splitting`, and the `pi` parent
  tag.
- [Dorado SAM specification](https://software-docs.nanoporetech.com/dorado/latest/basecaller/sam_spec/)
  defines the read-group model metadata and relevant per-read tags.
- [POD5 tools](https://software-docs.nanoporetech.com/pod5/latest/tools/)
  documents `pod5 filter`, strict missing-ID behavior, and multi-input caveats.
- [POD5 dataset API](https://software-docs.nanoporetech.com/pod5/latest/reference/api/dataset/)
  provides indexed read-ID lookup and selected iteration for a Python backend.

These are evolving tool interfaces. Implementation must probe and record the
installed tool versions and validate required flags before creating lineage
output.

## Findings

| ID | Severity | Finding |
| --- | --- | --- |
| SRB-C1 | Critical | There is no supported operation joining a frozen old QC cohort to selective POD5 basecalling and a descendant raw/downstream lineage |
| SRB-C2 | Critical | Old Dorado split-child IDs may not resolve to POD5 UUIDs because `pi` is not a default durable raw identity field |
| SRB-H1 | High | Input manifests prove POD5 content but current project registration has no relocation-aware source archive/catalog for later replay |
| SRB-H2 | High | Floating Dorado model selectors are not resolved into an immutable model identity in smftools compatibility/publication records |
| SRB-H3 | High | Current export QC precedence cannot express and freeze an arbitrary requested QC predicate against an exact preprocess generation |
| SRB-H4 | High | Raw force-redo advances one current pointer; it does not publish a side-by-side multi-stage processing lineage suitable for publication review |
| SRB-H5 | High | The project registry cannot expose multiple processing revisions of one experiment without either replacing pointers or triggering its duplicate-identity defense |
| SRB-H6 | High | Selected old-QC re-basecalling can be mistaken for a full unbiased new-model reanalysis unless scope and cohort conditioning are explicit |
| SRB-M1 | Medium | `pod5_origin` is basename-only and cannot independently identify duplicate basenames or relocated source files |
| SRB-M2 | Medium | New Dorado splitting, trimming, barcoding, and minimum-Q-score behavior can change selected-input to output cardinality and identity |
| SRB-M3 | Medium | Direct-modification re-basecalling must resolve compatible simplex and modification models and cannot reuse old MM/ML-derived QC |
| SRB-M4 | Medium | Subset-dependent duplicate clustering, fitted HMMs, and other cohort analyses may differ from full-universe processing even under the same parameters |

## Recommended direction

Implement the companion plan as a new semantic program rather than extending
the random POD5 subsampler or FASTQ exporter:

1. Freeze a versioned parent molecule selection.
2. Resolve every selected molecule to one source POD5 UUID with explicit
   evidence.
3. Validate source checksums and resolve the exact Dorado/model bundle.
4. Basecall directly by read-ID list, optionally materializing filtered POD5s.
5. Publish a descendant raw generation with parent/selection/basecall lineage.
6. Recompute requested downstream targets under a staged lineage root.
7. Validate counts, identities, artifacts, and refreshed QC deltas.
8. Publish the lineage atomically without changing the active lineage.
9. Promote it explicitly at experiment/project scope only after review.

The companion implementation plan is
[selective_pod5_rebasecalling_implementation_plan.md](selective_pod5_rebasecalling_implementation_plan.md).
