# Experiment input, alignment, and round-trip implementation plan

**Plan date:** 2026-07-30

**Repository:** `smftools`

**Program status:** In progress

**Repository state at planning time:** `2c5bbdf`
(`feature/semantic-variant-acceptance`; `origin/main` at `4e1b1e5`)

**Source audit:**
[input_ingestion_alignment_audit.md](input_ingestion_alignment_audit.md)

**Predecessor programs:**

- [experiment_project_partitioned_pipeline_implementation_plan.md](../completed/experiment_project_partitioned_pipeline_implementation_plan.md)
- [project_and_latent_partitioned_pipeline_implementation_plan.md](../completed/project_and_latent_partitioned_pipeline_implementation_plan.md)
- [semantic_dag_variant_preprocessing_implementation_plan.md](../completed/semantic_dag_variant_preprocessing_implementation_plan.md)

## Objective

Replace the current path- and cache-driven raw-input boundary with one explicit,
validated, content-identified ingestion system that supports:

- raw-signal, FASTQ, unaligned BAM, and existing aligned BAM entry;
- deterministic multi-file and paired-end input resolution;
- one stable barcode/sample identity contract;
- lossless mate segments and one molecule-level SMF observation;
- covered bases, uncovered insert gaps, and overlap conflicts without inventing
  signal;
- structured Dorado, minimap2, BWA-MEM2, and Bowtie2 alignment adapters;
- validated external alignments and later CRAM input;
- experiment/project exports that declare whether they are sequence-only or
  lossless re-ingestion bundles;
- immutable raw generations and, only after that lifecycle exists, safe
  append-only growth; and
- stable workflow-facing result, validation, and provenance contracts.

The program must preserve current supported single-source behavior unless an
item explicitly introduces a documented fail-fast correction or migration.
Each implementation item is intended to be a focused branch and PR cut from the
then-current `main`. Feature branches must not modify
`src/smftools/_version.py`.

The program also records a separate project-CLI completion lane for the named
set, sample-analysis, and embedding command gaps found during the same audit.
Those tasks do not block the ingestion dependency chain.

## Current baseline

The plan assumes the following behavior:

- `smftools experiment full` executes the semantic
  raw -> preprocess -> spatial -> HMM -> latent graph.
- Downstream partitioned stages, preprocess generations, project products, and
  latent/project model products already use explicit identity, validation,
  provenance, and transactional publication patterns.
- `ExperimentConfig` discovers POD5, FAST5, FASTQ, BAM, and H5AD files.
- A homogeneous FASTQ directory is converted to one
  `canonical_basecalls.bam`.
- A BAM directory is discovered but later treated as one BAM path and is not
  functional.
- Any supplied BAM is treated as sequence input to align again unless an
  internal derived output happens to exist.
- Alignment execution is hard-coded to Dorado or minimap2.
- Paired FASTQs receive BAM paired/read1/read2 flags and a shared query name,
  but the raw store requires globally unique `read_id` rows and has no
  mate-assembly layer.
- Experiment/project FASTQ export writes sequence, quality, and a minimal
  barcode/path manifest; ingestion does not consume that manifest.
- The raw semantic node hashes resolved configuration but does not identify the
  complete input contents as ordered source artifacts.
- Raw intermediate reuse is frequently based on output-path existence.
- The project CLI exposes registry, materialization, planning, validation,
  latent export, FASTQ export, and sample-store inspection, while named-set
  mutation and execution of planned sample-analysis/embedding targets remain
  API-only.

The new implementation must reuse the existing experiment/molecule identity,
semantic-node, artifact-validation, resource-envelope, region-catalog,
sidecar-manifest, and immutable-generation authorities. It must not create
parallel notions of project identity, stage completion, or compatibility.

## Program finding IDs

These IDs provide stable references for PR descriptions and acceptance tests.

| ID | Severity | Finding |
| --- | --- | --- |
| IAR-C1 | Critical | A discovered BAM directory is passed downstream as though the directory were one BAM |
| IAR-C2 | Critical | Paired mates share a query name but are neither assembled into one molecule nor compatible with raw-store unique row identity |
| IAR-H1 | High | Existing aligned BAMs are realigned; no authoritative aligned-input mode exists |
| IAR-H2 | High | Raw source identity is path/config based and intermediate reuse can ignore changed or appended inputs |
| IAR-H3 | High | Barcode/sample identity has no single precedence contract and already-demultiplexed non-split BAMs can lose labels |
| IAR-H4 | High | Paired FASTQ alignment does not preserve a first-class two-stream fragment contract |
| IAR-H5 | High | Exported FASTQs are not provenance-preserving or lossless re-ingestion bundles |
| IAR-H6 | High | Alignment execution is hard-coded to Dorado/minimap2 and unknown aligners fail late |
| IAR-M1 | Medium | Mixed-type directories silently select one input type by priority |
| IAR-M2 | Medium | FASTQ mate and barcode filename inference does not cover common Illumina/CASAVA names safely |
| IAR-M3 | Medium | Sequence-only FASTQ cannot preserve direct-modification MM/ML signal |
| IAR-M4 | Medium | SAM/CRAM and multi-alignment source collections have no explicit contract |
| IAR-M5 | Medium | Workflow `experiment run` accepts one staged file but no reproducible multi-file manifest/bundle |
| IAR-M6 | Medium | Project named sets and planned sample-analysis/embedding targets lack complete CLI execution surfaces |

## Agreed design contracts

These contracts constrain the PRs below. Changing one requires a design review
and an update to this plan before implementation proceeds.

### The resolved input manifest is the sole source-set authority

Every experiment raw request resolves to one versioned canonical input manifest,
whether the user supplies:

- one file;
- a directory;
- a user-authored manifest; or
- a re-ingestion bundle.

Directory scanning is a convenience resolver, not scientific identity. It must
produce the same canonical rows as an equivalent user manifest.

The canonical manifest owns:

- ordered source membership;
- content checksum and byte size;
- file kind and compression;
- sample/barcode/read-group declarations;
- pair/template and mate declarations;
- alignment role;
- modification-signal capability;
- trimming/filtering declarations where known; and
- source namespace.

Paths are provenance and resolution hints. Compatibility is based on canonical
content and semantic metadata, not only absolute paths or mtimes.

### Ambiguity fails before external execution

The resolver must reject, before basecalling/alignment:

- mixed source kinds without an explicit manifest policy;
- a BAM directory until the multi-alignment source contract is implemented;
- duplicate source paths or content rows;
- incomplete or multiply assigned mate pairs;
- conflicting barcode/sample declarations;
- unsupported source/alignment-mode combinations;
- direct-modification analysis from signal-incapable input; and
- unknown aligner names.

Legacy directory inference remains for homogeneous supported directories.
Behavior that previously silently ignored recognized files becomes an explicit
migration error.

### Input kind and alignment policy are separate

The initial public alignment modes are:

| Mode | Meaning |
| --- | --- |
| `align` | Produce an alignment through a configured adapter |
| `existing` | Validate and normalize an existing alignment without realigning reads |

There is no unvalidated `trust_existing` mode. "Existing" may copy,
coordinate-sort, index, or normalize an artifact into the raw staging
generation, but it may not change alignment placement.

Legacy BAM configs default to the current `align` behavior until the user
explicitly chooses `existing`. The migration must not guess from the presence
of mapped records whether realignment was intended.

`input_already_demuxed`, `skip_bam_split`, and `align_from_bam` retain their
specific meanings and do not alias alignment mode.

### Alignment outputs are owned artifacts

Every alignment route produces a versioned alignment manifest containing:

- source-manifest digest;
- reference bundle identity/checksum;
- adapter/external aligner name and version;
- normalized argv or external command provenance;
- single/paired layout;
- sort/index state;
- alignment BAM checksum;
- tag-preservation capabilities;
- read/reference counts and validation summary; and
- relative owned artifact paths.

Existing alignments are copied or normalized into the staging generation so a
completed raw generation remains relocatable and cannot be changed by mutating
an external source path.

### Aligner adapters are structured and shell-free

An adapter declares:

- accepted input layouts;
- required executable and version probe;
- reference-index preparation;
- argv construction as a list of arguments;
- paired-stream behavior;
- output type;
- sort/index behavior;
- expected tag preservation; and
- normalized provenance.

No adapter accepts an arbitrary shell command string. External workflows hand
off an already-produced alignment and manifest through `alignment_mode:
existing`.

The first adapter refactor must preserve Dorado and minimap2 behavior before
BWA-MEM2 or Bowtie2 is added.

### Reference identity is exact

Existing and adapter-produced alignments are validated against the exact
alignment reference bundle:

- FASTA content checksum;
- `@SQ` names, lengths, and ordering semantics;
- converted/deaminase reference transformation;
- alignment-region reduction and original-coordinate mapping; and
- reference interval-map identity.

For externally aligned conversion data, smftools must expose a stable prepared
alignment-reference bundle so the user can align against the same reference
records smftools later validates.

Filename agreement is never sufficient reference validation.

### Barcode/sample identity has one precedence contract

The precedence order is:

```text
explicit manifest declaration
  > validated BAM BC/RG/SM metadata
  > configured barcode-sequence classification
  > legacy filename fallback
```

Every route publishes the same barcode sidecar schema before raw molecule
metadata is built. Conflicts, missing labels, and unclassified fractions are
recorded in validation output.

Filename fallback is retained only for compatible legacy single-end inputs and
must emit a warning when it is the authority.

### A molecule and an alignment segment are distinct identities

For single-end and long-read input, one molecule normally has one primary
segment. For paired-end input, one molecule may have R1 and R2 segments.

Persistent identities are:

```text
template/read_id        original instrument/template name
segment_id              R1, R2, or single/long-read segment
segment_uid             experiment + template + segment
molecule_uid            experiment + template
```

The raw molecule spine has one row per `molecule_uid`. Lossless segment
artifacts may have multiple rows per molecule and must use `segment_uid` as
their unique key. Bare query name is not used as a segment-row primary key.

Discordant, secondary, supplementary, and singleton-mate state is explicit
metadata, not inferred from name suffixes downstream.

### Missing paired-end insert sequence stays missing

Materialization produces one molecule row:

- positions covered only by R1 use R1;
- positions covered only by R2 use R2;
- positions covered by neither mate remain `NaN` for SMF signal and false for
  base coverage;
- an overlap agreement produces one observation;
- an overlap SMF disagreement is `NaN` and is marked in a conflict mask;
- sequence disagreements select the higher-quality base; equal-quality
  disagreements become `N`; and
- overlapping mates never count as two molecules.

The versioned initial consensus contract applies to conversion/deaminase
paired-end data. Paired direct-modification input remains explicitly
unsupported until a probability-consensus contract is separately validated.

The existing outer alignment-span concept is not reused as observed base
coverage. New-schema outputs publish at least:

- `covered_base_mask`;
- `mate_coverage_count`; and
- `overlap_conflict_mask`.

Compatibility readers may derive covered-base masks for legacy single-segment
records from CIGAR aligned operations.

### Segment data remains lossless; consensus is reproducible

The raw generation retains normalized segment sequence, quality, CIGAR,
alignment flags, and per-segment SMF evidence. The molecule view is derived
deterministically from those segment artifacts with a recorded consensus
algorithm/schema version.

This allows consensus policy upgrades without discarding source segments or
realigning reads.

### Direct-modification capability is explicit

Input capability is not inferred from modality alone:

- POD5 plus compatible basecalling can produce direct signal;
- a modified BAM must retain valid MM/ML tags;
- a sequence-only FASTQ cannot be a lossless direct-modification source; and
- an aligner route that strips MM/ML is rejected for direct analysis unless a
  separately validated tag reattachment mechanism exists.

The initial program does not implement tag reattachment by read name.

### Raw publication is immutable and manifest-driven

Raw generation publication follows the established transactional pattern:

1. resolve and checksum the input manifest;
2. plan the raw semantic nodes;
3. create a unique staging generation;
4. build or reuse only validated content-addressed intermediate artifacts;
5. validate alignment, sidecars, segments, molecule spine, indexes, and
   manifests;
6. atomically publish the immutable generation;
7. atomically advance the raw current pointer;
8. update the experiment stage completion record; and
9. refresh consolidated/project discovery only after publication.

Path existence alone never establishes compatibility. Force-redo creates a new
generation/revision and does not write through the current generation.

### Append is an explicit source-set transition

Append-only behavior is introduced only after immutable raw generations exist.
The resolver classifies a new manifest relative to the current one as:

- identical;
- pure addition;
- removal;
- metadata mutation; or
- content mutation.

Only pure addition may use incremental raw extraction. It writes new source
partitions/shards and publishes a new generation referencing validated prior
immutable content. Removal or mutation requires a full new raw generation.

Downstream semantic nodes extend or recompute according to their declared
consumed channels; they never continue because an old intermediate filename
exists.

### Export products declare their re-ingestion capability

Sequence-only FASTQ export remains supported and is labeled as derived,
potentially QC/dedup-selected data.

A re-ingestion bundle contains:

```text
bundle/
  input_manifest.csv
  provenance.json
  reads/ or alignments/
  checksums.json
```

It preserves experiment namespace, molecule/template identity, sample/barcode,
pair layout, modality, filtering/deduplication state, trimming state, source
generation, and modification-signal capability.

Tag-dependent lossless export uses BAM, not FASTQ.

## Target ingestion architecture

```text
config + file/directory/user manifest/re-ingestion bundle
                             |
                             v
                  canonical input resolver
                             |
               resolved input manifest + digest
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
     raw signal         unaligned reads    existing alignment
     basecalling          align adapter       validation
          |                  |                  |
          +------------------+------------------+
                             |
                  owned alignment partitions
                  + alignment manifest
                             |
                  barcode/sample normalization
                             |
                     lossless segments
                             |
                molecule assembly/consensus
                             |
          immutable raw generation + source indexes
                             |
          preprocess -> spatial -> HMM -> latent -> project
```

The physical number of input files, alignment partitions, or extraction tasks
does not redefine scientific identity. It is recorded as source/artifact
provenance and may influence bounded execution planning only.

## Core schemas

Schemas are independently versioned. Exact field names may be refined in the
implementing PR, but semantic ownership must remain as defined here.

### Canonical input manifest

Required canonical row fields:

| Field | Meaning |
| --- | --- |
| `source_id` | Stable content/metadata-derived row identity |
| `path` | Resolvable source path recorded as provenance |
| `sha256` | Source content checksum |
| `size_bytes` | Source byte size |
| `source_kind` | POD5, FAST5, FASTQ, unaligned BAM, aligned BAM, or CRAM |
| `source_role` | Raw signal, reads, or alignment |
| `sample` | Explicit sample identity or null |
| `barcode` | Explicit barcode identity or null |
| `read_group` | Read-group identity or null |
| `pair_id` | File-pair identity or null |
| `mate` | R1, R2, or unpaired |
| `namespace` | Source/experiment namespace |
| `modification_capability` | Raw signal, MM/ML, conversion sequence, or sequence only |
| `trimmed` | True, false, or unknown |

The manifest-level metadata records schema version, ordered manifest digest,
resolution method, base directory, and warnings. Machine-local absolute path is
excluded from semantic identity.

### Alignment manifest

Required fields include:

- schema and adapter versions;
- source-manifest digest;
- reference-bundle digest;
- aligner/external tool name and version;
- normalized arguments;
- layout and partition list;
- each output BAM/BAI checksum;
- sort order and validation state;
- `@SQ` digest;
- primary/secondary/supplementary counts;
- pair/singleton/discordant counts;
- tag-capability summary; and
- relative artifact pointers.

### Segment and molecule schemas

Segment rows include:

- `segment_uid`;
- `molecule_uid`;
- original template/read ID;
- segment label;
- source ID;
- reference and reference-strand identity;
- reference start/end;
- CIGAR;
- mapping flags/quality;
- sequence, quality, mismatch, and SMF arrays; and
- barcode/sample provenance.

The molecule spine includes:

- one unique `molecule_uid`;
- original template/read ID;
- experiment identity;
- sample/barcode;
- segment count and pair state;
- reference/strand;
- outer fragment bounds;
- covered-base, overlap, and gap counts;
- discordance/conflict metrics; and
- pointers to segment and molecule-view shards.

### Re-ingestion bundle manifest

The bundle extends the canonical input manifest with:

- producing experiment/project identity;
- producing raw/preprocess generation;
- selection/QC/dedup policy;
- original and exported molecule IDs;
- collision-safe exported read/template IDs;
- pair layout;
- retained/lost tag capabilities; and
- provenance/checksum files.

## Configuration and migration contract

The exact public names must be finalized in IAR-01, but the initial shape is:

```yaml
input_data_path: /path/to/one/file/or/directory
input_manifest_path: null
alignment_mode: align
aligner: dorado
```

Rules:

- exactly one of `input_data_path` and `input_manifest_path` is authoritative;
- a bundle resolves through its included input manifest;
- existing configs without `alignment_mode` retain current alignment behavior;
- explicit `alignment_mode: existing` is required to consume mapped BAM/CRAM
  without realignment;
- `fastq_barcode_map` remains a compatibility input and is normalized into
  manifest declarations;
- `fastq_auto_pairing` remains during migration but delegates to the manifest
  resolver;
- deprecated or superseded flags receive one documented compatibility period;
- mixed directories and BAM directories fail immediately until their
  implementing PRs land;
- user-visible schema/default changes include migration notes and config tests.

## Delivery strategy

Use one focused branch/PR per item. Do not combine raw-generation migration,
paired molecule storage, new aligners, and append-only publication in one diff.

The primary dependency chain is:

```text
IAR-01 fail-fast vocabulary
    -> IAR-02 canonical input manifest
        -> IAR-03 raw source identity/intermediate ownership
            -> IAR-04 immutable raw generations

IAR-02 -> IAR-05 barcode/sample normalization

IAR-02 + IAR-03
    -> IAR-06 existing alignment ingestion

IAR-03 + IAR-06
    -> IAR-07 structured aligner adapters
        -> IAR-08 paired FASTQ/alignment contract

IAR-04 + IAR-05 + IAR-08
    -> IAR-09 molecule/segment raw storage
        -> IAR-10 paired molecule consensus

IAR-06 + IAR-09
    -> IAR-11 multi-BAM/CRAM source partitions

IAR-07 + IAR-08
    -> IAR-12 BWA-MEM2/Bowtie2 adapters

IAR-02 + IAR-04 + IAR-09
    -> IAR-13 re-ingestion/export bundles

IAR-04 + IAR-09 + IAR-11
    -> IAR-14 append-only raw growth

all core PRs
    -> IAR-15 workflow integration and acceptance
```

IAR-05 and IAR-06 can proceed in parallel after the manifest/source identity
foundation is stable. IAR-12 does not block paired consensus when minimap2 is
the first validated paired adapter. The project-CLI lane may proceed
independently.

## Ordered core PR backlog

| ID | Suggested branch | Primary outcome | Audit coverage | Depends on |
| --- | --- | --- | --- | --- |
| IAR-01 | `fix/input-contract-validation` | Fail-fast input/alignment vocabulary and current-bug guardrails | IAR-C1, IAR-M1, IAR-H6 | None |
| IAR-02 | `feature/canonical-input-manifest` | One deterministic manifest for files, directories, and user manifests | IAR-H2, IAR-M2, IAR-M5 | IAR-01 |
| IAR-03 | `feature/raw-source-artifact-identity` | Content-addressed raw source/intermediate compatibility | IAR-H2 | IAR-02 |
| IAR-04 | `feature/raw-immutable-generations` | Transactional raw publication and manifest-driven restart | IAR-H2 | IAR-03 |
| IAR-05 | `fix/barcode-sample-identity-contract` | Uniform precedence and sidecar publication | IAR-H3, IAR-M2 | IAR-02 |
| IAR-06 | `feature/existing-alignment-input` | Validate and ingest aligned BAM without realignment | IAR-H1, IAR-M3 | IAR-02, IAR-03 |
| IAR-07 | `feature/alignment-adapter-contract` | Structured Dorado/minimap2 adapters and alignment manifests | IAR-H6 | IAR-03, IAR-06 |
| IAR-08 | `feature/paired-fastq-alignment` | Common Illumina pairing and two-stream alignment | IAR-H4, IAR-M2 | IAR-02, IAR-07 |
| IAR-09 | `feature/raw-molecule-segments` | Lossless segment identity and one molecule spine row | IAR-C2 | IAR-04, IAR-05, IAR-08 |
| IAR-10 | `feature/paired-molecule-consensus` | Overlap consensus and uncovered-gap representation | IAR-C2, IAR-H4 | IAR-09 |
| IAR-11 | `feature/alignment-source-partitions` | Validated multi-BAM and existing CRAM ingestion | IAR-C1, IAR-M4 | IAR-06, IAR-09 |
| IAR-12 | `feature/bwa-bowtie-aligners` | BWA-MEM2 and Bowtie2 adapters | IAR-H6 | IAR-07, IAR-08 |
| IAR-13 | `feature/reingestion-export-bundles` | Typed sequence-only and lossless round-trip exports | IAR-H5, IAR-M3 | IAR-02, IAR-04, IAR-09 |
| IAR-14 | `feature/raw-append-generations` | Safe append-only source growth | IAR-H2 | IAR-04, IAR-09, IAR-11 |
| IAR-15 | `feature/input-alignment-acceptance` | Workflow manifest support, docs, and integrated acceptance | All core findings | IAR-01 through IAR-14 |

## IAR-01 — input contract validation and fail-fast guardrails

### Scope

- Add typed alignment-mode and source-role vocabulary to configuration.
- Validate supported aligner names before load execution.
- Reject mixed recognized input types with a message listing every type/count.
- Reject BAM directories with guidance that multi-BAM support is scheduled for
  IAR-11.
- Reject SAM/CRAM with explicit current support guidance rather than
  categorizing them as unrelated files.
- Reject impossible modality/capability combinations detectable from one file.
- Replace the unknown-aligner log-and-continue path with a raised,
  configuration-level error.
- Preserve homogeneous POD5/FAST5/FASTQ directory and single-file behavior.
- Document that existing BAM requires an explicit mode after IAR-06; before
  then, requesting it fails with a forward-compatible message.

### Primary files

- `src/smftools/config/discover_input_files.py`
- `src/smftools/config/experiment_config.py`
- `src/smftools/config/default.yaml`
- `src/smftools/cli/load_adata.py`
- `src/smftools/informatics/bam_functions.py`
- `tests/unit/config/test_LoadExperimentConfig.py`
- new focused input-discovery tests

### Required tests

- Every supported single file and homogeneous directory retains its type.
- Mixed POD5/FASTQ/BAM directories fail and list all recognized contents.
- BAM directory fails before executable checks or directory creation.
- Unknown aligner fails during config loading.
- Existing legacy BAM config still resolves to alignment mode.
- Unsupported direct FASTQ capability fails with actionable guidance only when
  lossless direct signal is required.
- Errors are deterministic across recursive and non-recursive discovery.

### Exit gate

No currently unsupported input collection or aligner reaches an external tool
or fails later as a missing intermediate artifact.

## IAR-02 — canonical input manifest

### Scope

- Define canonical input-manifest schema 1 and its validator.
- Accept a user CSV manifest and resolve a single file/directory into the same
  canonical row model.
- Stream SHA-256 calculation without loading large inputs into memory.
- Define deterministic ordering, row IDs, pair IDs, and manifest digest.
- Resolve relative user-manifest paths relative to the manifest location.
- Detect duplicate paths, duplicate content declarations, incomplete mates,
  conflicting namespaces, and conflicting metadata.
- Normalize `fastq_barcode_map` and auto-pair results into manifest fields.
- Add common Illumina/CASAVA filename parsing, but mark inferred metadata and
  never let inference override explicit declarations.
- Publish the resolved manifest and resolution report under the task-local raw
  staging area.
- Keep source paths outside the output tree read-only.

### Primary files

- new `src/smftools/informatics/input_manifest.py`
- `src/smftools/config/discover_input_files.py`
- `src/smftools/config/experiment_config.py`
- `src/smftools/cli/load_adata.py`
- `src/smftools/informatics/experiment_manifest.py`
- new `tests/unit/informatics/test_input_manifest.py`
- config and workflow-contract tests

### Required tests

- File, directory, and equivalent user manifest produce the same digest.
- Relocating a manifest plus its relative inputs preserves semantic identity.
- Changing file bytes at the same path changes source/manifest identity.
- Reordering user rows normalizes deterministically.
- Common `*_S1_L001_R1_001.fastq.gz` pairs resolve correctly.
- Multiple lanes for one declared sample remain distinct source rows.
- Ambiguous pair/file patterns require explicit metadata.
- Large-file hashing is streamed and bounded.
- Missing, unreadable, duplicate, and mutated-during-hash inputs fail cleanly.

### Exit gate

Every supported raw request has exactly one validated canonical manifest before
basecalling, alignment, demultiplexing, or raw artifact writes begin.

## IAR-03 — raw source artifact identity and intermediate ownership

### Scope

- Add the ordered input-manifest digest and source identities to raw semantic
  `NodeInputs.input_artifacts`.
- Include alignment-reference bundle identity in the raw source contract.
- Define versioned intermediate commit manifests for:
  - FAST5-to-POD5 conversion;
  - FASTQ-to-unaligned-BAM normalization;
  - basecalling;
  - alignment/sort/index;
  - barcode/UMI sidecars; and
  - direct-modification extraction.
- Replace output-exists reuse decisions with commit-manifest validation.
- Key reusable intermediate directories by semantic source/operation identity,
  not a fixed filename alone.
- Treat checksum mismatch, incomplete publication, and source mismatch as
  incompatible.
- Record external tool versions and normalized operation configuration.
- Ensure `force_redo_load_adata` creates an intentional execution revision
  without overwriting a compatible committed intermediate.

### Primary files

- `src/smftools/pipeline/experiment_graph.py`
- `src/smftools/cli/helpers.py`
- `src/smftools/cli/load_adata.py`
- `src/smftools/informatics/experiment_manifest.py`
- `src/smftools/informatics/sidecar_manifest.py`
- new raw-intermediate manifest helper
- semantic graph, raw lifecycle, and failure-injection tests

### Required tests

- Same manifest/config/reference produces compatible raw planning.
- Same path with changed bytes produces `stale_input`.
- Added, removed, and reordered sources produce the correct plan result.
- A fixed-name stale canonical BAM is never reused for a changed manifest.
- Missing/corrupt intermediate commit metadata forces rebuild.
- Tool/version provenance is recorded but invalidates only according to the
  declared operation compatibility policy.
- Force-redo never mutates a committed artifact.

### Exit gate

No raw intermediate is reused solely because its expected path exists, and raw
semantic planning explains source-content incompatibility.

## IAR-04 — immutable raw generations

### Scope

- Define raw generation schema and `current.json` pointer.
- Stage all owned raw outputs beneath one unique generation staging directory.
- Publish alignment artifacts, sidecars, segment/current ragged artifacts,
  molecule spine/indexes, reference catalogs, and generation manifest
  transactionally.
- Initially migrate the current one-record-per-read raw representation without
  changing scientific contents; IAR-09 extends the schema for segments.
- Validate every required artifact/checksum/pointer before current selection.
- Update experiment stage completion and consolidated spine only after current
  publication.
- Preserve prior complete generations on force-redo or replacement.
- Add compatibility readers for the current pre-generation raw layout when
  identity can be recovered; require explicit migration/recompute otherwise.
- Make downstream stage source selection resolve the selected raw generation.

### Primary files

- `src/smftools/cli/raw_adata.py`
- `src/smftools/cli/load_adata.py`
- `src/smftools/informatics/raw_store.py`
- `src/smftools/informatics/experiment_manifest.py`
- `src/smftools/informatics/experiment_spine.py`
- `src/smftools/cli/helpers.py`
- raw lifecycle, relocation, failure-injection, and downstream-source tests

### Required failure-injection tests

- Failure during basecalling/alignment normalization.
- Failure while writing a ragged shard or molecule index.
- Failure after spine creation but before generation commit.
- Failure after generation commit but before current-pointer swap.
- Corrupt artifact behind a nominally complete generation.
- Failed replacement leaves the prior current generation readable.
- Copied/relocated experiment tree resolves all owned artifacts.
- Legacy raw layout has deterministic compatibility or remediation behavior.

### Exit gate

Raw spine existence alone is never a completion signal. A partial or corrupt
raw generation is never selected by restart, downstream stages, consolidated
spine publication, or project registration.

## IAR-05 — barcode and sample identity normalization

### Scope

- Version one canonical barcode sidecar schema.
- Implement the agreed manifest > BAM metadata > sequence classification >
  filename fallback precedence.
- Normalize BC, RG ID, RG SM, Dorado bi/BM, and smftools classifier output into
  explicit source/value/confidence columns.
- Build the sidecar for every route, including
  `input_already_demuxed=True` plus `skip_bam_split=True`.
- Detect within-read and within-source conflicts.
- Record classified, unclassified, unknown, and conflicting counts/fractions.
- Keep split BAM generation optional and independent of metadata availability.
- Make raw metadata consume only the normalized sidecar contract.
- Preserve legacy filename behavior behind a warning and migration note.

### Primary files

- `src/smftools/cli/load_adata.py`
- `src/smftools/cli/raw_adata.py`
- `src/smftools/informatics/bam_functions.py`
- `src/smftools/informatics/barcode_sidecar.py`
- `src/smftools/informatics/sidecar_manifest.py`
- barcode extraction/labels/config tests

### Required tests

- Explicit manifest labels override compatible/incompatible lower authorities
  according to the declared conflict policy.
- Already-demultiplexed non-split BC-tagged BAM publishes correct labels.
- RG/SM-only input resolves sample identity.
- Dorado and smftools classification produce the same canonical columns.
- Unknown/unclassified rates are published and filter behavior is unchanged.
- Project-export names do not lose experiment namespace when a bundle manifest
  is present.
- Filename fallback warns and does not confuse R1/R2 with barcodes.

### Exit gate

Every raw molecule has a traceable barcode/sample authority, and no supported
non-split route depends on an absent sidecar.

## IAR-06 — existing aligned BAM input

### Scope

- Implement explicit `alignment_mode: existing` for one BAM.
- Validate BAM readability, header, sort order, index, primary alignments,
  sequence/quality/CIGAR availability, and reference compatibility.
- Validate converted/deaminase alignment-reference and interval-map identity.
- Validate paired flags and required tag capability without yet assembling
  mates.
- Normalize/copy the BAM and index into the raw generation without changing
  alignment placement.
- Record external aligner/tool provenance when available, and "unknown" as
  explicit provenance when not.
- Expose a stable prepared alignment-reference bundle/helper for users who must
  align conversion references externally.
- Reject direct-modification aligned input without valid MM/ML.
- Ensure no alignment executable is required or invoked in existing mode.

### Primary files

- new `src/smftools/informatics/alignment_manifest.py`
- new `src/smftools/informatics/alignment_validation.py`
- `src/smftools/cli/load_adata.py`
- `src/smftools/cli/raw_adata.py`
- `src/smftools/config/experiment_config.py`
- reference catalog/map helpers
- existing-alignment unit/integration tests

### Required tests

- Valid sorted/indexed aligned BAM is ingested without invoking an aligner.
- Unsorted input is normalized without changing alignment records.
- Missing index is created in staging, not beside the source.
- Reference name, length, checksum, and converted-reference mismatches fail.
- Source BAM remains byte-identical.
- MM/ML requirements are enforced for direct modality.
- BC/RG metadata survives normalization.
- Existing paired flags and mate fields survive normalization.
- Relocated completed output no longer depends on the source BAM.

### Exit gate

A user can intentionally provide one authoritative aligned BAM, receive a
validated owned raw alignment artifact, and prove that no realignment occurred.

## IAR-07 — structured alignment adapter contract

### Scope

- Define the adapter protocol and registry.
- Refactor Dorado and minimap2 execution behind adapters without changing
  supported scientific defaults.
- Give each adapter explicit single-end, paired-end, BAM, and tag-preservation
  capabilities.
- Move version probing, argv construction, reference index preparation,
  execution, sort/index, and provenance into the adapter boundary.
- Reject unsupported adapter/input combinations before execution.
- Publish alignment manifest schema 1 for adapter and existing-alignment routes.
- Key reference indexes by reference checksum, adapter/version, and semantic
  index parameters.
- Remove the generic unknown-aligner fallthrough.

### Primary files

- new `src/smftools/informatics/alignment_adapters/`
- `src/smftools/informatics/bam_functions.py`
- `src/smftools/cli/load_adata.py`
- `src/smftools/config/experiment_config.py`
- external-tool/version helpers
- adapter unit tests with subprocess fakes

### Required tests

- Dorado/minimap2 argv matches existing supported behavior.
- Arguments containing spaces/special characters remain distinct argv entries.
- Missing/unsupported executable versions fail before output staging.
- Adapter capability errors identify the source layout and remedy.
- Reference index cache identity changes only for semantic inputs.
- Failed aligner/sort/index publication leaves no committed alignment.
- Alignment manifest is deterministic and relocation-safe.

### Exit gate

Raw orchestration does not branch directly on aligner names, and every
alignment is produced or accepted through one validated manifest contract.

## IAR-08 — paired FASTQ and paired alignment contract

### Scope

- Extend manifest resolution for common Illumina/CASAVA filenames, lanes, and
  explicit R1/R2 declarations.
- Validate record-name synchronization and unequal file lengths before/while
  normalization.
- Preserve distinct R1/R2 streams through paired-capable adapters.
- Normalize template names without using the shared template name as segment
  identity.
- Preserve proper-pair, mate-reference, mate-position, orientation, and
  template-length fields.
- Treat orphan records as explicit singleton mates.
- Define discordant-pair validation/reporting without filtering by default.
- Prevent R1/R2 tokens from becoming inferred barcodes.
- Keep single-end FASTQ behavior unchanged.

### Primary files

- `src/smftools/informatics/input_manifest.py`
- `src/smftools/informatics/bam_functions.py`
- alignment adapters
- `src/smftools/cli/load_adata.py`
- paired FASTQ fixtures and focused tests

### Required tests

- CASAVA, `/1`/`/2`, `_R1`/`_R2`, multiple-lane, and explicit manifest pairs.
- Missing mate, duplicate mate, unsynchronized name, and unequal-length errors.
- Paired adapter receives two ordered streams.
- Proper-pair and singleton records survive sort/index.
- Barcode/sample identity is pair-consistent.
- Input record ordering does not change source or template identity.
- Existing paired BAM passes through the same normalized pair metadata.

### Exit gate

Paired input reaches raw extraction with lossless, validated mate identity and
alignment metadata; no intermediate step flattens it into ambiguous single-end
records.

## IAR-09 — molecule/segment raw storage

### Scope

- Version segment, molecule spine, ragged shard, and molecule-index schemas.
- Store each primary R1/R2/single/long-read alignment under unique
  `segment_uid`.
- Preserve shared `molecule_uid`/template identity separately.
- Publish one molecule spine row even when two segment rows exist.
- Retain per-segment sequence, quality, mismatch, signal, CIGAR, flags, and
  alignment metrics.
- Add molecule-to-segment and segment-to-shard indexes.
- Handle secondary/supplementary records as annotations or separate
  non-consensus segments according to existing primary-only policy.
- Update barcode, UMI, metrics, modkit/pysam signal, export, and project
  discovery joins to use the correct identity level.
- Provide a compatibility path in which legacy one-row reads become one-segment
  molecules without rewriting their scientific values.

### Primary files

- `src/smftools/informatics/ragged_store.py`
- `src/smftools/informatics/raw_store.py`
- `src/smftools/cli/raw_adata.py`
- `src/smftools/informatics/molecule_identity.py`
- `src/smftools/informatics/partition_read.py`
- export and project registry/catalog readers
- raw/ragged/identity/project tests

### Required tests

- One long read produces one molecule and one segment.
- One proper pair produces one molecule and two unique segments.
- Duplicate template names across experiments remain independent.
- Duplicate segment identity within an experiment fails.
- One molecule can be queried through both segment pointers.
- Legacy raw stores materialize identically as one-segment molecules.
- Relocation preserves molecule/segment pointers.
- Streaming remains bounded and rejects collisions across source partitions.

### Exit gate

Raw-store uniqueness is defined at the appropriate molecule and segment levels,
and a conventional paired BAM cannot fail merely because R1 and R2 share a
template name.

## IAR-10 — paired molecule consensus and gap representation

### Scope

- Add versioned conversion/deaminase pair-consensus planning.
- Orient both mates in reference coordinates before comparison.
- Materialize one molecule row from one or two primary segments.
- Publish `covered_base_mask`, `mate_coverage_count`, and
  `overlap_conflict_mask`.
- Leave uncovered insert positions as `NaN`/not covered.
- Apply the agreed conservative signal and sequence conflict policies.
- Calculate fragment, overlap, gap, conflict, singleton, and discordance
  metadata.
- Update downstream coverage, quality, mismatch, variant, duplicate, spatial,
  HMM, and plotting consumers to use covered-base semantics where required.
- Reject paired direct-modification consensus with precise guidance until a
  separate contract is implemented.

### Primary files

- `src/smftools/informatics/ragged_store.py`
- new paired consensus helper
- `src/smftools/informatics/partition_read.py`
- `src/smftools/preprocessing/`
- `src/smftools/plotting/`
- paired molecule and downstream compatibility tests

### Required tests

- Non-overlapping mates yield one molecule with `NaN` and false coverage in the
  intervening gap.
- Partial/full overlap yields one deterministic molecule row.
- Agreeing overlap calls remain one observation.
- Signal disagreement becomes `NaN` plus conflict mask.
- Sequence disagreement follows base-quality/tie policy.
- Insertions, deletions, soft clipping, reverse mate, and reference bounds.
- Singleton and discordant pairs retain explicit metadata.
- Downstream reducers never count two mates as two molecules.
- Read-span plots do not paint the uncovered insert as observed.

### Exit gate

Paired conversion/deaminase data has one scientifically correct molecule view,
with lossless source segments and explicit missing/conflicting positions.

## IAR-11 — multi-alignment source partitions and CRAM

### Scope

- Permit explicit manifests containing multiple compatible aligned BAM/CRAM
  sources.
- Treat each file as an owned source/alignment partition rather than building
  one mandatory monolithic merged BAM.
- Validate compatible reference bundles, modality, tag capability, barcode
  authority, and pair layout across partitions.
- Define template namespace/collision policy across lanes/files.
- Add CRAM reading/normalization with mandatory reference checksum validation.
- Extend streaming extraction, indexes, alignment manifests, and QC summaries
  across source partitions.
- Keep implicit BAM-directory ingestion disabled unless directory resolution
  can produce an unambiguous manifest.
- Reject cross-partition duplicate molecules unless the manifest explicitly
  identifies one logical source shard and the records are non-overlapping.

### Primary files

- input/alignment manifest and validation modules
- `src/smftools/cli/load_adata.py`
- `src/smftools/cli/raw_adata.py`
- `src/smftools/informatics/raw_store.py`
- BAM QC and sidecar helpers
- multi-source and CRAM tests

### Required tests

- Two BAM source partitions produce one valid experiment generation.
- Incompatible references/tags/modalities fail before extraction.
- Same template ID in distinct declared namespaces remains independent.
- Accidental duplicate molecule collision fails.
- CRAM succeeds only with the exact reference.
- Source order does not change semantic identity after canonical normalization.
- No extraction path passes a directory to an aligner/BAM reader.
- Memory remains bounded by one configured source/reference bucket.

### Exit gate

Multi-BAM/CRAM input has an explicit partitioned contract, and raw correctness
does not depend on concatenating every alignment into one file.

## IAR-12 — BWA-MEM2 and Bowtie2 adapters

### Scope

- Add BWA-MEM2 and Bowtie2 adapter implementations.
- Support single-end and paired-end FASTQ layouts.
- Build/reference content-addressed aligner indexes in owned cache locations.
- Record executable/version/index/argv provenance.
- Stream or stage SAM/BAM output into the shared sort/index/validation path.
- Validate required output tags/fields and declare tag-preservation limits.
- Reject direct MM/ML workflows that would lose modification tags.
- Add config defaults only where scientifically justified; do not change the
  existing default aligner in this PR.

### Primary files

- alignment adapter registry and new adapter modules
- `src/smftools/config/default.yaml`
- `src/smftools/config/experiment_config.py`
- container/dependency documentation if executables are packaged
- adapter integration fixtures/tests

### Required tests

- Single/paired argv and reference-index commands for both adapters.
- Executable absence/version reporting.
- Correct paired flags and shared alignment-manifest schema.
- Index reuse and invalidation by reference/adapter version.
- Paths with spaces remain safe.
- External-tool failure does not publish a generation.
- Tiny real-tool smoke tests where executables are available; deterministic
  fakes cover ordinary unit CI.

### Exit gate

Users can choose BWA-MEM2 or Bowtie2 through the same validated adapter contract
without changing downstream raw semantics.

## IAR-13 — re-ingestion and export bundles

### Scope

- Version bundle and provenance schemas.
- Preserve current FASTQ export as an explicitly sequence-only product.
- Add collision-safe template/read naming and identity map.
- Add experiment and project bundle export with canonical input manifest.
- Preserve paired layout and source namespace.
- Record QC/dedup selection, modality, trim state, source generation, and lost
  capabilities.
- Add BAM bundle export when barcode/RG/MM/ML/pair metadata must survive.
- Make experiment ingestion consume a bundle manifest directly.
- Reject a sequence-only bundle for direct-modification lossless analysis.
- Ensure project exports with duplicate bare IDs remain independently
  re-ingestible.

### Primary files

- `src/smftools/cli/export_fastq.py`
- `src/smftools/informatics/fastq_export.py`
- new bundle/export manifest helper
- `src/smftools/cli_entry.py`
- workflow contract and project registry readers
- export/re-ingestion round-trip tests

### Required tests

- Experiment FASTQ bundle -> fresh full raw generation preserves declared
  labels and molecule mapping.
- Project bundle with duplicate bare IDs remains collision-free.
- Paired bundle preserves R1/R2 and one molecule identity.
- BAM bundle preserves BC/RG/MM/ML and checksums.
- Sequence-only direct re-ingestion fails with an actionable capability error.
- Already filtered/deduplicated state is visible and never silently claimed as
  raw/unfiltered.
- Relocated bundle resolves relative paths and validates checksums.

### Exit gate

Every export declares whether it is sequence-only or lossless, and every
advertised re-ingestion path has an automated identity/capability round trip.

## IAR-14 — append-only raw generations

### Scope

- Add source-manifest transition classification.
- Permit incremental execution only for pure source additions.
- Reuse prior immutable source partitions, segments, molecule shards, and
  indexes by checksum/reference rather than copying mutable directories.
- Process only added sources and atomically publish a new complete raw
  generation.
- Rebuild aggregate molecule/index/catalog views deterministically.
- Detect collisions between added and existing template/segment identities.
- Propagate new raw channel fingerprints through preprocess/downstream semantic
  planning.
- Define append behavior for paired files: an incomplete pair is not a valid
  append; adding a mate to a previously published singleton is a mutation/full
  rebuild.
- Require full recompute for source removal, byte mutation, semantic metadata
  mutation, or reference change.
- Record reused/new artifact identities and counts.

### Primary files

- input manifest transition helper
- raw generation/lifecycle modules
- semantic experiment/preprocess graphs
- raw store/index publication
- downstream growth and failure-injection tests

### Required tests

- Identical manifest is compatible and does no work.
- Pure added FASTQ/BAM partition extracts only new content.
- Changed bytes at one existing path requires full recompute.
- Removed source requires full recompute.
- Added source with molecule collision fails before publication.
- Failed append leaves prior current generation intact.
- New current generation is complete after relocation.
- Downstream plan marks exactly the channels affected by added molecules.
- Fixed-name legacy intermediate files cannot mask the append.

### Exit gate

Adding supported source partitions is a transactional, explainable generation
transition; no append path mutates the current raw generation or reuses stale
intermediates.

## IAR-15 — workflow integration, documentation, and acceptance

### Scope

- Extend `experiment run` to accept one staged manifest or re-ingestion bundle
  while retaining read-only input staging and integrity checks.
- Keep arbitrary directory staging disallowed in workflow mode.
- Include canonical input/alignment/raw generation identities in
  `workflow_result.json`, versions, checksums, and validation.
- Update CLI help, configuration reference, lifecycle/migration, container, and
  basic usage documentation.
- Add an acceptance matrix mapping every audit finding and implementation item
  to automated coverage or an explicitly owned external-tool deferment.
- Add end-to-end profiles for:
  - single FASTQ;
  - FASTQ directory/manifest;
  - existing aligned BAM;
  - paired overlapping/non-overlapping Illumina;
  - multi-BAM partitions;
  - supported alternative aligners;
  - sequence-only and BAM re-ingestion bundles; and
  - append-only growth.
- Validate relocation and arbitrary container UID behavior for owned artifacts.
- Ensure stable legacy migration errors for unsupported prior patterns.

### Primary files

- `src/smftools/cli/workflow_contract.py`
- `src/smftools/cli_entry.py`
- result/version/checksum schemas
- `docs/source/`
- container smoke scripts/workflows
- acceptance catalogs and unit/integration/E2E tests

### Required verification

```text
venvs/venv-all/bin/python -m pytest -q <focused input/alignment/raw/export tests>
venvs/venv-all/bin/python -m pytest -m unit -q
venvs/venv-all/bin/python -m pytest -m integration -q
venvs/venv-all/bin/python -m pytest -m smoke -q
venvs/venv-all/bin/ruff check .
venvs/venv-all/bin/ruff format --check .
sphinx-build -W -b html docs/source docs/_build/html
```

Run applicable real-tool and container E2E tests outside the restricted unit
environment. Record absent external executables as explicit acceptance
deferments rather than weakening component assertions.

### Exit gate

All core audit findings have automated acceptance or an approved,
owner-assigned external validation, and the public docs describe actual input,
alignment, paired-molecule, export, and append behavior.

## Independent project-CLI completion lane

These findings came from the audit but are not dependencies of the raw-input
program. They should use separate branches and can proceed in parallel after
the current project semantic contracts are stable.

| ID | Suggested branch | Primary outcome | Audit coverage | Depends on |
| --- | --- | --- | --- | --- |
| PCLI-01 | `feature/project-set-commands` | Named set add/list/show/remove CLI | IAR-M6 | Existing registry set API |
| PCLI-02 | `feature/project-sample-analysis-workflow` | Plan/run/validate periodicity/sample-analysis products | IAR-M6 | Current semantic project graph |
| PCLI-03 | `feature/project-embedding-workflow` | Plan/run/validate shared embedding generations | IAR-M6 | Current embedding store |
| PCLI-04 | `feature/project-analysis-acceptance` | Unified project product help/docs/acceptance | IAR-M6 | PCLI-01 through PCLI-03 |

### PCLI-01 — named set commands

- Add set add/list/show/remove or deactivate commands without changing the
  registry's append-only experiment behavior.
- Resolve and display stable experiment membership.
- Validate missing/inactive/duplicate experiments.
- Ensure `--set` consumers use exactly the shown resolved membership.
- Add CLI/configuration documentation and focused registry/CLI tests.

Exit gate: users can create and inspect every named set used by project
planning/materialization without calling Python APIs.

### PCLI-02 — sample-analysis workflow

- Add a task-local `project sample-analysis run` interface or equivalent thin
  target selector.
- Use the existing semantic plan/result vocabulary and immutable sample-analysis
  artifacts.
- Add stable `workflow_result.json`, compatible skip, validation, failure, and
  source-change behavior.
- Do not overload generic genomic materialization output.

Exit gate: the `sample-analysis` target advertised by project planning has a
matching executable and validation path.

### PCLI-03 — project embedding workflow

- Add a task-local shared embedding execution interface.
- Preserve the existing selection definition, immutable generation, source
  fingerprint, append-growth, force-refit, and trusted-local model contracts.
- Make the model trust boundary explicit at the CLI.
- Add validation and result JSON without treating task-local latent coordinates
  as shared.

Exit gate: the `embedding` target advertised by project planning has a matching
safe executable and validation path.

### PCLI-04 — project analysis acceptance

- Unify help and documentation for selection, materialization,
  sample-analysis, and embedding targets.
- Prove plan/run/validate compatibility, relocation, source mutation, failure,
  force behavior, named-set selection, and duplicate bare read identity.
- Decide whether `project run` becomes a target-dispatch command or remains a
  materialization compatibility alias to explicit subcommands.

Exit gate: every public project plan target maps to one documented execution
and validation lifecycle.

## Schema and migration policy

- Version input manifest, alignment manifest, barcode sidecar, raw generation,
  segment, molecule spine/index, pair consensus, export bundle, append
  transition, result JSON, and versions schemas independently.
- Readers support older schemas only when identity and scientific meaning can
  be recovered without guessing.
- Legacy directory inference is normalized through the new manifest resolver.
- Legacy fixed intermediate filenames are never accepted as cache hits without
  compatible commit metadata.
- Legacy one-row raw records migrate as one-segment molecules.
- Existing mapped BAM behavior remains "align" unless the user explicitly
  selects existing-alignment mode.
- Mixed directories change from silent priority selection to a documented
  fail-fast migration.
- FASTQ filename barcode inference remains a warned fallback, not an
  authoritative new-manifest default.
- Old complete raw generations remain readable until a separately designed
  retention policy removes them.
- User-facing config/default/CLI changes receive migration notes and warnings
  where safe compatibility exists.

## Explicit non-goals

- Building a cluster/cloud workflow scheduler inside smftools.
- Accepting arbitrary shell command templates as aligners.
- Guessing whether a BAM should be realigned based on mapped flags.
- Treating absolute paths, mtimes, or output existence as scientific identity.
- Mutating a current raw generation or external source file in place.
- Combining incompatible references or modalities in one experiment.
- Making physical input partitioning or worker count part of molecule identity.
- Fabricating one gapped CIGAR to hide two paired-end segments.
- Treating paired mates as independent molecules.
- Filling an unsequenced paired-end insert with zeros or inferred SMF signal.
- Treating conflicting overlap calls as two observations.
- Reconstructing MM/ML probabilities from sequence-only FASTQ.
- Reattaching arbitrary lost BAM tags by bare query name in the initial
  program.
- Supporting paired direct-modification consensus without a separate validated
  probability model.
- Changing default QC/dedup filtering as part of ingestion refactoring.
- Rewriting completed SDV/PL identity and semantic graph contracts without a
  demonstrated defect.
- Bumping the package version on feature branches.

## Decision gates

Most core contracts are fixed above. The following choices must be resolved
before the named PR:

| Decision | Needed by |
| --- | --- |
| Exact public config names and compatibility-warning duration | IAR-01 |
| User manifest interchange format: CSV only versus CSV plus JSON metadata input | IAR-02 |
| Source checksum cache location and safe mutation detection during hashing | IAR-02 |
| Raw generation directory names and legacy pointer migration | IAR-04 |
| Exact BC versus RG/SM conflict reporting/filter policy | IAR-05 |
| Prepared conversion-reference CLI name and output bundle shape | IAR-06 |
| Minimum supported Dorado/minimap2 versions in adapter registry | IAR-07 |
| Default discordant/singleton pair keep/filter policy; recommended initial default is keep plus flag | IAR-08 |
| Segment storage encoding: normalized relational Parquet rows versus nested segment columns; recommended relational rows | IAR-09 |
| Whether legacy `read_span_mask` remains outer-span-only or becomes a deprecated alias; `covered_base_mask` remains authoritative either way | IAR-10 |
| CRAM publication retained as CRAM versus normalized owned BAM; recommended initial output is BAM | IAR-11 |
| Whether BWA means BWA-MEM2 only or also legacy BWA-MEM; recommended initial adapter is BWA-MEM2 only | IAR-12 |
| Bundle BAM granularity: one per experiment versus source/sample partitions | IAR-13 |
| Downstream nodes eligible for append-only extension versus conservative recompute | IAR-14 |
| Project `run` target dispatch versus explicit subcommand hierarchy | PCLI-04 |

## Program completion definition

The core IAR program is complete when:

1. Every experiment raw request resolves to one validated canonical input
   manifest.
2. Mixed/unsupported/ambiguous inputs fail before external execution.
3. Raw semantic planning identifies source contents and exact alignment
   reference artifacts.
4. Intermediate and raw generation reuse is manifest-driven, transactional,
   immutable, and relocation-safe.
5. Existing aligned BAM input can be validated and consumed without
   realignment.
6. Barcode/sample identity has one precedence and sidecar contract.
7. Dorado/minimap2 use structured adapters and BWA-MEM2/Bowtie2 are available
   through the same interface.
8. Paired FASTQ layout survives alignment with lossless segment identity.
9. One paired template becomes one molecule with conservative overlap
   consensus and missing uncovered insert positions.
10. Multi-BAM/CRAM source collections use validated source partitions.
11. Sequence-only and lossless exports are distinguished, validated, and
    re-ingestible according to declared capability.
12. Append-only growth creates a new complete raw generation and never reuses a
    stale fixed-name intermediate.
13. Workflow result/validation/version/checksum contracts include input,
    alignment, and raw generation identities.
14. Focused, unit, integration, smoke, failure-injection, relocation,
    external-tool, container, lint, format, and documentation gates pass or
    carry explicit owner-approved deferments.

The project-CLI lane is complete when every advertised project plan target has
matching documented execution/validation behavior and named sets are fully
manageable through the CLI.

## Implementation status

Update this ledger as the program proceeds:

- `Planned`: scope is defined but no implementation branch has started.
- `In progress`: record the branch and base `main` commit.
- `Implemented`: record focused/full verification and the feature commit.
- `Merged`: record the PR number and merge commit from `main`.
- `Blocked`: record the repeated external or design blocker and the decision
  needed; do not use this state merely for incomplete work.

Add a dated implementation record beneath the tables for each completed item,
following the predecessor plans. Do not mark an item merged until its merge is
visible on `main`.

### Core ingestion/alignment program

| ID | Status | Branch | Notes |
| --- | --- | --- | --- |
| IAR-01 | Merged | `fix/input-contract-validation` | PR #468; feature `3ad10a8`; merge `2b67189` |
| IAR-02 | Merged | `feature/canonical-input-manifest` | PR #469; feature `df80b65`; merge `acbbb0b` |
| IAR-03 | Merged | `feature/raw-source-artifact-identity` | PR #470; feature `ec60392`; merge `2e72a3b` |
| IAR-04 | Merged | `feature/raw-immutable-generations` | PR #472; feature `5db6d98`; merge `823c9d4` |
| IAR-05 | Merged | `fix/barcode-sample-identity-contract` | PR #473; feature `c8892b3`; merge `d7ac4f8` |
| IAR-06 | Merged | `feature/existing-alignment-input` | PR #474; feature `f555886`; CI fix `a12ba47`; merge `726b786` |
| IAR-07 | Merged | `feature/alignment-adapter-contract` | PR #475; feature `1c3d3a4`; merge `d961eba` |
| IAR-08 | Merged | `feature/paired-fastq-alignment` | PR #476; feature `0b1302b`; merge `24ddbf9` |
| IAR-09 | Merged | `feature/raw-molecule-segments` | PR #477; feature `81af8c2`; merge `8f8e869` |
| IAR-10 | Merged | `feature/paired-molecule-consensus` | PR #478; feature `d65b6d8`; merge `df42dfd` |
| IAR-11 | Merged | `feature/alignment-source-partitions` | PR #479; feature `422138e`; merge `c7c2010` |
| IAR-12 | Merged | `feature/bwa-bowtie-aligners` | PR #480; feature `8f558d9`; merge `7467659` |
| IAR-13 | Merged | `feature/reingestion-export-bundles` | PR #481; feature `aec916f`; merge `2c7c936` |
| IAR-14 | Merged | `feature/raw-append-generations` | PR #482; feature `39ec0e8`; merge `b60037f` |
| IAR-15 | In progress | split into three branches | See the 2026-08-12 IAR-15 record below |

### 2026-08-10 — IAR-01 implementation record

- Started `fix/input-contract-validation` from `origin/main` at `9edc5ac`.
- Added the backward-compatible `alignment_mode: align` vocabulary and derived input source roles.
- Added fail-fast mixed-input, BAM-directory, SAM/CRAM, direct-FASTQ, and aligner validation.
- Added focused configuration/discovery tests and migration documentation.
- Verification: 76 configuration tests, 106 smoke tests, focused raw process-pool coverage,
  Ruff check/format, and `sphinx-build -W` pass. The full unit run passed 1,600 tests before four
  unrelated failures exposed a stale canonical environment missing declared dependency `skops`;
  after refreshing `venv-all`, all nine tests in the affected ML module pass.
- Committed as `3ad10a8` and merged to `main` in PR #468 at merge commit `2b67189`.

### 2026-08-10 — IAR-02 progress record

- Started `feature/canonical-input-manifest` from `origin/main` at `2b67189` with no configured
  upstream branch.
- Added CSV schema 1 validation, deterministic content/metadata source identities, streamed
  SHA-256 with mutation detection and a task-local SQLite checksum cache, relative-path
  resolution, duplicate and metadata conflict detection, and relocation-invariant digests.
- Normalized FASTQ barcode overrides and explicit/common CASAVA pair metadata into canonical rows;
  ambiguous and incomplete pairs now fail before external execution.
- Integrated manifest-backed configs, workflow runtime staging, raw-task publication, experiment
  provenance, and mandatory raw-stage CSV/JSON/resolution-report artifacts.
- Verification: Ruff repository check, 120 focused manifest/config/workflow/raw/graph tests, the
  complete unit suite (1,621 passed, 9 skipped, 7 xfailed), and warning-strict Sphinx build pass.
- Committed as `df80b65` and merged to `main` in PR #469 at merge commit `acbbb0b`.

### 2026-08-10 — IAR-03 progress record

- Started `feature/raw-source-artifact-identity` from `origin/main` at `acbbb0b` with no configured
  upstream branch.
- Added ordered input-manifest/source checksums and the alignment-reference bundle to raw planning;
  source or reference byte changes now explain `stale_input`, while relocation and manifest row
  order remain compatible.
- Added schema-versioned intermediate commits with checksum validation, semantic operation keys,
  strict/provenance-only tool-version policies, atomic commit publication, and immutable
  force-redo revisions.
- Wired validated revision reuse into FAST5-to-POD5 conversion, FASTQ-to-BAM normalization,
  Dorado basecalling, and alignment/sort/index; legacy fixed-name files no longer authorize reuse.
- Applied the same commit contract to UMI sidecars, smftools and Dorado barcode sidecars, and raw
  direct-modification BED/TSV extraction. Committed alignment BAMs and indexes remain immutable
  even when optional demultiplexed-BAM cleanup is requested.
- Verification: repository-wide Ruff check, focused manifest/graph/raw tests, the complete unit
  suite (1,630 passed, 9 skipped, 7 xfailed), and warning-strict Sphinx build pass. The initial
  sandboxed unit run's 20 macOS semaphore failures all pass in the unsandboxed full run.
- Committed as `ec60392`.
- Merged to `main` in PR #470 at merge commit `2e72a3b`.
- The production CPU smoke then exposed non-UTF-8 bytes in Debian Samtools version metadata. Started
  `fix/raw-intermediate-version-decoding` from `2e72a3b` without an upstream and aligned the new
  intermediate version probe with the workflow collector's tolerant UTF-8 decoding. Verification:
  31 focused workflow/container tests and 106 smoke tests pass; repository-wide Ruff passes. The
  exact Docker smoke could not run locally because the Docker daemon is unavailable. Committed the
  follow-up as `81100f1` and merged it in PR #471 at merge commit `81d0f1b`.

### 2026-08-10 — IAR-04 progress record

- Started `feature/raw-immutable-generations` from `origin/main` at `81d0f1b` with no configured
  upstream branch.
- Added schema-versioned raw generation manifests, generation-root-relative artifact records,
  checksummed run-root dependencies, unique staging directories, validated atomic publication,
  and an atomic `current.json` selector with rollback to the prior complete generation.
- Snapshotted the raw spine, ragged store, normalized obs/molecule artifacts and indexes, reference
  and region catalogs, sidecar manifest, optional barcode index, and canonical input-manifest
  artifacts into each immutable generation. Content-addressed IAR-03 alignment and annotation
  intermediates remain shared immutable dependencies and are checksum-validated before selection.
- Changed raw restart, downstream path resolution, consolidated-spine generation, lifecycle
  provenance, and project registration/catalog discovery to select only the validated current raw
  generation. Stage completion and consolidated-spine refresh now happen after generation
  publication; failed replacements retain a validated `previous_complete` lifecycle record so
  restart and project discovery continue to expose the still-current prior generation.
- Added deterministic migration of lifecycle-compatible legacy raw layouts, while corrupt current
  selectors fail closed for downstream consumers and remain recoverable by an explicit raw rebuild.
- Added relocation, corruption, unsafe-pointer, incomplete-manifest, staging failure, pointer-swap
  failure, post-swap rollback, force-redo failure, legacy migration, downstream selection,
  consolidated-spine, and project-discovery coverage.
- Verification: 1,641 unit tests passed (9 skipped, 7 xfailed), 106 smoke tests passed (1 skipped),
  repository-wide Ruff check and touched-file format check passed, and warning-strict Sphinx build
  passed. The two process-pool tests that cannot initialize macOS semaphores inside the sandbox
  passed in the unsandboxed full unit run.
- Committed as `5db6d98` and merged to `main` in PR #472 at merge commit `823c9d4`.

### 2026-08-10 — IAR-05 progress record

- Started `fix/barcode-sample-identity-contract` from `origin/main` at `823c9d4` with no configured
  upstream branch.
- Added canonical barcode/sample identity schema 1 with manifest, BAM `BC`/`RG`/`SM`, configured
  sequence-classifier, and warned legacy filename authorities; selected source/confidence and
  lower-authority conflicts remain explicit per read.
- Published canonical Parquet and validation-report intermediates on every load route, registered
  both in the sidecar manifest, and made dense and partitioned raw metadata consume the normalized
  contract while preserving legacy sidecar reads and `skip_unclassified` behavior.
- Preserved declared FASTQ barcode/sample/read-group metadata in generated BAM `BC`, `RG`, and
  `@RG SM` fields, without misreporting internally generated source IDs as barcode authorities.
- Added classified, unclassified, unknown, and conflicting counts/fractions; retained experiment
  namespace in molecule grouping; documented the filename-fallback migration.
- Verification: 1,659 unit tests passed (9 skipped, 7 xfailed), 106 smoke tests passed (1 skipped),
  repository-wide Ruff check/format and warning-strict Sphinx build passed.
- Committed as `c8892b3`.
- Merged to `main` in PR #473 at merge commit `d7ac4f8`.

### 2026-08-11 — IAR-06 progress record

- Started `feature/existing-alignment-input` from `origin/main` at `d7ac4f8` with no configured
  upstream branch.
- Enabled explicit single-BAM `alignment_mode: existing` resolution while retaining legacy BAM
  realignment under `alignment_mode: align`; existing mode neither probes nor invokes an aligner.
- Added read-only source validation for exact prepared-reference records, coordinate order,
  primary sequence/quality/CIGAR fields, coherent MM/ML direct signal, paired-flag structure, and
  bounded external program provenance with explicit `unknown` fallback.
- Added content-addressed copy-or-sort normalization into an owned BAM/BAI pair, plus a relocatable,
  checksummed alignment manifest registered as a raw-generation dependency. Source BAM bytes and
  source-adjacent indexes remain untouched.
- Added a content-identified prepared alignment-reference helper that mirrors alignment-region
  reduction and conversion transforms and rejects invalid BED coordinates before extraction.
- Documented existing-alignment configuration, validation, migration, provenance, prepared-reference
  use, and the fail-fast paired-input boundary pending molecule-segment ingestion.
- Verification: 1,682 unit tests passed (9 skipped, 7 xfailed), 106 smoke tests passed (1 skipped),
  70 focused config/manifest/alignment/integration tests passed, repository-wide Ruff check/format
  passed, and warning-strict Sphinx build passed.
- Container CI then exposed an inherited IAR-04 path-resolution defect: materialization derived the
  run root from the nested raw-generation data directory, so it could not discover preprocess
  catalogs and misrouted `nan0_0minus1` to the raw ragged store. The follow-up now resolves the run
  root from generation-aware spine/source paths and adds an immutable-generation derived-layer
  regression. Verification: 89 affected raw-generation/preprocess/materialization/spatial tests,
  106 smoke tests, and repository-wide Ruff check/format pass.
- Committed the existing-alignment implementation as `f555886` and the container-CI raw-generation
  materialization correction as `a12ba47`.
- Merged to `main` in PR #474 at merge commit `726b786`.

### 2026-08-11 — IAR-07 progress record

- Started `feature/alignment-adapter-contract` from `origin/main` at `726b786` with no configured
  upstream branch.
- Added the authoritative Dorado/minimap2 adapter registry, typed capability and execution
  contracts, shell-free argv construction, fail-fast version gates, owned sort/index execution,
  partial-output cleanup, and semantic in-memory reference-index identities. Minimum supported
  versions are Dorado 0.7.0, minimap2 2.24.0, and external samtools 1.10.0.
- Raw orchestration now probes the selected adapter before task-output creation, rejects paired or
  lossy direct-modification routes before alignment staging, and commits generated BAM/BAI output
  only with a validated schema-1 alignment manifest. Existing and generated routes now share the
  same deterministic, relocation-safe manifest reader contract.
- Corrected the production direct-BAM smoke fixture to use `alignment_mode=existing`; its input is
  already coordinate-aligned and MM/ML-tagged, so minimap2 BAM-to-FASTQ realignment would be lossy.
  Added actual minimap2 execution coverage for the generated, sequence-only route.
- Verification: 1,692 unit tests passed (9 skipped, 7 xfailed); 106 smoke tests passed (1 skipped);
  45 focused adapter/config/existing-alignment tests passed; repository-wide Ruff check/format,
  `git diff --check`, and warning-strict Sphinx documentation build passed. The first sandboxed
  unit run had 20 process-pool failures because macOS semaphore syscalls were denied; the same full
  suite passed with normal process permissions.
- Committed as `1c3d3a4`.
- Merged to `main` in PR #475 at merge commit `d961eba`.

### 2026-08-11 — IAR-08 progress record

- Started `feature/paired-fastq-alignment` from `origin/main` at `d961eba` with no configured
  upstream branch.
- Implemented strict paired-FASTQ synchronization for `/1`/`/2`, `_R1`/`_R2`, and CASAVA mate
  annotations, including unequal-length, conflicting metadata, and partial-output failures.
- Added a minimap2 two-stream paired adapter route that preserves BC/RG comments and validates
  real proper-pair output, reciprocal mate placement, orientation, and template length.
- Added distinct raw segment identities (`template/1`, `template/2`) while retaining shared
  template identity and pair-state metadata. Metrics, BAM tags, barcode identity, and raw
  extraction now use the segment identity consistently.
- Existing paired BAM validation now accepts proper pairs and explicit singleton/discordant
  records, reports their counts without filtering, and rejects malformed mate fields.
- Verification: 1,705 unit tests passed (9 skipped, 7 xfailed); 106 smoke tests passed (1 skipped);
  44 focused paired/validation/raw-extraction tests passed; real minimap2 paired alignment passed;
  repository-wide Ruff check/format, `git diff --check`, and warning-strict Sphinx documentation
  build passed. The first sandboxed unit run had 20 process-pool failures because macOS semaphore
  syscalls were denied; the same full suite passed with normal process permissions.
- Committed as `0b1302b`; the branch has no configured upstream.
- Merged to `main` in PR #476 at merge commit `24ddbf9`.

### 2026-08-11 — IAR-09 progress record

- Started `feature/raw-molecule-segments` from `origin/main` at `24ddbf9` with no configured
  upstream branch.
- Added schema-v4 raw storage with stable `segment_uid`, shared template-based `molecule_uid`,
  one-row molecule spines, lossless segment catalogs, molecule-to-segment indexes, and
  segment-to-ragged-shard indexes. Existing molecule UID hashes and single-read scientific values
  remain stable.
- Published segment artifacts through immutable raw generations and made raw/dense relocation,
  derived indexes, project discovery, project export batching, and cross-experiment normalization
  identity-level aware. Primary-only extraction is now recorded as an explicit alignment-segment
  policy. New raw generations use schema 2; schema-1 generations remain valid without the new
  segment artifacts.
- Added paired-template materialization through segment pointers ahead of IAR-10 consensus, while
  retaining direct one-segment materialization for legacy/single-read stores. Proper pairs now
  produce one spine row and two independently queryable, collision-checked segment rows.
- Verification: 1,712 unit tests passed (9 skipped, 7 xfailed); 106 smoke tests passed (1 skipped);
  43 focused raw/query/generation/identity tests passed; repository-wide Ruff check/format,
  `git diff --check`, and warning-strict Sphinx documentation build passed. The first sandboxed
  unit run had multiprocessing failures because macOS semaphore syscalls were denied; the first
  unrestricted run then exposed six acceptance-fixture compatibility failures, which were fixed
  before the final green run.
- Committed as `81af8c2`; the branch has no configured upstream.
- Merged to `main` in PR #477 at merge commit `8f8e869`.

### 2026-08-11 — IAR-10 progress record

- Started `feature/paired-molecule-consensus` from `origin/main` at `8f8e869` with no configured
  upstream branch.
- Added versioned conversion/deaminase pair consensus over lossless segment shards, producing one
  molecule row across shard boundaries while retaining singleton state and rejecting discordant
  pairs and paired direct-modification inputs before scientific output.
- Published authoritative `covered_base_mask`, `mate_coverage_count`, and
  `overlap_conflict_mask` layers. Uncovered insert gaps remain missing; signal conflicts remain
  missing and are flagged; sequence conflicts use base quality with equal-quality ties becoming
  `N`.
- Recorded overlap, gap, conflict, singleton, and algorithm metadata and made mismatch-frequency,
  variant, and preprocessing coverage plots prefer observed-base coverage over the legacy outer
  span.
- Added paired overlap/gap/conflict/singleton/direct-rejection and cross-shard tests. Ruff check and
  format passed; 1,718 unit tests passed (9 skipped, 7 expected failures), 106 smoke tests passed
  (1 skipped), 29 focused preprocessing/spatial tests passed, and the warning-strict Sphinx build
  passed.
- Committed as `d65b6d8`; the branch has no configured upstream.
- Merged to `main` in PR #478 at merge commit `df42dfd`.

### 2026-08-11 — IAR-11 progress record

- Started `feature/alignment-source-partitions` from `origin/main` at `df42dfd` with no configured
  upstream branch.
- Added explicit-manifest-only multi-alignment admission for BAM, CRAM, and compatible mixed
  BAM/CRAM partitions while keeping implicit alignment-directory and multi-path ingestion
  disabled.
- Added exact CRAM reference-sequence MD5 validation and owned CRAM-to-BAM normalization. Every
  partition is validated against the same reference, modality/tag policy, and pair layout before
  extraction.
- Added a bounded SQLite-backed cross-partition template collision check. Repeated template names
  fail within one namespace and remain independent across explicitly distinct namespaces.
- Normalized each source into its own owned BAM/BAI/manifest, published per-source barcode
  authority, and streamed the canonical partition order without concatenating BAMs. Namespaced
  source-local IDs retain the original QNAME in `source_read_id` and receive collision-safe raw
  storage identities.
- The initial partition route requires already-demultiplexed inputs, `skip_bam_split=True`, no UMI
  annotation or BAM-to-BED generation, and the pysam backend for direct signal; unsupported
  processing routes fail before normalization.
- Added a real two-BAM end-to-end raw-generation test plus manifest, CRAM, collision, configuration,
  and streaming identity coverage. Ruff check/format passed; 1,729 unit tests passed (9 skipped,
  7 expected failures), 106 smoke tests passed (1 skipped), the partition E2E passed, and the
  warning-strict Sphinx build passed.
- Committed as `422138e`; the branch has no configured upstream.
- Merged to `main` in PR #479 at merge commit `c7c2010`.

### 2026-08-11 — IAR-12 progress record

- Started `feature/bwa-bowtie-aligners` from `origin/main` at `c7c2010` with no configured
  upstream branch.
- Added BWA-MEM2 and Bowtie2 adapters for single-end and paired-end sequence alignment through
  the shared adapter, sorting, indexing, validation, and provenance contracts. The existing
  default aligner remains unchanged.
- Added atomically published, content-addressed native reference-index caches. Cache identities
  bind the reference checksum, aligner and index-builder versions, adapter version, and index
  parameters; checksum manifests prevent reuse of incomplete or altered indexes.
- Added executable and minimum-version validation, stable argument construction, managed-option
  rejection, explicit BAM/MM/ML loss rejection, and documented tag-preservation limits.
- Added unit coverage for command arguments, paired layouts, version and executable failures,
  cache reuse/invalidation, paths containing spaces, and failed-index publication, plus
  conditional real-tool integration coverage for both aligners.
- Updated installation, container, API, and experiment-configuration documentation, including
  the partitioned alignment input behavior delivered by IAR-11.
- Ruff check and format passed; 1,746 unit tests passed (9 skipped, 7 expected failures), 106
  smoke tests passed (1 skipped), focused adapter/config/integration coverage passed (65 passed,
  4 real-tool tests skipped), and the warning-strict Sphinx build passed.
- Committed as `8f558d9`; the branch has no configured upstream.
- Merged to `main` in PR #480 at merge commit `7467659`.

### 2026-08-11 — IAR-13 progress record

- Started `feature/reingestion-export-bundles` from `origin/main` at `7467659` with no configured
  upstream branch.
- Added schema-versioned, checksummed sequence-only FASTQ and lossless BAM export bundles with an
  authoritative completion manifest, canonical relative-path input manifest, collision-safe
  exported identities, and a source-to-export identity map.
- Experiment and project exports now preserve declared sample/barcode/source namespaces, paired
  R1/R2 layout, selection/QC/dedup state, modality, trim state, source raw generation, and retained
  or lost capabilities. Project exports namespace duplicate bare source IDs safely.
- BAM bundles filter the owned alignment sources to selected reads while retaining coordinate
  alignment, pair flags, read groups, BC tags, and MM/ML direct-modification signal; direct bundles
  fail if any selected primary record lacks the required MM/ML capability.
- Canonical input resolution now accepts a relocated bundle manifest directly, validates all
  advertised sizes and SHA-256 checksums, and rejects sequence-only bundles when lossless direct
  signal is required. Legacy `export-fastq` remains available and now emits the explicit bundle
  contract alongside its compatibility manifest.
- Added unit coverage for experiment/project selection, duplicate identities, paired layout,
  relocation and tamper detection, direct capability rejection, and lossless BAM tags/checksums,
  plus a real-minimap2 export-to-fresh-raw-generation round trip.
- Verification: 1,752 unit tests passed (9 skipped, 7 expected failures); 106 smoke tests passed
  (1 skipped); 34 focused export/input-manifest tests and the real-minimap2 round trip passed;
  focused Ruff check/format, `git diff --check`, and the warning-strict Sphinx build passed.
- Committed as `aec916f`; the branch has no configured upstream.
- Merged to `main` in PR #481 at merge commit `2c7c936`.

### 2026-08-11 — IAR-14 progress record

- Started `feature/raw-append-generations` from `origin/main` at `2c7c936` with no configured
  upstream branch.
- Added deterministic canonical-manifest transition classification for identical, append-only,
  removed, replaced, content-mutated, and metadata-mutated source sets. Complete new FASTQ pairs
  can append; pair completion by changing an existing declaration remains a full recompute.
- Raw execution now freezes the full pre-execution config/source identity, processes only source
  IDs added by a pure append, and restores full-manifest provenance afterward. Raw config,
  alignment-reference, removal, byte, or semantic metadata changes retain the full-recompute path.
- Added transactional append assembly over the selected immutable generation. Prior and added raw
  shards are combined with collision-safe paths, aggregate molecule/segment/catalog/index views
  are rebuilt, reference plans are recomputed, and molecule or segment identity collisions fail
  before publication.
- New generations checksum-match unchanged files against the prior immutable generation and
  hardlink them into the new generation; mutable canonical paths never authorize reuse. Generation
  manifests validate and record the source transition, reused/added IDs, prior generation, and
  reused/new file and byte counts.
- Added source-growth semantic-plan coverage plus real existing-BAM and minimap2 FASTQ append
  round trips. Both routes retain the prior generation, process one added source, publish a
  complete relocated generation, and propagate the changed raw channel to dependent preprocessing.
- Verification: 1,760 unit tests passed (9 skipped, 7 expected failures); 106 smoke tests passed
  (1 skipped); 121 focused transition/raw-store/publication/planning/real-tool tests passed;
  repository-wide Ruff check/format, `git diff --check`, and warning-strict Sphinx build passed.
- Committed as `39ec0e8`; the branch has no configured upstream.

### 2026-08-12 — IAR-01 through IAR-14 pre-IAR-15 implementation audit

Re-ran every declared gate against `main` at `b60037f`. The ledger's verification claims are
accurate: 1,760 unit tests passed (9 skipped, 7 xfailed), 49 integration, 106 smoke,
repository-wide Ruff check/format clean, warning-strict Sphinx build succeeded. The **e2e lane was
never covered by any per-item verification record** and has 3 failures.

Design review of the highest-risk paths found the core invariants sound. `InputManifestRow.identity()`
excludes `path`/`source_id`/`inferred_fields` but retains `sha256`/`size_bytes`, so `source_id` is
content-inclusive and `APPEND_ONLY` is only reachable when every prior source kept identical bytes
and metadata — in-place mutation correctly reaches `CONTENT_MUTATED`, relocation correctly reaches
`IDENTICAL`. Adapters are genuinely shell-free with checked exit codes and no stderr deadlock.
Append assembly hardlinks the prior generation only into disposable staging, copies mutable
canonical shards, and rejects identity collisions before publication. No TODO/FIXME markers and no
swallowed exceptions across the ~12.7k inserted lines. Two apparent defects were investigated and
dismissed: the mixed `run_root`/`raw_root` reads in `raw_append.py` correctly mirror
`raw_store.py:699-728`, and the equal-quality consensus tie resolving to neither mate is documented
intent (`ragged_store.py:644`) flagged via `overlap_conflict_mask`.

**External-tool provisioning (2026-08-12):** the machine now resolves and executes dorado (models
symlinked from `~/dorado_models`), minimap2, samtools, modkit, `bwa-mem2` 2.2.1 (miniforge), and
`bowtie2` 2.5.5 (homebrew), with the Docker daemon up. This removed every tool-gated skip: the
combined integration+e2e run went from 59 passed / 7 skipped to **63 passed / 2 skipped**, and all
four `test_short_read_adapter_executes_with_native_index` cases (bwa-mem2 and bowtie2 × single and
paired) now execute for real. IAR-07/IAR-08/IAR-12 therefore have genuine external-tool coverage
rather than deferments. The 2 remaining skips are not tool-gated: an absent `statsmodels` package and
a Linux-only cgroup check.

Separate packaging bug found while inventorying: `statsmodels` is imported at
`tools/position_stats.py:68` but is declared in **no** `pyproject.toml` extra, so
`calculate_relative_risk_on_activity` fails for anyone who lacks it and `[all_2]` does not supply it.
It also bypasses the `require()` helper from `optional_imports`. Out of IAR scope; fix separately.

Findings, in severity order:

- **D1 (merged, PR #483; feature `70fc0cb`; merge `cbe7912`)** — `canoncall`/`modcall`
  ignored dorado's exit status, and `commit_intermediate` validated checksums without rejecting
  zero-byte outputs, so a failed basecall was published `state: complete` and reused permanently.
  Belongs to IAR-03's exit gate: checksum equality cannot establish artifact validity. Fixed at both
  layers, with validation-side rejection so already-poisoned caches heal.
- **D2 (partly open, fold into IAR-15)** — the 3 `test_load_adata_e2e` params fail rather than skip
  when `tests/_test_inputs/dorado_models` is absent. Resolved locally on 2026-08-12 by symlinking
  that gitignored path to the user's existing `~/dorado_models`; the full e2e lane then passed
  10/10, confirming the 3 failures were purely a fixture gap and not an IAR-01..14 regression. The
  missing **skip guard** is still open and still belongs to IAR-15's requirement to record absent
  external tools as explicit deferments — the IAR-14 FASTQ append test already does this correctly
  via `shutil.which("minimap2")`.
- **D3 + D6 (merged into one fix, commit `8a416ef`)** — investigating D3 uncovered a larger
  pre-existing bug (D6). `_sanitize_uns` keeps JSON-serializable values verbatim but stringifies
  anything else, and AnnData reads stored string lists back as numpy arrays; since
  `_bind_generation_spine` rereads and rewrites the spine on **every** publication (not just
  appends), `ragged_store` and `signal_columns` degraded into string reprs like `"['mod_a' 'mod_b']"`
  (numpy's repr, no commas). Readers already compensated defensively (`partition_read.py:253-265`,
  `partition_store.py:650-657`) so nothing broke functionally, but stored artifacts were wrong and
  every reader had to know to un-degrade them. Fixed by adding `uns_string_list` /
  `normalize_uns_string_lists` to `readwrite` (accepting list, array, and both string-repr forms for
  backward compatibility) and normalizing before all three spine writes. `signal_columns` is now
  also unioned across an append rather than taken from the added subset.
- **D4 (withdrawn — the finding was wrong)** — obs column sets legitimately differ between
  generations: optional columns such as `source_read_id` only appear for namespaced sources, so the
  union with null fill is the correct semantic, not silent drift. An equality check was implemented
  and then reverted after both append e2e round trips rejected it on `source_read_id`. A comment now
  documents this at `raw_append.py` so it is not "fixed" again.
- **D5 (fixed, commit `8a416ef`)** — hoisted the position axis, outer-fragment bounds, and three
  summary column offsets out of `collapse_paired_segments`' per-molecule loop.

Coverage gap that let D3/D6 through: the append e2e asserts row counts and reuse stats but never
`uns` fidelity. Now covered by focused tests in `test_raw_generation.py`, `test_raw_append.py`, and
`test_readwrite.py`. The IAR-15 acceptance matrix should close this class mechanically rather than by
inspection — note that two of the five findings only became visible by *running* the code, and one
(D4) was disproved by it.

### 2026-08-12 — IAR-15 split and progress record

IAR-15 as written bundles workflow code, docs, and a test matrix into one item, which would be a
large PR against this repo's small-focused-PR convention. Split into three branches:

| Part | Branch | Status |
| --- | --- | --- |
| Workflow bundle input, aligner versions, identity threading | `feature/workflow-bundle-input` | **Merged** PR #484; feature `65facfe`; merge `6111841` |
| Acceptance matrix, 8 e2e profiles, relocation/container-UID, skip guards (D2) | `feature/input-alignment-acceptance` | **In progress**, 2 of 6 items done; commits `1717b59`, `afd5578` |
| CLI help, config reference, lifecycle/migration, container, usage docs | `docs/input-alignment-reference` | Planned, written last so it describes shipped behavior |

Grounding the first part against the code changed its scope substantially. Prior PRs had already
satisfied more of IAR-15 than the scope list implies:

- **Bundle re-ingestion already worked** end to end. `_read_csv_declarations`
  (`input_manifest.py:427-432`) detects a `.json` manifest and routes it through
  `resolve_bundle_input_manifest`, and `resolve_input_manifest` resolves the manifest path before
  anchoring relative rows (`:461`, `:777`). So the workflow's symlink staging
  (`workflow_contract._stage_readonly_alias`) preserves bundle-relative artifact resolution. Verified
  empirically, then locked in with a test rather than rebuilt. The config-layer round trip was
  already proven by `test_sequence_export_bundle_reingests_as_fresh_raw_generation`.
- **Raw generation ids and input-manifest checksums already reach `workflow_result.json`** via the
  per-stage manifest entries that IAR-02/IAR-04 publish (`_collect_artifacts`, `:643`). No change
  needed.
- **Arbitrary directory staging was already rejected** (`_stage_readonly_alias` requires a concrete
  file; covered by `test_workflow_staging_rejects_directory_inputs`).

Two genuine gaps, both fixed in `65facfe`:

- `_required_external_tools` recognized only dorado and minimap2 as aligners, so a bwa-mem2 or
  bowtie2 run recorded **no aligner version** and strict mode never checked for the binary. Now
  covers all four, including `bowtie2-build` (a separate binary the bowtie2 adapter shells out to for
  its native index). `bwa-mem2` needs a `version` subcommand, not `--version`. The new test asserts
  every required tool has a version command — the invariant that was actually violated.
- `sources` entries now carry bundle kind, scope, modality, lost capabilities, and source generation
  ids. A file fingerprint cannot express that re-ingesting a `sequence_only` bundle drops
  direct-modification capability. Optional field, so the result schema version is unchanged.

### 2026-08-13 — BOTH LANES COMPLETE (read this first)

Every item in this plan is implemented, tested, and merged: IAR-01 through IAR-15 for the
input/alignment program, and PCLI-01 through PCLI-04 for the independent project-CLI lane recorded
at the end of this file. Nothing in either lane is outstanding.

The coverage records are `tests/acceptance/input_alignment_criteria.json` and
`tests/acceptance/project_cli_criteria.json`, each validated by a test that resolves every cited
test back to a real symbol. Read those rather than this file's older prose, whose scope lists
overstate what remained.

One follow-up is open rather than merged: `fix/statsmodels-optional-dependency` (see "Separate
known bug" below), which is pushed with a PR awaiting merge.

#### Where the work landed

| Merge | Contents |
| --- | --- |
| PRs #468–#482 | IAR-01 through IAR-14 |
| PR #484 | IAR-15 part 1: workflow bundle input, bundle provenance, aligner version coverage |
| PR #485 | IAR-15 part 2a: D2 skip guards, aligner e2e profile, FASTQ directory + paired Illumina profiles |
| PR #486 | IAR-15 part 2b: lossless-BAM round trip, relocation/foreign-UID validation, killed-run restart recovery, acceptance matrix |
| PR #487 | Ledger D3/D5/D6: uns string-list fidelity, signal-column union, consensus loop hoist, and the catalog entries flipped to automated |
| `docs/input-alignment-lifecycle` (pushed, PR open) | IAR-15 part 3: restart/immutability/bundle-re-ingestion documentation |

#### The record of what is covered

`tests/acceptance/input_alignment_criteria.json` is now the authoritative map from every audit
finding, implementation item, audit acceptance scenario, and ledger finding to the test that covers
it. `tests/unit/test_input_alignment_acceptance_catalog.py` resolves every cited test back to a real
symbol, so the map cannot rot silently. Read that file rather than reconstructing coverage from this
plan's prose. Two entries are not automated, both deliberately:

- `finding.iar_m6` — the project-CLI lane below, out of scope for this program.
- `item.iar_15` — the documentation half; flip it to `automated` once the docs PR merges, or leave
  it, since documentation is not test-covered either way. `ledger.d4` is recorded as `withdrawn`
  with the reasoning, so the disproved finding is not rediscovered and "fixed" again.

#### Environment (provisioned on this machine — verify, do not re-install)

All external tools resolve and execute: `dorado` 1.3.1, `minimap2`, `samtools`, `modkit`,
`bwa-mem2` 2.2.1 (miniforge), `bowtie2` 2.5.5 + `bowtie2-build` (homebrew), Docker daemon up.
`tests/_test_inputs/dorado_models` is a **symlink** to `~/dorado_models` (gitignored at
`.gitignore:29`, so it is invisible to git and must be recreated if the checkout is replaced). Use
`venvs/venv-all/bin/python`.

Green baseline: **1,787 unit** (9 skipped, 7 xfailed), **53 integration** (2 skipped), **106 smoke**
(1 skipped), **18 e2e** (1 skipped), Ruff check+format clean, warning-strict Sphinx build succeeds.
If e2e is 15 rather than 18, the dorado model symlink is missing.

#### Behavior worth knowing before touching this area again

- **A killed run is resumable.** A process that dies mid-stage leaves the stage record in `running`;
  the next attempt supersedes it (`superseded_attempt`) and reuses the retained complete generation
  (`restored_from_previous_complete`). Before PR #486 this state was fatal and permanent — every
  later invocation died with `invalid stage transition for 'raw': 'running' -> 'planned'`.
- **Immutability is detect-not-prevent.** Published artifacts keep their write bit on purpose, so an
  arbitrary container UID can manage the tree; corruption is caught by checksum at selection time.
  Do not "harden" this by chmod-ing published generations read-only.
- **Do not re-add an obs schema-equality check to `_combine_spines`** (see `ledger.d4`).
- **Bundle kinds are not interchangeable.** `lossless_bam` requires `alignment_mode: existing`;
  under `align` it is rejected on the source-role conflict.

#### Separate known bug — fixed, PR open

`statsmodels` was imported at `tools/position_stats.py:68` but declared in **no** `pyproject.toml`
extra, so `calculate_relative_risk_on_activity` failed for anyone who lacked it and `[all]`/`[all_2]`
did not supply it. It also bypassed the `require()` helper. Worse than recorded: the package was
absent from `venvs/venv-all` too, so `tests/unit/test_calculate_relative_risk_on_activity.py` had
been skipping via `importorskip` rather than ever running.

Fixed on `fix/statsmodels-optional-dependency` (pushed, PR awaiting merge): statsmodels joins the
`analysis` extra beside the other analysis dependencies, and the import routes through `require()`
so the failure names the extra to install. With it installed those 3 tests run and pass.

#### Method notes that paid off

Ground each item against the code before building it, and prefer a throwaway probe script in the
scratchpad over reasoning about behavior. Three of the four items in the final session turned up a
defect that only appeared once the behavior was exercised for real, and one earlier finding (D4) was
disproved by running the code. Probe scripts that touch the pipeline need an
`if __name__ == "__main__":` guard — the process pool uses spawn on macOS, and without it the child
re-executes the script's module level. Never `git add -A` right after a Sphinx build; that is now
prevented by `.gitignore`, but the build also writes `docs/source/api/generated/`.

### Project-CLI lane — COMPLETE (2026-08-13)

| ID | Status | Merge | Notes |
| --- | --- | --- | --- |
| PCLI-01 | Done | PR #489 | `project add-set/list-sets/show-set/remove-set`; one shared set resolver so shown membership is applied membership |
| PCLI-02 | Done | PR #490 | `project sample-analysis` under the workflow contract; also fixed `run --experiment` never reaching materialization |
| PCLI-03 | Done | PR #491 | `project embedding` with the model trust boundary explicit at the CLI |
| PCLI-04 | Done | branch `feature/project-analysis-acceptance` (pushed, PR open) | `project run --target` dispatch + acceptance catalog |

The coverage record is `tests/acceptance/project_cli_criteria.json`, validated by
`tests/unit/test_project_cli_acceptance_catalog.py`. Read it rather than re-deriving what is
proven. Every plan target now maps to one execution and validation lifecycle; `selection` is
planned-but-not-executable by design, and a test pins that plan targets minus run targets is
exactly `{selection}`.

Bugs found by exercising these paths, all fixed with regression tests: `project run --experiment X`
published a request naming a subset while pooling every experiment; a per-sample partition with no
analyzable read crashed on a numpy index dtype instead of contributing no rows; and embedding
growth counters read from the in-memory fit result reported every extension as adding zero
molecules.

**One deliberate deferral**, recorded as `gap.shared_experiment_identity` in the catalog: two
registered experiment ids can carry the same `experiment_uid`, colliding their molecule identities
project-wide. Production cannot reach it (the identity is keyed on the run root, and each pipeline
run owns one), but a hand-assembled tree can. A registration guard was written and reverted here
because the existing test fixtures write sibling raw stores under one parent — exactly the shape it
rejects — so closing this means reworking those fixtures first. That is the natural next change in
this area.

