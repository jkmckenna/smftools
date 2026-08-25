# Experiment input, alignment, and round-trip audit

> **Repository state reviewed:** not recorded, and not recoverable — this document
> carries neither a commit nor a date. Its companion plan
> (`completed/input_ingestion_alignment_implementation_plan.md`) landed as PRs
> #468–#493, so the audit predates those. Treat every claim as needing
> re-verification.

## Status and scope

This repository-local audit records the planning-time behavior and identified gaps at the boundary
between experiment inputs and the semantic experiment/project pipelines. Implementation progress
and landed corrections are tracked in the companion implementation plan ledger.

The reviewed surface includes:

- project commands introduced or strengthened by the semantic DAG work;
- FASTQ, BAM, POD5, and FAST5 input discovery;
- barcode and sample identity propagation;
- `smftools experiment full` raw ingestion;
- experiment and project FASTQ export;
- append and round-trip behavior;
- paired-end Illumina reads;
- existing aligned BAMs and alternative aligners.

The primary implementation paths are:

- `src/smftools/cli_entry.py`
- `src/smftools/config/discover_input_files.py`
- `src/smftools/config/experiment_config.py`
- `src/smftools/cli/load_adata.py`
- `src/smftools/cli/raw_adata.py`
- `src/smftools/informatics/bam_functions.py`
- `src/smftools/informatics/ragged_store.py`
- `src/smftools/informatics/raw_store.py`
- `src/smftools/pipeline/experiment_graph.py`
- `src/smftools/cli/export_fastq.py`
- `src/smftools/informatics/fastq_export.py`
- `src/smftools/cli/workflow_contract.py`

## Executive summary

The project-level semantic lifecycle is strongest around experiment
registration, compatibility planning, cross-experiment materialization,
validation, and artifact export. The experiment pipeline is also well
integrated after a valid raw store exists:

```text
raw -> preprocess -> spatial -> HMM -> latent
```

The largest remaining gaps are at the raw-input normalization boundary:

- a homogeneous FASTQ directory is combined into one canonical unaligned BAM;
- a directory of BAMs is discovered but is not consumed as a collection;
- an input BAM is treated as sequence input to align again, even when it is
  already aligned;
- only Dorado and minimap2 have executable alignment branches;
- barcode identity has multiple configuration-dependent sources and no single
  documented precedence contract;
- paired FASTQ records are marked as mates but are not assembled into one
  molecule downstream;
- exported FASTQs are valid derived inputs but are not lossless,
  provenance-preserving, or append-safe round-trip bundles;
- raw restart behavior still relies on existence-based intermediate BAM caches,
  so adding input files to an already-run directory is not a reliable append
  operation.

Until these boundaries are formalized, users should treat a changed input set
or a FASTQ export as a new experiment with a fresh output root.

## Current project command surface

The current `smftools project` commands fall into four groups:

| Area | Commands | Current role |
| --- | --- | --- |
| Project lifecycle | `init`, `add`, `remove`, `list` | Scaffold a project, register experiments by pointer, deactivate registrations, and inspect references |
| Semantic lifecycle | `plan`, `run`, `validate` | Explain compatibility, execute task-local project materialization, and validate result/source identity |
| Cross-experiment artifacts | `materialize`, `export-latent`, `export-fastq` | Pool genomic data, preserve task-local latent coordinate ownership, and export QC-selected reads |
| Sample-store inspection | `sample-store-list` | List cataloged sample/reference partitions |

### Strengths

- Registration points to experiment artifacts instead of copying them.
- Canonical references are harmonized using sequence identity.
- Duplicate bare instrument read IDs remain independently addressable across
  registered experiments through experiment and molecule identities.
- Materialization supports reference, experiment, named-set, modality, stage,
  genomic-window, layer, and read-metric selection.
- Large selections can be written as bounded partitioned Zarr output instead
  of one pooled in-memory AnnData.
- `project run` publishes a stable `workflow_result.json` and supports
  compatible reuse.
- `project validate` checks artifact integrity and compares the stored source
  plan with the current project state.
- Experiment-local latent coordinate spaces are not silently pooled.

### Remaining CLI gaps

- `project plan` accepts `sample-analysis` and `embedding` targets, while
  `project run` executes materialization only.
- Periodicity/sample analysis and shared project embeddings are available as
  Python APIs but not as equivalent workflow CLI commands.
- Named sets can be resolved and mutated through the registry API, and commands
  accept `--set`, but there are no public set add/list/remove commands.
- Planning, execution, and validation are not yet one unified lifecycle across
  materialization, sample analysis, and shared embedding products.

## Input discovery

`ExperimentConfig` accepts a file or directory through `input_data_path`.
Recognized inputs include POD5, FAST5, compressed or uncompressed FASTQ, BAM,
and H5AD.

For a directory containing more than one recognized type, the current
selection order is:

```text
POD5 > FAST5 > FASTQ > BAM > H5AD
```

The first present type wins. Lower-priority recognized files are reported in
discovery counts but are not rejected as a mixed input and are not processed.

| Input | Current behavior | Assessment |
| --- | --- | --- |
| Single POD5 | Basecall, align, and continue | Supported for configured ONT workflows |
| FAST5 file/directory | Convert to one POD5 artifact, then continue | Supported |
| Single FASTQ | Convert to canonical unaligned BAM, then align | Supported with identity caveats |
| Homogeneous FASTQ directory | Recursively discover and combine FASTQs into one canonical unaligned BAM | Supported as a fresh input set |
| Single BAM | Treat as an unaligned/basecalled source and align again | Partially supported |
| BAM directory | Discover all BAMs, but later pass the directory path as the BAM input | Not correctly supported |
| Existing aligned BAM | Align again unless an undocumented internal output path is pre-populated | No first-class support |
| SAM or CRAM | Not recognized by input discovery | Unsupported |
| Mixed-type directory | Select the highest-priority type and ignore the others | Ambiguous and unsafe |

The workflow-oriented `smftools experiment run --input` contract is stricter:
it requires one concrete staged file and explicitly rejects directory inputs.
This is reproducible, but it does not currently offer a manifest or bundle
alternative for multi-file FASTQ input.

## FASTQ normalization and identity

All discovered FASTQs are passed to `concatenate_fastqs_to_bam`, producing one
`canonical_basecalls.bam`. The BAM records receive a `BC` tag and, by default,
an `RG` tag.

### Barcode inference

Barcode assignment uses:

1. an explicit `fastq_barcode_map`, when provided;
2. otherwise, the final underscore-delimited token of the FASTQ filename;
3. otherwise, the filename stem.

FASTQ header metadata is not parsed as a sample/barcode source.

Examples of current filename inference:

| Filename | Inferred barcode |
| --- | --- |
| `bc01.fastq.gz` | `bc01` |
| `sample_bc01.fastq.gz` | `bc01` |
| `expA__bc01.fastq.gz` | `bc01` |
| `treatment_A.fastq.gz` | `A` |
| `sample_bc01_R1.fastq.gz` | `R1` |

The last two examples show why filename inference is not a sufficient sample
identity contract. In particular, project-export namespaces and common
paired-end suffixes can be discarded or misinterpreted.

### Alignment after FASTQ conversion

FASTQ input is not passed directly to an aligner. It is first represented as
an unaligned BAM and then handled by `align_and_sort_BAM`.

For minimap2 with `align_from_bam=False`, the BAM is converted back to a single
FASTQ file. That conversion drops BAM auxiliary tags and does not preserve a
general paired-file contract. With `align_from_bam=True`, the BAM is passed to
minimap2, but this is not an explicit existing-alignment mode.

## BAM input

### Single BAM

A single BAM avoids basecalling but does not avoid alignment. The input path
becomes `unaligned_output`, and smftools derives a separate
`*_aligned_sorted.bam`. If that derived output does not exist, alignment runs.

This behavior has several implications:

- an authoritative external alignment is not used as-is;
- minimap2's BAM-to-FASTQ path can discard BC, RG, MM/ML, and other tags;
- already-correct pairing, primary/secondary choices, and mapping qualities can
  be replaced;
- direct-modification signal may be lost when the chosen alignment path does
  not preserve MM/ML tags.

The following existing options do not mean "skip alignment":

| Option | Actual role |
| --- | --- |
| `input_already_demuxed` | Suppress barcode classification |
| `skip_bam_split` | Avoid writing per-barcode split BAMs |
| `align_from_bam` | Select minimap2's BAM input path instead of converting BAM to FASTQ |

There is an incidental internal cache behavior: placing an external BAM at the
exact derived `*_aligned_sorted.bam` path causes the loader to skip alignment.
This depends on private output naming, does not establish proper provenance,
and should not be documented as a supported workflow.

### BAM directory

Directory discovery sets `input_type="bam"` and records all matching
`input_files`. The load path does not merge, iterate, or select those paths.
It instead uses `input_data_path`, which remains the directory, as
`unaligned_output`. The aligner will therefore receive a directory where it
expects one BAM.

Multi-BAM input needs an explicit policy:

- reject it with a clear error;
- merge compatible BAMs while preserving headers and tags;
- or treat each BAM as a source partition and assemble one experiment store
  from validated partitions.

Silently discovering the files without implementing one of these policies is
misleading.

## Barcode and sample metadata

Barcode identity currently comes from several distinct mechanisms:

- an explicit FASTQ path-to-barcode map;
- FASTQ filename inference;
- BAM `BC` tags;
- BAM `RG`/read-group metadata;
- Dorado `BC` and per-end `bi` tags;
- Dorado reclassification from barcode sequence;
- smftools barcode extraction using a configured barcode kit;
- split BAM filenames and a derived barcode sidecar.

There is no single public precedence rule covering all of these sources.

The partitioned raw metadata path expects a barcode sidecar. If a `BC` column
is present in that sidecar, it becomes `barcode`; without a barcode column,
the fallback is `barcode="unknown"`.

One important inconsistent case is:

```yaml
input_already_demuxed: true
skip_bam_split: true
```

The split-BAM path can reconstruct a sidecar from split files, but the
non-split already-demultiplexed path does not establish the same sidecar.
Consequently, an input BAM may retain reads while losing usable sample labels
in the partitioned raw store.

A future precedence contract should be explicit and validated:

```text
input manifest/map
  > validated BAM BC/RG/SM metadata
  > configured sequence classification
  > filename fallback
```

Unknown and unclassified fractions should be reported as validation results,
not discovered only in downstream plots.

## `experiment full`

After successful raw ingestion, the semantic experiment graph is well
integrated:

```text
raw -> preprocess -> spatial -> HMM -> latent
```

`full_run_latent` can stop the configured full workflow after HMM. The semantic
graph propagates incompatibility through downstream stage dependencies and
checks stage completion records.

The main lifecycle weakness is the raw source boundary:

- the semantic raw node has no input artifact identity analogous to downstream
  stage source artifacts;
- raw compatibility includes resolved configuration, including input paths and
  discovered file lists, but does not checksum every source file's contents;
- replacing a source at the same path is not reliably detected;
- adding a FASTQ can change the resolved config while existing
  `canonical_basecalls.bam` and aligned BAMs are still reused based on path
  existence;
- `force_redo_load_adata` bypasses the outer raw-stage reuse decision but does
  not consistently invalidate every inner intermediate cache.

Therefore, adding files to an input directory is not a reliable incremental
append operation.

## FASTQ export and re-ingestion

Experiment export writes one FASTQ per selected sample/barcode. Project export
namespaces groups by experiment:

```text
<experiment_id>__<barcode>.fastq.gz
```

Sequence and quality are read directly from raw ragged shards in query
coordinates. By default, exports use the most appropriate available
QC/deduplication pass set. `allow_unfiltered` can export all raw reads.

Each record contains only:

```text
@read_id
sequence
+
quality
```

The accompanying `fastq_manifest.csv` contains:

```text
barcode,n_reads,path
```

### What works

- The output is valid FASTQ.
- A FASTQ export can be supplied to a new conversion/deaminase experiment with
  a fresh output root.
- Project filenames avoid collisions between barcode filenames from different
  registered experiments.

### What is not preserved

- experiment identity in the FASTQ header;
- `experiment_uid` and `molecule_uid`;
- sample/barcode metadata other than the filename and separate manifest;
- BAM BC/RG/MM/ML and other auxiliary tags;
- original alignment and reference assignment;
- variant evidence and other derived metadata;
- direct-modification probabilities;
- whether physical barcode bases were trimmed;
- whether the reads were already filtered or deduplicated.

The input loader does not consume `fastq_manifest.csv`. Automatic filename
inference also reduces `expA__bc01.fastq.gz` back to `bc01`, losing the project
namespace. If multiple project experiments contain the same bare read name,
re-ingesting all exports into one new experiment can violate the raw store's
experiment-global unique `read_id` requirement.

An exported FASTQ directory is therefore a derived fresh input, not a lossless
round-trip or an append bundle.

Direct-modification experiments require special treatment: ordinary FASTQ
cannot represent MM/ML modification probabilities. A tag-preserving BAM export
or the original signal/basecalled modified BAM is required for faithful
re-analysis.

## Paired-end Illumina input

### Desired scientific representation

Two mates from one sequenced template should be one molecular observation.
When they do not overlap, the unsequenced insert should remain missing:

```text
R1 coverage       ##########
R2 coverage                    ##########
molecule          ##########....##########
                              ^
                         missing / NaN
```

When they overlap, the shared positions should be reconciled into one
consensus:

```text
R1 coverage       ###############
R2 coverage             ###############
merged molecule   #####################
                         ^
                    overlap consensus
```

The expected merge rules are:

- R1-only position: use the R1 call;
- R2-only position: use the R2 call;
- neither mate: leave signal `NaN` and coverage false;
- both agree: retain one consensus call;
- both disagree: use a documented quality-aware rule or mark the position
  ambiguous/missing;
- never count overlapping mates as two independent molecules.

### Current FASTQ pairing behavior

The FASTQ converter can auto-pair limited filename patterns ending in R1/R2,
read1/read2, or 1/2. It normalizes `/1` and `/2` suffixes and writes separate
BAM records with:

- one shared query name;
- paired, read1, or read2 flags;
- mate initially marked unmapped;
- the same inferred barcode and read group.

Common Illumina names such as `sample_S1_L001_R1_001.fastq.gz` are not covered
reliably by the current end-anchored patterns. The fallback barcode heuristic
can also interpret `R1` as the barcode.

### Alignment limitation

The canonical paired BAM is not routed through a dedicated paired-end aligner
contract. In the normal minimap2 path it is converted into one FASTQ stream.
Separate mate streams, pairing expectations, and fragment reconstruction are
not explicit. Dorado is the default aligner but is not a general paired-end
Illumina alignment interface.

### Raw-store limitation

After alignment, raw extraction emits one ragged record per primary alignment
using `read.query_name` as `read_id`. Both mates normally have the same query
name. There is no mate-assembly step before validation.

The ragged and raw stores require:

- unique `read_id` within each frame;
- experiment-global unique `read_id` across streamed frames.

Thus a conventional pair is likely to fail as a duplicate ID. Giving mates
different IDs avoids the collision but incorrectly treats them as independent
molecules.

### Why a synthetic gapped CIGAR is insufficient

The ragged materializer already initializes SMF signal to `NaN`, so missing
positions are representable. However, one current ragged row owns one
alignment/CIGAR.

Representing two mates as a fabricated CIGAR such as `150M300N150M` would:

- obscure original mate provenance;
- complicate reverse-orientation and overlap consensus;
- lose paired flags and per-mate mapping quality;
- cause the current `read_span_mask` to mark the complete outer span,
  including the unsequenced `N` gap.

The correct model is a molecule with one or more alignment segments.

### Proposed molecule/segment model

```text
molecule
  molecule_id / template_id
  experiment_uid
  barcode / sample
  pair status
  reference
  fragment bounds

segments
  molecule_id
  segment_id (R1 or R2)
  reference_start
  CIGAR
  sequence
  quality
  SMF calls
```

Materialization would group segments by molecule and scatter both mates into
one reference-grid row.

Recommended layers include:

- merged SMF signal in `X`, with `NaN` when unobserved;
- `covered_base_mask`, true only where a mate supplies an aligned base;
- `mate_coverage_count`, with values 0, 1, or 2;
- `overlap_conflict_mask`;
- consensus sequence and quality layers.

Recommended molecule metadata includes:

- `paired` and `proper_pair`;
- R1/R2 reference starts and ends;
- fragment start/end and insert size;
- overlap and uncovered-gap lengths;
- overlap conflict count;
- singleton-mate and discordant-pair flags.

Pairs mapping to different references, unexpected orientations, or implausible
insert sizes should be retained as explicitly discordant or excluded through a
configured policy. A mapped mate with an unmapped partner can remain as a
single-segment molecule.

For conversion/deaminase SMF, mate calls should be interpreted in reference
coordinates before overlap consensus. An overlapping disagreement is one
uncertain observation, not two independent pieces of biological evidence.

## Existing alignments and alternative aligners

### Current aligner implementation

The aligner argument resolver can select arguments by aligner and sequencer,
but the executable wrapper has only two branches:

```text
minimap2
dorado
```

Configuring BWA, BWA-MEM2, Bowtie2, or another aligner reaches the
unknown-aligner branch and does not create the expected output. The later load
path can then fail on a missing intermediate rather than producing one clear
configuration error.

### Needed separation of concerns

Input format and alignment policy should be separate:

```yaml
input_type: fastq
alignment_mode: align
aligner: bwa-mem2
```

For an existing alignment:

```yaml
input_type: aligned_bam
alignment_mode: validate_existing
input_data_path: sample.aligned.sorted.bam
fasta: reference.fasta
```

Suggested alignment modes are:

| Mode | Behavior |
| --- | --- |
| `align` | Run a configured built-in aligner |
| `use_existing` | Trust a supplied alignment after minimum structural checks |
| `validate_existing` | Perform full compatibility checks, then consume without realignment |
| `external` | Consume an alignment plus a portable alignment manifest |

### Existing-alignment validation

Before accepting an alignment, smftools should verify:

- BAM/CRAM readability;
- coordinate sort order;
- presence or creation of an index;
- `@SQ` reference names and lengths;
- exact reference FASTA checksum;
- compatibility with converted-reference and interval-map conventions;
- valid primary alignments, query sequences, CIGARs, and qualities;
- molecule/segment identity rules;
- paired flags, mate references/coordinates, and fragment consistency;
- barcode/sample metadata source;
- required MM/ML tags for direct-modification analysis;
- preservation or explicit absence of optional move/current-signal tags.

SAM can be normalized to BAM. CRAM requires the matching reference and should
not be accepted without reference validation.

### Aligner adapters

Built-in aligners should use structured adapters rather than arbitrary shell
templates. An adapter should define:

- supported input layouts, including paired FASTQ streams;
- executable/version probing;
- argv construction without invoking a shell;
- reference indexing requirements;
- streaming or temporary output behavior;
- sort/index publication;
- tag-preservation expectations;
- a normalized alignment manifest.

Candidate adapters include Dorado, minimap2, BWA-MEM2, and Bowtie2. An
`external` adapter should validate a supplied result rather than execute an
unconstrained command.

For direct-modification BAMs, tag preservation is part of scientific
correctness. Converting a BAM to FASTQ before BWA/Bowtie2/minimap2 discards
MM/ML; either the aligner path must preserve/re-attach those tags by validated
read identity, or smftools must consume the externally aligned tagged BAM
without realignment.

Adding a paired-aware aligner does not alone fix paired-end SMF. Mate assembly
into one molecule remains a separate required step after alignment and before
raw-store publication.

## Proposed input manifest

A first-class input manifest should replace directory inference as the
authoritative multi-file contract. A conceptual row schema is:

| Field | Purpose |
| --- | --- |
| `path` | Source file path |
| `sha256` | Immutable source identity |
| `input_type` | POD5, FAST5, FASTQ, unaligned BAM, aligned BAM, or CRAM |
| `sample` | Stable sample identity |
| `barcode` | Stable barcode identity |
| `experiment_namespace` | Preserve project/export source identity |
| `pair_id` | Shared template/file-pair identity |
| `mate` | R1, R2, or unpaired |
| `read_group` | Desired or expected RG identity |
| `alignment_mode` | Align or use/validate existing |
| `modification_signal` | Whether MM/ML or raw signal is required/present |
| `trimmed` | Whether physical adapters/barcodes may be absent |

The manifest should be ordered, checksummed, stored with the raw stage, and
included in the semantic raw node's source identity.

Directory convenience can remain, but it should generate and display a
resolved manifest. Mixed input types, ambiguous mate pairing, duplicate source
files, and unsupported multi-BAM layouts should fail before external tools are
started.

## Proposed round-trip bundle

Experiment/project export intended for re-ingestion should be a bundle rather
than a directory of anonymous FASTQs:

```text
export/
  input_manifest.csv
  provenance.json
  reads/
    ...
```

The bundle should preserve or explicitly declare:

- experiment and molecule identities;
- sample and barcode labels;
- source grouping and mate identity;
- modality;
- filtering and deduplication state;
- trimming state;
- reference/alignment provenance;
- whether direct-modification signal is retained;
- collision-safe read-name policy.

FASTQ remains appropriate for derived sequence-only export. A BAM export is
needed when BC/RG/MM/ML or paired alignment metadata must survive.

## Incremental append semantics

Appending source files is a distinct lifecycle from restarting an identical
input. A safe design requires:

1. checksum the complete source manifest;
2. distinguish identical, append-only, removed, and mutated inputs;
3. publish raw data as immutable generations;
4. for append-only input, create new raw shards without modifying prior shards;
5. atomically advance a current-generation pointer only after validation;
6. invalidate or incrementally extend downstream products according to their
   semantic dependencies;
7. never treat the existence of an intermediate BAM as proof that it represents
   the current input set.

Without this generation model, a new source set should always use a fresh
output root.

## Prioritized improvements

### Priority 0: make current behavior explicit and safe

1. Reject mixed-type directories instead of silently applying type priority.
2. Reject BAM directories with a clear message until multi-BAM handling exists.
3. Add first-class `aligned_bam`/`validate_existing` ingestion.
4. Establish and validate one barcode/sample precedence contract.
5. Always publish a barcode sidecar for partitioned raw ingestion, including
   already-demultiplexed non-split BAMs.
6. Reject lossless direct-modification re-ingestion from ordinary FASTQ.
7. Validate supported aligner names during config loading.

### Priority 1: multi-file and paired-end correctness

1. Introduce the authoritative input manifest.
2. Support common Illumina/CASAVA paired filenames and explicit pair metadata.
3. Preserve paired layout through the aligner adapter.
4. Add the molecule/segment raw representation.
5. Implement quality-aware overlap consensus and uncovered-gap masks.
6. Support validated multi-BAM merge or source partitions.
7. Add a tag-preserving re-ingestion/export bundle.

### Priority 2: extensibility and lifecycle

1. Introduce structured alignment adapters.
2. Add BWA-MEM2 and Bowtie2 backends.
3. Add external alignment manifests and CRAM support.
4. Make raw inputs explicit semantic source artifacts.
5. Publish immutable raw generations and append-only extension.
6. Extend workflow `experiment run` to accept a manifest/bundle while retaining
   its reproducible single-source boundary.

### Project CLI follow-up

1. Add named-set management commands.
2. Add workflow executors for project sample analysis and shared embeddings, or
   narrow CLI planning targets to executable products.
3. Unify plan/run/validate compatibility inspection across project product
   types.

## Test and acceptance gaps

Existing focused tests cover project registration/materialization,
compatibility workflows, raw-store uniqueness, and FASTQ export selection and
namespacing. Direct acceptance coverage is still needed for:

| Scenario | Expected acceptance |
| --- | --- |
| Homogeneous FASTQ directory -> `experiment full` | Every manifest source appears in the raw generation |
| Mixed input directory | Fails before execution with all conflicting types listed |
| BAM directory | Explicit rejection or validated merge/partition behavior |
| Existing aligned BAM | No realignment; reference/tag/provenance validation recorded |
| Source file changed in place | Raw compatibility becomes incompatible |
| FASTQ appended to prior input set | Append-only generation or explicit rejection; never stale BAM reuse |
| Experiment export -> new `experiment full` | Labels and identity follow the declared bundle contract |
| Project export with duplicate bare read IDs | Collision-free re-ingestion |
| Direct-modification export/re-ingestion | MM/ML preservation or explicit unsupported error |
| Common Illumina R1/R2 filenames | Correct file pairing and sample/barcode assignment |
| Overlapping paired mates | One molecule with deterministic consensus |
| Non-overlapping paired mates | One molecule with `NaN` signal and false coverage in the gap |
| Discordant or singleton mates | Explicit metadata and configured keep/filter behavior |
| BWA-MEM2/Bowtie2 adapter | Stable argv, version, reference, paired layout, sort/index, and manifest |
| Existing CRAM | Exact reference validation and readable normalized ingestion |

External-tool end-to-end tests can remain in the local E2E tier, but component
tests should cover manifest resolution, pair assembly, consensus, gap masking,
alignment validation, and semantic invalidation without requiring external
executables.

## Current user guidance

Until the above contracts are implemented:

- use one homogeneous input type per experiment;
- use a fresh output root whenever the source file set changes;
- treat a FASTQ directory as a new experiment input, not an append operation;
- supply an explicit `fastq_barcode_map` instead of relying on filename
  inference for complex or paired filenames;
- do not expect a BAM directory to work;
- expect a supplied BAM to be aligned again;
- use Dorado or minimap2 as the only built-in aligners;
- do not assume paired Illumina mates become one SMF molecule;
- treat exported FASTQs as sequence-only derived data;
- retain the original signal or a tag-preserving BAM for direct-modification
  re-analysis.
