# Selective POD5 re-basecalling and processing-lineage implementation plan

**Plan date:** 2026-08-14

**Repository:** `smftools`

**Program status:** **Code-complete** as of 2026-08-17 — `SRB-01a/b` through
`SRB-09` are merged. The last lane, `SRB-09`, landed in PR #533 (`2f8aacc`;
main `4e2021e`).

What is left is evidence, not implementation: two of the catalog's 45 entries
are deferred on *data* rather than effort — cohort-versus-universe divergence
in dedup or a fitted HMM needs real multi-experiment data, and the
protected-data validation profile needs `NKG-03` to regenerate more than the
single pilot experiment. Both clear when the regeneration lane produces runs.
See
[generation_lifecycle_and_naming_implementation_plan.md](generation_lifecycle_and_naming_implementation_plan.md)
for `NKG-03`, which is the active lane across both programs.

**Implementation baseline:** `d862e3a` on `main` (after `EGL-13`; `a6ff9b5` was
the SRB-08b baseline)

**Current branch:** none — `feature/rebasecall-acceptance` merged in PR #533.

**Source audit:**
[selective_pod5_rebasecalling_audit.md](selective_pod5_rebasecalling_audit.md)

**Predecessor programs:**

- [input_ingestion_alignment_implementation_plan.md](../completed/input_ingestion_alignment_implementation_plan.md)
- [semantic_dag_variant_preprocessing_implementation_plan.md](../completed/semantic_dag_variant_preprocessing_implementation_plan.md)
- [experiment_project_partitioned_pipeline_implementation_plan.md](../completed/experiment_project_partitioned_pipeline_implementation_plan.md)

## Objective

Add a publication-safe workflow that can start from an established experiment
or project, select either all source signal or a frozen subset of prior
molecules, re-basecall those POD5 reads with a newly resolved Dorado model, and
publish a complete descendant processing lineage without mutating or silently
replacing the currently active results.

The completed program must:

- support all authoritative POD5 reads, all molecules in a parent raw
  generation, an explicit molecule/read list, or a QC predicate against one
  immutable preprocess generation;
- resolve old database identities to physical POD5 UUIDs, including namespaced
  and Dorado split-read cases;
- validate the exact source POD5 bytes after relocation;
- support canonical and compatible modified-base calling;
- resolve floating selectors such as `hac@latest` to full model identities;
- publish immutable selection, signal-resolution, basecall, raw-generation,
  downstream, and validation provenance;
- recompute QC/dedup and report parent-selection versus new-QC cohort deltas;
- preserve prior lineages and require explicit promotion;
- support project fan-out while preventing two processing lineages of the same
  biological experiment from being pooled as replicates; and
- remain usable through the existing semantic plan/run/validate and external
  workflow contracts.

Implementation is proceeding as focused branches/PRs cut from current `main`,
without feature-branch version bumps. `SRB-01` is split so `SRB-01a` establishes
the strictly read-only contract; freezing selections remains a run-time concern
and starts only after the planner is reviewed.

## Program finding IDs

The stable finding IDs are defined in the companion audit: `SRB-C1` through
`SRB-M4`.

## Implementation status

| Item | Status | Evidence |
|---|---|---|
| SRB-01a | **Merged** — PR #515 (`6ff2dd5`; main `5d9c438`) | Strict schema-1 JSON/YAML request parsing, bounded allowlisted QC predicates, exact immutable raw/preprocess parent resolution, distinct `all-signal`/`all-parent-molecules`/`qc`/`ids` counts, source availability, stable blocker/warning codes, and nested `experiment rebasecall plan [--json]`. Execution, downloads, selection freezing, publication, and promotion are explicitly unavailable. 21 focused tests, 106 smoke tests, and the full unit suite (1,948 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| SRB-01b | **Merged** — PR #518 (`94fbbd7`; main `edf08e5`) | Run preparation now requires explicit acceptance of a stable plan ID, revalidates immutable parent manifests, and atomically freezes a schema-versioned, content-addressed manifest plus deterministic Parquet rows. Selection identity is path-neutral and model-independent while source-column fingerprints detect changed inputs. Exact all-signal, molecule, and split-parent rows; reuse; tampering; stale acceptance; parent drift; blocked plans; and interrupted writes are covered. 24 focused planner/freezing tests, 106 smoke tests, and the full unit suite (1,978 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| SRB-02a | **Merged** — PR #516 (`c2a668e`; main `1325854`) | Identity schema 3/raw schema 5 preserve exact `basecall_read_id`, Dorado `pi` as `basecall_parent_read_id`, and POD5-index-validated `pod5_read_id` plus status/evidence through raw shards, segment/molecule catalogs, indexes, spines, and namespaced partitions. Split children share the validated signal parent without losing their own basecall IDs, including signal-feature lookup. Both pysam and samtools tag paths are covered. 81 focused tests, 106 smoke tests, and the full unit suite (1,955 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| SRB-02b | **Merged** — PR #517 (`0adfe0c`; main `a5dc558`) | A deterministic POD5 dataset index now drives exact selected-row resolution in durable-field, parent-field, legacy source/basecall/read-ID, then retained-BAM `pi` order. The planner reports bounded evidence, stable digests, duplicate split-parent references, duplicate-source ambiguity, and unreadable/unresolved blockers without writing artifacts. The checked-in POD5 fixture indexes 4/4 UUIDs; the real `241213` current generation selects 19,328 molecules and correctly blocks because its authoritative input manifest is BAM-only. 17 focused tests, 106 smoke tests, and the full unit suite (1,964 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| SRB-03a | **Merged** — PR #519 (`f417f64`; main `2fa79f4`) | Original and explicit relocated POD5 candidates are mutation-safely hashed and accepted only when byte size and SHA-256 match the canonical input-manifest row. Resolution has stable evidence/counts/failures and a path-independent digest; exact relocated paths feed the POD5 UUID index while mismatches, missing/unreadable sources, and unknown relocation identities block. The existing manifest remains authoritative, so no duplicative project catalog was added. The checked-in POD5 fixture relocates under a different name and still indexes all 4 UUIDs. 78 focused tests, 106 smoke tests, and the full unit suite (1,989 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| SRB-03b | **Merged** — PR #520 (`83c3367`; main `b5891b0`) | Accepted frozen selections now materialize as content-addressed, atomic, per-source filtered POD5 artifacts. Strict manifests retain source/selection identity, checksums, and requested/found/missing/duplicate UUID accounting; reuse revalidates both bytes and indexed UUID sets. Interrupted writes and source drift leave no published partial artifact, split-child references are deduplicated with preserved multiplicity counts, and a real checked-in POD5 subset remains fully valid after its original source is removed. 48 focused tests, 106 smoke tests, and the full unit suite (2,001 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| SRB-04a | **Merged** — PR #521 (`c856ca2`; main `676026f`) | Read-only planning now probes the installed Dorado build and supported basecaller flags, resolves POD5 run conditions against Dorado's structured chemistry catalog, selects exact full simplex/modification model names, hashes every installed model directory into a path-neutral bundle identity, and emits normalized `--read-ids`, splitting, Q-score, summary, moves, trim, and barcode arguments. Exact Dorado version, capabilities, model bytes, and invocation semantics enter the accepted Plan ID, so the same floating alias cannot reuse a changed bundle. Fake-executable, missing-capability/model, multi-chemistry, relocation, tamper, and checked-in POD5 metadata coverage passes; installed Dorado 1.3.1 resolves the fixture to `dna_r10.4.1_e8.2_400bps_hac@v5.2.0`. 58 focused tests and 106 smoke tests pass; Ruff and format are clean. **Deferred validation closed 2026-08-16, after the merge** — re-run on the host at `main` `676026f`: unit **2,014 passed, 8 skipped, 178 deselected, 7 xfailed** (4m08s), 106 smoke passed, `ruff check` clean, `ruff format --check` clean across 592 files, and `sphinx-build -W` succeeded. The sandbox run's 20 `SC_SEM_NSEMS_MAX` failures were confirmed to be the sandbox semaphore limit and nothing else: 1,994 + 20 = 2,014, exactly the host pass count. |
| SRB-04b | **Merged** — PR #522 (`2ae9bb7`; main `59eeb82`) | `pipeline/rebasecall_basecall.py` executes an accepted plan and publishes one immutable content-addressed basecall through the `SRB-03b` transaction: staged build, full validation, single `os.replace`, and staging removed on any failure so nothing partial looks reusable. Reuse keys on the frozen selection **and** the resolved model bundle, so two `hac@latest` requests over the same reads diverge once the installed models do. Validation is against what Dorado emitted — header Dorado version and `basecall_model` inside the resolved bundle, every record's `pi` parent inside the exact requested UUID set, duplicate IDs, split-child multiplicity, and absent reads counted; foreign parents and model disagreement are hard failures. Per `D2`, `generation_kind` is stamped here, with `SELECTION_GENERATION_KINDS` shared with the planner. Nothing derived from the staging path enters the manifest — the `EGL-01` pointer-leak hazard, designed against rather than rediscovered. 18 focused tests including a real pysam-written Dorado-shaped BAM; full unit suite 2,032 passed, 8 skipped, 179 deselected, 7 xfailed; 53 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. **Validated against the real basecaller**: the new opt-in `tests/e2e/cli/test_rebasecall_basecall_e2e.py` runs installed Dorado 1.3.1 over the checked-in POD5 fixture, basecalls exactly the 2 selected UUIDs, and validates the published manifest end to end in ~13s, skipping cleanly where dorado or a model directory is absent. |
| SRB-05a | **Merged** — PR #523 (`c22574d`; main `3e617e5`) | `pipeline/rebasecall_lineage.py` publishes a lineage as the `D1` map `stage -> generation id`, with no nested run tree: descendant generations go into the experiment's ordinary stage directories beside the parent's, and the lineage records which ids are its own. This required separating publication from selection in the shared helper — `staged_generation(..., select_current=False)`, threaded through `publish_raw_generation` — so a descendant is addressable immediately while `current.json` keeps resolving the parent; only `SRB-08` promotion changes that. Raw generation schema 3 adds an optional strictly-validated `lineage` block whose absence is meaningful, with `generation_kind` read from the basecall per `D2` rather than restated. `plan_raw_append` now refuses a lineage descendant as an append base. Exit gate covered directly: a killed stage leaves the prior lineage intact, staging empty, and the parent still current. 16 focused tests; full unit suite 2,048 passed, 8 skipped, 179 deselected, 7 xfailed; 53 integration, 106 smoke, 19 e2e; Ruff check/format and Sphinx `-W` clean. |
| SRB-05b | **Merged** — PR #524 (`c248f64`; main `f2b0985`) | `raw_adata` gains `lineage_provenance`: a descendant publishes beside the parent without advancing `current.json` and takes neither the completeness skip nor the append path. `pipeline/rebasecall_run.py` adds `derive_descendant_config` (parent config, new `calls.bam`, everything else inherited verbatim; `output_directory` deliberately unchanged so the descendant lands in the experiment's ordinary stage directories) and `run_lineage_raw_stage`, which runs the stage *inside* the lineage transaction so the exit gate holds through the real work. `LineageRawStageResult.to_dict()` is the engine-facing workflow payload. 7 focused unit tests plus **2 integration tests driving the real `raw_adata`** — a descendant publishes beside its parent with `current.json` untouched and schema 3 carrying the lineage block, and an ordinary generation records none. Full unit suite 2,055 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke, 19 e2e; Ruff check/format and Sphinx `-W` clean. |
| SRB-06a | **Merged** — PR #525 (`eaa4bcb`; main `fd9250a`) | The `D1` selector now reaches path computation: `get_adata_paths(lineage_generations=...)` pins which generation each stage resolves, `preprocess_adata` accepts that pin plus lineage provenance, and `run_lineage_raw_stage` runs preprocess after raw whenever the request's `downstream.target` is deeper than `raw`. **A hazard this exposed:** `staged_generation` ran `after_current` unconditionally, but preprocess and latent use that hook to publish the *canonical stage-root spine* that ordinary readers resolve — a non-selected descendant would have overwritten it and silently become the parent's answer. The hook is now split into `after_publish` (always, describing the generation) and `after_current` (only when the selector advances), covered directly. `validate_lineage_provenance` moved into `informatics/generation.py` as shared vocabulary. Preprocess records the same optional `lineage` block without a schema bump, since its validator compares versions with `!=`. Full unit suite 2,057 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. |
| SRB-06b | **Merged** — PR #526 (`244b880`; main `ef32cc0`) | `pipeline/rebasecall_transition.py` reconciles every selected origin molecule: one row each, survived or not, carrying origin identity, POD5 UUID/source, the basecall outputs and new raw molecules produced, recomputed QC/dedup flags, and a terminal status (`no_signal`, `no_call`, `dropped_in_raw`, `failed_qc`, `duplicate`, `passed`, `qc_not_run`). A split origin stays one row with children aggregated, so the row count equals the frozen selection's. `qc_not_run` is deliberate — a lineage that stopped before preprocess reports QC as not recomputed rather than leaving it absent, which would read as passing. **Exit gate met:** `reconcile_qc_transition` recomputes every published count from the published table alone and reports disagreement, including that terminal statuses sum to the whole selection; build and reconcile share one derivation so the check cannot pass by computing counts differently than the writer. The report is written after publication and outside lineage identity. The orchestrator now **refuses** targets deeper than preprocess rather than silently stopping there. 9 focused tests plus an end-to-end lineage reconciliation; full unit suite 2,067 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. |
| SRB-06c | **Merged** — PR #527 (`9d0aecf`; main `4c50d4b`) | `spatial_adata`, `hmm_adata`, and `latent_adata` take `lineage_generations`/`lineage_provenance`, publish with `select_current=False`, and record the `lineage` block. The orchestrator maps each target to its stages (`_TARGET_STAGES`) and pins every stage to the generations the lineage already published, so `full` runs end to end and the guard now rejects only unknown targets. As predicted, spatial and hmm call `publish_canonical_spine` *after* their `staged_generation` block, outside the hook `SRB-06a` gated — both now skip it for a descendant, which also reports its own generation spine rather than the stage-root path. **A bug only running it found:** the descendant spatial run silently *skipped*, because the completeness check saw the parent's spine; a lineage builds beside whatever is current, so the parent looking complete says nothing about this run's work. Fixed in all three, mirroring `raw_adata`. Having found it in spatial and fixed hmm/latent by inspection, hmm got its own real-CLI descendant test, which then caught a double writing its spine to the canonical path instead of staging. Full unit suite 2,070 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. |
| SRB-07a | **Merged** — PR #528 (`bd5a240`; main `c5a3207`) | Registry schema 5 adds `D1`'s lineage map. An entry's own `spines`/`catalogs` *are* the `original` lineage, synthesized on read, so a pre-schema-5 registry reads identically and nothing on disk is rewritten (`load_registry` never gated on schema version, so the bump is safe). `register_experiment_lineage` records a descendant without changing what the project resolves; `set_active_lineage` moves only the selector, leaving the prior lineage exactly as queryable, and refuses an incomplete lineage. `list_experiments(lineage=...)` resolves exactly one lineage per experiment and reports which; an unknown lineage is an error rather than a silent fall back to the original. `assert_one_lineage_per_experiment` rejects pooling two lineages of one experiment — the same biology processed twice would double-count every molecule — while distinct experiments on different lineages stay legitimate. 7 focused tests; full unit suite 2,077 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. |
| SRB-07b | **Merged** — PR #529 (`977be4a`; main `5000329`) | `ProjectCatalog.open(lineage=)` makes the catalog the single lineage-aware seam, so selection, materialization, and interval catalogs read one lineage instead of each caller deciding; `assert_one_lineage_per_experiment` runs on every `experiments()` result. **An inconsistency this exposed:** `open()` built the harmonized reference alias table straight from `registry["experiments"]`, bypassing lineage resolution, which would have left the reference view describing the original processing while queries returned a descendant's. `pipeline/rebasecall_project.py` fans one request across selected experiments: each resolves its own chemistry/model bundle (a project spans flow cells and kits, so one forced resolution would block the project on the odd experiment out or silently use a model chosen for another chemistry), a blocked experiment does not block the others, a failed one does not abort the run, and registration never changes what the project resolves. 9 focused tests; full unit suite 2,086 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. **Deferred:** named project lineage sets, listed as "if needed for publication review" — no need has appeared, and `list_experiments(lineage=<mapping>)` already expresses a per-experiment selection. |
| SRB-08a | **Merged** — PR #530 (`0838459`; main `1adc3f6`) | `pipeline/rebasecall_validate.py` is the check behind the registry's claim: `validate_rebasecall_lineage` revalidates the manifest, every named stage generation, the QC transition's reconciliation, and the basecall, writing a machine-readable report into the `validation.json` slot reserved since `SRB-05a`. `promote_rebasecall_lineage` runs it first and treats failure as the refusal, so **the exit gate is enforced rather than documented**; promoting a descendant without enough context to verify it is refused too, so the check cannot be sidestepped by omitting arguments. Rollback is the same operation — promoting a prior complete lineage, `original` included — and the descendant stays registered and queryable. Replayability is reported separately from completeness (a lineage can be complete and not replayable) and folded in only under `require_replayable`. A defect the tests caught: `failures` initially included the informational replay check, so a *complete* report could carry failures and a refusal message could name `replayable` when replay was never the issue. 11 focused tests; full unit suite 2,097 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and Sphinx `-W` clean. |
| SRB-08b | **Merged** — `a6ff9b5` on `main`. Landed as a fast-forward rather than a PR merge commit (GitHub could not load merge status; the branch was 1 ahead / 0 behind, so `merge --ff-only` from the `main` worktree was equivalent), so there is no `Merge pull request` commit for this one. | `docs/source/tutorials/selective_rebasecalling.md` leads with the exit gate's distinction — a table of the four selection modes against what each result supports, then why old-QC selection is scope-biased (a `qc` request selects reads whose QC came from the *old* basecalls, so it measures the new model on the old model's easy cases). Also covers refreshed QC and the transition report's terminal statuses, model pinning, direct-modification behavior, source retention with rough disk sizing, promotion/rollback/validation, a publication checklist, and migration. **A gap this exposed:** the tutorial described `smftools project rebasecall plan`, which the command surface lists but `SRB-07b` left as a library function; documentation referencing a command users cannot run is a defect, so the read-only command now exists with a test that it emits the stable JSON schema and writes nothing. **Deliberately not done:** a `2.21.0` release-notes file — `2.20.0.md` was added by the `release: 2.20.0` commit, so this repo authors release notes at release time, and the material a release author needs is in the tutorial. Full unit suite 2,098 passed, 8 skipped, 181 deselected, 7 xfailed; 55 integration, 106 smoke; Ruff check/format and warning-strict Sphinx clean. |
| SRB-09 | **Merged** — PR #533 (`2f8aacc`; main `4e2021e`) | Acceptance matrix and real-tool validation, in the established form: a versioned `tests/acceptance/*.json` catalog **plus** a unit test resolving every citation back to a real symbol, as `EGL-08` and the input/alignment lane did. 45 entries — 12 audit findings (`SRB-C1`..`M4`), 18 delivered items (`SRB-01a`..`09`), 13 minimum scenarios, 2 validation profiles — of which 43 are automated and 2 deferred. The validator rejects the ways a catalog rots: an unresolvable citation, a deferment without owner and reason, an "automated" entry carrying deferment vocabulary, a missing finding or item, and deferments exceeding a quarter of the catalog; it was checked to bite by pointing one citation at a nonexistent symbol. Writing it also closed a deferral recorded on a bad assumption — direct-modality QC recomputation *was* automatable with the existing corpus (`write_raw_store` + `execute_partitioned_preprocessing`), now `test_direct_modality_lineage_qc.py`, parametrized across direct/conversion/deaminase. **The 2 remaining deferrals are blocked on `NKG-03` data.** |

**Open integration gap, carried from `D2` (noted 2026-08-16).** The artifact
layout above puts basecalls at `<run>/basecalls/generations/<id>/` beside a
`current.json` — the *fifth generation kind*. `SRB-04b` instead publishes to its
own content-addressed root, and the generation-lifecycle plan still lists that
migration as open. Nothing is broken: a lineage references `basecall_id`, which
is stable either way. But moving basecalls onto the shared generation vocabulary
(including reading a flat legacy `<run>/basecalls/*.bam` as `legacy_in_place`) is
real remaining work, and it is the natural companion to `SRB-05b`.

## Agreed design contracts

### A processing lineage is the publication unit

A lineage is a complete, immutable descendant of an experiment's earlier
artifacts. It is not a second biological experiment and not an in-place
replacement raw generation.

A lineage identity covers:

```text
parent experiment_uid
+ parent raw generation ID
+ optional parent preprocess generation ID
+ frozen selection result ID
+ source-signal resolution ID
+ resolved basecall result ID
+ reference/config semantic identities
+ requested terminal downstream target
```

Execution builds beneath `.staging/<lineage_id>`, validates every required
artifact, then atomically publishes `lineages/<lineage_id>`. A failed lineage
never changes the active lineage or any current pointer in the parent run.

An implementation may internally run the ordinary raw -> preprocess -> spatial
-> HMM -> latent pipeline beneath the staged lineage root. This is preferable
to creating second implementations of those stages. The lineage manifest is
the outer transaction that keeps stage outputs together even where an
individual stage does not yet expose multiple generations in one run root.

### Selection scope is explicit

The initial selection modes are:

| Mode | Parent required | Selection |
| --- | --- | --- |
| `all-signal` | Raw input manifest | Every read UUID in all validated POD5 sources |
| `all-parent-molecules` | Raw generation | Every parent molecule that can be resolved to source signal |
| `qc` | Preprocess generation | Molecules satisfying an explicit predicate over allowlisted scalar obs columns |
| `ids` | Raw generation | Explicit `molecule_uid`, project-safe `(experiment_uid, read_id)`, or source POD5 UUID input |

`all-signal` and `all-parent-molecules` must not be aliases. The plan displays
the count difference before execution.

Selection is evaluated once and written to an immutable Parquet artifact. It
records the source generation, predicate syntax/version, columns consumed,
column fingerprints, selected molecule identities, and deterministic selection
digest. External tools consume this frozen artifact, never a live re-evaluation
of `current`.

### QC predicates are structured and safe

The request uses a versioned structured predicate, not Python `eval` or an
arbitrary SQL string. Initial operators should cover boolean equality,
numeric/string comparisons, membership, null checks, and nested `all`/`any`/
`not` composition over scalar observation columns.

Example request:

```yaml
schema_version: 1
name: publication-2026
source:
  raw_generation: current
  preprocess_generation: current
selection:
  mode: qc
  predicate:
    all:
      - {column: passes_read_qc, op: eq, value: true}
      - {column: passes_modification_qc, op: eq, value: true}
      - {column: passes_variant_qc, op: eq, value: true, missing: fail}
      - {column: passes_dedup, op: eq, value: true}
basecall:
  model: hac@latest
  modified_bases: []
  read_splitting: preserve
  trim: none
  emit_moves: true
  min_qscore: 0
signal:
  materialize: false
downstream:
  target: full
promotion:
  activate: false
```

Convenience presets such as `final-parent-qc` may expand to this schema, but
the expanded predicate is stored. A missing requested column fails unless the
request explicitly defines its missing-value policy.

### Parent selection and refreshed QC are separate facts

No parent QC result is copied as the new result. The descendant lineage runs
the configured preprocess and downstream semantic graph against the new calls.

The lineage publishes a transition table with at least:

```text
origin molecule identity
source POD5 UUID and source ID
selected_by_parent
source_signal_resolved
basecall_output identity/identities
new raw molecule identity/identities
new passes_read_qc
new passes_modification_qc
new passes_variant_qc, when present
new passes_qc
new is_duplicate
new passes_dedup
terminal status and reason
```

The default downstream cohort is the new lineage's own QC/dedup result. A user
may inspect the parent-selected set regardless of the new outcome, but a
publication export may not label all parent-selected rows as new-QC passing.

### Source-signal resolution is checksum-first

The source raw generation's copied input manifest is authoritative for source
IDs and SHA-256 values. Source resolution tries:

1. the recorded path, if it still matches size and SHA-256;
2. an explicit request-level source-ID/checksum relocation map;
3. a project source catalog keyed by source ID/checksum, if configured; or
4. a user-provided source root with deterministic manifest-relative mapping.

There is no automatic broad filesystem search and no basename-only match.
`pod5_origin` is diagnostic only.

The resolver publishes one row per source with recorded and resolved paths,
content validation, and replayability state. Paths are excluded from semantic
identity; checksums and source metadata are included.

### Molecule-to-POD5 mapping is complete or the plan blocks

The resolver uses explicit origin fields first, then validates candidate UUIDs
against a POD5 dataset index. For historical split children it may read the
retained source/aligned BAM and use `pi`. The resulting evidence source is
recorded per row.

Selected identities must partition into exactly one of:

- one resolved POD5 parent UUID;
- a deliberate duplicate reference to an already selected parent (for example,
  two old split children); or
- unresolved/ambiguous.

An unresolved or ambiguous requested molecule blocks execution by default.
The user may create a new request excluding named failures after reviewing the
plan; `--missing-ok` is not a publication-safe default.

Future raw generations should promote at least `pod5_read_id`,
`basecall_parent_read_id`, and `basecall_read_id` into durable scalar identity
fields so BAM fallback is unnecessary.

### Read splitting has an explicit identity policy

The request supports:

- `preserve`: use Dorado's configured/default splitting and publish a
  one-to-many parent/output map;
- `disable`: pass `--disable-read-splitting` and preserve one basecall record
  per source POD5 read where Dorado emits a call.

There is no implicit attempt to reproduce an unknown old Dorado default.
Splitting policy is scientific configuration and participates in compatibility.

New lineage molecule identity retains the parent experiment identity for a
one-to-one unsplit call. When one source signal produces multiple new reads,
each new molecule receives its own descendant identity and records
`origin_molecule_uid` plus `pod5_read_id`. Project joins may compare lineage
outputs through origin identity; ordinary project pooling still selects only
one lineage per experiment.

### Dorado requests may float; published model identities may not

The basecall adapter distinguishes:

- `requested_model_complex`, for example `hac@latest`;
- resolved full simplex model name;
- resolved modification model names;
- Dorado executable version;
- model directory artifact checksums or a deterministic model-bundle digest;
- normalized argv and scientific options; and
- BAM `@PG`/`@RG` metadata observed after execution.

Resolution uses the POD5 sequencing condition and installed Dorado model
catalog. A plan may report model resolution as blocked when source signal or a
compatible installed Dorado is unavailable. Reuse is based on the resolved
model identity, not only the floating request string.

For direct modality, the adapter must resolve one compatible simplex and set
of modification models. Old MM/ML probabilities and any QC derived from them
are not reused.

### Selective basecalling is complete and count-validated

The default compute route supplies the frozen POD5 UUID file to Dorado
`--read-ids`. Optional signal materialization writes filtered POD5 artifacts
per source, preserving source partition identity and avoiding one enormous
merge.

The basecall manifest validates:

- requested unique POD5 UUID count;
- UUIDs found in source indexes;
- Dorado source-parent UUIDs observed;
- output record count;
- split-child multiplicity;
- reads discarded or absent, with a reason when available;
- duplicate output IDs;
- model/header agreement; and
- BAM structural validity and checksum.

`min_qscore` defaults to zero at the basecaller boundary so smftools owns the
publication QC decision. Any nonzero basecaller filter is allowed only as
explicit scientific configuration and appears as pre-ingestion loss in the
transition report.

### Basecall configuration is separate from downstream configuration

Model, modifications, trimming, read splitting, barcoding, `emit_moves`, and
minimum basecaller Q-score form the basecall semantic configuration. Device,
batch size, threads, and worker/chunk layout are execution provenance unless
they are shown to change scientific output.

Alignment reference identity, alignment adapter/arguments, modality, raw
extraction semantics, preprocessing thresholds, and downstream analysis
settings continue to use the existing configuration and semantic node
authorities. The re-basecall program must not invent duplicate configuration
fields for them.

### Descendant raw generations declare their origin

Extend raw generation provenance without breaking schema-1/2 readers. A
descendant generation records:

```text
generation_kind: full_source | parent_universe | selected_cohort
origin experiment_uid
parent raw generation ID
parent preprocess generation ID, when selection used it
selection result ID
source-signal resolution ID
basecall result ID
origin-to-descendant identity-map artifact
```

The input POD5 manifest remains a source artifact. The frozen selection and
basecall manifests are additional dependencies, not encoded as fake POD5 input
rows.

**Amended 2026-08-14 by `D2` in
[generation_lifecycle_and_naming_implementation_plan.md](generation_lifecycle_and_naming_implementation_plan.md):**
`generation_kind` moves off this list and onto the **basecall** generation
manifest — the artifact whose contents the selection actually determines. The
descendant raw generation records `basecall_generation_id` and derives
`generation_kind` from it; a mirrored copy is allowed for convenience, but the
basecall manifest is authoritative and validation fails on disagreement. Every
other field above stays on raw. The driving case is sharing: under the reconciled
run-level basecall layout, one basecall serves several experiments (`260309` has
one SUP basecall for two), so on raw the same fact would be restated once per
descendant and the restatements could disagree. A legacy flat
`<run>/basecalls/*.bam` with no manifest reads as `full_source`.

### Project registration is lineage-aware

Each project experiment entry gains a backward-compatible lineage map:

```text
experiment
  experiment_uid
  active_lineage: original
  lineages
    original: existing stage pointers/generation IDs
    <lineage_id>: descendant stage pointers/generation IDs/status
```

Legacy entries are read as one `original` lineage. Project selections resolve
exactly one lineage per experiment: active by default, or an explicit named
lineage/project lineage set. A query that would include two lineages of the
same experiment fails before materialization.

`promote` atomically changes only the active-lineage selector after validating
the complete lineage. It does not delete prior artifacts, rewrite the
experiment's scientific identity, or masquerade as adding a new replicate.
Rollback is promotion of a prior complete lineage.

**Amended 2026-08-14 by `D1` in
[generation_lifecycle_and_naming_implementation_plan.md](generation_lifecycle_and_naming_implementation_plan.md):**
this registry entry is the **only** cross-stage selector in either program, and
the optional experiment-local `active.json` in the artifact layout below is
dropped — two experiment-scoped selectors can disagree with no rule for which
wins. EGL's `current.json` stays strictly within-kind ("which generation of
*this* stage is the default here"), so the two layers compose: a lineage is a map
`stage → generation id`, and the `original` lineage is that map defined
implicitly as each stage's `current.json`. An experiment consulted outside any
project resolves `original`. EGL therefore ships no experiment-scoped selector
for SRB to subsume.

### Planning is read-only; promotion is separate

The user-facing lifecycle is:

```text
request -> plan -> run -> validate -> review -> promote
```

Planning resolves parent generations, selection counts, source availability,
identity mapping, chemistry/model compatibility, estimated resources, and
expected downstream invalidation without writing scientific output. A plan may
write no cache or model downloads; if model download is required, it reports a
blocked external prerequisite or a distinct prepare action.

Run never promotes. Validate never promotes. Promotion requires a separate
explicit command and is the only action that changes a project's default
lineage.

## Proposed command surface

The experiment planner spelling was finalized in `SRB-01a` as a nested subgroup.
Later lifecycle commands should extend that subgroup rather than create a second
planner or validator:

```text
smftools experiment rebasecall plan CONFIG REQUEST.yaml [--json]
smftools experiment rebasecall run CONFIG REQUEST.yaml --output-root ROOT
smftools experiment rebasecall validate LINEAGE_ROOT [--json]
smftools experiment rebasecall promote CONFIG LINEAGE_ID

smftools project rebasecall plan PROJECT REQUEST.yaml [--set NAME] [--json]
smftools project rebasecall run PROJECT REQUEST.yaml --output-root ROOT [--set NAME]
smftools project rebasecall validate PROJECT LINEAGE_RESULT [--json]
smftools project rebasecall promote PROJECT LINEAGE_RESULT
```

The engine-facing result JSON follows the existing workflow contract and
contains per-experiment task results for project fan-out. A later CLI review
may prefer a nested `rebasecall plan|run|validate|promote` subgroup, but it must
not create a second planner or validator.

## Artifact layout

Per the EGL reconciliation, basecalls are run-level generation artifacts and a
lineage references generation IDs rather than embedding a BAM or a second nested
run tree:

```text
basecalls/
  current.json
  .staging/<basecall_generation_id>/
  generations/<basecall_generation_id>/
    generation_manifest.json
    calls.bam
    sequencing_summary.tsv               # when supported/requested
    selection/
      molecules.parquet
      predicate.json
      pod5_read_ids.txt
      source_resolution.parquet
      identity_map.parquet
    signal/
      manifest.json
      filtered/<source_id>.pod5           # optional

rebasecall_outputs/
  requests/<request_id>.json
  .staging/<lineage_id>/
  lineages/<lineage_id>/
    lineage_manifest.json
    request.json
    basecall_generation_id
    stage_generations.json                # raw/preprocess/spatial/hmm/latent IDs
    qc_transition.parquet
    validation.json
                                          # no active.json: the project registry's
                                          # active_lineage is the sole cross-stage selector
```

All pointers inside a published lineage are lineage-root relative. External
POD5 paths remain provenance/resolution hints unless filtered signal was
materialized. `validate --require-replayable` requires either accessible
checksum-matching source POD5s or complete filtered POD5 artifacts.

## Semantic graph

The new nodes are semantic orchestration nodes; physical source/read chunks
remain task-catalog rows:

```text
parent raw generation
  + optional parent preprocess generation
    -> rebasecall.selection

parent input manifest + source relocation map
    -> rebasecall.signal_resolution

selection + signal_resolution + Dorado/model config
    -> rebasecall.basecalls
    -> descendant experiment.raw.complete
    -> descendant experiment.preprocess.complete
    -> descendant experiment.spatial.complete
    -> descendant experiment.hmm.complete
    -> descendant experiment.latent.complete (when requested)
    -> rebasecall.qc_transition
    -> rebasecall.lineage.complete

complete lineage + explicit user action
    -> project.active_lineage_selector
```

Promotion is registry state mutation, not a scientific compute node hidden
inside `lineage.complete`.

## Delivery strategy

### SRB-01 — request, selection, and read-only planning contract

**Status:** `SRB-01a` merged in PR #515. `SRB-01b` is implemented on
`feature/rebasecall-selection-freezing` and awaiting commit/review. Accepted
plans can now freeze authoritative selections during run preparation without
starting Dorado or publishing lineage state.

**Scope**

- Define request, selection-predicate, plan, and result schemas.
- Resolve exact parent raw/preprocess generations rather than live stage paths.
- Implement `all-signal`, `all-parent-molecules`, `qc`, and `ids` planning.
- Freeze deterministic molecule selections and source column fingerprints only
  during run, after the read-only plan is accepted.
- Add configuration validation and stable reason codes.
- Finalize CLI spelling and update the CLI command map before implementation.

**Primary areas**

- new re-basecall planning module under `pipeline/` or `informatics/`;
- `pipeline/semantic_graph.py` and analysis registry adapters;
- `cli_entry.py` and `src/smftools/cli/AGENTS.md` command map;
- config defaults/parser for user-tunable request references only; and
- unit schema/planning tests.

**Exit gate**

A read-only plan can explain the exact parent, selection universe/count,
required sources, prospective model resolution, downstream target, and every
blocking reason without creating scientific artifacts.

### SRB-02 — durable POD5 origin identity and historical resolution

**Status:** `SRB-02a` merged in PR #516 and `SRB-02b` merged in PR #517.
Selection freezing is implemented in the follow-on `SRB-01b` branch.

**Delivery split**

- `SRB-02a` writes explicit origin identity for every newly ingested BAM row.
  The BAM QNAME is `basecall_read_id`; Dorado `pi`, when present, is
  `basecall_parent_read_id`; and only a parent-or-QNAME candidate found in the
  configured POD5 dataset is promoted to `pod5_read_id`. Resolution status and
  evidence remain explicit when the source is unavailable or the candidate is
  absent. These fields survive namespaces and every raw catalog/index/spine.
- `SRB-02b` resolves existing immutable generations. It owns the ordered
  explicit-field, `source_read_id`, POD5-index, and retained-BAM fallback;
  deterministic evidence rows; duplicate-parent accounting; and blockers for
  unresolved or ambiguous selected molecules.

**Scope**

- Add durable `pod5_read_id`, `basecall_read_id`, and parent/split identity to
  new raw artifacts.
- Build selected molecule -> source POD5 UUID resolution for current and legacy
  generations.
- Use `source_read_id`, POD5 indexes, and BAM `pi` fallback in the agreed order.
- Detect multiple old split children selecting one source parent.
- Publish deterministic resolution/evidence rows and actionable failures.

**Primary areas**

- `informatics/molecule_identity.py`;
- BAM tag extraction and raw scalar schema;
- raw store molecule/segment catalogs and indexes;
- POD5 dataset/index helper; and
- focused split/namespaced/multi-source tests.

**Exit gate**

Every selected historical molecule is either mapped unambiguously to a source
POD5 UUID with recorded evidence or blocks before basecalling.

### SRB-03 — checksum-based source resolver and optional signal materialization

**Status:** `SRB-03a` merged in PR #519. `SRB-03b` is implemented on
`feature/rebasecall-signal-materialization` and awaiting commit/review.
Exact-byte source resolution, atomic filtered signal publication, strict
artifact validation, and replay without the original sources are complete.

**Delivery split**

- `SRB-03a` owns mutation-safe checksum validation, deterministic path-neutral
  source decisions, stable failure codes, and validated-path handoff to the
  POD5 UUID index.
- `SRB-03b` owns atomic per-source filtered POD5 publication, artifact
  manifests, requested/found/missing UUID accounting, and replayability after
  original sources are removed.

**Scope**

- Resolve original, relocated, project-catalog, and explicit source paths by
  source ID/checksum.
- Add a project source catalog only if it is the minimum necessary authority;
  do not duplicate the canonical input manifest.
- Support direct selected-read access and optional per-source filtered POD5
  publication.
- Validate requested/found/missing/duplicate UUID counts.
- Add replayability validation and relocation tests.

**Primary areas**

- `informatics/input_manifest.py`;
- `informatics/pod5_functions.py` or a new focused source resolver;
- sidecar/artifact manifest helpers; and
- project registry source metadata.

**Exit gate**

A source plan survives path relocation when the same checksum-matching POD5s
are mapped explicitly, and optional filtered signal survives without originals.

### SRB-04 — selective Dorado adapter and immutable model/basecall manifest

**Status:** both halves are **merged** — `SRB-04a` model/capability resolution
(PR #521, `c856ca2`) and `SRB-04b` execution, BAM validation, and immutable
basecall publication (PR #522, `2ae9bb7`; main `59eeb82`), which meets both
exit-gate clauses under test.

**Two decisions worth carrying into `SRB-05`:**

- *One BAM, not one per source.* Signal materialization stays per source to
  avoid a large merge, but execution presents every resolved source through a
  staging directory and produces a single `calls.bam`, matching the reconciled
  run-level basecall layout in the generation-lifecycle plan
  (`calls.bam` + `read_to_pod5_origin.csv` beside one manifest). Partition
  identity is not lost: `read_to_pod5_origin.csv` carries
  `read_id, basecall_parent_read_id, pod5_read_id, pod5_source_id` per record.
- *No CLI surface yet.* Like `SRB-01b` and `SRB-03b`, this lane is library-level
  only; `experiment rebasecall plan` remains the sole command. The user-facing
  execution surface belongs to `SRB-08`.

**Delivery split**

- `SRB-04a` owns read-only executable/capability probes, POD5 chemistry
  resolution, exact installed simplex/modification model identity and checksums,
  normalized argv, stable blockers, and accepted-plan invalidation.
- `SRB-04b` owns atomic Dorado execution, output structure/header/read-parent
  validation, basecall manifests, and content-addressed reuse.

**Scope**

- Replace direct helper argument construction with a structured basecall
  adapter while preserving existing canonical/modified behavior.
- Add `--read-ids`, splitting policy, explicit minimum Q-score, summary, moves,
  trimming, and barcode options.
- Probe supported flags for the installed Dorado version.
- Resolve and record full simplex/modification model identities and model
  bundle digest.
- Validate output BAM headers, IDs, counts, structure, and checksums.
- Make content-addressed reuse depend on the frozen selection and resolved
  model identity.

**Primary areas**

- `informatics/basecalling.py`;
- `informatics/raw_intermediate_manifest.py`;
- new basecall manifest/adapter module;
- Dorado fake-executable integration tests; and
- opt-in real-Dorado E2E profile.

**Exit gate**

Two requests with the same floating alias but different resolved model bundles
cannot reuse one another, and an exact selected UUID set produces a validated
basecall manifest or fails without a reusable commit.

### SRB-05 — descendant raw generation and lineage transaction

**Status:** `SRB-05a` (transaction and provenance) is **merged** (PR #523,
`c22574d`; main `3e617e5`); `SRB-05b` (the real raw stage inside that
transaction) is implemented on `feature/rebasecall-lineage-raw-stage`
(`c248f64`). See the ledger rows.

**Scope amendment.** The third scope bullet below — "stage an ordinary
experiment run beneath a unique lineage root" — predates the artifact layout in
this document, which dropped the nested `lineages/<id>/run/` subtree in favour of
`stage_generations.json`. The implementation follows the layout: a descendant is
published into the parent experiment's ordinary stage directories, beside the
parent's generation and without advancing `current.json`, and the lineage records
which ids are its own. Only the *lineage container* is staged under a unique
root. This is `D1` in force — publication and selection are separate, and
selection stays with the project registry's `active_lineage`.

**Scope**

- Extend raw-generation provenance for full-source, parent-universe, and
  selected-cohort descendants.
- Add parent generation, selection, signal-resolution, basecall, and identity
  map dependencies.
- Stage an ordinary experiment run beneath a unique lineage root.
- Publish the complete outer lineage atomically and preserve the parent run.
- Prevent ordinary raw append semantics from treating a selected descendant as
  an append to the parent source universe.

**Primary areas**

- `informatics/raw_generation.py`;
- new processing-lineage publication/validation module;
- `cli/raw_adata.py`/`load_adata.py` integration;
- experiment workflow result contract; and
- failure-injection/rollback/relocation tests.

**Exit gate**

A killed basecall, raw stage, downstream stage, validation, or publish leaves
the parent and every prior complete lineage unchanged and discoverable.

### SRB-06 — downstream execution and QC transition reporting

**Status:** delivered in three parts — `SRB-06a` downstream execution mechanism
(PR #525), `SRB-06b` the QC transition report meeting the exit gate (PR #526),
and `SRB-06c` spatial/hmm/latent threading (`9d0aecf`). See the ledger rows.

**What the lane taught, worth carrying into `SRB-07`.** The same defect appeared
three times in different clothing: a descendant must not inherit the parent's
*selection* state. It showed up as the canonical stage-root spine being
overwritten (06a, via `staged_generation`'s hook), again as spatial and hmm
publishing that spine outside the hook (06c), and again as stages *skipping*
because the parent looked complete (06c). Any future kind that publishes a
descendant should be checked against all three, on a real run rather than by
inspection.

**Scope**

- Run existing semantic downstream targets inside the staged lineage.
- Recompute all sequence/alignment-dependent QC and dedup nodes.
- Publish origin-to-descendant row mapping and QC transition counts/reasons.
- Label cohort-dependent analyses and prevent claims of full-universe
  equivalence for selected lineages.
- Define behavior when a selected parent produces no new call, multiple split
  children, a different barcode, reference, or alignment.

**Primary areas**

- experiment semantic graph/workflow contract;
- preprocess generation and QC sidecars;
- experiment spine identity mapping;
- reporting/plot layout; and
- integration tests across modalities and target depths.

**Exit gate**

The terminal lineage report reconciles every selected origin molecule and can
reproduce parent-selection, new-raw, new-QC, and new-dedup counts exactly from
published artifacts.

### SRB-07 — project fan-out and lineage-aware registry

**Scope**

- Plan/run one lineage per selected project experiment.
- Resolve chemistry/model independently per experiment.
- Extend registry entries with backward-compatible lineage maps and one active
  selector.
- Add named project lineage sets if needed for publication review.
- Prohibit pooling two lineages of one experiment as replicates.
- Refresh stage/catalog pointers only on explicit promotion.

**Primary areas**

- `project/registry.py` and project graph;
- project workflow contract/CLI;
- materialization/source snapshot selection;
- registry schema migration; and
- multi-experiment/mixed-chemistry/duplicate-ID tests.

**Exit gate**

A project can hold original and re-basecalled lineages side by side, query each
explicitly, and change the active publication candidate atomically without
double-counting one experiment.

### SRB-08 — promotion, rollback, validation, and user documentation

**Scope**

- Add complete lineage validation, optional replayability validation, explicit
  promotion, and rollback-by-promotion.
- Document scope bias, refreshed QC, source retention, disk estimates, model
  pinning, direct-modification behavior, and migration.
- Add a publication checklist and machine-readable validation summary.
- Update release notes/config migration notes for user-facing CLI/schema
  changes.

**Required checks**

- warning-strict Sphinx build;
- command help/CLI criteria updates;
- relocation and failure-injection suites; and
- Ruff, unit, integration, and smoke gates appropriate to touched code.

**Exit gate**

Users cannot accidentally activate an incomplete lineage, and documentation
clearly distinguishes full-signal reanalysis from old-QC-selected reanalysis.

### SRB-09 — acceptance matrix and real-tool validation

**Scope**

- Add a versioned acceptance catalog resolving every finding and scenario to
  automated or explicitly owner-deferred evidence.
- Exercise fake-tool deterministic tests in ordinary CI.
- Exercise real Dorado/POD5 profiles when their tools/models are provisioned.
- Validate resource bounds and project fan-out on representative protected
  data outside public CI.

**Minimum scenarios**

- all POD5 signal versus all parent molecules yields the expected universe
  difference;
- exact QC predicate and explicit-ID selection;
- relocated multi-POD5 sources with duplicate basenames;
- namespaced identities and old split children resolved through `pi`;
- selected parent with no Dorado output and one parent with multiple new split
  children;
- `hac@latest` resolving differently after a model/tool change;
- canonical conversion/deaminase and direct modified-base workflows;
- changed alignment/barcode/QC/dedup outcomes;
- failure before and after lineage directory publication;
- project fan-out over at least two experiments and mixed compatible
  chemistries;
- original/new lineage query and explicit promotion/rollback;
- portable filtered-signal replay; and
- rejection of unsupported legacy chemistry or missing source signal.

**Exit gate**

Every `SRB-*` finding and the end-to-end publication workflow has executable
evidence or a named protected-data validation owner and reason.

## Dependency order

```text
SRB-01 request/planner
  -> SRB-02 origin identity
      -> SRB-03 source resolver
          -> SRB-04 Dorado adapter
              -> SRB-05 lineage/raw publication
                  -> SRB-06 downstream/QC transition
                      -> SRB-07 project registry
                          -> SRB-08 docs/promotion
                              -> SRB-09 acceptance
```

Small schema/test preparation may overlap, but no PR should expose selective
execution before exact identity resolution and missing-read failure semantics
are complete.

## Migration and compatibility policy

- Existing experiment/project commands and current pointers remain unchanged
  until a lineage is explicitly promoted.
- Existing raw/preprocess generations remain readable.
- Legacy project entries migrate logically to one `original` lineage without
  rewriting registered experiments.
- Historical rows lacking parent-read identity use validated BAM/POD5 fallback;
  they are never guessed from UUID shape or basename.
- `max_basecall_reads` remains a random development/testing cap and is not
  repurposed as publication selection.
- Existing `model: hac` configs retain their behavior for ordinary raw runs;
  re-basecall publication adds exact resolved-model provenance and may later
  motivate a separately reviewed improvement to ordinary basecalling.
- No promotion deletes historical lineages or source POD5 files.
- Source retention/deletion policy is a separate future program.

## Explicit non-goals

- Reconstructing raw signal from BAM/FASTQ when POD5/FAST5 is unavailable.
- Claiming a selected old-QC cohort is equivalent to re-basecalling all source
  signal.
- Copying old read/mapping/modification/variant QC onto new basecalls.
- Treating a floating model alias as reproducible publication identity.
- Searching arbitrary disks for matching POD5 files without user authority.
- Silently dropping unresolved selected reads.
- Registering a processing revision as an independent biological replicate.
- Pooling two lineages of one experiment in ordinary project materialization.
- Mutating a published raw/preprocess generation or completed lineage in place.
- Deleting source POD5s or historical lineages.
- Embedding an HPC/cloud scheduler inside smftools.
- Changing default QC thresholds, dedup algorithms, HMM semantics, or model
  fitting policies as part of this program.

## Decision gates resolved for SRB-01

1. Use the nested `experiment rebasecall plan|run|validate|promote` lifecycle.
2. Filtered POD5 materialization is optional (`signal.materialize: false` by
   default).
3. New-call read splitting defaults to `preserve`.
4. Promotion remains a separate explicit operation; no plan or run activates a
   lineage, and project review behavior is finalized with SRB-07.
5. Schema 1 uses bounded `all`/`any`/`not` predicates and safe scalar comparison,
   membership, and null operators over canonical QC mask columns only.
6. Schema 1 relocation maps are request-local. A project source catalog is added
   in SRB-03 only if checksum-based resolution proves it necessary.

These decisions do not alter the core safety contracts: exact parent
selection, checksum-validated signal, immutable resolved model identity,
refreshed QC, atomic lineage publication, and explicit promotion.

Two questions that would otherwise have landed in `SRB-01` are already settled
and should not be reopened here: where `generation_kind` lives (`D2`) and how
lineage selection relates to EGL's `current.json` (`D1`). Both are recorded in
[generation_lifecycle_and_naming_implementation_plan.md](generation_lifecycle_and_naming_implementation_plan.md)
and amended into the contracts above.

## Program completion definition

The program is complete when a user can:

1. inspect a year-old project's available raw/preprocess generations;
2. request all POD5 signal, all parent molecules, or an exact QC/ID subset;
3. obtain a read-only plan with counts, missing sources/IDs, model resolution,
   costs, and scientific scope warnings;
4. run only the selected POD5 UUIDs through a current compatible Dorado model;
5. receive an immutable descendant raw and requested downstream lineage;
6. audit the exact model, source bytes, origin/output identity mapping, and
   refreshed QC/dedup transitions;
7. compare original and new lineages without double-counting experiments;
8. validate or replay the result after supported relocation; and
9. explicitly promote or roll back the project's active publication lineage.

Until all nine outcomes are covered, the supported manual workaround should be
described as expert-only and must not be presented as a provenance-complete
publication workflow.
