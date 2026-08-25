# Experiment/project partitioned pipeline implementation plan

**Plan date:** 2026-07-21

**Repository:** `smftools`

**Branch at planning time:** `feature/general-improvements`

**Commit at planning time:** `0e4e4d249d877b6619bd65c16b3f80a15c6ed039`

**Source audit:** [experiment_project_partitioned_pipeline_audit.md](experiment_project_partitioned_pipeline_audit.md)

## Objective

Implement the audit recommendations through small, dependency-ordered changes that make `smftools experiment full` correct, bounded, queryable, relocatable, and reproducible across:

- raw -> preprocess -> spatial -> HMM, with later partition-aware stages inheriting the same contracts;
- locus and genome analysis modes;
- conversion, deaminase, and direct modalities;
- laptop, workstation, container, scheduler, and HPC resource profiles;
- experiment and project queries by molecule, reference interval, and barcode.

The program must preserve existing behavior unless a PR explicitly introduces a documented configuration/schema migration. Feature branches must not change `src/smftools/_version.py`.

## Agreed design contracts

These decisions are inputs to implementation, not open-ended suggestions.

### Storage and execution are independent

Persisted logical partitions, physical chunks, indexes, IDs, schemas, and checksums must not depend on the resources of the producing machine. Worker count, task fusion, memory ceilings, and in-flight limits are execution decisions recalculated by each consumer.

| Layer | Persisted? | Governing rule |
|---|---:|---|
| Logical partition | Yes | Deterministic reference/window/barcode organization and query pruning |
| Zarr chunk / Parquet row group | Yes | Portable independently readable unit sized against a minimum consumer profile |
| Compute batch | No | One or more portable units chosen from local memory headroom and algorithm expansion |
| Worker pool / in-flight set | No | Local requested/detected CPU, memory, task count, and I/O limits |

High-resource machines obtain throughput through concurrency, bounded task fusion, prefetch, store reuse, vectorization, node-local caching, and parallel writes to standard-sized output units. They do not create a dataset that requires equivalent resources to read.

### Resource resolution has two levels

- A run-level immutable `ResourceEnvelope` records requested and detected CPU/memory limits plus enforcement mode.
- A `PoolBudget` is recalculated before each pool using current memory headroom, current smftools process-tree use, stage-specific peak estimates, and the immutable envelope.
- CPU utilization and the number of live machine threads are not default hard-cap inputs. An optional cooperative/shared-machine policy may react to sustained load and must be explicit and logged.
- Running pools use bounded task admission rather than continuous pool resizing.

### HMM ownership is reference/barcode-specific

Users may fit distinct HMMs for each configured reference/barcode stratum. Read chunks within one stratum must share one immutable model artifact; they must never race to fit or overwrite it. Every HMM task output must record the exact model ID/checksum used.

### Genome mode has three BED scopes

| Scope | New configuration | Meaning |
|---|---|---|
| Alignment | `alignment_regions_bed` | Optional restriction of the FASTA/reference universe before alignment |
| Analysis | `analysis_regions_bed` | Shared downstream scope for preprocess, spatial, HMM, latent, and future analyses |
| Plotting | `plot_regions_bed` | Presentation-only intervals stitched from completed authoritative analysis cores |

All user BEDs use original FASTA coordinates. Reduced alignment references publish a mapping between stored local coordinates and original reference coordinates. Analysis cores are authoritative and non-overlapping; stage halos are context only. Plot intervals may span several analysis cores and must be stitched without duplicate boundary positions or ambiguous read/model provenance.

### Read identity is project-wide

Bare instrument `read_id` remains a data column. Project and cross-stage identity uses `(experiment_uid, read_id)` and, where useful, a deterministic compact `molecule_uid`. All scalar and derived indexes must preserve that identity.

### Publication is transactional

Task/stage artifacts are written to unique temporary locations, validated, and atomically published. A final spine is not a completion signal by itself. Compatible completed manifests drive restart/skip behavior.

## Delivery strategy

Use one focused branch/PR per item below. Do not combine HMM, BED scope, resource enforcement, and query-layout changes in one diff. Each PR names the audit IDs it addresses and carries forward compatibility readers until an explicit migration removes them.

The main dependency chain is:

```text
baseline/CI
    -> correctness hotfix
    -> artifact lifecycle
    -> resource envelope/budgets
    -> HMM ownership
    -> molecule identity
    -> region catalogs/shared planning
    -> lazy indexed readers
    -> plot stitching/streaming reductions
    -> observability and cross-machine acceptance
```

## Ordered PR backlog

| ID | Suggested branch | Primary outcome | Audit coverage | Depends on |
|---|---|---|---|---|
| PR-00 | `feature/partitioned-pipeline-baseline` | Dependency and CI/test baseline | H7 | None |
| PR-01 | `fix/genome-spatial-empty-regions` | Genome spatial succeeds without a plotting BED | C1 | PR-00 |
| PR-02 | `feature/stage-artifact-manifests` | Transactional artifact lifecycle and restart contract | H6 | PR-00 |
| PR-03 | `feature/resource-envelope` | Detected/requested resource contract and validation | H1, M4 | PR-00 |
| PR-04 | `feature/dynamic-pool-budget` | Per-pool admission, monitoring, and stage estimates | H1, H2, H5 | PR-03 |
| PR-05 | `feature/hmm-model-artifacts` | Immutable model IDs, atomic paths, and task provenance | C2, M2 | PR-02 |
| PR-06 | `feature/hmm-fit-apply-planning` | Single-owner fit followed by bounded chunked apply | C2, H2 | PR-04, PR-05 |
| PR-07 | `feature/molecule-identity-index` | Project molecule identity and derived read indexes | H3, H4, M3 | PR-02 |
| PR-08 | `feature/genome-region-catalogs` | Three BED schemas and original-coordinate mapping | H9 | PR-02 |
| PR-09 | `feature/shared-analysis-region-planner` | One inherited analysis scope across stages | H9 | PR-08 |
| PR-10 | `feature/lazy-partition-query` | Predicate-pruned, bounded base/derived reads | H3, H8, M2 | PR-03, PR-07, PR-09 |
| PR-11 | `feature/plot-region-stitching` | Plot-only cross-core assembly and manifests | H2, H3, H8, H9 | PR-04, PR-07, PR-09, PR-10 |
| PR-12 | `feature/streaming-project-reducers` | Bounded reducers, plots, and project export | H2, H5 | PR-04, PR-10 |
| PR-13 | `feature/progress-batch-results` | Complete progress/performance and batch status | M1, M5 | PR-02, PR-04 |
| PR-14 | `feature/partitioned-pipeline-acceptance` | Full modality/mode/cross-machine validation | All | PR-01 through PR-13 |

## Implementation status

| PR | Status | Branch | Verification summary |
|---|---|---|---|
| PR-00 | Implemented and merged in PR #385 | `feature/partitioned-pipeline-baseline` (completed) | Commit `dbafa44`, merged as `b94b59e`. Python 3.11 minimum storage stack: 31 passed; canonical smoke: 92 passed, 1 skipped; canonical unit: 855 passed, 9 skipped; Ruff and Sphinx `-W` clean. Dependency profiles and CI tiers followed in PRs #386 (`ff65dec`) and #387 (`9cd6463`). |
| PR-01 | Implemented and merged in PR #388 | `fix/genome-spatial-empty-regions` (completed) | Commit `132e469`, merged as `5511a1e`; genome spatial publishes valid output when no plotting regions are configured. |
| PR-02 | Implemented and merged in PR #389 | `feature/stage-artifact-manifests` (completed) | Commits `84297ef` and `ec4b89b`, merged as `976b3df`; transactional manifests and partitioned stage lifecycle are on `main`. |
| PR-03 | Implemented and merged in PR #390 | `feature/resource-envelope` (completed) | Commit `30cc26c`, merged as `79050ee`; detected/requested immutable `ResourceEnvelope` behavior is on `main`. |
| PR-04 | Implemented and merged in PR #391 | `feature/dynamic-pool-budget` (completed) | Commit `cc0152a`, merged to `main` as `43958ee`. Diff reviewed; focused `memory_guard` module: 32 passed; focused raw/plot modules: 51 passed; HMM partitioned module: 21 passed; Ruff and Sphinx `-W` clean. A larger combined macOS multiprocessing run exited without a pytest summary after reaching preprocess tests, with no assertion failure reported. |
| PR-05 | Implemented and merged in PR #392 | `feature/hmm-model-artifacts` (completed) | Commit `ab2f6e8`, merged to `main` as `ebd8418`. Immutable hashed model IDs, atomic checkpoint publication, model-level fit locking, checksums, training-selection provenance, forced-fit revisions, and task layer-to-model mappings. Focused HMM suite: 28 passed; Ruff and Sphinx `-W` clean. |
| PR-06 | Implemented and merged in PR #393 | `feature/hmm-fit-apply-planning` (completed) | Commit `6fa731b`, merged to `main` as `3c20034`. Explicit fit planning completes immutable per-group or shared-plus-adapted models before chunked apply. Deterministic selection defaults to at most 1,000 reads per fit and may be reduced by the task-memory ceiling. Focused planner/artifact/config tests: 25 passed; partitioned HMM tests: 23 passed; Ruff and Sphinx `-W` clean. |
| PR-07 | Implemented and merged in PR #394 | `feature/molecule-identity-index` (completed) | Commit `e7d6d7c`, merged to `main` as `38d7eeb`. Stable experiment/molecule identity, raw and derived indexes, collision-free pooled observation names, and index-only cross-stage lookup are on `main`. Unit suite: 925 passed, 9 skipped; Ruff and Sphinx `-W` clean. |
| PR-08 | Implemented and merged in PR #395 | `feature/genome-region-catalogs` (completed) | Commits `3f0af2f` and `2e71f74`, merged to `main` as `b36e011`; versioned alignment/analysis/plot BED catalogs and original-coordinate reference mapping are on `main`. |
| PR-09 | Implemented and merged in PR #396 | `feature/shared-analysis-region-planner` (completed) | Commit `7c0da28`, merged to `main` as `da61ada`; versioned shared authoritative-core planning is inherited by preprocess, spatial, HMM, latent, and shared stage input. |
| PR-10 | Implemented and merged in PR #397 | `feature/lazy-partition-query` (completed) | Commits `26b4b8c` and `ecabfba`, merged as `7f6496f`; predicate-pruned molecule/task indexes, pre-memory Zarr projection with minimal-install fallback, bounded physical chunks/query batches, and relocatable artifact paths. |
| PR-11 | Implemented and merged in PR #398 | `feature/plot-region-stitching` (completed) | Commits `5003aff`, `3cfd000`, and `51eff68`, merged as `8cbeb4e`; plot-only cross-core assembly, deterministic molecule selection, gap handling, and source manifests are on `main`. |
| PR-12 | Implemented and merged in PR #399 | `feature/streaming-project-reducers` (completed) | Commit `2b047cb`, merged to `main` as `e540382`; bounded reducers, pre-load plot subsampling, dense position-product ceilings, project preflight, and transactional partition-native project export are on `main`. |
| PR-13 | Implemented and merged in PR #400 | `feature/progress-batch-results` (completed) | Merge commit `64fb9e3`; lifecycle-scoped progress/performance logs, linked full summaries, and scheduler-visible batch results are on `main`. |
| PR-14 | Implemented and merged in PR #401 | `feature/partitioned-pipeline-acceptance` (completed) | Commits `a361d7c`, `4da5eb0`, and `c974abe`, merged to `main` as `baccc3e`; synthetic modality/mode and relocation coverage, the 40-criterion acceptance catalog, and portable platform/resource CI are on `main`. Thirty-two criteria are automated and eight post-implementation validations remain explicitly owner-deferred. |

### PR-04 implementation record — 2026-07-21

Implemented and merged contract:

- Added immutable, versioned `PoolBudget` snapshots using the run envelope, current system and cgroup headroom, recursive process-tree RSS, and private-memory accounting that avoids double-counting fork-shared pages.
- Re-resolve budgets before every central pool allocation, including replacement pools after watchdog failures, and before every bounded admission refill.
- Bound submitted futures while preserving input/result order; raw extraction and clustermap generators now retain backpressure rather than eagerly queueing all work.
- Added clear `PoolBudgetError` failure when one estimated task cannot fit and live preflights for sequential dispatch, reducers, plots, and major raw-stage external-tool boundaries.
- Added Linux watchdog fallback when the dedicated cgroup is unavailable, recursive worker-descendant sampling/termination, and sampling-only behavior when the cgroup is active.
- Added stage estimator names/version provenance plus predicted-versus-measured peak fields and calibration ratios in performance summaries.
- Wired task-plan estimates into preprocess, duplicate detection, spatial, and HMM dispatch without persisting machine-specific scheduling decisions into portable analysis artifacts.

Merge record:

- Feature commit: `cc0152a` (`feat: add dynamic pool budgets and bounded admission`).
- GitHub PR: #391.
- `main` merge commit: `43958ee`.
- The local combined macOS multiprocessing limitation remains recorded above; it did not indicate an assertion failure in the reviewed changes.

### PR-05 implementation record — 2026-07-21

Implemented and merged contract:

- Added canonical, versioned `HMMModelKey` identities covering fit ownership, reference, barcode, optional core bounds, signal label, architecture, fit configuration, and intentional forced-fit revision.
- Replaced fitted-model publication with short hashed paths safe across operating systems, atomic torch checkpoint writes, JSON commit metadata, file and semantic-content checksums, and conflict rejection for different content under one immutable ID.
- Added exclusive per-model fit locks with same-host abandoned-lock recovery so concurrent read chunks reuse one published checkpoint rather than racing to overwrite it. PR-06 will replace first-lock ownership with an explicit deterministic fit plan.
- Persisted training-selection digests without raw read identifiers and recorded relocatable artifact references, checksums, model IDs, and exact layer-to-model mappings in every partitioned HMM task Zarr and catalog row.
- Updated HMM parameter and convergence diagnostics to discover immutable hashed checkpoints while retaining legacy checkpoint discovery for older diagnostic files.
- Added focused coverage for collision resistance, effective fit-config hashing, checksum tampering, immutable-content conflicts, concurrent reuse, global-then-adapt artifacts, forced-fit revisions, and real task-store provenance. Verification: 28 focused tests passed; Ruff check and format check passed for touched files.

Merge record:

- Feature commit: `ab2f6e8` (`feat: add immutable HMM model artifacts`).
- GitHub PR: #392.
- `main` merge commit: `ebd8418`.

### PR-06 implementation record — 2026-07-22

Implemented contract:

- Added an explicit, deterministic fit plan that groups every configured model by its public `hmm_fit_strategy`, `hmm_groupby`, and `hmm_shared_scope` semantics before any apply worker starts.
- Added deterministic SHA-256-ranked fit membership with `hmm_max_fit_reads: 1000` and `hmm_fit_selection_seed: 0` defaults. Each fit uses all eligible reads when bounded, otherwise the smaller of the configured cap and the memory-safe read count derived from `target_task_memory_mb`.
- Added portable fit and fit-selection Parquet catalogs containing candidate/selected counts, selection digests and membership, resource bounds, fit dependencies, immutable model IDs, semantic model checksums, and relative artifact references.
- Separated HMM execution into base fits, dependent per-barcode adaptations for `shared_transitions`, and chunked immutable-artifact apply. Adaptation can update configured emissions/start probabilities while shared transitions remain frozen.
- Preserved the raw checkpoint-file SHA-256 for integrity checking and added a semantic model checksum for reproducible task provenance across equivalent fits whose Torch archive bytes differ.
- Added focused coverage for chunk-to-fit ownership, task-order invariance, configured and memory fit caps, shared base/adaptation assignment, invalid grouping, forced-fit revisioning, and config defaults/validation.

Verification:

- Planner, artifact, and config modules: 25 passed, 1 pre-existing warning.
- Partitioned HMM module: 23 passed; the sandboxed macOS run could not initialize multiprocessing semaphores, so the complete module was rerun outside that restriction and passed.
- Ruff check, Ruff format check, and `git diff --check`: clean.
- Sphinx warnings-as-errors HTML build: passed after enabling network access for external intersphinx inventories.

Merge record:

- Feature commit: `6fa731b` (`feat: separate HMM fit and apply planning`).
- GitHub PR: #393.
- `main` merge commit: `3c20034`.

### PR-07 implementation record — 2026-07-22

Implemented and merged contract:

- Added stable experiment and molecule identities while preserving the original instrument read ID.
- Published raw molecule indexes and bounded preprocess, spatial, and HMM read indexes with relocatable artifact pointers and immutable HMM model provenance.
- Made pooled observation identity collision-free and reversible across experiments, and added index-only cross-stage molecule lookup.
- Updated embedding/export joins, registry compatibility, schema registrations, and focused identity/index coverage.

Merge record:

- Feature commit: `e7d6d7c` (`feat: complete molecule identity catalog`).
- GitHub PR: #394.
- `main` merge commit: `38d7eeb`.

### PR-08 implementation record — 2026-07-22

Audit coverage: H9. Branch: `feature/genome-region-catalogs`.

Exact behavior:

- Add independent `alignment_regions_bed`, `analysis_regions_bed`, and `plot_regions_bed` configuration fields. All are interpreted as original-FASTA, 0-based, half-open coordinates.
- Normalize each configured BED into a versioned Parquet catalog with deterministic row ordering, stable content-derived region IDs, source-file SHA-256, preserved optional name/score/strand fields, and explicit overlap/adjacency annotations without merging records.
- Reject malformed rows, duplicate ambiguous names, invalid strands, non-positive intervals, coordinates outside original reference bounds, and references absent from the original FASTA with actionable errors.
- Publish `reference_interval_map.parquet` mapping every stored alignment/reference record to original FASTA coordinates. Direct and deaminase records map one-to-one; conversion records map every conversion/strand variant to the same original interval; alignment-reduced records retain their original interval offsets.
- Treat `fasta_regions_of_interest` as a deprecated alias for `alignment_regions_bed`. Equal values are accepted; conflicting simultaneous values fail instead of choosing silently.
- Keep `spatial_regions_bed` spatial-only. It is neither copied into `analysis_regions_bed` nor `plot_regions_bed`, and existing spatial execution behavior remains unchanged in PR-08.
- Publish catalogs and the reference map as relocatable raw-stage sidecars/provenance. PR-08 does not alter task planning; PR-09 consumes analysis catalogs and PR-11 consumes plot catalogs.

Compatibility boundary:

- Existing configs with only `fasta_regions_of_interest` retain alignment subsetting and receive the new alignment catalog plus deprecation warning.
- Existing configs with only `spatial_regions_bed` retain current spatial-only behavior.
- Runs without any BED fields publish a complete reference map and omit optional region catalogs.

Planned verification:

- Config default/override, alias, conflict, and legacy-spatial isolation tests.
- BED success, empty/comment-only, malformed, bounds, missing-reference, name, score, strand, overlap, and adjacency tests.
- Direct, deaminase, conversion-strand, reduced-reference, deterministic-ID/checksum, and relocatable-sidecar mapping tests.
- Focused raw/config suites, unit marker suite, Ruff, `git diff --check`, and Sphinx warnings-as-errors build.

Implemented result:

- Added typed independent alignment, analysis, and plot BED configuration plus conflict-safe `fasta_regions_of_interest` deprecation handling; legacy `spatial_regions_bed` remains spatial-only.
- Added schema-v1 deterministic BED3-BED6 catalogs with original-coordinate bounds validation, stable IDs, source SHA-256 metadata, explicit empty-catalog schemas, preserved record identity, and deterministic overlap/adjacency/name/score/strand semantics.
- Added schema-v1 `reference_interval_map.parquet` covering full and alignment-reduced references across direct, deaminase, and every conversion/strand alignment record, with original/stored bounds, length validation, source region identity, modality, and FASTA checksums.
- Published catalogs atomically as raw-stage sidecars, spine/experiment provenance, checksummed lifecycle artifacts, and project registry/catalog inputs. Registry schema advanced to 4 with backward-compatible reads.
- Documented configuration, migration, coordinate, error, and scope-consumption boundaries. PR-09 remains responsible for shared analysis task planning and PR-11 for plot-region stitching.

Verification:

- Pre-change config/legacy-spatial baseline: 18 passed.
- Focused config/catalog/raw lifecycle and storage suite: 85 passed.
- Focused mapping/project registry/catalog suite: 63 passed.
- A redundant post-format full unit rerun was stopped at user request after 656 passed, 9 skipped, and no failures; focused tests had already passed before mechanical formatting.
- Ruff check/format and `git diff --check`: clean.
- Sphinx warnings-as-errors HTML build: passed; generated build artifacts removed.

Merge record:

- Feature commits: `3f0af2f` (`feat: add genome region catalogs`) and `2e71f74` (`docs: mock pyarrow for autodoc`).
- GitHub PR: #395.
- `main` merge commit: `b36e011`.

### PR-09 implementation record — 2026-07-22

Implemented contract:

- Added a versioned shared analysis-region planner that consumes normalized catalog and reference-map pointers inherited from the experiment spine rather than reparsing configuration paths.
- Mapped original-FASTA analysis intervals into full or alignment-reduced stored coordinates, rejected analysis spans not covered by the alignment map, unioned overlapping records, and split ownership deterministically on portable storage-tile boundaries.
- Routed preprocess, spatial, HMM, latent, direct-modality Youden windows, and shared stage-input iteration through the same authoritative cores while allowing each consumer to request its required halo.
- Added stable analysis core IDs, source analysis region IDs, original-coordinate bounds, core/load bounds, and planner version to task metadata and task catalogs. Stage outputs continue to publish only authoritative core positions.
- Preserved full-reference locus behavior and legacy `spatial_regions_bed` behavior when no analysis catalog exists. Plot catalog changes do not alter the shared compute plan or downstream compute-stage compatibility hashes.
- Documented inherited planning and the compute/plot boundary.

Verification:

- Planner, preprocess dispatch, shared stage input, and latent coverage: 25 passed.
- Partitioned preprocess/spatial and HMM executor coverage: 44 passed outside the restricted macOS sandbox; the sandbox-only attempt had 38 passes and six semaphore-permission failures before worker creation.
- Stage lifecycle/hash coverage: 6 passed.
- Ruff checks and `git diff --check`: clean.
- Sphinx warnings-as-errors HTML build: passed; generated build artifacts removed.

Merge record:

- Feature commit: `7c0da28` (`feat: share analysis region planning across stages`).
- GitHub PR: #396.
- `main` merge commit: `da61ada`.

### PR-10 implementation record — 2026-07-22

Implemented contract:

- Added PyArrow dataset queries that push reference, genomic interval, sample/barcode, read-ID, and molecule-ID predicates into raw and derived Parquet indexes before task stores are opened.
- Made partitioned materialization automatically use lazy Zarr reads when available, project rows, genomic columns, and requested layers before `to_memory`, and assemble bounded row microbatches under a configurable query-memory allowance.
- Routed preprocess-owned `X`, preprocess/HMM layer overlays, and spatial read metrics through per-stage read indexes and stored `group_row` offsets, with backward-compatible catalog/eager fallbacks for older stores.
- Decoupled physical layout from logical task dimensions with bounded two-dimensional Zarr chunks and estimated 64 MiB Parquet row groups.
- Centralized relative artifact serialization/resolution, including absolute cross-volume fallback, and removed original-path dependencies when dense and raw stage trees are copied together.
- Added barcode fields to newly written molecule indexes and exposed barcode/molecule-UID selection through `materialize`.

Verification:

- Query, partition reader, raw store, relocation, chunk-layout, and path fallback coverage: 34 passed, plus focused barcode/molecule-index integration passed.
- Experiment spine, partitioned preprocess/spatial, and partitioned HMM integration: 49 passed outside the restricted macOS semaphore sandbox.
- Project registry/catalog/sample-store compatibility: 52 passed.
- Ruff format/check and `git diff --check`: clean.
- Sphinx warnings-as-errors HTML build: passed with intersphinx network access; generated build artifacts removed.

Commit record:

- Feature commit: `26b4b8c` (`feat: add lazy indexed partition queries`).
- CI compatibility commit: `ecabfba` (`fix: fall back when lazy dependencies are unavailable`). Python 3.11's minimum-storage environment omits optional `xarray`; explicit and automatic lazy requests retain eager fallback when that dependency is unavailable.
- GitHub PR: #397.
- `main` merge commit: `7f6496f`.

### PR-11 implementation record — 2026-07-22

Implemented contract:

- Added plot-region resolution that orders contributing authoritative cores, removes duplicated
  halo positions, aligns rows by stable molecule identity, and records source-task provenance.
- Added deterministic read selection before matrix loading, configurable labeled-`NaN` handling
  for unanalyzed gaps, and plot-source manifests for preprocess, spatial, and HMM outputs.
- Routed partitioned spatial and HMM plot generation through the shared stitching path while
  preserving full-reference and legacy behavior when no plot catalog is configured.
- Kept the chunk-materialization regression test in its parent process so its monkeypatched call
  recorder is portable and cannot fork a multithreaded pytest process on Linux.

Verification and merge record:

- Focused plot-region, partitioned executor, configuration, and stage-artifact coverage passed;
  Ruff format/check and `git diff --check` were clean.
- Feature commit: `5003aff` (`Add cross-core plot region stitching`).
- CI follow-ups: `3cfd000` and `51eff68`.
- GitHub PR: #398.
- `main` merge commit: `8cbeb4e`.

### PR-00 execution record — 2026-07-21

Pre-change baselines:

- The managed macOS sandbox produced 837 passes, 9 skips, and 17 failures. Fourteen failures were environmental: process pools could not query semaphore limits or bind local multiprocessing sockets.
- The same suite outside that sandbox produced 850 passes, 9 skips, and four code-behavior failures. Those four were stale plotting tests that still patched pre-lazy-import module paths or expected large matrices removed from return values by the memory-scaling refactor.

Implemented contract:

- Raised the package floor from Python 3.10 to Python 3.11 and aligned classifiers, Ruff, test CI, build CI, and installation guidance.
- Declared the production storage stack as `anndata>=0.12,<0.13`, `zarr>=3.1,<4`, `pyarrow>=15`, and `pandas>=2.1,<3`.
- The Pandas upper bound is required for the declared AnnData minimum: an isolated install initially selected Pandas 3, whose Arrow-backed string arrays AnnData 0.12.0 could not write to H5AD or Zarr. Pandas 2.1.0 resolves this incompatibility.
- Added a Python 3.11 CI job that installs exact storage minimums and exercises incremental Zarr, partition reads, raw Parquet stores, and the reusable fixture.
- Expanded normal CI from smoke-only to smoke plus the full unit marker set on Python 3.11 and 3.12.
- Added a deterministic eight-read fixture spanning two references and two barcodes. The builder supports conversion, deaminase, and direct modality labels and writes only relative store/BAM pointers.
- Updated the four stale plotting tests without restoring memory-heavy matrix return values or changing production behavior.

Local verification:

- Exact minimum stack on Python 3.11.9 (`anndata 0.12.0`, `zarr 3.1.0`, `pyarrow 15.0.0`, `pandas 2.1.0`): 31 passed; `pip check` clean.
- Canonical smoke suite: 92 passed, 1 skipped.
- Canonical unrestricted unit suite: 855 passed, 9 skipped, 102 deselected; no failures.
- `ruff check .` and `ruff format --check .`: clean.
- `sphinx-build -W -b html docs/source docs/_build/html`: clean when intersphinx network access is available.

Remaining PR-00 gate:

- Confirm the new CI jobs on GitHub's Linux runners. No E2E or real-experiment run is part of PR-00; those remain in PR-14 and will use isolated outputs beneath this repository's `dev/` tree as specified later in this plan.

## PR specifications

### PR-00: baseline, dependencies, and CI

Scope:

- Run the complete unit suite with `venvs/venv-all` and record failures by environment versus code behavior.
- Add the selected Parquet engine to the appropriate production dependency set.
- Set AnnData/Zarr bounds compatible with incremental Zarr v3 APIs and test the declared minimums.
- Make CI run unit tests in addition to smoke tests; keep external-tool E2E separately gated.
- Add a small deterministic partitioned-pipeline fixture used by later PRs.

Primary files:

- `pyproject.toml`
- `.github/workflows/ci.yml`
- `tests/conftest.py` and focused fixture helpers, following `tests/AGENTS.md`

Exit gate:

- Base/core installation can execute Parquet and Zarr production paths.
- Unit CI is green on Linux.
- Any platform-specific multiprocessing limitation is explicitly marked/gated rather than silently skipped.

### PR-01: genome spatial empty-region correctness

Scope:

- Give an empty dense-region catalog an explicit schema.
- Treat “genome mode with no plot BED” as valid: spatial tasks/reductions complete and dense products are absent.
- Publish a readable empty region catalog and completed stage manifest.

Primary files:

- `src/smftools/tools/partitioned_spatial.py`
- `tests/unit/test_spatial_partitioned_cli.py`
- a focused full-recipe integration test

Exit gate:

- Genome-only partitioned spatial succeeds with `spatial_regions_bed: null`.
- Existing locus behavior and legacy BED behavior remain unchanged.

### PR-02: stage and artifact lifecycle

Scope:

- Add versioned artifact/stage records with `planned`, `running`, `complete`, and `failed` states.
- Record config hash, input artifact IDs, schema version, expected/successful tasks, paths, checksums, timings, and outcome.
- Add atomic JSON/Parquet/H5AD/Zarr publication helpers where filesystem semantics permit.
- Make stage skip logic require a compatible `complete` record plus essential-artifact validation.
- Keep readers for current manifest/spine layouts.

Primary files:

- `src/smftools/informatics/experiment_manifest.py`
- `src/smftools/informatics/sidecar_manifest.py`
- `src/smftools/readwrite.py`
- stage wrappers under `src/smftools/cli/`
- project registry writers

Exit gate:

- Fault injection before final publication never produces a stage treated as complete.
- Restart reuses valid tasks and rejects corrupt/incompatible artifacts.

### PR-03: `ResourceEnvelope`

Scope:

- Detect logical CPU, process affinity, cgroup CPU quota, scheduler allocation, total/available memory, cgroup/job memory limit/current use, and platform enforcement capabilities.
- Resolve user values against detected hard limits.
- Validate percentages, byte/GB caps, threads, reserve, and task-memory settings.
- Log requested, detected, resolved, and enforcement values at command start.
- Centralize worker thread environment initialization.

Primary files:

- `src/smftools/memory_guard.py`
- `src/smftools/parallel_utils.py`
- `src/smftools/config/experiment_config.py`
- `src/smftools/config/default.yaml`
- `src/smftools/logging_utils.py` / `perf_log.py`

Exit gate:

- Resolved CPU never exceeds affinity/quota/scheduler allocation.
- Invalid resource values fail during configuration loading.
- Enforcement mode is explicit on Linux, macOS, and Windows.

### PR-04: dynamic `PoolBudget` and bounded admission

Scope:

- Recalculate current system/cgroup headroom and smftools process-tree use before every pool.
- Resolve stage-specific compute-batch size, worker count, and maximum in-flight tasks.
- Stop admitting work when headroom falls; fail clearly at the hard ceiling.
- Add Linux watchdog fallback when cgroup activation fails and recursive parent/child sampling elsewhere.
- Cover sequential execution, reducers, plotting, and external tools—not only pool workers.
- Introduce stage-specific peak estimators and estimator-version logging.

Exit gate:

- Synthetic over-budget process trees are constrained or fail cleanly.
- Results are invariant to worker count and task fusion.
- Logged predicted versus measured peaks are available for estimator calibration.

### PR-05: immutable HMM model artifacts

Scope:

- Introduce a collision-resistant model key/ID containing configured reference, barcode, optional core/window, label, architecture, and fit-config hash.
- Use reversible or hashed path components safe on Windows.
- Publish checkpoints atomically with schema, fit history, checksum, config hash, and training-selection metadata.
- Record `model_id`, checksum, and artifact reference in every HMM task Zarr and catalog row.
- Reject conflicting attempts to publish different content under one immutable ID.

Exit gate:

- Distinct references/barcodes/labels cannot collide on disk.
- Every persisted HMM layer can be traced to an immutable checkpoint.

### PR-06: HMM fit/apply separation

Scope:

- Group planned apply tasks by configured reference/barcode model key.
- Fit each key exactly once before apply workers start.
- Use all intended reads when bounded; otherwise use an explicitly configured deterministic bounded fit selection until a streaming-EM implementation is available.
- Apply the immutable artifact to read chunks in parallel.
- Align partitioned HMM configuration with the public `hmm_fit_strategy`, `hmm_groupby`, `hmm_shared_scope`, and adaptation fields.

Exit gate:

- Two or more read chunks in one stratum produce one fit and one checksum.
- One, two, and N workers produce identical task-to-model mappings and numerical results.
- `force_redo_hmm_fit` creates a new intentional artifact/version and never races an in-use checkpoint.

### PR-07: molecule identity and indexes

Scope:

- Add stable `experiment_uid` and deterministic `molecule_uid`; preserve original `read_id`.
- Enforce raw read-ID uniqueness across streaming flushes before final publication.
- Make pooled project `obs_names` unique and reversible.
- Add scalar molecule and derived read-to-task indexes with group path/row, reference/core, barcode, stage, schema, and model ID.
- Update embedding/export joins to use molecule identity.

Exit gate:

- Identical instrument read IDs in two experiments remain independently addressable.
- A molecule query locates all stages without opening every task store.

### PR-08: genome region catalogs

Scope:

- Add `alignment_regions_bed`, `analysis_regions_bed`, and `plot_regions_bed` config fields.
- Normalize 0-based half-open BEDs into versioned catalogs with stable IDs and source checksums.
- Publish `reference_interval_map.parquet` connecting stored/reduced/conversion-strand references to original FASTA coordinates.
- Preserve `fasta_regions_of_interest` as a deprecated alignment alias.
- Preserve legacy `spatial_regions_bed` behavior without silently converting it into pipeline-wide scope.

Exit gate:

- Every BED resolves through original coordinates for direct, conversion, and deaminase data.
- Overlap/adjacency/name/strand/error semantics are deterministic and documented.

### PR-09: shared analysis-region planner

Scope:

- Tile the union of analysis intervals into deterministic non-overlapping authoritative cores.
- Retain source region IDs while preventing duplicated computation from overlapping BED records.
- Let each stage request a stage-specific halo and publish core-only results.
- Route preprocess, spatial, HMM, latent, and shared stage input through the same inherited catalog.
- Record core/load bounds, analysis region IDs, and planner version in task catalogs.

Exit gate:

- All downstream stage catalogs cover the same requested analysis union.
- No authoritative position is duplicated or omitted at task boundaries.
- Changing only plot regions does not invalidate compute artifacts.

### PR-10: lazy indexed partition query

Scope:

- Query Parquet/DuckDB indexes before loading H5AD/Zarr.
- Push reference/window/barcode/molecule predicates into catalogs and datasets.
- Slice base, preprocess, spatial, and HMM Zarr arrays before `to_memory`.
- Assemble consumer-sized microbatches under the local resource envelope.
- Separate logical partition size from portable physical Zarr chunks/Parquet row groups.
- Centralize dataset-root-relative artifact references and cross-volume fallback behavior.

Initial chunk benchmarks, not permanent constants:

- Zarr: 8-32 MiB uncompressed per independently readable chunk.
- Parquet row groups: 32-128 MiB uncompressed.
- Parquet files: 128-512 MiB containing multiple row groups.

Exit gate:

- A small read/window/barcode query never eagerly loads a whole producer-sized task unless the result requires it.
- A copied/renamed dataset remains readable without original source BAM/POD5 paths.

### PR-11: plotting-only region stitching

Scope:

- Resolve each plot interval against completed authoritative task catalogs.
- Slice and stitch all overlapping cores in coordinate order, remove halo duplication, and align rows by molecule ID.
- Support a plot interval spanning any number of adjacent analysis windows.
- Fail by default on unanalyzed gaps; optionally render explicitly labeled `NaN` gaps.
- Microbatch barcodes/reads before loading and apply deterministic plot subsampling.
- Publish plot-to-source manifests with contributing tasks, layers, model IDs, selection seed, and artifact paths.

Exit gate:

- Cross-boundary plots contain the exact requested positions once each.
- Plot matrices and source manifests are invariant to analysis tile size, read chunk size, and worker count.

### PR-12: streaming reducers and project materialization

Scope:

- Replace list-plus-concat/vstack reductions with incremental, on-disk, or mergeable reducers.
- Subsample plots before loading their full source arrays.
- Bound dense position x position products by explicit byte/width limits.
- Preflight project materialization before allocating each part.
- Add partition-native project export that does not require a final in-memory `ad.concat`.

Exit gate:

- Reducers, plot builders, and project exports remain within the resolved budget.
- `allow_large` is not presented as compatible with a hard memory ceiling.

### PR-13: progress, performance, and batch status

Scope:

- Give raw its own lifecycle-scoped human and performance logs.
- Emit completed/total, throughput, ETA, per-task duration/retry, rows/bases, bytes read/written, and current/peak process-tree RSS.
- Consume futures in completion order while retaining deterministic result order.
- Link stage logs in a top-level `full` summary.
- Make batch commands return nonzero on partial failure and write a machine-readable summary.

Exit gate:

- Raw/preprocess/spatial/HMM skipped, failed, and completed outcomes are explicit.
- Batch schedulers can distinguish complete success from partial success.

### PR-14: acceptance and portability

Scope:

- Run the synthetic modality/mode matrix.
- Run the three real experiment validations described below.
- Transfer a completed tree from a high-resource profile to a low-resource profile and repeat indexed queries/plots without modifying authoritative artifacts.
- Add macOS and Windows storage/config/query smoke jobs and a capable Linux cgroup integration job.
- Capture cold/warm latency, files/chunks opened, bytes read, worker decisions, and peak RSS.

Exit gate:

- Every acceptance criterion in the audit is either automated and passing or explicitly deferred with owner/reason.

Implementation record — 2026-07-22:

- Expanded the deterministic raw-store fixture across conversion, deaminase, and direct modalities; locus and genome modes; top/bottom strand derivatives; two barcodes; and multiple physical shards per reference/barcode stratum.
- Added a relocation query test that copies the completed synthetic tree and performs an indexed molecule/window query from its new root.
- Added a schema-versioned 40-criterion acceptance catalog. Thirty-two criteria point to automated test evidence; eight are explicitly deferred with an owner and reason.
- Added macOS and Windows storage/config/query jobs plus a Linux runtime job that distinguishes real cgroup activation from the tested fallback and records cgroup capability data.
- Evidence-bearing suite: 221 passed and 2 platform/sandbox skips in the restricted run; all nine multiprocessing cases blocked by macOS semaphore restrictions passed outside that sandbox. Portable focused command: 39 passed and 2 local platform/sandbox skips. Ruff and diff checks are clean.
- Corrected the portable CI dependency profile to install both Xarray and Dask, which are required by ``anndata.experimental.read_lazy``. The exact clean Python 3.11 CI profile then passed: 39 passed, 2 platform/sandbox skips, and 9 deselected.

Merge record:

- Feature commits: `a361d7c` (`test: add partitioned pipeline acceptance coverage`), `4da5eb0`, and `c974abe` (portable lazy-query CI dependency corrections).
- GitHub PR: #401.
- `main` merge commit: `baccc3e`.

Program status:

- The dependency-ordered implementation backlog, PR-00 through PR-14, is complete and merged.
- The PR-14 exit gate is satisfied: all 40 audit acceptance criteria are either tied to automated test evidence or explicitly deferred with an owner and reason in `tests/acceptance/partitioned_pipeline_criteria.json`.
- Thirty-two criteria are automated. The following eight items remain as a post-implementation validation/operations backlog rather than unfinished implementation work:
  - protected real-data full runs for all modality/mode combinations;
  - real HMM numeric equivalence across 1, 2, and N workers;
  - representative high-depth end-to-end memory validation;
  - a true Linux HPC-to-macOS/Windows laptop tree transfer;
  - representative physical layout, decompression, and I/O amplification benchmarks;
  - HPC scaling validation while preserving portable physical output layout;
  - representative cold/warm query benchmarks with files/chunks opened, bytes read, and peak RSS;
  - exhaustive fault injection across every task, write, reduction, and plot boundary.

## Real-data validation program

Real data is local validation material, not a Git fixture. Do not commit absolute configs, BAM contents, read IDs, sample metadata, or generated outputs.

### Read-only source inventory

Experiment roots are lab-local and are not recorded here. They live in the
operator's validation manifest (see "Validation inputs" below); this table
records only what the plan needs to reason about -- the modality, the shape of
the input, and where the BAM sits *within* an experiment root.

| Modality | Template config | Basecalled BAM, relative to the experiment root | Size observed |
|---|---|---|---:|
| Direct | `experiment_config_v2_direct_12t_modkit_parallel_test.csv` | `<outputs>/raw_outputs/bam_outputs/hac_6mA_5mC_5hmC_calls.bam` | 420 MiB |
| Deaminase | `experiment_config.csv` | `<outputs>/informatics_outputs/bam_outputs/hac_canonical_basecalls.bam` | 1.7 GiB |
| Conversion | `experiment_config_v_full_conversion_12t_test.csv` | `<outputs>/informatics_outputs/bam_outputs/hac_canonical_basecalls.bam` | 241 MiB |

**Validation inputs.** Which experiment roots supply each modality is operator
state, not design rationale: it changes per machine and per lab, and naming
specific runs here would leak unpublished datasets into a tracked document.
Record the mapping in the analyses repository and point `SMFTOOLS_VALIDATION_ROOT`
at it.

The deaminase template currently names a root-level `hac_canonical_basecalls.bam`; the validation copy must point to the verified BAM under the run's `<outputs>/informatics_outputs/bam_outputs/`. Never edit the original template.

The observed FASTA records are locus-sized (approximately 0.2-4.7 kb). They are useful for natural locus validation. To exercise real-data genome paths and cross-window stitching without inventing molecular signals, validation configs should explicitly set `analysis_mode=genome` and a smaller `genome_tile_size` so each real reference spans multiple cores. This tests scheduling/storage semantics; results must be compared against a locus-mode materialization over the same requested coordinates.

### Isolated directory layout

The default validation root is repository-local and already excluded by the repository's `/dev/` ignore rule:

```text
<repo_root>/dev/real_data_validation/partitioned_pipeline/
    <git_short_sha>/
        validation_index.json
        <modality>/
            <mode>_<resource_profile>/
                config.csv
                run_manifest.json
                beds/
                    alignment_regions.bed
                    analysis_regions.bed
                    plot_regions.bed
                inputs/
                    optional deterministic smoke-subset BAMs
                outputs/
                comparisons/
                logs/
```

For this worktree, the default resolves to:

```text
<repo>/dev/real_data_validation/partitioned_pipeline/
```

Full/HPC validation may exceed the worktree's available storage or perform better on node-local/project scratch. In that case, set an explicit task-specific `SMFTOOLS_VALIDATION_ROOT` to an external scratch directory. Keep the generated config, validation manifest, comparison summaries, and an `artifact_root` pointer under the repository-local `dev/real_data_validation/` tree. A symlink may be added for local convenience, but manifests must not depend on the symlink for resolution.

Rules:

- Treat all three source experiment trees as read-only. Do not create validation outputs beside their existing canonical analyses.
- Default all configs, BEDs, smoke inputs, outputs, comparisons, and logs to the repository-local validation root.
- Use external scratch only through an explicit `SMFTOOLS_VALIDATION_ROOT`; record its resolved path and storage/filesystem identity in the local manifest.
- Never reuse an existing `output_directory` or edit an existing config.
- Never delete or overwrite an earlier validation run; use a new commit/scenario directory.
- Before writing, verify the output resolves below either the repository-local validation root or the explicitly configured scratch root and does not already contain an unrelated analysis.
- Put a validation sentinel/manifest in every created output root with commit, config checksum, source BAM path/stat identity, resource profile, command, and start/end outcome.
- Source BAMs are opened as inputs only. Alignment, indexes, splits, intermediate files, and analysis outputs go under the new validation output directory.
- Set deletion options false in validation configs unless a specific deletion behavior is under test.
- Never stage or commit real-data configs, BAM subsets, read identifiers, manifests containing absolute local paths, or generated results. The existing `/dev/` ignore rule is a guardrail, not permission to commit sensitive/local artifacts with `git add -f`.
- Repository-local validation writes stay inside the workspace. External scratch or HPC writes require the applicable execution approval at run time.

### Input preflight

Before each real run:

1. Confirm source BAM/config/FASTA/BED paths resolve and are readable.
2. Run `samtools quickcheck -v` or the configured pysam equivalent without modifying the BAM.
3. Inspect BAM header/read-group/barcode expectations and reference compatibility.
4. For direct data, sample records to confirm required MM/ML tags before selecting the modkit/pysam backend.
5. Record file size, modification time, header digest, and optional full checksum in `run_manifest.json`.
6. Validate the generated config through `ExperimentConfig` before starting external tools.
7. Print and confirm the resolved input, output, threads, memory, execution modes, and BED catalogs.

### Generated validation configs

Each validation config is a copied local template with only deliberate overrides. At minimum set or verify:

- `input_data_path`: preferred basecalled BAM above, not POD5/FAST5 and not an existing final H5AD;
- `output_directory`: unique path under the repository-local validation root or explicit scratch override;
- `experiment_name`: includes modality, mode, resource profile, and short commit;
- `threads`, `max_memory_gb`/`max_memory_percent`, and `target_task_memory_mb`;
- `analysis_mode`, `genome_tile_size`, and `genome_tile_halo`;
- `preprocess_execution_mode`, `spatial_execution_mode`, and `hmm_execution_mode`: `partitioned` for acceptance runs;
- alignment/analysis/plot BED fields appropriate to the scenario;
- modality-specific thresholds/backends inherited from the real template;
- all destructive intermediate-deletion flags disabled.

Do not use the direct template's existing `target_task_memory_mb=10240` for laptop validation. Resolve a bounded profile appropriate to the machine and record the override.

### Validation tiers

| Tier | Input | When | Purpose |
|---|---|---|---|
| A | Synthetic pytest fixture | Every PR | Fast correctness, edge cases, fault injection |
| B | Deterministic BAM subset from one modality | Relevant PRs | Real parser/tag/schema behavior with short feedback |
| C | Full preferred BAM for all three modalities | Milestone PRs 06, 11, 14 | End-to-end scientific/storage validation |
| D | Completed full tree copied between resource profiles/machines | PR-14 | HPC-to-laptop portability and query equivalence |

If a deterministic smoke BAM is needed, derive it under the scenario's validation `inputs/` directory from the preferred basecalled BAM using a recorded seed/selection method that preserves the header and modification tags. Never alter the source BAM. The subset manifest must record selected read IDs locally but must not be committed.

### Scenario matrix

The minimum real-data matrix is:

| Scenario | Direct | Deaminase | Conversion |
|---|---:|---:|---:|
| Natural locus, single worker | Required | Required | Required |
| Natural locus, detected parallel workers | Required | Required | Required |
| Forced genome mode, multiple cores | Required | Required | Required |
| Analysis BED spans multiple cores | Required | Required | Required |
| Plot BED contained in one core | Required | One modality sufficient before PR-14 | One modality sufficient before PR-14 |
| Plot BED spans at least two cores | Required | Required | Required |
| HMM reference/barcode has at least two read chunks | Required | Required | Required |
| Low-resource consumer query/plot | Required | Required | Required |

Resource-profile comparisons use identical authoritative inputs and analysis/plot BEDs. Only execution settings and output directory differ.

Suggested local profiles, always capped by detected availability:

- `single`: 1 worker and a conservative memory ceiling;
- `laptop`: at most 2 workers and approximately 4 GiB total ceiling if available;
- `parallel`: up to 8 workers or the detected allocation, with a recorded bounded ceiling;
- `hpc`: scheduler allocation and user cap, used only on the HPC.

### Comparison contract

Do not compare only whether commands exit zero. Record and compare:

- stage/task counts and completion manifests;
- molecule IDs/read IDs and per-reference/barcode counts;
- catalog schemas, core/load bounds, and region IDs;
- selected layer names, shapes, dtypes, absent-fill semantics, and deterministic sampled-value digests;
- HMM model IDs/checksums, training selection, and task-to-model mapping;
- plot interval coordinates, contributing task/model manifest, selected reads, and numeric matrix digest;
- files/chunks opened, logical/physical bytes read, wall time, throughput, and peak process-tree RSS;
- results across 1, 2, and N workers and across producer/consumer resource profiles.

Existing analysis outputs may be used as scientific reference points, but new storage layouts should be compared through explicit invariants and numeric summaries rather than byte-for-byte directory equality.

## Synthetic acceptance fixture

Build one compact deterministic fixture shared across PRs:

- two original references and conversion strand derivatives;
- two barcodes per reference;
- enough reads to force at least two read chunks per model stratum;
- adjacent and overlapping analysis BED records;
- a plotting interval within one core and another spanning at least two cores;
- reads that span boundaries, reads present in only one core, and an unanalyzed gap;
- duplicate instrument read IDs across two experiments;
- conversion, deaminase, and direct signal representations.

This fixture must support deterministic assertions without external Dorado/modkit binaries. External-tool behavior remains in separately marked integration/E2E tests.

## Decision register

The following choices must be confirmed before the named PR begins. Recommended defaults are recorded to prevent accidental divergence.

| Decision | Needed by | Recommended default |
|---|---|---|
| Minimum supported consumer | PR-03/PR-10 | 2 CPU, 4 GiB available RAM; benchmark an 8 GiB profile too |
| Preprocess semantics under analysis BED | PR-09 | Compute read-global QC once for reads overlapping the analysis union; publish positional results only in authoritative cores |
| HMM fit population when all reads exceed budget | PR-06 | Explicit deterministic cap with persisted membership/config, followed later by streaming EM |
| Plot interval crosses unanalyzed gap | PR-11 | Error by default; explicit labeled-`NaN` opt-in |
| Legacy `spatial_regions_bed` migration | PR-08 | Preserve current spatial-only behavior; require explicit opt-in to pipeline-wide analysis scope |
| Original versus reduced coordinates | PR-08 | User BEDs always use original FASTA coordinates; persisted map resolves stored coordinates |
| Portable chunk targets | PR-10 | Benchmark ranges from the audit; choose by measured uncompressed memory and I/O amplification |

## Per-PR working protocol

Before editing:

1. Re-read applicable root and nested `AGENTS.md`/`CLAUDE.md` instructions.
2. Update this plan with the active PR ID, audit IDs, exact behavior, compatibility boundary, and tests.
3. Inspect existing implementation/tests/docs before designing new modules.
4. Capture relevant synthetic and real-data baseline summaries.

During implementation:

1. Keep the diff scoped to one backlog item.
2. Add schema versions and backward readers with every persisted-format change.
3. Use config management for user-tunable values and constants for stable internal names.
4. Emit provenance/performance fields as part of the behavior, not as a follow-up.
5. Never weaken assertions or delete tests to make a change pass.

Before handoff:

1. Run focused unit tests, then smoke/unit markers appropriate to the change.
2. Run lint/format/type/docs checks in proportion to touched files and per nested instructions.
3. Run the required real-data validation tier only in a new isolated output directory.
4. Compare correctness plus resource/query metrics to the recorded baseline.
5. Update config docs, migration notes, and this plan's status/decision records.

## Definition of done

A backlog item is complete only when:

- its behavioral contract and audit IDs are explicit;
- focused tests cover success, empty input, invalid input, restart, and relevant cross-platform paths;
- persisted artifacts are versioned, traceable, relocatable, and backward-readable as promised;
- task/result identity is independent of worker count, task fusion, and producing machine;
- memory includes parent, workers, descendants, reducers, plots, and serialization where applicable;
- progress/performance decisions are inspectable;
- user-facing config changes include documentation and migration behavior;
- no release version is changed on the feature/fix branch;
- required real-data and synthetic acceptance evidence is recorded.

## Starting action

Begin with PR-00 to establish the dependency/test baseline, then PR-01 as the first behavior fix. Do not start HMM, BED-scope, or storage-layout changes until PR-02 through PR-04 provide artifact and resource primitives they need.

No real-data pipeline execution is part of creating this plan. The three experiment trees were inventoried read-only; test configs and validation outputs should be created under the repository-local validation root (or an explicitly recorded scratch override) only when the corresponding PR reaches its declared validation tier.
