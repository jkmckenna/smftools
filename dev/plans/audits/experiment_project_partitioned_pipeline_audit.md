# Experiment/project partitioned pipeline audit

> **Repository state reviewed:** `43958ee` — inferred: last commit on `main` on the stated audit date 2026-07-21.
> **427 commits on `main` since.** An audit describes the code at a moment; it
> goes stale rather than completing. Re-verify any specific claim before relying on it.

**Audit date:** 2026-07-21

**Repository:** `smftools`

**Branch:** `feature/general-improvements`

**Commit:** `0e4e4d249d877b6619bd65c16b3f80a15c6ed039`
**Primary scope:** `smftools experiment full` (`raw -> preprocess -> spatial -> hmm`), locus/genome modes, conversion/deaminase/direct modalities, project registration/materialization, partitioned storage and I/O, resource ceilings, progress/performance logging, portability, and read-level traceability.

## Executive conclusion

The new architecture is a strong foundation, but it does **not yet satisfy the full set of stated guarantees**.

The active pipeline now uses appropriate modern building blocks: read-relative Parquet shards for raw molecules, thin H5AD spines, task-local Zarr v3 stores, Parquet catalogs, bounded raw-extraction backpressure, incremental derived-layer writes, and project-level reference harmonization. These changes remove several previously monolithic data paths and are directionally correct for large experiments.

However, the audit found two release-blocking correctness problems and several high-severity scaling/operability gaps:

1. A genome-only spatial run with no BED regions fails with `KeyError: 'reference'`; therefore the default genome-mode `experiment full` path is currently broken.
2. Reference/barcode-specific HMM models are an appropriate and intended organization, and their logical identities are normally separated in checkpoint names. The defect occurs when one reference/barcode/window is split into read chunks: those chunks share one mutable checkpoint, can fit concurrently, race while overwriting it, and do not record which model version produced each task output. Even sequential execution fits the shared checkpoint from the first chunk rather than the complete reference/barcode stratum.
3. CPU and memory values constrain some Python worker pools, but they are not uniformly capped by detected usable CPU or currently available/cgroup memory. A configured memory value is not a cross-platform hard ceiling.
4. Task-local compute is partitioned, but several reducers, plot builders, dense-region products, HMM inputs, and project pooling paths can still scale with the entire selected dataset and exceed the task budget.
5. Raw and derived information is logically traceable by read ID, but fast lookup is incomplete: the raw barcode index is not consumed by the main reader, derived task catalogs do not map read IDs to task rows, and materialization scans the full spine plus candidate derived stores.
6. Cross-experiment read identity is ambiguous when two experiments contain the same read ID. The pooled object keeps an `experiment` column but leaves duplicate `obs_names`.
7. Completion/provenance records and performance logs are partial. Stage existence is often treated as completion, the experiment manifest only records raw, and Linux/sequential/raw memory observations are missing or can be attributed to the wrong stage.
8. The persisted formats are broadly portable, but resource portability is incomplete. Partitions planned on a large HPC can be opened eagerly as whole task stores on a laptop, and some paths become absolute. The producing machine's worker count and memory must remain provenance only; a consuming machine must independently plan bounded reads and compute without rewriting the authoritative dataset.
9. Genome BED semantics are incomplete and conflated. One BED can reduce the FASTA before alignment, while `spatial_regions_bed` simultaneously controls spatial compute windows and spatial dense products. There is no inherited BED scope shared by preprocess, spatial, HMM, latent, and future analyses, and no independent plotting BED capable of assembling one requested span across multiple completed analysis windows.

### Goal scorecard

| Desired outcome | Assessment | Summary |
|---|---|---|
| Modern, scalable storage structures | **Mostly met** | Parquet + Zarr + thin spines are appropriate; packaging/version and transactional-write gaps remain. |
| Partitioned compute and writing | **Partially met** | Main task execution is partitioned; several reductions, plots, model fitting, and project pooling are not bounded. |
| Resources flexible to user input | **Partially met** | Config values exist and influence central pools, but several paths bypass the central resolver. |
| CPU capped by detected availability | **Not met** | The central resolver does not inspect CPU count, affinity, cgroup quota, or scheduler allocation. |
| Memory capped by detected availability | **Not met as a guarantee** | Linux can get a hard cgroup cap when delegation succeeds; other/failure/sequential/parent paths are best-effort or unguarded. |
| Locus + genome support | **Not met** | Genome-only spatial currently fails without `spatial_regions_bed`. |
| Conversion + deaminase + direct support | **Partially verified** | Implementations exist, but the partitioned full-flow modality/mode matrix is not tested end to end. |
| Full read traceability | **Functionally partial** | Raw pointers and stage catalogs preserve lineage, but no stable project-wide molecule key or derived read-to-task index exists. |
| Fast reference-window x barcode retrieval | **Partially met** | Physical layout is promising, but active readers scan the full spine and do not consume the raw barcode index. |
| Progress/performance observability | **Partially met** | Human and JSONL logs exist for later stages; raw, Linux memory sampling, sequential work, reducers, and task progress are blind spots. |
| Multi-OS portability | **Partially met** | `pathlib`, spawn/forkserver handling, Parquet, HDF5, and Zarr help; paths, CI, dependency floors, and enforcement differ by OS. |
| HPC-to-laptop resource portability | **Partially met** | Logical partitions transfer, but eager derived-store reads, producer-sized task groups, absolute paths, and the lack of a consumer-side resource planner can make a transferred dataset impractical on a smaller machine. |
| Genome alignment/analysis/plot scopes | **Not met** | Alignment BED support exists and spatial has one combined compute/plot BED; a pipeline-wide analysis BED, independent plot BED, and original-coordinate mapping are absent. |

## What is working well

The following should be preserved while addressing the findings:

- Raw molecule arrays are stored in read coordinates and scattered onto a reference window only on demand (`informatics/ragged_store.py:1-6`). Integer arrays are narrowed to `int8` and floating signal arrays to `float32` (`ragged_store.py:185-263`).
- Raw shards are organized by escaped reference and coarse start bin, then sorted by barcode/read ID within each incoming group (`informatics/raw_store.py:126-128`, `raw_store.py:229-301`).
- Raw extraction uses bounded read buckets, a bounded parent accumulator, and at most `max_workers` in-flight results (`cli/raw_adata.py:680-750`). This directly addresses result-queue memory growth.
- The raw spine retains per-read `ragged_shard` and `ragged_row` pointers plus source BAM metadata (`informatics/raw_store.py:82-123`).
- Selected read IDs are pushed into PyArrow Parquet filters rather than filtering all decoded ragged rows in pandas (`informatics/ragged_store.py:274-301`).
- Preprocess, spatial, and HMM tasks are divided by reference, genomic core/halo, barcode, and read chunk (`preprocessing/dispatch_plan.py:82-160`).
- Preprocess and HMM derived layers, and spatial read matrices, are appended to Zarr incrementally instead of retaining a second complete output object (`preprocessing/partitioned_executor.py:315-330`; `tools/partitioned_hmm.py:196-251`; `tools/partitioned_spatial.py:402-431`).
- Preprocess uses a staging spine during duplicate reduction, addressing a real prior partial-completion failure (`preprocessing/partitioned_executor.py:1009-1031`).
- Stage pointers placed in spines are generally relative to the run root, and the consolidated experiment spine unions sibling spatial/HMM catalogs (`informatics/experiment_spine.py:10-27`, `experiment_spine.py:90-103`).
- Project registry paths are stored relative to the project and resolved centrally (`project/registry.py:153-180`, `registry.py:398-431`).
- Project set access provides a streamed, projected API that yields one experiment slice at a time (`project/set_store.py:94-155`). This is preferable to constructing a monolithic project object.
- Worker pools retry unfinished work with fewer processes after a broken pool, preserving completed results (`memory_guard.py:458-478`, `memory_guard.py:519-578`).
- Worker BLAS/OpenMP limits and non-GUI plotting defaults show good awareness of nested parallelism, although the implementations should be consolidated (`parallel_utils.py:22-93`; `memory_guard.py:379-405`).

## Current data flow and storage model

| Stage | Primary persisted structures | Selection/lineage mechanism | Main scaling behavior |
|---|---|---|---|
| Raw | Parquet ragged shards, `interval_catalog.parquet`, `barcode_index.parquet`, run-level `molecules.parquet`, `obs.parquet`, thin `spine.h5ad` | `read_id -> ragged_shard/ragged_row`; source BAM path in obs; reference plans in `uns` | Ragged arrays are bounded by input group, but scalar/index state remains O(total reads). |
| Preprocess | Task Zarr stores, task/catalog Parquet, reduced var/obs Parquet, thin spine | Catalog identifies reference/window/barcode/chunk store; spine points to catalog | Task compute is bounded; reductions and global duplicate state scale with all tasks/reads. |
| Spatial | Task Parquet summaries, read-metric Zarr stores, matrix sidecars, plot catalog, thin spine | Task catalog plus per-task `obs_names` | Task compute is bounded; global plotting/reduction and dense region products are not consistently bounded. |
| HMM | Task Zarr stores, task catalog, model checkpoints, plots, thin spine | Task catalog plus per-task `obs_names`; checkpoint filename encodes scope-like fields | Apply is chunked, but fit/checkpoint semantics are unsafe across chunks. Plot reductions can be global. |
| Experiment spine | H5AD obs joined from raw/preprocess; union of stage `uns` pointers | Read ID index and unioned catalogs | Loads all scalar obs into memory. Spatial/HMM read arrays stay in sidecars. |
| Project | JSON registry, reference UID/alias table, per-sample pointers, optional DuckDB query over reference/interval tables | Experiment ID + canonical reference UID + selected stage spine | Streamed set access is good; pooled materialization accumulates all parts and has a post-allocation guard. |

`cli/recipes.py:28-33` implements `experiment full` as a simple sequential call chain. This gives stage-level restart semantics, but it does not provide a transaction or a top-level completion record for the four-stage workflow.

For HMMs, this audit treats the ability to use a distinct model per reference/barcode as intended behavior. A genomic core/window and signal label may further qualify that model where explicitly intended. It does not recommend collapsing models across references or barcodes. The concern is whether every task has an unambiguous, immutable link to the correct model for its own stratum and whether any worker can overwrite that model after derived analyses have been written.

## Detailed findings

### Critical C1: genome-only spatial analysis fails without a BED file

**Evidence**

- `_dense_product_regions` adds full-reference regions only for locus plans and BED regions only for genome plans (`tools/partitioned_spatial.py:1145-1169`).
- When all references are genome-mode and `spatial_regions_bed` is unset, `records` is empty. The function creates `pd.DataFrame(records)` with no columns and immediately sorts on `reference/start/end` (`partitioned_spatial.py:1171`).
- `execute_partitioned_spatial` calls this unconditionally after task reduction (`partitioned_spatial.py:1473-1477`).
- An isolated runtime probe reproduced `KeyError: 'reference'` for one genome reference and an empty BED-region table.

**Impact**

- `smftools experiment full` cannot complete for a genome-only experiment under the default `spatial_regions_bed: null` configuration.
- This fails after spatial tasks and reductions have already written output, increasing restart ambiguity and wasted compute.

**Required correction**

- Construct the empty region catalog with an explicit schema/column list and return it without sorting, or guard the sort when empty.
- Confirm that “no dense products requested for genome mode without BED regions” is the intended behavior. If tiled dense products are desired, plan them explicitly rather than silently reusing full-locus behavior.
- Add a partitioned spatial test with only genome references and no BED input, and a full-flow test that reaches HMM from that case.

### Critical C2: HMM checkpoint ownership and publication are unsafe across read chunks

**Evidence**

- Under the active default `per_sample` behavior, the intended reference/barcode separation is present: `_annotate_task` passes `task.barcode` as `sample`, and derives `model_reference` from the reference plus core coordinates; the HMM signal/output prefix supplies the label (`tools/partitioned_hmm.py:92-110`, `partitioned_hmm.py:153-166`). Thus ordinary models for different references, barcodes, cores, or labels have distinct logical checkpoint keys.
- HMM tasks reuse the preprocessing task planner, which can split one reference/barcode/core into multiple read chunks (`preprocessing/dispatch_plan.py:134-159`; `tools/partitioned_hmm.py:864-869`). `chunk_index` is intentionally absent from the model key because those chunks belong to the same reference/barcode model, but there is no preceding single-owner fit step for that shared model.
- `HMMTrainer._path` combines kind, sample/barcode, reference/core, and label, so every read chunk in one intended model stratum targets the same checkpoint (`cli/hmm_adata.py:432-458`). Writes use direct `torch.save`, not a temporary file plus atomic replace.
- CPU HMM tasks are allowed to run in parallel (`partitioned_hmm.py:875-893`). Two chunks can both observe a missing checkpoint, fit different models from different reads, annotate their own data, and race while overwriting the same `.pt` file.
- Even in sequential mode, the first chunk trains the reference/barcode checkpoint and later chunks load it. The saved model is therefore fit from only the first read chunk, not the complete intended reference/barcode stratum. `force_redo_hmm_fit=True` makes every chunk refit and overwrite the same path even sequentially.
- HMM task Zarr metadata and the task catalog record task/reference/barcode/core/layers, but no model path, immutable model ID, checksum, fit-input identity, or model schema/config hash (`tools/partitioned_hmm.py:225-260`). After a checkpoint overwrite, the persisted task output cannot prove which fitted parameters produced it.
- Filename normalization is not injective: `_path` only changes `/` to `_`, so distinct logical identifiers such as `a/b` and `a_b` can resolve to the same model filename (`cli/hmm_adata.py:432-437`). This is an additional cross-stratum overwrite risk for unusual but valid reference/barcode/label names.
- Separately, the partitioned implementation reads `hmm_fit_scope` (`per_sample/global/global_then_adapt`), but this field is not declared or parsed by `ExperimentConfig`. The public config instead declares `hmm_fit_strategy`, `hmm_shared_scope`, `hmm_groupby`, and adaptation fields (`config/default.yaml:394-402`; `config/experiment_config.py:1103-1111`). That mismatch makes intended model ownership/configuration difficult to validate, even though per-reference/barcode fitting itself is desirable.

**Impact**

- Models for ordinary distinct reference/barcode strata are separated as intended; the problem is not the existence of those distinct fits.
- Read chunks within one reference/barcode stratum can be annotated using different fitted parameters, depending on chunking, scheduling order, and worker timing.
- The surviving checkpoint may not be the model used for all published layers, and the task catalog cannot reconstruct the mapping. Later parameter plots may therefore describe the last writer rather than the model that produced every task result.
- Non-injective filenames can allow two distinct logical strata to overwrite one another.
- User-configured ownership/shared-transition behavior cannot be validated reliably because the active and declared config vocabularies differ.

**Required correction**

Preserve distinct reference/barcode models, but give each intended model stratum a single owner through a two-phase plan:

1. Resolve explicit model keys such as `(reference, barcode, HMM label, architecture, fit-config hash)`, with core/window included only when intended, before dispatching apply tasks.
2. Fit exactly one model per key using all intended reads for that reference/barcode stratum, a deterministic bounded subsample, or a documented streaming sufficient-statistic reducer.
3. Publish an immutable, collision-resistant model artifact atomically and record its model ID/checksum plus fit-input selection metadata.
4. Dispatch chunked apply tasks that only load that artifact; every Zarr task and catalog row must record the assigned model ID/checksum.
5. Make the partitioned config vocabulary match the existing public HMM vocabulary, or introduce a documented migration with validation.

Acceptance tests must create at least two references, two barcodes per reference, and two read chunks per reference/barcode/core. They should assert distinct models between reference/barcode strata, one immutable model shared by the chunks within each stratum, no checkpoint overwrites, a persisted task-to-model mapping, identical results across worker counts, and deterministic reruns. Include identifier pairs that would collide under slash-to-underscore normalization.

### High H1: resource values are scheduling hints, not a uniform detected-resource ceiling

#### CPU

- `resolve_max_workers` caps by configured threads, item count, and estimated memory, but not `os.cpu_count()`, process affinity, cgroup `cpu.max`, container CPU quota, or scheduler allocation (`memory_guard.py:77-96`).
- Several external-tool and library calls receive `cfg.threads` directly, including alignment, demultiplexing, BAM processing, and plotting (`cli/load_adata.py:581-615`, `load_adata.py:1009-1047`; `tools/partitioned_spatial.py:1340-1396`; `tools/partitioned_hmm.py:799-803`).
- POD5 metadata extraction can start a `ProcessPoolExecutor(n_jobs=cfg.threads)` inside each raw extraction worker (`cli/raw_adata.py:394-417`, `informatics/pod5_functions.py:488-538`), allowing multiplicative nested process counts.
- Two separate worker initializers exist. `parallel_utils.configure_worker_threads` caps BLIS, Accelerate, Torch, and Matplotlib, while `memory_guard._limit_blas_threads_in_worker` omits those controls (`parallel_utils.py:63-93`; `memory_guard.py:379-405`). Most central task pools use the narrower initializer.

#### Memory

- The budget is based on total physical RAM, not currently available memory, cgroup/container headroom, or current process-tree use (`memory_guard.py:46-74`).
- Negative/zero/over-100 percent and nonpositive GB settings are not rejected; the resolver can silently clamp an invalid result to one byte (`memory_guard.py:64-74`). `ExperimentConfig.validate` does not validate resource fields (`config/experiment_config.py:2441-2484`), and the CLI config loader does not call `validate` (`cli/helpers.py:367-388`).
- Linux has a real aggregate cap only if creating/delegating a cgroup v2 child succeeds (`memory_guard.py:99-200`). Setup intentionally fails open.
- The worker watchdog unconditionally becomes a no-op on Linux (`memory_guard.py:245-273`), even when cgroup setup failed. Linux then has no enforcement fallback.
- On macOS/Windows the watchdog samples and kills only direct pool workers. It does not enforce a parent + descendants aggregate, does not include external tools or nested child pools, adds 20% per-worker tolerance, and permits three over-budget polls (`memory_guard.py:285-358`).
- Sequential execution has no RSS watchdog (`memory_guard.py:494-508`). Reductions, plots, config loading, spine loading, HDF5/Zarr writes, and external commands outside guarded pools are not covered.

**Impact**

The current design can reduce likely memory use, but it cannot promise “the workflow will remain below the user ceiling” on all supported operating systems. It also cannot promise that requested CPU is capped by CPU actually assigned to the process.

**Required correction**

Introduce one resolved resource object used everywhere, containing at least:

- detected logical CPU, affinity/quota/scheduler CPU, requested CPU, and final CPU;
- physical memory, `psutil.available`, cgroup/job limit and current use where available, requested cap, reserve, and final usable budget;
- enforcement mode (`cgroup`, Windows Job Object if implemented, tree watchdog, advisory) and whether it successfully activated.

Use this as an immutable run-level `ResourceEnvelope`, then derive a new `PoolBudget` before every process-pool allocation. The pool budget should snapshot current system-available memory, cgroup/job headroom, and current smftools process-tree RSS; subtract parent/output reserves; apply a stage-specific per-task peak estimate; and resolve worker count plus maximum in-flight work. It may shrink later pools as memory pressure changes, but it must never exceed the run-level user/detected ceiling.

The final worker/tool thread count should be the minimum of requested and detected usable CPU. Instantaneous machine CPU utilization or the number of live threads should not normally redefine this hard cap: those signals are transient and the OS scheduler already manages contention. An opt-in shared-machine/cooperative mode may sample sustained load and delay/reduce new work, but it should be logged because it reduces reproducibility. Linux should run the tree watchdog when cgroup activation fails. Non-Linux monitoring should include parent plus recursive descendants and should distinguish monitoring from enforceable hard limits in logs and documentation.

Do not continuously resize a live process pool. Recalculate before each pool, then use bounded task admission while it runs: stop launching new work when dynamic memory headroom falls, and fail cleanly if the process tree reaches its hard ceiling.

### High H2: stage-specific memory estimates and post-task work are not bounded

The task planner uses a single constant of eight bytes per loaded read-position (`preprocessing/dispatch_plan.py:13`, `dispatch_plan.py:134-156`). This approximates the base raw dense grids without optional signal layers, but it is reused for work with materially different live sets:

- Raw materialization can include four base layers plus any signal-feature layers (`informatics/ragged_store.py:404-419`, `ragged_store.py:513-528`).
- HMM materializes with `layers=None`, loading all source and available derived layers before annotation (`tools/partitioned_hmm.py:196-221`; `informatics/partition_read.py:822-830`). HMM annotation itself retains all output layers until annotation returns (`partitioned_hmm.py:199-207`).
- Spatial creates read x lag autocorrelation/count matrices and read x periodogram matrices in addition to X (`tools/partitioned_spatial.py:80-137`). These sizes depend on `autocorr_max_lag`, period range, and site types, not the reference-window width alone.
- Preprocess coverage reduction reads every task Zarr into lists of DataFrames and concatenates them (`preprocessing/partitioned_executor.py:393-452`). Read-stat reduction follows the same all-partials pattern (`partitioned_executor.py:575-593`).
- Spatial reduction reads all task Parquets into lists and concatenates (`tools/partitioned_spatial.py:448-500`).
- Spatial read clustermaps load and accumulate every chunk for a reference/window/barcode, then `vstack` before applying the plot row cap (`partitioned_spatial.py:692-803`). Periodicity plotting similarly accumulates and restacks all read power arrays (`partitioned_spatial.py:836-881`, `partitioned_spatial.py:994-1018`).
- Spatial dense-region generation materializes every qualifying read for an entire locus/BED region, then may build position x position matrices (`partitioned_spatial.py:1267-1401`). A read x position budget does not cover O(position²) products.
- HMM feature count/size plotting builds one Python record per read/layer before making a DataFrame (`tools/partitioned_hmm.py:342-400`).
- Raw streaming still retains O(total reads) scalar structures: molecule rows, barcode index rows, two read-pointer dicts, and all obs frames (`informatics/raw_store.py:201-206`, `raw_store.py:554-613`). Ragged memory is bounded, but total parent memory is not experiment-size-independent.

**Required correction**

- Define separate memory estimators for raw, preprocess, spatial, HMM fit, HMM apply, dense-region matrices, plotting, and project pooling.
- Include selected layer dtypes/counts, task output arrays, transient copies, metadata, and expected serializer overhead.
- Replace list + concat/vstack reducers with incremental aggregation, partitioned output, bounded reservoirs, or on-disk SQL/DuckDB reductions.
- Subsample before loading/vstacking plot data, not afterward.
- Set explicit maximum region width/position-matrix bytes and split or refuse over-budget P x P products.
- Allow a too-large single task to be subdivided automatically; reducing only the number of concurrent workers cannot fix a task that individually exceeds the budget.

### High H3: read traceability exists, but fast read and reference/barcode lookup is incomplete

#### Good lineage already present

- Raw spine rows map read IDs to ragged shards/rows and include reference interval, sample/barcode aliases, and source BAM path (`informatics/raw_store.py:82-123`).
- Stage stores retain read IDs as Zarr obs indices.
- The consolidated experiment spine unions preprocess, spatial, and HMM catalog pointers (`informatics/experiment_spine.py:90-103`).
- A run-level `molecules.parquet` gives a canonical raw row, reference, and sample (`informatics/raw_store.py:255-271`, `raw_store.py:598-600`).

#### Missing fast paths

- `materialize` first loads the entire H5AD spine and filters its pandas obs arrays for reference, interval, sample, and read ID (`informatics/partition_read.py:118-130`, `partition_read.py:163-197`). Query setup is therefore O(total experiment reads) in memory/time.
- The raw `barcode_index.parquet` is written and registered (`informatics/raw_store.py:283-301`, `raw_store.py:601-632`), but no production reader consumes it. Repository search found no use outside its writer/tests.
- The raw interval catalog is exposed to the project/DuckDB catalog, but the main materializer selects shards through full-spine per-read pointers rather than interval-catalog predicate pushdown.
- Preprocess/spatial/HMM task catalogs omit their `read_ids`; the only membership copy is inside each task Zarr obs. Catalog records include reference, core, barcode, and chunk but not a read-ID range, hash bucket, or per-read pointer (`preprocessing/dispatch_plan.py:33-38`; `tools/partitioned_spatial.py:433-445`; `tools/partitioned_hmm.py:255-260`).
- Derived-layer overlay filters catalogs by reference/window, but not selected barcode, then opens candidate Zarr stores and tests read-ID overlap (`informatics/partition_read.py:544-584`). Spatial read-metric overlay similarly opens every reference-matching task store (`partition_read.py:644-698`).
- The project’s modern per-sample store is a pointer/count catalog, not an alternate indexed read path; it ultimately returns the caller to full-spine `materialize` (`project/sample_store.py:51-68`, `sample_store.py:118-143`).

**Impact**

A read can usually be reconstructed if its experiment and read ID are known, but lookup cost grows with the full spine and number of derived task stores. The target workload—one reference window across barcodes—does not yet exploit the indexes already written.

**Recommended storage/query contract**

- Define a stable molecule primary key: `(experiment_uid, read_id)` plus an optional compact `molecule_uid` hash. Never treat bare `read_id` as project-global.
- Make a Parquet molecule dataset the query index, partitioned by canonical/exact reference and coarse start bin, sorted by barcode, interval, and molecule key. Include raw shard/row pointers.
- Add a derived read index with at least `(molecule_uid, stage, reference, core_start, core_end, barcode, group_path, group_row, model_id/schema_version)`; partition it by stage/reference/core or store a compact read-hash bucket.
- Query this index through `pyarrow.dataset` or DuckDB predicate pushdown before opening H5AD or Zarr.
- Keep Zarr for dense task arrays and Parquet for searchable scalar facts/indexes; do not duplicate entire dense matrices per project.
- Benchmark the primary query explicitly: a 10-100 kb window across all barcodes, a barcode subset within that window, and one read across all stages.

### High H4: project-wide read identity is ambiguous

- `normalize_part` adds `obs["experiment"]` but does not namespace the obs index (`project/set_store.py:77-91`).
- `project_adata` concatenates with the default `index_unique=None` (`project/catalog.py:288-290`).
- An isolated probe created two experiments with the same raw read ID. The pooled output had `obs_names == ['shared-read', 'shared-read']`, `obs_names.is_unique == False`, while the experiment column correctly contained both experiment IDs.
- Downstream project embedding code stores and compares `obs_names` without an experiment component (`project/embedding_store.py:129-139`, `embedding_store.py:290-350`), so collisions can conflate reads.

**Impact**

The user cannot unambiguously name or retrieve a pooled read with bare `obs_names`, and project analyses that map by read name may silently merge or overwrite records.

**Required correction**

- Adopt the project-wide molecule key described under H3.
- Preserve the original instrument read ID in a dedicated `read_id` column.
- Use a reversible composite key or immutable UID as pooled `obs_names`.
- Add collision tests for materialization, embeddings, exports, sample analysis, and write/read round trips.

### High H5: pooled project memory guardrails act after allocation and are incomplete

- `project_adata` materializes each experiment part, appends it, and only then adds its estimated bytes and rejects it (`project/catalog.py:257-280`). One oversized part is already resident when the guard fires.
- `_part_nbytes` counts only X and layers, omitting obs, obsm/read metrics, var, uns, sparse/lazy backing behavior, and concatenation temporaries (`project/catalog.py:207-212`).
- The final `ad.concat` can allocate another pooled object and perform outer-join dtype expansion after the running estimate passes (`project/catalog.py:288-290`).
- The CLI exposes only a fixed ~8 GiB threshold and `--allow-large`; it does not accept the experiment resource settings, detected available memory, a caller-provided byte ceiling, or streamed output (`cli_entry.py:622-695`).
- `iter_set_parts` is genuinely streamed, but defaults `layers=None` (all layers), and the CLI materialize path accumulates every yielded part (`project/set_store.py:108-119`; `project/catalog.py:257-273`).

**Required correction**

- Estimate a selection before materialization from catalog shape/dtypes and reject early.
- Use the same resolved resource object as experiment stages.
- Expose a configurable project memory limit and a partitioned project export that writes each experiment/barcode/window without a final in-memory concat.
- Treat `allow_large` as bypassing a warning, not as compatible with a memory ceiling.
- Account for the destination/concat allocation and read metrics in the estimate.

### High H6: completion, restart, and provenance are not transactional across the full workflow

- Spatial and HMM wrappers skip based on final spine existence alone (`cli/spatial_adata.py:58-73`; `cli/hmm_adata.py:614-631`). They do not validate a completion marker, expected catalogs, task count, config hash, or readable stores.
- Preprocess finalizes its spine before automated summary plotting (`preprocessing/partitioned_executor.py:1030-1046`). HMM finalizes its spine and experiment spine before final feature clustermaps (`tools/partitioned_hmm.py:904-920`). A plot failure leaves a “done” spine and a default restart skips the missing work.
- Task Zarr and Parquet outputs are written directly to final paths, and H5AD/Zarr writers overwrite their targets directly (`readwrite.py:979-1002`, `readwrite.py:1245-1321`). There is no stage-level temporary directory + atomic publish.
- The experiment manifest explicitly states that only raw records stage completion (`informatics/experiment_manifest.py:10-13`). Preprocess, spatial, and HMM never call `record_stage_completion`.
- Experiment and sidecar manifest JSON writes are direct, non-atomic read/modify/write operations and have no locking (`informatics/experiment_manifest.py:41-64`; `informatics/sidecar_manifest.py:30-49`; `project/registry.py:91-105`). Concurrent writers or process interruption can lose/corrupt metadata.
- Sidecar manifests store `str(Path(...))`, commonly absolute paths, while spines/registry generally use run-relative paths (`informatics/sidecar_manifest.py:41-49`). This weakens copy/move portability and creates two path semantics for the same artifacts.

**Required correction**

- Give every stage a manifest state machine such as `planned -> running -> complete/failed`, with config hash, source artifact IDs, expected task count, successful task count, schema versions, timings, and model/checksums.
- Write task artifacts to unique temporary paths; validate; atomically publish within the same filesystem.
- Publish the final spine and `complete` marker last.
- Make skip logic require a compatible complete record and validate essential artifacts.
- Use atomic JSON writes and file locking for project/experiment registries.
- Store relative paths with an explicit anchor/path kind; retain backward-compatible absolute-path reads.

### High H7: the runtime dependency and CI contract does not match the partitioned full pipeline

- Partitioned storage unconditionally calls `DataFrame.to_parquet/read_parquet`, but neither `pyarrow` nor `fastparquet` is a core or optional dependency in `pyproject.toml:42-52`.
- The code writes/edits Zarr v3 through recent `anndata.io.write_elem` and `zarr.open_group(..., use_consolidated=False)` APIs (`informatics/incremental_zarr.py:53-64`), while dependencies declare only `anndata>=0.10.0` and do not pin a compatible Zarr range (`pyproject.toml:42-52`). The installed audit environment used AnnData 0.12.19 and Zarr 3.1.5; the declared minimum was not validated.
- `experiment full` necessarily reaches HMM and plotting, whose CLI module requires the optional `torch` and `matplotlib` extras (`cli/hmm_adata.py:25-30`). There is no dedicated `full` extra or startup dependency check describing the complete requirement set.
- GitHub CI runs only on Ubuntu and the pytest job runs only `pytest -m smoke`, not unit/integration/e2e tests (`.github/workflows/ci.yml:52-79`). There are no macOS or Windows CI jobs despite the portability objective.

**Impact**

A normal base installation can reach the new partitioned path and fail at the first Parquet operation. Older but declared-valid AnnData/Zarr combinations may fail during incremental writes. Current unit regressions can merge without running in CI.

**Required correction**

- Add the selected Parquet engine as a required dependency for the core partitioned pipeline.
- Establish and test compatible AnnData/Zarr version bounds for the APIs and format used.
- Define/document an install target for `experiment full`, or move truly required dependencies into core.
- Run unit tests in CI, add the 3 x 2 full-flow matrix with lightweight fixtures, and add at least macOS plus Windows storage/config/query smoke jobs. Linux cgroup enforcement needs a separate capable integration runner.

### High H8: persisted partitions are not yet fully resource-independent across machines

The intended deployment model includes a full run on a large HPC followed by transfer of the complete analysis/project directory to a lower-resource laptop. This requires a strict separation between the immutable storage contract and the execution plan chosen on either machine.

**Evidence**

- Preprocess/HMM task sizes and spatial task sizes are chosen from the producing run's `target_task_memory_mb` and reference-window plan (`preprocessing/dispatch_plan.py:82-160`; `tools/partitioned_spatial.py:229-269`). Those task boundaries are useful provenance and physical organization, but they must not become the minimum memory needed by every future reader.
- `materialize(..., lazy=True)` offers a lazy base-partition path, but eager mode is the default (`informatics/partition_read.py:1-13`, `partition_read.py:765-790`).
- Derived preprocess/HMM overlay ignores the caller's lazy choice and calls `safe_read_zarr` for each candidate task before selecting shared reads and positions (`partition_read.py:524-584`). `safe_read_zarr` delegates to eager `anndata.read_zarr` (`readwrite.py:1324-1350`).
- Spatial read-metric overlay also eagerly loads each reference-matching task Zarr before subsetting selected reads (`partition_read.py:640-698`). A small reference-window/barcode/read query on a laptop can therefore load substantially larger HPC-created task stores.
- Dense-cache Zarr chunks span the complete position width of a full reference/tile, while derived preprocess chunks also use the full task width (`informatics/partition_store.py:522-539`; `preprocessing/partitioned_executor.py:356-363`). This favors whole-row analysis but may create excessive decompression/I/O amplification for small reference-window queries.
- Most stage pointers are relative, but dense-cache relocation can convert raw/catalog pointers to absolute paths, and cross-volume relative-path generation is not universally safe (`informatics/partition_store.py:448-469`; `informatics/partition_read.py:54-75`). A transferred tree may retain paths meaningful only on the HPC.
- The materialization/query API has no resolved consumer-side resource budget. It cannot automatically lower concurrency, select a bounded backend, or microbatch assembly based on the laptop's CPU/memory.

**Impact**

- The dataset may be format-readable after transfer but not operationally usable within the laptop's memory ceiling.
- Correctness can accidentally depend on task boundaries chosen using the HPC's resources.
- A user may be forced to repartition or regenerate authoritative results merely to inspect them on a smaller machine, creating duplicate data and provenance ambiguity.

**Required storage/execution contract**

Separate four layers that are currently too closely coupled:

| Layer | Purpose | Persistence | Resource behavior |
|---|---|---|---|
| Logical partition | Query pruning and identity, primarily reference/coarse genomic window plus barcode or barcode bucket | Stable and recorded | Independent of machine size |
| Physical chunk/row group | Minimum independently readable Zarr/Parquet unit | Stable and recorded | Sized for a documented minimum supported machine and primary query patterns |
| Compute batch | One or more physical chunks assigned together | Ephemeral | Recomputed from local algorithm memory expansion and current headroom |
| Worker pool/in-flight set | Parallel execution | Ephemeral | Recomputed from local CPU allocation, memory, task count, and I/O bandwidth |

A logical Zarr group or Parquet partition may be large; portability requires its internal chunks/row groups to be independently sliceable. Conversely, making every file extremely small creates metadata and filesystem overhead. Prefer reasonably sized Parquet files containing multiple bounded row groups and large logical Zarr groups containing bounded multidimensional chunks.

1. Treat logical identity, schemas, indexes, model IDs, and checksums as persistent data. Treat worker counts, in-flight limits, memory ceilings, and scheduler details as execution provenance only.
2. Never require the consuming machine to recreate the producing machine's process pool or memory allocation. On every invocation, resolve a new local resource envelope from user limits plus CPU affinity/quota/scheduler allocation and current memory headroom.
3. Define a minimum supported consumer profile rather than “most machines” informally—for example, a 2-CPU laptop with 4-8 GiB RAM. Keep authoritative logical partition boundaries stable after transfer, but size physical units from uncompressed working bytes and algorithm expansion, with a documented maximum. Chunk shapes must support both read-row and genomic-window selection; genome arrays must not require decompressing a whole task width for a small window.
4. Predicate-prune catalogs first, slice Zarr arrays lazily, and call `to_memory` only on bounded read x position blocks. Derived layers and spatial metrics must honor the same lazy/bounded selection as the base matrix.
5. Assemble results in consumer-sized microbatches. Lower-memory machines should reduce concurrency and batch size; they should not change result identity or numerical meaning. Higher-resource machines should gain throughput by processing more portable units concurrently and, where useful, fusing several adjacent chunks into one compute batch without changing the persisted layout.
6. Allow an optional machine-local cache/index optimized for the laptop, but keep it disposable and separate from the authoritative portable manifest. Repartitioning may be a performance optimization, never a prerequisite for correctness.
7. Store relocatable paths relative to an explicit dataset/project root. Preserve original absolute BAM/POD5 paths as provenance fields only; opening completed analysis partitions must not require those source paths to exist.
8. Record storage schema/chunking versions separately from the producing run's resource profile so compatibility checks do not reject a dataset merely because the consumer has fewer resources.
9. On HPC systems, amortize portable-unit overhead through task fusion, store-handle reuse, bounded asynchronous prefetch, node-local scratch caches, vectorized multi-chunk kernels, work stealing, and concurrent writes to separate standard-sized output shards. Pool sizing must still account for `workers x fused_batch_peak` plus parent/reducer/output memory and I/O limits.
10. For a reference/barcode HMM, stream or deterministically sample all relevant portable chunks into the single-owner fit, publish one immutable model, and then apply it across those chunks concurrently. Neither fitting scope nor model identity should depend on how many chunks a particular machine fuses into a task.

Initial physical-size targets should be treated as benchmark hypotheses, not format constants. Reasonable starting ranges are approximately 8-32 MiB uncompressed per Zarr chunk, 32-128 MiB uncompressed per Parquet row group, and 128-512 MiB per Parquet file containing several row groups. The acceptance criterion is measured bounded memory and low I/O amplification on the minimum consumer profile, not adherence to a particular number.

### High H9: genome alignment, analysis, and plotting BED scopes are not separated

The required genome-mode contract has three distinct interval scopes:

| Scope | Proposed configuration | Purpose |
|---|---|---|
| Alignment | `alignment_regions_bed` | Optionally restrict the FASTA/reference universe before alignment |
| Analysis | `analysis_regions_bed` | Restrict preprocess, spatial, HMM, latent, and subsequent computations while preserving bounded task cores/halos |
| Plotting | `plot_regions_bed` | Define presentation intervals assembled from already-computed analysis partitions without changing analysis scope |

**Current behavior and evidence**

- `fasta_regions_of_interest` is the current alignment-scope option. `load_adata` extracts each BED interval into a reduced FASTA before conversion/alignment (`cli/load_adata.py:422-449`; `informatics/fasta_functions.py:356-416`).
- Extracted FASTA records are named `chrom:start-end`, and alignment coordinates become local to that new record. For conversion data, strand-derived reference names further qualify those records. There is no persisted original-reference offset map that lets later BEDs consistently remain in original genomic coordinates.
- `spatial_regions_bed` is the only downstream BED field exposed by `ExperimentConfig` (`config/experiment_config.py:1180`, `experiment_config.py:2249`; `config/default.yaml:577`).
- Partitioned spatial uses that one BED both to replace ordinary genome tiles with BED-specific spatial tasks and to generate dense clustermap/position-matrix regions (`tools/partitioned_spatial.py:154-270`, `partitioned_spatial.py:1145-1171`, `partitioned_spatial.py:1460-1493`). Analysis scope and presentation scope are therefore coupled.
- Preprocess independently calls `plan_preprocess_tasks`, which iterates every standard reference window and does not accept a region catalog (`preprocessing/dispatch_plan.py:56-160`; `preprocessing/partitioned_executor.py:948-952`).
- HMM independently calls the same standard planner and ignores the spatial BED/task catalog (`tools/partitioned_hmm.py:864-893`).
- Latent analysis constructs units from every standard locus/genome core and also has no region input (`tools/partitioned_latent.py:51-83`, `partitioned_latent.py:596-610`).
- The current plot path does not implement a general plotting-only interval resolver that queries completed preprocess/spatial/HMM catalogs, slices all overlapping authoritative cores, removes halo duplication, and stitches them into one requested span.

**Impact**

- Users cannot define one genome analysis scope and know that every downstream stage used exactly that scope.
- HMM/latent outputs can cover different windows from spatial outputs even within the same `experiment full` run.
- Changing plotting intervals can require changing the spatial task plan or rerunning compute, although plotting should be a read-only view over completed results.
- When alignment used a reduced FASTA, later BEDs written in original chromosome coordinates may not match stored reference names or local positions.
- Cross-stage read/model provenance cannot state one normalized analysis-region identity.

**Required three-BED contract**

1. Parse all BED inputs as 0-based, half-open intervals and normalize them once into versioned Parquet catalogs. Record source path/checksum, original line/name, source reference, source start/end, stored reference, stored start/end, orientation/strand derivation where relevant, and a stable region ID.
2. Preserve `fasta_regions_of_interest` as a deprecated alias for `alignment_regions_bed`. Alignment restriction may change the physical reference FASTA, but must publish a `reference_interval_map` from every stored reference/local coordinate range back to the original FASTA reference and coordinates.
3. Make `analysis_regions_bed` an inherited experiment-level scope. Preprocess, spatial, HMM, latent, and future stages must consume the same normalized catalog from the experiment spine/manifest rather than reparsing a config path independently.
4. Tile the union of analysis intervals into deterministic non-overlapping authoritative cores. Each stage may load its own required halo, but it must publish only core positions and record both core and load bounds. Overlapping BED records should not duplicate computation, while their individual region IDs/names remain queryable provenance.
5. Define read inclusion consistently as reference-interval overlap, followed by stage QC masks. A read may participate in multiple adjacent core tasks as needed for context, but each derived position/read result must have one authoritative owner.
6. Make `plot_regions_bed` presentation-only. For each requested reference/barcode plot interval, query every completed stage catalog whose authoritative core overlaps the interval, slice before materialization, discard halo/non-owned positions, align rows by the project molecule key, and stitch positions in genomic order.
7. A plot interval may span any number of adjacent analysis windows. For example, `[110, 135)` must combine `[110, 120)` from analysis core `[100, 120)` with `[120, 135)` from core `[120, 140)` and render one clustermap without duplicate position 120 or duplicated reads.
8. Validate that every requested plot interval is covered by the union of completed analysis cores for every required stage/layer. Default to a clear error for an unanalyzed gap; an explicit opt-in mode may render missing spans as `NaN`, but must label them as absent analysis rather than biological missingness.
9. Materialize plots in barcode/read microbatches under the consumer's local `PoolBudget`. The plotting BED must not force all reads or all overlapping partitions into memory simultaneously.
10. Persist a plot-to-source manifest containing plot region ID, stage/layers, contributing task IDs/core ranges, model IDs where applicable, read selection/subsampling seed, and output artifact paths.

**Backward compatibility**

- Continue reading `fasta_regions_of_interest`, but warn that `alignment_regions_bed` is the replacement.
- Do not silently reinterpret legacy `spatial_regions_bed` as pipeline-wide analysis scope, because that would newly restrict preprocess/HMM/latent behavior. When the new fields are absent, preserve its current spatial-task-plus-spatial-product behavior with a deprecation warning and record that legacy scope explicitly.
- New configurations should use `analysis_regions_bed` and `plot_regions_bed`. A migration tool can copy one legacy spatial BED into both new fields only after the user explicitly chooses pipeline-wide analysis restriction.

### Medium M1: performance logs have major blind spots and raw attribution errors

- `setup_stage_logging` creates/rotates a human log and `PerfLogger` (`logging_utils.py:75-104`, `logging_utils.py:107-143`). Preprocess, spatial, and HMM wrappers call it.
- Raw goes through `load_adata_core`, which calls only `setup_logging`, not `setup_stage_logging` (`cli/load_adata.py:253-284`). A single `experiment full` invocation therefore produces no raw perf log.
- In a long-lived batch process, the previous experiment’s HMM `PerfLogger` remains in the ContextVar while the next experiment’s raw stage runs. Raw pools can be written into the preceding HMM perf file until preprocess rotates it.
- The Linux worker watchdog is a no-op, and memory samples are emitted only by that watchdog (`memory_guard.py:272-273`, `memory_guard.py:290-358`). Linux perf logs contain pool events but no periodic RSS samples even when cgroup enforcement is active.
- Sequential task execution emits start/end but no samples (`memory_guard.py:485-508`). Reductions, plotting in the parent, external tools, and writes are not sampled.
- `PerfLogger` supports pool start/sample/retry/end and a stage summary, but not per-task completed/total, throughput, ETA, task duration, bytes read/written, or stage substep timing (`perf_log.py:43-109`).
- Parallel result collection in `run_tasks_parallel` awaits futures in submission order (`memory_guard.py:533-548`), which delays progress visibility and result release behind one slow early task.

**Required correction**

- Use `setup_stage_logging` for raw and close it explicitly at stage exit, including skip/failure paths.
- Add a process-tree sampler independent of the enforcement mechanism, including Linux and sequential paths.
- Emit task/substep progress with completed/total, duration, retries, rows/bases, input/output bytes, and current/peak RSS.
- Consume futures in completion order while storing results by original index.
- Add a top-level `full` summary that links all four stage logs and their outcome.

### Medium M2: filesystem encoding and relative paths are not fully cross-platform

- Active raw/spatial/HMM task directories correctly percent-encode path components with a conservative safe set (`informatics/raw_store.py:126-128`; `tools/partitioned_spatial.py:44-48`; `tools/partitioned_hmm.py:42-45`).
- HMM model filenames only replace `/`; characters invalid on Windows (`:`, `*`, `?`, quotes, angle brackets, pipe), trailing spaces/dots, and reserved device names are not handled (`cli/hmm_adata.py:432-437`).
- Project per-sample directories interpolate experiment ID, reference, and sample without encoding (`project/sample_store.py:37-40`). Separators create unintended nesting; `..` is unsafe; and glob-based listing assumes exactly three path levels (`sample_store.py:103-114`).
- Other sluggers do not disambiguate collisions (`a/b` and `a?b` can both become `a_b`) or Windows reserved names (`informatics/partition_store.py:97-100`; `project/set_store.py:36-39`).
- `os.path.relpath` raises on Windows when source and anchor are on different drives. Both spine paths and project registry paths assume it always succeeds (`informatics/partition_read.py:54-75`; `project/registry.py:153-168`). This is realistic when raw input/BAM, output, and project are on different volumes.
- Dense cache relocation converts raw pointers to absolute paths when output differs from raw (`informatics/partition_store.py:448-469`, `partition_store.py:619-635`), reducing copy/move portability.

**Required correction**

Centralize reversible component encoding with collision resistance and Windows reserved-name handling. Centralize path serialization with `relative` and `absolute/cross-volume` variants, explicit anchors, and relocation tests. Do not expose logical names as raw path components.

### Medium M3: streaming raw write does not enforce experiment-global read-ID uniqueness

- `validate_ragged_frame` enforces uniqueness only within the frame passed to it (`informatics/ragged_store.py:203-246`).
- The streaming writer explicitly cannot validate the whole experiment (`informatics/raw_store.py:535-543`).
- Duplicate read IDs in separate streamed groups overwrite `shard_by_read`/`row_by_read` dictionary entries while both shard/molecule rows are written (`raw_store.py:255-271`). The final obs concat may have duplicate indices and mismatched pointers.

**Required correction**

Enforce uniqueness using the on-disk molecule index, an external sort, a disk-backed hash set, or a source guarantee that is explicitly validated per input BAM. Fail before publishing the raw spine. Test duplicates across references and across two flushes of the same reference.

### Medium M4: storage-mode selection is disconnected from the resolved workflow budget

- Automatic locus/genome planning compares one estimated dense matrix (`reads x reference_length x 8`) to `max_full_matrix_gb` (`informatics/storage_planner.py:64-85`).
- `max_full_matrix_gb` defaults to 8 GiB independently of `max_memory_percent/max_memory_gb` (`config/default.yaml:548-557`). A user can request a 4 GiB workflow cap while auto-planning an 8 GiB “full” matrix.
- Full dense cache construction materializes an entire reference; tiled cache construction materializes every overlapping read in a tile without barcode/read chunking (`informatics/partition_store.py:480-551`).

**Required correction**

Derive every full/tiled decision from the resolved usable budget and a stage-specific peak factor. Tile both positions and reads/barcodes where depth requires it. Store the resolved plan inputs and estimator version in the manifest so decisions are explainable and reproducible.

### Medium M5: batch workflow reports completion despite per-experiment errors

The batch command catches every exception, prints it, continues, and ends with “Batch processing complete” without a nonzero exit (`cli_entry.py:346-363`). For automated large projects, a partially failed `batch full` can appear successful to a scheduler.

Return a nonzero status when any experiment fails, and write a machine-readable batch summary with success/failure, stage, exception, and output/log paths.

## Modality and analysis-mode verification matrix

The implementation contains branches for all requested modalities, but current tests do not establish the complete contract.

| Pipeline surface | Conversion/locus | Deaminase/locus | Direct/locus | Conversion/genome | Deaminase/genome | Direct/genome |
|---|---|---|---|---|---|---|
| Partitioned raw | Implemented; unit-level pieces | Implemented; unit-level pieces | Implemented for modkit/pysam backends; focused tests | Same raw planner, not full-flow tested | Same raw planner, not full-flow tested | Same raw planner, not full-flow tested |
| Partitioned preprocess | Substantial unit coverage | Chimera/QC unit coverage | Youden/incremental-layer unit coverage | No full modality/mode path | No full modality/mode path | Specific genome tests exist |
| Partitioned spatial | Locus unit coverage | No modality-specific partitioned E2E | No modality-specific partitioned E2E | **Confirmed default failure after task reduction** | **Confirmed default failure after task reduction** | **Confirmed default failure after task reduction** |
| Partitioned HMM | Locus-focused unit coverage | Single-C behavior is partially exercised, not a full modality path | No direct-modality partitioned full test | Not full-flow tested; HMM chunk/model issue applies | Not full-flow tested; HMM chunk/model issue applies | Not full-flow tested; HMM chunk/model issue applies |
| `experiment full` | Mocked call-order only | Mocked call-order only | Mocked call-order only | Mocked call-order only | Mocked call-order only | Mocked call-order only |

Specific observations:

- `tests/unit/test_full_recipe.py:7-29` verifies only that four mocked functions are called in order.
- Existing CLI E2E tests call legacy/general `load_adata` and `spatial_adata` for three modality configs (`tests/e2e/cli/test_load_adata.py:16-26`; `tests/e2e/cli/test_spatial_adata.py:16-27`); they do not assert a fresh raw-only partitioned full flow through HMM.
- Genome tests exist for the storage planner, read path, preprocess task stitching, and direct Youden thresholds, but not for default partitioned spatial/HMM/full behavior.
- HMM partitioned tests overwhelmingly use conversion/locus fixtures and do not force the same model scope to span multiple read chunks.

## Validation performed during this audit

### Focused unit suite

Command used the canonical `venvs/venv-all` interpreter and covered the full recipe, resource/perf utilities, raw/ragged storage, materialization, planner/manifest/spine, preprocess/spatial/HMM partitioned executors, and project registry/catalog/set/sample/CLI modules.

**Result:** `187 passed, 5 failed, 3104 warnings in 79.96s`.

All five failures were multiprocessing tests in `tests/unit/test_memory_guard.py`. They failed before exercising assertions because the managed sandbox denied macOS semaphore/forkserver operations:

- `os.sysconf("SC_SEM_NSEMS_MAX") -> PermissionError: [Errno 1] Operation not permitted`
- forkserver Unix-socket `bind -> PermissionError: [Errno 1] Operation not permitted`

An attempted unsandboxed rerun was not available under the execution policy. The other selected tests, including storage, preprocess, spatial, HMM, and project tests, passed. These five results should be rerun in a normal macOS shell; they are not evidence of a code assertion regression, but they also mean the watchdog’s macOS integration behavior was not independently confirmed in this audit environment.

### Targeted runtime probes

1. **Genome spatial empty-region probe:** reproduced `KeyError: 'reference'` from `_dense_product_regions` with a genome-only plan and no BED rows.
2. **Project read-ID collision probe:** created two temporary modern experiments with the same read ID, registered them, and materialized their shared canonical reference. The result contained duplicate `obs_names`, confirming project identity ambiguity.

### Not performed

- External-tool raw E2E (Dorado, minimap2, modkit, samtools) was not rerun because it depends on tool installations and fixture outputs beyond a read-only audit.
- A real high-depth memory benchmark was not run. Static findings identify paths that cannot obey the stated ceiling by construction; measured calibration should follow after those paths are made observable and bounded.
- No source behavior was modified. This file is the only requested artifact.

## Recommended implementation sequence

### P0: correctness and release blockers

1. Fix empty genome spatial region catalogs and add genome-only spatial/full tests.
2. Preserve reference/barcode-specific HMM fitting, but publish one immutable model per configured reference/barcode model key before chunked apply; persist task-to-model IDs/checksums, align the public config, and add collision/multi-chunk determinism tests.
3. Add normalized alignment/analysis/plot region catalogs, an original-coordinate reference map, one inherited downstream analysis planner, and cross-task plot stitching. Preserve legacy BED behavior through explicit compatibility handling.
4. Declare/test Parquet and AnnData/Zarr dependencies required by the production format.
5. Introduce stage completion markers and make skip logic validate them before trusting an existing spine.

### P1: enforceable resource and retrieval architecture

6. Introduce an immutable run-level `ResourceEnvelope` plus a dynamic per-pool `PoolBudget`; use them for Python pools, nested libraries, plots, external tools, dense-cache planning, and project materialization.
7. Add a consumer-side query planner that recomputes local resources on every machine, predicate-prunes first, slices all base/derived Zarr arrays lazily, and assembles bounded microbatches independent of producer task size. Let high-resource consumers fuse and prefetch multiple portable chunks without rewriting them.
8. Add Linux fallback monitoring, process-tree sampling, parent/sequential coverage, and explicit enforcement-mode reporting.
9. Replace global concat/vstack plot/reducer paths with streaming/on-disk reductions and pre-load sampling.
10. Add a project-wide molecule UID and derived read-to-task index; route reference/window/barcode/read queries through Parquet/DuckDB predicate pushdown.
11. Make project export partition-native and preflighted instead of requiring one pooled AnnData.

### P2: hardening and operational quality

12. Centralize cross-platform component encoding and cross-volume path serialization.
13. Complete experiment manifest records for preprocess/spatial/HMM/full and atomically update JSON registries.
14. Add raw perf logging, process-tree sampling on every OS, task progress/ETA, I/O counters, and batch result summaries.
15. Put unit tests in CI and add macOS/Windows storage/query jobs plus a capable Linux cgroup integration job.

## Proposed acceptance criteria

The redesign should not be considered complete until the following are automated:

### Correctness

- Fresh `experiment full` succeeds for all six modality/mode combinations.
- Genome mode succeeds with and without each new BED scope and preserves documented legacy `spatial_regions_bed` behavior.
- Each configured reference/barcode model stratum receives its own deterministic model; all read chunks in that stratum reference the same immutable model ID/checksum and yield identical layer values at 1, 2, and N workers.
- Distinct logical reference/barcode/label identifiers cannot resolve to the same model artifact path, including names containing separators or underscores.
- Every published task in each catalog is readable, has the expected row/position shape, and references a completed source/model artifact.
- Duplicate raw read IDs across stream flushes fail before final publication.
- Duplicate instrument read IDs across experiments remain separately addressable by project molecule UID.

### Genome interval scopes

- Alignment-only, analysis-only, plot-only, all-three, and no-BED configurations have independent tests across conversion/deaminase/direct genome modes.
- All user BEDs use original FASTA coordinates. Runs using an alignment BED publish and round-trip a stored-reference/local-coordinate to original-reference coordinate map, including conversion top/bottom references.
- Preprocess, spatial, HMM, and latent catalogs contain only authoritative cores intersecting the normalized analysis union, share stable analysis region IDs, and never duplicate an owned position because source BED intervals overlap.
- Stage-specific halos may extend beyond an analysis core for context, but halo-only results are never published as authoritative analysis output.
- A plotting interval contained within one analysis core reads only that slice. A plotting interval spanning two or more adjacent cores stitches the exact requested coordinate range in order, with no duplicated boundary positions and read rows aligned by molecule ID.
- Plotting the same requested interval produces the same source-task/model manifest and numerical matrix regardless of analysis tile size, read-chunk size, worker count, or machine profile.
- A plot interval crossing an unanalyzed gap fails clearly by default. Explicit missing-span mode inserts and labels `NaN` positions without confusing absent analysis with missing molecular signal.
- Changing only `plot_regions_bed` never invalidates or reruns preprocess, spatial, HMM, or latent artifacts.

### Resource behavior

- A startup record reports requested, detected, and resolved CPU/memory plus enforcement mode.
- Resolved workers never exceed affinity/cgroup/scheduler CPU.
- Before every pool allocation, a logged `PoolBudget` recomputes available/cgroup memory headroom and current smftools process-tree use, then resolves stage-specific batch size, worker count, and maximum in-flight tasks without exceeding the immutable run envelope.
- Default CPU sizing is based on allocation/affinity/quota rather than an instantaneous utilization snapshot. Any optional shared-machine load-based throttling is explicit and logged.
- Configured memory fields reject invalid values.
- On Linux, a deliberately over-budget process tree is constrained by cgroup or the fallback is clearly active and tested.
- On macOS/Windows, parent + recursive child RSS is measured; inability to provide a hard ceiling is stated explicitly.
- Synthetic high-depth tasks remain within a documented tolerance of the resolved budget, including reductions, plots, serialization, and project export—not only worker task bodies.

### Cross-machine portability

- Build a representative full experiment/project on a Linux HPC profile (for example, 64 CPUs and a large memory budget), copy the complete tree to a macOS or Windows laptop profile (for example, 2 CPUs and 4 GiB), and open/query it without modifying authoritative artifacts.
- The laptop can retrieve one read across all stages and a reference window across selected/all barcodes while remaining below its resolved memory ceiling and using no more than its resolved CPU allocation.
- Base, preprocess, spatial, and HMM readers predicate-prune and slice before materialization; no test query eagerly loads a whole producer-sized task store unless the selected result itself requires it.
- The same logical query returns the same molecule IDs, model IDs, layers, and scalar results on both machines regardless of worker count or microbatch size.
- Moving or renaming the dataset/project root preserves every required partition/catalog/model link. Missing original HPC BAM/POD5 paths do not prevent completed-analysis queries.
- Producer resource settings remain inspectable as provenance but never override the consumer's locally resolved resource envelope.
- Logical partitions and physical chunks are identical across resource profiles; only compute-batch fusion, worker count, and in-flight limits change.
- Physical chunk/row-group benchmarks report uncompressed size, decompression/I/O amplification, and peak algorithm working memory on the minimum supported consumer profile. No single required read unit exceeds the documented portable bound.
- An HPC scaling test demonstrates higher throughput from concurrent units, multi-chunk task fusion, prefetch/cache reuse, or vectorization while writing the same standard-sized portable output units expected by the laptop reader.

### Query and traceability

- Given `(experiment_uid, read_id)`, one indexed query returns raw shard/row, source BAM/POD5 metadata, preprocess task/row, spatial task/row, HMM task/row/model ID, and QC masks without scanning all task stores.
- Given `(canonical_reference, start, end, barcode set)`, the query prunes molecule and derived partitions before opening array stores.
- Query benchmarks record cold/warm latency, files opened, bytes read, and peak RSS at representative project scale.
- Moving a complete run/project tree preserves all relative pointers; cross-volume Windows inputs use a documented resolvable path form.

### Completion and observability

- Fault injection after every task/write/reduction/plot step never leaves a stage that restart logic treats as complete.
- Stage/full manifests include config hash, input IDs, task totals, schema versions, timing, peak RSS, worker decisions, and output checksums/paths.
- Raw, preprocess, spatial, and HMM each emit progress and performance records; skipped/failed/completed outcomes are explicit.
- Batch commands exit nonzero on partial failure and produce a machine-readable summary.

## Bottom line

The partitioned redesign has successfully replaced the most damaging monolithic raw and per-task write patterns. Its storage primitives are suitable for the intended future architecture. The next work should focus less on adding another storage form and more on making the existing form **query-indexed, scope-correct, transactionally published, relocatable, and governed by a consumer-local resource model**.

Until C1 and C2 are fixed, `experiment full` should not be described as supporting genome mode or deterministic bounded HMM processing. Until H1/H2/H5 are addressed, `max_memory_*` should be documented as planning/advisory controls except where a successfully activated Linux cgroup provides a verified aggregate cap. Until H3/H4 are addressed, read-level traceability should be described as reconstructable within an experiment, not yet rapid or globally unambiguous across a project. Until H8 is addressed, HPC-to-laptop transfer should be described as format portability rather than guaranteed low-resource operational portability. Until H9 is addressed, `spatial_regions_bed` should be documented as a spatial-only combined compute/plot scope, not as a pipeline-wide genome analysis scope.
