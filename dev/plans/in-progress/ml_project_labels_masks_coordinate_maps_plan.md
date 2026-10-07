# ML project labels, position masks and coordinate maps (`MLX`)

**Status:** in progress. `MLX-01`–`MLX-03`, `MLX-05`–`MLX-07`, `MLX-09`–`MLX-11` merged (`MLX-11`: #652); `MLX-04`, `MLX-08` proposed.

## Problem

A project-scope activity-classifier study (one allele active vs inactive,
across experiments, by signal channel, span and region) cannot be expressed as
an ML plan today, for three independent reasons found while designing it:

1. **Labels come only from stored obs.** A dataset's `labels.column` and every
   `filters` key are read from each experiment's molecule index
   (`selection._read_identity_metadata`). Biological labels (cell type, sorted
   population, antibody gate) live in the project's sample sheet, keyed by
   experiment and barcode, and stage stores are immutable. `LabelSchema`
   already refuses any `source` but `obs` ("only 'obs' is currently
   supported"), so a non-obs source was anticipated but never built.
2. **One contiguous window per dataset.** Coordinates are a single
   `[start, end)` (`partition_dataset` coordinate resolution, `filters.start/end`
   or one interval-catalog row). Region-ablation studies need "everything but
   the enhancer" or "enhancer + promoter only" -- two or more windows.
3. **One reference's coordinates per dataset.** Planning requires exactly one
   resolved canonical reference (`orchestration/planning._input_schema`), and
   positions are each reference's own. Molecules of a structural variant (e.g.
   an enhancer deletion, 522 bp shorter) cannot share a dataset with the intact
   allele's molecules even over sequence both carry, because the same base sits
   at different positions.

## Work items

| item | status | scope |
|---|---|---|
| `MLX-01` external label table | merged | `labels.source: table` joined on declared keys |
| `MLX-02` position masks | merged | include/exclude windows within a dataset's span |
| `MLX-03` cross-reference coordinate maps | merged | place several references' molecules in one coordinate frame |
| `MLX-04` qualification | proposed | one real project study end to end |
| `MLX-05` real-store compatibility | merged | resolve channels and QC filters against real pipeline stores (`F66`) |
| `MLX-06` plan job runner | merged | bind a resolved plan to a dataset snapshot, split and partition dataset per fold; train and test-evaluate each fold |
| `MLX-07` training-only groups | merged | let single-class groups train without being held-out folds |
| `MLX-08` published fold runs | proposed | run `MLX-06` folds through the job service as immutable run artifacts |
| `MLX-09` partition-major reads | merged | open each store partition once per pass instead of once per batch (`F67`) |
| `MLX-10` whole-dataset reads, HMM-stage channels | merged | read a dataset's rows without a split (for embeddings), and resolve HMM/spatial stage read indexes |
| `MLX-11` block-sharded workers | merged | let N processes split the read by whole blocks (`F69`) |

Order: `MLX-01`, `MLX-05`, `MLX-06` (together they unblock a single-reference,
full-span pilot), then `MLX-02`, then `MLX-03`, which builds on `MLX-02`'s
guard. `MLX-07` whenever a task has single-class experiments worth training on.

### `MLX-01` — external label table

```python
"labels": {
    "source": "table",
    "table": "ml/labels/b6_vs_nk.parquet",   # project-relative
    "keys": ["experiment_id", "barcode", "reference"],
    "column": "label",
    "classes": {"inactive": 0, "active": 1},
    "positive_class": "active",
    "missing": "drop",
}
```

- **Keys.** Each key names a molecule-identity field resolved per row:
  `experiment_id` (from the project registry), `experiment_uid`, `barcode`,
  `sample`, `reference` (canonical, via the reference registry) and
  `physical_reference` (strand-level). Barcodes compare by number across
  spellings, reusing `barcode_number_key` (`BAL-01`), so a sheet's `4` matches
  a stored `barcode04`.
- **Join.** Many-to-one: duplicate keys in the table are an error. Rows with no
  match follow `missing` (`drop` default; `error` available). The table may
  carry extra columns; `group_by` and `filters` may name them, so a study keeps
  task fields (harvest, negative class) beside the label instead of encoding
  each task as its own file.
- **Provenance.** The table's sha256 and the key list enter the dataset
  snapshot identity, like the catalogs it is joined against; editing the table
  changes the snapshot.
- **Scope.** `project` scope only; an experiment-scope plan has no project to
  resolve the path against.
- Tests: join by each key kind, barcode spelling equivalence, duplicate keys
  refused, `missing: error`, snapshot id changes with table content, `group_by`
  on a table column.

**As implemented.** `LabelSpec.table`/`keys`; `MLPlan.to_dict` omits them for
obs labels, so plans written before this keep their `plan_hash`. Selection
loads the table once (`_load_label_table`: normalized keys, duplicates
refused, file sha256), and joins it per experiment before `filters`
(`_join_label_table`). A table column sharing a name with any column of the
stored molecule index is refused, including columns the plan does not load.
The table sha256 joins the selection identity payload and the dry-run report.
Tests: `test_table_*`, `test_filters_and_groups_may_name_table_columns`,
`test_selection_identity_changes_with_table_content` (selection), and
`test_table_label*`, `test_obs_labels_serialise_without_table_fields` (plan).

### `MLX-02` — position masks

```python
"positions": {"include": [[995, 3718]], "exclude": [[3127, 3528]]}
```

- Windows are half-open, in the dataset's coordinate frame (its reference, or
  `MLX-03`'s frame reference). The tensor width stays the span of `include`, so
  convolutional models keep true distances; excluded and gap positions are
  marked unavailable (`availability` mask false, value 0), never compacted
  together.
- Sklearn backends drop columns unavailable for every row, so NB/LR/RF fit only
  on visible positions; attributions are zero at masked positions (existing
  behaviour for unavailable positions).
- `filters.start/end` keep working and mean one `include` window.
- **Open:** whether every sklearn feature path already honours `availability`
  (verify before relying on it; add the column drop where it does not).
- Tests: masked positions carry no signal into any backend; widths and
  coordinates in the snapshot; a single-window mask is identical to the
  equivalent `filters.start/end` dataset.

**As implemented — differs from the above on one point.** Masked positions
are *not in the feature set*, rather than kept as unavailable columns. The
transform has no per-position exclusion: a masked cell would be imputed to a
constant and still emit its observed/design/padding indicator columns, and
for Bernoulli NB thousands of constant columns add a class-dependent offset
to every row's log-probability (ranking metrics unchanged, probabilities
shifted). So `PositionMask` (`include` minus `exclude`) resolves to disjoint
windows; selection counts features from them (refusing a window past the
reference); `snapshot_from_selection` writes one snapshot interval per
window; the reader builds the feature coordinates from all of them
(`_coordinates`) and `_position_columns` places values by position, so no
other reader change was needed. A multi-window plan's id hashes its
coordinates. Cost: a convolutional model sees kept windows joined, not at true
distance -- single-window masks (enhancer-only, intervening-only, ...) are
unaffected. `positions` is omitted from `MLPlan.to_dict` when unset, so
earlier plans keep their hash.

Tests: `test_positions_resolve_to_kept_windows_and_round_trip`,
`test_plans_without_positions_serialise_without_them`,
`test_position_declarations_are_validated` (plan);
`test_masked_dataset_holds_only_kept_positions` (batch values equal the kept
columns of the unmasked dataset), `test_trained_model_sees_only_kept_positions`
(runner). Real data: the pilot with the enhancer masked binds 4,289 of 4,690
positions, none in the enhancer, and a fold trains and evaluates in 197 s.

### `MLX-03` — cross-reference coordinate maps

```python
"coordinate_frame": {
    "reference": "allele_A",
    "maps": {"allele_A_deletion": "ml/coordinate_maps/deletion_to_A.parquet"},
}
```

- A map is a table `(source_position, frame_position)` for one canonical source
  reference; positions not listed have no counterpart in the frame. The
  project supplies it (from an alignment it owns); smftools validates that it is
  injective, strictly increasing and within both references' lengths, and
  hashes it into the snapshot.
- The partition reader places each molecule's values at mapped frame
  positions; frame positions its reference lacks are unavailable for that
  molecule.
- **Leakage guard.** Which positions a molecule *has* would identify its
  reference, and in a deletion-vs-intact task, its class. So a dataset with a
  coordinate frame must select (via `MLX-02`) only frame positions mapped in
  every selected reference; planning refuses otherwise, naming the offending
  windows. There is no override flag.
- Planning's one-canonical-reference rule becomes one *frame* reference.
- Tests: a molecule on a mapped reference lands at the frame positions of its
  aligned bases; a selection that includes an unmapped position is refused;
  snapshot changes with the map; an identity map reproduces the unmapped
  dataset exactly.

**As implemented.** `CoordinateFrame(reference, maps)` on the dataset
(project scope; omitted from `to_dict` when unset). Selection loads each map
(`_load_coordinate_maps`: integer, one-to-one, order-preserving), refuses a
selected reference neither frame nor mapped, and counts features in the frame
(`_frame_feature_count`), refusing any kept frame position a selected
reference lacks -- the leakage guard, no override. Map sha256s join the
selection identity and dry run. Planning takes the frame as the schema
reference; the snapshot lists every canonical reference. The reader gets
the maps (`build_partition_data_plan(coordinate_maps=...)`), reads each
reference separately with its own source span (real ragged stores refuse a
position window over several references), and translates a mapped read's
source positions into frame positions (`_mapped_position_columns`). Not done:
the identity-map equivalence test; tests instead check placement directly.

**Found on the way (`F68`).** The reader made the design mask "site and not
padding", so a read ending before the window gave its row a different design
row, and the position-by-channel schema refuses any batch whose rows differ.
Real data has such reads (partial enh-del molecules). Design is now the
reference's site mask alone; observed still requires coverage; the uniformity
check ignores rows that cover none of the window. Test:
`test_reads_outside_the_window_are_padding_not_a_design_conflict` (fails with
only that change reverted, with the same error real data raised).

Tests: `test_coordinate_frame.py` (deletion fixture written with
`write_experiment_store`): deletion reads land at their frame positions; a
position some reference lacks is refused, naming it; an unmapped reference is
refused; the map file is part of the identity; a mixed-reference job trains;
F68. Plan: `test_coordinate_frame_*`. Real data: B6 vs enh-del (DAFseq fresh,
top strand) -- the full span is refused naming exactly `[3151, 3673)`;
downstream-of-deletion selects 3,151 shared positions over B6 and enh-del
reads; above the deletion, 20 enh-del reads placed at frame `[3673, 4690)`
equal their stored values at source `[3151, 4168)` exactly. A trained fold
scored at chance (AP = positive fraction) on few training negatives -- a
plumbing result. Several experiments in that task are single-class (`MLX-07`).

### `MLX-04` — qualification

On a project with an intact allele, its enhancer deletion, and sorted
populations: label table from the sample sheet, one full-span single-reference
task (`MLX-01` only), one region-masked task (`MLX-02`), one intact-vs-deletion
task over shared sequence (`MLX-03`), each trained with leave-one-experiment-out
splits. Check per-fold row counts against the label table and that attributions
are zero outside the selected positions.

**`MLX-01` real-data check.** A project label table (one task, keyed on
experiment, barcode and strand-level reference) resolved 756,816 reads over
six experiments with zero label disagreements against the table; stored
kit-qualified barcodes matched the sheet's plain numbers. It needed `MLX-05`'s
first two fixes shimmed in the check script.

### `MLX-05` — real-store compatibility (`F66`)

Selection was only ever exercised on fabricated fixtures. Against stores the
pipeline actually writes:

- **Catalog.** Resolve a stage's written-store `catalog.parquet` (which lists
  `layers`, `has_x`), not the planner's `task_catalog.parquet`; register it in
  the project registry at `project add`, and keep the read-index fallback.
- **`X`.** Treat `has_x` as making layer `X` available.
- **Defaults.** Default channels name layers real stores write (`X` with the
  modality's site context), or fail at plan time naming the available layers.
- **QC filters.** Let `filters` reach the stage's `stage_obs.parquet` columns
  (`passes_qc`, `passes_dedup`, ...) for the stages a dataset reads, joined on
  `read_id` like the raw obs sidecar; the sidecar's sha256 joins the identity.
- Tests: fixtures written by the real stage writers (or copied from their
  schema), not hand-built catalogs; each failure above as a regression test.

**As implemented.** `_stage_task_catalog` prefers the written store's
`catalog.parquet` over the planner's `task_catalog.parquet` wherever both
exist; `has_x` makes `X` available; a missing layer's error lists what the
stage wrote and points to `X`; deaminase accepts `GpC` (a subset of its C
sites) as accessibility; `_read_identity_metadata` joins missing filter or
group columns from each read stage's `stage_obs.parquet` on `read_id`, after
the raw obs sidecar. Not done: the default channels still name the
`*_site_binary` layers of single-file preprocess output -- changing them
would change the hash of every plan relying on defaults, and it is not
established that single-file stores hold the same calls in `X`. Not done
either: registering the store catalog at `project add`; the read-index
fallback finds it. Tests: `test_selection_reads_the_written_store_catalog_and_x`,
`test_missing_layer_error_names_what_the_stage_wrote`,
`test_deaminase_gpc_subset_is_accessibility`,
`test_filters_reach_stage_obs_qc_flags`.

**Real-data check, no shims.** The `MLX-01` pilot selection (one task, six
experiments) resolves directly; with `passes_qc` and `passes_dedup` filters it
drops from 756,816 reads to 17,322. `plan_ml_workflow` (dry run: selection,
leave-one-experiment-out folds, model schema, job outputs) completes once the
one single-class experiment is excluded (`MLX-07`).

### `MLX-06` — plan job runner

Nothing outside the benchmark fixtures builds a `DatasetSnapshotManifest`,
`SplitManifest` and partition data plan from a resolved selection, so a plan
can be dry-run but not trained: the documented path binds them by hand. Add
one function that takes a parsed plan, a scope and a job name and returns the
bound partition dataset per fold (selection -> snapshot observations and
sources -> split manifest from the resolved split -> partition sources from
the registry's stage spines -> `build_partition_data_plan`), then runs the
job through the existing job service. Tests: a project fixture trains a
`bernoulli_nb` end to end; the bound snapshot id equals the dry run's
selection-derived identity.

**As implemented** (`orchestration/binding.py`). `bind_ml_job(plan, job,
project_dir=...)` resolves the job's selection, builds the snapshot
(`snapshot_from_selection`: one `ExperimentSource` per experiment, keyed on its
molecule-index sha256; observations from the selection identity table; one
interval over the selected span), turns each resolved fold into a
`SplitManifest` through `MLSplitResolution.to_manifest`, and binds each to the
experiments' stage spines with `build_partition_data_plan`. Selection now
carries those spine paths, run roots and generation ids on each selected
source (execution bindings, outside every identity hash).
`run_bound_train_job` fits each declared model per fold (the plan's balancing
profile applies), predicts the fold's test role and evaluates it, returning
results in memory. Model resolution is shared with the dry run
(`resolve_plan_model`). Not done: publishing through the job service
(`MLX-08`), and test-role prediction is materialized, so bounded by
`max_materialization_bytes`. Tests: `test_plan_job_runner.py` (stores written
by `write_experiment_store`): one fold per held-out experiment, train and
evaluate every fold, stable snapshot and split ids across binds.

**Real-data check.** The pilot binds on the real project in 83 s (5 folds,
16,827 train rows in the first). Training is correct but impractically slow:
~70 s per 64-row batch, because the reader reopens ~48 scattered partitions
per batch (`F67`, `MLX-09`). Stopped after 91 min; no real metrics yet.

### `MLX-07` — training-only groups

Leave-one-group-out refuses a fold whose held-out group lacks a class. A
study may still want such a group's rows in training (an experiment with only
active samples). Add a split option naming groups that only ever train, so
folds are the remaining groups and every fold's train set includes them.
Tests: the named groups appear in no test role and in every train role.

**As implemented.** Two ways, both `leave_one_group_out` only:
`train_groups` (named groups; unknown names refused), as planned, and
`single_class_groups: train`, which makes every group lacking a class
train-only automatically -- what a task grid needs, since which experiments
are single-class depends on the task and group tokens (experiment uids) are
opaque. Default `refuse` keeps the old behaviour; it is omitted from
`to_dict`, so earlier plan hashes are unchanged. Folds are the remaining
groups; with none left, refused. The rule that `leave_one_group_out` rejects
role lists now rejects only validation/test lists (test updated).

Tests: `test_single_class_groups_are_refused_by_default`,
`test_single_class_groups_train_in_every_fold_when_asked`,
`test_named_train_groups_never_hold_out`, `test_unknown_train_groups_are_refused`
(splitting); `test_single_class_groups_policy_is_validated_and_hash_neutral`
(plan). Real data (workflow dry runs): the pilot keeps its five folds and now
trains on the single-class experiment it had to exclude; B6 vs enh-del over
all seven experiments resolves to three scoreable folds, the four
single-class experiments training in each.

### `MLX-08` — published fold runs

Wrap `MLX-06`'s per-fold training in the job service (`run_train_job`), so each
fold run publishes its model bundle, predictions and evaluation as immutable
artifacts with the dataset snapshot and split ids, instead of returning them
in memory. Tests: a fold run's published manifest names the bound snapshot and
split ids; a failed fold leaves a failed run record, not partial artifacts.

### `MLX-09` — partition-major reads (`F67`)

`PartitionDataset` reads each batch with one `materialize` per experiment,
over rows in snapshot (molecule_uid hash) order, so a batch touches dozens of
partitions and each is opened from scratch through anndata. On a real store a
64-row batch costs ~70 s.

Read partition-major instead: per split, group the wanted reads by store
partition, open each partition once per pass (only the wanted rows, only the
requested matrix and design columns), and cut batches from the rows read.
Batch contents become partition-ordered; streamed sklearn fits are
order-independent, and Torch already shuffles within its buffer. Bound peak
memory by partition, not by split.

Tests: a fixture with many partitions per experiment opens each partition
once per pass (count opens); batches cover every split row exactly once;
streamed NB fit equals the materialized fit; on the real pilot, a fold's train
pass takes minutes, not hours.

**As implemented.** Two changes in `data/partition_dataset.py`, both inside
the reader (no change to `materialize`):

- *Partition-major order.* `ExperimentPartitionSource.stage_read_indexes`
  (bound by `MLX-06` from the selection) lets `build_partition_data_plan`
  key each row by its stored `(group_path, group_row)`; `read_order(split)`
  sorts rows by experiment, then key. Rows without a key keep canonical
  order, so a dataset bound without read indexes reads exactly as before.
  `materialize(split)` restores manifest order from `order_indices`.
- *Block reads.* Profiling showed the first change alone was not enough: one
  `materialize` call costs ~10 s fixed on a real store (it reloads the spine,
  ~3 s, and opens ~22 store sections) even for one partition. `iter_batches`
  now decodes blocks of whole batches (`PartitionReadPolicy.max_block_bytes`,
  512 MiB by default) in one call and slices them; batch boundaries, and so
  worker sharding, are unchanged.

Real pilot fold (16,827 train rows, 4,690 positions): a full train pass in
205 s, against ~70 s per 64-row batch before (~5 h). `run_bound_train_job`
also predicts the test role batch by batch: one held-out experiment (12,551
rows) was estimated at 2.5 GB to materialize, over the default budget. The
whole five-fold pilot (bernoulli_nb, leave-one-experiment-out) now trains and
evaluates in 17 min, held-out average precision 0.82-0.93. Tests:
`test_batches_read_one_partition_at_a_time`,
`test_materialized_split_keeps_manifest_order`,
`test_block_reads_match_batch_reads_with_fewer_store_reads`,
`test_read_order_is_canonical_without_read_keys`,
`test_test_role_is_predicted_in_batches_not_materialized`. Not done: caching a loaded
spine across `materialize` calls (would take the remaining fixed cost, but
changes `materialize`'s path-based fast paths).

### `MLX-10` — whole-dataset reads and HMM-stage channels

**Why.** A project embedding study (pool molecules from chosen samples, embed
them with PCA or a diffusion map, cluster with KNN/Leiden, plot, and order
per-sample clustermaps by cluster) needs exactly what the ML data path already
does -- label/group tables (`MLX-01`), QC and dedup filters (`MLX-05`),
multi-window masks (`MLX-02`), deletion coordinate maps (`MLX-03`) and fast
partition-major reads (`MLX-09`) -- but over *all* selected rows, with no
train/test split. `smftools project embedding` covers none of those
selections, and re-implementing them for embeddings would duplicate `MLX`.
Two gaps stand in the way:

1. **No split-free read.** `bind_ml_job` binds per fold of a job's split.
2. **HMM channels cannot be selected.** The registry records a
   `<stage>_read_index` only for preprocess (resolved through its current
   generation); for hmm and spatial it looks beside the stage's top-level
   spine, but generation layouts keep the read index under
   `generations/<id>/`. `_stage_read_index` then finds nothing for `hmm`, so a
   channel on an HMM layer (`C_all_accessible_features`,
   `C_all_footprint_features`) fails at stage membership. The HMM written
   catalog does list its layers.

**What.**

- `bind_ml_dataset(plan, dataset_name, *, project_dir=...) -> BoundDataset`:
  selection, snapshot and one `PartitionDataset` over every selected row
  (a single all-rows role), with `iter_batches()`/`materialize()` as for a fold.
  Unlabelled datasets allowed; a label table's extra columns (group labels)
  are carried on the bound rows for plotting.
- Stage read indexes and catalogs for hmm/spatial resolve through the stage's
  `current.json` generation, in selection (`_stage_read_index`,
  `_stage_task_catalog`) and in the registry at `project add`.
- Multi-stage datasets (a preprocess `X` channel plus HMM-layer channels) are
  already supported by the reader per stage; verify end to end on real stores.

**Tests.** A fixture whose hmm stage uses the generation layout resolves an
HMM-layer channel; `bind_ml_dataset` yields every selected row exactly once,
in manifest order from `materialize()`; a preprocess + hmm multi-channel
dataset reads both stages' values for the same molecules. Real data: a
fresh-NK embedding set (B6 active, B6 inactive, enh-del) over C sites,
`C_all_accessible_features` and `C_all_footprint_features`.

**As implemented.** `bind_ml_dataset(plan, dataset, project_dir=...,
group_by=...)` returns a `BoundDataset` (selection, snapshot, one
`PartitionDataset` over every row via a single all-rows role; `identity`,
`iter_batches()`, `materialize()`); `group_by` carries extra per-row columns
such as `Barcode`. A plan's `splits`, `models` and `jobs` may now be empty.
`_stage_read_index` falls back to the stage's current generation
(`resolve_current_generation`), which also makes the written catalog and stage
obs resolvable there; `_discover_catalogs` records `hmm_read_index` /
`spatial_read_index` the same way at `project add`. No reader change was needed
for preprocess + hmm channels.

Tests: `test_dataset_reads.py` (HMM stage published through
`staged_generation`, canonical spine at the stage root):
`test_hmm_channel_reads_through_its_generation` (values of both stages per
molecule; fails with the generation lookup reverted),
`test_whole_dataset_materializes_in_manifest_order`,
`test_registry_records_the_hmm_read_index_in_its_generation`,
`test_a_plan_may_declare_datasets_only`. Real data: the fresh-NK set (B6
active / B6 inactive / enh-del, downstream of the deletion in B6 coordinates)
binds 36,981 molecules over 3,151 positions with C-site, accessible-feature
and footprint-feature channels all populated; ~30 rows/s over two stages.
HMM channels use the C-site design mask, so they are read at C sites only.

### `MLX-11` — block-sharded workers (`F69`)

`iter_batches(worker_id, num_workers)` sharded by batch index, but since
`MLX-09` a worker decodes whole blocks of several batches. Every block held
batches of every worker, so N workers each decoded every block: parallel reads
were N times the work, and a real embedding read (~31k molecules, two stages,
four channels) ran on one core for 15+ minutes. Workers now take whole blocks
(block *k* -> worker *k* mod N); batches stay disjoint and complete, and each
block is decoded once. `BoundDataset.iter_batches` passes the sharding
through. Test: `test_workers_split_blocks_so_each_is_read_once` (12 block
reads instead of 6 with the old sharding), and
`test_whole_dataset_reads_split_across_workers`.

