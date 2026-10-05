# ML project labels, position masks and coordinate maps (`MLX`)

**Status:** in progress. `MLX-01` merged; `MLX-05` implemented on
`feature/mlx-05-real-store-selection`, not merged; `MLX-02`–`MLX-04`, `MLX-06`,
`MLX-07` proposed. `MLX-06` blocks any training from a plan.

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
| `MLX-02` position masks | proposed | include/exclude windows within a dataset's span |
| `MLX-03` cross-reference coordinate maps | proposed | place several references' molecules in one coordinate frame |
| `MLX-04` qualification | proposed | one real project study end to end |
| `MLX-05` real-store compatibility | implemented, not merged | resolve channels and QC filters against real pipeline stores (`F66`) |
| `MLX-06` plan job runner | proposed | bind a resolved plan to a dataset snapshot, split and partition dataset, and run its jobs |
| `MLX-07` training-only groups | proposed | let single-class groups train without being held-out folds |

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

### `MLX-07` — training-only groups

Leave-one-group-out refuses a fold whose held-out group lacks a class. A
study may still want such a group's rows in training (an experiment with only
active samples). Add a split option naming groups that only ever train, so
folds are the remaining groups and every fold's train set includes them.
Tests: the named groups appear in no test role and in every train role.

