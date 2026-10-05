# ML project labels, position masks and coordinate maps (`MLX`)

**Status:** proposed. Nothing implemented.

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
| `MLX-01` external label table | proposed | `labels.source: table` joined on declared keys |
| `MLX-02` position masks | proposed | include/exclude windows within a dataset's span |
| `MLX-03` cross-reference coordinate maps | proposed | place several references' molecules in one coordinate frame |
| `MLX-04` qualification | proposed | one real project study end to end |

Order: `MLX-01` first (it alone unblocks a single-reference, full-span pilot),
then `MLX-02`, then `MLX-03`, which builds on `MLX-02`'s guard.

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
