# Derived layers and lean generations (`DLS`)

Status: in progress. Items `DLS-01`--`DLS-08`; `DLS-01` implemented. Works
with `SLC` (`in-progress/storage_lifecycle_implementation_plan.md`), which
manages generations once written; `DLS` reduces what a generation writes.
Takes over the compact-feature-storage part of `HPS-08`
(`in-progress/hmm_parameter_scopes_implementation_plan.md`).

## Why

One experiment's current chain (260820, raw -> latent, 2026-10-17) is 37.6 GB:

| stage | total | layer store | `spine.h5ad` | `csvs/` | plots |
|---|---|---|---|---|---|
| raw | 12.3 G | (ragged calls) | 0.8 G | 1.3 G | -- |
| preprocess | 11.2 G | 5.7 G | 1.3 G | 3.2 G | 0.3 G |
| spatial | 4.1 G | 0.5 G | 1.3 G | 1.9 G | 0.4 G |
| hmm | 6.6 G | 2.2 G | 1.3 G | 1.9 G | 1.1 G |
| latent | 3.4 G | 0.1 G | 1.3 G | 1.9 G | 0.1 G |

- `csvs/` -- CSV copies of every spine's obs / var -- are 10.2 GB (27 %);
  nothing reads them.
- Every stage's spine carries the whole read table again (~6 GB of copies).
- Preprocess: `nan_half` 2.67 GB (46 % of its store) and `nan0_0minus1`
  0.44 GB are fixed transforms of `X` (2.38 GB, stored).
- HMM: the posterior is 1.02 GB (53 % of the store) as float; states,
  feature classes, lengths and merged layers ~0.8 GB -- all computable from
  the posterior in milliseconds. Storing them also freezes their settings:
  changing a merge distance (e.g. 60 -> 40 bp) re-runs every experiment's HMM
  stage.

Together ~20 of 38 GB per experiment of this size.

Raw intermediates add more: 260820 keeps three BAMs (27.7 GB), 102 GB across
the project's runs:

| intermediate | size | content (verified 2026-10-17) |
|---|---|---|
| `fastq-to-unaligned-bam` | 6.8 G | the FASTQ as BAM: 1,753,696 reads, name / SEQ / QUAL / `BC` / `RG` |
| `alignment-sort-index` | 10 G | every alignment (6.43M records) and alignment tags; 3,301 duplicate primaries (pre-`F64` rescue) |
| `restore-primary-sequences` | 10 G | the same records with one primary per read (1,753,696, unmapped included): its primaries reproduce the unaligned BAM exactly (name, SEQ, QUAL, `BC`, `RG`; identical checksums over all reads) |

CRAM 3.1 of `restored.bam`: 6.3 GB (-37 %), 98 s to write on 4 threads, full
decode as fast as BAM (71 s vs 72 s), every record identical (core fields
and tag values; tag order differs as CRAM rebuilds `MD` / `NM`). A
BAM-input run (SUP basecalls) keeps every basecall tag (`qs`,
`mv`, `ns`, `ts`, `ch`, `st`, `du`, barcode tags) on its aligned records.

## Principle

Store what is expensive or irreplaceable -- raw calls, `X`, dedup clusters,
fitted models, the HMM posterior, latent embeddings. Derive on request what
is a cheap deterministic function of stored layers, with its settings as
read-time parameters. A derived layer is reproducible: its function has a
version, and the version and settings are recorded wherever a result is
written (stage outputs, analysis manifests, exports). A stored layer always
wins over a derivation of the same name, so existing generations read as
before.

## Items

| id | item | status |
|---|---|---|
| `DLS-01` | `write_csv_exports` (default off): obs / var / keys CSV copies beside a written spine only when enabled; set when the experiment config loads, inherited by worker processes; not a semantic key (no stage becomes stale) | implemented |
| `DLS-02` | posterior provenance and encoding: the posterior layer records the model id, `fit_config_hash`, checksum, axis and encoding per task; stored as 8-bit (0-254 = p x 254, 255 = no call), decoded to float on read; float generations still read | proposed |
| `DLS-03` | HMM features derived from the posterior: one versioned function builds states, size classes, aggregates, lengths and merged layers from the posterior and the feature settings (`hmm_feature_sets`, `hmm_merge_layer_features`, posterior threshold); the HMM stage uses it for plots and summaries without storing the layers; `materialize(..., derived=...)` builds them on request with the generation's recorded settings or overrides, and records what it used | proposed |
| `DLS-04` | preprocess transforms derived from `X` (`nan_half`, `nan0_0minus1`, and any other fixed transform found by the same audit) | proposed |
| `DLS-05` | one read table per experiment: stage spines store only the columns they add and point at the experiment's table; `materialize` and `load_spine` join them; old spines (full obs) read unchanged | proposed |
| `DLS-06` | exports and external readers: bundles and `export-*` write derived layers out (an external tool reading the zarr does not see derivations) | proposed |
| `DLS-07` | `experiment full CONFIG --through hmm` as a CLI form of `full_run_latent: false` | proposed |
| `DLS-08` | raw intermediates: keep one copy of each expensive step. FASTQ input: the FASTQ (source) + one aligned file; the unaligned BAM is deleted once alignment is committed (rebuilt from the FASTQ in minutes). POD5 / basecall-BAM input: the basecall BAM (GPU hours, model-version dependent) + one aligned file. Legacy runs with `restored.bam`: it is the aligned copy and `aligned_sorted.bam` goes. The kept aligned file is CRAM 3.1 (lossless, -37 %); intermediate caching resolves it like a BAM; a deleted intermediate is recomputed on demand. Each deletion is checked first (record counts; for the unaligned BAM, read content against the aligned primaries). 260820: 27.7 -> ~6.3 GB; ~100 -> ~25 GB across the project. POD5 retention (archive after basecalling; needed by selective re-basecalling) is a separate decision | proposed |

## Order and acceptance

`DLS-01` first (largest saving, no format change). `DLS-02` before `DLS-03`
(derivations read the posterior with its provenance). Each derivation item is
accepted on real data: for a 260820 generation, the derived layers equal the
stored ones exactly (same settings), and the stage's plots are unchanged.
`DLS-05` last: it touches every stage writer and reader.

## Not in scope

Raw-store compaction (`SLC`); model payload compaction (`SLC-02`, done);
pruning and archiving generations (`SLC`).
