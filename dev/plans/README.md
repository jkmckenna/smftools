# Design records

Architecture audits and implementation plans for smftools. See `AGENTS.md` for
conventions: the three kinds of document, and the hard rules (no absolute paths,
no sequencing-run names, cite findings not datasets).

One line per document. Detail belongs in the document — an index that summarises
goes stale and is then worse than nothing.

## `audits/`

Investigations of the code as it is. An audit never "completes"; it goes stale.
Each carries a **Repository state reviewed** block naming the commit it describes
and how far `main` has moved since — between 190 and 538 commits, so treat every
specific claim as needing re-verification.

`load_preprocess_audit.md` is **superseded**: it describes the pre-partitioned
architecture, which no longer exists. `input_ingestion_alignment_audit.md` has no
recoverable anchor and says so.

| document | scope | plan it motivated |
|---|---|---|
| `experiment_project_partitioned_pipeline_audit.md` | partitioned experiment/project pipeline | `completed/experiment_project_partitioned_pipeline_implementation_plan.md` |
| `project_and_latent_partitioned_pipeline_audit.md` | project and latent stages | `completed/project_and_latent_partitioned_pipeline_implementation_plan.md` |
| `variant_preprocessing_incremental_reprocessing_audit.md` | incremental variant reprocessing | `completed/semantic_dag_variant_preprocessing_implementation_plan.md` |
| `input_ingestion_alignment_audit.md` | input ingestion and alignment | `completed/input_ingestion_alignment_implementation_plan.md` |
| `selective_pod5_rebasecalling_audit.md` | selective re-basecalling from pod5 | `in-progress/selective_pod5_rebasecalling_implementation_plan.md` |
| `ml_infrastructure_audit.md` | ML infrastructure as of 2026-07-30 | `completed/ml_implementation_ledger.md` |
| `ml_audit_second_opinion.md` | independent review of the ML infrastructure audit | `completed/ml_implementation_ledger.md` |
| `ml_behavior_inventory.md` | `ML-001` inventory of ML behaviour and migration surface | `completed/ml_implementation_ledger.md` |

## `completed/`

Every tracked item merged to `main` and verified against the code.

| document | scope |
|---|---|
| `duplicate_detection_scaling.md` | bitpacking, chunked union-find, permutation banding (`e18d593`) |
| `experiment_project_partitioned_pipeline_implementation_plan.md` | `PR-00`–`PR-14` |
| `materialize_read_cost_implementation_plan.md` | `MRC-01`–`MRC-05` `materialize` read cost (`F70`, `F71`): index-directed partition reads, spine cache, overlay of owning stores only, skip `X` for derived-only requests; one call ~16 s -> under 1 s |
| `demux_refresh_implementation_plan.md` | `DRF-01`–`DRF-03` demux refresh without re-extraction: sequencing-summary end calls fill `unclassified`, reassembly applies them and rewrites the segment catalog and pointer indexes, `experiment refresh-demux` publishes a sibling preprocess generation with re-chosen duplicate keepers (PR #725); `DRF-04` default keeper preference open |
| `read_periodicity_implementation_plan.md` | `RPG-01`–`RPG-05` per-read periodograms over regions for any plan channel (`site_context: all` for dense layers, `F73`), range narrowed to short regions, paired clustermaps, `smftools project/experiment periodicity`, shared cache key (`F72`), a block per worker (`F74`); equal to the spatial stage's periodograms |
| `read_periodicity_figures_implementation_plan.md` | `RPF-01`–`RPF-04` periodicity figures after first use: descending order, display coordinates, several groupings per run, grid figures, mean spectra with bootstrap bands and per-group summaries |
| `stage_context_qc_implementation_plan.md` | `SCQ-01`–`SCQ-04` (`SCQ-01`–`SCQ-03` merged, `SCQ-04` qualified) sequence-context QC in every run: modification bias per barcode x reference in preprocess, residual context bias and accessible-conditioned rates per HMM variant, backfill for finished stages; plot/QC settings, no stage invalidation |
| `barcode_allowlist_implementation_plan.md` | `BAL` (merged #639, qualified) -- `barcodes_to_include`, so a run carrying several experiments (e.g. two modalities on one flow cell) keeps each to its own barcodes |
| `site_context_bias_implementation_plan.md` | `SCB-01`–`SCB-04` modification-site sequence-context bias: strand-oriented contexts, per-offset enrichment, k-mer rates, group differences, figures; `smftools project/experiment context-bias` |
| `project_and_latent_partitioned_pipeline_implementation_plan.md` | `PL-15`–`PL-23` (PR #414) |
| `semantic_dag_variant_preprocessing_implementation_plan.md` | `SDV-01`–`SDV-14` |
| `input_ingestion_alignment_implementation_plan.md` | `IAR-01`–`IAR-15` (PRs #468–#488), `PCLI-01`–`PCLI-04` (PRs #489–#493); coverage in `tests/acceptance/*.json` |
| `ml_implementation_ledger.md` | `ML-001`–`ML-503`, the ML migration; plan and development ledger fused in one document |
| `ml700_benchmark_plan.md` | `ML-700` performance and scalability qualification |
| `smftools_raw_load_plan.md` | the v2.0.0 `raw`/`load` split; thin spine over a partitioned ragged store |
| `experiment_storage_schema.md` | formal parquet/zarr storage schema; all four phases, each narrower than first sketched |
| `project_sample_and_set_stores.md` | project-level per-sample and set stores; a set is a query, not a concat cache |
| `generation_lifecycle_and_naming_implementation_plan.md` | `EGL` generation lifecycle and experiment naming; the `NKG` rollout continues as a log in `logs/` |
| `portable_storage_roots_implementation_plan.md` | `PSR-01`–`PSR-20`; one tracked exception -- `PSR-19`'s in-band catalog updates on publish, `data scan` is the substitute |
| `basecall_stage_and_source_selection_implementation_plan.md` | `BCS-01`–`BCS-11`; two tracked exceptions -- `BCS-06` is a `full`-only pre-step rather than a semantic-DAG node, `BCS-09` ships as same-volume reporting rather than scheduling enforcement (no batch orchestrator exists yet to enforce within) |
| `stage_upstream_guard_implementation_plan.md` | `SUG-01` stage commands refuse to recompute stale upstream stages unless `--allow-upstream` (merged); `SUG-02` stage-only refit folded into `HPS-07` |
| `hmm_cross_reference_fit_implementation_plan.md` | `HXR-01`–`HXR-06` HMM fits shared across a declared group of references (`hmm_reference_groups`), accepted on a two-allele run; shared emissions (`HPS`) now cover the allele artefact by default |
| `generation_prune_scope_implementation_plan.md` | `EGL-03c` project / directory-wide prune planning -- superseded, folded into `SLC-09` |

## `in-progress/`

An active branch, some items merged and others open.

| document | scope |
|---|---|
| `alignment_rescue_sequence_implementation_plan.md` | `ARS` rescued reads keep their SEQ (minimap2 omits it on secondaries, so every rescued read was dropped at extraction, `F64`), repair of committed alignments, raw re-extraction |
| `hmm_context_emissions_implementation_plan.md` | `HCE` sequence-context-aware HMM emissions: `HCE-01`–`HCE-10` merged, `HCE-05` / `HCE-07` qualified; defaults `learned` + every-state weights for deaminase configs (`HPS`); `HCE-11` proposed |
| `ml_project_labels_masks_coordinate_maps_plan.md` | `MLX` project-scope (`MLX-01`–`MLX-03`, `MLX-05`–`MLX-07`, `MLX-09`–`MLX-11` merged) ML studies: external label table (`labels.source: table`), multi-window position masks, cross-reference coordinate maps with a leakage guard |
| `selective_pod5_rebasecalling_implementation_plan.md` | `SRB` selective POD5 re-basecalling and processing lineages; `SRB-01`–`SRB-09` merged, one open item: basecalls onto the shared generation layout |
| `duplicate_detection_span_agnostic_implementation_plan.md` | `DSA` span-agnostic duplicate detection; `DSA-01`–`DSA-04`, `DSA-06` merged, `DSA-05` measured (hierarchical-cap revisit open) |
| `transfer_time_analysis_bundling_plan.md` | `TAB` bundle a run's analysis-tree generations into few large files before moving them between drives; `TAB-01`, `TAB-02` merged, `TAB-03` blocked (no tested configuration beat plain rsync); zarr v3 sharding and coarser source-side partitioning both ruled out first, on real data |
| `motif_scanning_occupancy_implementation_plan.md` | `MOT-01`–`MOT-06` (`MOT-01`–`MOT-04` merged; `MOT-05`, `MOT-06` proposed) motif scanning (user-supplied motif file; built-in numpy engine, optional FIMO), bulk HMM-class tracks with motif lanes, and per-molecule motif occupancy from HMM classes (TF-sized, medium, nucleosome, accessible, uninformative), co-occupancy and group comparisons; an analysis + CLI, not a stage |
| `barcode_qc_end_identity_implementation_plan.md` | `BQC-01`–`BQC-05` (`BQC-01`–`BQC-03b` merged; `BQC-01b`, `BQC-04`, `BQC-05` proposed) per-end barcode evidence (names, edit distances, boundaries, clips) for every read; a ligation / detection / tagging model separating miscalls, unbarcoded rear ends and free-adapter tagging; extract-then-trim before alignment; with / without a spike-in; extends `EGL-31` |
| `storage_lifecycle_implementation_plan.md` | `SLC-01`–`SLC-12` disk-space lifecycle: compressed model payloads and compaction, model persistence policy and unread-model retirement (extends `MLR-16`), reference-aware generation pruning with archive / confirmed delete (replaces the `EGL-03b` blocker, folds in `EGL-03c`), pre-stage space check, one `data usage` report (`SLC-01`, `SLC-02`, `SLC-06`–`SLC-08`, `SLC-11` implemented) |
| `hmm_parameter_scopes_implementation_plan.md` | `HPS` transitions per sample x reference x strand, emissions shared per emission group with technical efficiency offsets (tied EM), per-bp axis, named fitting plans, layer sets, new HMM defaults (`HPS-01`–`HPS-05`, `HPS-07`, `HPS-11` merged); `HPS-07` layer sets, `HPS-12` genome-mode fitting; next `HPS-08`; `HPS-06` deferred |

## `proposed/`

A plan with no implementation branch yet.

| document | scope |
|---|---|
| `agent_files_plan.md` | restructuring the repo's `AGENTS.md`/`CLAUDE.md` files; explicitly not deployed |
| `pipeline_throughput_implementation_plan.md` | `THR-01`–`THR-06` -- single-threaded, pool-collapse and redundant-scan bottlenecks found running `experiment batch full` (`F53`–`F57`, `F59`): alignment rescue, duplicate-detection group sizing, latent, raw extraction |
| `ml_model_runs_interpretability_plan.md` | `MLR-01`–`MLR-14` model runs as self-describing records (task, data, molecules per fold, fitted models, evaluation, curves), applying saved runs to new data, interpretability per run (position importance, per-molecule attributions, attribution clustermaps beside the input), run comparison, fold-matrix cache, validation role, detector catalogue for CNNs, detector-scale CNNs (pattern-detector span from sub-nucleosome to full locus; theoretical and effective span), model classes and per-task zoos (additive, tabular non-linear, spatial, global sequence), pretraining (masked reconstruction / AE / VAE encoders) and fine-tuning with a declared leakage policy |
| `chimera_sensitivity_implementation_plan.md` | `CHS-01`–`CHS-04` detected vs estimated chimera rates: in-silico detector sensitivity, blind (same-allele / same-strand) chimeras, detector overlap |
| `meta_read_assembly_plan.md` | `MRA-01`–`MRA-09` assemble overlapping fragments of PCR copies of one molecule (rapid / transposase libraries, conversion chemistries) into full-length meta reads from `DSA` duplicate clusters -- strict shared-modification evidence, conflict-free layouts, consensus with support -- published as a processing lineage so spatial / HMM / latent run on it beside the read-level chain |

## `logs/` — not tracked

Append-only records that never reach "complete", and where measurements from
unpublished experiments land first.

| document | scope |
|---|---|
| `pipeline_findings.md` | `F17`–`F50`, findings from running the pipeline; append-only |
| `nkg_regeneration_rollout.md` | `NKG-01`–`NKG-06`; a naming scheme over twenty named experiments, so the identifiers are the content |

## Not tracked here

Project-specific drivers -- code and plans tied to one lab dataset rather than
to smftools -- belong in the analyses repository, not in the design records. The
ML migration's per-project driver is an example: it is named for the dataset it
migrates and describes that project's slice, not the library's design.
