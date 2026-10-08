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

## `in-progress/`

An active branch, some items merged and others open.

| document | scope |
|---|---|
| `alignment_rescue_sequence_implementation_plan.md` | `ARS` rescued reads keep their SEQ (minimap2 omits it on secondaries, so every rescued read was dropped at extraction, `F64`), repair of committed alignments, raw re-extraction |
| `hmm_context_emissions_implementation_plan.md` | `HCE` (`HCE-01`–`HCE-04`, `HCE-06`, `HCE-08`, `HCE-09` merged; `HCE-05` qualified: default `none`, `learned` opt-in; `HCE-07` proposed) sequence-context-aware HMM emissions: per-(state, context) modification probabilities as relative weights on the log-odds scale; `table` (e.g. naked-DNA calibration) or `learned` in EM with shrinkage; CpG kept separate; qualified on a multi-enzyme panel before any default changes |
| `ml_project_labels_masks_coordinate_maps_plan.md` | `MLX` project-scope (`MLX-01`–`MLX-03`, `MLX-05`–`MLX-07`, `MLX-09`–`MLX-11` merged) ML studies: external label table (`labels.source: table`), multi-window position masks, cross-reference coordinate maps with a leakage guard |
| `selective_pod5_rebasecalling_implementation_plan.md` | `SRB` selective POD5 re-basecalling and processing lineages; `SRB-01`–`SRB-09` merged, one open item: basecalls onto the shared generation layout |
| `duplicate_detection_span_agnostic_implementation_plan.md` | `DSA` span-agnostic duplicate detection; `DSA-01`–`DSA-04`, `DSA-06` merged, `DSA-05` measured (hierarchical-cap revisit open) |
| `transfer_time_analysis_bundling_plan.md` | `TAB` bundle a run's analysis-tree generations into few large files before moving them between drives; `TAB-01`, `TAB-02` merged, `TAB-03` blocked (no tested configuration beat plain rsync); zarr v3 sharding and coarser source-side partitioning both ruled out first, on real data |
| `motif_scanning_occupancy_implementation_plan.md` | `MOT-01`–`MOT-06` (`MOT-01` implemented) motif scanning (user-supplied motif file; built-in numpy engine, optional FIMO), bulk HMM-class tracks with motif lanes, and per-molecule motif occupancy from HMM classes (TF-sized, medium, nucleosome, accessible, uninformative), co-occupancy and group comparisons; an analysis + CLI, not a stage |

## `proposed/`

A plan with no implementation branch yet.

| document | scope |
|---|---|
| `agent_files_plan.md` | restructuring the repo's `AGENTS.md`/`CLAUDE.md` files; explicitly not deployed |
| `pipeline_throughput_implementation_plan.md` | `THR-01`–`THR-06` -- single-threaded, pool-collapse and redundant-scan bottlenecks found running `experiment batch full` (`F53`–`F57`, `F59`): alignment rescue, duplicate-detection group sizing, latent, raw extraction |
| `generation_prune_scope_implementation_plan.md` | `EGL-03c` -- fan the existing read-only, dry-run-only prune planner (`EGL-03a`) out to a project and to an arbitrary directory of run roots; does not touch `EGL-03b` (deletion), still blocked |

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
