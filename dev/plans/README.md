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
| `project_and_latent_partitioned_pipeline_implementation_plan.md` | `PL-15`–`PL-23` (PR #414) |
| `semantic_dag_variant_preprocessing_implementation_plan.md` | `SDV-01`–`SDV-14` |
| `input_ingestion_alignment_implementation_plan.md` | `IAR-01`–`IAR-15` (PRs #468–#488), `PCLI-01`–`PCLI-04` (PRs #489–#493); coverage in `tests/acceptance/*.json` |
| `ml_implementation_ledger.md` | `ML-001`–`ML-503`, the ML migration; plan and development ledger fused in one document |
| `ml700_benchmark_plan.md` | `ML-700` performance and scalability qualification |
| `smftools_raw_load_plan.md` | the v2.0.0 `raw`/`load` split; thin spine over a partitioned ragged store |
| `experiment_storage_schema.md` | formal parquet/zarr storage schema; all four phases, each narrower than first sketched |
| `project_sample_and_set_stores.md` | project-level per-sample and set stores; a set is a query, not a concat cache |
| `generation_lifecycle_and_naming_implementation_plan.md` | `EGL` generation lifecycle and experiment naming; the `NKG` rollout continues as a log in `logs/` |

## `in-progress/`

An active branch, some items merged and others open.

| document | scope |
|---|---|
| `selective_pod5_rebasecalling_implementation_plan.md` | `SRB` |
| `basecall_stage_and_source_selection_implementation_plan.md` | `BCS` basecalling as a stage and read-source selection; Phase 1 (selection) implemented, Phases 2-3 proposed |
| `portable_storage_roots_implementation_plan.md` | `PSR` portable storage roots and multi-location tracking; Phase 1 (offline raw data) merged, Phases 2-5 proposed |
| `duplicate_detection_span_agnostic_implementation_plan.md` | `DSA` span-agnostic duplicate detection; drafted, `DSA-05` real-data qualification open |

## `proposed/`

A plan with no implementation branch yet.

| document | scope |
|---|---|
| `agent_files_plan.md` | restructuring the repo's `AGENTS.md`/`CLAUDE.md` files; explicitly not deployed |

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
