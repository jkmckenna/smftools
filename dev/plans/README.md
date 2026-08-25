# Design records

Architecture audits and implementation plans for smftools. See `AGENTS.md` for
conventions: the three kinds of document, and the hard rules (no absolute paths,
no sequencing-run names, cite findings not datasets).

One line per document. Detail belongs in the document — an index that summarises
goes stale and is then worse than nothing.

## `audits/`

Investigations of the code as it is. An audit never "completes"; it goes stale.
Each should carry an **audited against `<sha>`** marker — none currently do,
which is worth fixing as each is next touched.

| document | scope | plan it motivated |
|---|---|---|
| `experiment_project_partitioned_pipeline_audit.md` | partitioned experiment/project pipeline | `completed/experiment_project_partitioned_pipeline_implementation_plan.md` |
| `project_and_latent_partitioned_pipeline_audit.md` | project and latent stages | `completed/project_and_latent_partitioned_pipeline_implementation_plan.md` |
| `variant_preprocessing_incremental_reprocessing_audit.md` | incremental variant reprocessing | `completed/semantic_dag_variant_preprocessing_implementation_plan.md` |
| `input_ingestion_alignment_audit.md` | input ingestion and alignment | `completed/input_ingestion_alignment_implementation_plan.md` |
| `selective_pod5_rebasecalling_audit.md` | selective re-basecalling from pod5 | `in-progress/selective_pod5_rebasecalling_implementation_plan.md` |

## `completed/`

Every tracked item merged to `main` and verified against the code.

| document | scope |
|---|---|
| `duplicate_detection_scaling.md` | bitpacking, chunked union-find, permutation banding (`e18d593`) |
| `experiment_project_partitioned_pipeline_implementation_plan.md` | `PR-00`–`PR-14` |
| `project_and_latent_partitioned_pipeline_implementation_plan.md` | `PL-15`–`PL-23` (PR #414) |
| `semantic_dag_variant_preprocessing_implementation_plan.md` | `SDV-01`–`SDV-14` |
| `input_ingestion_alignment_implementation_plan.md` | `IAR-01`–`IAR-15` (PRs #468–#488), `PCLI-01`–`PCLI-04` (PRs #489–#493); coverage in `tests/acceptance/*.json` |

## `in-progress/`

An active branch, some items merged and others open.

| document | scope |
|---|---|
| `selective_pod5_rebasecalling_implementation_plan.md` | `SRB` |
| `experiment_storage_schema.md` | formal parquet/zarr storage schema; phases 1-2 implemented |
| `project_sample_and_set_stores.md` | project-level per-sample and set stores; set store v2 implemented |
| `smftools_raw_load_plan.md` | thin molecule-index AnnData over distributed storage |

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
| `generation_lifecycle_and_naming_implementation_plan.md` | `EGL` generation lifecycle (complete), `NKG` regeneration (active), `F20`–`F50` findings |

That document is three programs fused into one and should be split — `EGL` into
`completed/`, `NKG` into `in-progress/`, the findings log staying here. It is
twice the length of its nearest neighbour, which is the signal.
