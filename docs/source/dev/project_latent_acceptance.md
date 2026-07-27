# Project and latent acceptance coverage

The project/partitioned-latent audit is enforced by focused tests rather than
one monolithic workflow fixture. This keeps failures attributable while still
covering every ownership, lifecycle, resource, and portability boundary.

| Acceptance dimension | Automated coverage |
| --- | --- |
| One and two experiments, including duplicate bare read IDs | `test_project_catalog.py::test_project_adata_concats_across_experiments`, `test_project_catalog.py::test_same_instrument_read_id_is_independently_addressable`, `test_project_latent_store.py::test_relocated_multi_experiment_latent_access_keeps_duplicate_read_identity` |
| Locus and genome analysis cores | `test_latent_partitioned_cli.py::test_analysis_units_are_reference_or_core_local` |
| Preprocess, spatial, and HMM latent sources | `test_latent_partitioned_cli.py::test_partitioned_latent_explicit_stage_selects_named_spine`, `test_latent_partitioned_cli.py::test_latent_cli_prefers_partitioned_hmm_spine` |
| Auto, partitioned, and legacy execution | `test_latent_partitioned_cli.py` source-resolution and compatibility tests |
| Full workflow latent default, opt-out, ordering, and failure propagation | `test_full_recipe.py::test_full_flow_runs_raw_preprocess_spatial_hmm_latent_in_order`, `test_full_recipe.py::test_full_flow_can_disable_latent`, `test_full_recipe.py::test_full_flow_records_latent_failure` |
| PCA/UMAP, NMF, and CP enabled/disabled behavior | `test_latent_partitioned_cli.py::test_fitted_latent_space_transforms_additional_reads`, `test_latent_partitioned_cli.py::test_latent_cp_memory_policy_is_deterministic`, `test_LoadExperimentConfig.py::test_disabled_latent_algorithms_ignore_unused_settings` |
| Roomy, fit-reducing, and minimum-unit resource profiles | `test_latent_resource.py`, `test_latent_partitioned_cli.py::test_latent_unit_applies_effective_fit_and_transform_counts`, `test_resource_runtime.py` |
| Fresh, restart, plot-only, config/source change, failure, force-redo, and compatible growth | Lifecycle tests in `test_latent_partitioned_cli.py`, including `test_latent_growth_reuses_model_and_transforms_only_new_rows` |
| Genomic materialization/export and scoped latent access/export | `test_project_catalog.py`, `test_project_latent_store.py` |
| Project embedding fit, exact read, extend, forced refit, and interrupted publication | `test_project_embedding_store.py` |
| Original and relocated experiment/project trees | `test_project_registry.py::test_project_survives_being_copied_to_a_different_absolute_path`, project embedding relocation, scoped latent export relocation, and duplicate-identity relocation tests |
| Unique identities, duplicate bare IDs, and one molecule in multiple cores | Project catalog identity tests and `test_project_latent_store.py::test_scoped_reader_yields_independent_owners_without_combining_coordinates` |
| Project registration refreshes new stage/index pointers | `test_project_registry.py::test_add_experiment_refresh_discovers_new_latent_stage_and_index` |
| Model identity, trust, tamper, dependency, and CP provenance | `test_latent_model_artifacts.py` and latent generation publication validation |
| Legacy periodicity, embedding, latent, and generic-materialization migration boundaries | `test_project_sample_analysis.py`, `test_project_embedding_store.py`, and `test_project_latent_store.py` |

Pull requests affecting this surface run focused project/latent/config/index
tests, the complete unit and integration tiers, smoke tests for touched CLI
paths, repository-wide Ruff and format checks, and the warning-as-error Sphinx
build. External-tool E2E workflows remain local because they require configured
Dorado, minimap2, or modkit inputs; their exclusion does not weaken the
partitioned project/latent storage contracts above.
