# Migrating legacy machine-learning workflows

smftools 2.x retains compatibility adapters for the original matrix-, AnnData-, wrapper-, and
sliding-window-oriented ML APIs. These adapters emit `FutureWarning` when called or instantiated
and are scheduled for removal in smftools 3.0. New workflows should use the versioned plan,
manifest-bound datasets, registered model recipes, job services, and immutable artifacts under
`smftools.machine_learning`.

## Ownership boundary

- `smftools.machine_learning` owns data selection and splits, input and mask contracts, registered
  models, training, inference, evaluation records, explanation execution, and artifact publication.
- `smftools.analysis.compute` owns pure transformations from stored results to summary tables.
- `smftools.analysis.plot` owns plots rendered from explicit result tables and output paths.
- Project code owns biological labels, selectors, class meaning, model choices, and action plans;
  it should not reimplement generic training, evaluation, explanation, or artifact management.

The old `analysis.compute.ml_cnn` and `analysis.compute.ml_explanations` paths remain importable,
but their model-execution implementations now live behind an internal compatibility boundary.
The pure functions in `analysis.compute.ml_metrics`, such as logit conversion and result-only
evaluation, remain available. Its estimator construction, fitting, and prediction helpers are
deprecated.

## Replacement map

| Legacy surface | Canonical replacement |
|---|---|
| `analysis.compute.ml_cnn.fit_simple_cnn` | `machine_learning.orchestration.train_partition_model` |
| `analysis.compute.ml_cnn.predict_cnn_scores` | `machine_learning.orchestration.apply_partition_model` |
| `analysis.compute.ml_cnn.integrated_gradients_attributions` | `machine_learning.orchestration.explain_partition_model` |
| `analysis.compute.ml_metrics.build_binary_classifier` | `machine_learning.models.BUILTIN_MODEL_REGISTRY` |
| `analysis.compute.ml_metrics.fit_classifier` | `machine_learning.training.fit_sklearn_partition_model` |
| `analysis.compute.ml_explanations.*` | `machine_learning.interpretability.explain_sklearn_model` |
| `analysis.compute.ml_splits.build_leave_one_group_out_splits` | `machine_learning.splitting.plan_ml_splits` |
| `machine_learning.data.AnnDataModule` and `build_anndata_loader` | `PartitionDataset` or `MaterializedDataset` bound to manifests |
| Prototype `CNNClassifier`, `MLPClassifier`, `RNNClassifier`, and Transformer classes | A validated registered recipe; currently `ResidualDilatedCNN1d` for neural classification |
| `TorchClassifierWrapper` and `train_lightning_model` | `fit_torch_partition_model` or a canonical train job |
| `SklearnModelWrapper` and `train_sklearn_model` | `fit_sklearn_partition_model` or a canonical train job |
| Legacy AnnData inference and sliding-window helpers | Explicit apply/evaluate jobs and immutable `PredictionResult` records |
| `ModelEvaluator` and `PostInferenceModelEvaluator` | `evaluate_predictions`, then `analysis.compute.ml_results` |

XGBoost remains a legacy-only candidate: it is not a registered built-in family. Register and test
a family against the same input, persistence, evaluation, and explanation contracts before using
it in a canonical workflow.

## Migration sequence

1. Express sample selection, labels, class order, split groups, balancing, models, and requested
   actions in an ML plan.
2. Resolve a project- or experiment-owned ML workspace. Do not construct output paths inside model
   or analysis code.
3. Materialize through `PartitionDataset`; use `MaterializedDataset` only when a validated external
   matrix must be bound to canonical dataset and split manifests.
4. Train and apply registered sklearn or plain-Torch families through the canonical services.
5. Evaluate natural-prevalence predictions first. Record any balanced sensitivity cohort as a
   distinct evaluation artifact rather than changing the held-out test membership.
6. Request explanations through the capability-aware interpretability service with explicit target,
   cohort, mask policy, and training-derived background.
7. Read immutable prediction, evaluation, history, and explanation records into
   `analysis.compute.ml_results`, then render plots with explicit paths under the resolved workspace.

Do not silence the compatibility warnings globally. They identify the remaining call sites that
must migrate before upgrading to smftools 3.0.
