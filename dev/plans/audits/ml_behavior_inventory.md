# smftools ML behavior and migration inventory

> **Repository state reviewed:** `b8b5a90` — recorded in this document as `Committed source`.
> **335 commits on `main` since.** An audit describes the code at a moment; it
> goes stale rather than completing. Re-verify any specific claim before relying on it.

**Work package:** ML-001

**Inventory date:** 2026-07-30

**Committed source:** `b8b5a90`

**Branch:** `feature/ml-behavior-inventory`

**Status:** `IN_PROGRESS`

This inventory describes committed `main` only. Directories named `archived` are excluded. The
uncommitted ML changes in the main worktree are listed separately as candidates;
they are not current package behavior or compatibility obligations.

## Conclusions

1. No production package workflow or CLI imports either committed ML implementation.
2. `smftools.machine_learning` is exposed through the lazy `smftools.ml` alias, but its only
   in-repository consumers are import smoke tests.
3. The committed `analysis.compute.ml_*` modules are pure array/table helpers consistent with the
   analysis contract except that `ml_cnn` also owns models, training, device selection, and
   attribution computation.
4. Neither surface publishes a reusable model artifact. The Lightning trainer can emit a framework
   checkpoint to a caller-selected directory; the analysis CNN retains its best `state_dict` only
   in memory.
5. The current AnnData/Lightning data path is not safe to preserve as canonical behavior: it eagerly
   tensorizes, mutates/fills missing data without an observed mask, uses row-random splits, derives
   labels from categorical codes, and returns the training set from validation/test loaders in the
   default zero-worker path.
6. The useful committed behavior should be migrated by composition into
   `smftools.machine_learning`; pure metric/split summaries and plots can remain under
   `smftools.analysis`.

## Consumer and public-surface inventory

| Surface | Current exposure/consumer | Compatibility assessment |
|---|---|---|
| `smftools.ml` | Lazy alias in `smftools.__init__` to `smftools.machine_learning` | Preserve as a compatibility alias; document `smftools.machine_learning` as canonical. |
| `smftools.machine_learning.{data,evaluation,inference,models,training,utils}` | Lazy submodule exports | Keep namespace; do not promise every prototype class as stable. |
| `smftools.machine_learning.*` implementation symbols | Import smoke tests only | Behavioral replacement is allowed, but removal should use an inventory-backed compatibility note. |
| `analysis.compute.ml_cnn/ml_metrics/ml_splits/ml_explanations` | No committed caller in `src/` or `tests/` | Treat as project-consumed candidates because the audits observed external use; retain adapters until that use is confirmed. |
| `analysis.plot.ml` | No committed caller in `src/` or `tests/` | Keep result-driven plots; migrate only assumptions tied to legacy model objects. |
| CLI | No ML imports or commands | Add nothing until ML-501. |
| Downstream project scripts | Referenced by the audits but not present in any local worktree | Consumer compatibility remains unverified and must be supplied or re-scanned before ML-503. |

The test suite contains smoke import coverage for `machine_learning` modules and no behavioral tests
for splitting, loaders, fitting, persistence, inference, metrics, or explanations.

## Committed `analysis` ML surface

| Module/symbols | Current behavior | Disposition |
|---|---|---|
| `ml_metrics.build_binary_classifier`, `fit_classifier`, `predict_binary_scores` | Builds/fits Bernoulli NB, RF, and optional XGBoost pipelines over `(n_obs, n_features)` arrays. | **Migrate/adapt** builders and prediction behind sklearn recipes/adapters; add logistic regression in ML-301. |
| `ml_metrics.logit_from_probability`, `normalize_pr_auc`, `evaluate_binary_classifier`, `make_metrics_row` | Pure binary result computation. Optional evaluation resampling can alter prevalence. | **Keep/adapt in analysis**; natural prevalence must be primary and sensitivity resampling explicitly named. |
| `ml_splits.validate_disjoint_groups`, `summarize_split`, `build_leave_one_group_out_splits` | Pure metadata/index helpers; only train/test roles; infeasible groups are silently omitted. | **Adapt** summaries as pure analysis; replace split authority with ML-201 manifests and explicit failure/reporting. |
| `ml_explanations.bernoulli_nb_logodds_contributions` | Exact NB log-odds decomposition for a specific pipeline shape. | **Migrate computation** to classical explanation adapter; result summaries may remain pure analysis. |
| `ml_explanations.tree_shap_contributions`, `xgboost_contributions` | RF SHAP probability contributions and XGBoost native margin contributions. | **Migrate/adapt** under capability-dispatched interpretability with method/output provenance. |
| `ml_cnn.CNNConfig`, config converters/default, `ResidualDilatedCNN1d` and building blocks | Configurable residual/dilated 1D CNN consuming `(batch, channel, position)`. | **Migrate** as the initial named plain-PyTorch family/recipe. |
| `ml_cnn.build_cnn_input` | Builds signal + observed and optional design/position/spacing channels. | **Adapt** into the modality-aware input schema; replace one generic channel list with ordered biological channels and per-channel masks. |
| `ml_cnn.build_cnn_baseline`, `integrated_gradients_attributions`, `_LogitWrapper` | Integrated Gradients-specific baseline and wrapper behavior. | **Migrate/adapt** into neural explanation adapters after ML-401; persist target, baseline, channel role, and mask policy. |
| `ml_cnn.split_train_validation`, `fit_simple_cnn` | Plain-PyTorch binary trainer with row-level internal validation split and in-memory early stopping. | **Replace**, preserving useful optimization behavior only after group/split manifests exist. |
| `ml_cnn.TrainedCNNModel`, `predict_cnn_scores`, device helper | In-memory inference wrapper and score generation. | **Adapt** to Torch predictor/model artifacts; device selection remains an execution concern. |
| `analysis.plot.ml` plotting functions | ROC, PR, score distributions, read heatmaps, and metric bars from explicit data/output paths. | **Keep/adapt** under `analysis.plot`; make stored prediction/history tables the inputs. |

No committed analysis ML function writes model weights or a model manifest. `fit_simple_cnn` restores
an in-memory best state and returns it.

## Committed `smftools.machine_learning` surface

| Area/symbols | Current behavior | Disposition |
|---|---|---|
| `data.AnnDataDataset`, `random_fill_nans` | Eagerly converts a complete AnnData matrix/layer/obsm to one tensor and replaces NaNs with random values. | **Replace** with bounded partition adapters and explicit observed masks. |
| `data.split_dataset`, `AnnDataModule`, `build_anndata_loader` | Row-random train/val/test split stored in `obs`/CSV; Lightning and raw loaders. | **Replace**, retaining a temporary adapter only if a real consumer is found. |
| `models.BaseTorchModel` | `nn.Module` base coupled to saliency, IG, DeepLift, occlusion, GradCAM, and AnnData writes. | **Decompose/deprecate**; keep models plain and move explanation/application behavior to services. |
| `CNNClassifier`, `MLPClassifier`, `RNNClassifier` | Configurable prototype classifiers over flat arrays. | **Inventory as candidates**; residual CNN from `analysis` is the accepted initial neural family. |
| `BaseTransformer`, `TransformerClassifier`, `MaskedTransformerPretrainer` | Scalar/feature Transformer prototypes with attention buffers and pretraining concepts. | **Adapt later** only after input/mask contracts and consumer evidence. |
| `DANNTransformerClassifier`, `DANNTransformer`, `GradReverse` | Domain-adversarial prototype components. | **Defer** until modality/domain-confounding requirements justify a tested task. |
| `PositionalEncoding`, `ScaledModel` | Reusable model utilities. | **Keep/adapt** only when a canonical family needs them. |
| `TorchClassifierWrapper` | Lightning training/validation/test/predict wrapper with metrics and plots. | **Defer/deprecate as core**; a future Lightning adapter may wrap canonical tasks/models. |
| `SklearnModelWrapper` | Fitted-object wrapper, evaluation, SHAP, and AnnData attribution writes. | **Decompose** into sklearn predictor, evaluation, and explanation adapters. |
| `train_lightning_model`, `run_sliding_window_lightning_training` | Lightning trainer, early stopping, optional checkpoint directory, window loop. | **Defer/replace** after plain-PyTorch vertical workflow. |
| `train_sklearn_model`, `run_sliding_window_sklearn_training` | Fits/evaluates wrapper and plots immediately. | **Replace** with backend-neutral job services and immutable results. |
| sklearn/Lightning/sliding-window inference functions | Mutate AnnData with split and prediction columns using stored observation names. | **Adapt** prediction logic; replace mutable AnnData as the authoritative result with prediction tables. |
| `ModelEvaluator`, `PostInferenceModelEvaluator`, `flatten_sliding_window_results` | Wrapper-oriented metric/plot utilities. | **Replace/adapt** behind the shared prediction/metric contract. |
| `detect_device` | Device selection helper. | **Keep one implementation** under execution utilities. |

## Confirmed behavioral defects and regression targets

| ID | Current behavior | Required regression target |
|---|---|---|
| B-001 | `AnnDataModule.val_dataloader()` returns `train_set` when `num_workers` is falsy. | Validation loader yields exactly validation membership for zero and nonzero workers. |
| B-002 | `AnnDataModule.test_dataloader()` returns `train_set` when `num_workers` is falsy. | Test loader yields exactly locked test membership for zero and nonzero workers. |
| B-003 | `build_anndata_loader(..., lightning=False)` passes `split_save_path` and `load_existing_split` positionally in reversed semantic positions. | Named-argument construction and split round-trip test. |
| B-004 | `random_fill_nans` mutates the selected source array and uses uncontrolled random filling. | Source remains unchanged; observed mask is preserved; any stochastic transform is seeded, train-only, and recorded. |
| B-005 | `AnnDataDataset` eagerly tensorizes the complete selected AnnData. | Partition fixture proves bounded row/layer/position reads under a memory limit. |
| B-006 | `split_dataset` randomly splits molecule rows with no biological grouping. | Composite `(experiment_uid, Sample)` disjointness and leakage refusal tests. |
| B-007 | Categorical labels become transient pandas category codes. | Explicit label mapping round-trip and inference compatibility tests. |
| B-008 | Repeated `setup()` calls reconstruct data and can change random NaN fills; sklearn fit/evaluate invoke setup repeatedly. | Dataset/split identity and transformed values remain locked across fit/evaluate/apply. |
| B-009 | Validation/test balancing is available inside wrappers and analysis evaluation without a durable sensitivity-analysis identity. | Natural-prevalence primary metrics; any alternate prevalence is separately named and persisted. |
| B-010 | `DANNTransformerClassifier` passes positional constructor arguments incompatibly through `TransformerClassifier`. | Constructor/forward test before any DANN model can be registered. |
| B-011 | Prototype inputs are inconsistent: flat `(B,F)`, CNN `(B,1,L)`, analysis CNN `(B,C,L)`, and Transformer `(B,S,D)`, with no compatibility schema. | Every model validates an ordered modality-aware input schema before execution. |
| B-012 | Checkpoints/results lack dataset, split, label, modality, channel-role, code, and dependency identity. | Round-trip immutable artifact test covering the complete manifest linkage. |

## Saved-artifact compatibility

| Artifact | Current format | Compatibility action |
|---|---|---|
| Analysis CNN result | In-memory `TrainedCNNModel` and `state_dict`; caller-defined persistence only | No package file format to preserve. Provide an explicit importer only if downstream artifacts are supplied. |
| Lightning checkpoint | Framework checkpoint in caller-selected directory | Treat as legacy/trusted resume input, not the canonical portable model; export canonical plain state/config when feasible. |
| AnnData split CSV/`obs` column | Mutable row names plus `train`/`val`/`test` strings | Read-only migration adapter may be offered; canonical output is a versioned split manifest with stable molecule/group IDs. |
| sklearn object | No package persistence | Canonical new artifacts use reviewed `.skops`; unsafe pickle/joblib remains explicit legacy input only. |
| AnnData prediction/attribution columns | Mutable `obs`/`obsm` keys | May be generated as a convenience view from immutable prediction/explanation artifacts, never treated as authority. |

## Current source anchors

These line anchors make the migration tables auditable against `b8b5a90`:

- `src/smftools/__init__.py:70` — public `smftools.ml` lazy alias.
- `src/smftools/analysis/compute/ml_cnn.py:20` — device/config/model/input/train/predict/IG
  surface; principal public anchors are lines 29, 133, 197, 283, 395, 491, and 521.
- `src/smftools/analysis/compute/ml_metrics.py:33` — sklearn builders; metrics begin at line
  172.
- `src/smftools/analysis/compute/ml_splits.py:16` — disjointness, summary, and leave-one-group-out
  functions.
- `src/smftools/analysis/compute/ml_explanations.py:15` — NB, RF SHAP, and XGBoost contribution
  functions.
- `src/smftools/analysis/plot/ml.py:20` — ML plot functions; the metric bar plot begins at line
  581.
- `src/smftools/machine_learning/data/anndata_data_module.py:25` — eager dataset; split function
  line 87; data module line 140; loader factory line 282.
- `src/smftools/machine_learning/data/preprocessing.py:6` — mutating random NaN fill.
- `src/smftools/machine_learning/models/base.py:13` — attribution-coupled base model.
- `src/smftools/machine_learning/models/{cnn.py:13,mlp.py:10,rnn.py:10,transformer.py:15}` —
  prototype neural families.
- `src/smftools/machine_learning/models/lightning_base.py:20` and
  `sklearn_models.py:18` — backend wrappers.
- `src/smftools/machine_learning/training/train_lightning_model.py:19` and
  `train_sklearn_model.py:7` — trainer entry points.
- `src/smftools/machine_learning/inference/{inference_utils.py:6,sklearn_inference.py:9,lightning_inference.py:13,sliding_window_inference.py:9}`
  — mutable AnnData inference surface.
- `src/smftools/machine_learning/evaluation/{evaluators.py:19,eval_utils.py:6}` — wrapper-oriented
  evaluation.

## External and uncommitted candidates

The separate main worktree is at the same commit but has modified
`analysis/compute/ml_cnn.py` and `analysis/plot/ml.py` plus untracked differential-abundance
modules. The audits describe Transformer, CNN-Transformer hybrid, GradientSHAP, and additional plot
behavior in those uncommitted files.

Classification: **evaluate after the committed inventory, do not preserve implicitly**. Before any
candidate is migrated, obtain its tests, actual project consumers, artifact examples, input/channel
semantics, and owner approval. Differential abundance is not model-training infrastructure and
should follow the analysis contract independently.

## ML-002 regression baseline handoff

ML-002 should first characterize B-001 through B-012 with tiny in-memory/partition fixtures. Tests
must describe current defects without blessing unsafe behavior as the new contract. The highest
priority tests are split membership, zero-worker val/test loaders, explicit labels, non-mutating
missingness handling, modality-aware channel compatibility, and artifact round trips.
