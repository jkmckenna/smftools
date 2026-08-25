# Second opinion on the smftools ML infrastructure audit

> **Repository state reviewed:** `15bcc01` — inferred: last commit on `main` on the stated date 2026-07-30.
> **319 commits on `main` since.** An audit describes the code at a moment; it
> goes stale rather than completing. Re-verify any specific claim before relying on it.

Date: 2026-07-30

## Scope and evidence

This report validates `dev/ml_infrastructure_audit.md` against:

- the current `~/git/smftools` main worktree at `4e1b1e5`, including its documented uncommitted
  changes to `analysis/compute/ml_cnn.py` and `analysis/plot/ml.py`;
- the checked-in `smftools.machine_learning`, `smftools.analysis`, partition-store, HMM-artifact,
  latent-artifact, and project-embedding implementations;
- the tests currently covering those areas; and
- primary documentation for PyTorch, Lightning, Hydra, W&B, MLflow, scikit-learn, BERT-style
  pretraining, model cards, and dataset documentation.

The downstream `Nkg2a_DAFseq_merged` project described by the first audit was not present in the
available filesystem. Claims about its scripts therefore could be checked only where the first
audit quoted or summarized them, not by a second independent read of that repository.

This is an architectural assessment only. No package code was changed or executed as a training
workflow.

## Executive conclusion

The first audit is directionally right about the central problem: smftools has multiple ML
implementations that evolved independently, and it needs one explicit ownership model for reusable
architectures, data adapters, training runs, and artifacts.

Its most important overstatement is calling `smftools.machine_learning` a "complete, working"
framework. It is better described as an earlier, broad prototype. It has good ideas—plain
`nn.Module` models, a Lightning wrapper, an AnnData data module, sklearn parity, and attribution
helpers—but its data splitting and loader paths contain correctness problems, it cannot consume the
partitioned stores without eagerly materializing all data, its missing-data behavior is not
reproducible, and its tests only verify imports. It should not be selected as the production
foundation without redesign and behavior tests.

The two package locations can have a legitimate distinction, but not the distinction they have
today:

- `smftools.analysis` should own pure downstream calculations and visualizations: split
  diagnostics, metric computation, statistical comparisons, and plots. It should not grow a
  second model zoo or a second training engine.
- `smftools.machine_learning` should own reusable model families, input schemas/transforms,
  partition-aware datasets, task heads, training adapters, inference, and ML artifacts.
- `smftools.hmm` should remain separate. Its models are part of a specific production pipeline
  stage with a different statistical and lifecycle contract. Reuse its artifact-writing primitives,
  not its entire class hierarchy or model key unchanged.

The recommended direction is therefore to converge reusable classifier code under
`machine_learning`, keep its models as ordinary PyTorch `nn.Module`s, and make Lightning an optional
training adapter rather than the model API. Use configurable model families plus named, immutable
architecture recipes; neither fully hard-coded architectures nor an unconstrained "configure every
layer" graph builder is the right default.

The partitioned stores are suitable for ML, but the current `AnnDataModule` is not. A new
partition-aware dataset must plan train/validation/test membership from the spine/catalog first,
then stream bounded rows from Zarr. Splits must normally be group-disjoint at the biological sample,
experiment, animal, or run level—not random molecule rows.

Classical models should use the same dataset, split, prediction, metric, and artifact contracts.
Bernoulli Naive Bayes, logistic regression, and random forests are useful first-class baselines;
Torch and sklearn should have separate adapters behind a small predictor protocol rather than a
shared inheritance tree. Interpretability should likewise be capability-based: direct parameters
for simple models, TreeSHAP for supported trees, Captum gradient methods for differentiable neural
models, and held-out permutation importance as the broad fallback.

Finally, the package should establish a local, tracker-neutral run and artifact format before
adopting W&B. W&B or MLflow can then mirror that record and provide comparison UI. A hosted tracker
must not become the only record of which data, split, configuration, and code produced a model.
Experiment-local runs should live below `ExperimentConfig.output_directory`; cross-experiment runs
should live below the project's `project_outputs/`. A separate versioned ML plan should declare
models, data, labels, group-aware splits, balancing, and train/apply/explain/plot jobs.

## Validation of the first audit

| Audit claim or recommendation | Verdict | Second-opinion finding |
|---|---|---|
| There are three independent live ML surfaces | **Substantially agree** | `analysis` classifiers, `machine_learning`, and HMM fitting are independent. HMM is not a duplicate classifier framework, though; it is a task-specific pipeline subsystem whose artifact utilities happen to be reusable. |
| `analysis.compute.ml_*` is the active project-facing code | **Agree for the audited tree** | The July 2026 changes are in `analysis/compute/ml_cnn.py`; `machine_learning` last received substantive organization work in 2025 and formatting/optional-import changes in January 2026. |
| `smftools.machine_learning` is a complete, working framework | **Disagree** | It is broad but not production-ready. Specific correctness and scalability gaps are listed below. Import-only smoke tests do not validate data isolation, training, checkpoint restoration, inference, or metrics. |
| Nothing outside `machine_learning` uses it | **Agree** | Package references outside the subtree are the lazy `smftools.ml` export and import smoke tests. No production CLI or analysis path consumes it. |
| The two locations duplicate training, models, metrics, and attribution | **Agree** | The duplication is real. The resolution should be one ML ownership boundary, not indefinite synchronization of both stacks. |
| HMM artifacts are the best package precedent | **Agree with an important qualification** | Atomic publication, checksums, portable relative records, immutable IDs, and conflict detection are strong. The HMM model ID does not include the separately stored training-selection digest, so its key is not sufficient unchanged for supervised training provenance. |
| Generalize `HMMModelKey` as the first/highest-leverage change | **Qualify** | Reuse lower-level artifact primitives and design principles. Define a classifier/run schema whose identity includes the dataset snapshot, split, input schema, architecture, training configuration, seed policy, and code revision. Do not simply rename `HMMModelKey`. |
| Use an HMM-style decorator registry for classifiers | **Mostly disagree** | A registry is useful, but decorators add import-order and hidden-registration behavior without a demonstrated plugin requirement. For three to several built-in families, one explicit `model_type -> ModelSpec(config_type, builder, schema_version)` mapping is clearer. Add plugin registration only if third-party models become a real use case. |
| Promote generic transformer code into `analysis.compute` | **Disagree on destination** | Generic trainable encoders and classifiers belong in `machine_learning.models`. Putting them in `analysis.compute` would deepen the boundary violation identified by the audit. Pure tokenization or metric functions may remain array-in/result-out utilities. |
| Adopt Lightning if `machine_learning` becomes the future | **Qualify** | Lightning can be useful, but models should remain framework-neutral `nn.Module`s. Adopt or retain a Lightning adapter only when its trainer features pay for the dependency and abstraction. |
| W&B is the highest-return immediate addition | **Disagree as a first step** | A local run manifest and artifact contract are higher priority. W&B is useful as an optional backend after that contract exists. Otherwise it creates a polished remote index over artifacts that remain locally ambiguous or incomplete. |
| Skip Hydra for now | **Agree** | smftools already has a package configuration hierarchy. Hydra would be justified for a dedicated training application with compositional sweeps, not as a second package-wide config system. |
| `lightning` dependency versus `pytorch_lightning` imports is a naming mismatch | **Qualify** | Current Lightning documentation installs `lightning` and uses `lightning`/`lightning.pytorch`. The legacy `pytorch-lightning` distribution and `pytorch_lightning` namespace still exist. This is modernization/consistency debt, not by itself proof that installation is broken. |
| `captum` is eagerly imported by `analysis.compute.ml_cnn` | **Disagree** | In the audited current file, Torch is imported at module load, while Captum is imported inside the attribution function. Torch is also currently a core dependency in `pyproject.toml`. |
| `joblib` is unused | **Only true for model persistence** | It is not used to persist package models, but it is used elsewhere in smftools for parallel computation. Its package-level dependency cannot be judged unused from model-save searches alone. |
| Add a single append-only `training_runs.jsonl` | **Qualify** | A JSONL index is a reasonable prototype but a weak canonical store under concurrency and schema evolution. Immutable per-run manifests should be authoritative; a Parquet table, SQLite database, MLflow store, or W&B project can be a rebuildable/queryable index. |

## Why both `analysis` and `machine_learning` exist, and what their point should be

The current split reflects history more than a stable architecture.

`smftools.analysis` has an explicit design contract: compute modules are pure array/table-in,
result-out functions; plot modules render a supplied result to an explicit path; filters return
boolean masks; and project-specific I/O stays outside the package. The active classifier work was
placed there because project scripts needed small, composable functions operating on matrices and
because that code did not depend on the older ML package.

`smftools.machine_learning` was built as a broader end-to-end framework. It includes model
architectures, Lightning and sklearn wrappers, AnnData loading, training, evaluation, inference,
sliding windows, and attribution. That is a coherent intended responsibility, but it was never
connected to production callers and has drifted.

The useful distinction is therefore **analysis of model results versus construction and execution
of models**, not "simple ML versus advanced ML" and not "plain PyTorch versus Lightning."

Recommended ownership:

```text
smftools/
├── analysis/
│   ├── compute/
│   │   ├── ml_metrics.py        # pure y/score -> metrics
│   │   ├── ml_splits.py         # pure metadata -> split plans/diagnostics
│   │   └── ...                  # non-ML statistical analysis
│   └── plot/
│       └── ml.py                # curves/comparisons -> figures
├── machine_learning/
│   ├── models/
│   │   ├── encoders.py          # CNN/transformer/hybrid representation models
│   │   ├── heads.py             # reconstruction/classification/domain heads
│   │   ├── configs.py           # validated architecture configs and named recipes
│   │   └── registry.py          # explicit built-in model specifications
│   ├── data/
│   │   ├── input_schema.py      # channels, tokens, coordinates, missingness semantics
│   │   ├── split_manifest.py    # persisted group-aware membership
│   │   ├── partition_dataset.py # bounded Zarr reads
│   │   └── transforms.py        # train-fitted transforms and augmentations
│   ├── tasks/
│   │   ├── pretraining.py       # corruption + reconstruction loss
│   │   └── classification.py    # supervised heads/loss
│   ├── training/
│   │   ├── pytorch_engine.py    # small reference engine
│   │   └── lightning_adapter.py # optional orchestration adapter
│   ├── inference/
│   ├── evaluation/
│   └── artifacts/
│       ├── manifests.py
│       └── store.py
└── hmm/                         # task-specific production pipeline remains separate
```

This is a conceptual target, not a recommendation to perform a broad move in one PR. The safe
migration would establish tested contracts first, move one vertical path, and retain compatibility
imports while downstream projects migrate.

### What may reasonably stay in `analysis`

- metric functions that know nothing about a model object;
- group split construction and leakage validation from metadata tables;
- confidence intervals, bootstrap comparisons, calibration calculations, and statistical tests;
- plotting training histories, ROC/PR curves, calibration, attribution summaries, and model
  comparisons from explicit result tables.

### What should not remain duplicated in `analysis`

- trainable architecture classes;
- optimizer/epoch/early-stopping loops;
- checkpoint or pretrained-weight loading;
- partition-aware training datasets;
- task heads and pretraining objectives;
- framework-specific trainers.

## `smftools.machine_learning` is a prototype, not a ready framework

The first audit correctly noticed that the subtree is feature-rich. It did not validate whether its
paths preserve train/validation/test isolation or work under their default arguments.

### Concrete local findings

1. **Default validation and test loaders return training data.**

   In `data/anndata_data_module.py`, the `num_workers` branches correctly use `self.val_set` and
   `self.test_set`, but the common false/`None` branches construct both loaders from
   `self.train_set`. Early stopping and reported test performance can therefore be calculated on
   training rows.

2. **The split is random by molecule row, not by biological or technical group.**

   `split_dataset` shuffles `np.arange(total_len)`. Molecules from the same sample, library,
   barcode, run, animal, or experiment can appear in all three sets. In SMF data those rows are
   correlated, so row-random performance can be optimistic. The newer
   `analysis.compute.ml_splits` is safer because it explicitly holds out groups, but it is not
   connected to the old data module.

3. **The dataset eagerly copies the complete selected matrix into RAM.**

   `AnnDataDataset.__init__` reads `adata.X`, a layer, or `obsm`, fills missing values, and converts
   the whole result to a Torch tensor. It does not use `partition_query.read_zarr_subset`,
   `partition_read.materialize`, catalog pruning, lazy slicing, or a memory budget.

4. **Missing values are filled randomly with no persisted seed or mask.**

   `random_fill_nans` mutates its input-like array and replaces NaNs with global NumPy random
   draws. The transformation is neither fitted nor recorded, and missingness is hidden from the
   model. By contrast, the active `analysis.compute.ml_cnn.build_cnn_input` preserves an
   observed-data channel and deterministically zero-fills the signal; that is a better starting
   semantic.

5. **The non-Lightning loader call swaps arguments.**

   The positional call to `split_dataset` passes `split_save_path` where `load_existing_split` is
   declared and `load_existing_split` where `split_save_path` is declared.

6. **Transformer masking is not implemented as an attention padding mask.**

   `BaseTransformer.encode` multiplies token embeddings by a mask and then calls
   `nn.TransformerEncoder(x)` without `src_key_padding_mask`. Zeroed inputs can still produce
   non-zero projected keys/values because of biases and subsequent layers. PyTorch exposes
   `src_key_padding_mask` specifically to exclude padded/unobserved keys.

7. **The classifier path does not accept a mask.**

   `TransformerClassifier.forward` calls `encode(x)` without a mask, so the available mask
   parameter is only used by the masked pretrainer. The old AnnData dataset also does not return a
   mask in its batch.

8. **Some advertised model classes cannot be instantiated as written.**

   `DANNTransformerClassifier.__init__` passes three positional arguments to a parent initializer
   that accepts two positional arguments after `self`. This is a direct indication that code
   presence is being mistaken for a supported capability.

9. **Training does not restore the best checkpoint before testing.**

   The Lightning helper configures a `ModelCheckpoint` only when a path is supplied and then calls
   `trainer.test(model, ...)` on the current in-memory state. It does not request the best
   checkpoint with `ckpt_path="best"`. With no checkpoint path, early stopping stops training but
   no best state is restored.

10. **Tests cover importability, not behavior.**

    The `tests/smoke/machine_learning` files parametrically import modules. There are no behavioral
    tests for split disjointness, loader identity, missingness, model shape contracts, a training
    step, checkpoint round-tripping, inference alignment, or partition streaming.

These findings do not imply that every class should be discarded. They mean the subtree should be
mined for interfaces and useful components, then made correct behind tests before being declared
the canonical API.

## Flexible model families versus hard-coded architectures

Use a hybrid design:

1. **Flexible family classes** expose a small set of structural parameters that genuinely define a
   family: channel widths, number of blocks/layers, dilation schedule, model dimension, attention
   heads, feed-forward dimension, dropout, pooling mode, and input-channel schema.
2. **Named recipes** freeze reviewed combinations such as `cnn_small_v1`,
   `masked_transformer_base_v1`, or `hybrid_activity_v1`.
3. **Artifacts store the fully resolved configuration**, recipe name, and architecture schema
   version. A recipe name alone is not reproducible because its defaults may change.
4. **Validation rejects nonsensical combinations**, for example `d_model % n_heads != 0`, unequal
   CNN channel/dilation lengths, an input schema incompatible with a checkpoint, or a coordinate
   vocabulary longer than a learned positional embedding.

This provides controlled experimentation without turning smftools into a general neural-network
graph builder.

### Why not hard-code every instance

- Pretraining and task fine-tuning often need different model capacity.
- SMF loci and feature-channel schemas can differ.
- Ablations need controlled changes in depth, width, pooling, and masks.
- Exact configurations can be serialized and reconstructed from `state_dict` artifacts.

### Why not make everything configurable

- Large configuration surfaces create untested combinations and unclear support promises.
- Arbitrary per-layer lists complicate compatibility and artifact loading.
- A user can accidentally compare architectures while believing only one parameter changed.
- The package would inherit responsibility for validating a general model-construction language.

The active `CNNConfig`, `TransformerConfig`, and `HybridConfig` are closer to the right pattern than
the constructor-only classes in the old ML package. They should eventually become validated,
versioned configs in the canonical ML namespace, with named presets layered on top.

### Registry recommendation

For built-in models, prefer one explicit table:

```text
model_type -> {
    config_type,
    builder,
    architecture_schema_version,
    compatible_input_schema_versions,
}
```

That removes the three parallel dictionaries noted by the first audit while keeping registration
visible and deterministic. A decorator registry is warranted later only if smftools explicitly
supports third-party architecture plugins.

## Classical and other non-neural models

Yes. scikit-learn models should be first-class ML backends, not treated as temporary baselines.
`scikit-learn>=1.2` is already a core smftools dependency, while the package's `ml-extended` extra
contains Captum, SHAP, Lightning, Hydra, and W&B. That dependency split is compatible with making
ordinary sklearn training available without installing the extended neural/interpretability stack.

Useful initial families are:

- **Bernoulli Naive Bayes** for binary accessibility/base-state features. It is fast, produces a
  valuable low-capacity baseline, and its per-class feature log probabilities are directly
  inspectable.
- **Logistic regression or an SGD linear classifier** for calibrated or rankable linear baselines,
  coefficient-based interpretation, class weighting, and sparse inputs.
- **Random forests and histogram/gradient-boosted trees** for nonlinear interactions with little
  representation engineering. XGBoost is already in the `analysis` optional extra and should be
  exposed through the same model contract if retained.
- **Linear or kernel SVMs** when sample counts and feature dimensions make them practical.
- **Nearest-neighbor methods** for small reference sets and diagnostic similarity analyses, though
  they are less attractive as promoted models because inference stores or searches the training
  reference set.
- **PCA/NMF, mixture models, and clustering** for unsupervised representation or discovery tasks.
  These need task-specific evaluation contracts rather than being forced through a classifier
  interface.
- **HMMs** where sequential latent-state modeling is the scientific task. Existing smftools HMMs
  should remain in their task-specific subsystem, even though they are non-neural.

The package should not make Torch and sklearn estimators inherit from the same base class. Instead,
define a small predictor protocol and use backend adapters:

```text
fit or load
predict / predict_proba or decision_function
input_schema
class_schema
artifact_serializer
capabilities: incremental_fit, sample_weight, native_explanation, probability_output
```

`TorchPredictor` and `SklearnPredictor` can satisfy that protocol through composition. Metrics,
prediction tables, run manifests, split manifests, and plotting then remain backend-neutral.

### One data contract, backend-specific materialization

The logical batch should identify `X`, `y`, observed/design masks, feature metadata, molecule IDs,
group IDs, and split provenance. Its physical representation can differ:

- Torch receives tensors plus explicit masks.
- sklearn receives dense NumPy arrays, sparse matrices, or tables after a fitted preprocessing
  pipeline.
- estimators with `partial_fit`, including Bernoulli, Multinomial, and Gaussian Naive Bayes, can
  consume bounded partition-store batches incrementally.
- random forests and many other estimators normally require the complete selected training matrix
  in memory. The partition store can still perform predicate pushdown and bounded reads, but the
  adapter must estimate/materialize the resulting matrix within a declared memory budget.

Classical estimators do not generally accept a per-position mask at `predict` time. Missingness
therefore becomes part of the feature schema: for example, fit a train-only imputer and optionally
append explicit observed/design-mask indicator columns in an sklearn `Pipeline`. The same fitted
pipeline must be stored with the estimator and reused unchanged for validation, test, and
inference. This is exactly the sort of leakage that sklearn pipelines are intended to prevent.

Tree models do not make missingness semantics disappear. Even if an implementation accepts NaNs,
structurally unmeasured positions and genuine biological values still need distinguishable input
semantics. A model must never learn from a validation/test-fitted imputer or feature selector.

### Persistence for sklearn models

Store the resolved preprocessing pipeline, estimator parameters, learned class order, feature
schema, training dependency versions, and held-out behavior tests. `joblib` is convenient but is
pickle-based: loading an untrusted artifact can execute code, and sklearn does not support loading
models across different sklearn versions. A safer `skops.io` representation or an ONNX inference
export may be appropriate where supported, but neither eliminates the need for the native
manifest and a tested input/output contract. The artifact format should record which loader is
required and whether the source must be trusted.

## Interpretability organization

Interpretability should be a separate subsystem composed around a fitted predictor, not methods on
every model class and not another copy under `analysis.compute`. Model-specific computation belongs
under `machine_learning.interpretability`; pure summaries, comparisons, and plotting of completed
explanations can remain in `analysis`.

A conceptual layout is:

```text
machine_learning/interpretability/
├── request.py               # target, cohort, baseline, mask policy, method parameters
├── result.py                # backend-neutral AttributionResult
├── baselines.py             # reproducible training-derived background/reference sets
├── neural/
│   ├── input_gradients.py   # saliency and gradient × input
│   ├── captum_methods.py    # Integrated Gradients, DeepLift, GradientSHAP
│   ├── gradcam.py           # selected convolutional layers only
│   └── attention.py         # named attention rollout / attention × gradient methods
├── tree/
│   └── tree_shap.py
├── linear/
│   └── coefficients.py
├── probabilistic/
│   └── naive_bayes.py
└── model_agnostic/
    ├── permutation.py
    └── kernel_shap.py
```

The method must match the model and the scientific question:

| Model or question | Preferred methods | Important constraint |
|---|---|---|
| Bernoulli/Multinomial Naive Bayes | class log-probability or log-odds differences; permutation importance | Direct parameters are often clearer than approximate SHAP values. |
| Logistic/linear model | standardized coefficients, odds ratios, permutation importance, optionally LinearSHAP | Coefficients are only comparable after accounting for feature scaling and correlation. |
| Random forest/boosted trees | TreeSHAP and held-out permutation importance | Impurity importance can favor high-cardinality features; SHAP background/dependence assumptions must be recorded. |
| Any fitted predictor | held-out permutation importance | It explains score degradation for a dataset/cohort, not a causal effect or necessarily single-molecule behavior. |
| Differentiable neural model | saliency, gradient × input, Integrated Gradients, DeepLift, GradientSHAP | Target output, baseline distribution, input transforms, masks, and convergence/error settings are part of the result. |
| CNN | Integrated Gradients plus LayerGradCam/Guided GradCAM at a declared convolutional layer | GradCAM is layer- and architecture-specific and generally coarser than input attribution. |
| Transformer | input/embedding attribution plus precisely named attention rollout or attention × gradient analysis | Attention weights alone are not a general explanation; avoid the ambiguous label `AttentionCAM` unless a specific algorithm is defined. |
| Expensive black-box model | KernelSHAP only for small, justified cohorts; otherwise permutation/ablation | KernelSHAP cost grows quickly and its feature masker/background encodes strong assumptions. |

Captum documents Integrated Gradients as attribution along a path from a baseline, GradientSHAP as
an expected-gradient approximation with independence/linearity assumptions, DeepLift with limited
nonlinearity-rule support, and LayerGradCam for models with convolutional layers. SHAP documents
TreeExplainer specifically for tree ensembles and KernelExplainer as a weighted-regression,
model-agnostic approximation. These are not interchangeable names for "feature importance."
Normalize informal names such as `GradSHAP` to the canonical implementation name
`GradientSHAP` in configurations and manifests.

### A common explanation artifact

Every method should produce an `AttributionResult` carrying:

- attribution values and their observation/feature axes;
- method name, implementation/version, and model/backend capabilities used;
- model ID, run ID, dataset snapshot ID, split/cohort ID, and stable observation IDs;
- explained output (class index and label, logit/probability/loss, or intermediate target);
- baseline/background dataset ID and sampling rule;
- target layer or attention algorithm where relevant;
- feature/input schema and observed/design/padding-mask policy;
- method parameters, random seed, convergence delta/error estimate where available; and
- any aggregation from channels/positions to genomic regions.

Background examples and baselines must be selected only from the training partition or from an
independent reference population. They must preserve the meaning of sequence/context channels and
structural design masks; an all-zero baseline is not scientifically neutral merely because an API
accepts it. Test observations may be explained, but the test set must not choose the baseline,
method, layer, hyperparameters, or reporting threshold.

Explanation arrays should be immutable run artifacts:

```text
runs/<run_id>/explanations/<explanation_id>/
├── manifest.json
├── values.zarr
├── feature_summary.parquet
├── group_summary.parquet
└── plots/
```

Keep raw per-molecule explanations in Zarr when large, and make tables/plots reproducible
derivatives. Aggregate separately by class, sample, reference, strand, and position so a large
class does not silently dominate a global mean. Report stability across seeds/folds and, where
feasible, explanation sensitivity or infidelity; do not present one attractive heatmap as proof of
a biological mechanism.

An appropriate support order is:

1. Naive Bayes log-odds and linear coefficients.
2. Held-out permutation importance for every predictor.
3. TreeSHAP for supported forest/boosting models.
4. Integrated Gradients, GradientSHAP, and DeepLift for validated neural inputs.
5. LayerGradCam for CNNs and explicitly defined attention analyses for transformers.
6. KernelSHAP only for bounded cases where a specialized explainer is unavailable.

## Plain PyTorch versus Lightning

This is an orchestration decision, not a model-class decision. A reusable smftools model should be a
plain `torch.nn.Module` either way. Lightning itself describes `LightningModule` as an organization
of training, validation, test, prediction, and optimizer logic around PyTorch; its Trainer handles
devices, loops, callbacks, distributed samplers, precision, and checkpoint state.

### Use plain PyTorch when

- a training path is short and stable;
- it runs on one CPU/GPU/MPS device;
- explicit control of masking, multiple losses, or unusual batch semantics is more valuable than
  trainer abstraction;
- package dependency weight and API stability matter;
- the main need is a small reference loop for tests and reproducible examples; or
- sklearn-like fit/predict ergonomics are sufficient.

The current three-architecture loop in `analysis.compute.ml_cnn` is within this range, although it
still needs artifact integration, group-aware split provenance, bounded validation batching, and
better reproducibility metadata.

### Use a Lightning adapter when several of these are real requirements

- reliable interruption and full-state resume;
- automatic mixed precision;
- gradient accumulation;
- multiple GPUs/nodes or pluggable distributed strategies;
- standardized callbacks for early stopping/checkpointing;
- consistent metric/logger integration;
- repeated pretraining and fine-tuning jobs where maintaining custom loop infrastructure has become
  material work; or
- a shared training application with multiple task modules and data modules.

Lightning checkpoints can contain model, optimizer, scheduler, callback, loop, hyperparameter, and
data-module state. That is valuable for resumable pretraining, but it does not replace smftools'
portable inference artifact. The latter should remain a plain model `state_dict` plus explicit
configuration and input schema. PyTorch recommends `state_dict` saving for flexibility, and
Lightning checkpoints can be converted or have their underlying module weights extracted.

### Recommended package relationship

```text
plain nn.Module + task/loss object + DataLoader
                 │
                 ├── small plain-PyTorch engine
                 └── optional LightningSystem/LightningDataModule adapter
```

Do not make every model inherit from a large classifier base class containing attribution,
AnnData mutation, metrics, and device detection. Composition keeps models usable in plain PyTorch,
Lightning, inference services, and attribution tools.

### A practical adoption gate

Do not migrate merely to remove a 50-line epoch loop. Adopt Lightning after a tested
partition-aware dataset and artifact contract exist, and when at least one production workflow
needs resume, precision management, or multi-device training. If Lightning is retained, modernize
new code around the currently documented `lightning`/`lightning.pytorch` namespace and pin/test a
supported minor range; Lightning's versioning policy allows some backward-incompatible minor
changes with deprecations.

## When Hydra is useful

Hydra is valuable when a dedicated training entry point needs:

- composable config groups such as dataset × input representation × encoder × task head × trainer;
- command-line overrides over a fully resolved hierarchical configuration;
- systematic multirun sweeps;
- launcher/sweeper plugins for local parallel jobs or cluster execution; and
- a separate output directory and resolved config for every launched job.

Hydra's official multirun behavior expands combinations of overridden parameters and can use
launcher/sweeper plugins. That is useful once model experiments are intentionally config-driven.

It is not currently justified as a package-wide configuration system because:

- smftools already resolves `default.yaml -> modality YAML -> user_defined_config.csv`;
- model construction is still split between package and project scripts;
- a second composition system would create unclear precedence and duplicated concepts; and
- neither a stable ML config schema nor a canonical training CLI exists yet.

Recommended order:

1. define ordinary typed/versioned ML configs and serialize their resolved values;
2. establish one training application and artifact contract;
3. add simple Click options or config-file input;
4. adopt Hydra only if composition and multirun management are demonstrably the bottleneck.

Hydra should remain an application-layer choice. Core model and data APIs should accept normal
Python config objects and should not require Hydra/OmegaConf objects.

## When W&B or MLflow is useful

An experiment tracker is useful when the team needs to:

- compare many runs and folds interactively;
- inspect training curves while a job is running;
- search by architecture, dataset, split, task, or metric;
- associate input dataset versions with output model versions;
- share dashboards and artifacts across machines; or
- coordinate larger sweeps.

W&B documents metrics logging, artifact versions, and input-dataset/output-model lineage. MLflow
Tracking records runs, parameters, code versions, metrics, datasets, and artifacts, and can start
with a local directory or SQLite-backed setup before moving to a shared server.

### Why tracking should be optional

- smftools is an installable bioinformatics package, not a single hosted training service.
- Projects may run offline or handle data that must not be uploaded.
- A user must be able to reconstruct an artifact without access to a SaaS account.
- Tracker integrations and credentials should not affect scientific behavior.
- Run IDs and manifests need stable package semantics independent of a vendor.

### Recommended policy

1. Every run writes the same complete local manifest and files.
2. A tracker adapter may mirror parameters, scalar metrics, curves, safe aggregate tables, and
   artifacts.
3. Tracker failure must not corrupt or invalidate a completed local run.
4. Raw molecule identifiers or sensitive sample metadata are not uploaded by default.
5. The local manifest records the optional external tracker name, project, and run ID.

W&B is reasonable for a team already using its hosted UI. MLflow is attractive for a local-first or
self-hosted workflow. A simple local CSV/TensorBoard logger can be enough initially. The choice can
be deferred if the local run schema is tracker-neutral.

## Using partitioned data stores for train/validation/test

Yes—the partitioned stores are a strong fit for ML, but through a purpose-built dataset rather than
the current eager `AnnDataDataset`.

Existing capabilities already provide most storage primitives:

- a spine with molecule-level metadata and stable row identity;
- Parquet indexes with predicate pushdown for reference, sample, barcode, read ID, molecule UID, and
  genomic interval;
- cataloged Zarr group paths and row offsets;
- `read_zarr_subset` for bounded row, position, and layer projection;
- lazy slicing before `to_memory` where optional dependencies support it; and
- a query memory budget.

### Recommended data flow

```text
spine/catalog metadata
        │
        ├── resolve eligible molecules and labels
        ├── assign group-disjoint train/val/test membership
        ├── persist immutable split manifest
        └── map each member to partition path + row + coordinate projection
                                      │
                                      ▼
                         partition-aware Dataset
                                      │
                         bounded Zarr row batches
                                      │
                    transform -> tensor + masks + label
                                      │
                                      ▼
                                  DataLoader
```

### Dataset style

PyTorch supports both map-style and iterable-style datasets:

- A **map-style dataset** is suitable when the split manifest maps every logical example to a
  partition and row. Implementing batched fetch (`__getitems__`) can group requested rows by
  partition and avoid one Zarr open/read per molecule.
- An **IterableDataset** is suitable when sequential partition/chunk reads are substantially
  cheaper than random access. Its worker copies must be explicitly sharded with
  `get_worker_info()` to prevent duplicated examples.

For the current stores, a map-style manifest plus partition-grouped batch fetch is likely the best
first implementation because it preserves deterministic membership and normal samplers. An
iterable dataset becomes attractive for very large pretraining corpora where streaming throughput
matters more than random row access.

### Split at the right unit

The split unit must match the intended generalization claim:

| Intended claim | Minimum holdout group |
|---|---|
| Generalize to new molecules from the same technical library | molecule rows may be split, but this is the weakest and usually least interesting claim |
| Generalize to a new library/barcode from the same sample | library or barcode |
| Generalize to a new biological sample | sample/replicate |
| Generalize across animals/donors | animal/donor |
| Generalize across sequencing runs/batches | run |
| Generalize across experiments/datasets | experiment |

For most label-classification claims, molecules from one biological sample should not be divided
across train, validation, and test. Use a stored group column and a group-aware strategy such as
leave-one-group-out, `GroupKFold`, or `StratifiedGroupKFold` where its constraints are satisfiable.
Scikit-learn defines `StratifiedGroupKFold` specifically to preserve class proportions as much as
possible while keeping groups non-overlapping.

Nested choices matter:

- The **test set** is locked and used for final reporting, not architecture selection.
- The **validation set** selects architecture, early stopping, thresholds, and calibration.
- Cross-validation folds used for comparison should be children of one parent experiment/run, with
  a separate final model fit recorded afterward.
- Normalization, imputation statistics, vocabulary selection, feature selection, and calibration
  are fitted on training rows only, then applied unchanged to validation/test/inference. This
  avoids leakage.

### What the split manifest should contain

At minimum:

- stable dataset snapshot ID;
- stable split ID and split-schema version;
- molecule UID or a resolvable project-local row reference;
- `train`, `validation`, or `test`;
- grouping fields and held-out group;
- label as resolved at split time;
- partition path/ID and row reference;
- reference/coordinate/input-layer selection;
- split algorithm, seed, and parameters;
- counts and class balance per split and group; and
- a digest over ordered membership.

For a distributable pretrained model, store aggregate counts and membership digests rather than
private identifiers. Within the local smftools project, retain a protected split table with the
actual molecule UIDs so the run can be reproduced. A digest alone proves equality but cannot
reconstruct membership after the fact.

## Mask semantics: training, validation, test, and inference

"Mask" currently refers to several different concepts. They must be named and handled separately.

| Mask | Meaning | Train | Validation/test | Normal inference |
|---|---|---:|---:|---:|
| `split_membership` | Which rows belong to a data split | Select rows | Select rows | Not a model input |
| `observed_mask` | Which assay positions are actually observed versus missing | Yes | Yes | Yes |
| `padding_mask` / attention key-padding mask | Which sequence positions are padding/non-tokens | Yes | Yes | Yes |
| `design_mask` / feature-availability mask | Positions intentionally excluded by the experimental/model design | Yes if unavailable at deployment | Same policy | Same policy |
| `corruption_mask` | Positions hidden to create a self-supervised reconstruction target | Generate during pretraining | Generate only for reconstruction evaluation | No, unless scoring the pretraining task |
| `loss_mask` | Positions/examples contributing to a particular loss | Yes | Yes for the same evaluation objective | No loss at inference |
| `augmentation_mask` | Random feature dropout or span masking for regularization | Train only | No | No |
| `ablation_mask` | Features deliberately removed to study dependence | Optional experiment | Optional experiment | Only an explicitly labeled ablation, not ordinary deployment |

### Non-negotiable rules

1. A model must never infer missingness solely from the filled signal value if that value is also a
   valid observation. Carry `observed_mask` explicitly.
2. Transformer padding/missing positions must be excluded using correct attention-mask semantics,
   not only by multiplying embeddings by zero.
3. If a region will be unavailable in real deployment, train and validate with the same design
   mask. Applying that mask only at inference creates distribution shift.
4. A training-only corruption/augmentation mask should not appear in normal inference.
5. Applying a new mask only at inference is valid for a named sensitivity or ablation analysis, but
   its performance must not be presented as the model's trained operating condition.
6. Split membership is provenance and data selection. It should not be fed as a predictive feature.
7. Masks, fills, and positional channels are part of the input schema and artifact compatibility
   key. Two checkpoints with different mask semantics are not interchangeable.

The active `build_cnn_input` already has the useful idea of separate signal, observed, optional
design, positional, spacing, and condition channels. The canonical ML API should generalize and
version that input schema instead of using the old `random_fill_nans` behavior.

## Pretrained encoders versus fine-tuned label classifiers

Treat these as different artifact kinds with explicit lineage.

### Pretrained encoder artifact

A reusable pretrained artifact contains:

- encoder family and fully resolved architecture config;
- tokenizer/value vocabulary and coordinate-axis schema;
- input-channel and mask schema versions;
- pretraining objective and corruption policy;
- pretraining dataset snapshot and split digest;
- optimizer/schedule/seed/determinism metadata;
- encoder `state_dict`;
- optional pretraining head state for resuming/evaluating reconstruction;
- training/validation reconstruction curves and held-out metrics;
- code and dependency versions; and
- intended uses and limitations.

### Fine-tuned classifier artifact

A fine-tuned child contains:

- immutable parent encoder model ID and checksum;
- task/head type, label vocabulary, and class mapping;
- whether the encoder was frozen, partially unfrozen, or fully fine-tuned;
- layer-wise learning-rate or freeze schedule;
- task dataset snapshot and group-aware split ID;
- threshold/calibration policy selected on validation only;
- final encoder plus head `state_dict` for portable inference;
- classification metrics, curves, and subgroup results; and
- an explicit declaration of which parent components changed.

The lineage should look like:

```text
pretraining run
      └── pretrained encoder model
              ├── fine-tuning run: activity labels
              │       └── activity classifier model
              └── fine-tuning run: cell-type labels
                      └── cell-type classifier model
```

BERT is the standard precedent: a reusable encoder is pretrained with a masked reconstruction
objective and fine-tuned with a task-specific output layer. For smftools, this is useful only if the
unlabeled corpus is meaningfully larger or more diverse than the labeled tasks and transfer is
tested against from-scratch baselines. "Pretrained" should not become a label for every intermediate
checkpoint.

### Model class design

Prefer an encoder/head composition:

```text
encoder.encode(x, masks) -> per-position and/or pooled representation
pretraining head(representation) -> reconstruction logits
classification head(pooled representation) -> class logits
domain head(pooled representation) -> domain logits
```

This avoids duplicating an entire transformer for every task and makes parent-child artifact
lineage natural. Task systems own losses; encoder classes do not need to inherit from a
classifier-shaped base class.

## Locating ML outputs in an experiment or project

The ML model itself should never infer an output directory. Directory selection belongs to the
application/orchestration layer, which should resolve one explicit workspace and inject an artifact
writer or `RunPaths` object into training, application, explanation, and plotting services.

The existing smftools layout gives a natural two-scope rule:

| Invocation scope | Proposed ML root | Appropriate use |
|---|---|---|
| One experiment config | `<ExperimentConfig.output_directory>/ml_outputs/` | Training, application, or evaluation whose eligible data all come from that experiment. |
| One smftools project | `<project_dir>/project_outputs/ml/` | Training or comparison across registered experiments, project sets, or samples. |

`ml_outputs` is a proposed sibling stage directory, not an existing implemented constant. It
follows the current experiment convention in which raw, preprocess, spatial, HMM, latent, variant,
and chimeric outputs are siblings under `ExperimentConfig.output_directory`. The project location
follows the documented rule that materialized and derived cross-experiment results belong under
`project_outputs/`.

The caller should pass exactly one scope locator—an experiment config or a project directory—rather
than letting ML search parent directories, inspect the current working directory, or accept a
different arbitrary output path in every model method. If an eligible dataset spans more than one
experiment, require project scope. A project run reads registered experiment artifacts through
`registry.json` and named sets and must not write results back into those registered experiment
trees.

The resolved context should contain at least:

```text
MLWorkspace
├── scope_kind: experiment | project
├── root: resolved ML output root
├── scope_id
├── experiment_name + experiment config hash + output root
│   OR
│   project registry digest + selected set + registered experiment IDs
└── active ML-plan digest
```

From that context, one path resolver creates all dataset, run, model, explanation, and index paths.
Compute objects receive paths or an artifact-writer interface; they do not concatenate directory
names. Publication should use portable relative references, staging followed by atomic rename, and
immutable IDs, following existing smftools artifact conventions.

A pretrained model may be loaded from a model artifact in a different experiment, project, or
approved shared registry. The predictions and explanations still go into the **active** workspace
as a new run that records the source model ID, URI/path, and checksum. Plotting similarly reads
immutable run/model/explanation IDs and writes reproducible derivatives into the active run or a
dedicated comparison run; it must not rediscover "the latest" checkpoint by filename.

When a useful experiment model becomes project-wide, use an explicit registration/promotion action:
either create a portable project pointer to the checksummed immutable artifact or publish a
project-owned copy with preserved parent lineage. Do not give the same mutable directory two
identities.

## Organizing ML artifacts inside either scope

smftools already has project output roots, portable relative paths, checksummed generations, HMM
model sidecars, source fingerprints, and latent/embedding model dependency metadata. ML should use
the same layout below either resolved root.

Recommended conceptual layout, where `<ml_root>` is one of the two roots above:

```text
<ml_root>/
├── datasets/
│   └── <dataset_snapshot_id>/
│       ├── manifest.json
│       ├── split_<split_id>.parquet
│       ├── summary.parquet
│       └── DATASET_CARD.md
├── runs/
│   └── <run_id>/
│       ├── run_manifest.json
│       ├── resolved_config.json
│       ├── environment.json
│       ├── history.parquet
│       ├── metrics.parquet
│       ├── validation_predictions.parquet
│       ├── test_predictions.parquet
│       ├── plots/
│       │   ├── learning_curves.png
│       │   ├── roc_pr.png
│       │   └── calibration.png
│       ├── explanations/
│       │   └── <explanation_id>/
│       │       ├── manifest.json
│       │       ├── values.zarr
│       │       └── feature_summary.parquet
│       ├── checkpoints/
│       │   ├── last.ckpt
│       │   └── best.ckpt
│       └── logs/
├── models/
│   └── <model_id>/
│       ├── model_manifest.json
│       ├── <backend_artifact>       # e.g. weights.pt, pipeline.skops, or model.onnx
│       ├── MODEL_CARD.md
│       └── optional_safe_aggregate_evaluations/
└── index/
    ├── runs.parquet
    └── models.parquet
```

The exact filenames are less important than the separation of concepts:

- A **dataset snapshot** defines data identity and eligible examples.
- A **split** defines role membership for that snapshot.
- A **run** is one execution, including failed/interrupted runs and fold children.
- A **checkpoint** is resumable training state at a point in a run.
- A **model artifact** is a validated, immutable output intended for reuse/inference.
- An **index** is a query aid rebuildable from authoritative manifests.

The run manifest is authoritative for where outputs went. W&B, MLflow, TensorBoard, a Parquet
index, or a UI may index or mirror them, but must not redefine the filesystem location.

## How users should declare models, data, splits, labels, and actions

Use a separate, versioned, validated **ML plan** (for example `ml_plan.yaml`) for nested ML intent.
Do not place the entire plan in the existing three-column experiment CSV, and do not reuse the
scaffolded `project.yaml`: current smftools documentation explicitly says `project.yaml` is
human-curated and not read by smftools.

The experiment CSV should remain authoritative for experiment acquisition/processing and its
`output_directory`. The project registry and named sets should remain authoritative for
cross-experiment membership. A future CLI can receive `--ml-plan <path>` plus exactly one of
`--experiment-config <path>` or `--project-dir <path>`. If convenient, the experiment CSV may
contain one scalar `ml_plan_path`, but the plan content and precedence should remain separate and
unambiguous.

The plan should define named reusable objects, then jobs that refer to those names:

```yaml
schema_version: 1

scope:
  kind: project
  set: dafseq_training

datasets:
  activity_reads:
    source:
      stage: preprocess
      layer: binary_accessibility
      references: [Nkg2a]
    samples:
      include:
        - exp_01/sample_A
        - exp_01/sample_B
        - exp_02/sample_C
        - exp_02/sample_D
        - exp_03/sample_E
    filters:
      mapping_quality_min: 20
    labels:
      source: obs
      column: activity_status
      classes:
        inactive: 0
        active: 1
      missing: drop
      positive_class: active

  new_activity_reads:
    source:
      stage: preprocess
      layer: binary_accessibility
      references: [Nkg2a]
    samples:
      include: [exp_04/sample_F]

splits:
  sample_holdout:
    strategy: explicit_groups
    group_by: [experiment_id, sample_id]
    train_groups: [exp_01/sample_A, exp_01/sample_B, exp_02/sample_C]
    validation_groups: [exp_02/sample_D]
    test_groups: [exp_03/sample_E]
    seed: 42

balancing:
  weighted_training:
    train:
      method: class_weight
    validation:
      method: natural
    test:
      method: natural

models:
  nb_baseline:
    backend: sklearn
    family: bernoulli_nb
    parameters:
      alpha: 1.0
  forest_v1:
    backend: sklearn
    family: random_forest
    parameters:
      n_estimators: 500
      class_weight: balanced
  cnn_small:
    backend: torch
    recipe: cnn_small_v1
    overrides:
      channels: [32, 64, 128]
    initialization:
      kind: scratch

jobs:
  train_activity:
    action: train
    dataset: activity_reads
    split: sample_holdout
    balancing: weighted_training
    models: [nb_baseline, forest_v1, cnn_small]
    evaluate: [validation, test]
    explain: [native, permutation]

  apply_activity:
    action: apply
    model: model:<immutable_model_id>
    dataset: new_activity_reads

  compare_activity:
    action: plot
    runs: [run:<immutable_run_id_1>, run:<immutable_run_id_2>]
    plots: [learning_curves, roc_pr, calibration, feature_importance]
```

This is a proposed schema, not a currently supported configuration file. Its important properties
are the separation of identities and policies:

- **Dataset selection** states which registered experiments, samples, references, intervals,
  layers, and row filters are eligible. Users should select stable experiment/sample IDs exposed by
  a registry or spine, not filesystem globs.
- **Label schema** explicitly maps source values to class IDs, records class order and the positive
  class, and defines what happens to missing/unknown labels. Inferring transient pandas category
  codes is unsafe because category order may differ at inference.
- **Split policy** alone assigns eligible biological groups to train, validation, and test.
  Explicit group lists should be supported alongside seeded stratified-group generation. The
  resolved membership is persisted as a split manifest before fitting.
- **Balancing policy** is separate from split membership. Class weights, weighted samplers,
  downsampling, or upsampling normally apply only to training. Validation and locked test should
  retain natural prevalence for primary metrics. A separately named evaluation-prevalence
  sensitivity analysis may be useful, but cannot replace the natural held-out result.
- **Model declarations** name an estimator family/recipe, backend, resolved parameters, and
  initialization lineage. The run freezes all defaults; changing the YAML name later cannot change
  an existing artifact.
- **Jobs** declare intent—train, apply, evaluate, explain, or plot—and reference named definitions
  or immutable artifact IDs. An apply or plot job should never retrain. A selector such as
  `best_from: train_activity` may be accepted at planning time only if it resolves to a specific
  model ID and selection metric in the run manifest.

Validate the plan schema and all references before reading molecule matrices. Report the resolved
sample counts, class counts, group intersections, estimated memory, and output workspace as a
dry-run plan. Refuse overlapping train/validation/test groups, absent class labels, impossible
stratification, a model lacking the requested capability, or a cross-experiment job in experiment
scope.

Hydra may later generate or override these ordinary typed plans for sweeps, but core APIs should
consume the resolved schema rather than Hydra objects. This keeps one scientific configuration
contract for Click, Python, notebooks, Lightning, and plain PyTorch/sklearn execution.

### Run-manifest minimum fields

Identity and status:

- run ID, parent run/fold ID, task type, start/end timestamps, status, and failure reason;
- smftools repository commit, project repository commit, dirty-tree flag, and preferably a patch
  digest when running dirty code;
- Python, dependency, platform, accelerator, and device metadata.

Data:

- dataset snapshot ID and manifest checksum;
- split ID, group columns, membership digest, and per-split/group counts;
- selected experiments, samples, references, coordinate ranges, layers, and label mapping;
- input schema, feature transforms, imputation/fill policy, and all mask semantics.

Training:

- fully resolved architecture and recipe;
- initialization kind (`scratch`, `pretrained`, `resume`) and parent model/checkpoint ID;
- loss, optimizer, scheduler, batch size, accumulation, precision, epochs/steps, early-stopping
  rule, and checkpoint rule;
- all seeds and determinism settings.

Results:

- best epoch/checkpoint and selection metric;
- per-epoch/step history;
- validation and locked-test metrics with class balance;
- fold metrics and aggregate uncertainty where applicable;
- prediction-table checksums and plot paths;
- final model artifact ID/checksum;
- optional W&B/MLflow tracker reference.

PyTorch notes that exact reproducibility is not guaranteed across releases and platforms even with
identical seeds. Therefore dependencies, device/backend, and deterministic settings are part of the
scientific record, not incidental debugging details.

### Training curves and prediction tables

Store curves as tidy Parquet/CSV data, not only images:

```text
run_id | fold | epoch | split | metric | value | n_examples | timestamp
```

Plots should be reproducible derivatives of those tables.

Not every backend has epochs. Iterative neural, boosting, or `partial_fit` estimators may report a
step/epoch trajectory; a closed-form or one-shot sklearn estimator should instead record fit
duration, cross-validation/fold metrics, calibration/threshold curves, and final held-out metrics.
Do not manufacture a "training curve" for a model that has no meaningful iterative history. Use
nullable `epoch`/`step` fields or a separate event-kind column so the same tidy table can represent
both cases.

Store validation/test prediction tables with stable project-local molecule IDs, truth, score,
predicted class, group, and any safe evaluation strata. These enable threshold recomputation,
calibration, subgroup analysis, and independent metric verification. For exported/public artifacts,
replace row-level private data with aggregate metrics and digests as appropriate.

### Model and dataset cards

Each promoted model should include a short model card describing intended use, unsupported use,
input schema, training data scope, evaluation design, performance across relevant groups,
limitations, and parent-model lineage. Dataset snapshots should have corresponding documentation of
composition, collection/processing, labels, exclusions, and known biases. The model-card and
datasheet literature provides established reporting frameworks; smftools can use a compact
domain-specific subset rather than inventing prose ad hoc for every run.

## How to adapt the existing artifact precedents

Reuse from HMM/latent/embedding artifacts:

- canonical JSON and stable hashes;
- checksummed files;
- atomic staging and publication;
- conflict detection;
- immutable generation/model directories;
- portable relative references;
- schema and implementation versions;
- dependency versions where serialized objects require them;
- source and membership fingerprints;
- separation of current pointers from immutable generations.

Do not copy unchanged:

- `HMMModelKey` fields, which are specific to fit scopes and do not put the separately recorded
  training selection into `model_id`;
- pickle as the default neural-network exchange format;
- a model-only identity that omits the split and input schema;
- silent reliance on a mutable named variant directory.

For neural models, publish a plain `state_dict` plus configuration/input schemas for portable
inference, and optionally retain a framework checkpoint within the originating run for resume.
PyTorch recommends `state_dict` saving as the flexible model-loading approach.

### Semantic run identity versus content checksum

Keep both:

- `run_id`: unique execution identity; two attempts with identical configs still have distinct run
  IDs.
- `model_key` or semantic ID: hashes the architecture, input schema, dataset, split/training scope,
  parent model, and training configuration.
- `weights_sha256`: hashes the actual published weights.
- `model_id`: may be derived from the semantic key plus weight checksum, or be an opaque immutable
  ID with both recorded in its manifest.

This avoids the HMM-style ambiguity where a semantic key can identify a training intention while a
different stochastic result triggers a conflict. Repeated attempts should be representable and
comparable, not forced to overwrite or masquerade as one artifact.

## Recommended priorities

1. **Declare the boundary.** Make `machine_learning` the future owner of reusable trainable models
   and data/training infrastructure; keep pure result computations/plots in `analysis`; keep HMM
   task-specific.
2. **Define and test the input/mask schema.** Signal, observed, design, padding, coordinate,
   condition, corruption, and loss masks need explicit shapes and semantics before moving models.
3. **Define a group-aware split manifest.** Make sample/experiment leakage tests mandatory.
4. **Build a bounded partition-aware dataset.** Reuse the spine, catalogs, predicate pushdown, and
   `read_zarr_subset`; do not eagerly tensorize a complete AnnData.
5. **Define the ML workspace and plan schemas.** Resolve either experiment-local or project-local
   output roots and validate named datasets, labels, splits, balancing, models, and jobs before
   matrix reads.
6. **Define local run/model/explanation manifests and artifact publication.** Reuse HMM/latent
   primitives but include dataset/split/input/code identity appropriate to classifiers.
7. **Promote tested vertical baselines.** Start with Bernoulli Naive Bayes or logistic regression
   plus one active neural architecture; use the same predictions, metrics, artifacts, and
   behavioral tests.
8. **Add encoder/head lineage for pretraining and fine-tuning.** Do this when there is a demonstrated
   reusable pretraining corpus and downstream transfer benchmark.
9. **Add optional tracking.** Mirror the local schema to W&B, MLflow, or a simple local logger; do
   not make tracking the canonical record.
10. **Adopt Lightning only behind an adapter and a production requirement.** Resume/mixed
   precision/multi-device pretraining are good triggers.
11. **Revisit Hydra last.** Use it for a dedicated config-driven multirun application if manual
    experiment composition becomes the bottleneck.

## Decisions to avoid

- Do not add a third training framework alongside the two existing paths.
- Do not move additional model classes into `analysis.compute`.
- Do not treat import smoke tests as evidence that a framework works.
- Do not split correlated molecules randomly when claiming sample-level generalization.
- Do not conflate missingness, attention padding, feature design, split membership, and
  self-supervised corruption under one `mask` argument.
- Do not make architecture presets mutable without recording their resolved configuration.
- Do not upload sample-level or molecule-level data to a tracker by default.
- Do not make W&B/MLflow the only copy of scientific provenance.
- Do not use validation/test data to fit imputation, transforms, thresholds, or feature selection.
- Do not treat a Lightning checkpoint as the only portable inference format.
- Do not let model classes guess output paths or write across experiment/project scope boundaries.
- Do not encode nested ML plans in the flat experiment CSV or silently read the human-only
  `project.yaml`.
- Do not rebalance validation/test by default or let balancing redefine split membership.
- Do not claim an attribution method is model-agnostic when it requires gradients, a convolutional
  layer, or a tree structure.
- Do not load untrusted pickle/joblib model artifacts.

## Research references

Framework and data loading:

- [PyTorch `torch.utils.data` documentation](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch `TransformerEncoder` mask API](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness)
- [Lightning `Trainer` documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- [LightningModule documentation](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html)
- [LightningDataModule documentation](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
- [Lightning checkpoint contents and restoration](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
- [Lightning versioning policy](https://lightning.ai/docs/pytorch/stable/versioning.html)

Configuration and tracking:

- [Hydra introduction and config composition](https://hydra.cc/docs/intro/)
- [Hydra multirun documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [W&B experiment logging](https://docs.wandb.ai/guides/track/log/)
- [W&B artifact overview and input/output lineage](https://docs.wandb.ai/models/artifacts)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking)

Evaluation, pretraining, and documentation:

- [scikit-learn `StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [scikit-learn common leakage and preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn `Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn Naive Bayes and incremental fitting](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [scikit-learn model persistence and security/version constraints](https://scikit-learn.org/stable/model_persistence.html)
- [Captum attribution algorithm descriptions](https://captum.ai/docs/attribution_algorithms)
- [Captum algorithm comparison matrix](https://captum.ai/docs/algorithms_comparison_matrix)
- [SHAP `TreeExplainer`](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
- [SHAP `KernelExplainer`](https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

## Final assessment

The first audit successfully found the organizational fracture, the unused dependencies, the
parallel model families, and the need for model/data provenance. Its proposed destination and
priority order need revision.

The key decision is not "Lightning or plain PyTorch" and not "W&B or JSONL." It is to establish one
framework-neutral scientific contract:

```text
versioned input schema
  + group-disjoint split manifest
  + partition-aware dataset
  + experiment/project ML workspace
  + versioned ML plan
  + configurable neural or classical model recipe
  + explicit task/loss
  + immutable run/model/explanation provenance
```

Once that exists, plain PyTorch and Lightning can be interchangeable execution strategies, and W&B
or MLflow can be interchangeable tracking backends. Without it, adopting any of those tools would
standardize orchestration around still-ambiguous data, masking, and artifact semantics.
