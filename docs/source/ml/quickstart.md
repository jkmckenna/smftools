# Machine learning quick start

Training a classifier over a partitioned SMF experiment, for both supported backends.

Every code block here is written against the current API. The plan documents are validated by the
same parser the package uses, and the training calls use the canonical dispatch — no convenience
wrappers that only exist in documentation.

## 1. Declare a plan

The plan says what to train, on which rows, and how. Nothing is inferred from the working
directory, and unknown keys are rejected rather than ignored.

```python
from smftools.machine_learning.plan import parse_ml_plan

document = {
    "schema_version": 1,
    "scope": {"kind": "experiment"},
    "datasets": {
        "accessibility": {
            "modalities": ["conversion"],
            "channel_policy": "single_modality",
            "channels": [
                {
                    "name": "accessibility",
                    "biological_role": "accessibility",
                    "sources": [
                        {
                            "modality": "conversion",
                            "stage": "preprocess",
                            "layer": "GpC_site_binary",
                            "site_context": "GpC",
                        }
                    ],
                }
            ],
            "labels": {"column": "activity", "classes": {"inactive": 0, "active": 1}},
        }
    },
    "splits": {
        "by_replicate": {"strategy": "leave_one_group_out", "group_by": ["sample_id"]}
    },
    "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
    "jobs": {
        "train_nb": {
            "action": "train",
            "dataset": "accessibility",
            "split": "by_replicate",
            "models": ["nb"],
        }
    },
}

plan = parse_ml_plan(document)
```

See the [plan reference](plan_reference.md) for every key. Two things catch people out: the scope
key is `set`, not `set_name`, and **sklearn models declare `family` while Torch models declare
`recipe`** — declaring the wrong one is an error, not a silent fallback.

## 2. Dry run before you train

`plan_ml_workflow` resolves the whole plan — selection counts, split membership, model schemas,
output paths, optional dependencies — without training or writing anything.

```python
from smftools.machine_learning.orchestration import plan_ml_workflow

report = plan_ml_workflow(plan, experiment_config=config)
```

Do this first. It catches an empty cohort, a split that leaves a class absent from a role, or a
missing optional dependency in a second, rather than after a long read.

## 3. Train

`train_partition_model` dispatches to the right engine for the model's backend.

```python
from smftools.machine_learning.models.registry import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.orchestration import train_partition_model

resolved = BUILTIN_MODEL_REGISTRY.resolve(
    "bernoulli_nb", input_schema=dataset.plan.dataset.input_schema
)
result = train_partition_model(dataset, resolved)

result.model.fit_mode              # 'partial_fit'
result.n_training_observations
result.balance.result_counts
```

`dataset` here is the partition dataset the workflow resolves for a job. `bind_ml_job` builds it
from a plan: the dataset snapshot, one split manifest per fold, and the experiments' stage spines.
`run_bound_train_job` then trains every model the job declares on each fold and evaluates it on
that fold's test role:

```python
from smftools.machine_learning.orchestration import bind_ml_job, run_bound_train_job

bound = bind_ml_job(plan, "train_nb", project_dir="path/to/project")
for run in run_bound_train_job(bound):
    print(run.fold_name, run.model_name, run.evaluation.metrics)
```

Position-agnostic CNNs with a bounded detector span come as recipes -- `rcnn_subnucleosome_v1`
(113 bp), `rcnn_2_3_nucleosomes_v1` (513), `rcnn_4_6_nucleosomes_v1` (1,025),
`rcnn_full_locus_v1` (5,121) -- e.g. `{"backend": "torch", "recipe": "rcnn_subnucleosome_v1"}`.
Each fold model's run record notes its theoretical receptive field and measured effective span
(`detector_scale` in `models.json`).

Neural models stop early on a validation role. With leave-one-group-out, declare one inside each
fold: the held-out experiment stays the test set, and a seeded fraction of the training
experiments' molecules (stratified by experiment x class) becomes validation --
`validation_fraction: 0.15` on the split (`validation_by: groups` holds out whole training
experiments instead). Final models take the same fraction. Every model of the job then trains
on the same, smaller train role.

Results come back in memory. To keep them, `train_and_publish` runs the same job through the
train job service and publishes one immutable run in the workspace (`project_outputs/ml/` for a
project):

```python
from smftools.machine_learning.orchestration import train_and_publish
from smftools.machine_learning.orchestration import runs

run = train_and_publish(bound, project_dir="path/to/project", tags={"task": "b6_vs_nk/promoter"})
run.run_id, run.model_ids["nb"]        # fold name -> published model id
run.read(runs.METRICS)                 # per model, fold and metric
run.summary["nb"]["roc_auc"]["mean"]   # mean over folds
```

The run holds the plan, environment and seeds; `data/membership.parquet` (every molecule's role
and class per fold, so the training and evaluation sets are exact); `models.json` (each fold
model, published as its own model bundle and reloadable with `load_published_sklearn_model` or
`load_published_torch_model`); held-out predictions; metrics, including average precision at a
fixed positive prevalence (`prevalence`, default 10 %); ROC / PR curves; training history; and a
fold summary. The workspace run index (`index/runs.json`) lists every run with its tags and
summary. A failed job still publishes a failed run manifest, then raises.

To reuse a model, train a final one on every selected row as well, then apply it to any dataset
with the same channels and positions (and classes, if it has labels) through a plan's apply job:

```python
run = train_and_publish(bound, project_dir="path/to/project", final_model=True)
final_id = run.model_ids["nb"]["final"]

# plan_new declares the new dataset and a job
#   {"action": "apply", "dataset": "new_cohort", "model": "nb"}  (or "model": "model:<id>")
from smftools.machine_learning.orchestration import apply_and_publish

applied = apply_and_publish(plan_new, "apply", model_id=final_id, project_dir="path/to/project")
applied.read(runs.APPLIED_PREDICTIONS)   # molecule, class probabilities (and truth, if labeled)
```

The apply run's manifest names the source model, and `model.json` its originating train run.
Labeled data also gets metrics, curves and a summary; mismatched channels, positions or classes
are refused, naming the difference. Final torch models wait for a validation role.

To explain a run's model, out of fold -- each held-out molecule by the fold model that held it
out:

```python
from smftools.machine_learning.orchestration import explain_run
from smftools.machine_learning.orchestration import explanations

explained = explain_run(run.run_id, model="nb", method="NaiveBayesLogOdds",
                        project_dir="path/to/project", max_per_fold=2000)
molecules, matrix = explained.attributions(fold)   # matrix: molecules x channels x positions
explained.read(explanations.IMPORTANCE)            # per fold, channel, position
explained.summary["nb"]["fold_consistency_spearman"]
```

Without `method=`, each family's default is used (its `default_explanation`). Families also
declare a model class -- `additive`, `tabular_nonlinear`, `spatial`, `global_sequence` -- recorded
with each run's models and shown in comparisons. Gradient methods and detector catalogues are much faster with the CNN on a GPU:
`explain_run(..., device="mps")` (or `"cuda"`, `"auto"`; default `"cpu"`), likewise
`detector_catalogue_run` and `apply_and_publish`. Methods follow the model: `NaiveBayesLogOdds` (naive Bayes), `TreeSHAP` (random forest),
`IntegratedGradients` and other Captum methods (torch) give per-molecule matrices;
`LinearCoefficients` and `PermutationImportance` give position importance only. Classical
attributions sum each position's transformed features (signal and indicators). Up to
`max_per_fold` held-out molecules per fold are explained (class-stratified, seeded). The explain
run names the fold models it used and is refused if the run's data have changed since training.

Per-molecule explanations also keep the explained inputs, and the run draws a default figure
(`figures/attributions.png`): each channel's input beside its attributions, one row per molecule,
blocked by true class, with held-out-experiment and score strips. Other figures come from the
record without re-reading data:

```python
from smftools.machine_learning.orchestration import plot_explanation

molecules, attributions, inputs = explained.pooled()
plot_explanation(explained, "by_ndr_state.png", order="bins", bins=ndr_state_of(molecules),
                 bin_order=["enhancer only open", "both open", "promoter only open", "neither open"],
                 bin_name="NDR state", coordinate_labels=tss_relative,
                 extra_panels=[{"name": "HMM accessible", "matrix": hmm_layer(molecules)}])
```

`order` is `"label"` (default), `"score"` (highest first) or `"bins"`; attributions use a
diverging scale symmetric about zero. Columns are the positions observed in at least one drawn
molecule (`columns="observed"`, the default -- for site channels, the sites; `"all"` draws every
position), labelled with real coordinates and separated only at mask-window breaks; the trace
above each panel is split by true class. Inputs are coloured by the channel's biological role (accessibility
green, methylation red; unobserved grey). Within each class (or bin) block, rows are clustered
(`within="hierarchical"`, on `cluster_on="attributions"` or `"inputs"`) or ranked by
out-of-fold score (`within="score"`); the score strip has a 0-1 colour bar. Extra panels and strips must be aligned with the pooled
molecules.

For a CNN, the detector catalogue describes what each final-layer detector responds to, out of
fold: its top windows (one per molecule, at its effective span), their mean input pattern,
where on the locus they sit, the detector's AUROC on its own, top-window enrichment, and
groups of redundant detectors:

```python
from smftools.machine_learning.orchestration import detector_catalogue_run
from smftools.machine_learning.orchestration import detectors

catalogue = detector_catalogue_run(run.run_id, model="cnn_sub", project_dir="path/to/project")
catalogue.read(detectors.DETECTORS)   # per fold and detector
catalogue.read(detectors.WINDOWS)     # each detector's top molecules and positions
```

To compare models across runs -- on the same held-out experiments, molecule for molecule:

```python
from smftools.machine_learning.orchestration import compare_runs, select_runs

runs = select_runs(project_dir="path/to/project", tags={"cell": "fresh_b6_vs_nk"})
comparison = compare_runs(runs["run_id"], project_dir="path/to/project",
                          reference="full_locus/rf", n_bootstrap=200)
comparison.summary       # per entry and metric: fold mean, SD, bootstrap interval
comparison.differences   # paired differences to the reference, folds better
comparison.write("project_outputs/ml_comparisons/regions")   # tables, settings, figure
```

Entries are named by the tags that differ among the runs, then the model. Folds are matched by
held-out group (folds not shared by every entry are dropped, with a warning); within each fold,
metrics are recomputed on the molecules every entry predicted. Two kinds of interval are
reported: `ci_low` / `ci_high` from a seeded bootstrap over molecules within folds
(class-stratified, the same resamples for every entry) -- molecule sampling only -- and
`fold_ci_low` / `fold_ci_high` from resampling held-out experiments, with an exact paired
sign-flip test (`sign_flip_p`) on the per-fold differences: what a new batch would see. With
n folds the smallest attainable `sign_flip_p` is 2 / 2**n (0.0625 for 5).

For a whole-cohort analysis with no folds -- an embedding, clustering -- `bind_ml_dataset` reads
every selected row of one dataset through the same selection (label tables, filters, `positions`,
coordinate frames), and the plan may declare datasets only:

```python
from smftools.machine_learning.orchestration import bind_ml_dataset

bound = bind_ml_dataset(plan, "reads", project_dir="path/to/project", group_by=["Barcode"])
bound.identity          # one row per molecule, in snapshot order
for batch in bound.iter_batches():
    ...
```

A bound job's folds share one decoded-row cache, so each molecule is read from the stores once
however many folds and models the job has (`PartitionReadPolicy(row_cache_bytes=...)`; the default
is the materialization budget, `0` turns it off). Results equal an uncached read.

### sklearn streams by default

For families declaring `incremental_fit` — `bernoulli_nb` today — training reads in bounded batches
with **no row ceiling**, because a streamed sklearn fit is numerically identical to a materialized
one. You do not ask for this.

`logistic_regression` and `random_forest` have no `partial_fit`, so they materialize the train
split and are bounded by `max_materialization_bytes`. Above that they refuse, and the refusal names
the streaming-capable alternatives.

To force the materialized path:

```python
from smftools.machine_learning.orchestration import SklearnTrainOptions

result = train_partition_model(
    dataset, resolved, sklearn_options=SklearnTrainOptions(streaming=False)
)
```

### Torch asks first

```python
from smftools.machine_learning.orchestration import TorchTrainOptions, train_partition_model
from smftools.machine_learning.training import TorchTrainingConfig

resolved = BUILTIN_MODEL_REGISTRY.resolve(
    "residual_dilated_cnn", input_schema=dataset.plan.dataset.input_schema
)
result = train_partition_model(
    dataset,
    resolved,
    torch_options=TorchTrainOptions(
        streaming=True,
        training_config=TorchTrainingConfig(max_epochs=20, device="auto"),
    ),
)

result.model.best_epoch
result.model.validation_loss
result.model.test_loss
```

Torch does **not** stream by default, and the asymmetry with sklearn is deliberate: a streamed
Torch fit shuffles within a buffer rather than globally, so it produces different weights at the
same seed. Silently switching strategy would hand you a different model. If a materialized Torch
fit exceeds the budget, the refusal names `TorchTrainOptions(streaming=True)` and says that weights
will differ.

:::{note}
Sizing a Torch run from the data budget does not work — process memory is dominated by model
activations, which nothing in the data plane estimates. See
[performance and limits](performance.md).
:::

## 4. What you get back

A training result carries the fitted model plus the provenance that makes it reproducible:

- `result.model.transform.transform_id` — the fitted transform, from train rows only
- `result.balance.resolution_id` — which rows were selected and how
- `result.model.dataset_snapshot_id` and `.split_id` — the exact cohort and membership

These are content-addressed. Two runs over the same rows with the same declarations produce the
same identities; anything that genuinely changes the result changes them too. The
[architecture guide](architecture.md) traces the whole chain.

## Where to go next

- [Plan reference](plan_reference.md) — every key, every enumerated value.
- [Performance and limits](performance.md) — what is supported, what is refused, what is not
  modelled at all.
- [ML migration guide](../tutorials/ml_migration.md) — replacing the legacy
  `analysis.compute.ml_*` entry points.
