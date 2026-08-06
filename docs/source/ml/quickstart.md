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

`dataset` here is the partition dataset the workflow resolves for a job. Constructing one directly
means binding a dataset snapshot, a split manifest, and local stage spines by hand — an advanced
path, not the normal one.

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
