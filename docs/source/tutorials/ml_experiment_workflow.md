# Tutorial: training a model on one experiment

Walking a classifier from a preprocessed experiment to a published model with traceable
provenance.

:::{note}
This is a narrative walkthrough against **your own** preprocessed experiment, not a runnable
notebook — the ML data plane reads partitioned stores, and no bundled dataset is one. Every call
shown is current API; the data is yours.
:::

## Before you start

You need an experiment that has been through `preprocess`, so that a stage spine and partition
store exist. The ML layer reads those; it does not read raw BAMs or a monolithic `.h5ad`.

You also need a label. Classification needs a column in `obs` that says what each read *is* —
typically an activity or genotype call — and it needs to be present for the reads you intend to
train on.

## 1. Decide the biological question first

The plan makes you state the question before any code runs, which is the point. Three decisions
matter more than anything else you will tune later:

**What are you predicting?** The label column, and what its classes mean.

**What can the model see?** One channel or two. For conversion SMF, `GpC_site_binary` is
accessibility and `CpG_site_binary` is endogenous methylation. Giving the model both is not
automatically better — if endogenous methylation trivially separates your classes, you will learn
that rather than the accessibility biology you were asking about.

**What must it generalise across?** This becomes `group_by`, and it is the decision most often got
wrong. If you want a model that works on a new replicate, group by replicate. Grouping by
something finer — read ID, or a field that repeats within a replicate — produces a model that
looks excellent and generalises to nothing.

## 2. Write the plan

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
            "labels": {
                "column": "activity",
                "classes": {"inactive": 0, "active": 1},
                "positive_class": "active",
            },
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

Start with `bernoulli_nb`. It is fast, it streams, and its explanations are exact rather than
approximate — which makes it a good instrument for finding out whether your labels and channels
make sense before you spend time on a CNN.

## 3. Dry run

```python
from smftools.machine_learning.orchestration import plan_ml_workflow

report = plan_ml_workflow(plan, experiment_config=config)
```

Read the selection counts before anything else. The common surprises:

- **Far fewer rows than expected** — a reference or sample filter is excluding more than intended,
  or labels are missing for most reads and being dropped.
- **A class absent from a role** — one replicate has only one class, so leave-one-out folds cannot
  evaluate. This raises rather than producing a meaningless metric.
- **Wildly uneven groups** — one replicate dominating training means a leave-one-out fold on it is
  the only honest estimate you have.

Fix these here. They are cheap now and expensive after a long read.

## 4. Train

```python
from smftools.machine_learning.models.registry import BUILTIN_MODEL_REGISTRY
from smftools.machine_learning.orchestration import train_partition_model

resolved = BUILTIN_MODEL_REGISTRY.resolve(
    "bernoulli_nb", input_schema=dataset.plan.dataset.input_schema
)
result = train_partition_model(dataset, resolved)
```

`bernoulli_nb` declares `incremental_fit`, so this streams and has no row ceiling. Nothing to
configure.

If you switch to `random_forest` or `logistic_regression`, training materialises the train split
and is bounded — around 51,000 rows at 1,000 positions with one channel. Above that it refuses and
names the alternatives. See [performance and limits](../ml/performance.md).

## 5. Read the result honestly

```python
result.balance.result_counts          # class counts actually trained on
result.model.transform.transform_id   # fitted preprocessing, train rows only
result.model.split_id                 # which membership
```

A leave-one-group-out plan gives one fold per replicate. **Look at the spread, not the mean.** If
three replicates score 0.85 and one scores 0.55, the honest summary is "works on three of four
replicates and we do not know why the fourth differs" — not 0.775.

## 6. Ask what the model learned

For a Bernoulli NB, per-feature log odds are exact and free:

```python
"jobs": {
    "explain_nb": {
        "action": "explain",
        "dataset": "accessibility",
        "model": "nb",
        "source_job": "train_nb",
        "explain": ["NaiveBayesLogOdds"],
    }
}
```

Attributions are aligned to genomic coordinates, so you can ask whether the positions the model
weights correspond to a footprint you recognise. If the model is accurate but its weight sits on
positions with no biological interpretation, that is worth understanding before trusting it —
often it is a batch or coverage artefact tracking your label.

`PermutationImportance` answers a different and usually more useful question: how much held-out
performance actually depends on a feature. It is computed on validation or test rows, never train.
See [choosing an interpretability method](../ml/interpretability.md).

## Where to go next

- Scaling to many experiments: [project-level tutorial](ml_project_workflow.md).
- Why a split, balancing choice, or mask behaves as it does:
  [splits, balancing, and masks](../ml/splits_and_masks.md).
- Publishing and loading models: [artifacts, provenance, and trust](../ml/artifacts_and_trust.md).
