# ML plan reference

An ML plan is the declarative description of what to train, on which rows, and how to evaluate and
explain it. It is validated strictly — unknown keys are rejected rather than ignored, so a typo
fails loudly instead of silently changing nothing.

```python
from smftools.machine_learning.plan import parse_ml_plan

plan = parse_ml_plan(document)   # document is a dict, e.g. from yaml.safe_load
plan.plan_hash                   # content identity, feeds every downstream artifact
```

Current schema version: **1**.

## A minimal plan

Every required key, nothing optional:

```python
{
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
```

## Top level

| Key | Required | Notes |
| --- | --- | --- |
| `schema_version` | yes | Integer. Currently `1`. |
| `scope` | yes | `{"kind": "experiment" \| "project", "set": <name>}`. Note the key is `set`, not `set_name`. |
| `datasets` | yes | Named dataset declarations. |
| `splits` | yes | Named split declarations. |
| `balancing` | no | Named balancing profiles. Omitting means natural prevalence everywhere. |
| `models` | yes | Named model declarations. |
| `jobs` | yes | Named jobs. |
| `tracking` | no | `{"provider": "none", ...}`. Tracker integrations are deferred; `none` is the only supported provider. |

## `datasets`

| Key | Required | Notes |
| --- | --- | --- |
| `modalities` | yes | Subset of `deaminase`, `conversion`, `direct`. |
| `channel_policy` | yes | `single_modality` for one modality; `harmonized` or `union` for several. The allowed values depend on how many modalities you declared. |
| `channels` | yes | Ordered biological channels; see below. |
| `experiments`, `samples` | no | `{"include": [...], "exclude": [...]}`. |
| `references` | no | Reference names to restrict to. |
| `filters` | no | Free-form additional selection on per-read metadata: the molecule index, the raw obs, then the obs of each stage the dataset reads (e.g. preprocess `passes_qc`, `passes_dedup`). |
| `labels` | no | Required for any dataset a `train` job uses. |

Each **channel** separates the biological meaning from the physical layer it comes from:

```python
{
    "name": "endogenous_methylation",
    "biological_role": "endogenous_methylation",
    "sources": [
        {"modality": "conversion", "stage": "preprocess",
         "layer": "CpG_site_binary", "site_context": "CpG"}
    ],
}
```

Multiple sources let one biological channel be populated from different physical layers per
modality — that is what makes a mixed-modality dataset coherent rather than a concatenation.

Which layer to name depends on how the store was written. Stores from the partitioned preprocess
stage hold binary site calls (0/1, NaN where unobserved) in `X` and write no `*_site_binary`
layers, so declare `"layer": "X"` with the site context to select — `C` or `GpC` for a deaminase,
`GpC` or `CpG` for conversion. The default channels name the `*_site_binary` layers of the older
single-file preprocess output. If a declared layer is absent, selection lists the layers the stage
did write.

**Labels**: `column` and `classes` are required; `source` defaults to `obs`, `missing` to `drop`,
and `positive_class` is optional but recommended for binary tasks so downstream metrics know which
class is positive.

With `source: obs` the label column is read from each experiment's stored molecule metadata. With
`source: table` it comes from a table in the project, joined to each molecule on the fields named
in `keys` -- for labels that live in a sample sheet rather than in any stored stage:

```python
"labels": {
    "source": "table",
    "table": "ml/labels/b6_vs_nk.parquet",     # project-relative .parquet or .csv
    "keys": ["experiment_id", "barcode", "reference"],
    "column": "label",
    "classes": {"inactive": 0, "active": 1},
    "positive_class": "active",
}
```

- `keys` may name `experiment_id`, `experiment_uid`, `barcode`, `sample`, `reference` (canonical,
  through the project's reference registry) and `physical_reference` (strand-level).
- Barcodes compare by number, so a sheet's `4` matches a stored `barcode04` or `NB04`.
- A key may appear in only one table row. Molecules with no row follow `missing`.
- The table's other columns are available to `filters` and to a split's `group_by`, so task
  fields can live beside the label. A table column may not reuse the name of a stored metadata
  column.
- The table's checksum is part of the dataset selection identity: editing a label changes it.
- Project scope only, since the path is resolved against the project directory.

## `splits`

| Key | Required | Notes |
| --- | --- | --- |
| `strategy` | yes | `explicit_groups`, `leave_one_group_out`, or `stratified_group`. |
| `group_by` | yes | Fields defining a group, e.g. `["sample_id"]`. |
| `train_groups`, `validation_groups`, `test_groups` | for `explicit_groups` | Group names per role. |
| `fractions` | for `stratified_group` | Role fractions. |
| `seed` | no | Defaults to `0`. |

Splits are always resolved on **whole groups**. A group cannot appear in two roles — that is how
leakage is prevented structurally rather than by convention.

## `balancing`

A named profile per role:

```python
{"weighted": {"train": {"method": "class_weight"}}}
```

Train accepts `natural`, `class_weight`, `weighted_sampler`, `downsample`, `upsample`.
**Validation and test accept only `natural`** — primary evaluation cohorts keep their real
prevalence, and asking for anything else is an error rather than a warning.

Note `weighted_sampler` is Torch-only, and it cannot be combined with streaming training because it
samples with replacement across the whole split.

## `models`

| Backend | Required key | Forbidden key |
| --- | --- | --- |
| `sklearn` | `family` | `recipe` |
| `torch` | `recipe` | `family` |

This asymmetry is enforced, and mixing them up is one of the easier mistakes to make:

```python
{"nb":  {"backend": "sklearn", "family": "bernoulli_nb"}}
{"cnn": {"backend": "torch",   "recipe": "residual_dilated_cnn"}}
```

Registered names today: `bernoulli_nb`, `logistic_regression`, `random_forest` (sklearn) and
`residual_dilated_cnn` (torch). Optional `parameters`, `overrides`, and `initialization` refine a
declaration; `initialization` defaults to `{"kind": "scratch"}`.

## `jobs`

Actions are `train`, `apply`, `evaluate`, `explain`, and `plot`. Each has its own required fields,
and the validator rejects fields that do not belong to the action:

| Action | Requires | Must not declare |
| --- | --- | --- |
| `train` | `dataset` (with labels), `split`, at least one entry in `models` | `model`, `source_job`, `runs`, `plots` |
| `apply` | `dataset`, `model`, and a `source_job` referencing a train job | — |
| `evaluate` | `dataset`, `source_job` referencing an apply or train job | — |
| `explain` | `dataset`, `model`, and at least one method in `explain` | — |

A worked chain — train, apply, evaluate, explain:

```python
"jobs": {
    "train_cnn": {"action": "train", "dataset": "reads", "split": "by_replicate",
                  "balancing": "weighted", "models": ["cnn"]},
    "apply_cnn": {"action": "apply", "dataset": "reads", "model": "cnn",
                  "source_job": "train_cnn"},
    "evaluate_cnn": {"action": "evaluate", "dataset": "reads", "source_job": "apply_cnn"},
    "explain_cnn": {"action": "explain", "dataset": "reads", "model": "cnn",
                    "source_job": "train_cnn", "explain": ["IntegratedGradients"]},
}
```

A `source_job` cannot reference itself, and every cross-reference — dataset, split, balancing
profile, model, source job — is checked against the plan before anything runs.

## Validating without running

`plan_ml_workflow` resolves a whole plan against a real experiment or project and reports selection
counts, split membership, model schemas, output paths, and optional-dependency availability,
without training or writing artifacts:

```python
from smftools.machine_learning.orchestration import plan_ml_workflow

report = plan_ml_workflow(plan, experiment_config=config)   # or project_dir=...
```

Exactly one scope input is mandatory and it must match `plan.scope`. No path is inferred from the
working directory.
