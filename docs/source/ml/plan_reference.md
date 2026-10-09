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
| `splits` | yes | Named split declarations; may be empty for a plan read only through `bind_ml_dataset`. |
| `balancing` | no | Named balancing profiles. Omitting means natural prevalence everywhere. |
| `models` | yes | Named model declarations; may be empty. |
| `jobs` | yes | Named jobs; may be empty. |
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
| `positions` | no | `{"include": [[start, end), ...], "exclude": [...]}`: the reference positions used as features. See below. |
| `coordinate_frame` | no | `{"reference": <name>, "maps": {<reference>: <table>}}`: put several references' molecules in one reference's coordinates. See below. |

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

**Positions**: by default a dataset uses every position of its reference (or `filters.start` to
`filters.end`). `positions` keeps only `include` windows minus `exclude` windows, half-open, in the
reference's forward coordinates — for region studies such as "everything but the enhancer":

```python
"positions": {"include": [[995, 3718]], "exclude": [[3127, 3528]]}
```

Only kept positions become features: a masked position contributes no signal and no mask
indicator, rather than an imputed constant. Kept windows are placed side by side, so a
convolutional model sees them joined; a single window keeps true distances. `positions` cannot be
combined with `filters.start`/`end`.

**Coordinate frame**: molecules of a structural variant (e.g. an enhancer deletion) sit at
different positions than the same bases on the intact allele. `coordinate_frame` places them in
the frame reference's coordinates through a project-relative map per other reference — a table of
`source_position`, `frame_position` pairs, one-to-one and in order; a source position it omits has
no frame counterpart:

```python
"references": ["6B6", "6B6_enh_del"],
"coordinate_frame": {"reference": "6B6", "maps": {"6B6_enh_del": "ml/maps/del_to_b6.parquet"}},
"positions": {"include": [[0, 3151]]},
```

Which positions a molecule *has* would identify its reference — in an intact-vs-deletion task, its
class. So every selected frame position must exist on every selected reference, and selection
refuses otherwise, naming the frame positions some reference lacks; select shared sequence with
`positions`. There is no override. Each map's checksum is part of the dataset identity. Project
scope only.

## `splits`

| Key | Required | Notes |
| --- | --- | --- |
| `strategy` | yes | `explicit_groups`, `leave_one_group_out`, or `stratified_group`. |
| `group_by` | yes | Fields defining a group, e.g. `["sample_id"]`. |
| `train_groups`, `validation_groups`, `test_groups` | for `explicit_groups` | Group names per role. `leave_one_group_out` accepts `train_groups` only: groups that train in every fold and are never held out. |
| `single_class_groups` | no | `leave_one_group_out` only. `refuse` (default) fails when a group lacks a class, since it cannot be scored as a test fold; `train` keeps such groups in every fold's train role instead. |
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

Train also takes `max_per_class` (with `natural`, `downsample` or `class_weight`) and `seed`:

```python
{"capped": {"train": {"method": "downsample", "max_per_class": 2000, "seed": 1}}}
```

`max_per_class` caps every class at that many training molecules per fold. With `downsample`, each
class gets the smaller of the cap and the smallest class. Use it to compare datasets of different
sizes on equal terms, or to draw learning curves (one profile per size). `seed` draws the training
cohort independently of the job seed, so repeat draws change only the cohort. The counts actually
trained are recorded per fold model (`n_train`, `train_class_counts` in the run's `models.json`).
Test and validation sets are never capped.

A job's balancing profile applies even when training options are passed explicitly (as Torch
training always does). Options that declare a *different* balancing are refused.

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

An sklearn model may declare `"calibration": "sigmoid"`: after fitting, Platt scaling (with Platt's
smoothed targets) of its positive-class log-odds is fitted on the fold's validation molecules, so
the split needs a `validation_fraction` (or validation groups). Predictions, metrics and applied
scores then use the calibrated probabilities. The ranking of molecules is unchanged, so AUROC and
AUPRC are too, except that ties from saturated probabilities are broken by the exact log-odds.
Naive Bayes in particular adds one log-likelihood ratio per site, so its raw probabilities are
extreme on long reads. Explanations use the uncalibrated estimator.

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
