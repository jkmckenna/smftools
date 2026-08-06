# Tutorial: training across a project

Pooling several experiments into one model, and the identity problems that only appear once you do.

:::{note}
A narrative walkthrough against your own project, like the
[experiment-local tutorial](ml_experiment_workflow.md). Read that one first — this covers only
what changes at project scope.
:::

## What actually changes

Mechanically, very little: the scope becomes `project`, and selection can span experiments.

```python
"scope": {"kind": "project", "set": "nkg2a_active_vs_inactive"},
```

Scientifically, three things change, and each has bitten real analyses.

## 1. Read identity must be collision-free

Read IDs are unique within an experiment and **not** across experiments. Two experiments can
easily contain the same basecaller-assigned read ID for entirely different molecules.

The package keys pooled rows on `molecule_uid`, which combines experiment identity with read
identity, so a collision is impossible rather than unlikely. You do not have to do anything for
this — but it is why a project result is keyed the way it is, and why a cached per-experiment
result computed under bare read IDs cannot be pooled.

If you see an error about molecule identity when pooling, it is this: something upstream was
computed before identity was collision-free, and it needs recomputing rather than coercing.

## 2. Grouping is a scientific choice, not a formality

At experiment scope, `group_by: ["sample_id"]` usually means "replicate", and that is usually
right.

At project scope you have to decide what a fold should hold out:

| `group_by` | Holds out | Answers |
| --- | --- | --- |
| `["sample_id"]` | One replicate, possibly within the same experiment | Does this generalise to another replicate? |
| `["experiment_uid"]` | A whole experiment | Does this generalise to another experiment — different day, prep, flow cell? |

**These give very different numbers, and the second is usually the one you want.** A model that
generalises across replicates within one experiment but collapses on the next experiment has
learned batch structure. Grouping by experiment is the honest test, and it is the one people skip
because it scores worse.

The package enforces disjointness of whatever field you name. It cannot tell you that `sample_id`
does not separate experiments.

## 3. Labels must mean the same thing everywhere

Pooling assumes `activity == "active"` means the same thing in every experiment. That assumption
is easy to violate quietly:

- a threshold that drifted between analyses;
- a class name reused for a different construct;
- one experiment labelled by a different person or method.

Nothing in the package can detect this. It will train happily on incoherent labels and give you a
mediocre model with no indication why. Check the label provenance across experiments before
pooling; if two experiments labelled differently, either relabel or keep them separate.

## Selecting a cohort

```python
"datasets": {
    "reads": {
        "modalities": ["conversion"],
        "channel_policy": "single_modality",
        "experiments": {"include": ["exp_a", "exp_b", "exp_c"]},
        "samples": {"exclude": ["exp_b_rep2"]},
        "references": ["6B6_top"],
        "channels": [...],
        "labels": {"column": "activity", "classes": {"inactive": 0, "active": 1}},
    }
}
```

Excluding a known-bad replicate here, in the declaration, is better than filtering upstream:
the exclusion becomes part of the dataset snapshot identity, so the resulting model records that
it was trained without `exp_b_rep2` rather than leaving that in someone's memory.

## Mixed modalities

If a project spans deaminase and conversion experiments, one biological channel can draw from
different physical layers per modality:

```python
{
    "name": "accessibility",
    "biological_role": "accessibility",
    "sources": [
        {"modality": "deaminase", "stage": "preprocess",
         "layer": "C_site_binary", "site_context": "C"},
        {"modality": "conversion", "stage": "preprocess",
         "layer": "GpC_site_binary", "site_context": "GpC"},
    ],
}
```

Set `channel_policy` to `harmonized` or `union` — `single_modality` is only valid for one
modality.

This is a genuine scientific claim: that a C site in deaminase data and a GpC site in conversion
data measure the same underlying accessibility well enough to pool. Sometimes true, sometimes not.
The `availability` mask records per read which channels were actually available, so the model is
not fed a fabricated zero where a modality simply cannot measure something — but whether pooling
is *valid* remains your call.

## Scale

Project cohorts are where the materialisation ceiling starts to matter. At a million reads no
split materialises, so:

- **sklearn** — use a family declaring `incremental_fit`, which streams by default with no
  ceiling. `bernoulli_nb` qualifies.
- **Torch** — pass `TorchTrainOptions(streaming=True)`. Not the default, because a streamed fit
  shuffles within a buffer and produces different weights from a materialised one.

Size Torch runs empirically: process memory is dominated by model activations, which no budget in
the data plane estimates. See [performance and limits](../ml/performance.md).

## Where to go next

- [Splits, balancing, and masks](../ml/splits_and_masks.md) — what is and is not enforced.
- [Artifacts, provenance, and trust](../ml/artifacts_and_trust.md) — what a published model records.
