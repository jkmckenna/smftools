# Splits, balancing, and masks

The three things most likely to make a result quietly wrong. Each is enforced by the package
rather than left to discipline, and this page says where the enforcement stops.

## Splits are resolved on groups, never rows

A split assigns whole **groups** to roles. A group is defined by `group_by` — typically
`["sample_id"]`, meaning a biological or technical replicate.

```python
"splits": {
    "by_replicate": {"strategy": "leave_one_group_out", "group_by": ["sample_id"]}
}
```

A group cannot appear in two roles. Attempting it raises rather than warning, so the common
failure mode — reads from one replicate landing in both train and test, inflating every metric —
is unrepresentable rather than discouraged.

Three strategies:

| Strategy | Use when |
| --- | --- |
| `explicit_groups` | You know exactly which replicates are train, validation, and test. |
| `leave_one_group_out` | You want per-replicate folds and an honest estimate of cross-replicate generalisation. |
| `stratified_group` | You want seeded proportional assignment; set `fractions`. |

**Grouping is only as good as the field.** If `sample_id` does not actually separate the thing you
want to generalise across — two libraries from one animal, say — the split is group-disjoint and
still leaky. The package enforces disjointness of the field you name; it cannot know whether you
named the right one.

## Balancing: train may be reshaped, evaluation may not

```python
"balancing": {"weighted": {"train": {"method": "class_weight"}}}
```

| Role | Allowed methods |
| --- | --- |
| `train` | `natural`, `class_weight`, `weighted_sampler`, `downsample`, `upsample` |
| `validation`, `test` | `natural` only |

Evaluation cohorts keep their real prevalence. Asking to resample them is an error, not a warning,
because a balanced test set silently changes what precision and recall mean.

Choosing a train method:

- `class_weight` — reweights the loss. Keeps every row. The usual first choice.
- `downsample` — discards majority rows. Fast, throws away data.
- `upsample` — repeats minority rows. Keeps data, risks overfitting the repeats.
- `weighted_sampler` — Torch only, and **incompatible with streaming**, because it samples with
  replacement across the whole split and therefore needs random access.

Whatever you choose, the resolution is recorded: `result.balance.resolution_id` covers the
selected indices, the counts, and the seed.

## Masks: seven declarable kinds, four produced

Masks keep distinct meanings distinct instead of folding them into the signal. The input schema
declares them, and there are seven kinds:

`observed`, `availability`, `design`, `padding`, `attention`, `corruption`, `loss`

The partition data plane produces **four**:

| Kind | Meaning |
| --- | --- |
| `observed` | This position was actually measured for this read. |
| `availability` | This channel is available for this read at all. |
| `design` | This position is a design site for the channel's site context — a GpC for accessibility, a CpG for endogenous methylation. |
| `padding` | This position lies outside the read's span. |

They are separate on purpose. A position can be a design site that was not observed, which is not
the same as a position that was observed as zero, which is not the same as a position outside the
read. Collapsing them into one "missing" flag loses the distinction that makes SMF data
interpretable.

:::{warning}
The remaining three kinds — `attention`, `corruption`, and `loss` — are part of the declared
vocabulary but are **not produced by the partition reader**. `MLPartitionBatch.mask_arrays`
returns only the four kinds above; a declared mask of another kind is silently omitted rather
than rejected. `corruption` is reserved for pretraining, which is scoped but not built, and
`attention` for attention-capable models, of which none are registered. Declare only the four
produced kinds unless you are building the machinery that produces the others.
:::

## Transforms are fitted on train rows only

`fit_feature_transform` refuses any role other than `train`. There is no flag to relax it.

The fitted state — per-column fill values, and centres and scales when standardising — is
content-addressed as `transform_id`, computed from declared content rather than execution
strategy. Reading in batches of 16 or 512, or streaming rather than materialising, produces the
same identity for the same rows.

Streaming a fit costs a number of passes that depends entirely on the spec, and the default costs
none:

| `imputation` / `scaling` | Data passes |
| --- | --- |
| `constant` / `none` (default) | **0** — both statistics are declared, not learned |
| `constant` / `standard` | 1 |
| `mean` or `most_frequent` / `none` | 1 |
| `mean` or `most_frequent` / `standard` | 2 |
| `median` / any | refused — an exact median needs the whole column in memory |

`median` is refused for a streamed fit rather than approximated, because a near-median statistic
would change fitted models with no signal to the caller. Use `mean` or `most_frequent`, or
materialise the split if the median is scientifically required and the split fits.

## What is enforced, and what is not

Enforced:

- group-disjoint splits;
- natural prevalence in validation and test;
- train-only transform fitting;
- the locked test role, unread until early stopping has selected the best validation state.

Not enforced, and yours to get right:

- whether `group_by` names the axis you actually need to generalise across;
- whether the label column means what you think across every experiment in a project;
- whether a class present in train is present in validation and test — absent classes raise, but a
  class down to two reads will pass and tell you very little.
