# Artifacts, provenance, and trust

Fitted models are published as immutable, content-addressed artifacts. This page covers what gets
written, what makes a result traceable, and where the security boundary sits when loading a model
someone else produced.

## Publication is transactional

A run builds into a unique staging location, is validated, and is then published atomically. A
failure leaves the previous state intact rather than a half-written directory that looks complete.

Every published artifact carries a checksum inventory, so a truncated or altered payload is
detected on load rather than silently used.

## What makes a model traceable

A fitted model records the identities of everything that produced it:

| Field | Answers |
| --- | --- |
| `dataset_snapshot_id` | Which rows, which channels, which label schema |
| `split_id` | Which group-disjoint membership |
| `transform.transform_id` | Which fitted preprocessing, from train rows only |
| `balance.resolution_id` | Which rows were selected, with which weights and seed |
| `training_config` (Torch) | Optimizer, early stopping, seed, device, shuffle buffer |

These are content-addressed. Two runs over the same rows with the same declarations produce the
same identities. Anything that genuinely changes the result changes them.

The converse matters as much: identities are computed from **declared content**, never from
execution strategy. Batch size, worker count, and streaming-versus-materialised do not change a
transform's identity, because they do not change the transform. Where a knob genuinely does change
results it is part of the declared config — `shuffle_buffer_batches` for streamed Torch training,
`example_batch_size` for attributions — so the identity reflects it rather than hiding it.

:::{note}
`transform_id` hashes the fitted statistics at fixed precision rather than raw float64. Without
that, summation order leaks in and `batch_size` — a pure performance knob — changes a transform's
identity and therefore the lineage of every model fitted with it.
:::

## Loading a model is a trust decision

Deserialising a model executes code. The package treats that as a security boundary rather than an
implementation detail, and the two backends handle it differently.

**sklearn artifacts use `skops`, not pickle.** On save, every type `skops` considers untrusted is
inspected, and anything outside the reviewed set is refused rather than written. On load, the same
check runs again, so a payload containing an unapproved type raises instead of executing.

**Torch artifacts load with `weights_only=True`.** The checkpoint carries tensors, not arbitrary
objects, so loading cannot execute code embedded in the file.

What this does and does not buy you:

- It protects against a model file that has been tampered with or crafted to execute code on load.
- It does **not** tell you whether a model is scientifically sound, or trained on the data its
  manifest claims. Provenance identities let you check that a model matches a dataset you trust;
  they cannot vouch for a dataset you have not seen.

Optimizer state is deliberately not persisted. Published models are inference artifacts; resuming
training from a checkpoint is not a supported workflow.

## Versioned schemas

Artifact and contract schemas are versioned, and a breaking change bumps the version rather than
reinterpreting old data. Two bumps to be aware of when reading older artifacts:

- **Feature transform schema 1 → 2.** Version 2 quantises the fitted statistics before hashing
  them into `transform_id`. Transform IDs published under version 1 do not match their version 2
  recomputation.
- **Torch training config 1 → 2.** Version 2 adds `shuffle_buffer_batches`. Configs persisted
  under version 1 lack the field and will not load.

Both changes exist so that recorded provenance determines the result. A reproducibility record
that does not is not one.

## Promotion

A mutable alias can point at a published model so downstream work refers to "the current model"
rather than an identity string. The alias is validated against the artifact it names, and the
artifact itself stays immutable — promotion moves a pointer, it never rewrites a run.
