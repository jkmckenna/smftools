# Machine learning: architecture and ownership

`smftools.machine_learning` owns trainable models and everything they need: data selection,
splits, transforms, training, inference, evaluation orchestration, interpretability, and immutable
artifacts. `smftools.analysis` owns pure result computation and plotting.

The boundary is what makes results reproducible, so it is worth stating precisely.

## Who owns what

| Concern | Owner |
| --- | --- |
| Input, mask, label, and capability contracts | `machine_learning.contracts` |
| Versioned ML plan (datasets, splits, models, jobs) | `machine_learning.plan` |
| Dataset snapshot, source, and split manifests | `machine_learning.manifests` |
| Workspace and run-path resolution | `machine_learning.workspace` |
| Row selection from experiments and projects | `machine_learning.selection`, `machine_learning.splitting` |
| Partition reads, transforms, balancing | `machine_learning.data` |
| Model families, recipes, registry | `machine_learning.models` |
| Training engines (sklearn and plain PyTorch) | `machine_learning.training` |
| Prediction records and backend adapters | `machine_learning.inference` |
| Evaluation orchestration | `machine_learning.evaluation` |
| Attribution computation | `machine_learning.interpretability` |
| Immutable run, model, and result artifacts | `machine_learning.artifacts` |
| Job services (train/apply/evaluate/explain) | `machine_learning.orchestration` |
| Pure metric summaries and result plots | `smftools.analysis` |

`smftools.ml` is a compatibility alias for `smftools.machine_learning`. The full name is canonical.

The rule that keeps the two halves apart: **`analysis` receives results and returns numbers or
figures.** It never trains, never reads a partition store, and never decides what a model is. If a
function needs a fitted model or a dataset, it belongs in `machine_learning`.

## The contract chain

Every fitted model is reachable from the data that produced it through a chain of content-addressed
identities. Each link is a SHA-256 over the declared content of the one before it:

```text
ML plan            plan_hash
  └─ dataset snapshot   snapshot_id      selection, input schema, label schema, sources
       └─ split         split_id         group-disjoint train/validation/test membership
            ├─ transform    transform_id  fill values, centres, scales, fitted on train only
            ├─ balance      resolution_id selected indices, class weights, seed
            └─ model                      architecture recipe plus the three above
                 ├─ predictions
                 └─ explanations  request_id -> result_id
```

Two consequences worth internalising:

- **Changing anything upstream changes every identity downstream.** That is the point. A model
  fitted on a different split is a different model, and its identity says so.
- **Identities are computed from declared content, never from execution strategy.** Reading in
  batches of 16 or 512 produces the same `transform_id`; streaming a fit or materialising it
  produces the same fitted transform. Where a knob genuinely changes results — `example_batch_size`
  for attributions, `shuffle_buffer_batches` for streamed Torch training — it is part of the
  declared request or config, so the identity reflects it honestly rather than hiding it.

## Leakage prevention is structural

Train-only discipline is enforced by the types, not by convention:

- `fit_feature_transform` refuses any role other than `train`.
- Splits are resolved on whole groups; a group cannot span two roles.
- Validation and test cohorts keep natural prevalence — resampling them raises.
- The locked test role is not read until early stopping has selected and restored the best
  validation state.

You do not opt into these. There is no flag that turns them off.

## Guaranteed, optional, and deprecated

The acceptance criterion for this documentation is that it distinguishes these, so it is explicit.

**Guaranteed.** Contracts, plans, manifests, workspace resolution, partition reads, transforms and
balancing, the sklearn and plain-PyTorch training paths, prediction and evaluation records,
classical and Captum-backed explanations, immutable artifacts, and the job services. These have
versioned schemas; a breaking change bumps the schema version and carries a migration note.

**Optional.** Captum and SHAP (attribution methods), and `ml-extended` dependencies generally.
These are imported lazily, after preflight validation, so a missing extra fails with an actionable
message rather than an import error at the top of a run.

**Proposed, not built.** PyTorch Lightning, experiment trackers, and Hydra composition are
recorded as deferred integrations. Nothing in the package depends on them, and no documented
behaviour assumes them. A pretrained-encoder and fine-tuned-head lineage is likewise scoped but
not implemented.

**Deprecated.** The legacy `smftools.analysis.compute.ml_*` execution entry points are behind
compatibility adapters with a 3.0 removal window. See the
[ML migration guide](../tutorials/ml_migration.md) for the replacement map.

## Scale: two reading modes

The data plane offers a bounded batch reader and a full-split materialiser, and the choice is not
cosmetic. `materialize()` runs a conservative memory preflight and **refuses** above a budget, so
at production scale it is unavailable; `iter_batches()` holds bounded memory at any split size.
Training follows the same split: the streaming fit paths work where the materialising ones refuse.

[Performance and limits](performance.md) covers which workloads are supported, which are refused,
and — importantly — which memory term is not modelled at all.
