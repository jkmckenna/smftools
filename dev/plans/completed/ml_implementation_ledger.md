# smftools ML implementation plan and development ledger

**Plan date:** 2026-07-30

**Repository:** `smftools`

**Program status:** `COMPLETE` — all 30 core work packages are merged and CI-verified. Five remain
open by design: `ML-205` and `ML-304` are `PROPOSED` behind recorded start gates, and
`ML-600`/`ML-601`/`ML-602` are `DEFERRED` pending production triggers.

**Released:** 2.19.0 cut in PR #466 (`8d33859`), CI run `31414597966` green across all eight jobs.
`_version.py` reads `2.19.0`; wheel and sdist built and verified in `dist/`. Not tagged, not
uploaded.

**CI verification:** run `31410967617` on `main` at `264c002` (2026-08-10) passed all eight jobs —
`pytest` on 3.11 and 3.12, `storage-minimums`, `lint`, `format`, `docs`, and `build` on both
Pythons. Merges #460, #463, and #464, which had produced no workflow run, are ancestors of that
commit, so the validation covers them cumulatively.

**Superseded status line:** `IN_PROGRESS` — P0–P5 are complete except ML-304, which remains gated on a
real pretraining corpus. P2 was reopened on 2026-08-05: ML-700 found that no training path uses the
streaming reader, capping training ~15x below a real experiment, so `ML-204` (streaming training)
is now the active next package and is sequenced ahead of ML-700's published limits, ML-701, and
ML-702

**Repository state at planning time:** `12635d5` (detached worktree)

**ML-000 revalidation state:** `b8b5a90` (`main`, after the 2.19.0 development-version bump)

**Source audits:**
[ml_infrastructure_audit.md](ml_infrastructure_audit.md) and
[ml_audit_second_opinion.md](ml_audit_second_opinion.md)

**Reference plan style:**
`dev/plans/completed/semantic_dag_variant_preprocessing_implementation_plan.md`

## Objective

Incrementally establish one backend-neutral, partition-aware, provenance-complete ML system for
smftools without breaking current active consumers or turning optional orchestration tools into the
scientific API.

The completed program must:

- give reusable trainable models, data adapters, execution, inference, explanation computation, and
  ML artifacts one canonical owner under `smftools.machine_learning`;
- preserve the pure result-compute and result-plot contracts under `smftools.analysis`;
- keep HMM fitting task-specific while reusing appropriate immutable-artifact patterns;
- support sklearn and plain-PyTorch models through shared data, split, prediction, metric, and
  artifact contracts;
- stream or boundedly materialize selected partition-store data without eager whole-experiment
  tensorization;
- make biological-group-disjoint splits, train-only transforms, explicit label mappings, and named
  mask semantics enforceable invariants;
- organize experiment-local and project-level outputs through one resolved ML workspace;
- let users declare datasets, samples, labels, splits, balancing, models, and
  train/apply/evaluate/explain/plot actions in one versioned ML plan;
- distinguish reusable pretrained encoders from task-specific fine-tuned model artifacts;
- publish immutable local run/model/prediction/explanation provenance and reproducible result tables;
- migrate current consumers through releasable compatibility checkpoints; and
- keep Lightning, trackers, and Hydra optional and off the critical path.

This is an incremental feature program. Each implementation item is intended to be a focused branch
and PR cut from the then-current `main`. Feature branches must not bump
`src/smftools/_version.py`. No work package may silently change split membership, class semantics,
mask behavior, or the model selected for inference.

The planning worktree is older than the main tree audited in the source reports. Before ML-001
starts, create its branch from current `main` and repeat the inventory against that exact source
state. Do not implement from `12635d5` merely because this ledger is stored here.

## Current baseline

The plan assumes the following audit findings, which ML-001 must revalidate against its
implementation branch:

- reusable ML behavior is split between active `analysis.compute.ml_*` code,
  `smftools.machine_learning`, and downstream project scripts;
- `smftools.machine_learning` contains useful model, sklearn, Lightning, evaluation, inference, and
  attribution ideas but is not connected to a production package workflow;
- current ML smoke tests verify imports rather than split, loader, masking, optimization,
  persistence, inference, or attribution behavior;
- the existing AnnData data module eagerly materializes data and has correctness defects in
  validation/test loader handling;
- partitioned stores already expose stable row identities, metadata planning, bounded row/layer
  projection, and project registry/set selection suitable for a new ML data plane;
- experiment output ownership already resolves through `ExperimentConfig.output_directory`;
- project-derived outputs already belong below `project_outputs/`;
- scikit-learn and Torch are core dependencies, while Captum, SHAP, Lightning, Hydra, and W&B are
  extended optional dependencies;
- HMM, latent, embedding, sidecar, and generation code provide useful precedents for checksums,
  atomic publication, portable references, manifests, and immutable artifacts; and
- no canonical ML plan, dataset/split manifest, run/model manifest, workspace resolver, or
  cross-backend predictor contract currently exists.

This program must reuse proven repository identity, resource, path, logging, optional-import, and
artifact utilities rather than create parallel authorities.

## Program finding IDs

These IDs provide stable references for PR descriptions, tests, and deferment decisions.

| ID | Severity | Finding |
|---|---|---|
| ML-C1 | Critical | Current split/loader behavior can mix or misidentify train, validation, and test data and does not enforce biological-group isolation |
| ML-C2 | Critical | Current ML data access eagerly materializes AnnData and cannot safely scale over partitioned experiment/project stores |
| ML-C3 | Critical | There is no authoritative record joining selected data, split, resolved model/training configuration, code, metrics, and reusable model output |
| ML-H1 | High | Reusable models, training, evaluation, and explanations have duplicate owners under `analysis`, `machine_learning`, and project scripts |
| ML-H2 | High | Signal, missingness, design, padding, attention, corruption, and loss masks lack one explicit compatibility contract |
| ML-H3 | High | Users cannot declaratively select models, samples, labels, splits, balancing, and train/apply/explain/plot actions |
| ML-H4 | High | ML code has no canonical experiment/project output resolver or immutable local artifact layout |
| ML-H5 | High | sklearn and PyTorch implementations lack one predictor, prediction, metric, and capability vocabulary |
| ML-H6 | High | Pretrained encoders, fine-tuned task heads, checkpoints, and promoted inference models lack explicit distinct identities and lineage |
| ML-H7 | High | Interpretability is duplicated and methods are not dispatched or recorded by model capability, baseline, cohort, target, and mask policy |
| ML-H8 | High | ML tests largely establish importability rather than behavioral correctness, leakage prevention, scalability, or persistence |
| ML-M1 | Medium | Architecture definitions need controlled flexibility plus immutable named recipes rather than hard-coded instances or unconstrained graphs |
| ML-M2 | Medium | sklearn persistence needs an explicit version/trust/security policy |
| ML-M3 | Medium | Local metrics, curves, predictions, explanations, model cards, and dataset cards lack one durable organization |
| ML-M4 | Medium | Optional Lightning, tracker, and Hydra dependencies exist without demonstrated adoption gates or stable core contracts |
| ML-M5 | Medium | Active consumers need a staged compatibility and deprecation path before duplicate implementations can be removed |

## How to use this ledger

Use these status values consistently:

| Status | Meaning |
|---|---|
| `PROPOSED` | Scoped in this ledger but not approved for implementation. |
| `READY` | Decisions and dependencies are complete; the package is safe to start. |
| `IN_PROGRESS` | One focused implementation change is actively being developed. |
| `BLOCKED` | Work cannot proceed until a named decision or dependency is resolved. |
| `DONE` | Acceptance criteria are met and validation evidence is recorded. |
| `DEFERRED` | Intentionally postponed; the reason and revisit trigger are recorded. |
| `CANCELLED` | Rejected or superseded; the reason is recorded. |

For each work package:

1. Move it from `PROPOSED` to `READY` only after its decision gates are resolved.
2. Record the branch/PR and owner when work starts.
3. Keep implementation changes small enough to review independently.
4. Add or update tests in the same change as behavior.
5. Record exact validation commands and outcomes before marking it `DONE`.
6. Update dependent packages and the roadmap table when scope changes.
7. Add architectural decisions to the decision log rather than silently changing this plan.

Checkboxes are a quick visual index, while the `Status` field is authoritative:

- `[ ]` not done;
- `[-]` active or blocked; and
- `[x]` done.

## Desired outcome

smftools has one coherent ML system in which:

- `smftools.machine_learning` owns reusable trainable models, data adapters, model execution,
  evaluation adapters, interpretability computation, and ML artifact contracts;
- `smftools.analysis` owns pure result computations and plots, without a second model zoo or
  training engine;
- `smftools.hmm` remains a task-specific production subsystem while sharing lower-level artifact
  design patterns where useful;
- experiment-local ML outputs are resolved below one experiment's `output_directory`;
- cross-experiment ML outputs are resolved below one project's `project_outputs/`;
- a versioned ML plan declares datasets, labels, group-aware splits, balancing, model recipes, and
  train/apply/evaluate/explain/plot jobs;
- sklearn and PyTorch models consume the same logical data/split contracts through separate
  backend adapters;
- pretrained encoders and fine-tuned task models have explicit parent-child lineage;
- masks have named, tested semantics;
- local immutable manifests are the authoritative scientific record; and
- Lightning, W&B/MLflow, and Hydra remain optional orchestration integrations.

## Non-goals

This plan does not:

- turn smftools into a general neural-network graph-construction framework;
- make Lightning, Hydra, W&B, MLflow, SHAP, or Captum mandatory for basic sklearn/PyTorch use;
- move HMM fitting into the generic classifier hierarchy;
- make `project.yaml` a machine-read ML configuration;
- permit random molecule-level splits to stand in for sample-level generalization;
- require every model backend to expose identical internal methods;
- promise automatic loading of arbitrary third-party pickle/joblib artifacts;
- migrate all existing ML code in a single PR; or
- add code merely to match the directory layout shown here before a vertical workflow needs it.

## Proposed design contracts

These contracts constrain the work packages below. Items already marked `ACCEPTED` in the decision
log are the audit-backed default. Open items must be resolved by their named gate. Changing an
accepted contract requires design review and an update to this ledger before implementation
proceeds.

### Ownership boundary

```text
smftools.machine_learning
├── schemas/contracts       input, masks, labels, splits, plans, capabilities
├── workspace/artifacts     output resolution, manifests, immutable publication
├── data                    experiment/project selection and partitioned-store batches
├── models                  plain PyTorch families, sklearn builders, model recipes
├── tasks                   classification/pretraining heads, losses, target semantics
├── training                plain PyTorch and sklearn engines
├── inference               backend adapters and prediction records
├── evaluation              predictor-neutral evaluation orchestration
├── interpretability        model-specific attribution computation
└── orchestration           train/apply/evaluate/explain job services

smftools.analysis
├── compute                 pure metrics, comparisons, aggregation, diagnostics
└── plot                    result + explicit output path -> plot

smftools.cli
└── thin experiment/project command wrappers

smftools.hmm
└── task-specific HMM fitting and inference
```

The tree is conceptual. Work packages should add only the modules needed for a tested vertical
slice. Exact file boundaries are decided during package design review.

### Core contracts are backend-neutral; execution adapters are not

Dataset identity, split membership, label/mask/input schemas, prediction tables, metrics, run/model
manifests, and explanation results are shared scientific contracts. sklearn and PyTorch retain
separate builders, fitted-object adapters, materialization strategies, and serializers. They satisfy
a small predictor/capability protocol through composition rather than inheriting from one model
base class.

Lightning may wrap a plain PyTorch model/task/data workflow later. Hydra may produce a resolved
ordinary ML plan later. A tracker may mirror local events and artifacts later. None is permitted to
become the only API or provenance source.

### Dataset identity and split membership are separate

A dataset snapshot identifies eligible observations, features, source artifacts, filters, labels,
and input schema. A split manifest assigns eligible biological groups to train, validation, and
test. Balancing changes training sampling or loss weighting, never membership in the locked
validation/test roles.

Split planning and fitted preprocessing precede model dispatch so all backends receive the same
scientific cohort definition.

### Experiment modality and biological channel meaning are separate

Projects may register experiments from multiple modalities. The project registry already records
each experiment's modality from its spine and the project catalog can select experiments or named
sets by modality. The ML planner must propagate that experiment-level modality into dataset, split,
prediction, and evaluation records rather than assume it is present on every materialized molecule
row.

An input channel is defined by both its physical source and its declared biological meaning. Site
context alone does not determine meaning:

| Modality | Default physical input | Default biological role |
|---|---|---|
| deaminase | `C_site_binary` | accessibility |
| conversion | GpC binary sites | accessibility |
| conversion | CpG binary sites | endogenous DNA methylation, unless the plan explicitly declares accessibility |
| direct | A, GpC, and/or CpG binary/modification layers | each role must be declared independently |

These are plan defaults, not inference rules inside a model. Every ordered channel specification
records modality applicability, source stage/layer, site context, biological role, coordinate
projection, dtype/transform, and observed/design-mask policy. A checkpoint trained with CpG as
endogenous methylation is incompatible with an equal-shaped input that interprets CpG as
accessibility.

A mixed-modality job must choose one explicit policy:

1. select one modality;
2. harmonize modality-specific physical layers into the same canonical biological channel roles;
   or
3. use a declared union-channel multimodal schema with per-channel availability masks and an
   explicit modality covariate/capability.

Unavailable channels are masked, never encoded as measured zero. Planning must reject ambiguous
roles or silently incompatible channel unions. Evaluation reports class counts and performance by
modality, and the split design must make modality confounding visible.

### Planning is read-only and explainable

Plan validation and dry-run selection may read configs, registries, spines, catalogs, manifests, and
metadata. It performs no model fitting and publishes no completed artifact. It reports resolved
scope, sources, samples, labels, group/class counts, disjointness, estimated matrix/batch memory,
requested model capabilities, optional dependencies, and intended output paths.

Invalid or infeasible work fails before large matrix reads.

### Artifact publication is immutable and local-first

A run is one execution attempt. A checkpoint is resumable backend state within that run. A promoted
model artifact is immutable and suitable for validated reuse. Dataset, split, prediction, and
explanation artifacts have independent versioned schemas and checksums.

Publication uses staging, validation, checksums, and atomic promotion. Tracker records and indexes
are rebuildable mirrors; per-run manifests are authoritative. Loading a pickle/joblib-derived
artifact requires an explicit trust policy.

### Models do not own filesystem placement

One application-layer workspace resolver receives exactly one experiment config or project
directory. It injects resolved run/artifact paths. Model, task, metric, and attribution objects
never search the current working directory, scan parents, reinterpret `project.yaml`, or construct
unscoped output roots.

Project runs may read registered experiment artifacts but never write back into their experiment
trees.

### Interpretability is capability- and question-specific

Native parameters are preferred for Naive Bayes and linear models; TreeSHAP is used for supported
tree ensembles; held-out permutation importance is the broad predictor-neutral fallback; Captum
gradient methods require differentiable neural inputs; and GradCAM requires a declared compatible
convolutional layer. Attention weights are not presented as a general explanation by themselves.

Every explanation identifies the fitted model, dataset/cohort, output/target, feature axes,
background/baseline, mask policy, method implementation/version, and parameters.

### Runtime flow

```text
experiment config OR project registry
                    +
              versioned ML plan
                    |
                    v
       validate and resolve workspace
                    |
                    v
    plan eligible rows and group-disjoint split
                    |
                    v
      publish dataset + split manifests
                    |
         +----------+-----------+
         |                      |
         v                      v
 sklearn materializer      partition batch reader
         |                      |
         v                      v
 sklearn predictor         plain PyTorch predictor
         +----------+-----------+
                    |
                    v
 prediction / metric / explanation records
                    |
                    v
       immutable run and model artifacts
                    |
          +---------+----------+
          |                    |
          v                    v
 pure analysis plots      optional tracker mirror
```

### Output roots

Decision `D-004` fixes the canonical scope roots:

```text
experiment scope:
<ExperimentConfig.output_directory>/ml_outputs/

project scope:
<project_dir>/project_outputs/ml/
```

The application layer resolves one active workspace. Models never search for directories or write
outside the injected run/artifact paths.

## Non-negotiable scientific invariants

Every completed vertical workflow must enforce:

1. Train, validation, and test biological groups are disjoint under the declared grouping columns.
2. Imputation, normalization, feature selection, calibration, thresholds, and backgrounds are fit
   on training data only.
3. Class labels and positive-class semantics are explicit and persisted.
4. Natural validation/test prevalence is preserved for primary evaluation.
5. Signal, observed, design, padding, corruption, attention, and loss masks are not conflated.
6. Dataset and split membership are resolved and checksummed before fitting.
7. Applying a model validates input-schema compatibility before prediction.
8. Predictions identify stable observations, model artifacts, datasets, and split/cohort roles.
9. Explanation baselines/backgrounds cannot be selected from the locked test set.
10. Local manifests remain sufficient to understand a run without access to a hosted tracker.
11. Experiment modality, physical source layer/site context, and biological channel role are
    explicit and persisted.
12. Missing or inapplicable modality channels use declared masks and are never represented as
    measured zero.
13. Mixed-modality evaluation reports per-modality support and performance and checks whether
    modality is confounded with the target label.

## Roadmap and critical path

The critical path is:

```text
P0 decisions
  -> P1 contracts/workspace/artifacts
  -> P2 partitioned data and splits
  -> P3 sklearn baseline
  -> P3 PyTorch baseline
  -> P4 evaluation/interpretability
  -> P5 orchestration and migration
```

Lightning, tracking, and Hydra are off the critical path.

| Phase | Work packages | Exit condition | Status |
|---|---|---|---|
| P0 — decisions and baseline | `ML-000`–`ML-002` | Boundary, compatibility, naming, and test baseline approved. | `DONE` |
| P1 — core contracts | `ML-100`–`ML-105` | Plans, schemas, workspace, and immutable artifact manifests are usable without a trainer. | `DONE` |
| P2 — data plane | `ML-200`–`ML-203` | Experiment/project rows can be split without leakage and read within memory limits. | `DONE` |
| P3 — model backends | `ML-300`–`ML-304` | One sklearn and one plain-PyTorch workflow share contracts and round-trip artifacts. | `DONE` |
| P4 — evaluation and explanation | `ML-400`–`ML-403` | Backend-neutral predictions/metrics and capability-aware explanations are reproducible. | `DONE` |
| P5 — user workflows and migration | `ML-500`–`ML-504` | Users can plan/train/apply/evaluate/explain/plot and legacy duplication is retired safely. | `DONE` |
| P6 — optional scale integrations | `ML-600`–`ML-602` | Optional tools add value without changing core scientific semantics. | `DEFERRED` |
| P7 — stabilization | `ML-700`–`ML-702` | Documentation, performance evidence, security review, and release migration are complete. | `IN_PROGRESS` |

**P2 reopened 2026-08-05.** `ML-204` was added after ML-700 found that no training path consumes
the streaming reader, capping training at ~85,011 rows — 15x short of a real experiment. P2's exit
condition ("read within memory limits") was met for *reads*; it was never true for *training
reads*. ML-204 is sequenced ahead of ML-701/ML-702 so the documentation and release review describe
a system that runs at lab scale.

### Backward-compatible rollout checkpoints

Every checkpoint must be releasable without unfinished later work:

1. **After ML-105:** typed schemas, workspace resolution, and immutable artifact primitives exist,
   but no current training entry point changes.
2. **After ML-203:** users/developers can dry-run dataset selection and group-aware splits against
   partition stores; existing trainers remain unchanged.
3. **After ML-301:** a package-internal sklearn vertical path can train/apply/publish artifacts;
   no legacy model path is removed.
4. **After ML-303:** one canonical plain-PyTorch path exists beside sklearn and produces the same
   prediction/metric contracts; Lightning is still unnecessary.
5. **After ML-403:** explanations are immutable, capability-dispatched outputs; requesting
   explanation remains optional and cannot affect the fitted model.
6. **After ML-501/ML-502:** an approved Python/CLI surface and reproducible plots are available,
   while compatibility wrappers preserve current consumers.
7. **After ML-503:** at least one real consumer uses the canonical path and has before/after
   acceptance evidence.
8. **After ML-504:** duplicate training/model code is deprecated or removed according to the
   approved compatibility window.
9. **ML-600 through ML-602:** each optional integration is independently releasable and may remain
   deferred indefinitely.

Every PR that writes a new persistent schema must add its reader, validator, version constant, and
migration/rejection behavior in the same change. Performance-sensitive changes must include a
bounded-memory test or benchmark fixture.

## Ordered core PR backlog

| ID | Suggested branch | Primary outcome | Finding coverage | Depends on | Status |
|---|---|---|---|---|---|
| ML-000 | `feature/ml-contract-decisions` | Resolve blocking contracts and first vertical use case | All | — | `DONE` |
| ML-001 | `feature/ml-behavior-inventory` | Current-source symbol/consumer/artifact migration map | ML-H1, ML-H8, ML-M5 | ML-000 | `DONE` |
| ML-002 | `test/ml-behavior-baseline` | Behavioral fixtures and regression characterization | ML-C1, ML-H2, ML-H8 | ML-001 | `DONE` |
| ML-100 | `feature/ml-plan-schema` | Typed versioned user ML plan | ML-H3 | ML-000 | `DONE` |
| ML-101 | `feature/ml-input-contracts` | Input, mask, label, and capability schemas | ML-H2, ML-H5 | ML-000 | `DONE` |
| ML-102 | `feature/ml-dataset-split-manifests` | Dataset identity and split provenance | ML-C1, ML-C3 | ML-101 | `DONE` |
| ML-103 | `feature/ml-workspace-resolution` | Canonical experiment/project output ownership | ML-H4 | ML-100 | `DONE` |
| ML-104 | `feature/ml-artifact-schemas` | Run/model/checkpoint/prediction/explanation schemas | ML-C3, ML-H6, ML-H7, ML-M2, ML-M3 | ML-101, ML-102, ML-103 | `DONE` |
| ML-105 | `feature/ml-artifact-publication` | Immutable publication and rebuildable indexes | ML-C3, ML-H4, ML-M3 | ML-104 | `DONE` |
| ML-200 | `feature/ml-data-selection-plan` | Metadata-first experiment/project selection | ML-C2, ML-H3 | ML-102, ML-103 | `DONE` |
| ML-201 | `feature/ml-group-splits` | Explicit and generated group-disjoint splits | ML-C1 | ML-200 | `DONE` |
| ML-202 | `feature/ml-partition-dataset` | Bounded partition-aware batches/materialization | ML-C2 | ML-101, ML-201 | `DONE` |
| ML-203 | `feature/ml-train-transforms-balancing` | Train-only transforms and balancing | ML-C1, ML-H2 | ML-201, ML-202 | `DONE` |
| ML-204 | `feature/ml-streaming-training` | Training fits driven from streamed batches | ML-C2 | ML-202, ML-203, ML-301, ML-303 | `DONE` |
| ML-205 | `feature/ml-activation-memory` | Torch activation-memory estimate and preflight | ML-C2 | ML-204, ML-303 | `PROPOSED` |
| ML-300 | `feature/ml-predictor-registry` | Predictor protocol, capabilities, explicit registry | ML-H5 | ML-101, ML-104 | `DONE` |
| ML-301 | `feature/ml-sklearn-vertical` | NB/logistic/RF training, application, and artifacts | ML-H5, ML-M2 | ML-203, ML-300, ML-105 | `DONE` |
| ML-302 | `feature/ml-torch-model-families` | Canonical configurable Torch families and recipes | ML-H1, ML-M1 | ML-101, ML-300 | `DONE` |
| ML-303 | `feature/ml-torch-vertical` | Plain-PyTorch train/apply/artifact workflow | ML-C1, ML-H1 | ML-202, ML-203, ML-302, ML-105 | `DONE` |
| ML-304 | `feature/ml-pretrained-lineage` | Encoder/pretraining/fine-tuned-head lineage | ML-H6 | ML-303 | `PROPOSED` |
| ML-400 | `feature/ml-evaluation-contract` | Shared predictions, metrics, curves, and folds | ML-C3, ML-H5, ML-M3 | ML-301, ML-303 | `DONE` |
| ML-401 | `feature/ml-explanation-contract` | Explanation request/result and artifact schemas | ML-H7, ML-M3 | ML-104, ML-400 | `DONE` |
| ML-402 | `feature/ml-classical-explanations` | NB/linear/permutation/TreeSHAP adapters | ML-H7 | ML-301, ML-401 | `DONE` |
| ML-403 | `feature/ml-neural-explanations` | Captum and validated attention adapters | ML-H7 | ML-303, ML-401 | `DONE` |
| ML-500 | `feature/ml-job-services` | Backend-neutral plan/train/apply/evaluate/explain services | ML-H3, ML-H4 | ML-301, ML-303, ML-400 | `DONE` |
| ML-501 | `feature/ml-user-orchestration` | Dry-run and approved Python/Click surface | ML-H3 | ML-500 | `DONE` |
| ML-502 | `feature/ml-analysis-results` | Pure metric summaries and reproducible plots | ML-H1, ML-M3 | ML-400, ML-401 | `DONE` |
| ML-503 | `feature/ml-consumer-migration` | One real consumer migrated with parity evidence | ML-H1, ML-M5 | ML-500, ML-502 | `DONE` |
| ML-504 | `fix/ml-legacy-convergence` | Compatibility, deprecation, and duplicate removal | ML-H1, ML-M5 | ML-503 | `DONE` |
| ML-600 | `feature/ml-lightning-adapter` | Optional Lightning execution adapter | ML-M4 | ML-303 + trigger | `DEFERRED` |
| ML-601 | `feature/ml-tracker-adapters` | Optional W&B/MLflow/local tracker mirrors | ML-M4 | ML-105, ML-400 + trigger | `DEFERRED` |
| ML-602 | `feature/ml-hydra-application` | Optional Hydra composition/multirun layer | ML-M4 | ML-100, ML-501 + trigger | `DEFERRED` |
| ML-700 | `test/ml-scale-qualification` | Performance, memory, and scalability evidence | ML-C2 | ML-301, ML-303, ML-500 | `DONE` |
| ML-701 | `feature/ml-documentation` | User, API, artifact, and migration documentation | ML-M3, ML-M5 | Public work packages, ML-204 | `DONE` |
| ML-702 | `feature/ml-program-acceptance` | Security, compatibility, migration, and release acceptance | ML-M2, ML-M5 | ML-204, ML-504, ML-700, ML-701 | `DONE` |

## Suggested primary file map

This map makes likely ownership visible before implementation. Exact new module names may be
adjusted during ML-000/ML-001, but changes should remain within these responsibility boundaries.
Do not create files merely to realize this diagram; add them when a vertical slice needs them.

| Work package | Suggested primary files or areas |
|---|---|
| ML-000–ML-002 | this ledger, both audits, current public modules/consumers, new focused fixtures under `tests/unit/machine_learning/` and `tests/integration/machine_learning/` |
| ML-100 | new `src/smftools/machine_learning/plan.py` or `plan/`; schema-version constants; plan fixtures |
| ML-101 | new `src/smftools/machine_learning/schemas/` or focused `contracts.py`; model capability definitions |
| ML-102 | new dataset/split manifest modules under `machine_learning/data/` or `artifacts/`; Parquet/JSON fixtures |
| ML-103 | new `src/smftools/machine_learning/workspace.py`; `src/smftools/cli/helpers.py`; project path helpers; constants |
| ML-104–ML-105 | new `src/smftools/machine_learning/artifacts/`; proven helpers from `readwrite.py`, HMM artifacts, latent/embedding stores |
| ML-200–ML-202 | `machine_learning/data/`; `informatics/partition_read.py`, partition/catalog query APIs, project registry/catalog/set APIs |
| ML-203 | `machine_learning/data/transforms.py`, balancing policy module, sklearn pipeline builders |
| ML-300 | `machine_learning/models/protocols.py`, explicit registry/spec module, backend adapters |
| ML-204 | `machine_learning/training/sklearn_backend.py`, `training/torch_backend.py`, `data/transforms.py` (streaming accumulators), `data/balancing.py` (metadata-only resolution) |
| ML-205 | activation-memory estimator beside `machine_learning/data/partition_dataset.py`; Torch preflight in `training/torch_backend.py`; `tests/acceptance/ml_scale_thresholds.json` |
| ML-301 | `machine_learning/models/sklearn_models.py`, `training/train_sklearn_model.py`, sklearn inference adapter |
| ML-302 | `machine_learning/models/` plus selected active model/config code migrated from `analysis.compute.ml_cnn` |
| ML-303 | `machine_learning/training/torch_engine.py`, tasks/losses, Torch inference adapter |
| ML-304 | `machine_learning/models/encoders.py`, task heads, pretraining/fine-tuning services and lineage manifests |
| ML-400 | `machine_learning/evaluation/` for orchestration/records; pure metric kernels under `analysis.compute` where appropriate |
| ML-401–ML-403 | new `machine_learning/interpretability/`; optional Captum/SHAP imports; explanation artifact modules |
| ML-500 | new `machine_learning/orchestration/` or focused job-service modules; package logging utilities |
| ML-501 | Python public exports; if approved, thin wrappers in `cli_entry.py`, `cli/`, and `cli/project_cmd.py` |
| ML-502 | `analysis/compute/ml_*`, `analysis/plot/ml.py`, their `__init__.py` inventories, focused analysis tests |
| ML-503–ML-504 | exact files identified by ML-001; compatibility adapters, exports, deprecation tests, downstream consumer plan/scripts |
| ML-600 | Lightning adapter modules only; no changes to plain model inheritance |
| ML-601 | tracker-neutral events plus optional backend adapters; privacy/redaction tests |
| ML-602 | dedicated training-application Hydra config groups that resolve to ML-100 objects |
| ML-700 | benchmarks/fixtures with documented execution policy; no mandatory slow benchmark in normal unit collection |
| ML-701 | `docs/source/ml/` (index, architecture, performance, quick starts, guidance), `docs/source/cli.md`, tutorials, curated API autosummary page, schema references, migration/release notes |
| ML-702 | `pyproject.toml`, optional imports/dependency profiles, security/relocation/compatibility tests, release documentation |

Before editing under `analysis`, `cli`, `docs`, or `tests`, re-read that subtree's `AGENTS.md`.
Agents must not edit or create `AGENTS.md`/`CLAUDE.md` files.

## P0 — decisions and baseline

### ML-000 — Record architectural decisions

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: none
- Purpose: convert the audit's recommended direction into explicit accepted or rejected decisions.
- Deliverables:

  - resolve every decision in the decision log that blocks P1;
  - record public namespace and compatibility policy;
  - confirm the initial vertical use case, labels, source layer, and grouping unit;
  - confirm whether the first CLI surface is experiment-scoped, project-scoped, or Python-only; and
  - agree on PR sizing and review ownership.

- Acceptance:

  - no P1 item has an unresolved blocking decision;
  - rejected alternatives and rationale are recorded; and
  - one representative small fixture/workflow is named for end-to-end acceptance.

- Resolved first vertical use case:

  - **scope:** a project-scoped, Python-driven binary classification workflow over registered
    **deaminase** experiment partitions;
  - **input:** the deaminase default, preprocessing-stage `C_site_binary` as one accessibility
    channel projected to one declared reference interval, with signal and observed-mask semantics
    kept separate;
  - **label:** a user-selected `obs` column; the acceptance fixture uses `activity_status` with the
    explicit mapping `inactive: 0`, `active: 1` and does not make that column or vocabulary a
    package-wide default;
  - **split unit:** the composite biological group `(experiment_uid, Sample)`, with disjoint train,
    validation, and test groups and immutable molecule membership;
  - **first backend:** Bernoulli Naive Bayes in a fitted sklearn pipeline, followed by the committed
    residual-dilated 1D CNN recipe using the same dataset, split, prediction, metric, and artifact
    contracts;
  - **balancing:** training-only class weighting or sampling; validation and test retain natural
    prevalence; and
  - **interface:** internal/public Python services first. Click commands are designed only after the
    vertical workflow and dry-run contract are stable.

- Current-main revalidation evidence:

  - `b8b5a90` contains no ML source changes after the audit's committed baseline `4e1b1e5`; the
    intervening commits are the 2.18.0 release and 2.19.0 development-version transition.
  - The audit's Transformer, CNN-Transformer hybrid, GradientSHAP, and differential-abundance
    descriptions came from uncommitted files in a different worktree. They are not part of current
    `main` and therefore are migration candidates to inventory, not current package behavior to
    preserve.
  - Current `analysis.compute.ml_cnn` owns one configurable residual-dilated CNN, its plain-PyTorch
    trainer, input construction, and Integrated Gradients. `analysis.compute.ml_metrics` owns
    Bernoulli NB, random forest, and optional XGBoost pipelines.
  - `smftools.machine_learning` still contains the separate AnnData/Lightning/sklearn framework and
    remains disconnected from production consumers. The top-level `smftools.ml` lazy alias still
    resolves to it.
  - Experiment output ownership still comes from `ExperimentConfig.output_directory`; project
    derived outputs still belong below `project_outputs/`; `project.yaml` remains human-only.
  - Partition readers still provide metadata-first selection, stable `experiment_uid` and
    `molecule_uid`, projected layers/intervals, query memory limits, and streamed project set
    members.
  - HMM, latent, registry, and project-store code still provide canonical JSON, checksums, atomic
    writes, immutable identities, and portable relative-path precedents.
  - sklearn and Torch remain core dependencies. Captum, SHAP, Lightning, Hydra, and W&B remain in
    the optional `ml-extended` dependency group.

- PR and review policy:

  - one ledger work package per branch/PR by default;
  - a persistent-schema PR must include its schema version, writer, reader, validator, rejection or
    migration behavior, and focused tests together;
  - the repository maintainer approves scientific semantics and public compatibility;
  - the implementation author owns code, tests, validation evidence, and ledger updates; and
  - changes crossing `analysis`, CLI, docs, or tests must follow those subtrees' local contracts and
    may be split when they cannot remain independently reviewable.

- Evidence/notes: decisions D-004, D-006, D-007, D-010, D-014, D-015, D-016, D-017, and
  D-023–D-025 below record the accepted ML-000 direction. The local decision gate was reviewed
  interactively on 2026-07-30; ML-001 is ready.

### ML-001 — Inventory behavior and migration surface

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-000
- Purpose: produce a current, line-level migration map before moving or replacing implementations.
- Scope:

  - `smftools.machine_learning`;
  - active `analysis.compute.ml_*` and `analysis.plot.ml`;
  - project scripts that call those APIs, if available;
  - HMM/latent/embedding artifact primitives worth reusing;
  - optional dependency import behavior; and
  - existing public/lazy exports.

- Deliverables:

  - table of public and project-consumed symbols;
  - classification of each symbol as keep, adapt, migrate, deprecate, or delete;
  - saved artifact compatibility inventory;
  - current input/output shape and label behavior;
  - identified correctness defects with regression-test targets; and
  - deprecation risk assessment.

- Acceptance:

  - every live ML symbol has an intended destination;
  - no project-consumed interface is removed without a compatibility plan; and
  - archived directories remain excluded.

- Evidence/notes: [ml_behavior_inventory.md](ml_behavior_inventory.md) inventories the committed
  `b8b5a90` symbols/consumers, shapes and label behavior, artifact compatibility, migration
  dispositions, twelve regression targets, and separately scoped external/uncommitted candidates.
  No archived code was included and no committed production/CLI consumer was found.

### ML-002 — Establish behavioral test baseline

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-001
- Purpose: replace importability as the only evidence of ML correctness.
- Deliverables:

  - deterministic tiny binary and multiclass fixtures;
  - tiny partitioned experiment fixture with multiple samples/groups;
  - current-behavior tests where backward compatibility is required;
  - expected-failure or defect-characterization tests for known bugs; and
  - marker/runtime plan for unit, integration, and end-to-end coverage.

- Required initial assertions:

  - loader identity and train/validation/test disjointness;
  - input/mask shape contracts;
  - label ordering persistence;
  - a single optimization step;
  - checkpoint/model round-trip prediction equality;
  - validation/test transforms are not fit;
  - partition reads stay within requested rows/columns; and
  - optional imports fail with actionable extra-install messages.

- Acceptance:

  - tests fail for the known incorrect loader/split paths they target;
  - fixtures contain at least three biological groups;
  - tests do not depend on external data or network access; and
  - expected runtime and markers are documented.

- Evidence/notes: local branch `test/ml-behavior-baseline` adds two unit modules and one
  integration module. Validation on 2026-07-30:

  - targeted pytest: `9 passed, 7 xfailed` in 3.19 seconds;
  - full unit marker suite: `1236 passed, 9 skipped, 106 deselected, 7 xfailed` in
    205.58 seconds with normal multiprocessing permissions;
  - Ruff check: passed; and
  - Ruff format check: passed.

  The seven strict expected failures cover B-001–B-004, B-007, B-008, and B-010. Passing tests
  cover three-group binary and multiclass fixtures, group-disjoint folds, modality-relevant
  signal/observed/design-mask shapes, one plain-PyTorch optimization step, state-dict prediction
  round trip, train-only sklearn imputer state, bounded partition row/layer/position projection,
  and actionable optional-import errors. Mark `DONE` after the focused PR merges.

## P1 — core contracts, workspace, and artifacts

### ML-100 — Define typed/versioned ML plan

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-000
- Purpose: provide one validated user declaration for datasets, labels, splits, balancing, models,
  and jobs.
- Proposed top-level sections:

  - `schema_version`;
  - `scope`;
  - `datasets`;
  - `splits`;
  - `balancing`;
  - `models`;
  - `jobs`; and
  - optional `tracking`.

- Deliverables:

  - ordinary typed Python representation independent of Hydra/OmegaConf;
  - YAML/JSON loader with explicit precedence;
  - dataset declarations that select one or more registered modalities and define ordered physical
    source-to-biological-role channel mappings;
  - reference validation between named objects;
  - canonical resolved serialization;
  - actionable validation errors;
  - schema migration/version policy; and
  - fixtures for train, apply, evaluate, explain, and plot plans.

- Acceptance:

  - invalid references, unknown fields, duplicate names, incompatible actions, and unsupported schema
    versions fail before matrix reads;
  - resolved defaults are serializable and hash-stable;
  - the plan is not embedded wholesale in the experiment CSV;
  - `project.yaml` is not read as the ML plan; and
  - core objects do not import Hydra/OmegaConf.

- Evidence/notes: local branch `feature/ml-plan-schema` adds an immutable stdlib-dataclass plan
  representation and strict YAML/JSON loaders under `smftools.machine_learning.plan`. The
  implementation provides:

  - explicit `defaults < file < overrides` precedence and recursive mapping/replacement semantics;
  - strict unknown-field, duplicate-key/name, reference, action, and schema-version validation;
  - project/experiment scope without resolving filesystem paths;
  - stable experiment/sample selections, explicit labels, group splits, training-only balancing,
    sklearn/Torch model declarations, and train/apply/evaluate/explain/plot jobs;
  - deaminase `C_site_binary` accessibility and conversion GpC-accessibility/CpG-endogenous
    methylation defaults;
  - explicit direct-SMF channel declarations plus harmonized/union mixed-modality policies; and
  - resolved round-trip serialization and SHA-256 plan identity without Hydra/OmegaConf imports.

  Validation on 2026-07-30:

  - focused ML-100 tests: `17 passed` in 0.05 seconds;
  - new plus legacy ML unit tests: `25 passed, 7 xfailed` in 4.60 seconds;
  - full unit marker suite: `1253 passed, 9 skipped, 106 deselected, 7 xfailed` in
    202.81 seconds with normal multiprocessing permissions; and
  - Ruff check and format check: passed.

  ML-101 retains ownership of detailed mask/input capability schemas, ML-103 retains workspace
  path resolution, and this work package remains `IN_PROGRESS` until its focused PR merges.

### ML-101 — Define input, mask, label, and capability schemas

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-000
- Purpose: make data and model compatibility explicit before model consolidation.
- Deliverables:

  - versioned input schema with ordered feature/channel names, experiment modality applicability,
    physical source stage/layer, site context, biological role, coordinate system, dtype, shape,
    reference, and transform identity;
  - named per-channel mask structures for observed, availability/design, padding, attention,
    corruption, and loss masks;
  - label schema with source field, exact value-to-class mapping, class order, positive class,
    missing/unknown behavior, and task type;
  - predictor capability flags such as probability output, incremental fit, sample weights,
    position masks, gradients, convolutional layers, and attention data; and
  - compatibility validation and canonical JSON.

- Acceptance:

  - every mask has documented shape, polarity, and consumer;
  - deaminase, conversion, and direct-SMF channel defaults round-trip with explicit biological
    roles;
  - equal-shaped schemas with different channel order or CpG biological meaning are incompatible;
  - an unavailable modality channel cannot be represented as a measured zero;
  - labels never rely on transient pandas categorical codes;
  - incompatible checkpoint/input schemas fail before execution;
  - unused masks cannot be silently accepted by a backend; and
  - round-trip serialization tests cover schema versions.

- Evidence/notes: local branch `feature/ml-input-contracts` adds immutable versioned contracts under
  `smftools.machine_learning.contracts` for:

  - ordered input channels resolved from ML-100 declarations, including modality-specific physical
    sources, biological roles, reference coordinates, shape, canonical dtype, and transform ID;
  - seven distinct boolean masks—observed, availability, design, padding, attention, corruption,
    and loss—with fixed axes, true polarity, consumers, and allowed execution phases;
  - explicit binary/multiclass label vocabularies with contiguous class IDs, class order, positive
    class, and missing/unknown behavior independent of pandas categorical codes;
  - backend-neutral predictor capabilities for probabilities, incremental fit, sample weights,
    position masks, gradients, convolutional layers, attention data, and supported/required masks;
  - exact schema hashes and actionable input-compatibility failures; and
  - mask shape, boolean dtype, availability/observation, corruption/observation, and
    padding/attention relationship validation.

  The schema consumes ML-100 dataset/channel/label declarations without reparsing ML plans. It
  performs no matrix reads, model adaptation, training, or workspace resolution. Validation on
  2026-07-30:

  - focused ML-101 tests: `21 passed` in approximately 0.05 seconds;
  - combined ML-100/ML-101/legacy ML tests: `46 passed, 7 xfailed` in 9.47 seconds;
  - full unit marker suite: `1274 passed, 9 skipped, 106 deselected, 7 xfailed` in
    202.97 seconds with normal multiprocessing permissions;
  - Sphinx warning-as-error HTML build: passed after enabling network access for intersphinx;
  - generated Sphinx source artifacts were removed after validation; and
  - Ruff check and format check: passed.

  Merged to `main` in PR #435 (`95b654e`).

### ML-102 — Define dataset and split manifests

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-101
- Purpose: separate eligible data identity from train/validation/test role assignment.
- Deliverables:

  - dataset snapshot manifest;
  - split manifest stored as stable row/group membership plus summary;
  - source experiment/project/set, experiment modality, and stage identities;
  - selected samples, references, intervals, physical layers, resolved biological channels,
    filters, and labels;
  - membership digests and counts by class/group/split/modality; and
  - portable references to source catalogs/spines.

- Acceptance:

  - identical resolved membership produces the same snapshot/split digest;
  - changing selection or grouping changes the appropriate digest;
  - manifests can be inspected without loading matrices;
  - train/validation/test membership is auditable at stable observation and group levels; and
  - every selected experiment has a known modality and an unambiguous resolved channel schema;
  - stale or changed source artifacts are detected.

- Evidence/notes:

  Added `machine_learning.manifests` as a path-neutral contract layer with:

  - immutable, versioned dataset snapshots and split manifests with strict readers;
  - portable relative source-artifact references and source-generation/fingerprint identities;
  - stable experiment/read-derived molecule identities and explicit experiment modalities;
  - resolved selection, input schema, label schema, filters, interval, sample, and reference
    provenance;
  - order-independent source, dataset-membership, split-membership, snapshot, and split digests;
  - dataset and per-split sample/class/modality/group counts;
  - exact split coverage and biological-group leakage validation; and
  - stale-source and serialized-content tamper detection.

  The layer consumes ML-101 schemas but performs no matrix reads, filesystem resolution, split
  generation, balancing, or model work. Validation on 2026-07-30:

  - focused ML-102/ML-101/ML-100 tests: `52 passed`;
  - all machine-learning unit tests: `60 passed, 7 xfailed`;
  - full unit marker suite: `1288 passed, 9 skipped, 106 deselected, 7 xfailed` in
    204.33 seconds with normal multiprocessing permissions;
  - Sphinx warning-as-error HTML build: passed after enabling network access for intersphinx;
  - generated Sphinx source/build artifacts were removed after validation; and
  - Ruff check, format, and `git diff --check`: passed.

  Merged to `main` in PR #438 (`e4dfed7`).

  Merged to `main` in PR #436 (`d80830f`).

### ML-103 — Resolve experiment/project ML workspaces

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-100
- Purpose: ensure all outputs have one deterministic owner and root.
- Deliverables:

  - `MLWorkspace`/equivalent value object;
  - experiment resolver using `ExperimentConfig.output_directory`;
  - project resolver using `<project_dir>/project_outputs/ml`;
  - one canonical proposed experiment ML directory name;
  - run path bundle with no model-owned path concatenation;
  - containment/path-traversal checks; and
  - workspace identity recorded in resolved plans/manifests.

- Acceptance:

  - caller supplies exactly one experiment or project scope;
  - multi-experiment selection is rejected in experiment scope;
  - project jobs never write into registered experiment trees;
  - moving a complete experiment/project preserves relative artifact references; and
  - all job types report intended paths in dry-run mode.

- Evidence/notes:

  Added `machine_learning.workspace` and canonical ML directory constants with:

  - one read-only resolver requiring exactly one experiment config or initialized project;
  - experiment ownership at `<ExperimentConfig.output_directory>/ml_outputs`;
  - project ownership at `<project_dir>/project_outputs/ml`;
  - path-neutral workspace IDs derived from scope kind and stable caller/owner identity;
  - deterministic dataset, run, model, and rebuildable-index roots;
  - a run path bundle for manifests, plans, environment, history, metrics, predictions, plots,
    explanations, checkpoints, and logs;
  - intended-path dry-run reports for train/apply/evaluate/explain/plot jobs;
  - portable workspace-relative serialization/resolution; and
  - strict component, containment, traversal, cross-scope, initialized-project, and
    multi-experiment experiment-scope checks.

  Resolution does not create directories or publish artifacts. Validation on 2026-07-30:

  - focused ML-103 tests: `23 passed`;
  - all machine-learning unit tests: `83 passed, 7 xfailed`;
  - full unit marker suite: `1311 passed, 9 skipped, 106 deselected, 7 xfailed` in
    203.48 seconds with normal multiprocessing permissions;
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx;
  - generated Sphinx source/build artifacts were removed after validation; and
  - Ruff check, format, and `git diff --check`: passed.

  Merged to `main` in PR #437 (`606c653`).

### ML-104 — Define run, model, checkpoint, prediction, and explanation manifests

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-101, ML-102, ML-103
- Purpose: establish tracker-neutral scientific provenance.
- Deliverables:

  - versioned manifest schemas;
  - distinct execution `run_id`, semantic model key, immutable `model_id`, and content checksums;
  - run state lifecycle (`planned`, `running`, `completed`, `failed`, `cancelled`);
  - model parent/child lineage;
  - environment, dependency, code revision, dirty-tree, seed, and device fields;
  - resolved plan/config references;
  - prediction and explanation table identities; and
  - trust/loader metadata for serialized sklearn artifacts.

- Acceptance:

  - repeated identical training attempts have distinct run IDs;
  - weights/checkpoint digests identify actual bytes;
  - model identity includes input schema, dataset, split, architecture, and parent lineage;
  - failure manifests retain enough context to diagnose interrupted work;
  - a promoted model can be understood without its originating tracker; and
  - schemas reject missing required provenance.

- Evidence/notes:

  - added strict, immutable, versioned run, model, checkpoint, prediction, and explanation
    manifests under `smftools.machine_learning.artifacts`;
  - separated unique execution run IDs, semantic model keys, content-addressed model/checkpoint
    IDs, and checksums of the serialized artifact bytes;
  - captured run lifecycle and failure context, workspace/plan/config/data/split identities,
    reproducibility environment, seeds, device, model lineage, prediction cohorts, and
    explanation target/baseline/mask semantics;
  - added explicit serialization policy metadata, including unsafe pickle/joblib gating and
    reviewed-type allowlists for `skops`;
  - extended deterministic run paths with `resolved_config.json`;
  - focused artifact tests: `19 passed`;
  - all machine-learning tests: `102 passed, 7 xfailed`;
  - full unit marker suite: `1330 passed, 9 skipped, 106 deselected, 7 xfailed` in
    203.38 seconds;
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx;
  - generated Sphinx source/build artifacts were removed after validation; and
  - Ruff check, format, and `git diff --check`: passed.

### ML-105 — Implement immutable artifact publication and rebuildable indexes

- Checkbox: `x`
- Status: `DONE`
- Depends on: ML-104
- Purpose: publish artifacts safely under the resolved workspace.
- Deliverables:

  - staging and atomic publication;
  - checksummed file inventory;
  - conflict detection and safe retry behavior;
  - run/model index rebuild from authoritative manifests;
  - portable relative references;
  - optional current/promoted aliases that cannot mutate immutable content; and
  - cleanup rules for abandoned staging data.

- Acceptance:

  - concurrent publication cannot silently overwrite a model;
  - interrupted publication leaves no apparently complete artifact;
  - index deletion and rebuild preserve authoritative records;
  - checksum mismatch is detected;
  - absolute external paths are not required after moving a complete workspace; and
  - tests reuse appropriate HMM/latent artifact patterns without copying their domain-specific keys.

- Evidence/notes:

  - added exact checksummed inventories for complete immutable run and reusable-model bundles;
  - publication copies declared sources into transaction-specific staging, verifies source and
    staged bytes, validates the manifest and complete inventory, and atomically renames the
    finished directory into its authoritative location;
  - per-identity publication locks and post-lock conflict checks make identical concurrent
    retries idempotent while rejecting rebinding or corrupted existing content;
  - child checkpoint, prediction, and explanation files remain within the complete run inventory
    rather than mutating a completed run after publication;
  - run/model indexes rebuild deterministically from validated authoritative manifests and retain
    only portable workspace-relative manifest references;
  - mutable named model aliases validate their target manifest checksum and can be repointed
    without modifying either immutable model bundle;
  - staging cleanup is threshold-bounded, scope-contained, symlink-safe, and also reclaims stale
    publication lock files;
  - focused publication tests: `14 passed`;
  - all machine-learning tests: `116 passed, 7 xfailed`;
  - full unit marker suite: `1344 passed, 9 skipped, 106 deselected, 7 xfailed` in
    204.60 seconds with normal multiprocessing permissions;
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx;
  - generated Sphinx source/build artifacts were removed from the worktree after validation; and
  - Ruff check, format, and `git diff --check`: passed.

  Merged to `main` in PR #439 (`d31f94c`).

## P2 — data plane and leakage prevention

### ML-200 — Plan eligible data from experiment/project metadata

- Checkbox: `x`
- Status: `DONE`
- Depends on: ML-102, ML-103
- Purpose: resolve data from spines/catalogs before touching large matrices.
- Deliverables:

  - experiment source resolver by modality/stage/layer/reference/sample/filter;
  - project source resolver through registry and named sets with modality propagation;
  - resolver from modality-specific physical layers/site contexts to declared biological channels;
  - stable molecule/group identity table;
  - capability to report selected row/feature counts and estimated materialization size;
  - source fingerprinting; and
  - dry-run selection summary.

- Acceptance:

  - metadata planning performs no eager full-matrix load;
  - source stage and layer ambiguity fails clearly;
  - selected experiment modality and sample/reference identities are preserved;
  - unknown modality, ambiguous CpG meaning, or incompatible mixed-modality channel mapping fails
    before matrix reads;
  - project selection works across registered experiments without copying source data; and
  - changed catalogs or source membership invalidate stale plans.

- Evidence/notes: local branch `feature/ml-data-selection-plan` adds a metadata-only selection
  planner under `smftools.machine_learning.selection`. The implementation:

  - resolves experiment manifests or project registry/named-set entries without opening H5AD/Zarr
    feature matrices or copying project data;
  - binds registered experiment modality and canonical/physical reference identity to explicit
    modality-specific channel sources, including current immutable preprocess generations;
  - selects stable molecule/sample/group/label metadata through Parquet molecule and stage indexes;
  - validates layer availability and deaminase/conversion/CpG biological semantics before
    materialization;
  - reports observation, feature, class, modality, and sample counts plus a conservative
    materialization-size estimate and explainable dry-run summary; and
  - fingerprints selected membership, task/interval catalogs, physical channel mappings, and source
    metadata so changed membership or feature catalogs change the selection identity.

  Validation on 2026-07-30:

  - focused ML-200 tests: `6 passed`;
  - all machine-learning tests: `122 passed, 7 xfailed`;
  - full unit marker suite: `1350 passed, 9 skipped, 106 deselected, 7 xfailed` in
    203.80 seconds with normal multiprocessing permissions;
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx;
  - generated Sphinx source/build artifacts were moved out of the worktree after validation; and
  - Ruff check, format, and `git diff --check`: passed.

  Merged to `main` in PR #440 (`15bcc01`).

### ML-201 — Build and validate group-aware splits

- Checkbox: `x`
- Status: `DONE`
- Depends on: ML-200
- Purpose: make biological grouping, not molecule rows, the default split unit.
- Deliverables:

  - explicit group-list splits;
  - seeded stratified-group holdout/fold strategies;
  - leave-one-group-out support where appropriate;
  - disjointness and class-by-modality feasibility validation;
  - split summaries with class/group/modality counts; and
  - immutable split manifest publication.

- Acceptance:

  - no grouping key appears in more than one split;
  - impossible class/group stratification fails rather than degrading silently;
  - deterministic seeds reproduce membership;
  - explicit sample assignments are preserved exactly;
  - locked test membership cannot be changed by balancing; and
  - modality/label confounding and absent class-by-modality cells are reported rather than hidden;
  - split tests cover experiment and project scopes.

- Evidence/notes: local branch `feature/ml-group-splits` adds group-aware split resolution under
  `smftools.machine_learning.splitting` and extends the plan schema with
  `leave_one_group_out`. The implementation:

  - preserves explicit train/validation/test group lists exactly, including user-facing
    `experiment_id/sample` tokens backed by stable experiment/group identities;
  - resolves seeded stratified-group train/validation/test assignments with requested fractions,
    deterministic identities, and bounded linear-per-attempt metadata scoring;
  - produces deterministic leave-one-group-out train/test folds;
  - rejects missing labels, incomplete group coverage, biological-group leakage, single-class
    roles, and class support that makes stratification impossible;
  - reports observation/group/class/modality counts and every class-by-modality cell, including
    zero-support cells and explicit confounding warnings;
  - marks validation/test roles as locked and exposes immutable molecule/group assignments; and
  - verifies plan, membership, label, modality, and grouping identity before creating the existing
    immutable `SplitManifest`.

  Validation on 2026-07-30:

  - focused ML-201/plan tests: `25 passed`;
  - all machine-learning tests: `130 passed, 7 xfailed`;
  - full unit marker suite: `1358 passed, 9 skipped, 106 deselected, 7 xfailed` in
    201.25 seconds with normal multiprocessing permissions;
  - Ruff check, format, and `git diff --check`: passed;
  - Sphinx warning-as-error HTML build: passed on 2026-08-01 with network access for
    intersphinx; and
  - generated Sphinx source/build artifacts were moved out of the worktree after validation.

### ML-202 — Build bounded partition-aware dataset reads

- Checkbox: `x`
- Status: `DONE`
- Depends on: ML-101, ML-201
- Purpose: read selected rows and features from partition stores within explicit memory limits.
- Deliverables:

  - dataset/index plan using spine/catalog row locations;
  - bounded row/position/layer projection through existing partition readers, grouped by modality
    and resolved into one ordered channel schema;
  - batch iterator suitable for PyTorch;
  - bounded materializer suitable for sklearn/sparse matrices;
  - deterministic sample ordering and worker behavior;
  - memory preflight and refusal path; and
  - optional caching with source/split/schema identity.

- Acceptance:

  - no complete AnnData tensorization is required;
  - batch rows and labels match the split manifest;
  - feature coordinates, biological channels, and per-channel masks remain aligned;
  - modality-inapplicable union channels carry availability masks and never measured-zero fills;
  - memory use is bounded by declared batch/materialization policy;
  - multiple workers do not duplicate or omit observations;
  - empty/partial partitions and variable reference lengths have defined behavior; and
  - end-to-end fixtures prove partition-to-batch correctness.

- Evidence/notes:

  Local implementation on `feature/ml-partition-dataset` adds path-neutral manifest-to-spine
  bindings, a stable source/split/schema/coordinate plan identity, explicit batch and full-split
  memory policies, deterministic worker-sharded batches, and a preflighted dense sklearn view.
  Reads project only the required molecule IDs, interval, and modality-specific layers through the
  existing partition reader. Values remain NaN when unavailable or unobserved and carry separate
  observed, availability, design, and padding masks. Union-channel schemas now declare
  observation-specific design masks so modality-specific C/GpC/CpG designs remain exact.

  Validation on 2026-08-01:

  - focused contract/unit/integration tests: `29 passed`;
  - all machine-learning unit and integration tests: `139 passed, 7 xfailed`;
  - full unit marker suite: `1363 passed, 9 skipped, 109 deselected, 7 xfailed` in
    207.94 seconds with normal multiprocessing permissions;
  - real partition fixtures cover mixed deaminase/conversion channels, partial interval rows,
    variable reference lengths, masks, deterministic multi-worker coverage, and materialization;
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx; and
  - generated Sphinx source/build artifacts were moved out of the worktree after validation.

### ML-203 — Add fitted transforms and train-only balancing

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-201, ML-202
- Purpose: provide backend-appropriate preprocessing without leakage.
- Deliverables:

  - fitted transform protocol with serialized state;
  - sklearn pipelines for imputation/indicators/scaling where applicable;
  - Torch transforms with equivalent documented semantics;
  - train-only class weights, weighted sampler, downsampling, and optional upsampling;
  - natural-prevalence validation/test default;
  - optional separately named evaluation-prevalence sensitivity analysis; and
  - transform/balance provenance.

- Acceptance:

  - transform `fit` sees training observations only;
  - validation/test roles are never resampled by training balance configuration;
  - observed/design indicators survive materialization;
  - class-weight ordering matches persisted class order;
  - inference reuses fitted state unchanged; and
  - tests deliberately expose preprocessing leakage if introduced.

- Evidence/notes: local branch `feature/ml-train-transforms-balancing` adds:

  - a backend-neutral immutable fitted-transform protocol and checksummed JSON state, fit only from
    materialized data whose manifest role is `train`;
  - constant/mean/median/most-frequent imputation, optional standard scaling, stable feature names,
    and independent observed/design/availability/padding indicator features;
  - an sklearn `Pipeline` with flattened indicators and a Torch adapter with channel-first values
    plus separate scientific masks, both applying the same fitted state without refitting;
  - class weights and weighted sampling in persisted label-schema order, plus deterministic
    train-only downsampling/upsampling;
  - natural-prevalence primary validation/test resolutions and separately named, non-mutating
    balanced evaluation-sensitivity cohorts; and
  - checksummed dataset/split/molecule provenance, selected indices, counts, and weights.

  Validation evidence:

  - focused transform/balancing tests: `12 passed`;
  - all machine-learning unit and integration tests: `151 passed, 7 xfailed`;
  - full unit marker suite: `1375 passed, 9 skipped, 109 deselected, 7 xfailed` in 207.00 seconds;
  - Ruff check, Ruff format check, and `git diff --check`: passed; and
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx; generated
    source/build artifacts were moved out of the worktree after validation.

### ML-204 — Train from streamed batches without full materialization

- Checkbox: `[x]`
- Status: `DONE` — merged in PR #458 (`23a5dfc`); orchestration wiring followed in PR #460
- Depends on: ML-202, ML-203, ML-301, ML-303 (all merged)
- Discovered by: ML-700 while wiring Sweep B. Added 2026-08-05.
- Purpose: let a training run consume a production-scale experiment. Today none can.

#### Problem

The bounded streaming reader delivered by ML-202 is correct, and ML-700 demonstrated that it holds
memory across a 300-batch run (quartile growth decaying to 3.8% of one batch estimate). **No
training code uses it.**

- `training/torch_backend.py` materializes `train` (line 403), `validation` (line 404), and `test`
  (line 514), and never calls `iter_batches`.
- `training/sklearn_backend.py` materializes `train` at line 145, *before* the incremental branch.
  Line 167 transforms the whole array; line 175 then chunks it for `partial_fit`. The advertised
  `incremental_fit` capability therefore bounds memory inside the estimator only, and nothing at
  the data boundary.

Verified: against a budget one byte below the train estimate, `bernoulli_nb` with
`incremental=True`, `bernoulli_nb` with `incremental=False`, `random_forest`, and
`residual_dilated_cnn` all refuse at preflight. Incremental and non-incremental refuse at the
identical boundary.

With the default 2 GiB budget and a 60% train fraction the ceiling is ~85,011 total rows at 1,000
positions / 1 channel — a 15x shortfall against a 1.3M-read experiment — falling to ~2,294 rows
(567x short) at 20,000 positions / 2 channels.

Raising `max_materialization_bytes` is not the fix. At 1.3M reads and 1,000 positions the train
split alone estimates ~33 GiB; the budget is behaving correctly.

#### The actual design difficulty

Streaming the model fit is the easy half. The hard half is that train-only transform fitting and
balancing currently require the materialized train array:

- `fit_feature_transform` computes per-column fill values and, when standardizing, `np.mean` and
  `np.std` over `axis=0` (`data/transforms.py:356-360`).
- `resolve_role_balance` resolves class membership and selected indices from
  `MLMaterializedPartitionData` (`data/balancing.py:291`).

Streamability differs per statistic and must not be papered over:

| Statistic | Streamable? | Approach |
| --- | --- | --- |
| `imputation="constant"` | yes | no data pass needed |
| `imputation="mean"`, standardize centers/scales | yes | one pass; Welford for variance |
| `imputation="mode"` | yes | one pass; SMF calls are `{0,1}` so the counter is tiny |
| `imputation="median"` | **no** | exact median needs the full column distribution |

**Balancing needs no data pass at all.** `PartitionReadEntry` already carries `class_id`
(`data/partition_dataset.py:127`), so class counts and deterministic selected membership can be
resolved from the read plan's metadata before any batch is decoded. This should be exploited
rather than reimplemented as a streaming reduction.

#### Blocking design decision — `transform_id` is execution-strategy dependent

Found 2026-08-05 while validating the streaming transform fit against the materialized one.

`FittedFeatureTransform.transform_id` is a SHA-256 over an identity dict containing
`centers.tolist()` and `scales.tolist()` — **unrounded float64 values**. Those values depend on
summation order, so a batched accumulation and a single `np.mean` over the whole array differ in
the last bit. Both are numerically correct.

Measured on identical data and an identical spec (`imputation="constant"`, `scaling="standard"`):

| Comparison | Result |
| --- | --- |
| streamed vs materialized `centers` | max abs diff 5.55e-17, `allclose=True`, `array_equal=False` |
| streamed vs materialized `scales` | max abs diff 5.55e-16, `allclose=True`, `array_equal=False` |
| `transform_id` across `batch_size` ∈ {16, 32, 64, 128} | **4 distinct IDs** |

The last row is the one that matters. `batch_size` is a pure performance knob with no scientific
meaning, and it changes the transform's identity — which propagates into model lineage. Two runs
over the same rows with the same spec produce different provenance because someone tuned a
throughput setting.

This is a defect in the **merged ML-203 contract**, not something ML-204 introduces. It was
unobservable while exactly one execution strategy existed; streaming makes it visible.

Note the fill values for `mean` imputation *did* match bitwise. That is luck specific to this
data: SMF calls are exactly `0.0`/`1.0`, so their sums are exact in float64. Continuous signal
would drift there too.

**Options:**

1. **Quantize the hashed statistics** (recommended). Round `fill_values`, `centers`, and `scales`
   to a fixed significant-digit count for the identity dict only; keep full precision in the
   stored arrays. 12 significant digits sits far above the ~1e-16 discrepancy and far below any
   scientifically meaningful distinction. Requires bumping `ML_FEATURE_TRANSFORM_VERSION` and a
   migration note, because previously published `transform_id` values change. ML-702-visible.
2. **Accept divergent IDs** and assert only numerical equivalence. Rejected: it makes ID equality
   useless as a "same transform" test and lets execution mode rewrite a model's lineage, which is
   what the immutable-artifact design exists to prevent.
3. **Make streaming bitwise-match** `np.mean`'s pairwise summation. Not viable — it depends on
   NumPy internals and batch boundaries.

**Resolved 2026-08-05 — option 1 implemented.** `ML_FEATURE_TRANSFORM_VERSION` bumped 1 → 2.
`transform_id` is now computed over `_identity_digest_payload()`, which renders `fill_values`,
`centers`, and `scales` at 12 significant digits as decimal strings; negative zero is normalized.
The quantization applies to the digest input only — stored arrays and `to_dict()` keep full
precision, so published artifacts and their `from_dict()` round trip are byte-unchanged.

Verified after the change:

| Property | Result |
| --- | --- |
| streamed vs materialized `transform_id`, all 5 streamable specs | **IDENTICAL** |
| distinct IDs across `batch_size` ∈ {1, 5, 16, 48} | **1** |
| `to_dict`/`from_dict` round trip | bitwise-equal arrays, ID preserved |

Migration note for ML-702: transform IDs published under schema version 1 do not match their
version 2 recomputation. No version 1 artifacts exist outside development at the time of the
change.

#### Deliverables

- A two-pass streaming fit contract: pass 1 accumulates transform statistics from batches, pass 2
  applies the frozen transform and drives the model. Pass 1 must hold only accumulators.
- Balance resolution computed from read-plan metadata, with no data pass.
- `fit_sklearn_partition_model`: drive `partial_fit` from `iter_batches` for families declaring
  `incremental_fit`, keeping the existing chunk determinism and class-order contract.
- `fit_torch_partition_model`: drive train and validation loaders from `iter_batches`; keep the
  locked test role unread until early stopping has selected and restored the best state.
- An explicit decision on `imputation="median"` under streaming: refuse with a message naming the
  streamable alternatives, or adopt a named approximation with recorded error bounds. Do not
  silently substitute a different statistic.
- Non-incremental families (`random_forest`, `logistic_regression`) keep the materialization
  ceiling. Their refusal message must name streaming-capable families as the alternative.
- Provenance: the fitted-transform checksum must remain derivable from train rows only and must
  match between the streamed and materialized paths for a dataset small enough to run both.

#### Acceptance

- A dataset whose train split exceeds `max_materialization_bytes` trains successfully through the
  streaming path for every family declaring `incremental_fit`, and for Torch.
- Streamed and materialized fits agree to numerical tolerance on a dataset that fits both, with
  identical fitted-transform and balance checksums.
- Peak memory during a streaming fit is bounded by the batch budget plus accumulators, verified
  with the ML-700 trajectory method rather than a single peak-RSS reading.
- Validation and test roles keep natural prevalence; no streaming shortcut introduces
  cross-split leakage, and the ML-203 train-only fingerprint still holds.
- Non-streamable configurations fail before reads with a message naming a supported alternative.
- Refusal messages for non-incremental families name the streaming-capable families.

#### Notes

Sizing this honestly: it changes the two vertical slices delivered by ML-301 and ML-303 plus the
ML-203 transform contract, so it is not a small package. It is also the difference between an ML
system that runs on the lab's real experiments and one that does not.

### ML-205 — Estimate Torch activation memory

- Checkbox: `[ ]`
- Status: `PROPOSED` — scoped, with the start gate below
- Depends on: ML-204, ML-303
- Discovered by: ML-700 while measuring streaming-fit memory. Added 2026-08-06.
- Purpose: give Torch training a memory preflight, as the data plane already has for reads.

#### Problem

Every memory guardrail in the ML data plane models **data** bytes.
`partition_dataset._bytes_per_row` estimates the decoded batch; nothing estimates the model.
During Torch training the model dominates:

| Measurement | Value |
| --- | ---: |
| Torch streaming-fit plateau RSS (3 epochs, 200 positions) | 2,172 MB |
| Ratio to the data batch estimate | **12,557x** |
| sklearn streaming-fit plateau, same shape | 132 MB (307x — same order as a plain read at 281x) |

So the gap is a property of neural models, not of streaming. Three consequences, all currently
unhandled:

1. `max_batch_bytes` does not control Torch training memory. Tuning it will not prevent an OOM
   caused by model width or depth.
2. No supported/slow/refused boundary for Torch training can be derived from the data estimator,
   which is why `tests/acceptance/ml_scale_thresholds.json` records Torch as
   `"supported, but memory is NOT modelled"` with `ceiling: unknown`.
3. A user sizing a training job from the data budget under-predicts by three to four orders of
   magnitude.

An attempt to derive a fixed multiplier failed and should not be retried in that form: across
batch size x sequence length the ratio spanned 4,409x to 20,833x, and the grid was confounded
because batch count was not held constant. A real model must vary the **architecture**
(`block_channels`, `stem_channels`, `hidden_dim`, `dilations`) rather than only the data shape.

#### Start gate

Do not begin on measurement appetite alone. Start when either:

- a Torch training run is OOM-killed at production scale, giving a concrete failure to model
  against; or
- a user or workflow needs a training-memory preflight — for example scheduling training on a
  shared HPC allocation, where refusing early beats being killed late.

Until then the honest published guidance is "size Torch training empirically", which
`ml_scale_thresholds.json` states.

#### Deliverables

- An activation-memory estimate as a function of the registered architecture and batch shape,
  living beside the data estimator rather than inside a model class.
- A preflight for Torch training mirroring `MLMemoryBudgetError`: refuse before allocation, with a
  message naming the architecture term that dominates.
- Calibration against measured peak RSS with repeats and controlled batch counts, using the
  ML-700 trajectory method rather than single peak readings.
- Published Torch limits replacing the `ceiling: unknown` entry in the thresholds file, and
  removal of the corresponding `unmodelled` register entry.

#### Acceptance

- The estimate bounds measured worst-case peak RSS across the calibration grid, in the
  conservative direction, with the headroom distribution published as ML-700 did for data reads.
- Refusal fires before allocation and names the dominant term.
- Changing a registered architecture's width or depth moves the estimate in the correct direction
  by roughly the predicted amount.
- `ml_scale_thresholds.json` no longer lists Torch activation memory as unmodelled, and the
  per-PR guard in `tests/unit/machine_learning/test_ml_scale_thresholds.py` is updated so the
  register cannot silently empty.

#### Notes

Scope risk: an activation estimator is architecture-specific, and only `residual_dilated_cnn` is
registered today. Estimating for one family is tractable; a general estimator over arbitrary
`nn.Module` graphs is not, and should not be attempted. If more families are registered before
this starts, prefer a per-recipe declared estimate over a universal analyzer.

## P3 — backend-neutral prediction and model implementations

### ML-300 — Define predictor protocol and explicit model registry

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-101, ML-104
- Purpose: unify capabilities and artifacts without forcing sklearn and Torch into one inheritance
  hierarchy.
- Deliverables:

  - small predictor protocol for loading, prediction, scores/probabilities, schemas, and
    capabilities;
  - `SklearnPredictor` and `TorchPredictor` adapters;
  - explicit built-in registry mapping names to config type, builder, schema versions, and
    capabilities;
  - named immutable model recipes; and
  - third-party plugin decision deferred unless a real consumer exists.

- Acceptance:

  - evaluation/application code does not branch on concrete model classes;
  - unsupported probability/explanation/mask requests fail via capabilities;
  - model recipes declare supported modality/channel-schema capabilities;
  - registry behavior is deterministic and does not depend on import side effects;
  - resolved recipe values, not only recipe names, are persisted; and
  - backend-specific objects do not leak into plan schemas.

- Evidence/notes: local branch `feature/ml-predictor-registry` adds:

  - a runtime-checkable backend-neutral predictor/loader protocol with ordered class predictions,
    score matrices, probability matrices, input-schema checks, and pre-execution capability/mask
    rejection;
  - composition-based fitted sklearn and plain-`nn.Module` Torch adapters, including sklearn
    `classes_` validation, channel-first Torch inputs, distinct mask forwarding, and restoration of
    Torch training/evaluation state;
  - immutable checksummed recipe records with deep-frozen parameters, declared modality/channel
    compatibility, strict round trips, and fully resolved `ResolvedDefinition` parameters;
  - an immutable deterministic registry with no decorator/import-side-effect registration and
    tamper checks between recipes, typed configs, capabilities, and builders;
  - initial estimator-only definitions for Bernoulli Naive Bayes, logistic regression, and random
    forest; training, trusted persistence/loading, and artifact publication remain ML-301; and
  - no canonical Torch family registration yet: the residual-dilated CNN and its reviewed recipe
    remain correctly scoped to ML-302 rather than depending on legacy `analysis` model code.

  Validation evidence:

  - focused predictor/registry tests: `22 passed`;
  - ML unit, integration, and smoke tests: `194 passed, 7 xfailed`;
  - full unit marker suite: `1397 passed, 9 skipped, 109 deselected, 7 xfailed` in 206.92 seconds;
  - Ruff check, Ruff format check, and `git diff --check`: passed; and
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx; generated
    source/build artifacts were moved out of the worktree after validation.

### ML-301 — Deliver sklearn vertical slice

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-203, ML-300, ML-105
- Purpose: establish a low-cost complete workflow before neural orchestration grows.
- Initial supported models:

  - Bernoulli Naive Bayes;
  - logistic regression; and
  - random forest.

- Deliverables:

  - validated builders and parameters;
  - fitted sklearn pipeline artifacts;
  - incremental `partial_fit` path only where the estimator supports it;
  - memory-preflighted materialization for non-incremental estimators;
  - consistent probability/decision-score records;
  - dependency/version and trust metadata; and
  - native parameter summaries.

- Acceptance:

  - each model trains and applies from the same dataset/split schema;
  - model round trip reproduces predictions within tolerance;
  - learned `classes_` order matches the label schema;
  - RF/non-incremental training refuses an unsafe materialization estimate;
  - no untrusted pickle/joblib artifact is loaded implicitly;
  - Bernoulli NB provides a useful baseline on the canonical fixture; and
  - one complete local run/model artifact is published.

- Evidence/notes: local branch `feature/ml-sklearn-vertical` adds:

  - one manifest-bound training entry point for Bernoulli NB, logistic regression, and random
    forest using the same input, label, transform, balancing, dataset snapshot, and split
    contracts;
  - capability-gated `partial_fit` for Bernoulli NB, full-fit materialization through the
    partition store's conservative memory preflight for logistic regression/random forest, exact
    persisted class ordering, native parameter summaries, and explicit rejection of Torch-only
    weighted sampling;
  - immutable application records with class IDs, decision/probability score matrices, class
    order, molecule IDs, and split/phase identity;
  - canonical `.skops` serialization with checksum-validated publication, exact dependency
    versions, exact registered-estimator allowlists, no pickle/joblib fallback, default version
    mismatch refusal, and reconstruction of input/label/transform/architecture contracts; and
  - complete local run plus model publication and prediction-preserving load tests.

  Validation evidence:

  - focused sklearn vertical tests: `9 passed`;
  - focused registry/training/inference/transform/balance/artifact tests: `57 passed`;
  - ML unit, integration, and smoke tests: `274 passed, 7 xfailed`;
  - full unit marker suite: `1406 passed, 9 skipped, 109 deselected, 7 xfailed` in 206.12 seconds;
  - Ruff check, Ruff format check, and `git diff --check`: passed; and
  - Sphinx warning-as-error HTML build: passed with network access for intersphinx; generated
    source/build artifacts were moved out of the worktree after validation.

### ML-302 — Consolidate configurable PyTorch model families

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-101, ML-300
- Purpose: replace duplicate architecture ownership with validated families plus named recipes.
- Candidate families:

  - residual/dilated 1D CNN;
  - token or feature transformer;
  - CNN-transformer hybrid; and
  - shared encoder plus task heads.

- Deliverables:

  - plain `torch.nn.Module` implementations;
  - typed validated family configuration;
  - named reviewed recipes;
  - input/mask compatibility declarations;
  - architecture schema versions;
  - migration decision for useful existing modules, including domain-adversarial components; and
  - shape/forward/state-dict tests.

- Acceptance:

  - depth/width/model dimension can vary only through supported validated parameters;
  - nonsensical combinations fail on construction;
  - models contain no filesystem, AnnData mutation, trainer, or plotting behavior;
  - forward outputs and mask behavior are documented and tested;
  - a stored resolved config reconstructs the exact architecture; and
  - useful active code is migrated rather than independently rewritten without reason.

- Evidence/notes: local branch `feature/ml-torch-model-families` adds:

  - one canonical plain-`nn.Module` residual/dilated 1D CNN under
    `machine_learning.models`, migrated from the active `analysis.compute.ml_cnn` implementation;
    the analysis module now preserves its established model/config/builder imports as compatibility
    aliases rather than owning a second architecture copy;
  - a frozen, strictly parsed family configuration with validated positive dimensions, matching
    block/dilation depth, odd length-preserving kernels, finite dropout, configurable widths/depth,
    binary or multiclass output dimension, squeeze/excitation, and attention pooling;
  - the immutable `residual_dilated_cnn_v1` registry recipe, architecture schema version 1,
    explicit Torch capabilities, input-channel compatibility validation, and exact resolved-config
    reconstruction before state-dict loading;
  - separate observed, availability, design, and padding mask arguments in channel-first Torch
    layout, with invalid signal zeroed before convolution, masked after every residual stage, and
    excluded from final average/max/attention pooling; mask shapes, polarity, dtype, empty rows, and
    non-finite valid values fail explicitly;
  - contract-layout mask transposition in `TorchPredictor`, so partition/transform masks remain
    aligned with the channel-first tensors already emitted by `TorchFeatureTransform`;
  - lazy compatibility exports for optional plotting, sklearn-artifact, and Lightning wrappers, so
    importing or using the plain Torch family does not import Lightning; and
  - a deliberate migration decision to retain the existing MLP/RNN/simple CNN/transformer,
    masked-pretrainer, and domain-adversarial code as unregistered compatibility/prototype modules.
    Transformer, CNN-transformer, domain-adversarial, and shared-encoder recipes remain deferred
    until a real task establishes their input, label, mask, attention-output, and transfer contracts.

  Validation evidence:

  - focused family/registry/predictor/legacy tests: `45 passed`;
  - ML unit, integration, and smoke tests: `290 passed, 7 xfailed`;
  - full unit marker suite: `1422 passed, 9 skipped, 109 deselected, 7 xfailed` in 207.58 seconds;
  - public import, legacy alias, state-dict reconstruction, and isolated no-Lightning import probes:
    passed;
  - Ruff check, Ruff format check, and `git diff --check`: passed; and
  - clean Sphinx warning-as-error HTML build: passed with network access for intersphinx; generated
    source/build artifacts were moved out of the worktree after validation.

### ML-303 — Deliver plain-PyTorch vertical slice

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-202, ML-203, ML-302, ML-105
- Purpose: establish the canonical neural workflow without requiring Lightning.
- Deliverables:

  - task/loss object separate from the model;
  - minimal explicit training/validation loop;
  - deterministic device selection and seed recording;
  - early stopping and best-state restoration;
  - resumable state only if included in the approved MVP;
  - plain `state_dict` inference artifact plus resolved config/schema; and
  - tidy history and prediction records.

- Acceptance:

  - one CPU optimization step and short fit succeed in unit/integration tests;
  - validation uses the validation loader and test uses the locked test loader;
  - the best checkpoint/state is restored before final evaluation;
  - missing values are handled reproducibly with explicit masks;
  - saved/loaded inference outputs match;
  - no Captum, Lightning, Hydra, or tracker is required; and
  - the same evaluation schema used by ML-301 is produced.

- Evidence/notes: local branch `feature/ml-torch-vertical` adds:

  - a strict versioned `TorchTrainingConfig`, a binary/multiclass `ClassificationTask` that owns
    loss semantics outside the model, tidy immutable epoch rows, and fitted/result records carrying
    the exact architecture, schemas, fitted transform, split/dataset IDs, seed/device policy,
    best epoch, held-out losses, and balancing resolution;
  - a minimal explicit Adam training loop using partition-store memory-preflighted materialization,
    train-only fitted imputation/scaling, channel-first signals, separate observed/availability/
    design/padding masks, deterministic seeded initialization/loading, validation-only early
    stopping, and in-memory best-state restoration;
  - locked test-role materialization and evaluation only after the best validation state is
    restored; test loss is never used for optimization or model selection;
  - shared natural/class-weight/weighted-sampler/downsample/upsample balancing behavior, while
    rejecting appended Torch mask-indicator features because masks remain distinct model inputs;
  - Torch prediction records with the same molecule/class/score/probability/class-order/split field
    contract as the sklearn vertical slice;
  - immutable `torch-state-dict` model publication containing canonical JSON metadata plus CPU
    tensors, exact registry architecture/key/shape/dtype validation, dependency versions, training
    history, fitted transform, and input/label schemas; loading always uses `weights_only=True`,
    reconstructs the registered model before strict state loading, and reproduces predictions; and
  - lazy legacy Lightning/AnnData/high-level imports across `data`, `training`, and `inference`, so
    the canonical Torch train/apply/publish path imports without Lightning, Captum, Hydra, or a
    tracker.

  Resumable optimizer checkpoints are deliberately excluded from this MVP. Correct resume support
  must persist optimizer/scheduler, epoch/step, random-generator, sampler, and callback state rather
  than presenting an inference state dict as resumable training state.

  Validation evidence:

  - focused Torch vertical tests: `10 passed`;
  - focused Torch/sklearn/transform/balance/predictor compatibility tests: `53 passed` before the
    final mask-indicator rejection test was added, followed by focused revalidation;
  - ML unit, integration, and smoke tests: `300 passed, 7 xfailed`;
  - full unit marker suite: `1432 passed, 9 skipped, 109 deselected, 7 xfailed` in 210.73 seconds;
  - deterministic same-seed state/prediction, restored-best validation loss, train/validation/test
    access order, safe-load flag, and isolated no-Lightning import probes: passed;
  - Ruff check, Ruff format check, and `git diff --check`: passed; and
  - clean Sphinx warning-as-error HTML build: passed with network access for intersphinx; generated
    source/build artifacts were moved out of the worktree after validation.

### ML-304 — Add pretrained encoder and fine-tuned head lineage

- Checkbox: `[ ]`
- Status: `PROPOSED` — gated on the start gate below; not implemented
- Depends on: ML-303
- Purpose: distinguish reusable representation learning from task-specific classification.
- Start gate:

  - a meaningfully larger/diverse unlabeled corpus exists; and
  - a from-scratch downstream baseline is available for comparison.

- Deliverables:

  - encoder/head composition;
  - pretraining task and corruption/loss-mask schema;
  - encoder artifact distinct from trainer checkpoint;
  - fine-tuned model referencing parent encoder and exact initialization policy;
  - freeze/unfreeze schedule provenance; and
  - transfer benchmark against from-scratch training.

- Acceptance:

  - pretrained artifact loads without a classification head;
  - fine-tuned artifact records parent model/checksum;
  - classifier label schema is not embedded as an encoder invariant;
  - corruption masks cannot leak into observed/design semantics;
  - transfer performance and uncertainty are compared to from-scratch; and
  - unsupported input-schema transfer fails clearly.

- Evidence/notes: none. This section previously carried a `[x] DONE` checkbox and an evidence
  block describing the Captum attribution adapters — that is ML-403's work, misfiled here, and
  ML-403 records it in its own section. No encoder/head composition, pretraining task, encoder
  artifact, or transfer benchmark exists. `machine_learning/artifacts/model.py` reserves
  `"pretrained"`/`"fine_tuned"` origin values and `contracts.py` reserves a `pretraining_task`
  mask consumer, but those are schema slots only. The operational status table's `PROPOSED` is
  authoritative.

## P4 — evaluation, interpretability, and plotting

### ML-400 — Standardize predictions and metrics

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-301, ML-303
- Purpose: make downstream evaluation independent of model backend.
- Deliverables:

  - prediction table schema with observation, experiment modality, group, split/cohort, truth,
    class, score, and model identity;
  - binary and multiclass metric records;
  - ROC/PR, calibration, confusion, threshold, and class-balance summaries;
  - fold aggregation and uncertainty policy;
  - training history/event schema that permits models without epochs; and
  - threshold/calibration fitting provenance.

- Acceptance:

  - sklearn and Torch predictions enter the same evaluator;
  - metrics can be recomputed from stored prediction tables;
  - validation-selected thresholds are not refit on test;
  - class order and positive class are explicit;
  - mixed-modality runs report support and metrics by modality alongside pooled metrics;
  - one-shot sklearn fits do not receive fabricated epoch curves; and
  - row-level sensitive outputs can be omitted or redacted for export.

- Evidence/notes: local branch `feature/ml-evaluation-contract` adds:

  - one immutable `PredictionResult` used concretely by both sklearn and Torch, retaining the old
    backend result names as compatibility aliases while adding ordered experiment, modality,
    optional group, truth, positive-class, cohort, and model identity fields;
  - validated full-row table export/restoration with explicit include, omit, or salted-hash identity
    policies, allowing metrics to be recomputed from stored prediction rows without applying a model;
  - immutable scalar metric, ROC/PR/calibration curve, confusion, natural class-balance,
    threshold/calibration provenance, training-event/history, evaluation-result, and fold-summary
    records with explicit persisted class order and nullable undefined slice metrics;
  - deterministic binary and multiclass evaluation at natural prevalence, including pooled and
    per-modality support/metrics, one-vs-rest class summaries, calibration and decision curves, and
    confusion matrices;
  - F1 and Youden-J validation/train threshold selection whose model, class order, positive class,
    split, and cohort provenance is carried into test evaluation; test fitting and incompatible
    threshold/calibration reuse are rejected;
  - equal-fold metric aggregation with contributing fold identities and sample-standard-deviation
    uncertainty; and
  - history adapters that preserve real Torch epochs but represent one-shot sklearn fitting as a
    non-epoch completion event rather than a fabricated learning curve.

  The existing `PredictionManifest` remains the immutable on-disk artifact/table identity; these
  records are the backend-independent in-memory/table evaluation contract. Artifact publication and
  job orchestration remain owned by ML-500 rather than being coupled into metric computation.

  Validation evidence:

  - focused evaluation plus sklearn/Torch vertical tests: `25 passed`;
  - ML unit and smoke area: `234 passed, 7 xfailed`;
  - full unit marker suite: `1438 passed, 9 skipped, 112 deselected, 7 xfailed` in 207.74 seconds;
  - prediction-table round trip, redacted export, validation-to-test threshold isolation,
    binary/multiclass behavior, mixed-modality support, fold uncertainty, and non-epoch sklearn
    history tests: passed;
  - repository-wide Ruff check, Ruff format check, and `git diff --check`: passed; and
  - clean Sphinx warning-as-error HTML build: passed with network access for intersphinx; generated
    source/build artifacts were removed after validation.

### ML-401 — Implement interpretability request/result contracts

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-104, ML-400
- Purpose: standardize explanation inputs and outputs before adding methods.
- Deliverables:

  - `InterpretabilityRequest`/equivalent with method, target, cohort, baseline/background, layer,
    mask policy, aggregation, and parameters;
  - `AttributionResult` with aligned observation/feature axes and provenance;
  - training-only background/reference sampling;
  - explanation artifact layout and IDs;
  - canonical method names (`GradientSHAP`, not ambiguous aliases); and
  - capability dispatch.

- Acceptance:

  - unsupported model/method combinations fail before expensive computation;
  - explained output and target class are recorded;
  - baselines/backgrounds are checksummed and never selected from locked test data;
  - attribution axes validate against the input schema and retain physical site context plus
    declared biological channel role;
  - test observations may be explained without tuning explanation choices on test; and
  - large raw explanations can be stored in chunked Zarr.

- Evidence/notes:

  - added exact, versioned, content-addressed explanation requests covering target, cohort,
    baseline, layer, masks, aggregation, method parameters, and train/validation-only decision
    provenance;
  - added capability preflight for the canonical classical, tree, gradient, layer, and attention
    method vocabulary, rejecting ambiguous aliases and unsupported model/method combinations before
    execution;
  - added deterministic training-only background sampling with checksummed values, masks,
    observations, experiments, modalities, schema identity, and split-manifest provenance;
  - added immutable attribution results with explicit axes, input-schema validation, physical site
    context and biological channel roles, plus support for position-only layer/attention results;
  - mapped runtime results into ML-104 explanation manifests under canonical Zarr/Parquet artifact
    paths without importing Captum or SHAP; and
  - validation passed: `245 passed, 7 xfailed` across ML unit/smoke tests; `1444 passed, 9 skipped,
    117 deselected, 7 xfailed` in the full unit-marker suite; repository-wide Ruff check and format
    check; `git diff --check`; and a clean warning-as-error Sphinx HTML build. Generated docs
    artifacts were removed after validation.

### ML-402 — Add classical and tree explanation adapters

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-301, ML-401
- Initial methods:

  - Naive Bayes log-probability/log-odds differences;
  - standardized linear coefficients/odds ratios;
  - held-out permutation importance for any predictor; and
  - TreeSHAP for supported forest/boosting models.

- Acceptance:

  - direct native parameters are preferred where they answer the question;
  - permutation importance records cohort, metric, repeat count, and seed;
  - TreeSHAP records model output and feature-perturbation/background policy;
  - aggregate results preserve class/sample/reference/position identity;
  - SHAP remains optional under the extended ML dependency; and
  - explanation tests use known synthetic feature effects.

- Evidence/notes:

  - added a strict canonical sklearn dispatcher for exact Bernoulli Naive Bayes target/reference
    log-odds, logistic-regression coefficients or odds ratios, deterministic held-out permutation
    importance, and TreeSHAP;
  - extended common attribution results with explicit transformed-feature metadata so biological
    signals and observed/design/availability/padding indicators remain distinct while retaining
    coordinate, physical site context, and biological channel role;
  - permutation importance rejects train/inference cohorts and records the held-out split/cohort,
    target-aware metric, repeat count, seed, baseline score, and per-feature variability;
  - TreeSHAP validates and records model output, feature-perturbation policy, additivity checking,
    SHAP version, and the exact checksummed training background before importing SHAP lazily from
    the `ml-extended` extra;
  - native-method synthetic tests reconstruct Bernoulli NB posterior log odds exactly and verify
    fitted-unit logistic coefficients/odds ratios; TreeSHAP and permutation tests cover known
    conversion-modality feature schemas and deterministic provenance;
  - existing `analysis.compute.ml_explanations` entry points remain unchanged for compatibility;
    consumer migration and any deprecation are deferred to ML-503/ML-504, and release-note
    generation remains release-branch work under repository policy; and
  - validation passed: `250 passed, 7 xfailed` across ML unit/smoke tests; `1449 passed, 9 skipped,
    117 deselected, 7 xfailed` in the full unit-marker suite; repository-wide Ruff check and format
    check; `git diff --check`; and a clean warning-as-error Sphinx HTML build. Generated docs
    artifacts were removed after validation.

### ML-403 — Add neural explanation adapters

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-303, ML-401
- Initial methods:

  - saliency and gradient × input;
  - Integrated Gradients;
  - GradientSHAP;
  - DeepLift;
  - LayerGradCam for declared CNN layers; and
  - precisely named attention rollout or attention × gradient methods, if justified.

- Acceptance:

  - target output, baseline, steps/samples, convergence delta, mask policy, and layer are recorded;
  - attribution respects observed/design/padding masks;
  - LayerGradCam is exposed only for compatible convolutional layers;
  - attention weights alone are not labeled as definitive explanations;
  - Captum is optional and imported only when needed;
  - chunking prevents unbounded attribution batches; and
  - synthetic completeness/sensitivity checks are included where the method supports them.

- Evidence/notes:

  - added bounded, mask-aware Saliency, InputXGradient, Integrated Gradients, DeepLift,
    GradientSHAP, LayerGradCam, and GuidedGradCam adapters over canonical fitted Torch models;
  - retained checksummed training backgrounds, declared-layer gating, deterministic stochastic
    attribution, convergence evidence, immutable provenance, and lazy Captum imports; and
  - left attention explanations capability-gated until a validated registered attention model
    exists rather than presenting raw attention weights as definitive explanations.

## P5 — user workflows, CLI, and migration

### ML-500 — Implement backend-neutral job services

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-301, ML-303, ML-400
- Purpose: provide testable Python orchestration independent of Click, Hydra, or Lightning.
- Job types:

  - plan/dry run;
  - train;
  - apply/predict;
  - evaluate;
  - explain; and
  - plot/compare.

- Deliverables:

  - typed service entry points accepting resolved plan/workspace objects;
  - explicit state transitions and failure manifests;
  - immutable model/run resolution;
  - cancellation/interruption behavior;
  - structured logging through package logging utilities; and
  - no project-specific constants.

- Acceptance:

  - apply/evaluate/explain/plot never retrain;
  - selectors such as "best from run" resolve once to an immutable model ID and metric;
  - outputs remain inside the active workspace;
  - services are callable without Click;
  - failures publish a diagnostic run record without a false completed marker; and
  - repeated applications of one model produce distinct run identities.

- Evidence/notes:

  - added a framework-independent `machine_learning.orchestration` package with typed resolved-job,
    dry-run, operation, staged-artifact, cancellation, terminal-outcome, and error contracts;
  - added explicit exact-ID, alias, and validation-metric best-from-run model selection that
    validates published bundles and resolves mutable selectors once to immutable model IDs;
  - added backend-neutral dispatch over the canonical sklearn/plain-Torch train, apply, evaluation,
    and explanation engines; only the training dispatcher calls fitting functions;
  - added action-specific train/apply/evaluate/explain/plot lifecycle services callable without
    Click, Hydra, Lightning, or a hosted tracker;
  - every attempt receives a fresh run UUID, writes only below the injected workspace, stages
    resolved plan/config plus explicit output files, and atomically publishes one terminal run
    bundle through the existing artifact service;
  - planned/running/completed/failed/cancelled transitions are explicit and emitted as structured
    records through package logging; failed and cancelled attempts publish diagnostic manifests
    without partial output artifacts or false completion;
  - cooperative tokens check before execution, at explicit phase boundaries, and before
    publication; `KeyboardInterrupt` publishes cancellation evidence before propagating;
  - plot/compare accepts only immutable source-run UUIDs and explicit contained output paths;
    result computation/plot implementations remain correctly scoped to ML-502;
  - focused orchestration tests: `20 passed`; all ML unit/smoke tests: `282 passed, 7 xfailed`;
    unrestricted full unit-marker suite: `1475 passed, 9 skipped, 123 deselected, 7 xfailed`;
    warning-as-error Sphinx HTML build, repository-wide Ruff check/format, and `git diff --check`
    passed; generated documentation outputs were removed after validation.

### ML-501 — Add dry-run planning and user-facing orchestration

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-500
- Purpose: let users safely see data, classes, splits, memory, models, and outputs before execution.
- Decision gate:

  - choose the initial public interface: Python-only, experiment CLI, project CLI, or both CLI
    scopes.

- If CLI is approved:

  - keep Click wrappers thin;
  - use the established wrapper/core split;
  - place one-experiment commands under `smftools experiment`;
  - place cross-experiment commands under `smftools project`;
  - update the CLI command map and documentation; and
  - test Click parsing separately from core behavior.

- Acceptance:

  - dry run performs selection/split/schema checks without training;
  - it reports output root, dataset/group/class counts, overlap checks, memory estimate, models,
    actions, and optional dependency requirements;
  - users can select explicit samples/groups through stable IDs;
  - invalid plans fail with field-level guidance; and
  - experiment/project scope cannot be inferred accidentally from the working directory.

- Evidence/notes:

  - resolved D-017 as Python-only for the initial public surface; no Click command, Hydra
    application, Lightning adapter, or hosted-tracker writer was added;
  - added an immutable, JSON-ready workflow preview that composes the existing metadata
    selector, group-safe split planner, input/label schemas, explicit model registry, and
    experiment/project workspace resolver without fitting models or writing files;
  - requires an explicit experiment config or initialized project path matching the declared
    scope and never derives ownership or output paths from the working directory;
  - reports requested experiment/sample/reference selectors and explicit split groups, selected
    sample/class/modality counts, per-fold observation/group/class support, disjointness and
    coverage checks, total and per-role memory estimates, resolved model architectures and
    capabilities, balancing policies, job actions, output roots/layout, and expected artifacts;
  - reports Captum, SHAP, and W&B requirements from requested explanations/tracking together with
    installed availability; plain sklearn/PyTorch services remain the default and Lightning and
    Hydra are explicitly not required;
  - model, dataset, split, label, tracking, and scope failures are surfaced with their plan field
    paths; named model recipes are validated against the resolved modality/channel/mask schema;
  - focused planner/adjacent tests: `47 passed`; machine-learning unit directory: `251 passed, 7
    xfailed`; the unrestricted full unit-marker suite exited successfully; warning-as-error Sphinx
    HTML build, repository-wide Ruff check/format, and `git diff --check` passed; generated
    documentation outputs were removed after validation.

### ML-502 — Integrate pure analysis summaries and plots

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-400, ML-401
- Purpose: preserve the `analysis` contract while avoiding duplicated model/training code.
- Deliverables:

  - pure metric/comparison utilities retained or moved under `analysis.compute`;
  - plotting functions under `analysis.plot` that accept result tables and explicit output paths;
  - learning/history, ROC/PR, calibration, confusion, feature importance, and attribution plots;
  - no artifact discovery or model loading inside plotting functions; and
  - compatibility wrappers only where needed during migration.

- Acceptance:

  - compute functions have no filesystem or AnnData dependency unless explicitly exempted;
  - plot functions do not train, apply, or select models;
  - figures can be rebuilt from stored tidy tables;
  - plots label split/cohort/model/class semantics; and
  - analysis behavior has unit tests independent of ML optional extras where possible.

- Evidence/notes:

  - added pure adapters from canonical training, evaluation, fold, balance, confusion, curve, and
    explanation records to fixed-schema tidy pandas tables without filesystem or AnnData access;
  - added table-only renderers for training history, ROC/PR, calibration, metric comparison,
    confusion matrices, feature importance, and attribution summaries with explicit output paths;
  - renderers preserve model, split, cohort, scope/modality, class, coordinate, channel, and
    biological-role semantics where applicable and never discover artifacts or load/apply models;
  - focused analysis tests: `6 passed`; combined analysis and machine-learning tests: `334 passed,
    7 xfailed`; the unrestricted full unit-marker suite exited successfully; warning-as-error
    Sphinx HTML build, repository-wide Ruff check/format, and `git diff --check` passed; generated
    documentation outputs were removed after validation.

### ML-503 — Migrate active ML consumers

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-500, ML-502
- Purpose: prove the new infrastructure against real workflows before removing old paths.
- Deliverables:

  - select one representative active consumer;
  - map its project-specific constants/data selection into an ML plan;
  - replace generic project-side infrastructure with package services;
  - compare split membership, predictions, metrics, and explanations against an accepted baseline;
  - document expected behavior changes; and
  - collect usability/performance feedback.

- Acceptance:

  - project-specific biological vocabulary remains in the project/plan;
  - generic training/artifact/explanation code no longer lives in the migrated project scripts;
  - any metric changes are explained by intentional correctness fixes;
  - artifacts are reproducible from the plan and registered inputs; and
  - the workflow does not require the deprecated training path.

- Evidence/notes:

  - PR #454 merged ML-502 to `main` at `1e49998`; the no-upstream
    `feature/ml-consumer-migration` branch was created from that commit;
  - the ML-001 inventory found no committed production or CLI consumer in smftools and identified
    the external `Nkg2a_DAFseq_merged/claude_scripts/ml/` workflow as the real active consumer;
  - the real read-only consumer was subsequently located at the lab project path supplied by the
    user, and the canonical destination is the initialized sibling
    `Nkg2a_DAFseq_merged_v2` project;
  - selected the bounded real slice `Cseda01 / active_b6_vs_inactive_b6 /
    emseq_enhancer_masked / C_only / random_forest`, with two locked experiments and three
    leave-one-replicate-out folds;
  - preserved the accepted 1,339-row OOF matrix, prediction, metric, and SHAP evidence in a
    checksummed project-owned dataset bundle. Current live inputs contain 1,282 locked rows because
    barcode 21 has 57 fewer reads, and the mutable legacy loader also fails if the later 260622
    experiment is admitted due to feature-layout drift;
  - added a validated package-level materialized-dataset protocol/adapter so project drivers can
    bind already resolved real matrices to canonical dataset/split manifests without duplicating
    training infrastructure;
  - the v2 project now owns only the biological plan and orchestration driver under
    `project_scripts/ml/`; outputs resolve through the canonical project workspace below
    `project_outputs/ml/`. The legacy project remained read-only and no recent file writes were
    detected there;
  - canonical run IDs are `69158c49-9791-4e85-b403-f7e78dcbfb10`,
    `e78ed1c5-b45a-4ca8-98e3-b555f46ea105`, and
    `6cd12e3c-83e0-4eb0-9882-f1cc2e20fa27`; all run and safe `.skops` model bundles pass checksum
    validation;
  - accepted-vs-canonical prediction score correlations are 0.9875, 0.9847, and 0.9894, class
    agreement is 97.88%, 94.69%, and 94.61%, and mean-absolute TreeSHAP profile correlations are
    0.9841, 0.9899, and 0.9900;
  - canonical natural-prevalence ROC AUC is 0.9404, 0.9268, and 0.9305. These are intentionally
    distinct from the legacy `target_eval_freq=0.1` metrics, which resampled the held-out cohort;
    parity reports retain that difference explicitly rather than treating it as a regression;
  - validation: focused backend/job tests `40 passed`; all ML unit tests `257 passed, 7 xfailed`;
    full unit-marker suite `1491 passed, 9 skipped, 123 deselected, 7 xfailed`; repository-wide
    Ruff check/format, `git diff --check`, and warning-as-error Sphinx HTML build passed. The first
    sandboxed full-unit/docs attempts failed only on blocked multiprocessing semaphores and DNS;
    unrestricted reruns passed, and generated documentation outputs were removed.

### ML-504 — Deprecate and remove duplicate legacy paths

- Checkbox: `[x]`
- Status: `DONE`
- Depends on: ML-503
- Purpose: converge ownership without abruptly breaking active users.
- Deliverables:

  - deprecation warnings and migration guide;
  - compatibility adapters where inexpensive and safe;
  - removal schedule tied to normal compatibility policy;
  - deletion or simplification of duplicate model/training/attribution code;
  - retained pure analysis metrics/plots; and
  - update lazy exports and optional imports.

- Acceptance:

  - all known consumers have a documented replacement;
  - deprecation tests verify warnings and compatibility behavior;
  - no second model zoo or training engine remains under `analysis`;
  - useful legacy functionality is either migrated or explicitly rejected;
  - removal is not bundled with unrelated refactoring; and
  - user-facing changes have migration/release notes.

- Evidence/notes:

  - moved the legacy matrix-CNN training/inference/attribution implementation and fitted-estimator
    explanation implementations out of `analysis` into an explicitly temporary internal
    `machine_learning.compatibility` namespace; the old import paths are warning adapters;
  - moved unconstrained sklearn construction/fitting/prediction out of `analysis.compute.ml_metrics`
    while retaining its pure probability, metric, and result-summary functions;
  - added one standardized `FutureWarning` contract with a smftools 3.0 removal target and concrete
    replacements for legacy analysis, AnnData data modules/splits, prototype models, sklearn and
    Lightning wrappers/trainers, AnnData inference, sliding-window helpers, and mutable evaluators;
  - retained pure result-driven analysis and plotting APIs, preserved compatibility behavior and
    hidden historical aliases, and documented XGBoost as legacy-only until it satisfies the
    canonical registry/persistence/evaluation/explanation contracts;
  - added the user-facing ML migration guide and replacement table; no code or artifacts in the
    external Nkg2a projects were changed;
  - validation: focused convergence tests `9 passed`; combined ML/analysis/integration suite
    `276 passed, 7 xfailed`; smoke `106 passed, 1 skipped`; unrestricted full unit-marker suite
    `1500 passed, 9 skipped, 123 deselected, 7 xfailed`; repository-wide Ruff check/format,
    `git diff --check`, and warning-as-error Sphinx HTML build passed. The sandboxed full-unit run's
    20 failures were exclusively the known blocked macOS semaphore calls; the unrestricted rerun
    passed, and generated documentation source outputs were moved to `/private/tmp`.

## P6 — optional scale integrations

These packages begin `DEFERRED`. Move one to `READY` only when its trigger is documented.

### ML-600 — Optional Lightning adapter

- Checkbox: `[ ]`
- Status: `DEFERRED`
- Depends on: ML-303 and a production trigger
- Start triggers:

  - full-state resume is a real need;
  - mixed precision or gradient accumulation is required;
  - multi-device/distributed training is required; or
  - maintaining repeated custom pretraining loops has become material work.

- Deliverables:

  - Lightning system/data-module adapters around the same model/task/data contracts;
  - current supported `lightning.pytorch` namespace and version range;
  - best-checkpoint restoration;
  - conversion/publication of plain inference `state_dict` artifacts; and
  - parity tests against the plain engine.

- Acceptance:

  - model classes remain plain `nn.Module`s;
  - core inference artifacts do not require Lightning;
  - resume restores optimizer/scheduler/callback state as advertised;
  - validation/test loader identity is correct; and
  - Lightning remains in the optional extended dependency.

- Revisit trigger/evidence: —

### ML-601 — Optional tracker adapters

- Checkbox: `[ ]`
- Status: `DEFERRED`
- Depends on: ML-105, ML-400 and a production trigger
- Candidate backends: W&B, MLflow, or local TensorBoard/CSV.
- Start triggers:

  - interactive comparison across many runs is a demonstrated need;
  - team/shared dashboards are required; or
  - sweep/artifact lineage is hard to manage from the local index.

- Deliverables:

  - tracker-neutral event interface;
  - adapters that mirror local parameters, metrics, curves, and safe artifacts;
  - external run references in local manifests;
  - privacy/redaction defaults; and
  - offline/failure behavior.

- Acceptance:

  - tracker failure cannot invalidate a locally completed run;
  - credentials never enter configs, logs, or artifacts;
  - raw molecule/sample identifiers are not uploaded by default;
  - the local workspace is sufficient without the tracker; and
  - enabling a tracker does not change scientific results.

- Revisit trigger/evidence: —

### ML-602 — Optional Hydra application layer

- Checkbox: `[ ]`
- Status: `DEFERRED`
- Depends on: ML-100, ML-501 and a sweep trigger
- Start triggers:

  - dataset × model × task composition is repeatedly cumbersome;
  - systematic multiruns are common; or
  - launcher/sweeper integration is required.

- Deliverables:

  - Hydra config groups that resolve into the ordinary ML plan;
  - explicit precedence over package defaults/user plan/CLI overrides;
  - run directory behavior reconciled with `MLWorkspace`; and
  - sweep parent/child run lineage.

- Acceptance:

  - core schemas and services never require OmegaConf objects;
  - Hydra cannot redirect artifacts outside the active workspace accidentally;
  - each sweep child receives a fully resolved immutable plan; and
  - non-Hydra CLI/Python workflows remain supported.

- Revisit trigger/evidence: —

## P7 — stabilization and release readiness

### ML-700 — Performance and scalability qualification

- Checkbox: `x`
- Status: `DONE`
- Depends on: ML-301, ML-303, ML-500
- Deliverables:

  - benchmark matrix by rows, features, partitions, model backend, and device;
  - peak-memory measurements;
  - partition read throughput and worker-scaling measurements;
  - sklearn materialization refusal thresholds;
  - explanation chunk-size guidance; and
  - performance regression thresholds for representative fixtures.

- Acceptance:

  - published limits distinguish supported, slow, and refused workloads;
  - partition reads demonstrate bounded memory;
  - no benchmark uses validation/test leakage to improve throughput;
  - failures provide actionable memory/batch suggestions; and
  - benchmarks are reproducible from recorded environment/configuration.

- Evidence/notes: benchmark plan drafted 2026-08-04 in
  [ml700_benchmark_plan.md](ml700_benchmark_plan.md) on `test/ml-scale-qualification`. Central
  finding to validate: the `2x`/`3x` memory constants in `partition_dataset._bytes_per_row` are
  analytic and have never been compared to a measured peak, so the package's primary published
  artifact is the measured `peak_rss / estimate` headroom ratio. Harness not yet implemented.

### ML-701 — Documentation and examples

- Checkbox: `x`
- Status: `DONE`
- Depends on: each public work package, and ML-204

#### Delivery sequence (owner decision, 2026-08-06)

Eight live deliverables, split so a single `-W` failure cannot block everything:

1. **PR1 (done)** — architecture/ownership guide and performance/limits page.
2. PR2 — quick starts (sklearn, plain Torch) and the ML-plan schema reference.
3. PR3 — guidance pages: splits/balancing/masks, artifacts/promotion/trust, interpretability
   method selection.
4. PR4 — experiment-local and project-level tutorials, curated API reference, acceptance sweep.

#### API-reference blocker, measured 2026-08-06

The API page is **not** a simple addition and is deliberately last. Adding all 66 documentable ML
modules to `autosummary` **hard-aborts** the docs build: ten modules subclass bases from packages
in `autodoc_mock_imports` (`nn.Module`, `pl.LightningDataModule`, `Dataset`, `TransformerMixin`),
and under Python 3.12 subclassing a `Mock` raises `TypeError: __type_params__ must be set to a
tuple`. Sphinx does not degrade to a warning; it raises `Extension error` and stops.

Unmocking `torch` and `sklearn` — both core dependencies, so genuinely installed — is not the fix.
It removes the abort but leaves eight ML modules failing *and* regresses three currently-passing
modules (`analysis.compute.ml_cnn`, `analysis.compute.ml_metrics`,
`preprocessing.flag_duplicate_reads`), taking the build from 2 warnings to 35.

Owner decision: document a **curated subset** covering the canonical public surface and skip the
ten, recording the gap rather than restructuring merged ML-302/ML-303 model code for a docs-only
benefit.

Separately noted for PR4: `machine_learning.__all__` currently exports only eight subpackages and
omits `artifacts`, `contracts`, `manifests`, `plan`, `selection`, `splitting`, and `workspace` —
the core contract modules a user needs. The declared public surface is narrower than the real one.

- Deliverables:

  - architecture/ownership guide;
  - ML-plan schema reference and examples;
  - experiment-local and project-level tutorials;
  - sklearn and PyTorch quick starts;
  - split/balancing/mask guidance;
  - artifact, promotion, and trust/security guide;
  - interpretability method-selection guide;
  - pretrained/fine-tuned lineage guide when ML-304 lands; and
  - generated API documentation entries.

- Acceptance:

  - examples use partitioned stores and stable sample IDs;
  - docs distinguish proposed/optional features from guaranteed behavior;
  - every CLI change appears in `docs/source/cli.md`;
  - docs build succeeds with warnings treated as errors; and
  - optional dependency installation instructions match `pyproject.toml`.

- Evidence/notes: —

### ML-702 — Security, compatibility, and release readiness

- Checkbox: `x`
- Status: `DONE`
- Depends on: ML-204, ML-504, ML-700, ML-701
- Deliverables:

  - artifact trust/threat review;
  - path containment and symlink behavior review;
  - supported Python/sklearn/Torch/Lightning version matrix;
  - public API and deprecation review;
  - final migration notes;
  - changelog/release-note entry; and
  - full targeted and package validation.

- Acceptance:

  - untrusted pickle/joblib loading is prohibited or requires explicit trusted opt-in;
  - manifest/path inputs cannot escape the active workspace unexpectedly;
  - supported dependency versions have round-trip artifact tests;
  - backward-incompatible changes follow release policy;
  - unit, integration, end-to-end, lint, type, and docs checks required by touched areas pass; and
  - known limitations and deferred integrations are documented.

- Evidence/notes: —

## Suggested PR sequence

This sequence favors small reviewable changes. Work packages may take multiple PRs, but unrelated
phases should not be bundled merely to reduce PR count.

| PR | Primary content | Expected behavior change |
|---|---|---|
| 1 | ML-000/ML-001 decision and migration records | None |
| 2 | ML-002 behavioral fixtures/tests for current defects | Tests/characterization only |
| 3 | ML-100 plan schema and validation | New plan parser; no training |
| 4 | ML-101 input/mask/label schemas | New internal contracts only |
| 5 | ML-102 dataset/split manifests | New provenance structures |
| 6 | ML-103 workspace resolver | New path resolution; no trainer |
| 7 | ML-104/ML-105 artifact manifest/publication primitives | New local artifact APIs |
| 8 | ML-200/ML-201 metadata planning and split resolution | New dry-run data planning |
| 9 | ML-202 partition-aware batches/materialization | New data plane |
| 10 | ML-203 transforms/balancing | New train-only preprocessing policies |
| 11 | ML-300 predictor protocol/registry | New backend-neutral interface |
| 12 | ML-301 sklearn vertical slice | First supported end-to-end training path |
| 13 | ML-302 model-family consolidation | New canonical plain Torch models |
| 14 | ML-303 plain-PyTorch vertical slice | Canonical neural training path |
| 15 | ML-400 prediction/metric standardization | Unified evaluation outputs |
| 16 | ML-401/ML-402 classical explanations | Common explanation artifacts |
| 17 | ML-403 neural explanations | Optional Captum-backed explanations |
| 18 | ML-500/ML-501 services and approved user interface | User-facing orchestration |
| 19 | ML-502 plotting integration | Reproducible analysis plots |
| 20+ | ML-503/ML-504 consumer migration and staged deprecation | Legacy path transition |
| 21 | ML-204 streaming training reads | **Training works at production scale**; refusal ceiling lifted for streaming-capable families |
| 22 | ML-700 published limits (completes after ML-204) | Performance limits reflect the streaming reality |
| 23 | ML-701 documentation | Docs describe the post-streaming system |
| 24 | ML-702 security/compatibility/release | Release readiness |

## Validation matrix

Each PR runs the smallest relevant subset, while phase exits run the accumulated matrix using the
repository's selected interpreter policy.

| Area | Minimum evidence |
|---|---|
| Schemas/plans | Unit tests for valid/invalid documents, canonical serialization, version rejection, and reference validation |
| Workspace/artifacts | Unit tests for scope selection, path containment, atomic publication, checksum conflicts, portability, and index rebuild |
| Splits | Unit/property-style tests for group disjointness, determinism, explicit membership, impossible stratification, and class summaries |
| Partition data | Integration tests using a tiny real partition store; bounded reads, ordering, masks, coordinates, and multi-worker coverage |
| sklearn | Fit/apply/round-trip tests for NB, logistic regression, and RF; class order, pipeline leakage, memory refusal, and trusted loading |
| PyTorch | Forward shapes, masks, one-step optimization, best-state restore, deterministic fixture, state-dict round trip, CPU fallback |
| Evaluation | Known prediction-table metrics, threshold/calibration isolation, folds, multiclass behavior, and prevalence reporting |
| Interpretability | Synthetic known-feature tests, baseline provenance, shape alignment, capability failures, and optional-dependency paths |
| CLI/services | Core tests independent of Click plus parser tests and experiment/project scope end-to-end tests |
| Migration | Before/after membership, predictions, metrics, artifact, warning, and documented behavior comparisons |
| Docs | `sphinx-build -W -b html docs/source docs/_build/html` for any docstring or docs change |
| Package | Appropriate smoke/unit/integration/e2e markers, `ruff check`, format check, and configured type checks |

## Definition of done for a public vertical workflow

A workflow is not production-ready until all applicable statements are true:

- [ ] The user can validate and dry-run a versioned ML plan.
- [ ] Experiment/project scope and output root are explicit.
- [ ] Dataset and split manifests exist before fitting.
- [ ] Group-disjointness and class feasibility pass.
- [ ] Memory/materialization estimates pass.
- [ ] Label and mask schemas are persisted.
- [ ] Train-only transforms and balancing are enforced.
- [ ] Training or application produces immutable run records.
- [ ] The model artifact round-trips and validates input compatibility.
- [ ] Predictions and metrics are reproducible from stored tables.
- [ ] Explanations, if requested, record baseline/cohort/target/method semantics.
- [ ] Plots can be regenerated from stored result tables.
- [ ] Local provenance is sufficient without W&B/MLflow.
- [ ] Security/trust behavior for serialized objects is explicit.
- [ ] Tests, documentation, and migration notes are complete.

## Decision log

Items marked `ACCEPTED` represent the current audit recommendation but may still be revised through
an explicit ledger update before implementation.

| ID | Decision | Current disposition | Blocks | Rationale/notes |
|---|---|---|---|---|
| D-001 | Canonical reusable ML owner | `ACCEPTED`: `smftools.machine_learning` | P1+ | Keeps model/training/data infrastructure out of pure `analysis`. |
| D-002 | HMM ownership | `ACCEPTED`: remain task-specific | P1+ | Reuse artifact primitives, not HMM class/key semantics. |
| D-003 | Neural model API | `ACCEPTED`: plain `nn.Module` | P3 | Lightning may wrap models later. |
| D-004 | ML output roots and experiment dirname | `ACCEPTED`: scope-resolved local roots | ML-103 | Experiment runs use `<ExperimentConfig.output_directory>/ml_outputs/`; project runs use `<project_dir>/project_outputs/ml/`. Models never resolve paths themselves. |
| D-005 | User configuration | `ACCEPTED`: separate versioned ML plan | ML-100 | Avoid nested config in flat CSV and human-only `project.yaml`. |
| D-006 | Typed config implementation | `ACCEPTED`: stdlib dataclasses plus explicit validators | ML-100/101 | Use ordinary immutable dataclasses, strict mapping/YAML parsers, unknown-key rejection, and schema-version dispatch. Do not add a schema-framework dependency until nested-union complexity demonstrates a need. |
| D-007 | Public namespace | `ACCEPTED`: canonical `smftools.machine_learning`, compatible `smftools.ml` alias | ML-001/500 | New documented APIs use the descriptive namespace. Preserve the existing lazy alias and avoid broad top-level symbol re-exports during migration. |
| D-008 | Predictor abstraction | `ACCEPTED`: protocol + adapters | ML-300 | Avoid shared sklearn/Torch inheritance hierarchy. |
| D-009 | Initial sklearn models | `ACCEPTED`: NB, logistic, RF | ML-301 | Low-cost interpretable baselines plus nonlinear comparator. |
| D-010 | Initial neural family | `ACCEPTED`: residual-dilated 1D CNN recipe | ML-302/303 | Freeze a named versioned recipe derived from the committed `ResidualDilatedCNN1d` defaults, with a small validated override surface. Transformer/hybrid candidates require ML-001 inventory evidence before adoption. |
| D-011 | Split default | `ACCEPTED`: biological group-disjoint | ML-201 | Molecule-row random split is unsafe for generalization claims. |
| D-012 | Validation/test balancing | `ACCEPTED`: natural primary prevalence | ML-203/400 | Balance training only; separate sensitivity evaluation if needed. |
| D-013 | Artifact authority | `ACCEPTED`: immutable local manifests | ML-104/105 | Trackers remain optional mirrors. |
| D-014 | sklearn persistence | `ACCEPTED`: `skops.io` preferred, unsafe formats gated | ML-104/301/702 | Package-produced supported sklearn pipelines use checksummed `.skops` artifacts and an explicit reviewed-type allowlist. Pickle/joblib loading is legacy/trusted-only behind an explicit unsafe-load flag; ONNX may be an optional inference export but is not authoritative. Record exact sklearn/skops/numpy/scipy versions and reject unsupported environments by default. |
| D-015 | Index format | `ACCEPTED`: rebuildable Parquet indexes first | ML-105 | Authoritative per-artifact JSON manifests rebuild portable run/model Parquet indexes. Add SQLite only if measured concurrent query/update requirements justify a second index backend. |
| D-016 | Project model promotion | `ACCEPTED`: project-owned immutable copy | ML-105/500 | Promotion atomically copies a verified experiment/shared artifact into the project workspace, preserving source model ID, URI/path, checksum, and lineage. Loading by external pointer is allowed but is not project promotion. |
| D-017 | Initial user interface | `ACCEPTED`: Python service first | ML-501 | Stabilize plan validation, dry-run, and one vertical workflow as normal Python APIs before adding thin experiment/project Click commands. |
| D-018 | Lightning adoption | `DEFERRED` | ML-600 | Requires a documented scale/resume trigger. |
| D-019 | Tracker adoption | `DEFERRED` | ML-601 | Requires a documented collaboration/search trigger. |
| D-020 | Hydra adoption | `DEFERRED` | ML-602 | Requires a documented composition/sweep trigger. |
| D-021 | Legacy removal timing | `ACCEPTED`: warn throughout 2.x, remove in 3.0 | ML-504/702 | The real Nkg2a consumer has migrated and no committed package consumer depends on the legacy execution paths. Standardized `FutureWarning` messages identify the canonical replacement and 3.0 removal boundary; ML-702 must verify no active consumer remains before deletion. |
| D-022 | Canonical explanation vocabulary | `ACCEPTED`: precise method names | ML-401+ | Do not use ambiguous `GradSHAP`/`AttentionCAM` labels in manifests. |
| D-023 | First vertical acceptance workflow | `ACCEPTED`: project deaminase binary-label classification | ML-001+ | Use deaminase `C_site_binary` as an accessibility channel, an explicit fixture `activity_status` mapping, `(experiment_uid, Sample)` group holdout, Bernoulli NB first, then the residual CNN under shared contracts. The fixture label is not a package default. |
| D-024 | PR sizing and review ownership | `ACCEPTED`: one work package per PR | All | The maintainer approves scientific/public-contract decisions; the author owns tests and evidence. Persistent schemas remain complete vertical changes rather than being split from their readers/validators. |
| D-025 | Mixed-modality projects and channel semantics | `ACCEPTED`: registry modality plus explicit biological channel roles | ML-100–ML-102/ML-200–ML-202/ML-300/ML-400+ | Preserve each experiment's registered modality. Deaminase defaults to C accessibility; conversion defaults to GpC accessibility plus CpG endogenous methylation but may explicitly declare both as accessibility; direct SMF may independently use A, GpC, and/or CpG. Physical site context never silently determines biological meaning. |

## Risk register

| ID | Risk | Likelihood/impact | Mitigation | Status |
|---|---|---|---|---|
| R-001 | Scope expands into a complete rewrite before any usable workflow lands | High/high | Preserve vertical slices and PR order; defer optional integrations. | Open |
| R-002 | Biological leakage through molecule-level splitting or fitted preprocessing | High/high | Split manifests, group-disjoint tests, train-only transform protocol. | Open |
| R-003 | Partition adapter silently materializes an unsafe dataset | Medium/high | Metadata preflight, explicit memory budget, refusal tests. | Open |
| R-004 | Mask polarity/shape differs across models | High/high | Versioned named mask schema and compatibility tests before consolidation. | Open |
| R-005 | Saved models cannot be reconstructed after dependency upgrades | Medium/high | Record versions, plain Torch state dicts, schema versions, round-trip compatibility tests. | Open |
| R-006 | sklearn serialization executes untrusted code | Medium/high | Safer formats where supported; explicit trust gate; never implicit load. | Open |
| R-007 | Experiment/project artifacts write into the wrong ownership scope | Medium/high | One workspace resolver, containment checks, multi-experiment scope rejection. | Open |
| R-008 | Trackers become the only provenance record or expose sensitive metadata | Medium/high | Local authority, opt-in adapters, redaction defaults, offline tests. | Open |
| R-009 | Interpretability output is overclaimed biologically | High/high | Method capability table, baseline provenance, stability checks, explicit limitations. | Open |
| R-010 | Legacy consumers break during consolidation | Medium/high | Inventory, adapters, deprecation window, representative migration before removal. | Open |
| R-011 | Optional dependencies break import/test collection | Medium/medium | Lazy imports through optional dependency helpers and dependency-profile tests. | Open |
| R-012 | Architecture configurability creates unsupported combinations | Medium/medium | Small validated parameter surfaces plus named immutable recipes. | Open |
| R-013 | Cross-experiment identities become non-portable | Medium/high | Registry-based stable IDs, relative references, source digests, move tests. | Open |
| R-014 | Test suite becomes too slow for normal development | Medium/medium | Tiny fixtures, marker discipline, unit/integration separation, no external data. | Open |
| R-015 | A mixed-modality model learns modality or missing-channel patterns instead of biology | High/high | Explicit channel roles and availability masks, class-by-modality feasibility checks, group-disjoint splits, modality-stratified metrics, and held-out-modality tests when cross-modality generalization is claimed. | Open |

## Program completion definition

The core program is complete when:

1. Reusable trainable models, data adapters, execution, inference, explanation computation, and
   artifact contracts have one canonical owner under `smftools.machine_learning`.
2. `smftools.analysis` contains only pure ML result computation and explicit-output-path plotting,
   not a second model zoo or training engine.
3. A versioned ML plan declares experiment/project scope, datasets, labels, samples/groups, splits,
   balancing, model recipes, and job actions with deterministic validation.
4. Experiment-local and cross-experiment project jobs resolve outputs through one workspace
   contract and never infer or cross ownership boundaries.
5. Dataset and split manifests are inspectable before matrices are read, and automated tests prove
   biological-group disjointness.
6. Partition-store training/application reads are bounded and retain stable molecule, feature,
   coordinate, label, and mask alignment.
7. Experiment modality and each ordered channel's physical source, site context, biological role,
   applicability, and observed/design masks are explicit and part of model compatibility.
8. Mixed-modality datasets either harmonize equivalent biological roles or use an explicit
   union-channel schema, and evaluation reports class support and performance by modality.
9. Train-only preprocessing/balancing and natural held-out evaluation prevalence are enforced.
10. Bernoulli Naive Bayes, logistic regression, random forest, and one plain-PyTorch recipe produce
   the same prediction/metric/run artifact contracts.
11. Model family configuration is validated, resolved, versioned, and reproducible from an immutable
   recipe artifact.
12. Pretrained encoders and fine-tuned task models, if enabled, have explicit parent-child lineage
    and are benchmarked against from-scratch training.
13. Run, checkpoint, model, prediction, metric, and explanation artifacts are immutable,
    checksummed, portable, and locally understandable without a hosted tracker.
14. Interpretability dispatch and manifests identify method capability, target, baseline/background,
    cohort, mask policy, feature axes, and implementation version.
15. Users can dry-run, train, apply, evaluate, explain, and plot through the approved Python/CLI
    surface without accidental retraining or mutable "latest" model discovery.
16. At least one active real-world consumer is migrated with accepted before/after split,
    prediction, metric, artifact, usability, and performance evidence.
17. Duplicate legacy training/model/attribution paths are removed or covered by a documented,
    tested deprecation window.
18. Focused, unit, integration, end-to-end, optional-dependency, relocation, failure-injection,
    security, lint, format, documentation, and applicable performance gates pass.

ML-600 through ML-602 are not required for core completion unless their adoption triggers are
accepted. If they remain deferred, their dependency behavior and rationale must be documented.

## Implementation status

This is the operational status ledger. Update it with PR number, merge commit, and concise evidence
when a work package lands.

| Work package | Status | Branch | PR/merge/evidence |
|---|---|---|---|
| ML-000 | `DONE` | `feature/ml-contract-decisions` | Revalidated `b8b5a90`; decisions recorded in D-004/D-006/D-007/D-010/D-014–D-017/D-023–D-025; local ignored ledger reviewed interactively |
| ML-001 | `DONE` | `feature/ml-behavior-inventory` | Local ignored `ml_behavior_inventory.md`; committed behavior mapped separately from external/uncommitted candidates |
| ML-002 | `DONE` | `test/ml-behavior-baseline` | Merged to `main` in PR #433 (`fb38b0e`); targeted: 9 passed, 7 strict xfailed; full unit suite: 1236 passed, 9 skipped, 7 xfailed; Ruff check/format passed |
| ML-100 | `DONE` | `feature/ml-plan-schema` | Merged to `main` in PR #434 (`af3b649`); typed plan, strict loaders, modality-aware channels, and all action declarations validated |
| ML-101 | `DONE` | `feature/ml-input-contracts` | Merged to `main` in PR #435 (`95b654e`); versioned input/label/seven-mask/capability contracts validated |
| ML-102 | `DONE` | `feature/ml-dataset-split-manifests` | Merged to `main` in PR #436 (`d80830f`); immutable path-neutral dataset/source/observation/split manifests |
| ML-103 | `DONE` | `feature/ml-workspace-resolution` | Merged to `main` in PR #437 (`606c653`); scope-safe read-only workspace and run-path resolution |
| ML-104 | `DONE` | `feature/ml-artifact-schemas` | Merged to `main` in PR #438 (`e4dfed7`); immutable tracker-neutral artifact manifest schemas |
| ML-105 | `DONE` | `feature/ml-artifact-publication` | Merged to `main` in PR #439 (`d31f94c`); immutable publication and rebuildable indexes |
| ML-200 | `DONE` | `feature/ml-data-selection-plan` | Merged to `main` in PR #440 (`15bcc01`); metadata-only selection, channel resolution, identity/sizing, and stale-plan fingerprints |
| ML-201 | `DONE` | `feature/ml-group-splits` | Merged to `main` in PR #441 (`1b30922`); deterministic group-disjoint split resolution and immutable manifest conversion |
| ML-202 | `DONE` | `feature/ml-partition-dataset` | Merged to `main` in PR #442 (`8d4b18d`); bounded mixed-modality partition batches and memory-refusing materialization |
| ML-203 | `DONE` | `feature/ml-train-transforms-balancing` | Merged to `main` in PR #443 (`ad6ddf0`); checksummed train-only transforms and balancing with backend-appropriate adapters |
| ML-204 | `DONE` | `feature/ml-streaming-training` | Merged to `main` in PR #458 (`23a5dfc`); streaming transform fit, metadata-only balancing, and both backends fitting from batches. Training is no longer capped by `max_materialization_bytes` for incremental sklearn families or Torch |
| ML-205 | `PROPOSED` | `feature/ml-activation-memory` | Added 2026-08-06, discovered by ML-700. Torch process RSS is 12,557x the data batch estimate because activations are unmodelled; no supported/slow/refused boundary for Torch training can be derived from the data estimator. Gated on a start trigger: a production-scale OOM, or a workflow needing a training-memory preflight |
| ML-300 | `DONE` | `feature/ml-predictor-registry` | Merged to `main` in PR #444 (`b993688`); predictor protocols/adapters and explicit modality-aware built-in registry |
| ML-301 | `DONE` | `feature/ml-sklearn-vertical` | Merged to `main` in PR #445 (`1a0b54c`); sklearn train/apply and safe immutable `.skops` artifacts |
| ML-302 | `DONE` | `feature/ml-torch-model-families` | Merged to `main` in PR #446 (`12afb14`); canonical residual/dilated CNN family, strict recipe/config/mask contracts, analysis compatibility aliases, and lazy optional orchestration imports |
| ML-303 | `DONE` | `feature/ml-torch-vertical` | Merged to `main` in PR #447 (`8fb7405`); plain-Torch train/validate/test/apply and safe immutable state-dict artifact vertical |
| ML-304 | `PROPOSED` | `feature/ml-pretrained-lineage` | — |
| ML-400 | `DONE` | `feature/ml-evaluation-contract` | Merged to `main` in PR #448 (`cee787c`); shared prediction/evaluation/history/fold contracts, threshold isolation, mixed-modality summaries, and redacted table export |
| ML-401 | `DONE` | `feature/ml-explanation-contract` | Merged to `main` in PR #449 (`6a63a7a`); canonical request/result contracts, capability preflight, training-only backgrounds, and ML-104 artifact mapping |
| ML-402 | `DONE` | `feature/ml-classical-explanations` | Merged to `main` in PR #450 (`af77bd1`); native NB/linear, held-out permutation, lazy TreeSHAP, and transformed-feature provenance |
| ML-403 | `DONE` | `feature/ml-neural-explanations` | Merged to `main` in PR #451 (`7c1bd5d`); bounded mask-aware Captum input/layer explanations |
| ML-500 | `DONE` | `feature/ml-job-services` | Merged to `main` in PR #452 (`d69692c`); backend-neutral job services and terminal lifecycle publication |
| ML-501 | `DONE` | `feature/ml-user-orchestration` | Merged to `main` in PR #453 (`8e693d7`); immutable Python workflow dry-run with explicit scope, full preflight summaries, model-schema validation, and dependency reporting |
| ML-502 | `DONE` | `feature/ml-analysis-results` | Merged to `main` in PR #454 (`1e49998`); pure canonical-result tables and explicit-path reproducible plots |
| ML-503 | `DONE` | `feature/ml-consumer-migration` | Merged to `main` in PR #455 (`fbb652d`); real Nkg2a RF consumer migrated with immutable three-fold parity evidence |
| ML-504 | `DONE` | `fix/ml-legacy-convergence` | Merged to `main` in PR #456 (`f74a42f`); 3.0 warning window, analysis ownership convergence, migration guide, and compatibility tests validated |
| ML-600 | `DEFERRED` | `feature/ml-lightning-adapter` | Requires production trigger |
| ML-601 | `DEFERRED` | `feature/ml-tracker-adapters` | Requires tracking/collaboration trigger |
| ML-602 | `DEFERRED` | `feature/ml-hydra-application` | Requires composition/sweep trigger |
| ML-700 | `DONE` | `test/ml-scale-qualification` | Merged to `main` in PR #457 (`9b5bf2f`, harness) and PR #459 (`4fe69a1`, published limits). Sweeps A/C/D and the refusal boundary ran; `tests/acceptance/ml_scale_thresholds.json` carries the limits, taxonomy, and regression thresholds, guarded per PR by `test_ml_scale_thresholds.py`. Estimator verdict: the `2x`/`3x` constants hold, no change warranted. Documented gaps: no CUDA device on the measuring host, and explanation memory across chunk sizes unmeasured |
| ML-701 | `DONE` | `feature/ml-documentation*` | Merged to `main` in PR #461 (architecture, performance), #462 (splits/masks, interpretability, artifacts/trust), and #463 (curated API reference, tutorials). Nine ML pages, `sphinx-build -W` green. Sixteen tests pin documented claims to the code. API page covers 61 of 66 modules; the five omitted are deprecated or gated behind unbuilt Lightning, and `test_ml_api_surface.py` fails if that set grows |
| ML-702 | `DONE` | `feature/ml-program-acceptance` | Merged to `main` in PR #464 (`9c4195f`). Security and path-containment reviewed adversarially with 17 committed probes; supported-version matrix, deprecation review, and `release-notes/2.19.0.md` published. **Not yet CI-verified** — merges #460, #463, and #464 produced no workflow run, so the program is complete but unvalidated on Python 3.11, `storage-minimums`, `lint`, and `build`. One contract question left open: unproduced mask kinds are silently dropped rather than rejected |

## Change log

Add one row whenever this ledger's scope, status, decisions, or sequencing changes.

| Date | Work package/decision | Change | Author/PR |
|---|---|---|---|
| 2026-07-30 | Ledger creation | Converted both ML audits into phased work packages, decision gates, acceptance criteria, validation matrix, and risks. | — |
| 2026-07-30 | Plan-format alignment | Aligned metadata, baseline, findings, contracts, rollout checkpoints, backlog, file map, completion definition, and implementation status with the semantic-DAG reference plan. No package code implemented. | — |
| 2026-07-30 | ML-000 current-main revalidation | Revalidated the audit against `b8b5a90`, distinguished committed behavior from the audit's uncommitted external-worktree additions, selected the first vertical acceptance workflow, and resolved the P0 architecture/output/configuration/persistence/index/promotion/interface decisions. No package code changed. | `feature/ml-contract-decisions` |
| 2026-07-30 | D-025 mixed-modality input contract | Confirmed that projects may mix modalities; separated physical site context from biological channel role; added deaminase, conversion, and direct-SMF defaults plus mixed-modality masks, compatibility, split, evaluation, and confounding requirements. Marked ML-000 done and ML-001 ready. | local ignored ledger |
| 2026-07-30 | ML-001 behavior inventory | Inventoried committed ML symbols, consumers, input/output behavior, artifacts, migration destinations, and regression targets B-001–B-012. External/uncommitted candidates remain separate. Marked ML-001 done and ML-002 ready. | local ignored `ml_behavior_inventory.md` |
| 2026-07-30 | ML-002 local test baseline | Added deterministic binary/multiclass and partition fixtures, seven strict known-defect xfails, and passing mask/optimization/round-trip/train-only-transform/projection/optional-import contracts. Targeted suite, full unit suite, and Ruff pass. | `test/ml-behavior-baseline` |
| 2026-07-30 | ML-002 merged / P0 complete | Verified PR #433 merge commit `fb38b0e` contains the ML-002 baseline, marked ML-002 and P0 done, and created `feature/ml-plan-schema` from the merged mainline without upstream tracking. | `feature/ml-plan-schema` |
| 2026-07-30 | ML-100 local implementation | Added the immutable typed plan, strict loaders/validation, modality-aware ordered channels, named split/balancing/model/job declarations, canonical round trips/hashes, and focused fixtures for all five actions. Focused ML, legacy ML, full unit, and Ruff checks pass. | `feature/ml-plan-schema` |
| 2026-07-30 | ML-100 merged / ML-101 started | Verified PR #434 merge commit `af3b649` contains the typed ML plan, marked ML-100 done, and created `feature/ml-input-contracts` from the merged mainline without upstream tracking. | `feature/ml-input-contracts` |
| 2026-07-30 | ML-101 local implementation | Added immutable input, label, seven-mask, and predictor-capability contracts; exact schema compatibility/hashes; phase/consumer checks; and mask shape/relationship validation. Focused, combined ML, full unit, docs, and Ruff checks pass. | `feature/ml-input-contracts` |
| 2026-07-30 | ML-101 merged / sequence correction / ML-102 started | Verified PR #435 merge commit `95b654e`, marked ML-101 done, corrected the PR sequence so path-neutral dataset/split manifests follow their ML-101 dependency before ML-103 workspace resolution, and created `feature/ml-dataset-split-manifests` without upstream tracking. | `feature/ml-dataset-split-manifests` |
| 2026-07-30 | ML-102 local implementation | Added versioned path-neutral dataset/source/observation/split manifests with deterministic identities, summaries, strict round trips, group leakage prevention, portable source references, and stale/tamper checks. Focused ML, all ML, full unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-dataset-split-manifests` |
| 2026-07-30 | ML-102 merged / ML-103 started | Verified PR #436 merge commit `d80830f`, marked ML-102 done, and created `feature/ml-workspace-resolution` from the merged mainline without upstream tracking. | `feature/ml-workspace-resolution` |
| 2026-07-30 | ML-103 local implementation | Added canonical experiment/project ML workspace resolution, deterministic run path bundles, portable contained references, all-action dry-run reports, and cross-scope/traversal safeguards. Focused ML, all ML, full unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-workspace-resolution` |
| 2026-07-30 | ML-103 merged / ML-104 started | Verified PR #437 merge commit `606c653`, marked ML-103 done, and created `feature/ml-artifact-schemas` from the merged mainline without upstream tracking. | `feature/ml-artifact-schemas` |
| 2026-07-30 | ML-104 local implementation | Added tracker-neutral immutable run/model/checkpoint/prediction/explanation schemas, lifecycle and failure provenance, content identities, lineage, serialization trust policy, and resolved-config run paths. Focused, all-ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-artifact-schemas` |
| 2026-07-30 | ML-104 merged / ML-105 started | Verified PR #438 merge commit `e4dfed7`, marked ML-104 done, and created `feature/ml-artifact-publication` from the merged mainline without upstream tracking. | `feature/ml-artifact-publication` |
| 2026-07-30 | ML-105 local implementation | Added staged and atomically published immutable run/model bundles, exact checksum inventories, conflict-safe retries, deterministic manifest-derived indexes, validated mutable model aliases, relocation checks, and bounded staging/lock cleanup. Focused, all-ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-artifact-publication` |
| 2026-07-30 | ML-105 merged / ML-200 started | Verified PR #439 merge commit `d31f94c`, marked ML-105 done, and created `feature/ml-data-selection-plan` from the merged mainline without upstream tracking. | `feature/ml-data-selection-plan` |
| 2026-07-30 | ML-200 local implementation | Added metadata-only experiment/project selection, modality-aware biological channel resolution, stable molecule/group/label identity, dry-run sizing, and catalog/membership fingerprints. Focused, all-ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-data-selection-plan` |
| 2026-07-30 | ML-200 merged / ML-201 started | Verified PR #440 merge commit `15bcc01`, marked ML-200 done, and created `feature/ml-group-splits` from the merged mainline without upstream tracking. | `feature/ml-group-splits` |
| 2026-07-30 | ML-201 local implementation | Added exact explicit groups, seeded scalable stratified groups, leave-one-group-out folds, locked evaluation roles, feasibility/confounding diagnostics, and validated immutable split-manifest conversion. Focused, all-ML, full-unit, Ruff, formatting, and diff checks pass. | `feature/ml-group-splits` |
| 2026-08-01 | ML-201 docs validation | Completed the warning-as-error Sphinx build with network access for intersphinx; generated source/build artifacts were moved out of the worktree afterward. | `feature/ml-group-splits` |
| 2026-08-01 | ML-201 merged / ML-202 started | Verified PR #441 merge commit `1b30922`, marked ML-201 done, and created `feature/ml-partition-dataset` from the merged mainline without upstream tracking. | `feature/ml-partition-dataset` |
| 2026-08-01 | ML-202 local implementation | Added manifest-bound stage-spine sources, deterministic bounded batches, mixed-modality union-channel projection, observed/availability/design/padding masks, multi-worker sharding, and memory-refusing sklearn materialization. Focused, all-ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-partition-dataset` |
| 2026-08-01 | ML-202 merged / ML-203 started | Verified PR #442 merge commit `8d4b18d`, marked ML-202 done, and created `feature/ml-train-transforms-balancing` from the merged mainline without upstream tracking. | `feature/ml-train-transforms-balancing` |
| 2026-08-01 | ML-203 local implementation | Added checksummed train-only fitted transforms shared by sklearn/Torch, flattened sklearn mask indicators, channel-first Torch tensors with separate masks, persisted-class-order weights/sampling, deterministic training resampling, and natural primary evaluation cohorts with named sensitivity alternatives. Focused, all-ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-train-transforms-balancing` |
| 2026-08-01 | ML-203 merged / P2 complete / ML-300 started | Verified PR #443 merge commit `ad6ddf0`, marked ML-203 and the core data-plane phase done, and created `feature/ml-predictor-registry` from merged main without upstream tracking. | `feature/ml-predictor-registry` |
| 2026-08-01 | ML-300 local implementation | Added backend-neutral predictor/loader protocols, composition-based sklearn/Torch adapters, immutable modality/channel-aware recipes, and a deterministic explicit registry with the three approved sklearn estimator builders. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-predictor-registry` |
| 2026-08-01 | ML-300 merged / ML-301 started | Verified PR #444 merge commit `b993688`, marked ML-300 done, and created `feature/ml-sklearn-vertical` from merged main without upstream tracking. | `feature/ml-sklearn-vertical` |
| 2026-08-01 | ML-301 local implementation | Added shared sklearn training/application contracts for NB/logistic/RF, capability-gated incremental fitting, bounded non-incremental materialization, ordered prediction records, exact-versioned and explicitly allowlisted `.skops` publication/loading, and a complete immutable run/model round trip. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-sklearn-vertical` |
| 2026-08-01 | ML-301 merged / ML-302 started | Verified PR #445 merge commit `1a0b54c`, marked ML-301 done, and created `feature/ml-torch-model-families` from merged main without upstream tracking. Confirmed the accepted first family remains the committed residual/dilated CNN; transformer, hybrid, and domain-adversarial candidates remain unregistered pending consumer evidence. | `feature/ml-torch-model-families` |
| 2026-08-01 | ML-302 local implementation | Migrated the active residual/dilated CNN to canonical plain-Torch ownership with a strict configurable architecture, named registry recipe, explicit separate masks, exact reconstruction/state-dict tests, analysis compatibility aliases, and a lazy optional-import boundary. Deferred transformer/hybrid/domain-adversarial/shared-encoder promotion pending task evidence. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-torch-model-families` |
| 2026-08-01 | ML-302 merged / ML-303 started | Verified PR #446 merge commit `12afb14`, marked ML-302 done, and started the plain-PyTorch vertical slice from merged main. | `feature/ml-torch-vertical` |
| 2026-08-01 | ML-303 local implementation | Added a strict plain-Torch task/training engine, validation-only early stopping and best-state restoration, locked post-selection test evaluation, shared transforms/masks/balancing, sklearn-shaped prediction records, canonical metadata plus safe `weights_only` state-dict publication/loading, and lazy optional-Lightning package exports. Explicitly deferred resumable optimizer checkpoints. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-torch-vertical` |
| 2026-08-01 | ML-303 merged / ML-400 started | Verified PR #447 merge commit `8fb7405`, marked ML-303 merged, retained ML-304 behind its corpus/benchmark gate, and started the shared backend-neutral evaluation contract from merged main. | `feature/ml-evaluation-contract` |
| 2026-08-01 | ML-400 local implementation | Unified sklearn/Torch prediction rows; added reproducible binary/multiclass, pooled/per-modality evaluation; validation-bound threshold/calibration provenance; ROC/PR/calibration/confusion/balance records; fold uncertainty; non-fabricated training histories; and redacted exports. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-evaluation-contract` |
| 2026-08-01 | ML-400 merged / ML-401 started | Verified PR #448 merge commit `cee787c`, marked ML-400 merged, and started backend-neutral explanation request/result contracts from merged main. | `feature/ml-explanation-contract` |
| 2026-08-01 | ML-401 local implementation | Added canonical content-addressed explanation requests/results, capability preflight, training-only checksummed backgrounds, schema-aligned attribution axes, and canonical Zarr/Parquet artifact mapping. Focused ML, full-unit, docs, Ruff, formatting, and diff checks pass. Captum/SHAP execution remains scoped to ML-402/ML-403. | `feature/ml-explanation-contract` |
| 2026-08-01 | ML-401 merged / ML-402 started | Verified PR #449 merge commit `6a63a7a`, marked ML-401 merged, and started classical explanation adapters from merged main. | `feature/ml-classical-explanations` |
| 2026-08-01 | ML-402 local implementation | Added canonical native NB/linear explanations, held-out deterministic permutation importance, lazy background-aware TreeSHAP, and explicit transformed-feature provenance. Preserved legacy analysis entry points pending the planned migration phase. Focused ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-classical-explanations` |
| 2026-08-01 | ML-402 merged / ML-403 started | Verified PR #450 merge commit `af77bd1`, marked ML-402 merged, and started mask-aware neural explanation adapters from merged main. | `feature/ml-neural-explanations` |
| 2026-08-01 | ML-403 local implementation | Added bounded, mask-aware Captum input-gradient and declared-layer GradCAM adapters with checksummed training backgrounds, deterministic stochastic attribution, convergence evidence, and immutable provenance. Attention explanations remain gated until a registered attention-capable model exists. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-neural-explanations` |
| 2026-08-01 | ML-403 merged / ML-500 started | Verified PR #451 merge commit `7c1bd5d`, marked P4 complete, and started backend-neutral job services from merged main. | `feature/ml-job-services` |
| 2026-08-01 | ML-500 local implementation | Added immutable model resolution, backend-neutral scientific dispatch, read-only job previews, scoped atomic run lifecycle execution, structured logging, cancellation/interruption behavior, and diagnostic failure publication. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-job-services` |
| 2026-08-01 | ML-500 merged / ML-501 started | Verified PR #452 merge commit `d69692c` and started dry-run planning plus the approved user-facing orchestration surface from merged main. | `feature/ml-user-orchestration` |
| 2026-08-01 | ML-501 local implementation | Added the Python-only immutable workflow dry run with explicit experiment/project ownership, selection/split/schema/model preflight, selectors/counts/overlap/memory/output summaries, and optional dependency availability. Focused, ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-user-orchestration` |
| 2026-08-01 | ML-501 merged / ML-502 started | Verified PR #453 merge commit `8e693d7`, marked ML-501 merged, and started pure analysis result summaries and reproducible plotting from merged main. | `feature/ml-analysis-results` |
| 2026-08-01 | ML-502 local implementation | Added pure fixed-schema adapters for canonical ML results and table-only explicit-path renderers for histories, curves, calibration, metric comparisons, confusion matrices, feature importance, and attribution summaries. Focused, combined analysis/ML, full-unit, docs, Ruff, formatting, and diff checks pass. | `feature/ml-analysis-results` |
| 2026-08-02 | ML-502 merged / ML-503 consumer gate | Verified PR #454 merge commit `1e49998`, created the no-upstream consumer-migration branch from merged main, and rescanned smftools plus the sibling worktrees. The recorded active Nkg2a project consumer and its baselines are unavailable, so ML-503 is blocked rather than substituting a synthetic workflow. | `feature/ml-consumer-migration` |
| 2026-08-02 | ML-503 real consumer migration | Added the validated materialized-dataset bridge, migrated the accepted Nkg2a Cseda01 RF slice into its v2 smftools project, published and validated three immutable fold/model bundles, and recorded prediction, natural-metric, TreeSHAP, source-drift, and intentional legacy-evaluation differences. | `feature/ml-consumer-migration` |
| 2026-08-03 | ML-503 merged / ML-504 ready | Verified PR #455 merge commit `fbb652d` contains the materialized-dataset bridge, marked the dependency-complete legacy-convergence package ready, and reconciled stale roadmap/backlog statuses with the implementation ledger. | `fix/ml-legacy-convergence` |
| 2026-08-03 | ML-504 local implementation / P5 complete | Moved analysis-owned execution behind temporary machine-learning compatibility adapters, added standardized 3.0 deprecations across legacy APIs, preserved pure analysis, documented all replacements, and passed focused, ML, smoke, full-unit, Ruff, formatting, diff, and warning-as-error docs gates. Marked ML-700 ready after merge. | `fix/ml-legacy-convergence` |
| 2026-08-04 | ML-504 merged / ML-700 active | Verified PR #456 merge commit `f74a42f` on `origin/main` contains the legacy-convergence work, marked ML-504 merged rather than pending review, and promoted ML-700 to the active next package. Also refiled the `dev/` audits and ledger under status subdirectories and repointed the reference-plan path into `smftools-work-1/dev/completed/`. | `fix/ml-legacy-convergence` |
| 2026-08-04 | ML-304 status correction | The ML-304 section body carried `[x] DONE` with an evidence block that actually described ML-403's Captum attribution adapters. Confirmed against the package that no encoder/head composition, pretraining task, encoder artifact, or transfer benchmark exists — only reserved schema slots. Reset the section to `[ ] PROPOSED` to match the operational status table. | — |
| 2026-08-04 | ML-700 started | Cut `test/ml-scale-qualification` from `f74a42f` without upstream tracking and drafted the benchmark plan: four sweeps over rows/features/partitions/backend/device, an RSS-sampled measurement protocol, the supported/slow/refused taxonomy, a train-only-transform leakage guard, and committed regression thresholds. Framed the package around calibrating the previously unvalidated `2x`/`3x` estimator constants. | `test/ml-scale-qualification` |
| 2026-08-04 | ML-700 measurement protocol replaced | Implemented the harness and parameterized real-store fixtures, then found the planned in-process RSS protocol invalid: a warm CPython arena under-reports peak allocation by up to 150x (82 KB measured for a 12 MB workload at two warmups). Replaced memory measurement with cold single-shot subprocesses (`benchmarks/_isolated.py`); timing keeps the in-process warmup path. | `test/ml-scale-qualification` |
| 2026-08-04 | ML-700 preliminary headroom finding | First cold-process measurements violate the headroom invariant at 500 positions (peak 1.49x estimate). Row decomposition gives `peak ≈ 1.64 MB + 12.7 KB/row` against an assumed 21.1 KB/row: the linear term is conservative but no fixed term is modelled, so small workloads are the unsafe regime. Single-shot and noisy — recorded as preliminary, not acted on. | `test/ml-scale-qualification` |
| 2026-08-05 | ML-700 preliminary finding retracted | Added N-repeat independent cold processes and an init-separation control. With N=7 the "violated invariant" disappears: worst-case headroom 0.875 cold and 0.595 prewarmed. The 1.49x reading was a tail sample — the same cell gave 9.40 MB, 1.77 MB, and 3.01 MB across single runs. The ~1.6 MB fixed-overhead claim is also withdrawn: the median row fit has a negative intercept at R^2=0.994. Retraction recorded in the plan rather than deleted. | `test/ml-scale-qualification` |
| 2026-08-05 | ML-700 bounded-batch criterion met | Added `sweep_memory_repeated` and `sweep_bounded_batch_memory`. Streaming high-water RSS appeared to blow the batch estimate by 17-48x, but the per-batch trajectory shows growth decelerating from ~515 KB/batch to ~78 KB/batch across equal halves — an allocator plateau, not accumulation. `iter_batches` holds bounded live memory and the acceptance criterion is met. Recorded for ML-701: `max_batch_bytes` bounds one decoded batch, not process RSS, and the publishable claim is that process RSS plateaus independently of split size. | `test/ml-scale-qualification` |
| 2026-08-10 | 2.19.0 released | Cut 2.19.0: bumped `_version.py` from `2.19.0.dev0`, following the single `chore: release` commit pattern from 2.18.0. Auditing the release note against the commit range rather than from memory found two gaps — it recorded the analysis side only as a deprecation, omitting the `analysis.compute.ml_results` and `analysis.plot.ml_results` modules the program added and the declared ML output locations, and it carried the drafting date rather than the release date. PR #465 was deliberately left out, since it ships no user-facing code and 2.18.0 sets the precedent that notes cover user-visible change. Rebuilt the artifacts afterwards: the sdist ships `docs/`, so the first build carried the stale note. Merged as PR #466 (`8d33859`); CI run `31414597966` green across all eight jobs; wheel and sdist verified with `twine check`, correct version metadata, and the shipped tree matching `main`. | `chore/release-2.19.0` |
| 2026-08-10 | CI verified — program complete | Diagnosed why merges #460, #463, and #464 had no workflow run and why #459 and #462 were cancelled. Cause of the cancellations: `github.ref` is `refs/heads/main` for every push, so an unconditional `cancel-in-progress` made each merge cancel the previous merge's validation. Fixed by making cancellation conditional on `pull_request`, and stopped a manual dispatch from killing the weekly Extended CI run. Separately fixed the scheduled Extended CI failure in `test_latent_resource_decision_uses_live_runtime_headroom`, which hardcoded a 1.0 GiB cap and so depended on ambient process RSS: the integration suite reaches ~688 MiB before that file is collected and the ML coverage took it to ~784 MiB, enough for Linux CI but not macOS. Cap now derives from live RSS; verified under 2 GiB of ballast that the old cap reproduces the failure and the new one passes. Merged as PR #465; run `31410967617` on `main` is green across all eight jobs, giving the program its first full validation on Python 3.11, `storage-minimums`, `lint`, and `build`. | `fix/ci-concurrency-main` |
| 2026-08-10 | Ledger reconciliation / program closed | Reconciled statuses across all three surfaces after finding drift in both directions. **ML-105, ML-200, ML-201, and ML-202 section headers had read `[-] IN_PROGRESS` since PRs #439-#442**, months of drift the operational table never reflected — the same defect class as the ML-304 mis-marking found at the start of this program. ML-204 still read `READY` despite merging in #458; ML-700 read `PARTIAL`, ML-701 `IN_PROGRESS`, ML-702 `BLOCKED`. All now agree: 30 `DONE`, 2 `PROPOSED` (ML-205, ML-304), 3 `DEFERRED` (ML-600/601/602), verified by a cross-check asserting every section status matches its operational-table row. Program status set to `COMPLETE — pending CI verification`, because merges #460, #463, and #464 produced no workflow run. | — |
| 2026-08-06 | ML-702 — security review, version matrix, release notes | Performed the security review **adversarially rather than by reading code**, and it holds: every deserialization site in the ML package is safe (`yaml.SafeLoader`, skops with a reviewed allowlist and no pickle fallback, `torch.load(weights_only=True)`), and pickle/joblib policies refuse construction without an explicit unsafe flag. Path containment rejected every probe — run-id traversal (`../../outside`, `a/../../../outside`, `/etc`, `..`, nested, empty), escaping portable references, absolute references, and reverse traversal through `resolve_reference`. **The symlink case is the interesting one**: a link living inside the workspace but pointing outside is rejected, because containment resolves before comparing; a refactor swapping `resolve()` for a non-following equivalent would silently reopen it, which is why that probe is now a committed test. Added `test_ml_security_boundaries.py` (17 tests). Documented the supported-version matrix and the exact-match artifact version policy (a model saved under scikit-learn 1.9.0 will not load under 1.9.1 without `allow_version_mismatch`), plus the deprecation surface and 3.0.0 removal window. Wrote `release-notes/2.19.0.md` covering the program, both breaking schema bumps, and the known limitations. Gates: Ruff clean; docs `-W` green; per-PR 1681 passed, 9 skipped, 7 xfailed; integration 46 passed, 2 skipped. | `feature/ml-program-acceptance` |
| 2026-08-06 | ML-701 PR4 — API reference and tutorials / package COMPLETE | **Corrected the API blocker estimate: five modules fail autodoc, not ten.** The earlier figure came from grepping for mock-subclassing, which over-counted — some mocked bases (`nn.Module`, `TransformerMixin`) document fine and only certain ones (lightning, `torch.utils.data.Dataset`) break. Measured precisely with `sphinx.ext.autodoc.mock` per module: 61 of 66 import clean. All five failures are `anndata_data_module`, `sliding_window_inference`, `lightning_base`, `train_lightning_model`, `train_sklearn_model` — every one deprecated or gated behind unbuilt Lightning, which makes the omission defensible rather than arbitrary. Widened `__all__` from 8 to 15 to include `artifacts`, `contracts`, `manifests`, `plan`, `selection`, `splitting`, `workspace`; the quick start already told readers to import them. Added `api/machine_learning.md` (61 modules) and two narrative tutorials. **Adding the API page immediately caught a malformed RST table in `streaming_transforms.py`** written during ML-204 — the `mode`→`most_frequent` rename pushed two rows past the column margin, invisible until the module first went through autodoc. `test_ml_api_surface.py` pins the exclusion set so a new undocumented module fails rather than vanishing. | `feature/ml-documentation-api` |
| 2026-08-06 | ML-701 PR3 — guidance pages | Added `splits_and_masks.md`, `interpretability.md`, and `artifacts_and_trust.md`; seven ML pages now build `-W` green. **Found while writing: the mask vocabulary declares seven kinds but the partition reader produces four.** `MLPartitionBatch.mask_arrays` filters on `if mask.kind in values`, so a declared `attention`, `corruption`, or `loss` mask is silently omitted rather than rejected — no error, no mask. Documented as a warning telling users to declare only the produced four, and pinned by `test_ml_documented_mask_contract.py` so the warning cannot outlive the behaviour. Whether silent omission should instead raise is left as an open question for ML-702. The splits page also states plainly what is *not* enforced — notably that group disjointness is only as good as the `group_by` field, since the package cannot know whether the named axis is the one you need to generalise across. | `feature/ml-documentation-guidance` |
| 2026-08-06 | ML-701 PR2 — quick start and plan reference | Added `docs/source/ml/quickstart.md` and `plan_reference.md`, rebased onto the merged streaming-wiring fix so the quick start documents a facade that actually streams. **Every plan example was executed against `parse_ml_plan` before being written down, and three would otherwise have been wrong**: the scope key is `set` not `set_name`; sklearn models declare `family` while Torch models declare `recipe` (each rejects the other); and an `explain` job requires at least one method in `explain`. Added `tests/unit/machine_learning/test_ml_plan_documented_rules.py` (7 tests) pinning the documented rules to the parser rather than the prose, so a schema change fails the build and names the page to update. `sphinx-build -W` green across five ML pages. | `feature/ml-documentation` |
| 2026-08-06 | ML-204 follow-up — streaming reaches the orchestration facade | Writing ML-701's quick starts surfaced that ML-204 shipped the streaming engines but never wired them in: `train_partition_model` called only the materializing fits, the streaming ones were not exported from `training/__init__.py`, and their only callers in the package were the benchmark modules. A user on the approved orchestration surface still hit the ~85,000-row ceiling ML-204 existed to remove. Exported both streaming fits and added a `streaming` option to `SklearnTrainOptions` and `TorchTrainOptions`. **Defaults are deliberately asymmetric, grounded in measurement**: sklearn defaults to streaming whenever the family declares `incremental_fit`, because a streamed sklearn fit is numerically identical to the materialized one; Torch defaults to materializing, because a streamed Torch fit shuffles within a buffer and produces different weights, so switching silently would hand the user a different model. Materialization refusals at the training boundary are re-raised naming the streaming remedy, with the original preserved as `__cause__`. Added 8 dispatch tests including a parity test asserting the sklearn default only holds while streamed and materialized fits agree. Gates: Ruff clean; per-PR 1648 passed; integration 46 passed. | `fix/ml-streaming-orchestration` |
| 2026-08-06 | ML-701 PR1 — architecture and performance pages | Cut `feature/ml-documentation` from merged `main` (`4fe69a1`, PR #459). Established the docs baseline builds green before touching anything. Added `docs/source/ml/` with an index, an architecture/ownership guide, and a performance/limits page, registered in the root toctree; `sphinx-build -W` passes and build artifacts were cleaned per `docs/source/AGENTS.md`. Grounded the ownership guide in the **actual** package layout rather than the ledger's conceptual tree, which differs (no `schemas/` or `tasks/` directories; contracts live in top-level modules). The performance page states the refusal formula and worked examples rather than duplicating the ceiling table, so it cannot drift from `tests/acceptance/ml_scale_thresholds.json` and the docs build does not depend on the test tree. Measured the API-reference blocker and recorded the curated-subset decision. | `feature/ml-documentation` |
| 2026-08-06 | ML-700 Sweep C — worker scaling / all deliverables covered | Ran the worker-scaling sweep for the first time. **Sharding is free**: per-shard throughput is flat at 549-567 rows/second across `num_workers` in {1,2,4,8}, shard time scales as exactly 1/N (efficiency 100-103%), shards partition the split exactly, and no cell was flagged unstable. Recorded the honest limit of the measurement — shards were read one at a time in a single process, so this establishes that the sharding arithmetic costs nothing, **not** that N workers deliver an N-times wall-clock speedup; real parallel workers contend for disk and memory bandwidth, which is unmeasured. The per-PR guard asserts that distinction survives in the published text, so "sharding is free" cannot decay into "N workers are N times faster". With this, every ML-700 deliverable has evidence: benchmark matrix, peak memory, partition throughput and worker scaling, refusal thresholds, explanation chunk guidance, and committed regression thresholds. | `test/ml-scale-qualification-limits` |
| 2026-08-06 | ML-700 Sweep D — explanation chunking | Added `measure_explanation_chunking`. **Wall time has an interior optimum**: at 400 rows x 400 positions, `example_batch_size=8` was fastest, with 1 at 1.99x and 512 at 2.76x — larger chunks are not faster, contradicting the natural assumption. **`example_batch_size` is a scientific parameter, not only a performance knob**: attributions repeat bitwise at a fixed chunk size but differ by ~1.1e-3 relative across chunk sizes (max abs 2.27e-4 against a max value of 0.204), consistent with batch-size-dependent convolution kernel selection accumulating through the residual blocks. Investigated whether this was a fourth instance of a knob leaking into content identity and it is **not**: `example_batch_size` is already in `request.parameters` and therefore `request_id`, which `result_id` incorporates, so differing chunk sizes are correctly recorded as different results. Also ruled out dropout — `explain_torch_model` sets eval mode at `neural.py:512`. Guidance published: pick a chunk size and hold it across runs intended for comparison. Memory across chunk sizes was **not** measured reliably (sequential in-process runs warm the arena; one cell reported 0.1 MB against another's 46.8 MB) and is recorded as an open caveat. | `test/ml-scale-qualification-limits` |
| 2026-08-06 | ML-700 limits shipped as data / ML-205 catalogued | Owner decisions: ML-700 ships machine-readable limits and ML-701 writes the prose (matching the ledger's file-ownership table, which assigns docs to ML-701 — the `docs/source/ml_performance.md` in the benchmark plan was never ML-700's to write); the per-PR guard runs under the existing `unit` marker with the measured subset staying `integration`; ML-205 is `PROPOSED` with a recorded start gate. Added `tests/acceptance/ml_scale_thresholds.json` (limits, workload taxonomy, regression thresholds, and an `unmodelled` register), a fixture-free per-PR guard recomputing every published ceiling from `_bytes_per_row`, and a measured `integration` subset. Placed the thresholds file under `tests/acceptance/` rather than `src/` per the `partitioned_pipeline_criteria.json` precedent — it is a test expectation, not shipped runtime data. Note `-m integration` does not run on pull requests (only weekly via `extended-ci.yml`), which is why the arithmetic guard is `unit`-marked. | `test/ml-scale-qualification-limits` |
| 2026-08-05 | ML-700 pt2 started / Torch sizing rule not derivable | Cut `test/ml-scale-qualification-limits` from merged `main` (`23a5dfc`, PR #458) and restored the stashed trajectory tooling. Attempted a Torch memory scaling law over batch size x sequence length: ratios span 4,409x-20,833x with the (64, 400) cell breaking the trend, and the grid is confounded because batch count was not held constant (batch-16 cells sample ~450 batches, batch-64 ~114). **Recorded as not-a-law rather than published as one.** What holds: Torch plateau RSS is 3-4 orders above the data batch estimate with a non-constant ratio, so no multiplier converts a data budget into a Torch prediction, and the data budget cannot size a Torch job. Forces a scoping decision on ML-700's limits deliverable, which the plan assumed derives from the data estimator: publish limits for reads/materialization/sklearn and document Torch as empirically sized, or build an activation-memory estimator as its own package. Recommended the former. | `test/ml-scale-qualification-limits` |
| 2026-08-05 | ML-204 streaming-fit memory verified / activation gap found | Added a `streaming_fit` trajectory mode sampling RSS as each batch leaves the reader. **Both backends are bounded**, closing ML-204's last acceptance criterion: sklearn decays 52x first-to-final quartile (plateau 132 MB); Torch adds +1,696.78 MB in its first decile then +0.00 and +0.25 MB in its last two (plateau 2,172 MB over 3 epochs). **New finding for ML-700's limits: Torch process RSS is 12,557x the data batch estimate**, because the working set is dominated by model activations, which no ML data-plane budget models — `partition_dataset` estimates data bytes only. So `max_batch_bytes` does not control Torch training memory, and a supported/slow/refused taxonomy for Torch cannot be derived from the data estimator. sklearn's plateau (307x) sits in the same order as reads (281x), so this is a neural-model property, not a streaming one. Also recalibrated the trajectory verdict from batch-estimate-relative to plateau-relative, then recorded that it remains threshold-sensitive: under the new criterion the read control flips to `accumulating` at a 6.46% tail on a run previously measuring 1.4%. Third instance of a fixed RSS threshold flipping between runs — the verdict is a reading aid, not a CI gate. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 shuffle buffer moved into persisted provenance | `shuffle_buffer_batches` was a call argument, so two streamed models with identical recorded `training_config` could have different weights. Same defect class as the `transform_id` fix, inverted: there a non-scientific knob leaked *into* identity, here a scientific knob was *missing from* provenance. Moved into `TorchTrainingConfig` with `TORCH_TRAINING_CONFIG_VERSION` 1 → 2, validated as a positive integer, and carried through `to_dict`/`from_dict` into the persisted artifact. Added a test asserting both halves — that the buffer genuinely changes the fitted model and that the value is recorded — so the field cannot decay into unverified documentation. Migration for ML-702: Torch training configs persisted under version 1 lack the field and will not load. Gates: Ruff clean; unit+smoke+integration 1671 passed, 10 skipped, 7 xfailed. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 Torch streaming fit | Added `fit_torch_partition_model_streaming` and `_StreamingLoader`, a re-iterable loader that decodes partition batches and substitutes for `DataLoader` inside `_evaluate`. Removes all three materializations (train/validation/test). **Two deliberate semantic differences, documented rather than hidden:** shuffling is buffered over `shuffle_buffer_batches` decoded batches rather than global, so fitted weights differ from the materialized path at identical seeds; and each epoch re-reads the store, trading wall time for the ability to run at all. Weight parity is therefore asserted nowhere — instead the tests assert the locked-test contract (a read-recording dataset proves no `test` read precedes the last train/validation read, and no `materialize` call occurs), balance `resolution_id` and `transform_id` parity, seed reproducibility, and training where the materialized path raises `MLMemoryBudgetError`. `weighted_sampler` is refused: it samples with replacement across the whole split, needing the random access streaming avoids; the message names `class_weight` as the equivalent reweighting. Gates: Ruff clean; unit+smoke+integration 1670 passed, 10 skipped, 7 xfailed. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 sklearn streaming fit | Added `fit_sklearn_partition_model_streaming`: balance from plan metadata, transform from streamed batches, then one pass feeding `partial_fit`. Selection multiplicity is applied per batch via `np.bincount`, so upsample duplication and downsample dropping work without materializing. Documented the one deliberate divergence: streaming feeds rows in canonical batch order rather than the balance's permuted order, which is equivalent only for order-independent incremental updates — asserted in tests rather than assumed. Only `bernoulli_nb` declares `incremental_fit`; `logistic_regression` and `random_forest` are refused with a message naming the streaming-capable families and `max_materialization_bytes`. Verified identical `feature_log_prob_`, `class_log_prior_`, predictions, `n_training_observations`, balance `resolution_id`, and `transform_id` against the materialized fit for natural/class_weight/downsample/upsample. **A streaming fit succeeds where `fit_sklearn_partition_model(incremental=True)` raises `MLMemoryBudgetError`** — the acceptance criterion, now covered. Also added `FixtureSpec.imbalanced`, because the balanced default made downsample and upsample no-ops and the parity claim vacuous. Gates: Ruff clean; unit+smoke+integration 1665 passed, 10 skipped, 7 xfailed. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 metadata-only balance resolution | Confirmed by reading `_resolve` that balancing consumes only `labels`, `split`, and `molecule_uids` — never feature values. Refactored the core to `_resolve_from_labels(labels, molecule_uids, role, ...)` with `_resolve` retained as a thin wrapper, then added `resolve_role_balance_from_plan`, which resolves a cohort from `PartitionReadEntry` metadata with **zero data reads**. Verified identical `resolution_id`, `selected_indices`, and `selected_molecule_digest` against the materialized path for all five methods (natural, class_weight, weighted_sampler, downsample, upsample), including seeded resampling order, on both stubs and real Zarr stores. Added 8 unit tests and 12 integration tests; the integration file asserts the zero-pass claim by passing a source that raises if read. Gates: Ruff clean; ML unit+integration 313 passed, 7 xfailed; unit+smoke 1637 passed, 9 skipped, 7 xfailed. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 transform_id defect fixed | Implemented digest quantization: `ML_FEATURE_TRANSFORM_VERSION` 1 → 2, `transform_id` hashes `_identity_digest_payload()` rendering the three fitted statistics at 12 significant digits with normalized negative zero. Digest input only — stored arrays and `to_dict()` keep full precision, so artifact round trips are byte-unchanged. Streamed and materialized fits now produce identical IDs across all five streamable specs, and IDs are stable across batch sizes. Added `tests/unit/machine_learning/test_ml_streaming_transforms.py` (23 tests) including a named regression guard for the batch-size defect. Gates: Ruff check/format clean; unit marker suite 1523 passed, 9 skipped, 7 xfailed; smoke+integration 116 passed, 2 skipped. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 started / transform_id defect found | Cut `feature/ml-streaming-training` from merged `main` (`9b5bf2f`, PR #457). Implemented `data/streaming_transforms.py`: a pass planner and bounded accumulators that fit a train-only transform from batches. Established that the **default spec costs zero data passes** (`imputation="constant"`, `scaling="none"` are declared, not learned) and that applying a transform already streams, so only fitting needed work. `median` is refused with guidance rather than silently approximated. Validation against the materialized fit matched bitwise for fill values but exposed a blocking defect in the merged ML-203 contract: `transform_id` hashes unrounded float64 moments, so it varies with summation order — **four distinct IDs across `batch_size` ∈ {16,32,64,128}** on identical data. Owner decision required before ML-204 can claim provenance parity. | `feature/ml-streaming-training` |
| 2026-08-05 | ML-204 sequenced ahead of P7 | Owner decision: ML-204 runs before ML-700's published limits, ML-701, and ML-702. Marked ML-204 `READY`, ML-700 `PARTIAL — paused for ML-204` (its harness is complete and is an ML-204 prerequisite; its remaining deliverable is the limits, which ML-204 changes), and ML-701/ML-702 `BLOCKED` with the reason recorded. Reopened P2 in the phase table, added PRs 21-24 to the suggested sequence, and rewrote "Current next action". | — |
| 2026-08-05 | ML-204 proposed | Drafted the streaming-training work package from the ML-700 Sweep B finding and registered it in the status table. Scoped the real difficulty: the model fit is the easy half, while train-only transform fitting needs per-column statistics that are streamable for constant/mean/mode/standardize but **not** for `median`, which needs an explicit refuse-or-approximate decision. Recorded that balancing needs no data pass at all, since `PartitionReadEntry.class_id` already carries labels in the read plan. Package touches ML-203, ML-301, and ML-303 contracts, so it is not small. | — |
| 2026-08-05 | ML-700 Sweep B — no training path streams | Wiring Sweep B surfaced an architectural gap. `torch_backend` materializes train/validation/test (lines 403, 404, 514) and never calls `iter_batches`; `sklearn_backend` materializes train at line 145 *before* the incremental branch, so `partial_fit` chunks an already-materialized, already-transformed array. Verified empirically: at one byte below the train estimate, `bernoulli_nb` incremental=True, incremental=False, `random_forest`, and `residual_dilated_cnn` all refuse at preflight. With the default 2 GiB budget the ceiling is ~85,011 total rows at 1,000 positions/1 channel — a 15x shortfall against a 1.3M-read experiment, rising to 567x at 20,000 positions. **No training path can consume a production-scale experiment**, while the bounded streaming reader that could is unused by training code. Supersedes the plan's predicted "sklearn has a ceiling Torch does not" — both do. Recommended as its own work package, not a benchmark line. | `test/ml-scale-qualification` |
| 2026-08-05 | ML-700 plateau demonstrated | Added `measure_batch_trajectory` (isolated `trajectory` mode returning per-batch RSS). A 300-batch run gives monotonic quartile decay 739,937 → 108,046 → 27,676 → 17,270 bytes/batch, ending at 3.8% of the batch estimate — accumulation excluded, since retaining a batch per iteration cannot decay. The earlier 38-batch run was non-monotonic and returned the opposite verdict, so run length is part of the method. Caveat recorded: Q4 creep extrapolates to ~350 MB at production scale, so the publishable claim is "sublinear and decaying", not "constant memory". | `test/ml-scale-qualification` |
| 2026-08-05 | ML-700 instrument limitation documented | RSS has now produced a false finding in both directions: it under-reports for repeated in-process work (82 KB for a 12 MB workload) and over-reports for streaming work (48x the true per-batch bound). Neither is a package defect. Future ML-700 work and `thresholds.json` must state which direction applies to a workload shape before quoting a number. | `test/ml-scale-qualification` |
| 2026-08-05 | ML-700 estimator verdict | No change to `_bytes_per_row` warranted. Worst-case slope 12,950 B/row against 21,102 assumed (1.6x margin); median slope 9,507 (2.2x). The constants are over-conservative rather than unsafe. One sample touched headroom 1.030 in a cell whose max ran 3.65x its median; tail attribution is unresolved and needs tracemalloc-class instrumentation, not more RSS samples. Recommendation recorded: leave the constants alone — over-conservatism costs a config override, under-conservatism costs a mid-run OOM. | `test/ml-scale-qualification` |

## Current next action

**Begin ML-204.** Create `feature/ml-streaming-training` from current `main` without upstream
tracking.

ML-700 is paused at a deliberate point: its harness is complete (and ML-204's acceptance criteria
depend on it), while its remaining deliverable — the published supported/slow/refused limits — is
exactly what ML-204 will change. Finish ML-700's limits after streaming lands, then ML-701, then
ML-702.

Sequence: `ML-204` → `ML-700` (limits) → `ML-701` → `ML-702`.

Start ML-204 with the two design decisions that gate implementation, because both change the
public contract:

1. `imputation="median"` under streaming — refuse with a message naming the streamable
   alternatives, or adopt a named approximation with recorded error bounds. Do not silently
   substitute a different statistic.
2. Whether non-incremental families (`random_forest`, `logistic_regression`) keep the
   materialization ceiling permanently, or gain a documented sampling path.

Exploit the cheap win first: balancing needs no data pass, because `PartitionReadEntry.class_id`
already carries labels in the read plan.

Preserve the external legacy Nkg2a project as historical accepted evidence. Keep Hydra, Lightning,
hosted tracker writers, and ML-304 outside the work package unless their recorded trigger is met.
