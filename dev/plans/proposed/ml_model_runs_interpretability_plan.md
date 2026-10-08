# Model runs, reuse, interpretability and comparison (`MLR`)

**Status:** proposed. Nothing implemented. One PR per item, in order.

## Why

A trained model should be a self-describing record: what task and data it was
trained and evaluated on, its fitted state, its metrics, and its
interpretations -- reusable later on new data, and comparable with other
models on the same evaluation. Today a project trains through
`bind_ml_job` -> `run_bound_train_job`, scores each fold and keeps only
metrics (`nkg2a_final` adds a prediction table by hand); every fitted model is
discarded, nothing records which molecules trained it, interpretability is
never run, and model comparisons are tables written by hand.

## What exists (the ML program, `ML-000`-`ML-702`, completed)

- Plans, selection, splits: `plan.py`, `selection.py`, `splitting.py`
  (`leave_one_group_out`, `explicit_groups`, `stratified_group`; roles
  train / validation / test), label tables (molecule-level keys since
  `feature/ml-molecule-label-keys`).
- Model registry (`models/registry.py`): `bernoulli_nb`,
  `logistic_regression`, `random_forest` (sklearn), `residual_dilated_cnn`
  (torch); versioned recipes, validated configs, capabilities. Training:
  `training/sklearn_backend.py` (streaming for `partial_fit` families),
  `training/torch_backend.py` (epochs, early stopping, per-epoch history).
- Artifacts: `workspace.py` (project `ml/{datasets,runs,models,index}`),
  `artifacts/` (`RunManifest`, `ModelManifest` with lineage,
  `PredictionManifest`, `ExplanationManifest`, atomic `publish_bundle`),
  `models/sklearn_artifacts.py` (skops, no pickle) and
  `models/torch_artifacts.py` (state dicts) with `load_published_*`.
- Job service (`orchestration/service.py`): immutable train / apply /
  evaluate / explain / plot lifecycles around a caller's operation.
- Evaluation (`evaluation/`): predictions, metrics, ROC/PR curves, folds,
  training history.
- Interpretability (`interpretability/`): `NaiveBayesLogOdds`,
  `LinearCoefficients`, `PermutationImportance`, `TreeSHAP`; `Saliency`,
  `InputXGradient`, `IntegratedGradients`, `GradientSHAP`, `LayerGradCam`
  (Captum); training-background sampling; explanation artifact layout.
  Plot: `analysis/plot/ml_results.plot_attribution_summary`.
- Older neural classes (`models/mlp.py`, `cnn.py`, `rnn.py`,
  `transformer.py` incl. a domain-adversarial transformer,
  `lightning_base.py`) predate the registry.

**Gaps.** No single call trains *and* publishes a plan's job (the service
needs an operation the caller assembles); no "apply a saved run to another
dataset"; per-molecule attributions are computed but not stored as a
molecules x positions matrix with molecule identities, and there is no figure
of them beside the input; no fixed-prevalence AUPRC; no report comparing runs;
each model re-reads the data; only one neural family is registered; no XGBoost
or SVM.

**Precedent** (`Nkg2a_DAFseq_merged/claude_scripts/ml/`): out-of-fold random
forest SHAP per held-out molecule stitched across folds into one aligned
figure, fold models saved and reloaded (`f1_activity_rf_explanations.py`);
per-site layer integrated gradients from a transformer, chunked for memory
(`transformer_apply.py`); fold-to-fold consistency of attribution tracks
(`f1_activity_cnn_consistency.py`). This plan makes that pattern standard.

## Design

### A model run is the unit

One directory per train job under the workspace's `runs/` (its models listed
in the manifest's `model_keys`), published atomically and never edited. As
built in `MLR-01` (`orchestration/runs.py`):

```
runs/<run_id>/
  run_manifest.json   plan hash, job, dataset snapshot id (molecules and their
                      class ids), split id (digest of the fold split ids),
                      environment (smftools version, commit, dirty tree,
                      packages), seeds, artifact checksums, state
  resolved_plan.json, resolved_config.json   (options, prevalence, fold split ids)
  tags.json           caller labels (e.g. a project's task id)
  data/membership.parquet   per fold: every molecule's role and class
  data/splits.json    per fold: split id, held-out group, role / class counts
  models.json         per model and fold: the published model id
  predictions/test.parquet  held-out truth, prediction, class probabilities
  metrics.parquet     every metric per model and fold, plus normalised and
                      fixed-prevalence average precision
  curves.parquet      ROC / PR / calibration points
  history.parquet     training events (torch: per-epoch losses)
  summary.json        per model: mean / SD / n over folds
models/<model_id>/    each fold model (ModelManifest + skops / state dict),
                      `originating_run_id` pointing back at the run
```

Since `MLR-02`: fold `final` models, and apply runs (their own runs, linked
by `source_model_ids`). Planned: `explanations/<id>/` and figures (`MLR-03`, `MLR-04`). Not yet
recorded: a content hash of a coordinate map (the plan hash covers its path).

The workspace `index/` lists runs with their tags and headline metrics, so a
project can find "every model on task X" without walking directories.

### Applying a saved run

`apply` takes one published model (a run's final model, or a fold model)
and any dataset selection whose input schema matches (same channels,
positions -- through a coordinate map if needed -- and classes): predictions,
and metrics when labels exist, published as an apply run whose manifest names
the source model (runs are immutable, so it is not added inside the train
run).

### Interpretability belongs to the run

`explanations/<id>/` holds one method on one evaluation set: the method and
parameters, the background (for SHAP-style methods), the evaluation set's
molecule UIDs, and

- **position importance**: per position, the global score per fold model
  (mean |attribution|, NB log-odds, permutation drop) and the fold-to-fold
  consistency (rank correlation);
- **per-molecule attributions**: a molecules x positions matrix (float32,
  chunked, with UIDs and frame positions), the positive-class contribution of
  each position to each molecule's score, from the fold model that held the
  molecule out (out-of-fold) or the chosen model for applied data;
- **detector catalogue** (convolutional models): what each final-layer
  detector (channel) responds to -- its top-activating input windows (at the
  detector's span) across the evaluation set, their mean pattern per input
  channel, where on the locus they occur, how often in each class, and
  detectors grouped by pattern similarity. Attributions say where a
  molecule's score came from; the catalogue says what patterns the model
  looks for (the CNN analogue of motif discovery).

### The attribution clustermap

For an explanation record: the input layer(s) (site calls, HMM accessible /
footprint, read from the same dataset) and the attribution matrix side by
side, one row order for all panels, TSS-relative columns, with strips for the
true label, predicted score and fold (and optionally Leiden / NDR state when a
latent embedding is supplied). Row order: by predicted score, by label then
hierarchical, or by a supplied binning. Built on
`plotting.latent_plotting.plot_latent_ordered_clustermap` (colour-stable
strips, extra strips), a diverging colour scale for attributions.

### Comparing runs

A comparison selects runs (by tags) evaluated on the same folds and reports,
per fold and overall: each metric, paired differences between runs, and
uncertainty (bootstrap over molecules within folds; across folds), with a
figure (models x tasks, per-fold points). It reads only run records.

### Speed: one read per task

Fold feature matrices are cached per (dataset snapshot, split, transform) in
the workspace `datasets/`, so a second model on the same task does not re-read
the stores.

## Work items

| item | status | scope |
|---|---|---|
| `MLR-01` train-and-publish | done (PR #700) | one call: bind a plan job, train each model per fold, publish run / data / models / evaluation records and index; fixed-prevalence AUPRC (reweighted, subsampled) in smftools metrics |
| `MLR-02` final models and apply | implemented (`feature/ml-final-models-apply`) | optional all-groups final model; apply a run to another dataset with records |
| `MLR-03` explanation records | proposed | position importance, per-molecule attribution matrices, and the CNN detector catalogue per run and evaluation set, out-of-fold; fold consistency |
| `MLR-04` attribution clustermap | proposed | input layers beside attributions, shared row order, label / score / fold strips; detector catalogue figures |
| `MLR-05` run comparison | proposed | select runs by tags; paired per-fold metrics, bootstrap intervals, figures |
| `MLR-06` fold-matrix cache | proposed | read each task's data once for every model |
| `MLR-07` validation role | proposed | a stratified validation fraction of each fold's training molecules (default) or held-out training experiments, for early stopping and tuning; the test experiment stays whole; final models too |
| `MLR-08` detector-scale CNNs | proposed | position-agnostic residual dilated CNNs whose pattern detectors have a stated, enforced maximum span (receptive field): sub-nucleosome, 2-3, 4-6 nucleosomes, full locus; effective span measured per run |
| `MLR-09` further neural families | proposed | MLP / transformer ported to registry configs; project-registered families |
| `MLR-10` qualification | proposed | `nkg2a_final` region / model grid through `MLR-01`-`MLR-05`; parity with its current metrics |

### `MLR-01` -- train-and-publish

`orchestration.train_and_publish(bound, *, workspace | project_dir, tags,
sklearn_options, torch_options, registry, prevalence=0.10, ...)` -> one run
per bound train job, composing `iter_bound_train_job` (fold runs one at a
time, each model published and released) and the service's train lifecycle
(failures publish a failed run and raise); the final model moves to
`MLR-02`; fold models through
`publish_sklearn_model` / `publish_torch_model`; per-fold molecule lists;
predictions, metrics, curves, history. `evaluation.metrics` gains
`average_precision_at_prevalence` (reweighted, and subsampled with draws and a
seed). The index records tags and headline metrics.

Tests: a fixture project trains NB and RF; every record exists and validates;
reloading a fold model reproduces its predictions; the molecule lists equal
the split; the fixed-prevalence metric equals a direct computation; a second
call with the same plan publishes a new run (immutable) and the index lists
both.

Evidence: `tests/integration/machine_learning/test_train_and_publish.py` (NB
and RF over three held-out experiments; a residual CNN with an explicit
train / validation / test split -- torch training needs a validation role,
which leave-one-group-out lacks until `MLR-07`; a failed run) and
`tests/unit/machine_learning/test_ml_prevalence_metric.py`.

### `MLR-02` -- final models and apply

`train_and_publish(..., final_model=True)` fits each model on every selected
row (fold `"final"`: model id, membership, split record; no held-out
evaluation). `apply_and_publish(plan, job, model_id=...)` applies one
published model to an apply job's dataset and publishes an apply run (not a
record inside the immutable train run): manifest `source_model_ids`,
`model.json` (originating train run), `data/molecules.parquet`,
`predictions/applied.parquet`, and for labeled data metrics, curves and
summary. Channels, positions and classes are checked first. Final torch
models are refused until `MLR-07`. Applying a fold ensemble is left for
later.

Tests (`test_final_models_and_apply.py`): a final model is fit on every row
and reloads; applying it to labeled data publishes predictions, metrics and
lineage, indexed; unlabeled data gets predictions only; a re-applied fold
model reproduces its held-out predictions; `model:<id>` jobs, a missing
model id, a position mismatch (named) and final torch models are handled.

### `MLR-03` -- explanation records

`explain_run(run, method, *, evaluation="held_out" | dataset selection,
parameters, background=...)`; dispatches to `interpretability` by model
capability; stores importance and the attribution matrix (chunked, UIDs,
positions); out-of-fold by default. For convolutional runs, method
`DetectorCatalogue` (parameters: top windows per detector, similarity
threshold for grouping) reads final-layer activations before pooling.

Tests: NB log-odds attributions equal the closed form; TreeSHAP rows sum to
the model output minus the base value; each held-out molecule is explained by
the fold model that held it out; fold consistency on a planted signal; a CNN
trained on a planted pattern has a detector whose top windows recover it, at
the planted locations, enriched in the planted class.

### `MLR-04` -- attribution clustermap

Detector catalogue figures: per detector (or group), the mean top-window
pattern per input channel, its locus position histogram and class
enrichment.

Tests: panels share rows; strips match their rows; colour scale symmetric
about 0; row orders as requested; figure written per explanation; catalogue
figures match the stored windows.

### `MLR-05` -- run comparison

Tests: runs on different folds are refused (or compared on the shared folds
with a warning); paired differences equal direct computation; bootstrap is
seeded.

### `MLR-06` -- fold-matrix cache

Tests: a second model reuses the cache (no store reads); the cache key changes
with the snapshot, split or transform; results equal an uncached run.

### `MLR-07` -- validation role

Test is always a whole held-out experiment (the estimate of performance on a
new batch). Validation comes from the training experiments, for early
stopping and tuning only:

- **Molecule-level (default).** `leave_one_group_out` gains
  `validation_fraction` (e.g. 0.15): in each fold, that fraction of the
  training molecules becomes validation, stratified by group (experiment) x
  class so every training experiment and class is represented in proportion;
  seeded (`seed`), recorded per molecule in the run's membership.
- **Group-level (option).** `validation: groups` holds out whole training
  experiments instead (mirrors the test condition, at the cost of training
  batches -- about a quarter with four training experiments); for comparison
  on the CNN ladder (`MLR-08`).
- **Final models** (`MLR-02`): the same fraction of every selected row, so
  final torch models become possible (refused until now).
- **Isolation rule.** Split manifests now refuse any group in two roles. With
  a molecule-level fraction, train and validation share experiments; the rule
  becomes: test is isolated from train and validation; train and validation
  may share groups only when the split declares a molecule-level validation
  fraction (recorded in the manifest).
- **Caveat recorded with the run.** Molecule-level validation shares batch
  effects with training, so validation loss is optimistic relative to a new
  batch and early stopping may stop somewhat late; reported performance stays
  honest (test is a whole experiment).

Tests: test molecules never appear in train or validation, and test is one
whole experiment per fold; the validation fraction is met within rounding per
experiment x class; seeded and reproducible; the relaxed isolation rule
accepts the declared shared groups and still refuses test leakage or an
undeclared shared group; group-level validation holds out whole training
experiments; early stopping reads validation only; a torch final model trains
with the fraction; leave-one-group-out folds are unchanged without the option.

### `MLR-08` -- detector-scale CNNs

Question: how large must a single pattern detector be for activity to be
readable -- sub-nucleosome, 2-3 nucleosomes, 4-6, or the whole locus -- with
models otherwise alike?

The existing `residual_dilated_cnn` is the base: convolutions, then global
average / max / content-attention pooling and a small head -- no positional
encoding, no flatten over positions, so it learns what a feature looks like,
not where it is. Such a model is a bag of local detectors:

- each final-layer position is a pattern detector that sees a window of the
  input no wider than the **receptive field (RF)** -- the detector's maximum
  span;
- pooling summarises each detector over the whole molecule (average: how much
  of the pattern; max: whether it occurs anywhere; attention: a
  content-weighted sum) and discards where the detections were;
- the head combines those summaries.

So every model uses the whole molecule; the RF bounds the largest single
pattern a detector can recognise, and so the largest distance across which the
model can relate two features. The ladder varies that span.

Inputs are on a bp grid (site calls with an observed mask), so spans are in
bp: `RF = 1 + (stem_kernel - 1) + sum over blocks of 2 (kernel - 1) dilation`
(the theoretical maximum).

- **Effective span.** Influence concentrates at a detector's centre and falls
  off towards its edges, so the effective span is often well below the
  theoretical one. Each trained run records both: the theoretical RF, and the
  effective span measured from gradients of final-layer detector outputs with
  respect to the inputs on held-out molecules (the centred width holding 50 %
  and 90 % of the |gradient| mass, averaged over detectors and positions). A
  rung whose effective span falls far below its label is reported as such.
- **Sparse sites.** A window of fixed width holds however many C (or GpC)
  sites fall in it, which varies along the locus; spans are reported with the
  site density they cover (sites per window, by locus position).
- `ResidualCNNConfig` gains `receptive_field` (computed, recorded in the run
  manifest) and an optional `max_receptive_field` (refused when exceeded).
- Squeeze-excite pools over the whole molecule inside every block, so a model
  with SE sees the full locus whatever its dilations: bounded models require
  `use_se: false` (validated against `max_receptive_field`).
- Recipes, alike in depth / width as far as the ladder allows (dilations grow
  gradually -- large jumps leave gaps, "gridding"; a downsampling stage is the
  alternative for the full locus):

  | recipe | target | example dilations (kernel 5) | RF |
  |---|---|---|---|
  | `rcnn_subnucleosome` | < ~150 bp | 1, 1, 2, 2, 3, 4 | 113 bp |
  | `rcnn_2_3_nucleosomes` | ~400-600 bp | 1, 2, 4, 8, 16, 32 | 513 bp |
  | `rcnn_4_6_nucleosomes` | ~800-1,200 bp | 1, 2, 4, 8, 16, 32, 64 | 1,025 bp |
  | `rcnn_full_locus` | >= 4.7 kb | 1, 2, 4, 8, 16, 32, 64, 128, 256, 128 | 5,121 bp |

- At full-locus RF a CNN can infer absolute position from padding at the
  molecule's edges: position-agnostic by design, not in effect -- the
  comparison point, recorded as such.
- **Sparse site inputs** (C / GpC calls at their sites) beside **dense HMM
  layers** (accessible / footprint / lengths at every bp, `site_context: all`)
  share the bp grid, each with its own observed mask. Two fixes to the
  residual CNN first:
  1. *Missing vs unmodified.* Unobserved values are zero-filled and the mask is
     not an input, so "no site" and "site, unmodified" both read 0. Give the
     model the mask: an observed-indicator channel per sparse channel (or a
     signed encoding, +1 modified / -1 unmodified / 0 no site).
  2. *Propagation between sites.* Features are zeroed after the stem and every
     block wherever no channel is observed; with site channels alone that is
     most positions, so dilated taps between sites read zeros and context
     flows only through sites (worse for GpC). Zero by the read's span (first
     to last observed position) instead.
  Length layers are rescaled (log length, or length classes) before sitting
  beside 0 / 1 channels.
- **HMM inputs do not respect the receptive field.** Each position's HMM call
  comes from decoding the whole read, and a length layer stamps a feature's
  full extent at each of its positions, so a "sub-nucleosome" CNN on HMM
  layers can use information from far outside its window. The ladder bounds
  information only with raw site inputs. Three input arms per recipe: sites
  only (the context experiment), HMM layers only, sites + HMM -- reported
  separately. HMM inputs also inherit the HMM's choices (enzyme context bias,
  variant, merging): the channel's layer name records which.
- Needs the validation role (`MLR-07`) for early stopping.

Tests: with a sparse site channel, "no site" and "unmodified site" give
different features (mask as input); features propagate across positions
between sites within the read span; the computed RF equals the formula; the effective span is at most the
theoretical RF and, for a model whose kernels are fixed to concentrate at the
centre, measurably smaller; perturbing one input position
changes pre-pooling features only within RF / 2 of it (empirical bound) for
bounded recipes and anywhere with SE on; `max_receptive_field` refuses an
over-wide config; translating a feature within the molecule leaves the
prediction unchanged up to edge effects (position-agnostic); each recipe
trains a few epochs, saves, reloads and explains (integrated gradients).

### `MLR-09` -- further neural families

MLP and transformer as `models/<arch>.py` with a config dataclass, builder
and registry recipe on the residual CNN's input contract (channel-first
values, observed mask); a project may build a registry from the built-ins
plus its own families and pass it to training (experimental architectures
live in the project until they earn a place in smftools).

Tests: each family trains a few epochs on a fixture, saves and reloads with
identical predictions, and explains with integrated gradients.

### `MLR-10` -- qualification

The `nkg2a_final` cell already compared by hand (fresh B6 vs NK; full locus,
E + P, E, P, intervening, downstream, E/P masked; NB, RF; NDR baselines)
through `MLR-01`-`MLR-05`: metrics equal the project's current tables;
attribution clustermaps for the RF and NB promoter / E + P models; then the
receptive-field ladder on the full locus.

## Project side (`nkg2a_final`)

- Runs replace the per-task `result*.json` / `folds*.csv` / predictions files;
  run tags carry the task id and model name.
- `metadata/ml_sets.yaml`: which tasks x models to run, evaluation sets to
  explain (held-out; applied cohorts), figure orderings (score, label, latent
  Leiden / NDR bins), comparisons to report.
- Experimental neural architectures in `project_scripts/ml/models/`,
  registered through a project registry.

## Out of scope

- XGBoost and SVM families (not planned; the registry can take them later).
- Hosted trackers (W&B / MLflow) and Hydra -- `ML-601` / `ML-602`, deferred.
- Hyperparameter search beyond what the validation role enables.
- Pretrained encoders (`ML-304`, gated).
