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
| `MLR-02` final models and apply | done (PR #701) | optional all-groups final model; apply a run to another dataset with records |
| `MLR-03` explanation records | done (PR #702) | out-of-fold position importance and per-molecule attribution matrices per run; fold consistency |
| `MLR-03b` detector catalogue | done (PR #714) | CNN detector catalogue (split from `MLR-03`) |
| `MLR-04` attribution clustermap | done (PR #704) | input layers beside attributions, shared row order, label / score / fold strips; detector catalogue figures |
| `MLR-05` run comparison | done (PR #705) | select runs by tags; paired per-fold metrics, bootstrap intervals, figures |
| `MLR-06` fold-matrix cache | done (PR #715) | read each task's data once for every model |
| `MLR-07` validation role | done (PR #706) | a stratified validation fraction of each fold's training molecules (default) or held-out training experiments, for early stopping and tuning; the test experiment stays whole; final models too |
| `MLR-08` detector-scale CNNs | done (PR #709) | position-agnostic residual dilated CNNs whose pattern detectors have a stated, enforced maximum span (receptive field): sub-nucleosome, 2-3, 4-6 nucleosomes, full locus; effective span measured per run |
| `MLR-09` model classes and the zoo | part 1 done (PR #716); part 2 (`conv_scanner`) implemented (`feature/ml-conv-scanner`); MLP / transformer pending | a capability class per registry family (additive, tabular non-linear, spatial, global sequence); class-aware defaults; MLP, multiscale CNN and transformer recipes; per-task zoo declarations |
| `MLR-10` qualification | in progress | `nkg2a_final` region / model grid through `MLR-01`-`MLR-05`; parity with its current metrics |
| `MLR-11` pretraining and fine-tuning | proposed | encoder / head split; a `pretrain` action (masked-site reconstruction, autoencoder, VAE) publishing head-less encoders; fine-tuning through `initialization`; pretraining-corpus leakage policy; transfer benchmark (absorbs `ML-304`) |

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

As built (`orchestration/explanations.py`): `explain_run(run_id, model=,
method=, parameters=None, max_per_fold=2000, background_size=100, seed=0)`
publishes an explain run whose manifest names the run's fold models (the
explain service now accepts several models of one key and training run).
Each fold's held-out molecules (class-stratified, seeded sample) are read
batch by batch -- never the whole split -- and explained by that fold's
model; background-dependent methods sample the fold's training molecules.
Records: `request.json`, `data/molecules.parquet` (fold, model, matrix row,
truth, out-of-fold score), `attributions/index.json` + `fold_NN.npy`
(float32 molecules x channels x positions; classical attributions summed over
each position's transformed features), `importance.parquet` (mean |a|, mean,
mean per true class; or a global method's value), `consistency.parquet`
(Spearman between folds), `summary.json` (mean consistency, top positions).
The re-bound snapshot and fold splits must equal the run's. Evaluating an
applied dataset, and layer methods (Grad-CAM), are left for later.

Tests (`test_explanation_records.py`): NB contributions plus the prior give
the fold model's log posterior odds; TreeSHAP rows plus the base value give
the stored out-of-fold probability; every held-out molecule is explained once
by its fold's model; importance on the planted signal, fold consistency;
seeded class-stratified sampling; a global method gives importance only;
indexing; unknown model / method and changed data refused; integrated
gradients for a torch run.

### `MLR-03b` -- detector catalogue

For convolutional runs, method `DetectorCatalogue` (parameters: top windows
per detector, similarity threshold for grouping) reads final-layer
activations before pooling, stored as an explain run like `MLR-03`.

As built (`orchestration/detectors.py`): `detector_catalogue_run(run_id,
model=, max_per_fold=2000, top_windows=50, window=None, group_similarity=0.8)`,
an explain run like `MLR-03`. Per molecule and detector only the maximum
activation and its position are kept (a full activation map would be GBs);
top windows are one per molecule, sized by the fold model's recorded 90 %
effective span; detectors are grouped by Spearman correlation of their
per-molecule maxima. Records: maxima, windows, mean patterns, a detector table
(AUROC alone, log2 enrichment of top windows, centre position mean / SD,
group) and `plot_detector_catalogue` (the largest fold's most predictive
detectors). The fixture project gained `signal`, `n_positions` and
`reads_per_barcode`.

Tests (`test_detector_catalogue.py`): a CNN trained on a planted motif (at
random positions in active molecules) has, in every fold, a most-predictive
detector whose mean pattern is the motif (correlation > 0.7; observed: exact),
AUROC > 0.75, spread centres, enriched top windows; windows one per molecule
and ranked; classical models refused.

### `MLR-04` -- attribution clustermap

As built: explain runs with per-molecule attributions also store each fold's
explained inputs (`attributions/inputs_NN.npy`, NaN where unobserved) and
draw `figures/attributions.png` (all folds, blocks by true class).
`analysis.plot.ml_results.plot_attribution_clustermap` (and
`attribution_row_layout`) renders from arrays alone: per channel, input beside
attributions; extra panels (e.g. HMM layers) and strips aligned with the
molecules; true-class (fixed colours), held-out and continuous score strips;
orders `label`, `score` or `bins` (given order, e.g. NDR states);
`coordinate_labels` (e.g. TSS-relative); a seeded class-stratified row cap.
`orchestration.plot_explanation` draws from a record. The shared clustermap
gains continuous strips.

Tests (`test_ml_attribution_clustermap.py`, `test_explanation_records.py`):
panels and strips share one row order; returned row ids are the drawn
molecules; symmetric scale; row orders; seeded class-stratified sampling;
shape checks; stored inputs equal the re-read data; default figure recorded;
custom orderings from the record; no figure for global methods.

Detector catalogue figures move to `MLR-03b`: per detector (or group), the
mean top-window pattern per input channel, its locus position histogram and
class enrichment; tests that they match the stored windows.

### `MLR-05` -- run comparison

As built (`orchestration/comparison.py`): `select_runs(tags=...)` from the run
index; `compare_runs(run_ids, models=, metrics=, reference=, label_tags=,
prevalence=0.10, n_bootstrap=200, seed=0)` from run records only. Folds are
matched by held-out group (unshared folds dropped with a warning; none shared
refused); within a fold, metrics are recomputed on the molecules every entry
predicted (counts of dropped molecules reported; disagreeing truth refused).
Metrics: AUROC, average precision, normalised AP, normalised AP at a fixed
prevalence (reweighted), with fast implementations equal to scikit-learn's.
Uncertainty: per-fold values, mean and SD across folds, folds better; a
class-stratified bootstrap over molecules within folds (shared resamples),
giving intervals for each entry's fold-mean and each paired difference.
Added after the `MLR-10` qualification (whose fold SDs of 2-3.5 dwarfed
the molecule intervals): a between-experiment interval (folds resampled) for
every mean and difference, and an exact paired sign-flip test on per-fold
differences. `RunComparison.write` saves tables, settings (source run ids) and
`plot_run_comparison`'s figure. Comparisons are recomputed, not published as
runs.

Tests (`test_run_comparison.py`, `test_ml_comparison_metrics.py`): selection
by tags; fold metrics and paired differences equal direct computation; seeded
bootstrap; shared-fold comparison with a warning; bad metric / reference
refused; written tables, settings and figure; fast metrics equal
scikit-learn's (with ties and weights).

### `MLR-06` -- fold-matrix cache

As built: a **decoded-row cache** rather than per-fold matrices. A row's
decoded content depends only on its molecule (and the snapshot's positions,
channels, coordinate maps), not on its split, so `PartitionRowCache`, shared
by every fold, final split and model of a bound job, answers the reader's
`_read_batch`: each molecule is decoded from the stores once per bound job,
and batching is unchanged, so results equal an uncached read exactly.
`PartitionReadPolicy.row_cache_bytes` (default: the materialization budget;
0 disables) caps it; past the cap rows are read as before. Not persisted
across processes (a later run of the same task reads again) -- a disk layer
in the workspace `datasets/` is a possible follow-up.

Tests (`test_row_cache.py`): every molecule decoded once over 3 folds x 2
models (vs > 3x without the cache); predictions equal an uncached run; a
capped cache stays within budget with equal results; a cache refuses a
different snapshot / positions. The plan's original tests: a second model reuses the cache (no store reads); the cache key changes
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

As built: `leave_one_group_out` takes `validation_fraction` and
`validation_by` (`molecules` default, or `groups`); omitted when unset, so
earlier plan hashes and split ids are unchanged. Folds draw validation from a
generator fixed by the split seed and fold name. `SplitManifest.shared_roles`
(empty or train + validation) relaxes isolation for the declared pair only,
recorded in the split identity when used. Final splits
(`final_split_assignments`) take the same validation, so final torch models
train; torch fits without a test role record `test_loss = None`. Sklearn
models train on the train role only, so a declared fraction also removes
those molecules from their training (the same training set for every model
of a job).

Tests (`test_validation_fraction.py`): test molecules never appear in train or validation, and test is one
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

As built: `ResidualCNNConfig` gains `receptive_field` (computed),
`max_receptive_field` (refused when exceeded or with squeeze-excite),
`mask_channels` (validity mask appended as input channels) and
`span_masking` (features kept and pooled over each read's first-to-last valid
span); all default off and are left out of `to_dict` when unset, so
`residual_dilated_cnn_v1` and published models keep their identity.
`effective_span` measures the 50 % / 90 % influence widths from gradients of
final-layer features. Recipes `rcnn_subnucleosome_v1` (113 bp),
`rcnn_2_3_nucleosomes_v1` (513), `rcnn_4_6_nucleosomes_v1` (1,025),
`rcnn_full_locus_v1` (5,121): 64-channel blocks, squeeze-excite off, mask
channels and span masking on, `max_receptive_field` = their span. Train runs
record each residual CNN fold model's `detector_scale` (theoretical field,
effective spans, measured on up to 32 held-out molecules) in `models.json`.
The three input arms (sites / HMM / both) are plan choices (channels), not
code.

Tests (`test_ml_detector_scale_cnn.py`, `test_detector_scale_runs.py`): with a sparse site channel, "no site" and "unmodified site" give
different features (mask as input); features propagate across positions
between sites within the read span; the computed RF equals the formula; the effective span is at most the
theoretical RF and, for a model whose kernels are fixed to concentrate at the
centre, measurably smaller; perturbing one input position
changes pre-pooling features only within RF / 2 of it (empirical bound) for
bounded recipes and anywhere with SE on; `max_receptive_field` refuses an
over-wide config; translating a feature within the molecule leaves the
prediction unchanged up to edge effects (position-agnostic); each recipe
trains a few epochs, saves, reloads and explains (integrated gradients).

### `MLR-09` -- model classes and the zoo

A *family* is one implementation and a *recipe* one configured variant; a
**model class** is what a model can represent, which is what a task's zoo
should span:

| class | represents | families (now / planned) | default explanation |
|---|---|---|---|
| `additive` | independent per-position evidence | naive Bayes, logistic regression | exact contributions (log-odds, coefficients) |
| `tabular_nonlinear` | interactions among any positions, no adjacency | random forest, MLP | TreeSHAP; integrated gradients (MLP) |
| `spatial` | local, translation-invariant patterns of bounded span | residual dilated CNN (`MLR-08` ladder), multiscale CNN | integrated gradients; detector catalogue (`MLR-03b`); effective span |
| `global_sequence` | dependencies between any positions | transformer | integrated gradients; attention |

- `ModelFamilyDefinition` gains `model_class` (one of the above); run
  records and the index carry it; `compare_runs` can name and group entries by
  class; `explain_run` defaults its method by class (and family).
- A task's zoo is the model list of its train job: every member sees the same
  molecules and folds, so the paired comparison (`MLR-05`) is across classes
  by construction. Projects declare zoos per task type (`ml_sets.yaml`).
- New members on the residual CNN's input contract (channel-first values,
  validity masks; mask channels / span masking as in `MLR-08`):
  - **MLP** (`tabular_nonlinear`, torch): flattened positions x channels
    plus validity; the non-linear counterpart of the random forest with
    gradients.
  - **Multiscale CNN** (`spatial`): parallel branches of different spans
    (e.g. the ladder's) concatenated before pooling -- one model, several
    detector scales, each branch's span recorded.
  - **Transformer** (`global_sequence`): tokens per position (or per patch
    of positions, for 4.7 kb inputs), masked attention over valid positions.
    **Positional encoding is a recipe-level decision, recorded:** absolute
    encodings make the model position-aware (unlike the CNN ladder);
    relative encodings keep it closer to position-agnostic. Both are
    legitimate; they answer different questions.
- A project may register its own families (with a class) and pass the
  registry to training; experimental ones live in the project until they earn
  a place in smftools.

Part 2 as built -- **convolutional scanners** (`models/conv_scanner.py`,
family `conv_scanner`, class `spatial`), motivated by the `MLR-10` ladder
(detector span barely mattered; ~16 detectors sufficed at the
sub-nucleosome span) and its catalogue (strong detectors fired at single
positions -- with mask channels the C-site layout is a sequence fingerprint,
so position leaks into "position-agnostic" models):

- a stack of conv layers (`filters`, `kernel_sizes`, `dilations`), optional
  max-pool `downsample` between layers, per-filter pooling (`max`, `avg`,
  `attention`, or `adaptive_bins` for coarse position), a linear (or
  `head_hidden`) head; mask channels / span masking via the residual CNN's
  shared `MaskedConvInputs`; `receptive_field` and `feature_stride`, which
  effective spans and detector catalogues use to map feature positions back
  to input positions (and any CNN with a receptive field now records its
  detector scale);
- recipes `motif_scanner_k21/k51/k151_v1` (one layer, 16 filters, max pool,
  linear: 0.7-4.9k parameters), `adaptive_scanner_k51_v1` (1.9k),
  `two_layer_scanner_v1` (2.9k, receptive field 50, stride 4),
  `downsampling_scanner_v1` (28.7k, ~744 bp, stride 64).
- Tests: formulas, validation, round trip, shapes, translation invariance of
  global max vs position-awareness of adaptive bins, effective span with
  downsampling; on a planted motif a 1-layer scanner's filter weights *are*
  the motif in every fold, catalogue / IG / detector scale work, downsampled
  catalogue positions stay in the molecule.
- Project grid to run: inputs (C + mask; C without mask; HMM accessible at
  every position; HMM accessible + footprint length) x scanners -- the HMM
  inputs carry no site-layout fingerprint, so filters must learn shapes that
  can occur anywhere.

Part 1 as built: `MODEL_CLASSES`; `ModelFamilyDefinition.model_class` and
`default_explanation` (optional, validated; every built-in declares both --
nb / logistic regression additive, random forest tabular non-linear,
residual CNN spatial); `models.json` records each model's class;
`compare_runs` entries carry it; `explain_run` defaults its method to the
family's. Pending: MLP, multiscale CNN and transformer families.

Tests: every built-in family declares a class; runs, index and comparisons
carry it; default explanation per class; each new family trains a few epochs
on the fixture with a validation fraction, publishes, reloads with identical
predictions and explains with integrated gradients; the multiscale CNN
records each branch's span; transformer recipes record their positional
encoding, and a relative-encoding transformer's prediction is unchanged by
translating a pattern away from the edges.

### `MLR-10` -- qualification

The `nkg2a_final` cell already compared by hand (fresh B6 vs NK; full locus,
E + P, E, P, intervening, downstream, E/P masked; NB, RF; NDR baselines)
through `MLR-01`-`MLR-05`: metrics equal the project's current tables;
attribution clustermaps for the RF and NB promoter / E + P models; then the
receptive-field ladder on the full locus.

### `MLR-11` -- pretraining and fine-tuning

Absorbs `ML-304` (proposed in the completed ML program, never built; its
schema slots exist: lineage kinds `pretrained` / `fine_tuned`, plan models'
`initialization`, a reserved `pretraining_task` mask role).

- **Encoder / head split.** Torch families expose an encoder (everything
  before the head -- for the residual CNN, stem + blocks + pooling) and a head;
  a classifier head, a reconstruction decoder or a VAE's latent heads attach
  to the same encoder.
- **`pretrain` action.** A plan job with a dataset (labels not required),
  an objective and its corruption policy:
  - masked-site reconstruction (hide a fraction of observed calls, predict
    them; loss only at hidden observed sites);
  - autoencoder reconstruction;
  - VAE (reconstruction + KL; the latent is also a per-molecule embedding,
    an alternative input to the latent analyses).
  It publishes an **encoder artifact**: lineage `pretrained`, no classifier
  head, no label schema, with its objective, corruption policy and corpus.
- **Corruption masks stay distinct** from validity: a hidden site is
  observed-but-masked, never "no site" or "unobserved"; with `mask_channels`
  the model must not be able to read the corruption from its validity input.
- **Fine-tuning** through a model's `initialization`:
  `{"kind": "pretrained", "model": "model:<encoder id>", "freeze": "encoder" |
  "none" | {"schedule": ...}}`; a new head is attached; the run records the
  parent encoder (id, checksum) and the freeze schedule; the model's lineage
  is `fine_tuned`.
- **Leakage policy (declared, recorded).** Unlabeled pretraining could use a
  fold's held-out experiment, inflating fine-tuned scores. Policies:
  `per_fold` (pretrain without the fold's test experiment -- honest, one
  encoder per fold), `external` (a separate corpus: other experiments,
  cells, alleles) or `transductive` (everything, declared as such). Fine-tune
  runs refuse a `per_fold` encoder from another fold.
- **Transfer benchmark.** Fine-tuned against from-scratch on the same folds
  (`compare_runs`, entries tagged by initialization), including a frozen-
  encoder linear probe.

Tests: an encoder artifact loads without a head or label schema; a
fine-tuned model records its parent and freeze schedule and reloads; a
`per_fold` encoder is refused for another fold; masked-site loss reads only
hidden observed sites and the corruption is not visible through validity
channels; a VAE's latent is exported per molecule; the transfer comparison
pairs fine-tuned and from-scratch entries on shared folds.

## Project side (`nkg2a_final`)

- Runs replace the per-task `result*.json` / `folds*.csv` / predictions files;
  run tags carry the task id and model name.
- `metadata/ml_sets.yaml`: which tasks x models (a zoo per task type, one or
  more models per class, `MLR-09`) to run, evaluation sets to
  explain (held-out; applied cohorts), figure orderings (score, label, latent
  Leiden / NDR bins), comparisons to report.
- Experimental neural architectures in `project_scripts/ml/models/`,
  registered through a project registry.

## Out of scope

- XGBoost and SVM families (not planned; the registry can take them later).
- Hosted trackers (W&B / MLflow) and Hydra -- `ML-601` / `ML-602`, deferred.
- Hyperparameter search beyond what the validation role enables.
- Pretrained encoders now planned as `MLR-11`.
