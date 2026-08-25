# smftools ML infrastructure audit

Date: 2026-07-30. Scope: `~/git/smftools` at its current working-tree state (includes uncommitted
changes from active development — see "State of the audited tree" below), cross-referenced against
the downstream consumer project `Nkg2a_DAFseq_merged/claude_scripts/ml/`. Written for whoever picks
up the ML-infrastructure reorganization work next; read the "Bottom line" at the end first if you
want the short version.

## State of the audited tree

This audit describes `~/git/smftools` (main worktree, branch `main`, HEAD `4e1b1e57` at audit
time), **including uncommitted working-tree changes**: `analysis/compute/ml_cnn.py` and
`analysis/plot/ml.py` are locally modified (+752/-88 lines net), and `analysis/compute/
differential_abundance.py` + `analysis/plot/differential_abundance.py` are new, untracked files.
These changes added the Transformer/CNN-Transformer-hybrid architectures, GradientSHAP support, and
differential-abundance tooling described below — they are real and in active use for training right
now, just not yet committed.

**`~/git/smftools-work-2` (where this file lives) is a *different, older* worktree** — checked out
at `12635d53` (2026-07-22, four commits and the above uncommitted work behind `main`). Don't expect
`smftools-work-2`'s own `src/` to match anything described here; this audit is about the *package's
current state*, not about the specific worktree that happens to host this document.

## Bottom line

There are **three independent ML surfaces** inside `smftools`, built at different times, that don't
talk to each other:

1. **`analysis/compute/ml_cnn.py` + `ml_metrics.py` + `ml_explanations.py` + `ml_splits.py` +
   `analysis/plot/ml.py`** — the "tier 2" library the current DAFseq project (`claude_scripts/ml/`)
   actually uses. Plain functions and a few `nn.Module`s, no Lightning, no base-class hierarchy.
   Actively developed (this week).
2. **`smftools.machine_learning`** — a second, more architecturally mature ML framework
   (PyTorch Lightning `LightningModule`/`LightningDataModule`, a `BaseTorchModel` class hierarchy
   with built-in Captum/Grad-CAM interpretability, sklearn-model wrapper with a matching interface,
   evaluators). **Nothing in `smftools` imports it, and nothing in the current project uses it.** It
   looks like an earlier or parallel attempt at exactly the kind of framework this audit was asked
   to design — see §5 before building anything new.
3. **`hmm/`** — a from-scratch PyTorch HMM with EM fitting, a class registry (`@register_hmm`), and
   the most mature artifact-persistence system in the package (content-addressed, atomic-write,
   conflict-checked, file-locked). This is production code (wired into the CLI pipeline), not
   analysis-tier code, but its *artifact system* is the best existing precedent for the
   "how do we track training instances" question this audit was asked to answer.

Plus a fourth, dead one: `tools/archived/classifiers.py` — an earlier CNN/RNN/Transformer
implementation, archived, not imported anywhere.

None of the three live surfaces share a config-dataclass hierarchy, an artifact format, or a model
registry, despite each having independently reinvented a version of at least one of those three
things. The highest-leverage fix is not "adopt Lightning" or "add Hydra" — it's **picking one of the
three existing artifact-identity patterns (§5.3) and one of the two existing model-registry patterns
(§5.2) and making everything else use them**, since duplicated-but-incompatible versions of both
already exist in the tree today.

---

## 1. Current layout

### 1.1 `analysis/compute/` + `analysis/plot/` — the tier-2 library `claude_scripts/ml/` uses

```
analysis/compute/
├── ml_cnn.py              1,073 ln   CNN/Transformer/Hybrid nn.Module architectures,
│                                     training loop, Captum attribution (IG/DeepLift/GradientSHAP)
├── ml_metrics.py            227 ln   sklearn Pipeline builders (NB/RF/XGBoost) + eval metrics
├── ml_explanations.py       133 ln   NB log-odds / RF SHAP / XGBoost native-TreeSHAP attribution
├── ml_splits.py              112 ln   leave-one-group-out CV fold builder + leakage checks
├── dimensionality_reduction.py 331 ln  PCA/diffusion-map → UMAP → Leiden pipeline
├── clustering.py              87 ln   Ward-linkage row ordering for heatmaps (not a model)
├── differential_abundance.py 151 ln   log2-odds-ratio + bootstrap/permutation test (new, uncommitted)
├── ep_classification.py      122 ln   rule-based NDR classifier from HMM output (not ML)
├── hmm_features.py            79 ln   interval extraction from HMM layers (post-processing)
├── metrics_store.py           89 ln   Zarr-backed metrics store (storage, not ML)
├── pearson.py                  58 ln   correlation matrix helper
├── autocorrelation.py        320 ln   periodicity analysis
├── ls_periodicity.py         379 ln   Lomb-Scargle/FFT periodogram
└── read_cache.py             432 ln   Zarr read-cache I/O

analysis/plot/
├── ml.py                     789 ln   ROC/PR curves, training curves, score heatmaps, metric bars
├── embeddings.py             645 ln   PCA/UMAP scatter, density grid, cluster-composition bars
├── differential_abundance.py 100 ln   forest plot for compute/differential_abundance.py (new)
├── autocorr.py                465 ln   autocorrelation/periodogram plots
├── histograms.py              267 ln   interval histograms + Gaussian fits
├── heatmaps.py                 66 ln   Pearson heatmap
└── locus.py                     9 ln   stub only — plot_locus_map() "planned," not implemented
```

**Design contract** (stated in `analysis/compute/__init__.py`'s docstring and confirmed by
`ml_cnn.py`'s own comment at line ~264): this tier is pure, array-in/array-out, no AnnData, no file
I/O, and **deliberately does not depend on tier-3 project code** — it's meant to be called from
project-level driver scripts, not from `smftools.cli`. That contract is upheld: nothing in `cli/`
imports `analysis.compute.ml_cnn`/`ml_metrics`/`ml_explanations`/`ml_splits`, and nothing in
`analysis/plot/ml.py`/`embeddings.py`/`differential_abundance.py` is imported by `cli/` either.
`analysis/compute/dimensionality_reduction.py` is the one exception — it's used both by
`claude_scripts/` (tier 3) and internally by `project/embedding_store.py` and `analysis/plot/
embeddings.py`.

**Model classes here** (all in `ml_cnn.py`): `ResidualDilatedCNN1d`, `TokenTransformerClassifier`,
`CNNTransformerHybrid`, plus shared building blocks `SqueezeExcite1d`, `ResidualDilatedBlock1d`,
`AttentionPooling1d`, `PoolingClassifierHead`, and `_LogitWrapper` (a Captum adapter). Three parallel
config dataclasses (`CNNConfig`/`TransformerConfig`/`HybridConfig`) plus one shared result dataclass
(`TrainedCNNModel`, carrying `model_type: str` for save/load dispatch). A small dict-based registry
(`_SEQUENCE_MODEL_BUILDERS`/`_SEQUENCE_CONFIG_TO_DICT`/`_SEQUENCE_CONFIG_FROM_DICT`, keyed on
`"cnn"`/`"transformer"`/`"hybrid"`) does config (de)serialization and model construction generically
by `model_type` string — this is the newest piece (added this week) and is the closest thing this
tier has to a registry pattern.

**No model persistence lives in this tier at all.** `fit_simple_cnn`/`fit_simple_transformer`/
`fit_cnn_transformer_hybrid` keep a best-epoch `state_dict` in memory (for early stopping) but never
write to disk — saving/loading is entirely the caller's job (currently `claude_scripts/ml/
model_artifacts.py`, tier 3 — see §2).

### 1.2 `smftools.machine_learning` — the disconnected Lightning framework

```
machine_learning/
├── models/
│   ├── base.py            286 ln   BaseTorchModel(nn.Module) — saliency, IG, DeepLift, occlusion,
│   │                                Grad-CAM, apply_attributions_to_adata()
│   ├── cnn.py, mlp.py, rnn.py, transformer.py, positional.py, wrappers.py
│   │                                CNNClassifier / MLPClassifier / RNNClassifier /
│   │                                TransformerClassifier / DANNTransformerClassifier /
│   │                                MaskedTransformerPretrainer / DANNTransformer
│   │                                (all subclass BaseTorchModel except the utility ones)
│   ├── lightning_base.py  381 ln   TorchClassifierWrapper(pl.LightningModule) — full train/val/
│   │                                test/predict loop, ROC/PR logging
│   └── sklearn_models.py  295 ln   SklearnModelWrapper — same-shaped interface, sklearn backend
├── data/
│   └── anndata_data_module.py 341 ln   AnnDataDataset + AnnDataModule(pl.LightningDataModule)
├── training/
│   ├── train_lightning_model.py 154 ln   pl.Trainer + EarlyStopping/ModelCheckpoint wrapper
│   └── train_sklearn_model.py   110 ln
├── evaluation/
│   ├── evaluators.py       241 ln   ModelEvaluator, PostInferenceModelEvaluator
│   └── eval_utils.py        30 ln
├── inference/               4 files   sklearn/lightning/sliding-window inference runners
└── utils/                   device.py, grl.py (gradient-reversal for domain adaptation)
```

22 files, 2,806 lines. Every file goes through `smftools.optional_imports.require(...)` so the
package still imports cleanly without `torch`/`pytorch_lightning`/`shap` installed — a pattern
`analysis/compute/ml_cnn.py` does *not* follow (it imports `torch`/`captum` eagerly at module load).

This is a **complete, working, Lightning-based classifier framework** — `BaseTorchModel` already has
the interpretability methods (saliency, IG, DeepLift, occlusion, Grad-CAM) that `ml_cnn.py`
re-implements independently via a different mechanism (Captum + `_LogitWrapper`), and
`TorchClassifierWrapper` already has the train/val/test loop, early stopping, and ROC/PR logging
that `ml_cnn.py`'s `_fit_binary_torch_classifier` reimplements by hand. **Nothing calls any of it.**
Confirmed by full-tree grep: the only reference anywhere in `smftools` outside `machine_learning/`
itself is the lazy `__getattr__` re-export in `smftools/__init__.py` (`"ml":
"smftools.machine_learning"`). `cli/` doesn't use it. `claude_scripts/ml/` doesn't use it (it uses
`analysis.compute.ml_cnn` instead — presumably because that's what got extended for the DAFseq
project's specific needs, or because whoever wrote `claude_scripts/ml/` didn't know this package
existed).

There's also a `DANNTransformerClassifier`/`DANNTransformer` (domain-adversarial, via a
gradient-reversal layer in `utils/grl.py`) — genuinely more sophisticated than anything in
`ml_cnn.py`, and directly relevant if you ever want a model that's invariant to (say) enzyme or run
identity while still predicting activity state.

### 1.3 `hmm/` — production model-fitting code, not analysis-tier, but the best artifact system

```
hmm/
├── HMM.py                2,422 ln   BaseHMM(nn.Module) + registry (@register_hmm) +
│                                     SingleBernoulliHMM / MultiBernoulliHMM /
│                                     DistanceBinnedSingleBernoulliHMM + create_hmm() factory
├── model_artifacts.py      389 ln   content-addressed checkpoint system — see §5.3
├── fit_plan.py              310 ln   deterministic, hash-seeded fit planning for distributed fitting
├── call_hmm_peaks.py, nucleosome_hmm_refinement.py, display_hmm.py, hmm_readwrite.py
│                                     post-processing / plotting / legacy save-load
└── archived/                        4 superseded files
```

Wired directly into `cli/hmm_adata.py` (1,610 lines) — HMM fitting is a *pipeline stage*
(BAM→AnnData feature generation), not a downstream analysis, so its being in `cli/`-adjacent code
rather than `analysis/` is architecturally correct, not a layering violation. What's relevant to
this audit is purely `model_artifacts.py`'s design — detailed in §5.3, because it's the strongest
existing answer to "how do we track which model was trained on what."

### 1.4 `smftools.cli`

Click-based (`cli_entry.py`, `@click.group`), with ~30 subcommands. The actual stage logic lives in
`cli/` as plain functions (`raw_adata.py`, `load_adata.py`, `preprocess_adata.py`, `hmm_adata.py`,
`latent_adata.py`, etc. — 13,519 lines across 17 files), framework-agnostic (only `cli_entry.py`
itself imports `click`). `cli/latent_adata.py` computes PCA/UMAP/Leiden via a **separate, older**
implementation (`tools/calculate_umap.py`, `tools/calculate_leiden.py`), not via `analysis/compute/
dimensionality_reduction.py` — a third parallel implementation of the same operation, worth
noting even though it's out of scope for the classifier-focused parts of this audit.

---

## 2. What's delegated to `claude_scripts/ml/` (tier 3, project-level)

18 files, ~9,600 lines, in the downstream `Nkg2a_DAFseq_merged` project (not part of `smftools`
itself). Two groups:

**A. Genuinely project-specific drivers** (correctly tier-3 — reference `comparisons.yaml`, this
project's `REGION_SPANS`/enzyme/genotype vocabulary, this project's Zarr cache layout):
`b6_cross_dataset_activity.py` (2,821 ln — the main variant-comparison driver, ~26 model variants
A–Z), `full_locus_wt_enh_del_clustermap.py` (601 ln), `region_masking_comparison.py` (347 ln),
`f1_activity_classifiers.py` (674 ln), `f1_activity_tasks.py` (213 ln), `f1_activity_rf_
explanations.py` (572 ln), `f1_activity_cnn_explanations.py` (734 ln), `f1_activity_cnn_
consistency.py` (240 ln), `f1_activity_classifier_barplots.py` (305 ln), `masks.py` (303 ln — region
mask vocabulary), `io.py` (248 ln — project data loader).

**B. Generic infrastructure that landed in tier 3 by circumstance, not by design** — the masked-site
transformer pretraining track:

| File | Generic core | Project-specific wrapper |
|---|---|---|
| `transformer_models.py` (187 ln) | **All of it.** `MaskedSiteTransformer`/`MaskedSiteTransformerClassifier`, token vocab, sinusoidal PE. Zero project imports. | — |
| `transformer_io.py` (752 ln) | `matrix_to_tokens`, `tokenize_onto_union_axis`, `pool_sample_table_onto_union_axis` (~150-200 ln) | `TransformerDatasetBundle`, run/barcode literals, Zarr cache conventions (~500 ln) |
| `transformer_pretrain.py` (775 ln) | `TrainConfig`, `TokenMatrixDataset`, `MaskingCollator`, `_run_epoch`, `_metrics_from_logits`, `_select_device`, `_split_indices` (~80% of file) | `run_pretraining`, `build_argparser`/`main`, output-path wiring |
| `transformer_finetune.py` (375 ln) | `fit_finetune_classifier`, `predict_finetune_scores` (~half) | `build_finetune_sample_table`, hardcoded enzyme names, `run_finetune` |
| `transformer_apply.py` (345 ln) | `chunked_ig_attributions` (general Captum `LayerIntegratedGradients`, chunked over batch — mirrors `ml_cnn.py`'s own OOM-avoidance chunking) | `allele_family`, `_activity_labels`, `run_apply_to_corpus` |

**Concrete, already-visible cost of this misplacement**: `analysis/compute/ml_cnn.py` (tier 2) has
its own `_sinusoidal_position_table` (line ~259), with a docstring that says, verbatim, it's
"duplicated here rather than imported since tier-2 (smftools) does not depend on tier-3 (project)
code." That's a correct call *given the current package boundary* — but it's evidence the boundary
is in the wrong place. `transformer_models.py` has zero project dependencies; it could move into
`smftools` today with no changes, and then `ml_cnn.py` could import the real thing instead of
maintaining a parallel copy.

---

## 3. Artifact/checkpoint saving — three incompatible systems, one of them good

| System | Location | Format | Content-addressed? | Atomic write? | Conflict detection? |
|---|---|---|---|---|---|
| HMM | `hmm/model_artifacts.py` | `torch.save` + `.json` sidecar | **Yes** (SHA-256 of canonical key → `model_id`) | **Yes** (`atomic_torch_save`: tmp file + fsync + `os.replace`) | **Yes** (`publish_checkpoint` raises `HMMArtifactConflictError` on hash mismatch rather than silently overwriting) |
| Latent/embedding models | `latent_model_artifacts.py`, `project/embedding_store.py` | `pickle.dump` | Yes (`LatentModelArtifact`/`latent_model_id`, independently reimplemented) | Partial | Partial |
| CNN/Transformer/Hybrid classifiers | `claude_scripts/ml/model_artifacts.py` (**tier 3, not in `smftools` at all**) | `torch.save(state_dict)` / `joblib.dump` + `.npy` sidecars + `metadata.json` | **No** — directory name (`<fold_dir>/artifacts/`) is the only identity, no hash | No | No — a second training run with the same variant name silently overwrites |

The HMM system is the one to standardize on: canonical dataclass key → deterministic hash → `model_id`
→ sharded path → atomic write → locked concurrent-fit protection → checksum-verified load. Nobody
else in the tree needs the file-locking part (classifier training in this project is single-worker),
but the hash-identity + atomic-write + conflict-check trio is exactly what's missing from
`claude_scripts/ml/model_artifacts.py` today, and its absence is a real risk: two people (or two
sessions) training a variant with the same name at the same time, or re-running training after a
code change without renaming the variant, currently overwrites the previous model with no warning
and no way to tell after the fact which config actually produced the artifact on disk.

Also worth noting: **`joblib` is a declared dependency but is never actually used for
persistence anywhere in the tree** (grep confirms zero `joblib.dump`/`joblib.load` I/O calls outside
`claude_scripts/ml/model_artifacts.py`, which is tier-3/outside-`smftools` anyway) — `pickle` and
`torch.save` are what the package itself actually uses.

---

## 4. Model class organization — current state and a concrete proposal

**Current state**: three unrelated class hierarchies (`ml_cnn.py`'s free-standing `nn.Module`s with
no common base beyond `nn.Module` itself; `machine_learning/models/base.py`'s `BaseTorchModel`
hierarchy; `hmm/HMM.py`'s `BaseHMM` hierarchy with a decorator registry). No shared base class, no
shared config-dataclass base, two different dict/decorator registry idioms
(`_SEQUENCE_MODEL_BUILDERS` plain dicts vs. `@register_hmm` decorator + `_HMM_REGISTRY` dict) doing
conceptually the same job.

**Proposal**, scoped to the classifier-model surface this audit was asked to focus on (not
re-architecting HMM, which is working production code):

1. **Adopt the HMM registry idiom for classifiers, not the ad-hoc dict pattern.** A
   `@register_model("cnn")`-style decorator, mirroring `hmm/HMM.py`'s `register_hmm`, would replace
   `ml_cnn.py`'s three parallel `_SEQUENCE_CONFIG_TO_DICT`/`_SEQUENCE_CONFIG_FROM_DICT`/
   `_SEQUENCE_MODEL_BUILDERS` dicts with one registration point per model class, and would extend
   cleanly to `machine_learning/`'s `CNNClassifier`/`MLPClassifier`/`RNNClassifier`/
   `TransformerClassifier`/`DANNTransformerClassifier` if/when that framework gets reconnected
   (§5.1) — right now, adding a new model type means editing three separate dicts in `ml_cnn.py`;
   with a decorator, it means adding the decorator to the new class and nothing else.

2. **One base config dataclass**, something like:
   ```python
   @dataclass
   class ModelConfig:
       model_type: str  # registry key, redundant with the decorator but needed for serialization
       in_channels: int
   ```
   with `CNNConfig`/`TransformerConfig`/`HybridConfig` (and eventually `machine_learning/`'s
   configs, which currently don't even exist as dataclasses — `CNNClassifier`/`MLPClassifier`/etc.
   take constructor kwargs directly, no config object at all) inheriting from it. This gives
   `sequence_config_to_dict`/`from_dict` a single generic implementation (`dataclasses.asdict` +
   tuple/list coercion for the fields that need it) instead of three hand-written pairs.

3. **A thin `BaseClassifier` protocol** (not necessarily a base class, could be a `Protocol` if you
   want to keep `ml_cnn.py`'s current free-function style rather than adopting
   `machine_learning/`'s OOP style) that guarantees: `forward(x) -> logit`, `.config` attribute,
   `.model_type` attribute. This is *almost* what `TrainedCNNModel` already provides as a wrapper —
   the gap is that it's a wrapper around a model+metadata bundle, not a contract the model itself
   satisfies, so generic code (attribution, artifact save/load) has to know about `TrainedCNNModel`
   specifically rather than "any registered classifier."

4. **Decide, deliberately, whether `machine_learning/` is dead code to delete or a framework to
   finish connecting.** Right now it's neither — it's fully-built, unused, and silently drifting out
   of sync with `ml_cnn.py`'s newer patterns (e.g. `machine_learning/models/positional.py`'s
   `PositionalEncoding` is a third, independent sinusoidal-PE implementation, alongside `ml_cnn.py`'s
   and `transformer_models.py`'s). Whichever way this goes, leaving it as-is means every future
   contributor has to independently discover it exists and decide for themselves whether to use it
   — that discovery cost is the actual problem, more than which framework wins.

---

## 5. Would Lightning / wandb / Hydra help?

Checked actual usage, not just `pyproject.toml` extras: **`wandb` has zero imports anywhere in the
package. `hydra`/`omegaconf` have zero imports anywhere.** Both are declared in the `ml-extended`/
`all` extras but are currently pure aspiration — nothing in the tree uses them. `pytorch-lightning`
(declared as `lightning`, imported as `pytorch_lightning` — a naming mismatch worth fixing
regardless of what else changes) *is* used, but only inside the disconnected `machine_learning/`
package.

### 5.1 Lightning

Worth adopting, *if and only if* `machine_learning/`'s `TorchClassifierWrapper`/`AnnDataModule` get
actually connected to the current work rather than left parallel. The value proposition is real:
`ml_cnn.py`'s `_fit_binary_torch_classifier` hand-rolls exactly what `LightningModule.training_step`
+ `Trainer(callbacks=[EarlyStopping, ModelCheckpoint])` already provides (early stopping on val AUC,
best-checkpoint tracking, per-epoch history) — `machine_learning/models/lightning_base.py` already
built this once. The honest cost: migrating `CNNConfig`/`TransformerConfig`/`HybridConfig`-based
models into `LightningModule`s is real work, not a config flag, and the current CV-fold-per-process
training pattern (each fold trains a fresh model in a plain Python loop, no distributed/multi-GPU
need) doesn't obviously need what Lightning is best at (multi-GPU/multi-node orchestration,
gradient accumulation, mixed precision at scale). The `Trainer`'s early-stopping/checkpointing alone
would still be a net simplification even single-GPU. **Recommendation: don't adopt it opportunistically
for one new model — decide whether `machine_learning/` is the future, and if so, migrate `ml_cnn.py`'s
three architectures into it deliberately as one project, rather than letting a third partial Lightning
integration accumulate.**

### 5.2 Weights & Biases

Valuable specifically for the problem this audit flags in §6 (no cross-variant training-run index).
Every variant's `cv_metrics_summary.csv` today is a flat file in its own directory tree; W&B (or any
experiment tracker) would give a queryable, comparable, timestamped record for free, plus training
curves already computed (`TrainedCNNModel.history`) are exactly the kind of thing it's built to log.
Low integration cost relative to payoff — this is the single highest-ROI addition of the three, and
doesn't require resolving the `machine_learning/` question first (it can be added to the current
`ml_cnn.py`-based training loop directly: a handful of `wandb.log(...)` calls inside
`_fit_binary_torch_classifier`'s epoch loop, and one `wandb.init(...)`/`wandb.finish()` around each
`fit_simple_*` call). Self-hosted `mlflow` is a reasonable alternative if avoiding an external SaaS
dependency matters — same value proposition, more setup.

### 5.3 Hydra / OmegaConf

Lowest priority of the three, and possibly the wrong tool for this codebase's actual shape. Hydra's
value is composable, override-able YAML configs for large sweep/multirun setups; this project's
current sweep pattern (variants A–Z, each a Python function call with explicit kwargs, run via
one-off scripts) is not YAML-driven at all, and introducing Hydra would mean either (a) a large
rewrite of every `train_variant_*` function into Hydra's config-injection style, or (b) a thin,
low-value Hydra shim around code that stays procedural underneath. The `ExperimentConfig` dataclass
in `config/experiment_config.py` (the pipeline-wide config for the BAM→AnnData CLI stages) is a much
closer fit for Hydra/OmegaConf if it's ever adopted — but that's a separate, larger decision about
the CLI pipeline, not something this classifier-focused audit should recommend piggybacking on.
**Recommendation: skip for now**, revisit only if variant sweeps grow enough (many more than 26)
that hand-writing each training script stops scaling.

---

## 6. Training instances / data provenance — nothing tracks this today

Confirmed by exhaustive search: **no central index anywhere** — no CSV/JSON/SQLite file that maps
model variant → task/enzyme/region/layer used → training date → CV results, queryable across
variants. What exists instead:

- Per-variant `cv_metrics_summary.csv` under each variant's own `OUTPUT_ROOT/models/<variant_name>/`
  — accurate but siloed, one file per model, no cross-model view.
- `f1_activity_classifiers.py`'s `collect_existing_summaries`/`_merge_leaf_summary` — aggregates
  *within that one driver's* task/enzyme/mask/feature-set slices, but not across drivers (doesn't
  include `b6_cross_dataset_activity.py`'s or the transformer track's results).
- Hand-written README.md status tables (e.g. `ml_outputs/b6_cross_dataset_activity/README.md`) —
  human-maintained prose, not machine-queryable, and already incomplete (one README only covers
  4 of the ~26 variants that now exist).
- No timestamp, git commit hash, or config hash is stamped on any classifier artifact anywhere.

**Proposal**: a single append-only `training_runs.jsonl` (or a small SQLite file, if querying by
multiple fields matters) at a project-level location, one record per `fit_simple_*`/
`_train_sklearn_multi_celltype_variant` call, written automatically by `run_cv_and_final_model`/
its sklearn equivalent (i.e. inside the shared training helpers, not something each caller has to
remember to log). Minimum useful fields, modeled directly on what `HMMModelKey` already tracks for
HMM fits: `variant_name`, `model_type`, `task_id`, `enzyme`, `region_key`, `layer`, `git_commit`
(`smftools` *and* the project repo, since both can change independently), `config_hash` (hash of the
resolved `CNNConfig`/etc., so "did the architecture change between two runs with the same variant
name" is answerable), `train_timestamp`, `cv_metrics` (inline summary, not just a path), `artifact_path`.
This is a strict subset of what `hmm/model_artifacts.py`'s `HMMModelKey` + `.json` sidecar already
capture per-checkpoint — the fastest path to this is generalizing that existing dataclass rather than
designing a new schema from scratch, and it would also retroactively fix `claude_scripts/ml/
model_artifacts.py`'s silent-overwrite problem (§3) as a side effect, since a content-hash mismatch
against the same variant name is precisely what `HMMArtifactConflictError` already catches.

---

## 7. Model type taxonomy

Current inventory, by category (this maps directly onto what exists today, not an aspirational
list):

| Category | What exists | Where |
|---|---|---|
| **Supervised classifiers, trained from scratch per task** | CNN, Transformer, CNN-Transformer hybrid, Bernoulli NB, Random Forest, XGBoost | `analysis/compute/ml_cnn.py` + `ml_metrics.py`, driven by `claude_scripts/ml/b6_cross_dataset_activity.py` |
| **Self-supervised pretrained encoder + fine-tuned classification head** | `MaskedSiteTransformer` (BERT-style masked-site reconstruction) → `MaskedSiteTransformerClassifier` (frozen-encoder linear probe by default) | `claude_scripts/ml/transformer_{models,pretrain,finetune,apply}.py` — **tier 3 only**, no package-side equivalent (this is the strongest promotion candidate — see §2) |
| **Domain-adversarial classifiers** | `DANNTransformerClassifier`/`DANNTransformer` (gradient-reversal for domain-invariance) | `machine_learning/models/transformer.py` — **built, unused** |
| **Unsupervised/statistical sequence models** | Bernoulli HMM (single-state and multi-state, distance-binned variant) | `hmm/HMM.py` — production, unrelated code path |
| **Dimensionality reduction / latent embedding** | PCA, UMAP, diffusion maps (not learned generative models — no VAE) | `analysis/compute/dimensionality_reduction.py`, `tools/calculate_umap.py`/`calculate_leiden.py` (a second, older implementation of overlapping functionality), `latent_model_artifacts.py` |
| **VAEs / autoencoders** | **None exist anywhere in the tree.** No `Encoder`/`Decoder`/`VAE`/`Autoencoder` class, no reparameterization-trick code, no reconstruction-loss-plus-KL training loop, found anywhere in `smftools` (checked `machine_learning/`, `analysis/`, `hmm/`, `tools/`). | — |

If a VAE/autoencoder gets built, `machine_learning/models/base.py`'s `BaseTorchModel` is the more
natural home than `ml_cnn.py` — it already has the attribution/interpretability plumbing a
supervised classifier needs but a generative model mostly doesn't, so a VAE would likely want its
own smaller base class (encode/decode/reparameterize/loss) rather than inheriting
`BaseTorchModel`'s classifier-shaped interface. Worth deciding *before* building one, not after,
since retrofitting a base-class hierarchy onto an existing VAE is more disruptive than designing the
hierarchy to anticipate it.

---

## 8. Prioritized recommendations

Roughly in order of (impact / effort), not necessarily the order to execute them:

1. **Generalize `HMMModelKey`'s content-addressed artifact pattern into a shared
   `smftools`-level utility**, and have `claude_scripts/ml/model_artifacts.py` (and eventually
   `latent_model_artifacts.py`/`project/embedding_store.py`) use it instead of each maintaining an
   independent, weaker version. Fixes the silent-overwrite gap (§3) and gives §6's training-instance
   tracking almost for free, since the artifact key *is* most of a training-run record.
2. **Promote `transformer_models.py` into `smftools` (e.g. `analysis/compute/transformer.py` or a
   new `analysis/compute/sequence_models/` subpackage)** and delete `ml_cnn.py`'s duplicate
   `_sinusoidal_position_table`/`TokenTransformerClassifier` in favor of importing the real thing.
   Zero project dependencies in the source file today, so this is close to a pure move, not a
   rewrite. Do this *before* deciding anything about `machine_learning/`, since it's independently
   worth doing regardless of that larger question.
3. **Decide the `machine_learning/` question explicitly** (finish connecting it, or archive it like
   `tools/archived/classifiers.py`) rather than letting a third partial framework keep silently
   existing. This blocks a real decision on §5.1 (Lightning) and affects where new model classes
   should go.
4. **Add lightweight experiment tracking** (W&B or self-hosted MLflow — see §5.2) to the current
   `ml_cnn.py`-based training loop now; don't wait on #3, since it's additive to the existing
   procedural code either way.
5. **Skip Hydra for now** (§5.3) — revisit only if the variant count keeps growing well past its
   current ~26 and hand-written training scripts become the bottleneck.
6. Smaller cleanups noticed along the way, lowest priority: fix the `lightning`/`pytorch_lightning`
   import-name mismatch in `pyproject.toml`'s extras; delete or clearly mark `requirements.txt` as
   stale (it disagrees with `pyproject.toml` on which ML deps are core vs. optional, and isn't
   referenced anywhere); reconcile `cli/latent_adata.py`'s use of `tools/calculate_umap.py`/
   `calculate_leiden.py` vs. `analysis/compute/dimensionality_reduction.py`'s overlapping
   implementation of the same operations.
