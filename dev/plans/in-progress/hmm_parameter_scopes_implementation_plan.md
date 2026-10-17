# HMM parameter scopes, fitting plans and a project model store

Status: in progress. Items `HPS-01`--`HPS-13`; `HPS-01`--`HPS-05`, `HPS-07`, `HPS-08`, `HPS-11`--`HPS-13` implemented. Builds on `HCE` (context
emissions, `HCE-07` per-state weights), `HXR` (fits shared across
references), `SUG-02` (stage-only refit) and `SLC` (storage lifecycle).

## Why

An accessibility HMM has two kinds of parameters that answer to different
things:

- **transitions** (and start probabilities) describe chromatin -- how long
  accessible and protected stretches last. They differ legitimately between
  cell types, genotypes, alleles and treatments;
- **emissions** describe the measurement -- how often each state is
  modified, and the enzyme's sequence-context preference. They belong to the
  chemistry: enzyme x dose (x batch).

Today both are fitted together per sample x reference (`per_group`), or a
shared fit is followed by per-sample emission adaptation
(`shared_transitions`) -- the opposite of what comparisons need. With free
per-sample emissions, "more modification" can be explained either as more
accessible chromatin or as a more efficient enzyme, and the fit chooses
freely. On a two-allele deaminase run per-allele fits inverted the raw
allele differences in 67-83 % of samples (`HXR-06`); one shared fit fixed
it, but `HXR` also forces the alleles to share transitions, which a
deletion may genuinely change.

Samples treated with the same enzyme and dose are still not modified
equally (permeabilisation, nuclei per reaction, enzyme lot), so fully
shared emissions would read a less efficiently modified sample as closed
chromatin. The middle ground: shared emission shape (state levels, per-state
context weights) per emission group, plus one shrunk per-sample efficiency
offset, with transitions free per sample x reference x strand.

Separately, a project spans many experiments, and comparing emission models
or fitting plans currently means regenerating each experiment's HMM stage;
models live inside each experiment's HMM generation, and variant layers
(`HCE-06`) exist only within one stage run.

## Position axis

The input builder keeps only modification-site columns, so one HMM step is
"from one site to the next" whatever the bp gap: transitions depend on site
density, which differs between references (alleles), regions and targets
(GpC vs all C); bp-level layers come from filling between sites afterwards.
An HMM handles unobserved positions natively (emission likelihood 1), so
fitting on the full per-bp grid -- NaN where there is no site -- makes a
transition a per-bp probability, state durations bp lengths, and decoding a
sparse per-bp input give a posterior at every bp. Cost: the grid is ~4x the
site axis on C-targeting amplicons. This subsumes the distance-binned model
and composes with context emissions and shared emissions.

## Model

For read site *i* of fit group *g* (sample x reference x strand) in state *k*:

    logit P(modified) = theta[e(g), k] + w[e(g), k, c(i)] + delta[g]

- `e(g)`: the emission group (`hmm_emission_groupby`, e.g. enzyme x dose x
  methbase), shared by every fit group in it;
- `theta`, `w`: state levels and per-state context weights (`HCE-07`),
  shrunk and bounded as now;
- `delta[g]`: per-fit-group efficiency offset, Gaussian-shrunk to 0
  (`hmm_efficiency_prior_sd`), optionally anchored to one state's background
  or to a control reference in the same sample;
- transitions and start probabilities: per fit group (`hmm_groupby`).

Scopes:

| `hmm_emission_scope` | emissions | use |
|---|---|---|
| `fit_group` (default, today) | free per fit group | single-sample QC |
| `shared` | per emission group, `delta = 0` | uniform efficiency |
| `shared_efficiency` | per emission group + `delta[g]` | comparisons in general |

Fitting: EM with tied parameters -- each fit group's E-step uses its own
transitions and its emissions (`theta`, `w`, `delta[g]`); the emission M-step
pools sufficient statistics (hits / trials per state x context x fit group)
across the emission group and solves the penalised logistic model on that
small table; transitions update per group. Read selection: equal shares per
fit group within an emission group (as `HXR-02`).

## Items

| id | item | status |
|---|---|---|
| `HPS-01` | config: `hmm_emission_scope` (`fit_group` default \| `shared` \| `shared_efficiency`), `hmm_emission_groupby`, and the technical factors an offset may follow -- `hmm_efficiency_covariates` (numeric sample-sheet columns, log scale, one slope per emission group, e.g. enzyme per cell) and `hmm_efficiency_groupby` (categorical, e.g. batch); `hmm_efficiency_prior_sd`, `hmm_tied_rounds`, `hmm_tied_transition_iters`. Groups come from the existing `sample_sheet_path`. Biological factors are never offsets (see the per-sample offset finding below). Fingerprints and fit ids unchanged at the defaults | implemented |
| `HPS-02` | planner: under a shared scope every fit is per sample (needs `per_group` with `sample` in `hmm_groupby`) and carries its emission group, efficiency key and log covariate values (in its fit id); missing sheet rows / columns and non-positive covariates refused | implemented |
| `HPS-03` | tied EM (`hmm/tied_fit.py`): per member, transitions re-estimated with emissions fixed (the model's own EM, axis-aware); posterior hits / trials per state x context pooled; one penalised binomial fit of state levels, context weights (learned: ridge from `hmm_context_shrinkage`, bounded; table: fixed), covariate slopes (centred within the group) and categorical offsets (Gaussian prior). Simulation: shared levels within 0.25 log-odds, dose slope and per-member transitions recovered | implemented |
| `HPS-04` | artifacts: one tied job per model x emission group publishes an ordinary model per member (shared emissions + its offset + its transitions), so apply, layers, `HXR` pooling and caching are unchanged; the fit catalog records `emission_group` and `efficiency_offset`, artifacts the slopes and any constant covariate | implemented |
| `HPS-05` | named fitting plans: `hmm_variants` entries are full plans -- besides `hmm_context_*` they may set the axis, emission scope / groupings / efficiency covariates, reference groups, fit strategy and grouping, distance-aware transitions, starting probabilities, iteration limits and read selection (layer-naming settings stay global). Every spec is planned, fitted and given input under its own settings; `auto` scope resolves per plan; the fit catalog records each fit's `emission_scope` and `fit_config_hash` | implemented |
| `HPS-06` | project HMM model store: content-addressed models with manifests (plan hash, training selection, source generations, samples and experiments spanned), so one emission model can serve several experiments of the same chemistry | proposed |
| `HPS-07` | layer sets (`cli/hmm_layer_sets.py`): `smftools experiment hmm-plan CONFIG --plan NAME [--list] [--force]` fits and decodes one plan (`default` or an `hmm_variants` entry) from the current spatial spine into `hmm_adata_outputs/layer_sets/<plan>/generations/<id>/` via `staged_generation`; manifest records the plan, per-model `fit_config_hash` and the source spatial generation; unchanged settings and input reuse the current layer set; `--list` flags changed settings / superseded input. Run-root inference, generation listing, the reference graph, prune / archive and project registry discovery (`hmm_layer_set:<plan>` spines) know layer-set containers. Not in v1: a merged multi-plan reader view | implemented |
| `HPS-08` | storage: layer sets are generations under `SLC` (reference graph, prune, archive) -- done with `HPS-07`; posteriors optional (`hmm_write_posterior`); compact feature storage moved to `DLS-02` / `DLS-03` (`derived_layers_storage_implementation_plan.md`) | implemented |
| `HPS-09` | acceptance (below) | direct-fit comparison done; pipeline acceptance after `HPS-01`--`HPS-04` |
| `HPS-10` | docs: choosing scopes for a comparison, config reference | proposed |
| `HPS-11` | position axis: `hmm_axis: sites` (default) \| `bp`. `bp` places the site calls on the window's full per-bp grid (single and multi-channel; NaN elsewhere; coverage masking unchanged), decodes every position, sets `transition_counting = "span"` for fits and adaptations, and scales a new model's configured per-site starting transitions by the observed site spacing (stay `s ** (1 / spacing)`); refused with `hmm_distance_aware`; fingerprints and fit ids unchanged at `sites` | implemented |
| `HPS-12` | genome mode: a sample's genome-mode tasks form one fit group (all chromosomes and tiles) -- one model per sample, fitted on a bounded uniform subset of its reads (hash-ranked as before; each read fitted in the first tile window that owns it) and applied to every tile; no per-region fits for now. Context codes are built per loaded window (`ContextBernoulliHMM` position offset), at fit, pooled fit and apply, so a chromosome is never coded whole; locus references are unchanged (their window is the reference). Shared emission scopes tie the per-sample genome fits as usual | implemented |
| `HPS-13` | tied-fit throughput: an emission group's members built, held and re-estimated by persistent worker processes (one share each), the pooled emission solve in the parent -- identical results; groups run one at a time over all threads; per-round progress log. Found on the 260820 regeneration: 161 members fitted serially in one worker took ~80 min with eleven threads idle | implemented |

### `HPS-07` design: layer sets

**What a layer set is.** The decoded layers of one named fitting plan
(`HPS-05`: the default plan or an `hmm_variants` entry) for one experiment,
published as its own generation -- an HMM generation that holds one plan,
beside the HMM stage rather than inside it. Fitting and decoding a plan never
regenerates the HMM stage, and the HMM stage's current generation is never
touched.

**Layout and lifecycle -- the HMM stage's own machinery.**

    hmm_adata_outputs/
      generations/<id>/            the HMM stage (unchanged)
      layer_sets/<plan>/
        current.json
        retention.json             (pins, as any stage container)
        generations/<id>/          spine.h5ad, store/, models/, fit_catalog,
                                   generation_manifest.json

A layer-set generation is staged and published with `staged_generation`
exactly as the HMM stage is: built under `.staging/`, `current.json` moved
only on success, earlier generations kept. Its manifest records `stage:
hmm_layer_set`, the plan name, the plan's settings and `fit_config_hash`, the
smftools version, and the input it was decoded from.

**Input and lineage.** A layer set is decoded from the experiment's current
spatial generation -- the HMM stage's own input -- so its spine carries the
same pointers into the raw / preprocess / spatial stores as an HMM spine
does. Consequences, all free:

- `materialize(<layer-set spine>)` returns the molecules with the upstream
  layers plus that plan's HMM layers -- readers take a spine path as now;
- the `SLC` reference graph sees the pointers, so a current layer set keeps
  the generations it reads alive, and pruning treats old layer-set
  generations like any other (`generations ... prune`, listing kind
  `hmm_layer_set:<plan>`);
- the manifest names the spatial generation it was built from; when that is
  no longer current, `experiment plan` and the listing report the layer set
  as stale (built on superseded input) rather than silently mixing.

Running a plan while upstream stages are stale is allowed and recorded
("decoded from spatial `<id>`, stale at the time") -- this is `SUG-02`'s
stage-only refit, for the HMM.

**Command.**

    smftools experiment hmm-plan CONFIG --plan NAME [--list]

- `NAME` is `default` or an `hmm_variants` key of the config; the run fits
  and decodes only that plan (the stage executor restricted to one spec), so
  a plan can be added to the config and run without touching the HMM stage;
- `--list` shows each plan's current layer set, its `fit_config_hash`, the
  spatial generation it was decoded from and whether that is still current.

Layer names are the plan's (`C_<variant>_...`, or `C_...` for `default`), so
a layer set and the HMM stage can be read side by side and joined on
`molecule_uid` / position without renaming.

**Projects.** The project registry gains, per experiment, the current layer
sets (`spines["hmm_layer_set:<plan>"]`), refreshed with the other spines; a
project analysis names the plan it reads, e.g. a latent set channel
`("hmm", layer, plan)`. Registry pointers protect the layer sets they name
(`--protect-from`).

**Storage.** A layer set costs roughly one HMM generation (layers for every
molecule). The plan's own `hmm_write_posterior` decides whether the float
posterior is written; `SLC` prunes superseded layer-set generations.

**Not in v1.** A merged view (one `materialize` call returning the HMM stage's
layers and several layer sets' layers) -- callers join on `molecule_uid`;
making the HMM stage itself a layer set of the default plan; a project-wide
emission model across experiments (`HPS-06`).

**Tests.** A layer set of a variant publishes beside an unchanged HMM stage
(its current generation id and files untouched); its spine materializes the
plan's layers with the upstream layers; a second run with the same settings
reuses the models (fit ids); a changed plan publishes a new generation; the
reference graph protects the spatial generation a current layer set reads;
`--list` reports a layer set built on a superseded spatial generation as
stale; an unknown plan name is refused.

## Acceptance (`HPS-09`)

Direct fits, same reads and settings, learned per-state context emissions,
arms `fit_group` / emissions shared per sample (across its references) /
`shared` per emission group / `shared_efficiency`:

1. **Same cells, several enzymes and doses** (a six-enzyme x three-dose panel):
   chromatin is identical, so accessibility should agree across doses of one
   enzyme and across enzymes; the efficiency offsets should rise with dose.
2. **Two alleles in the same cells, one enzyme and dose, two polyamine
   conditions:** HMM allele ratios track raw ratios (`HXR-06` metric) while
   alleles keep separate transitions.

`shared_efficiency` should match or beat `fit_group` on both; `shared`
without offsets is expected to fail where efficiency differs. The same arms
on the `bp` axis (`HPS-11`) are compared on footprint lengths in bp and on
the same agreement metrics.

### Results (direct fits; tied EM; learned per-state context weights)

Same cells, six enzymes x three doses (B6 allele; the other allele alike):

| arm | between-enzyme r | between-dose r | accessible spread across doses | protected median (bp) | 120-200 bp share | < 50 bp share |
|---|---|---|---|---|---|---|
| `fit_group`, sites | 0.65 | 0.90 | 0.024 | 110 | 0.34 | 0.25 |
| `shared`, sites | 0.66 | 0.90 | **0.053** | 111 | 0.34 | 0.27 |
| `shared_efficiency`, sites | 0.65 | 0.90 | 0.031 | 109 | 0.34 | 0.27 |
| `fit_group`, bp | **0.68** | 0.91 | 0.030 | 117 | **0.37** | **0.22** |
| `shared_efficiency`, bp | **0.67** | 0.91 | 0.038 | 118 | **0.37** | **0.23** |

Efficiency offsets rise with dose for five of six enzymes without being told
the dose (one enzyme's run backwards -- a sample to check).

Two alleles in the same cells (30 samples), enhancer-deletion / intact,
outside the deletion:

| arm | HMM ratio direction = raw | median abs log2 gap | protected median intact / deletion (bp) | < 50 bp share |
|---|---|---|---|---|
| `fit_group`, sites | **33 %** | 0.20 | 72 / 51 | 0.40-0.50 |
| per sample (alleles share), sites | 87 % | 0.11 | 65 / 58 | 0.42-0.48 |
| `shared`, sites | 93 % | 0.09 | 70 / 55 | 0.42-0.49 |
| `shared_efficiency`, sites | 87 % | 0.10 | 63 / 57 | 0.44-0.50 |
| `fit_group`, bp | **33 %** | 0.22 | 124 / 114 | 0.20-0.24 |
| `shared_efficiency`, bp | 83 % | 0.18 | 124 / 135 | 0.21-0.24 |

Polyamine condition (0.5 / 0 mM, same mouse, cell type and allele): raw 0.62;
HMM 0.75 (`fit_group`), **0.44** (`shared`), 0.70 (`shared_efficiency`);
offsets +0.56 / -0.20.

- Free emissions per fit group invert allele comparisons on either axis;
  emission sharing fixes that, the axis does not.
- Shared emissions without offsets turn efficiency differences (dose,
  polyamine) into chromatin; offsets restore within-enzyme stability.
- The bp axis gives physical footprints (fewer sub-50 bp fragments, more
  nucleosome-sized stretches; on a site-sparse deaminase run the site axis
  halves footprint lengths) and better agreement between enzymes; it widens
  HMM allele ratios relative to raw and slightly loosens dose stability.
- On a per-bp grid EM must count every step inside the read
  (`transition_counting = "span"`, implemented) and start from per-bp-scaled
  transitions: counting only observed adjacent pairs overestimates protected
  persistence (~20 % on a 1-site-per-4-bp simulation), and from 0.5 / 0.5 EM
  barely moves.

Recommended target: `shared_efficiency` on the bp axis.

### Efficiency offsets must not follow biological factors

Fitted per sample on the two-allele run (one enzyme, one dose, same cell
input everywhere), the offsets tracked cell type: active NK +1.09 / +0.77,
inactive NK +0.85 / -0.07, CD4 T +0.21 / -0.55, B +0.07 / -0.77 (two polyamine
conditions), with replicates agreeing to ~0.1-0.2 -- the ordering of the
locus's activity, estimated from the locus itself. An offset free across
cell types absorbs the chromatin differences under study. Offsets therefore
follow only declared technical factors; with a constant reaction (that run)
there is no offset and `shared_efficiency` reduces to `shared`. A per-reaction
control (naked DNA, plasmid or a silent locus at real coverage) would allow
measuring efficiency across cell types (`hmm_efficiency_anchor`, future).

### Defaults

- `hmm_axis: bp` (`hmm_distance_aware` implies `sites`, with a warning).
- Deaminase configs: `hmm_context_model: learned`, `hmm_context_states: all`
  (`deaminase.yaml`); other modalities keep `none` until qualified. An
  explicit CSV `hmm_context_model,none` overrides the modality default.
- `hmm_emission_scope: auto`, `hmm_emission_groupby: [enzyme]`,
  `hmm_efficiency_covariates: [enzyme_per_cell]`: `shared_efficiency` when
  the sample sheet gives every sample those columns, else `fit_group` with a
  warning; the fit catalog records the effective scope. Explicit scopes stay
  strict.
- Stores fitted before these defaults fingerprint as the legacy `sites` /
  no context / `fit_group` setup, so their HMM stage (and later stages) is
  stale under the new defaults; raw, preprocess and spatial are not.

