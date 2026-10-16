# HMM parameter scopes, fitting plans and a project model store

Status: in progress. Items `HPS-01`--`HPS-11`; `HPS-01`--`HPS-04`, `HPS-11` implemented. Builds on `HCE` (context
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
| `HPS-05` | named fitting plans: a versioned spec (architecture, emission / transition scopes, context model, reference groups, read selection) with a hash; config `hmm_plans: {name: spec}` generalising `hmm_variants` | proposed |
| `HPS-06` | project HMM model store: content-addressed models with manifests (plan hash, training selection, source generations, samples and experiments spanned), so one emission model can serve several experiments of the same chemistry | proposed |
| `HPS-07` | layer sets: decoding a plan writes a per-experiment layer-set generation (lineage: model ids, the spatial / preprocess generations read) beside the HMM stage; readers select `(layer, plan)`; fitting a new plan does not regenerate the HMM stage (`SUG-02`) | proposed |
| `HPS-08` | storage: layer sets are generations under `SLC` (reference graph, prune, archive); posteriors optional, features stored compactly | proposed |
| `HPS-09` | acceptance (below) | direct-fit comparison done; pipeline acceptance after `HPS-01`--`HPS-04` |
| `HPS-10` | docs: choosing scopes for a comparison, config reference | proposed |
| `HPS-11` | position axis: `hmm_axis: sites` (default) \| `bp`. `bp` places the site calls on the window's full per-bp grid (single and multi-channel; NaN elsewhere; coverage masking unchanged), decodes every position, sets `transition_counting = "span"` for fits and adaptations, and scales a new model's configured per-site starting transitions by the observed site spacing (stay `s ** (1 / spacing)`); refused with `hmm_distance_aware`; fingerprints and fit ids unchanged at `sites` | implemented |

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

