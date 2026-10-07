# Sequence-context-aware HMM emissions (`HCE`)

**Status:** in progress. `HCE-01`–`HCE-04` merged; `HCE-05` qualified (not adopted as the default); `HCE-06`, `HCE-08`, `HCE-09` merged; `HCE-07` proposed. One PR per item, in order. The
default stays `none` unless `HCE-05` qualifies a mode.

## Question

The accessibility HMM gives each state one modification probability, shared
by every site: an accessible C is assumed equally likely to be modified
whatever its neighbours. Context-bias analyses (`SCB`) show it is not:
deaminases differ several-fold by flanking base (e.g. a G at +1 or -1 cutting
the rate 4-6x for some enzymes). An accessible site in a disfavoured context
that was not modified looks like protection, so the HMM is expected to call
spurious short footprints, and fragment accessible patches, where disfavoured
contexts cluster. Can position-specific emissions -- one probability per
(state, sequence context) -- remove that?

## Design

**Emissions by context.** Each design site gets a context index: its
strand-oriented, centred k-mer (default k = 3: N-C-N, 16 contexts for C sites;
the same orientation as `SCB`), from the reference sequence. The emission
becomes `p(state, context)`.

**Relative weights, fitted level.** A context's effect is a relative weight
`w(context)` -- its modification rate over the overall rate -- applied on the
log-odds scale, so probabilities stay in (0, 1):

    logit p(state, context) = logit p(state) + log w(context)

`p(state)` is fitted by EM per sample as today; the weights carry only the
shape of the enzyme's preference. Absolute per-context rates are not used:
they mix preference with the sample's accessibility and with where each
context happens to sit on the locus, and would double-count accessibility.
Weights apply to the modified (accessible) state by default; optionally to
every state (`hmm_context_states`).

**Where weights come from** (`hmm_context_model`):

| mode | weights | use |
|---|---|---|
| `none` | 1 | current behaviour, the default |
| `table` | read from a table: per group (e.g. enzyme), k-mer -> weight | weights measured where every site is accessible (naked DNA treated with the enzyme), or exported from a qualified `learned` fit |
| `learned` | estimated inside EM: per (state, context) emission from sites weighted by their state posteriors, shrunk toward the state's overall rate | no calibration data needed |

**Why not weights from the sample's raw calls.** In cells a k-mer's rate is
confounded with chromatin and methylation: on one amplicon most contexts sit
at a handful of sites, so a context at open sites looks favoured and a
correction from it would erase real accessibility; and methylated CpG resists
deamination, so CpG would be "corrected" away as a disfavoured context.
`learned` conditions on the state posteriors instead, which removes the
accessibility confound; `table` from naked DNA removes both.

**CpG** (`hmm_context_cpg`): `separate` (default for `learned`: CpG contexts
get their own emissions, never pooled with non-CpG, and are reported so a
methylation effect is visible rather than absorbed), `exclude` (CpG sites are
not HMM input), or `none` (treated like any context).

**Shrinkage.** A learned context emission is
`(sum_g gamma_k * obs + m * p_k) / (sum_g gamma_k + m)`: `m` pseudo-observations
(`hmm_context_shrinkage`, default 50) at the state's overall rate, so a rare
context cannot run away. Weights are bounded (`hmm_context_weight_bounds`,
default 0.1-10).

## Configuration (sketch)

```yaml
hmm_context_model: none          # none | table | learned
hmm_context_k: 3                 # odd; centred k-mer
hmm_context_states: [modified]   # modified (the higher-emission state) | all
hmm_context_table: null          # path; mode table
hmm_context_table_group: enzyme  # sample-sheet column selecting the table's group
hmm_context_cpg: separate        # separate | exclude | none
hmm_context_shrinkage: 50
hmm_context_weight_bounds: [0.1, 10]
```

## Work items

| item | status | scope |
|---|---|---|
| `HCE-01` context indices and weight tables | merged | per-position context index for a reference and strand; the weight-table format; `context-bias` exports it |
| `HCE-02` context emissions, `table` mode | merged | a context-aware Bernoulli emission with fixed weights; EM fits the per-state level |
| `HCE-03` `learned` mode | merged | per-(state, context) emissions in the M-step with shrinkage; CpG handling; the fitted weights saved as a table |
| `HCE-04` pipeline integration | merged | config, partitioned fit/apply, model artifacts, fingerprint |
| `HCE-05` qualification | qualified: default stays `none`, `learned` opt-in | on a panel of several enzymes applied to the same cells |
| `HCE-06` HMM variants | merged | several emission models in one HMM stage: namespaced layers, plots comparing them |
| `HCE-07` context weights for every state (learned) | proposed | the protected state's background modification follows the enzyme's preference too |
| `HCE-08` HMM vs raw per-molecule scatter | merged | per read: HMM accessible fraction against the raw modified-site fraction, per barcode, variants overlaid |
| `HCE-09` HMM fractions at observed sites | merged | per read: the share of the model's observed sites inside each feature, beside the raw modified-site fraction |
| `HCE-10` per-read fraction backfill | implemented, not merged | `HCE-06`/`-08`/`-09` per-read fractions and figures for HMM generations made before them |

### `HCE-01` — context indices and weight tables

`analysis.compute.site_context_bias` gains `context_index(sequence, strand,
positions, k)` -> per-position context codes (strand-oriented as
`site_contexts`; `N`-containing windows get an "ambiguous" code that is never
weighted) and a CpG flag. Weight-table format (parquet/CSV): `group`, `k`,
`kmer`, `weight` (relative rate), `n_sites`, `observed`, `source`
(`naked_dna` | `learned` | `cells`); `context-bias --export-weights` writes it
from `kmer_rates` (`log2_relative_rate`), marking cell-derived tables
`cells` so their caveat travels with them.

Tests: codes match `site_contexts`; bottom strand reverse-complemented;
ambiguous windows; table round trip; export from a `context-bias` run.

As built: `strand_window` is the one window reader for both; codes follow
`context_kmers(k)` (C-centred, 4^(k-1)); positions that are not a C on the
modified strand, or whose window touches `N`, get `NOT_A_CONTEXT` (-1).
Weights are smoothed -- `(modified + 0.5) / (observed + 1)` over the group's
overall rate -- so a context never modified in the data is unlikely rather
than impossible (a weight of 0 would forbid modification in the HMM).
`context-bias --export-weights [--weights-source cells|naked_dna]` writes
`context_weights_k<k>.parquet` for each k >= 3; `weights_for(table, group, k)`
returns weights in code order (1 for an absent k-mer).

### `HCE-02` — context emissions, `table` mode

`ContextBernoulliHMM(SingleBernoulliHMM)`: `_log_emission` gathers
`logit^-1(logit p_k + s_k log w[c_i])` by the per-position context `c_i`
(`s_k` = 1 for weighted states, 0 otherwise). The M-step for `p_k` solves the
weighted level by a few Newton steps on the log-odds (closed form only when
all weights are 1, which reduces to today's update). Save/load carries the
weights and context map.

Tests: all-ones weights reproduce `SingleBernoulliHMM` exactly (fit and
posteriors); on simulated reads with a planted context preference and known
states, posteriors with the true weights recover the states better than
without (fewer spurious protected calls at disfavoured contexts).

As built: `ContextBernoulliHMM` (registered `context_single`) takes
`log_weights` per context code and `position_codes` per reference position
(`set_contexts`); columns map to codes through the coordinates of the fit or
decode call. `SingleBernoulliHMM.fit_em`'s emission update moved to
`_emission_m_step` (closed form, unchanged); the context model solves
weighted states by Newton steps on the log-odds. With every weight 1 it
delegates to the plain emission, so fits and posteriors are identical.
Weights act on the odds, not the rate: a table holds rate ratios, which match
odds ratios for modest rates and keep probabilities in (0, 1) where they
diverge. On a simulation (60 reads, 900 bp, half the contexts at 0.2x, half
at 1.6x) sites called correctly rose 95.8% -> 96.4%, accessible sites in
disfavoured contexts 93.8% -> 95.2%: long, site-dense blocks are largely
recovered by the HMM's smoothing already; short patches are `HCE-05`'s to
measure.

### `HCE-03` — `learned` mode

EM accumulates `gamma * obs` and `gamma` per (state, context) with
`scatter_add` (the per-state sums today, split by context), then the shrunk
update above; CpG contexts per `hmm_context_cpg`. Works with both fit
strategies (`per_group`; `shared_transitions` learns contexts per group with
shared transitions). The fitted per-context emissions, divided by the state's
level, are saved as a weight table (`source: learned`), so a qualified fit can
be reused as `table` elsewhere.

Tests: on simulated data the learned weights converge to the planted ones;
shrinkage holds a context with few sites at the state rate; with no context
effect the fit matches `none`; CpG `separate` keeps CpG apart.

As built: `ContextBernoulliHMM(learn=True, shrinkage=m, weight_bounds=...,
cpg_codes=..., cpg=...)`, modified state only. Each EM iteration: closed-form
state levels, then per-context emissions of the modified state shrunk toward
its level, stored as bounded log weights on the odds. Learned weights are
relative to the state's overall rate, which already averages over contexts,
so they are identified up to a constant the level absorbs: compare centred
weights (a simulation recovers the centred planted weights within 0.2 on the
log scale, correlation > 0.95). `cpg="exclude"` drops CpG sites from the
model's input; `separate`/`none` learn them as any context, flagged `cpg` in
`weight_table(group, k)` (the `HCE-01` format, source `learned`).

### `HCE-04` — pipeline integration

Config keys above; the partitioned HMM fit and apply compute context indices
per reference and strand from the spine's references (as `context-bias`) and
pass them with the observations; model artifacts carry mode, k, weights,
table group; the HMM stage fingerprint includes mode, parameters and the
table's content hash (a changed table refits, cf. `F72`). Missing table
group for a sample: an error, unless `hmm_context_table_group` falls back to
`none` explicitly.

Tests: a fixture HMM run in each mode; changing the table changes the
fingerprint; `none` leaves existing outputs and fingerprints unchanged.

As built: a single-channel model becomes `context_single` when
`hmm_context_model` is not `none` (refused with `hmm_distance_aware`). Each fit
task computes `context_setup` from its materialized reads -- the reference's
forward sequence (`sequences_from_uns`, padding removed), strand from the
reference name, codes over every reference position, CpG codes, and either
unit weights (learned) or the table weights of the one `hmm_context_table_group`
value the fit's reads share -- and hands it to `HMMTrainer.context_setup`,
which sets it on new models and on adapted copies (keeping a learned shared
fit's weights as the start). Weights, position codes and CpG codes are
module buffers, so trainer checkpoints carry them; the context settings ride
in the checkpoint's override, so `create_hmm` rebuilds the model as fitted.
Fingerprints: every `hmm_context_*` key is dropped from the stage config and
the fit-config hash while the model is `none` -- existing stages and models
keep their hashes -- and with `table` the table's content hash joins both.

### `HCE-05` — qualification

On cells treated with several enzymes (same chromatin, different
preferences): one enzyme panel, 18 samples (six deaminase preparations x three
doses), each of two alleles, 8,291 QC/dedup-passing molecules. Every arm fits
and decodes the same reads with the experiment's own HMM settings (the
stage's input preparation, `context_setup`, `create_hmm`; 1,000-read fits),
differing only in emissions: `none`, `learned` (0.1-10x), a table from
`context-bias` on the same cells (negative control), and two learned bound
variants. Pre-re-extraction stores: valid between arms on identical reads,
absolute values will move. The enzyme with ~4 reads per dose is left out of
between-enzyme summaries.

| arm (allele 1 / allele 2) | between-enzyme r (worst pair) | footprint-length JS | context gap | per-read SNR |
|---|---|---|---|---|
| `none` | 0.597 / 0.629 (0.34 / 0.43) | 0.199 / 0.184 | 0.063 / 0.059 | 13.1 / 11.2 |
| `learned` | 0.626 / 0.654 (0.50 / 0.55) | 0.302 / 0.290 | 0.018 / 0.016 | 14.8 / 13.3 |
| cell table | 0.620 / 0.647 (0.45 / 0.50) | 0.312 / 0.296 | 0.035 / 0.033 | 14.7 / 13.4 |
| `learned`, 0.25-2x | 0.611 / 0.641 (0.47 / 0.52) | 0.353 / 0.328 | 0.017 / 0.015 | 14.6 / 13.7 |
| `learned`, <= 1x | 0.275 / 0.352 (-0.24 / -0.15) | 0.462 / 0.436 | 0.014 / 0.017 | 14.0 / 12.4 |

Learned weights transfer between alleles (median centred r 0.98) and doses
(0.98-0.99). Diagnostics on the same fits:

- Residual context bias of accessible calls (RMS log2 over C-centred 3-mers):
  raw modification 0.93 / 0.76, `none` 0.32 / 0.28, `learned` 0.15 / 0.17,
  cell table 0.18 / 0.18. Part of what `learned` flattens is xCG -- likely
  CpG methylation resisting deamination in cells, which `cpg: separate`
  absorbs into the weights.
- Footprint-length classes (share of sites): with `learned`, large protected
  stretches (200+ bp) fall from 25-27% to 18% and accessible sites rise from
  26% to 31-32%; nucleosome-sized footprints barely change (39-41% -> 38%).
  In clustermaps, `none`'s long stretches split into nucleosome-sized blocks
  separated by short accessible gaps: adjacent nucleosomes merged across
  linkers whose few C sites sit in contexts the enzyme modifies poorly.
- Periodicity: every input peaks at 189 bp, same width (~8 bp). The plain HMM
  often lowers per-read SNR below the raw calls; `learned` raises it, most for
  the enzyme with the strongest preference (+2.5 SNR, above the raw calls).

The lower between-enzyme footprint-length agreement under `learned` looks
like the loss of a shared artefact (every enzyme's `none` merges nucleosomes
alike), not a defect -- an inference, without ground truth for footprint
lengths. The one-sided bound is ill-posed as built: weights are relative to
the average context, so a cap of 1 under-predicts every favoured context.

**Decision:** the default stays `none`; `learned` is an opt-in (`HCE-06`
lets a project carry it alongside). Open before reconsidering the default:
a `learned` + `cpg: exclude` arm; the re-extracted stores; naked DNA, where
no site is protected and merged-nucleosome artefacts cannot arise.

### `HCE-06` — HMM variants

Several emission configurations in one HMM stage run, so downstream analyses
can choose any of them:

```yaml
hmm_context_model: none   # the default variant: layer names and hashes unchanged
hmm_variants:
  learned: {hmm_context_model: learned}
  cells:   {hmm_context_model: table, hmm_context_table: <path>, hmm_context_table_group: enzyme}
```

- Each variant overrides `hmm_context_*` settings; every variant fits and
  decodes the same reads in the same run (same generation).
- Layers: the default variant keeps today's names; each other variant writes
  namespaced layers, `<label>_<variant>_<feature>` (e.g.
  `C_learned_all_accessible_features`, `..._lengths`, merged layers). The
  stage catalog lists them, so ML plans, latent and periodicity sets select a
  variant by layer name.
- Model artifacts keyed by variant; the stage fingerprint and fit-config hash
  carry the variants (none configured: unchanged).
- Plots compare the variants in the existing figures rather than repeating
  them: clustermaps gain columns for each variant's feature layers, one read
  order across all columns; feature count and size histograms overlay one
  colour per variant, translucent fills with solid outlines so overlapping
  distributions stay distinguishable, the legend naming each variant.
- Not a default: a project turns variants on in its configs (best with a
  re-run it needs anyway); runtime and layer storage grow per variant.

- Per molecule, the fraction of the read's own span in each feature group
  (all-accessible, all-footprint) per variant, stored as `<layer>_fraction`
  in the stage's read table, and plotted as preprocess plots per-read
  modification rates: per reference window a panel per feature, at each
  barcode the variants' violins side by side, each colour translucent with a
  solid edge and median over the reads' jittered values.

Tests: no variants -> identical layers, artifacts and hashes; two variants ->
both layer sets, artifacts per variant, catalog lists both; clustermap columns
and histogram overlays per variant.

As built: `hmm_variants` (`ExperimentConfig`; a mapping, or JSON/YAML text in
CSV configs) may set only `hmm_context_*` keys; names must be identifiers and
may not be the leading word of a feature layer (`all`, `merged`, the
configured feature names), so `<label>_<variant>_<feature>` stays
unambiguous. `HMMModelSpec` carries `variant`, `base_label` and `overrides`;
`spec.config(cfg)` applies them. The default spec, its fit-config hash and
its fit ids are unchanged by adding variants (its models are reused);
multi-channel specs get no variants. `variant_layer_groups` pairs each
variant layer with its default layer for the clustermaps
(`extra_hmm_layers` / `extra_length_layers` columns), the count and size
histograms (shared bins, translucent fills, solid step outlines, legend
beside the panels) and the per-molecule fractions. An empty `hmm_variants`
is absent from the stage fingerprint; variants enter it, with any variant
weight table by content. Fractions are computed over each task's core window,
so a read split across genome chunks is not double counted; accessible and
footprint fractions need not sum to 1 (feature intervals are filled across
gaps between sites and can meet at their edges).

### `HCE-07` — context weights for every state (learned)

Learned mode weights the modified state only; the protected state's
background modification follows the same enzyme chemistry. Learn a weight
vector per state (shrunk, bounded), so a stray modification inside a
nucleosome at a favoured context is not read as accessibility. Qualify as
`HCE-05`, against `learned`.

## Out of scope

Naked-DNA calibration data (none yet; the `table` mode and format are ready
for it). Methylation modelling beyond keeping CpG contexts separate. Contexts
for other emission models (multi-channel, distance-binned) until the single
Bernoulli case qualifies.

### `HCE-08` — HMM vs raw per-molecule scatter

How far does each HMM (variant) move a molecule away from its raw signal? In
the same core window the stage already reports each read's accessible and
footprint fractions (`HCE-06`, `<layer>_fraction`). Beside them it stores the
read's raw modified-site fraction at the model's own sites --
`<model>_site_modified_fraction`, modified / observed calls of the HMM input
(one per model; variants share their input) -- and plots, per reference
window, a grid with one panel per barcode: x the raw fraction, y the HMM
accessible fraction, every variant overlaid in its colour (translucent
points, at most 2,000 reads per variant and panel), the identity line, and
each variant's Pearson r in the legend.

Tests: the raw fraction equals modified / observed of the model input over
the core; one figure per reference window, a panel per barcode, every variant
drawn; no `_fraction` columns -> no figure.

### `HCE-09` — HMM fractions at observed sites

`<layer>_fraction` (`HCE-06`) counts every position of a read's span, the
gaps between sites included; the raw modified-site fraction (`HCE-08`)
counts observed sites. `<layer>_site_fraction` -- for each accessible and
footprint layer of every variant -- is the share of the read's observed
model sites (in the core) inside the feature: the HMM's call at exactly the
sites the raw fraction counts, so the two differ only in the call (as the
periodicity inputs `accessible` vs `accessible_all`). The per-molecule violin
figure gains a panel per feature at sites; the accessible one carries the
raw modified-site fraction as a grey violin beside the variants.

Tests: equals observed-and-in-feature / observed from the model input and
the stored layer; raw drawn only in the accessible-at-sites panel.

### `HCE-10` — per-read fraction backfill

HMM generations made before `HCE-06` have none of the per-read fractions, so
neither the per-molecule violins nor the HMM-vs-raw scatters can be drawn
from them. `smftools experiment|project context-qc --stage hmm-fractions`
re-materializes each task's reads with their decoded layers (as the `SCQ-03`
HMM backfill does) and computes what the stage now stores per read --
`<layer>_fraction`, `<layer>_site_fraction`, `<model>_site_modified_fraction`
-- into `<generation>/molecule_fractions/molecule_fractions.parquet` (keyed by
read and task), then draws the violin and scatter figures from it into
`molecule_fractions/plots/features/` (the plot functions take an
`obs_reader`). The generation's read table, `plots/` and manifests are not
changed. Part of the default `context-qc` stages.

Tests: every backfilled column equals the stage-written one on a fixture with
a variant (span fractions included: re-materialized layers keep NaN outside
a read); figures written; the generation's plots unchanged; existing output
kept without `--refresh`.

