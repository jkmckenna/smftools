# Sequence-context-aware HMM emissions (`HCE`)

**Status:** in progress. `HCE-01`–`HCE-03` merged; `HCE-04` implemented. One PR per item, in order. The
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
| `HCE-04` pipeline integration | implemented, not merged | config, partitioned fit/apply, model artifacts, fingerprint |
| `HCE-05` qualification | proposed | on a panel of several enzymes applied to the same cells |

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
preferences), `none` vs `learned` (and `table` when calibration data exist):

1. Agreement between enzymes -- per-position mean accessibility, footprint
   length distributions, Leiden composition -- should rise with correction.
2. Footprint calls against local context: before correction, footprint
   frequency should rise with the local density of disfavoured contexts;
   after, that dependence should largely vanish.
3. Periodicity (`RPG`): nucleosome periodograms on the corrected accessible
   layer should be sharper (SNR, FWHM).
4. Transfer: weights learned on one allele (or half the reads) applied to the
   other; learned vs naked-DNA weights when available.
5. Dose: the learned shape across doses of one enzyme.

A mode becomes recommended only if 1-3 improve without 4 degrading.

## Out of scope

Naked-DNA calibration data (none yet; the `table` mode and format are ready
for it). Methylation modelling beyond keeping CpG contexts separate. Contexts
for other emission models (multi-channel, distance-binned) until the single
Bernoulli case qualifies.
