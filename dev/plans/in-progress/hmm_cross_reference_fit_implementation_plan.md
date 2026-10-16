# HMM fits shared across references (alleles) -- implementation plan

Status: implemented and accepted on real data (`HXR-01`--`HXR-06`); stays
in progress until a store has been regenerated with it. Built on the merged context-emission work (`HCE`,
`in-progress/hmm_context_emissions_implementation_plan.md`, through
`HCE-07`).

## Why

Partitioned HMM fitting groups reads by `hmm_groupby` (default `sample`,
`reference`, `methbase`), and `reference` is mandatory
(`REQUIRED_HMM_GROUP_FIELDS` in `src/smftools/hmm/fit_plan.py`). When one
sample carries two alleles of a locus as two references -- e.g. an intact
allele and an enhancer-deletion allele of the same locus -- each allele gets
its own HMM parameters, and allele comparisons of HMM states then carry the
difference between two fits, not between two chromatins.

Measured on a deaminase run with intact and enhancer-deletion alleles in the
same cells: raw deamination is equal or lower on the deleted allele (promoter
0.214 vs 0.232; intervening region 0.221 vs 0.237), while HMM accessibility
is far higher (0.275 vs 0.168; 0.272 vs 0.171). Downstream analyses that
compare alleles on HMM layers (NDR calls, accessible / footprint size
classes, HMM-based periodicity) are unusable until the two alleles share
a fit. `hmm_fit_strategy: shared_transitions` does not help: its shared
scope must also include `reference`.

## Goal

One fit per sample (and methbase) across a declared group of references,
applied to every read of each reference; per-reference behaviour unchanged
when not requested.

## Items

| id | item | status |
|---|---|---|
| `HXR-01` | config: `hmm_reference_groups` (name -> references) declares which references share a fit; a member is a reference or a base name (every strand; strands fit apart); unlisted references fit alone. `reference` stays required in `hmm_groupby` / `hmm_shared_scope` and now means "reference group" -- explicit groups instead of the proposed default-pool-everything, so a spike-in can never pool by omission | implemented |
| `HXR-02` | fit planning: `_group_tasks` keys a grouped reference on its group and strand (windows dropped; each member must have one analysis core, else an error); equal shares per reference (`hmm_max_fit_reads` and the memory ceiling split between the references present; a short reference gives all it has, the rest is not redistributed); a pooled fit's `reference` is its members joined with `+` and its id records each member's window. Ungrouped fits and their ids are unchanged; a group with one reference present for a sample is an ordinary fit | implemented |
| `HXR-03` | fitting: members are materialized on their own references and stacked left-aligned (NaN padding; transitions and levels are position-independent). `ContextBernoulliHMM` takes one coordinate row per read (N, L) into the members' concatenated context codes, so each read keeps its reference's contexts and the fit learns one set of (per-state) weights. `PER`, `GLOBAL` and `ADAPT` all pool; not with `hmm_distance_aware` (refused at planning) | implemented |
| `HXR-04` | apply and provenance: every member's tasks map to the pooled fit; a context model is given the task's own reference codes before decoding (shared weights kept); `fit_catalog` lists `references` and `members` (windows, candidate / selected counts), `fit_selection` each read's `read_reference`, checkpoint training metadata `reference_n_reads`; layers and names unchanged | implemented |
| `HXR-05` | invalidation and docs: `hmm_reference_groups` is omitted from stage fingerprints while empty (existing stores stay current) and, when set, makes the HMM stage and later ones stale, not preprocess or spatial; the experiment-config tutorial documents allele setups | implemented |
| `HXR-06` | tests: unit -- grouping keys, strands, equal shares, stable ids (and unchanged ids for ungrouped fits), shared-transition pooling, refusals; model -- per-read coordinates equal shared ones exactly; synthetic -- two references of different lengths and sequences, identical chromatin and enzyme, unequal reads: the pooled fit recovers both preferences and the two references carry the same calling bias (difference 0.002), while separate fits differ (0.032); end-to-end -- one fit, both alleles decoded, plain and learned-every-state models; acceptance -- on a two-allele deaminase run, HMM accessible tracks raw rates per allele (direction and ratio) after the shared fit | done (below) |

## Coordination

`HCE` (in progress) changes `fit_plan.py`, model specs and per-reference
context setup. `HXR-02` / `HXR-03` touch the same code: implement on top of
the merged `HCE` items, or agree the order with that work before starting.
`HXR-03` must keep per-read context codes per reference (a shared fit learns
one set of context weights from both references' contexts).

## Not in scope

- Raw re-extraction or preprocessing changes.
- Choosing which references to pool for a given project: that is project
  configuration (`hmm_reference_groups`).
- A stage-only rerun mode (rerunning `hmm` / `spatial` without re-checking
  raw when raw is stale): a separate concern, noted here because a
  shared-fit refit is most useful exactly when upstream stages must not move.

## Acceptance (`HXR-06`)

A deaminase run with intact and enhancer-deletion alleles in the same cells
(30 samples: active and inactive NK cells, CD4 T and B cells, two
polyamine doses). Each sample was fitted directly with the run's HMM
settings, four ways: plain or learned every-state emissions (`HCE-07`), one
fit per allele or one shared fit through this planner (equal reads per
allele). Enhancer-deletion / intact ratio, outside the deletion, median over
samples:

| cells | raw rate | per-allele fit | shared fit | shared fit, learned every state |
|---|---|---|---|---|
| active NK | 0.57-0.67 | 0.85-0.92 | 0.28-0.50 | 0.31-0.58 |
| inactive NK | 0.91-0.92 | 1.16-1.52 | 0.88 | 0.90-0.92 |
| CD4 T | 0.96 | 1.10-1.23 | 0.94 | 0.95-0.96 |
| B | 0.99-1.00 | 0.95-0.96 | 0.99-1.00 | 0.99-1.00 |

- The HMM ratio's direction matches the raw one in 33 % (plain) / 17 %
  (learned) of samples with per-allele fits and in 100 % / 93 % with a
  shared fit (the misses: B cells, ratio ~1).
- Median |log2(HMM ratio / raw ratio)|: 0.33 per allele, 0.035 shared
  (plain), 0.012 shared (learned every state).
- Per-allele fits nearly erased the deletion's effect in active NK cells and
  invented a gain in the other cells; a shared fit keeps the raw difference
  and, as a two-state call, sharpens it.

