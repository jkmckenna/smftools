# HMM fits shared across references (alleles) -- implementation plan

Status: proposed. Items `HXR-01`--`HXR-06`. Touches the same planning code as
the in-progress context-emission work (`HCE`,
`in-progress/hmm_context_emissions_implementation_plan.md`); sequence after or
alongside it (see "Coordination").

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
| `HXR-01` | config: `reference` optional in `hmm_groupby` / `hmm_shared_scope`; `hmm_reference_groups` (name -> references) declares which references may share a fit (default when `reference` is omitted: every reference of the experiment except spike-ins / unconfigured controls, so a spike-in amplicon never pools with the locus) | proposed |
| `HXR-02` | fit planning: `_group_tasks` keys on the reference group instead of the reference (and drops core / load windows from the key when tasks of one group differ only by reference); bounded read selection draws an equal share per reference (`hmm_max_fit_reads` split across the group's references, deterministic hash ranking as now) so the larger allele does not dominate; `fit_id` payload records the pooled references | proposed |
| `HXR-03` | fitting: a fit consumes reads of several references -- each read keeps its own reference's position axis and, with `HCE`, its own reference's per-position context codes; verify `PER`, `GLOBAL` and `ADAPT` fit kinds on mixed-reference batches | proposed |
| `HXR-04` | apply and provenance: every apply task maps to its group's fit; `fit_catalog` / `fit_selection` list the pooled references and per-reference read counts; layers and names unchanged | proposed |
| `HXR-05` | invalidation and docs: the changed `hmm_groupby` is part of the HMM stage's semantic configuration, so stores refit the HMM only (no raw / preprocess dependence); config reference and docs page for allele-comparison setups | proposed |
| `HXR-06` | tests: unit -- grouping keys, equal-share selection, fit ids stable across runs; synthetic -- two references of different lengths with identical simulated chromatin give one fit and matching per-reference state rates, while per-reference fits on unequal read counts reproduce the artefact; acceptance -- on a two-allele deaminase run, HMM accessible tracks raw rates per allele (direction and ratio) after the shared fit | proposed |

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
