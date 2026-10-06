# Modification-site sequence-context bias (`SCB`)

**Status:** proposed. Nothing implemented. One PR per item, in order.

## Question

How much does a site's sequence context shape whether it is modified? For a
channel's site type (C sites for a deaminase, GpC or CpG for conversion),
take the reference window of `2 * flank + 1` bases centred on each site,
oriented to the modified strand, and compare the contexts of *modified*
calls with the contexts of *all observed* calls at the same site type. Which
flanking positions and bases raise or lower modification? How does that
differ between enzymes, doses or samples?

Nothing in smftools does this. `append_base_context` classifies each position
(C / GpC / CpG / other C) but never looks beyond the immediate neighbours.

## Design

A read-only aggregation over existing stores, not a pipeline stage: it writes
tables and figures, never a store, and depends on no stage beyond the one
holding the binarized calls.

**Selection reuses the ML data path.** Molecules come from a datasets-only ML
plan through `bind_ml_dataset` (`MLX-10`): the channel names the layer and site
context, `filters` apply QC/dedup flags, a label table or obs column supplies
groups, `positions` restricts the window, and reads are partition-major and
parallel (`MLX-09`, `MLX-11`, `MRC`). No second selection language.

**Counting is streamed.** For each batch and each (group, physical reference,
position): observed calls (0 or 1 on a design site) and modified calls (1).
Memory is proportional to sites x groups, not molecules.

**Contexts are strand-oriented.** Reference sequences come from the spine's
`References` (forward strand). A top-strand site's context is the forward
window; a bottom-strand site (a G in the forward sequence, as
`append_base_context` defines it) gets the reverse complement, so every
context reads 5'->3' on the modified strand with the site at the centre.
Windows running off a reference end are padded with `N` and counted as such.

**Statistics, per group:**

- *Per site:* observed, modified, rate, context.
- *Per offset and base* (a sequence-logo view of the bias): for each offset
  `-flank..+flank` and base, the base's frequency weighted by modified calls
  and by observed calls; enrichment `log2((m_b/M) / (o_b/O))` with a
  pseudocount. Observed calls are the background: every potential site of the
  type, weighted by how often it was measured.
- *Per k-mer* (`k` odd, `<= 2 * flank + 1`, centred): modification rate with
  Wilson interval, observed and modified calls, and the number of *distinct
  sites* contributing -- on an amplicon most contexts occur at one or two
  positions, so a context resting on one site must show as such.
- *Between groups:* the offset x base enrichment difference of each group
  from a declared reference group (e.g. one enzyme against another): on one
  locus the positional confound largely cancels in such a difference, which
  makes it the most interpretable output.

## Work items

| item | status | scope |
|---|---|---|
| `SCB-01` counting + statistics | proposed | library: stream a bound dataset into per-site counts; per-offset, per-k-mer and between-group tables |
| `SCB-02` figures | proposed | offset x base enrichment heatmaps and logos, k-mer rate plots, group-difference heatmaps |
| `SCB-03` CLI | proposed | `smftools project context-bias` over a plan dataset |
| `SCB-04` qualification | proposed | a real enzyme panel |

### `SCB-01` — counting and statistics

`smftools.analysis.compute.site_context_bias`:

- `count_site_calls(bound, *, group_by=None, workers=1) -> SiteCounts` --
  streams `bound.iter_batches(...)` (split across worker processes as
  `MLX-11` allows), accumulates observed and modified calls per (group,
  physical reference, frame position) on the channel's design sites.
- `site_contexts(counts, references, *, flank) -> pd.DataFrame` -- adds each
  site's strand-oriented context.
- `offset_enrichment(...)`, `kmer_rates(..., k)`, `group_differences(...,
  reference_group)` -- the tables above, pure functions of the site table, so
  re-running with another `flank` or `k` needs no re-read when the site
  counts are kept.

Tests: a hand-built reference and molecules with a planted preference (e.g.
modification only when `+1` is `T`): enrichment is positive exactly there; a
bottom-strand site yields the reverse-complement context; `N` padding at
reference ends; k-mer distinct-site counts; streamed counts equal counts from
one materialized matrix; worker split gives identical counts.

### `SCB-02` — figures

`smftools.analysis.plot.site_context_bias`: offset x base enrichment heatmap
per group (diverging colours, centre site marked); an information-content
style logo of enrichment; k-mer rate dot plot with intervals and distinct-site
counts; group-difference heatmaps. Smoke tests write each figure.

### `SCB-03` — CLI

```
smftools project context-bias PROJECT_DIR --plan PLAN --dataset NAME \
    [--flank 3] [--kmer 1 --kmer 3 --kmer 5] [--group-by COLUMN] \
    [--reference-group VALUE] [--workers N] --output DIR
```

Writes `site_counts.parquet` (the re-usable counts), `sites.parquet`,
`offset_enrichment.csv`, `kmer_rates.csv`, `group_differences.csv`, the
figures, and `run.json` (plan hash, dataset, flank, k, groups, smftools
version). `--group-by` takes a label-table column or an identity column
(`Barcode`, `experiment_id`). An experiment-scope plan works through the same
command with `--experiment-dir`.

Tests: CLI on a fixture project writes every output; a second run with a
different `--flank` reuses `site_counts.parquet`.

### `SCB-04` — qualification

On a real enzyme panel (several deaminases at several doses on one locus):
per-enzyme offset enrichment and differences between enzymes, run time with 1
and 8 workers, and agreement of counts with a direct count from materialized
matrices on one sample.

## Out of scope

Restricting to accessible stretches (HMM): not requested. Project-specific
set definitions, scripts and output organisation live in the project, which
writes the plan and calls the CLI.
