# Sequence-context QC in the preprocess and HMM stages (`SCQ`)

**Status:** proposed. Nothing implemented. One PR per item, in order.
`SCQ-02` follows `HCE-06` (HMM variants).

## Why

`context-bias` (`SCB`) answers how a site's sequence context shapes
modification, but only when someone writes a plan and runs it. Run as part of
every stage, the same analysis becomes routine QC per barcode and reference --
an enzyme with an unexpected preference, a strand flip, a reference mismatch,
CpG methylation showing through -- and, on HMM calls, a check that each HMM
reads chromatin rather than sequence (`HCE-05`: the plain HMM leaves a
residual context bias of accessible calls of ~0.3 RMS log2 over 3-mers, learned
emissions ~0.15). Project-level `context-bias` stays for comparisons across
samples and experiments.

## Design

**Shared core.** Both stages tally per (barcode, physical reference, position):
observed and modified calls (`accumulate_site_calls`, `site_table`), then
contexts and statistics from `analysis.compute.site_context_bias`
(`site_contexts`, `offset_enrichment`, `kmer_rates`), and figures from
`analysis.plot.site_context_bias`. CpG contexts are reported separately
(a CpG flag on every table, CpG rows marked in figures), so methylation shows
as such rather than as enzyme preference.

**Site types follow the modality:** C sites for deaminase; GpC and CpG for
conversion (the flanks beyond the dinucleotide carry the information); none
for direct modalities until a use appears.

**Plot/QC settings, not semantic ones.** `stage_context_qc` (on by default),
`stage_context_qc_flank` (3), `stage_context_qc_kmers` ([1, 3]) join the
stages' plot-only config keys (`_STAGE_PLOT_CONFIG_KEYS`): switching them
never makes a finished stage stale; the outputs appear the next time a stage
runs, or through `SCQ-03` for finished ones.

**Genome mode.** Tables per reference are kept; figures pool references per
barcode.

## Work items

| item | status | scope |
|---|---|---|
| `SCQ-01` preprocess context QC | proposed | modification bias per barcode x reference from passing reads; tables and figures |
| `SCQ-02` HMM context QC | proposed | per variant: residual context bias of state calls; modification rate within accessible-called sites |
| `SCQ-03` backfill for finished stages | proposed | the same outputs from a finished stage, without re-running it |
| `SCQ-04` qualification | proposed | a real run: counts equal `context-bias` on the same reads; run time |

### `SCQ-01` — preprocess context QC

QC and dedup flags are decided after the per-task pass (read statistics and
duplicate reductions), so tallies restricted to passing reads cannot be built
inside the tasks. A closing step reads the finished store partition by
partition -- site calls of the stage's site types, passing reads only (the
final obs flags) -- over worker processes, accumulating site tallies; memory
scales with sites x barcodes.

Outputs under the preprocess generation: `context_qc/site_counts.parquet`,
`offset_enrichment.csv`, `kmer_rates.csv` (each with barcode, reference,
site type, CpG flag); figures in a `context_qc` plot category -- per barcode x
reference an enrichment logo and offset heatmap, k-mer rates; one grid of
logos across barcodes per reference -- registered as plot artifacts.

Tests: tallies equal a direct count of passing reads on a fixture; failing and
duplicate reads excluded; bottom-strand contexts reverse-complemented; site
types per modality; settings off -> no outputs; the stage config hash is
unchanged by the settings.

### `SCQ-02` — HMM context QC (after `HCE-06`)

During apply, decoded state calls are at hand; per variant, tally per site:

1. **Residual bias of state calls**: accessible calls per C-centred k-mer,
   relative to the mean over k-mers. Flat if the HMM reads chromatin, not
   sequence.
2. **Modification within accessible-called sites**: modified / observed per
   k-mer where the state is accessible -- the enzyme's preference with most
   of the chromatin effect removed; exportable as a weight table (`HCE-01`
   format).

Figures per barcode x reference overlay the variants, one colour each
(translucent fills, solid outlines, as `HCE-06`'s histograms): residual-bias
k-mer profiles and accessible-conditioned k-mer rates.

Tests: tallies equal a direct count from decoded layers; one figure carries
every variant; no variants -> the default alone.

### `SCQ-03` — backfill for finished stages

`smftools experiment context-qc EXPERIMENT_DIR --stage preprocess|hmm` (and a
project form iterating experiments) writes the stage's context-QC outputs
from its current generation without re-running it -- for runs made before
`SCQ`, until they are regenerated.

Tests: backfilled outputs equal those the stage writes itself on a fixture.

### `SCQ-04` — qualification

On a real run: per barcode, preprocess tallies equal `context-bias` on the
same passing reads; HMM tallies equal counts from the decoded layers; the
added run time and output size per stage.

## Out of scope

Comparisons across samples, experiments or projects (project `context-bias`
and its sets); choosing or fitting HMM context weights (`HCE`).
