# Sequence-context QC in the preprocess and HMM stages (`SCQ`)

**Status:** in progress. `SCQ-01`, `SCQ-02` merged, `SCQ-03` implemented. One PR per item, in order.
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
| `SCQ-01` preprocess context QC | merged | modification bias per barcode x reference from passing reads; tables and figures |
| `SCQ-02` HMM context QC | merged | per variant: residual context bias of state calls; modification rate within accessible-called sites |
| `SCQ-03` backfill for finished stages | implemented, not merged | the same outputs from a finished stage, without re-running it |
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

As built (`preprocessing/stage_context_qc.py`): after the stage obs is
written, `write_stage_context_qc` reads each barcode's task stores (`X`, the
core-cropped calls, and the `{reference}_{site}` var flags) in a worker
process. Passing reads are `passes_dedup` where dedup ran, else `passes_qc` --
the population the mismatch and segment clustermaps show. A call is modified
at >= 0.5. Tables: `site_counts.parquet`, `sites.parquet` (with context and
`cpg`), `offset_enrichment.csv`, `kmer_rates.csv` and `run.json`, under
`<generation>/context_qc/` (sidecar `preprocess_context_qc`). Statistics are
per (reference, site type, CpG flag) with the barcode as the group;
references without a stored sequence are skipped with a warning. Figures
(category `context_qc`) are per reference x site type x CpG flag -- an
enrichment logo, an offset heatmap and k-mer rates for each k > 1 -- each with
one panel per barcode, so the planned per-barcode logos and the cross-barcode
grid are the same figure. Barcodes with < 10,000 calls are left out of the
figures (not the tables); names lose their shared kit prefix. Tables are
written even with `emit_automated_plots` off; a failure is logged and never
blocks publication. The settings join `_NON_SEMANTIC_STAGE_CONFIG_KEYS`
(every stage) and the preprocess/HMM plot keys.

On the 260923 enzyme panel generation (928k reads, 34 barcodes, dedup-passing
reads only): tallies 7 s with 8 workers, 18 figures 7 s.

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

As built (`tools/hmm_context_qc.py`): in `execute_hmm_task`, for every
single-channel spec (default and variants) whose `<label>_all_accessible_features`
layer exists, the model input (`_prepare_model_input` with the variant's
config) and the state layer give per core site: observed, modified,
accessible (observed and accessible-called) and modified-while-accessible
reads, saved as a task partial. After all tasks, `write_hmm_context_qc`
reduces them to `<generation>/context_qc/`: `site_counts.parquet`,
`sites.parquet`, `kmer_rates.csv` (columns `measure` = `residual_bias` |
`accessible_rate`, model, variant, CpG flag, per barcode) and per model x
variant an `accessible_weights_<model>_<variant>_k<k>.parquet` weight table
(new source `accessible`; references pooled; group = barcode, so it loads
with `hmm_context_table_group: Barcode`). The partials are removed.
Figures (`context_qc` category): per reference x model x barcode x measure,
k-mer rates of the largest k with every variant overlaid (default first),
non-CpG and CpG panels; residual bias on the relative scale, the
accessible-conditioned rate on the absolute one; barcodes under 10,000 calls
left out. A failed tally or reduction is logged and never blocks publication.

260923 panel, current HMM generation (default only, intact top alleles, 42
tasks): 56 figures; mean |log2 relative rate| over 3-mers (non-CpG, BALB)
0.18 for the state calls vs 0.36 for modification within accessible sites.

### `SCQ-03` — backfill for finished stages

`smftools experiment context-qc EXPERIMENT_DIR --stage preprocess|hmm` (and a
project form iterating experiments) writes the stage's context-QC outputs
from its current generation without re-running it -- for runs made before
`SCQ`, until they are regenerated.

Tests: backfilled outputs equal those the stage writes itself on a fixture.

As built (`tools/context_qc_backfill.py`; `smftools experiment context-qc
EXPERIMENT_DIR [--stage ...] [--config CSV]`, `smftools project context-qc
PROJECT_DIR [--experiment ID ...]`, both with `--workers`, `--refresh`,
`--no-figures`): the config is the one recorded in `experiment_manifest.json`
unless a file is given; the generation is the stage's `current.json`
selection. Preprocess reuses `write_stage_context_qc` on the generation's
spine obs and task stores; HMM re-materializes each task's reads over its
core from the generation spine (decoded layers and input calls), tallies,
then reduces as the stage does -- in worker processes, as it is the slow
part. Existing outputs are kept unless `--refresh`; `run.json` gains
`backfilled: true`.

A published generation is otherwise left as it was: a preprocess
generation's manifest checksums `plots/`, `plots/catalog.parquet` and
`sidecar_manifest.json`, and is re-validated on reuse, so writing there would
make a finished stage read as corrupt and re-run. Backfilled figures go to
`<generation>/context_qc/plots/context_qc/` with their own catalog, and no
sidecar is registered (the stage-written outputs keep their places in
`plots/context_qc/` and the sidecar manifest).

### `SCQ-04` — qualification

On a real run: per barcode, preprocess tallies equal `context-bias` on the
same passing reads; HMM tallies equal counts from the decoded layers; the
added run time and output size per stage.

## Out of scope

Comparisons across samples, experiments or projects (project `context-bias`
and its sets); choosing or fitting HMM context weights (`HCE`).
