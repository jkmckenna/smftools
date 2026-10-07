# Periodicity figures: orientation, groupings, grids, spectra (`RPF`)

**Status:** in progress. `RPF-01` implemented. Follows `RPG` (completed). One PR
per item, in order.

## Why

First use of `smftools project periodicity` on real projects asked for:

- rows by peak period **descending**; no colour bar for the input panel (the
  periodogram keeps its own);
- positions oriented as the project's other clustermaps: a display coordinate
  (e.g. relative to a TSS, upstream on the left) rather than raw reference
  positions;
- several groupings of the same molecules without recomputing: each grouping
  is a full read and periodogram pass today (one CLI run per `--group-by`),
  minutes to tens of minutes per pass on a large set;
- grids of groups -- dose x enzyme, condition x cell type -- in one figure;
- a comparison between groups that share a region (e.g. two alleles of the
  same cells): mean spectra with uncertainty, and per-sample summaries.

## Work items

| item | status | scope |
|---|---|---|
| `RPF-01` clustermap presentation | implemented, not merged | descending order, no input colour bar, display coordinates |
| `RPF-02` several groupings per run | proposed | `--group-by` repeatable; one pass; figures per grouping |
| `RPF-03` grid figure | proposed | groups laid out by two labels, each cell input + periodogram |
| `RPF-04` mean spectra and summaries | proposed | mean power per period with bootstrap bands; per-group summary table |

### `RPF-01` — clustermap presentation

`plot_read_periodicity_clustermap` / `periodicity_row_order`:
`descending=True` (peak period largest first; `False` keeps ascending);
`input_colorbar=False`; `coordinate_origin` and `coordinate_reverse` give a
display coordinate `d = origin - position` (reverse) or `position - origin`,
columns ordered by `d` ascending (upstream on the left when reversed against a
forward reference), ticks in `d`. CLI: `--ascending`, `--coordinate-origin`,
`--coordinate-reverse`; figure options, not part of the results' cache key,
so a cached run redraws with them.

Tests: row order descending/ascending with bins; reversed coordinates put the
largest position first and label ticks in display units; smoke figures.

### `RPF-02` — several groupings per run

`compute_read_periodicity(group_by=[...])`: statistics carry one column per
grouping; the plot sample is chosen per combination of all groupings (so every
group of every grouping keeps up to N molecules). `--group-by` repeats; the
cache key holds the list. Figures go to `figures/<grouping>/<region>/`. One run
replaces one per grouping (a 75k-molecule set with four groupings: one pass
instead of four).

Tests: two groupings in one run equal two single-grouping runs (statistics and
power); figures per grouping; the plot sample covers every group of each.

### `RPF-03` — grid figure

`plot_read_periodicity_grid(cells, layout, row_labels, col_labels, ...)`:
`layout` is rows of cell keys (None = empty), each cell the arrays one
clustermap takes; each cell draws its input and periodogram heatmaps (rows by
peak period within the cell), with shared colour scales across the grid, one
periodogram colour bar, row/column labels, the cell's read count and median
peak in its title. `read_results(output_dir)` loads saved results (statistics,
power, plot values, grids) without a cache key, for drawing from finished
runs.

Tests: a 2 x 3 grid with an empty cell; shared scales; loader round trip.

### `RPF-04` — mean spectra and summaries

`plot_mean_spectra(panels, periods, ...)`: per panel, one curve per series
(mean power per period over its reads) with a bootstrap 95% band (reads
resampled), the median peak period marked; panels in a grid with labels.
`periodicity_summary(stats, by=[...])`: per group, reads scored, median peak
period, median SNR, fraction `peak_at_edge` -- the table for per-sample
comparisons.

Tests: bands contain the mean; a planted shift between two series shows in
the curves and the summary.

## Out of scope

Which groups go in which grid, and paired statistics between conditions,
are project analyses built on these.
