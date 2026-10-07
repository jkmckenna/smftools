# Per-read periodicity over regions (`RPG`)

**Status:** in progress. `RPG-01`–`RPG-03` merged; `RPG-04` implemented. One PR per item, in order.

## Question

How regularly spaced are the protected and accessible stretches of each
molecule, over a region of interest, and how does that differ between groups?
The spatial stage already answers part of this -- a Lomb-Scargle periodogram
per read over the whole read, on one site type, plotted per barcode -- but it
cannot be pointed at a region, a molecule selection, a different layer, or
shown beside the data it was computed from.

## What exists

- `analysis.compute.ls_periodicity`: `ls_periodogram_from_signal` and
  `analyze_ls_periodicity_direct` -- polynomial detrend, Lomb-Scargle over a
  period grid, peak period, SNR, peak power, FWHM -- on `(positions, signal)`.
- Spatial stage (`tools.partitioned_spatial`): per read and site type, the
  periodogram over 80-400 bp at 1 bp steps, detrend degree 2, at least 40
  sites; stored as `<site>_lomb_scargle_power` (reads x periods) with
  `<site>_ls_*` per-read statistics, read back with
  `materialize(spatial_spine, read_metrics=True)`. Its per-barcode read
  clustermaps cluster each panel on its own.
- Selection: `bind_ml_dataset` (`MLX-10`) streams any plan dataset --
  molecules, QC/dedup filters, groups, position masks, coordinate frames --
  with parallel block reads (`MLX-11`, `MRC`). `context-bias` (`SCB`) is the
  same pattern: a library function, a thin CLI, cached results.

## Design

**Input is a plan channel.** Stage, layer and site context name the signal:

| input | channel `stage` / `layer` / `site_context` |
|---|---|
| C-site calls | `preprocess` / `X` / `C` (or `GpC`) |
| HMM layer at C (or GpC) sites only | `hmm` / e.g. `C_all_accessible_features` / `C` |
| HMM layer at every position | `hmm` / layer / `all` (new, `RPG-01`) |

Lomb-Scargle takes irregular positions, so sparse site-restricted signals and
dense tracks go through the same code.

**Unit: (read, region).** Regions are half-open intervals in the dataset's
frame coordinates, given on the command line or as a BED-like file, or one
region per contiguous window of the plan's `positions` mask. Each read
contributes once per region it covers: at least `min_coverage` (default 0.8) of
the region's design positions observed, and at least `min_sites` (default 40)
observed values.

**Period range narrows to the region.** A periodogram needs several cycles
of the longest period it reports. The requested range (default 80-400 bp, as
the spatial stage) is cut per region to
`max_period = min(requested_max, region_length / min_cycles)` (`min_cycles`
default 3), and the peak-search range is clipped to it. When the narrowed
maximum falls below the requested minimum period, or below the peak range's
lower bound, the region is skipped with status `region_too_short` and a
warning. The effective ranges are recorded per region (outputs, `run.json`,
figure titles). One period grid per region keeps every read's periodogram on
the same axis.

**Per read and region**: detrend (polynomial, default degree 2), Lomb-Scargle
power on the region's period grid (normalized, so reads compare), peak period
within the peak range, SNR, peak power, FWHM, number of sites, status (`ok`,
`too_few_sites`, `low_coverage`, `no_peak`, `region_too_short`).

**Dense layers.** On an HMM layer at every position the signal is a step
function: its periodogram carries harmonics and partly reflects the HMM's own
length model. It is offered, documented as such, and the C-site input remains
the independent readout.

## Work items

| item | status | scope |
|---|---|---|
| `RPG-01` dense channels | merged | `site_context: all` in ML plans |
| `RPG-02` compute | merged | streamed per-(read, region) periodograms and statistics |
| `RPG-03` figures | merged | paired input / periodogram clustermaps in one row order |
| `RPG-04` CLI | implemented, not merged | `smftools project periodicity`, `smftools experiment periodicity` |
| `RPG-05` qualification | proposed | agreement with stored spatial periodograms; timings |

### `RPG-01` — `site_context: all`

Every position of the source layer is a design position; observed where the
value is finite and the read covers it. Refused on the `raw` and `preprocess`
stages, which hold site calls, so modification calls are never read off their
sites; a deaminase channel must still be declared accessibility.

Also here (`F73`): QC/dedup filter columns live only in preprocess obs, and
were looked up only in the stages a dataset's channels read, so a dataset
reading only the HMM stage failed. Preprocess obs is now always searched.

Tests: design mask is all positions within coverage; refused on preprocess
`X`; an HMM layer read with `all` equals the dense layer, with `C` equals it at
C sites.

### `RPG-02` — compute

`smftools.analysis.compute.read_periodicity` (pure): `period_grid(region,
period_range, peak_range, min_cycles)` -> grid, effective ranges, or a skip
reason; `read_periodograms(positions, values, observed, grid, ...)` -> power
matrix and per-read statistics for one batch and region, reusing
`ls_periodogram_from_signal` / `analyze_ls_periodicity_direct`.

`smftools.tools.read_periodicity`: `compute_read_periodicity(plan, dataset,
*, regions, channel, group_by, ..., workers)` streams `bind_ml_dataset`
batches (worker processes by whole blocks) into, per region, a statistics
table (`molecule_uid`, group, region, status, peak period, SNR, peak power,
FWHM, sites, coverage) and a power matrix (reads x periods) with its period
axis.

Tests: a synthetic signal with planted period recovers it per read; the
range narrows for a short region and the region is skipped below the floor;
coverage and site thresholds; a dense step signal and its site-restricted
sampling; 1 vs N workers identical; on the spatial-stage parameters a read's
periodogram equals `analyze_ls_periodicity_direct` on the same values.

As built: the period grid is descending, 1 bp steps, as the spatial stage
stores it. Statuses `ok`, `region_too_short`, `low_coverage`, `too_few_sites`,
`no_signal` (flat after detrending). Results come back in the dataset's
molecule order whatever the worker count. Checked on a real enzyme panel
(4,121 molecules, full amplicon, C sites): power and peak period identical
to the stored spatial-stage periodograms for the reads compared.

Also here (`F74`): with several workers, blocks are capped at an equal share,
so a dataset smaller than one memory-sized block no longer goes to a single
worker (that panel: 83 s -> 44 s with 8 workers).

### `RPG-03` — figures

`smftools.analysis.plot.read_periodicity.plot_read_periodicity_clustermap`,
one figure per (group, region):

- Two panels, one row order. Left: the input layer over the region's
  positions (binary calls, or the layer's own colours -- an HMM accessible
  layer green on grey). Right: the periodogram, `magma`, period axis.
- Rows sorted by peak period (option: binned by a column first). Reads
  without a valid peak are left out and counted in the title. An explicit
  row order may be passed instead (e.g. Leiden bins), so other figures can
  reuse it.
- Top marginal axes: mean signal per position above the input panel, mean
  power per period above the periodogram, the peak-search band shaded and the
  median peak period marked.
- Deterministic subsampling to `max_reads` rows; shared colour scale within
  a figure.

Smoke tests write the figure for site-restricted and dense inputs, an
explicit order, and a region with every read filtered out.

As built: input colours follow the HMM clustermaps (zero `#eee6d9`, no data
`#D0D0D0`, signal `#2E7D32`; binary inputs in two colours, continuous ones as
a ramp); columns no shown read observed are dropped by default
(`observed_columns_only`), so a site-restricted input shows its sites
side by side with true positions on the ticks; periods ascend left to right;
the counts of reads shown and left out go in the title.

### `RPG-04` — CLI

```
smftools project periodicity PROJECT_DIR --plan PLAN --dataset NAME --output DIR \
    [--channel C] [--region START-END ...] [--regions-file FILE] [--group-by COLUMN] \
    [--period-range 80 400] [--peak-range 150 250] [--min-cycles 3] \
    [--poly-degree 2] [--min-sites 40] [--min-coverage 0.8] \
    [--max-reads-per-plot N] [--workers N] [--refresh] [--no-figures]
smftools experiment periodicity EXPERIMENT_DIR ...   # same options
```

Writes `read_periodicity.parquet` (statistics), per region
`power_<region>.npy` with `periods_<region>.parquet`, `regions.parquet` (each
region's effective ranges or skip reason), figures, and `run.json`.

Results are cached under a key of plan hash, dataset, channel, grouping,
regions and every parameter, **plus a content hash of every file the dataset
references** (label table, coordinate maps). That key is a shared helper, also
used by `context-bias`: today its key misses those files, so an in-place edit
of a label table reuses stale counts (`F72`).

Tests: CLI on a fixture project writes every output; a re-run reuses the
cache; editing the label table invalidates it (both commands).

As built: figures per region in `figures/<region>/`, one per group plus
`all_groups.png` (binned by group); the input values of up to
1.25 x `--max-reads-per-plot` molecules per group are kept during the one
pass (each worker picks the same molecules from the identity table) and saved
(`plot_values_<region>.npz`), so a re-run redraws without reading. Plot values
are kept with or without figures, so `--no-figures` does not change the key.
Per-read `peak_at_edge` marks a peak on a bound of the peak-search range: on a
real panel 0.1% of reads over the full amplicon, 31% in a 600 bp region whose
range narrowed to 80-200 bp (peaks pile up at the 200 bp ceiling). A real
run (4,121 molecules, two regions, six groups, 8 workers): 48 s; cached
re-run with figures 3 s.

### `RPG-05` — qualification

On a real project: full-amplicon C-site periodograms with the spatial
stage's parameters equal the stored spatial-stage periodograms for the same
reads; a narrowed region; dense HMM vs site-restricted HMM inputs; 1 vs 8
workers identical, with timings.

## Out of scope

Project-specific sets, regions and output layout live in the project, which
writes the plan and calls the CLI. Joining periodicity into latent
clustermaps is a project figure that passes its own row order (`RPG-03`).
