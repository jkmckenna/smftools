# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`smftools.analysis` is the downstream statistical analysis library within smftools.
It contains pure analysis functions with no project-specific knowledge, organised into four
subpackages that mirror the `smftools.tools` / `smftools.plotting` convention.

## Subpackage structure

```
smftools/analysis/
  compute/     Array-in, result-out. No file I/O, no AnnData dependency (ep_classification excepted).
  plot/        Result + output_path → figure written to disk.
  filters/     DataFrame-in, boolean-array-out. Obs and var selection masks.
  config/      Static configuration objects; no runtime inputs.
```

### `compute/` modules

| Module | Key public functions |
|---|---|
| `autocorrelation` | `binary_autocorrelation_with_spacing()`, `weighted_mean_autocorr()`, `compute_replicate_curve()` |
| `pearson` | `nan_pearson_matrix()`, `make_ticks()` |
| `hmm_features` | `extract_intervals_from_row()` |
| `ep_classification` | `classify_position()`, `add_ep_obs_columns()` |
| `ls_periodicity` | `analyze_ls_periodicity()`, `analyze_fft_periodicity()` |
| `dimensionality_reduction` | `run_pipeline()`, `coverage_filter()`, `make_features_raw()`, `make_features_acf()` |
| `read_cache` | `load_layer()`, `load_var_info()`, `load_obs_metadata()`, `is_cached()` |

### `plot/` modules

| Module | Key public functions |
|---|---|
| `heatmaps` | `plot_pearson_heatmap()` |
| `histograms` | `plot_interval_histogram()`, `gaussian_fit_plot()` |
| `autocorr` | `plot_autocorr_overlay()`, `plot_ls_overlay()`, `plot_metric_barplot_paired()` |
| `locus` | `plot_locus_map()` — planned |

### `filters/` modules

| Module | Key public functions |
|---|---|
| `obs_filters` | `build_obs_mask()`, `max_cigar_deletion()` |
| `position_filters` | `build_position_mask()` |

### `config/` modules

| Module | Contents |
|---|---|
| `hmm_histogram` | `HISTOGRAM_CONFIGS` — binning, rolling-mean, and peak-calling settings keyed by HMM layer name |

## Design contract

- `compute/` functions take numpy arrays or DataFrames and return results. No side effects, no file I/O.
- `plot/` functions accept results + an explicit `output_path: Path` and write a figure to disk.
- `filters/` functions take `obs` or `var` DataFrames and return `np.ndarray` boolean masks.
- `config/` modules contain static objects; import and use directly.
- No function reads from a project `constants.py`. Project-specific values are always passed as parameters.
- The one exception: `ep_classification.add_ep_obs_columns()` accepts an `AnnData` because it writes
  new obs columns back — this mirrors the `smftools.tools` pattern for in-place AnnData annotation.

## Adding a new module

1. Decide which subpackage: returns a result → `compute/`; writes a figure → `plot/`;
   builds a boolean mask → `filters/`; static config → `config/`.
2. Create the file with a module-level docstring stating inputs and outputs.
3. Add the function to the table in the relevant `__init__.py` docstring.
4. No project constants — pass everything as parameters.

## Typical import pattern (from a project driver)

```python
from smftools.analysis.compute import pearson, autocorrelation, read_cache
from smftools.analysis.plot import heatmaps, autocorr as plot_autocorr
from smftools.analysis.filters.obs_filters import build_obs_mask
from smftools.analysis.filters.position_filters import build_position_mask
from smftools.analysis.config.hmm_histogram import HISTOGRAM_CONFIGS
```

## Read cache pattern

Direct zarr layer access bypasses AnnData overhead (~138 ms per barcode vs ~1 s for backed h5ad):

```python
import zarr, numpy as np
from smftools.analysis.compute.read_cache import load_layer, is_cached

# parquet cache (current)
df, coords = load_layer(cache_root, barcode, ref_strand, "C_site_binary")
mat = df.to_numpy()[:, pos_mask]

# zarr direct access (target — sort obs by barcode before writing)
z = zarr.open(zarr_path, mode="r")
barcodes = z["obs"]["Barcode"][:]
index = {bc: (int(np.where(barcodes == bc)[0][0]),
              int(np.where(barcodes == bc)[0][-1]) + 1)
         for bc in np.unique(barcodes)}
start, end = index["NB04"]
arr = z["layers"]["C_site_binary"][start:end, :]
```
