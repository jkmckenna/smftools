## Analysis: `an`

Downstream statistical analysis library. Pure functions with no project-specific knowledge,
organised into four subpackages by role:

| Subpackage | Input | Output |
|---|---|---|
| `compute` | numpy arrays / DataFrames | arrays / dicts (no I/O) |
| `plot` | results + `output_path` | figure written to disk |
| `filters` | obs or var DataFrame | boolean `np.ndarray` mask |
| `config` | none (static) | configuration dicts |

```{eval-rst}
.. automodule:: smftools.analysis
   :no-members:
   :show-inheritance:
```

### Compute

Pure statistical compute functions. No AnnData dependency (except `ep_classification`).

```{eval-rst}
.. autosummary::
   :toctree: generated/analysis_compute

   smftools.analysis.compute.autocorrelation
   smftools.analysis.compute.pearson
   smftools.analysis.compute.hmm_features
   smftools.analysis.compute.ep_classification
   smftools.analysis.compute.ls_periodicity
   smftools.analysis.compute.dimensionality_reduction
   smftools.analysis.compute.read_cache
```

### Plot

Figure rendering. Accepts results and an explicit `output_path`; writes a figure to disk.

```{eval-rst}
.. autosummary::
   :toctree: generated/analysis_plot

   smftools.analysis.plot.heatmaps
   smftools.analysis.plot.histograms
   smftools.analysis.plot.autocorr
   smftools.analysis.plot.locus
```

### Filters

Boolean mask builders for obs-level and var-level (position) selection.

```{eval-rst}
.. autosummary::
   :toctree: generated/analysis_filters

   smftools.analysis.filters.obs_filters
   smftools.analysis.filters.position_filters
```

### Config

Static configuration objects; no runtime inputs.

```{eval-rst}
.. autosummary::
   :toctree: generated/analysis_config

   smftools.analysis.config.hmm_histogram
```
