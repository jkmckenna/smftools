## Plotting: `pl`

Figure-rendering functions used by the CLI pipeline stages and available for direct use in
notebooks/scripts.

```{eval-rst}
.. autosummary::
   :toctree: generated/plotting

   smftools.plotting.autocorrelation_plotting
   smftools.plotting.chimeric_plotting
   smftools.plotting.classifiers
   smftools.plotting.general_plotting
   smftools.plotting.hmm_plotting
   smftools.plotting.latent_plotting
   smftools.plotting.plotting_utils
   smftools.plotting.pod5_plotting
   smftools.plotting.position_stats
   smftools.plotting.preprocess_plotting
   smftools.plotting.qc_plotting
   smftools.plotting.spatial_plotting
   smftools.plotting.umi_plotting
   smftools.plotting.variant_plotting
```

`qc_plotting` includes `plot_read_qc_histograms` (read length/quality/mapping QC, used by
`smftools preprocess`) and `plot_reference_barcode_chimera_rate` (the deaminase PCR-chimera-rate
heatmap written by `smftools raw`).

```{eval-rst}
.. automodule:: smftools.plotting
   :no-members:
   :show-inheritance:
```
