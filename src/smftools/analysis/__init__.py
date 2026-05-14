"""
smftools.analysis — downstream statistical analysis library.

Subpackages
-----------
compute     Pure statistical functions (array-in, result-out). No AnnData dependency.
plot        Figure rendering functions (result + output_path → file written to disk).
filters     Boolean mask builders for obs and var selections.
config      Static configuration objects (histogram binning, peak-calling parameters).

Design contract
---------------
- compute/ functions take numpy arrays or DataFrames and return results. No I/O.
- plot/ functions take results + an output Path and write a figure file.
- filters/ functions take obs/var DataFrames and return boolean masks.
- config/ modules contain static dicts/objects; no runtime inputs.
- No function in this package requires AnnData as input (ep_classification is
  the one exception: add_ep_obs_columns takes AnnData to write obs columns back).
"""
