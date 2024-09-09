# Basic Usage

Import SmfTools:

```
import smftools as smf
```

## Informatics Module Usage

Many use cases for smftools begin here. For most users, the call below will be sufficient to convert any raw SMF dataset to an AnnData object:

```
config_path = "/Path_to_experiment_config.csv"
smf.inform(config_path)
```


## Troubleshooting
For more advanced usage and help troubleshooting, the API and tutorials for each of the modules is still being developed.
However, you can currently learn about the functions contained within the module by calling:

```
smf.inform.__all__
```

This lists the functions within any given module. If you want to see the associated docstring for a given function, here is an example:

```
print(smf.inform.pod5_to_adata.__doc__)
```

These docstrings will provide a brief description of the function and also tell you the input parameters and what the function returns.
In some cases, usage examples will also be provided in the docstring in the form of doctests.