# AnnData schema registry

smftools maintains a versioned AnnData schema registry to document the expected
structure of each processing stage (raw, preprocess, spatial, hmm, and future).
The registry captures the intended keys, data types, creators, and notes for
each AnnData slot (`obs`, `var`, `obsm`, `varm`, `layers`, `obsp`, `uns`).

The runtime pipeline also records a live schema snapshot in
`adata.uns["smftools"]["schema"]`, along with a `schema_registry_version` field
to link each file to the registry version used for documentation.

```{include} _generated_schema_tables.md
```
