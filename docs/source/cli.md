# Command-line interface

```{click} smftools.cli_entry:cli
:prog: smftools
:nested: full
```

## Read-only project planning

Use `smftools project plan PROJECT_DIR TARGET CANONICAL_REFERENCE` to inspect a
project analysis dependency plan without publishing artifacts or changing the
project registry. Targets are `selection`, `materialization`,
`sample-analysis`, and `embedding`. Add `--json` for deterministic
machine-readable output; the other selection and projection options mirror
project materialization.
