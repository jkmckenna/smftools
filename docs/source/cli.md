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

## External workflow contract

Workflow engines should use `smftools experiment run` instead of rewriting an
experiment config or parsing stage logs:

```shell
smftools experiment run experiment.csv \
  --target full \
  --output-root "${TASK_OUTPUT}" \
  --input "${STAGED_BAM}" \
  --fasta "${STAGED_FASTA}" \
  --cpus "${TASK_CPUS}" \
  --memory-gb "${TASK_MEMORY_GB}" \
  --strict
```

The command writes a task-local runtime config, `software_versions.json`, and
`workflow_result.json` inside `--output-root`. The result records the semantic
plan, terminal outcome, generation and result IDs, relative artifact pointers,
available checksums, schemas, timings, structured failures, and the bounded
resource decision. Success, compatible reuse, and failure are represented by
the stable outcomes `success`, `compatible_skip`, and `failed`.

`--input` and `--fasta` accept concrete local files and local `file://` URIs.
Directory and remote URI inputs are not accepted in workflow mode; stage them
to one task-local file first. Read-only aliases are created inside the output
root so indexes and sidecars are also owned by the task. Overrides are applied
to the task-local config copy, and the source config and staged inputs are
integrity-checked without being rewritten. CPU and memory overrides can only
reduce the resolved config/host envelope. A requested CUDA or MPS accelerator
must also be available, and a CPU-only config cannot be expanded to an
accelerator.

Validate a completed or relocated bundle without writing:

```shell
smftools experiment validate "${TASK_OUTPUT}" --json
```

Validation exits nonzero for a failed result, incomplete or semantically
incompatible stage, missing/corrupt artifact, checksum mismatch, or pointer
that is absolute or escapes the output root. Internal workflow pointers are
relative to the output root, so moving the complete directory preserves the
contract.

Use `smftools versions --json` for the stable smftools/Python record. Repeat
`--tool` with a supported workflow executable (`dorado`, `pod5`, `minimap2`,
`modkit`, `gzip`, `multiqc`, `samtools`, `bedtools`, or
`bedGraphToBigWig`) to probe external versions explicitly. A workflow result
automatically records the tools required by the stages it is about to execute
and configured model identities; `--strict` fails before computation if one is
unavailable. In the [production CPU container](containers.md), the versions
record also includes the image, immutable tag, source revision, execution
profile, and runtime-supplied registry digest.

Project materialization uses the same result schema and task-local ownership:

```shell
smftools project run PROJECT_DIR CANONICAL_REFERENCE \
  --output-root "${TASK_OUTPUT}" \
  --layers C_site_binary \
  --cpus "${TASK_CPUS}" \
  --memory-gb "${TASK_MEMORY_GB}"

smftools project validate PROJECT_DIR "${TASK_OUTPUT}" --json
```

An unchanged project selection and projection request reuses a checksum-valid
materialization with the `compatible_skip` outcome. Project validation compares
the current semantic source plan with the published plan and reports stale
membership or feature inputs.
