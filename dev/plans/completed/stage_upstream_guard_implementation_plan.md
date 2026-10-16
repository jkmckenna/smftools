# Stage commands never recompute earlier stages silently (`SUG`)

**Status:** completed. `SUG-01` merged; `SUG-02` folded into `HPS-07` (layer sets: fitting and decoding a new HMM plan without regenerating the stage), `in-progress/hmm_parameter_scopes_implementation_plan.md`.

## Why

`smftools experiment preprocess | spatial | hmm | latent` execute the whole
semantic plan up to their target (`cli/recipes.run_experiment_target` ->
`pipeline/experiment_graph.execute_experiment_target`). When an earlier stage
is stale -- e.g. raw after an extraction algorithm bump -- a command meant to
refresh one stage recomputes everything upstream. Observed on a deaminase
run: a spatial-only rerun started a full raw re-extraction (algorithm 4 -> 5)
and rewrote raw-stage manifest records before it was stopped; nothing in the
command's name or output said it would. `smftools experiment plan` showed the
stale raw stage, but nothing tied the stage command to it.

## Items

| id | item | status |
|---|---|---|
| `SUG-01` | stage commands refuse to replace earlier stages' existing outputs: `execute_experiment_target(..., allow_upstream=False)` raises `UpstreamRecomputeRefusedError` before any stage runs when an earlier decision is `stale_config`, `stale_algorithm`, `stale_input`, `invalid_artifact` or `dependent_recompute` (missing earlier stages are still built, so a first run of a later stage works as before); `preprocess`, `spatial`, `hmm`, `latent` and `batch` default to refusing and take `--allow-upstream`; `full`, `run` and library callers keep `allow_upstream=True`; tests; CLI docs | implemented |
| `SUG-02` | stage-only mode: run the target stage on the current upstream generations even though they are stale (e.g. refit the HMM on an existing preprocess without re-extracting), recorded as such in the stage's provenance so the plan still reports the upstream staleness | folded into `HPS-07` |
