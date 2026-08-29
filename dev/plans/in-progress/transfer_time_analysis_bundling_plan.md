# Transfer-time analysis bundling (`TAB`)

**Status:** in progress. `TAB-01` (`data bundle-analysis`) is implemented on
`feature/tab-01-bundle-analysis`. `TAB-02` (`data unbundle-analysis`) is
implemented on `feature/tab-02-unbundle-analysis` -- see its note for the
verification scope that actually shipped. `TAB-03` (real-data round-trip
qualification) remains open.

## Problem

An experiment's analysis tree (`preprocess`/`spatial`/`hmm`/`latent` outputs) is
written as many small, independent partition stores -- one zarr group per
`(reference, genomic core-window, barcode, read-chunk)` task, each roughly
1-90MB and a couple hundred files, per `informatics/partition_read.py`'s
`materialize()` and each stage's partitioned task executor
(`preprocessing/partitioned_executor.py`,
`tools/partitioned_hmm.py`/`partitioned_spatial.py`). A real, moderate-size
lab's `analyses/` tree measured over 1,600 zarr stores and, when copied whole
to another drive, over 2.4 million files for ~600GB of data. rsync between two
local drives on that tree slowed from ~100MB/s to under 1MB/s once it reached
the zarr-chunk-dense parts of the tree -- the per-file negotiation overhead
dominates, not the byte count.

**Two things were ruled out first, on real data, not assumed:**

- **Zarr v3 sharding** (`anndata.settings.override(auto_shard_zarr_v3=True)`,
  available in the installed anndata 0.12.19 + zarr 3.1.5). Tested against two
  real stores of different shapes: a metadata-heavy store (208 files, 68.6MB)
  took 405s to write (vs 0.24s unsharded), read 300x slower, used 25% more
  disk, and only cut file count 2.9x; a matrix-heavy store (276 files, 1.0MB,
  real 472x4690 `X`) took 14.4s to write (vs 0.56s), used 3.6x more disk, and
  only cut file count 1.1x. Sharding's default 1GB target shard size is wildly
  oversized against smftools' actual ~1-90MB per-store sizes, and sharding is
  an *intra-array* mechanism regardless -- it cannot address that the file
  count comes from having thousands of separate *stores*, not one store with
  too many chunks. Not revisited by this plan.
- **Coarser partitioning at the source** (fewer, bigger stores written by the
  pipeline itself). Task granularity is deliberately bounded to a 512MB
  working-memory budget (`preprocessing/dispatch_plan.py::plan_preprocess_tasks`),
  audited in `dev/plans/audits/pipeline_scaling_audit.md`, and each task's
  store is an independently resumable, kill-safe unit tied to the generation
  lifecycle. The file-count cost is the known, accepted price of that
  guarantee. Reopening it to reduce file count would need re-validating the
  scaling audit's guarantees, not a config change -- out of scope here.

What both rule-outs leave standing: the write path is sound for what it
optimizes (bounded memory, parallel/resumable tasks), and the file-count cost
only actually hurts at one specific moment -- **moving the tree to another
drive**. That is a transfer-time problem, and the fix belongs at transfer
time: bundle a run's already-written output into few, large files before
copying it, and unbundle on the far end so the destination is an ordinary,
directly scannable analysis tree again.

## Design

**Bundle granularity: one archive per published generation.** A generation
directory (`<stage>_outputs/generations/<generation_id>/`) is already the
codebase's immutability boundary -- `staged_generation` publishes it atomically
and it never changes again once complete. That makes it the natural unit to
archive: a generation's bundle needs to be created exactly once, ever, and a
later run only adds new bundles for new generations rather than touching old
ones. Bundling anything coarser (a whole stage, a whole run) would force a
full re-bundle on every incremental sync; anything finer (per partition store)
barely improves on today's file count.

**Format: plain, uncompressed `tar`.** Zarr chunk data is typically already
compressed by its own codec (blosc/zstd); re-compressing already-compressed
bytes inside a tar spends CPU for negligible size benefit, and risks a rare
slight *inflation* on incompressible data. Uncompressed tar also creates and
extracts faster, since there is no CPU-bound compression pass -- consistent
with this plan's local-drive-to-drive motivating case (see Problem above).
Leave compression a flag, not a hardcoded assumption, for whoever ends up
bundling over a genuinely slow network link instead.

**Self-contained, so unbundling never phones home.** Tarring the whole
generation directory includes that generation's own `generation_manifest.json`
(checksums and all) inside the archive. Unbundling can therefore validate a
freshly-extracted generation from the manifest it just extracted alone -- no
new checksum scheme invented, and no need to reach back to the source
machine to confirm the copy is intact. (What "validate" actually means
turned out narrower than first sketched here -- see `TAB-02`'s
implementation note below, after Work items.) This mirrors `PSR-01`'s
"surviving a detached archive" principle and `BCS-07`'s recorded-identity
validation.

**Three separate steps, not one command.** Bundle locally; transfer the
(now few, large) files with whatever the user already uses -- `rsync`, Finder,
`scp`; unbundle on the destination. Collapsing this into one network-aware
command would duplicate transport logic smftools has no reason to own. The
existing `rsync`/`data sync` tooling is already the right tool once file count
stops being the problem.

**`current.json` is deliberately out of scope for bundling.** Which
generation is *current* per stage is a small, cheap-to-sync pointer file, and
`data sync` (`data/run_sync.py`) already reconciles it correctly between two
attached analysis locations, additively, without ever guessing which side
wins. Bundling's job ends at moving a generation's bulk content efficiently;
pointer reconciliation stays exactly what it already is.

```text
smftools data bundle-analysis RUN_ROOT --to BUNDLE_DIR [--stage NAME] [--generation ID]
    # tars every *complete* generation not already bundled into
    # BUNDLE_DIR/<stage>/<generation_id>.tar

<transfer BUNDLE_DIR to the destination however -- rsync, Finder, scp>

smftools data unbundle-analysis BUNDLE_DIR --to RUN_ROOT
    # extracts each bundle to a staging dir under RUN_ROOT's matching
    # generation path, atomically renames into place, then re-verifies
    # whatever per-artifact checksums the extracted manifest itself records

smftools data sync RUN_ROOT ...
    # reconciles current.json the same way it already does today
```

## Work items

| item | status | evidence |
|---|---|---|
| `TAB-01` `data bundle-analysis`: tar every complete, not-yet-bundled generation under a run root | implemented | `tests/unit/data/test_analysis_bundle.py`, `tests/unit/test_data_cli.py` (`test_bundle_analysis_cli_*`) |
| `TAB-02` `data unbundle-analysis`: extract, stage-then-rename, verify | implemented | `tests/unit/data/test_analysis_bundle.py` (`test_unbundle_analysis_generations_*`), `tests/unit/test_data_cli.py` (`test_unbundle_analysis_cli_*`) |
| `TAB-03` real-data round-trip qualification: bundle, copy, unbundle, validate a real partitioned store; confirm file-count and wall-clock improvement over a plain `rsync` of the same tree | proposed | -- |

**`TAB-02` shipped generic checksum re-verification, not a dispatch to each
stage's own semantic validator, and that turned out to be the right call
rather than a shortfall.** The Design section above originally sketched
calling each stage's existing `resolve_current_*_generation`/
`validate_*_generation` -- but reading `validate_raw_generation` and
`validate_preprocess_generation` (only `basecall`, `raw`, and `preprocess`
have one at all; `spatial`/`hmm`/`latent`/`variant`/`chimeric` do not) showed
they check far more than content integrity: `validate_raw_generation` in
particular re-derives relative pointer safety from the *destination's*
`final_dir`/`run_root`, checking the extracted spine's `.uns` pointers
against the whole run's directory layout, not just the one generation being
unbundled. That is real, valuable machinery for *publishing* a generation,
but unbundling only needs to answer "did the round trip preserve every
byte" -- a narrower, purely content-integrity question the pipeline already
answered once when it first published the generation. Re-deriving the
broader business-logic checks by partially wiring into validators built for
a different caller, without having fully verified every argument they
expect, was the real risk here, not the narrower scope.

What shipped instead: a bundle's own recorded checksum is verified before
extracting (proves the *transfer* was not corrupted), and after extracting,
every artifact `sha256` the generation's own manifest records is
re-verified generically (proves the *tar round-trip* preserved what the
original pipeline vouched for) -- reusing the same `artifacts: {<key>:
{"path", "sha256"}}` shape `basecall`/`raw`/`preprocess` already write via
`artifact_record(..., checksum=True)`, without needing per-stage knowledge
of what those artifacts mean. `spatial`/`hmm`/`latent` do not record
per-artifact checksums in their manifests today, so unbundling one of those
can only confirm the manifest parses and its `generation_id` matches --
reported honestly through a `checksums_verified: False` result field, never
silently claimed as full verification. Extending those stages' manifests to
record artifact checksums (mirroring what `raw`/`preprocess`/`basecall`
already do) would let `TAB-02` verify them fully too, but that is a
manifest-schema change to those stages, not `TAB-02`'s own job.

## Decided

- **Bundle per generation, not per run or per partition store** (2026-08-28).
  Matches the codebase's own immutability boundary; see Design above.
- **Uncompressed tar by default** (2026-08-28). Zarr chunk data is usually
  already compressed; re-compressing spends CPU for little gain on a local
  drive-to-drive transfer, the motivating case here.
- **Bundle/transfer/unbundle stay three separate steps**, not one network-
  aware command (2026-08-28). Transport is already solved elsewhere; smftools'
  job is only removing the file-count penalty, not reimplementing `rsync`.
- **`current.json` reconciliation is explicitly not this plan's job**
  (2026-08-28). `data sync` already does it correctly and cheaply.
- **Neither zarr v3 sharding nor coarser source-side partitioning is
  revisited by this plan** (2026-08-28). Both were tested/considered and
  rejected before this plan was written -- see Problem above.
