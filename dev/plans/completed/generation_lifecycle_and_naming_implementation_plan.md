# Generation lifecycle, experiment naming, and cross-version analysis management

**Status:** complete. Every `EGL` item is terminal — 21 merged, 4 done, 1
complete, 1 superseded and reverted, and `EGL-01` partial with its remainder
shipped as `EGL-01b`. Verified by reading the status table 2026-08-26.

The `NKG` rollout continues, but it is a **log**, not a plan: it lives untracked
in `logs/nkg_regeneration_rollout.md` and never reaches "complete" by
construction, so it does not hold this document open.

Two companion documents are kept **untracked**, under `logs/`:

- `nkg_regeneration_rollout.md` — the `NKG-01`..`NKG-06` rollout. It is a naming
  scheme applied to twenty specific experiments, so the run identifiers are its
  content rather than incidental illustration; scrubbing them would leave
  nothing behind.
- `pipeline_findings.md` — `F17`..`F50`, findings from *running* the pipeline.
  Append-only, so it never reaches "complete".

Findings are cited here by id (`F33`), which is stable across file moves in a
way links are not.

**MERGED 2026-08-18.** `EGL-14` through `EGL-18` landed in PR #536 (main
`6705318`) -- as the combined branch rather than the five per-lane PRs that were
prepared, so all five arrived in one merge. `EGL-22`/`EGL-23` followed in PR
#539 (`709c35d`; main `8fb8da1`). Current `main` is **`8fb8da1`**, green at
2,212 unit / 106 smoke / 55 integration with Ruff clean.

**Everything in the EGL code program is now merged. Nothing is committed and
waiting.** Three merged remote branches remain to be deleted by hand
(`fix/staging-pointers-and-chimeric-coercion`, `fix/egl15-...`,
`fix/egl16-...`, plus `fix/coordinate-projection-consistency` now that #539 is
in); this environment has no push credentials.

**Where we are (2026-08-17).** Both code programs are merged. The EGL lanes
`EGL-01`..`EGL-08` closed with PR #514 (`c8d9b62`); the sibling SRB program
closed with `SRB-09` in PR #533 (`2f8aacc`; main `4e2021e`) — see
[selective_pod5_rebasecalling_implementation_plan.md](selective_pod5_rebasecalling_implementation_plan.md)
for its ledger rather than a copy of it here.

The active work is no longer a code lane. It is `NKG-03`: regenerating the 20
runs, starting with the `241213` pilot. Running that pilot has produced every
finding since — `F10`, `F9`/`F11`, and then `F12`–`F15`, each surfaced only by
running the thing rather than by reading it.

**The pilot now completes correctly through dedup and latent.** `EGL-13`'s exit
gate is met (`is_duplicate` 0 → 2,350; latent completes), but only because
`EGL-14` fixed a second, independent cause of the same symptom. That validation
ran into a fresh store, `store_f12f13_validation/`, because of `F15`: a code fix
does not invalidate stage compatibility and there is no force flag, so the
first re-run after `EGL-14`/`EGL-15` did nothing at all.

**The pilot has been re-run with all four fixes** (2026-08-17 17:12–17:20,
canonical `store/`; preprocess `552fc438`, hmm `43232f22`, latent `d0bd2544`).
Three generations now sit on disk, each isolating one change:

| | baseline `43ddbe08` | +`EGL-14/15` `c899677a` | +`EGL-16` `552fc438` |
|---|---|---|---|
| `is_duplicate` | 0 | 2,350 | 2,350 |
| `passes_dedup` | 3,861 | 1,511 | 1,511 |
| chimeric among `passes_qc` | 100.0% | 100.0% | **3.6%** |
| dedup-passing **and** non-chimeric | 1 | 1 | **1,405** |
| HMM clustermap PNGs | 0 | 6 | **180** |
| `No reads for …` warnings | 174 | 174 | **0** |

180 is the full grid — 15 barcodes × 2 references × 6 layers — with 15 of 30
panels carrying ≥20 analyzable reads (max 177). `EGL-16` left dedup identical,
which is the right independence: the two fixes govern different filters and
neither perturbs the other.

The pilot config now carries `variant_chimera_min_adjacent_sites,2` (backup at
`experiment_config.csv.bak`); adding it is also what forced the recompute, since
`F15` leaves no other way.

What remains:

1. **Two numbers want a human read**, both computed correctly for the first time
   rather than silently wrong: the **60.9% duplicate rate** and the **3.6%
   chimera rate**. Whether those are right is a judgment about the library, not
   about the code.
2. **Merge `EGL-14`/`EGL-15`/`EGL-16`/`EGL-17`**, then the other 19 configs
   and the regeneration runs.
3. **`F16` -- restore conversion-aware variant calling.** Explains the
   transition band around 2069-2629 the clustermaps expose (sites 2054 and
   2519 bracket it) and is the root cause of `F14`. Superseded note: `6B6_top`-aligned reads systematically carry
   `6BALB_cJ_top` calls through that stretch, where the pre-partition figure
   for the same barcode showed solid self-reference. Consistent with the 4-6
   breakpoints per QC-passing read measured under `F14`, and possibly a
   property of the informative-site catalog in that region rather than of the
   reads. Unresolved; recorded rather than assumed benign.

The pilot also unblocks the two `SRB-09` acceptance deferrals waiting on
regenerated data rather than on effort.

Standing blocker, unchanged: destructive `EGL-03b` pruning stays blocked until
byte-reproducibility evidence is authoritative.

## Implementation status

| Item | Status | Evidence |
|---|---|---|
| EGL-01 | **Merged, partial** — the *extension* half shipped; the *consolidation* half deliberately did not (now `EGL-01b`) | Three commits, all on `main`: shared helper `informatics/generation.py` (PR #502, `1825fe6`) — `staged_generation`, `resolve_current_generation`, `remap_staged_paths`, `publish_canonical_spine`, `rebind_staged_spine_pointers`, `has_published_generations`, 175 lines of unit tests; `hmm` (PR #503, `9e8c985`); `spatial` + the stale-source fix (PR #504, `a2e7e0f`). `chimeric` deferred by design. Validated against real runs of `241213`, not only unit tests — which is what caught the pointer leak (6 leaked pointers on hmm's first real run, 10 rebound on spatial's). |
| EGL-01b | **Merged** — composition PR #505 (`ad31a7c`) and consolidation PR #506 (`70075f3`) | `resolve_stage_generation` and stage-owned experiment-spine composition are merged. `raw`, `preprocess`, `latent`, and project embeddings now share the transaction/resolver while preserving their pointer schemas; 1,869 unit tests passed, and all 7 `241213` generations remained readable with 0 listing defects. |
| EGL-02 | **Merged** (PR #500, `9234810`, main `7829db3`) | `informatics/generation_listing.py`, `cli/generations.py`, `experiment generations` + `project generations`, 20 unit tests. Full unit suite 1834 passed; the 4 `test_ml_sklearn_vertical.py` failures were pre-existing (missing `skops`), confirmed on a stashed clean tree. **Stale as of 2026-08-14:** `skops` is now installed in `venvs/venv-all` and those 4 pass — stop citing them as the expected baseline (see the invocation note below). **Docs + acceptance entry still outstanding** — see EGL-08. |
| EGL-03 | **EGL-03a merged** — PR #507 (`1190e10`; main `bf1df24`) | External atomic `retention.json` registry, inventory schema v2 pin state, reasoned pin/unpin, and a dry-run-only prune planner. 43 focused tests and 1,892 full unit tests pass; the real `241213` tree remains 7/7 readable and reports 335,926,792 policy-candidate bytes but 0 reclaimable bytes. No deletion or force path until byte reproducibility is authoritative. |
| EGL-04 | **Merged** — EGL-04a PR #511 (`d0ffaa3`; main `3434bb3`), EGL-04b PR #512 (`4639abd`; main `eacfdff`) | `experiment rename-id` provides complete preflight, a durable prepared/committed journal, rollback, manifest history with immutable UID, config rewrites, UID-keyed project registry/path updates, explicit-list set rewrites, and per-sample pointer/store moves; query SQL and immutable generations stay untouched. 6 focused transaction tests, 88 combined identity/registry/project tests, and the full 1,923-test unit suite pass; Ruff, format, and Sphinx `-W` are clean. |
| EGL-05 | **Merged** — EGL-05a PR #508 (`205b383`; main `12f4e2d`), EGL-05b PR #509 (`8e8ac42`; main `2bbff5e`) | Periodicity and embedding definitions include independently bumpable algorithm versions plus `SEMANTIC_GRAPH_DEFINITION_VERSION`; existing caches remain untouched and naturally miss under the new keys. `project analyses list [--stale] [--json]` inventories current, stale, and invalid entries using definitions/manifests plus file metadata only, including legacy missing-identity reasons and bytes on disk; it never reads results or unpickles models. The real `nkg2a_final` project produced a valid empty schema (no caches yet). 9 focused identity/inventory tests and 1,901 full unit tests pass; Ruff, format, and Sphinx `-W` are clean. `set_store.py` remains deliberately cacheless. |
| EGL-06 | **Merged** — PR #510 (`6530f94`; main `cbbe2f5`) | Central completion-only stamping refreshes top-level `smftools_version`, `graph_definition_version`, and optional `git_commit` for direct and full-pipeline stage publication. Non-complete transitions preserve the last successful identity, and a later completion without Git metadata removes a stale commit. 28 focused manifest tests and 86 manifest/workflow/graph tests pass; the full unit suite is 1,905 passed, 8 skipped, 178 deselected, 7 xfailed. Ruff check/format and Sphinx `-W` are clean. |
| EGL-07 | **Merged** — PR #513 (`6439be3`; main `ce0f7a0`) | `experiment plan` and `project plan` accept `--upgrade-impact`, producing schema-versioned JSON or grouped human output over unchanged `PlanState` decisions. Reports identify triggers, downstream dependents, blockers, and historical timing coverage; experiment estimates use manifest `elapsed_seconds`, while unavailable project timing stays explicit rather than inferring compatibility from cache-specific artifacts. 59 focused graph/CLI tests and 85 pipeline/workflow tests pass; full unit suite 1,927 passed, 8 skipped, 178 deselected, 7 xfailed; Ruff, format, and Sphinx `-W` are clean. |
| EGL-08 | **Merged** — PR #514 (`729dd10`; main `c8d9b62`) | The directory-organization guide already contained the EGL-01/EGL-02/EGL-03 generation and retention material. This close-out adds the dedicated cross-version analysis tutorial, strengthens the spatial rerun test to prove the prior generation remains resolvable after `current.json` advances, and records five automated acceptance criteria spanning generations, retention, canonical identity, scoped cache invalidation, and read-only upgrade impact. 125 focused tests and the full unit suite (1,927 passed, 8 skipped, 178 deselected, 7 xfailed) pass; Ruff, format, and Sphinx `-W` are clean. |
| EGL-12 | **Merged** — PR #534 (`150ca76`; main `cd68dc0`) — `F10`, found 2026-08-17 when the pilot re-run aborted | Directory content hashes counted OS metadata, so opening a published generation in Finder reported it as corrupt. `constants.is_os_metadata` is now applied to every artifact-validity checksum; 13 focused tests, including that real content changes still register. No published generation in this project contained OS metadata at publication time, so nothing already recorded was invalidated. |
| EGL-11 | **Superseded and reverted** by EGL-13, 2026-08-17 — `F9`, found by running the `241213` pilot | `position_valid` is computed against the unfiltered read population before QC/dedup exist, so ~1,360 real positions per reference read as zero and `latent` fails. Implemented as a second pass writing `position_valid_analysis`, then rolled back: it corrected the denominator but left the coverage statistic wired into a membership-named column, so `F11` would have survived it. |
| EGL-14 | **Merged** — PR #536 (`37c700b`; main `6705318`) — `F12` | Artifact pointers bind to `output_dir` for the whole of execute and rebind to `publication_dir` once, after the last mid-execute consumer. The rebind deliberately sits after the summary plots, not immediately after dedup: plots materialize slices too, so an early rebind fixes dedup and leaves preprocess plotting reading a path that does not exist — caught by writing the fix wrong first. `_overlay_preprocess_var` now warns on a claimed-but-missing catalog while staying quiet for a raw slice that has no preprocess stage. No published artifact changes: `_bind_generation_spine` rewrites these pointers at publish, so every generation on disk already carries correct ones. 5 focused tests, 3 of which fail against `main`; the other 2 are over-correction guards. **Validated on real data** — see EGL-13's exit gate below. |
| EGL-15 | **Merged** — PR #536 (`d23265c`; main `6705318`) — `F13` | `plotting_utils.coerce_bool_series` replaces `~s.astype(bool)` at all three sites that negate `chimeric_variant_sites` (two in `hmm_plotting`, one dormant in `spatial_plotting`). Follows the truthy vocabulary `HMM._coerce_bool` already established; missing and unrecognized values are False, so an unexpected label degrades to "keep the read and plot it" rather than aborting a stage. 25 focused tests including a guard that the filter still excludes when reads genuinely are chimeric. Verified 0 → 15 PNGs on a direct call against the published generation. **Necessary but not sufficient in practice** — `F14` keeps the pilot's clustermaps nearly empty for an unrelated reason. |
| EGL-13 | **Merged** — PR #535 (`4fc0e60`; main `d862e3a`) — `F9` + `F11`; **exit gate MET 2026-08-17** | Separates structural membership from coverage density. `position_in_<reference>` becomes membership (`positions.isin`); coverage keeps `<reference>_position_valid`; dedup drops the density intersection (`min_overlap_positions` is the correct per-pair guard); latent measures density from the matrix it is about to factorize. No published artifact changes, so no regeneration is owed for the fix itself. 9 focused tests, 6 of which failed against `main` before the change; full suites green on the branch (2,128 unit, 106 smoke, 55 integration; Ruff check/format clean). **Exit gate met 2026-08-17**, but only once `F12` was also fixed: the first re-run after this merge still reported 0 duplicates, because dedup was never seeing the columns this lane corrected. With `EGL-14` in place the same pilot reports `is_duplicate` 2,350, `passes_dedup` 1,511 (from 3,861), 2,983 clusters of size > 1, max cluster 39, reason `sequence_cluster`; latent completes and publishes. This lane's own contribution is independently confirmed — membership is 4683/4683 and the dedup comparison mask resolves to 154 sites, both 0 before. |
| D1–D3 | **Decided** 2026-08-14 | See "Decisions taken" below. D3 closes identity gates 4 and 7 with inferred compatibility plus strict validation. |
| EGL-17 | **Merged** — PR #536 (`9793ec5` + `a53ebcc`; main `6705318`) | Variant segment clustermaps for the partitioned pipeline. The renderer (`plot_variant_segment_clustermaps`) was never missing — it is the same function that drew the pre-partition figures — but only `cli/variant_adata.py` called it, because it reads dense `*_variant_segments` / `*_variant_call` layers while the partitioned store keeps variant evidence sparse by design. `preprocessing/partitioned_variant_plots.py` rasterizes the stored sparse evidence onto a read × position grid and hands it to that renderer rather than reimplementing it. Segments are read back from `events.parquet`, not recomputed, so the picture cannot drift from the flags it explains. Segment geometry stays the *raw* interpolation while the mismatch-type strip shows the `EGL-16`-gated verdict — seeing the disagreement is how the threshold gets checked. Panels plot the analysed (dedup-passing) population, matching the old `deduplicated/` output. Adds `variant_segments` to the preprocess plot categories. Verified end to end on `241213`: 30 panels, one per reference × barcode, and the `barcode02` panel renders 103 dedup-passing reads against 105 in the pre-partition figure with the same structure. `a53ebcc` fixes catalog registration, which the first commit omitted — the plots existed on disk but contributed 0 catalog rows, invisible to anything discovering plots through it. 7 focused tests on the rasterizer. |
| EGL-18 | **Merged** — PR #536 (`7844aea`; main `6705318`) — `F16` | **Validated on the pilot** (`store_egl18_validation`, preprocess `9c9deca7`). Informative sites/read 22 -> 16 for top-strand reads, all six C/T pairs excluded, G/A retained. Chimera rate among QC-passing 3.6% -> 3.1%; the small move is because `EGL-16`'s threshold was already masking the damage. Unmasked at `min_adjacent_sites=1` the real effect is **66.4% -> 13.9%**. Also *recovers* evidence: median no-calls per read 3 -> 2, because at surviving C/A-type sites a converted read showing `T` now matches `{C,T}` instead of no-calling, so 448 more reads carry usable variant evidence. Dedup is byte-identical (1,511 / 2,350), confirming independence. |
| EGL-18 (spec) | superseded by the row above | Strand-aware conversion site typing: acceptance chosen by the read's mapped strand (`top`->`C->T`, `bottom`->`G->A`) rather than a union. Closes the C/T miscalling at 46.6-70.7%. Makes informative-site status read-dependent, so it can no longer be a single precomputed catalog column. See the lane spec below. |
| EGL-19 | **Merged** — PR #541 (`55a3cc6` + `0bd44e0`; main `3169585`) | Per-molecule deamination segmentation as a preprocess sidecar `deamination/`. Rebuilt twice on measurement. (1) Counting only converted positions over-called chimeras ~300x (5,965 vs 18) — those molecules carried 32 `G->A` events against **651 retained Gs**, G-position conversion 0.036 vs 0.317 at C. Every C/G position is now evidence. (2) Efficiency estimated per read, since efficiency *is* the footprint signal and a fixed value reads protection as a switch. (3) Greedy binary segmentation replaced by optimal partitioning with PELT pruning: on `top|bottom|top` no single split improves the fit, so it reported 1 switch where there were 3 (best sub-window gain 1.15e-14). Prefix-sum costs took it from ~46 billion operations (not finishing) to 17.6 ms/read. On `251105`: 18 chimeric, median 1 segment. | Per-molecule deamination segmentation as a preprocess sidecar `deamination/`, mirroring `variant/`. Per-position state from the same CIGAR walk (`C->T`, `C` retained, `G->A`, `G` retained); segments are runs of >= 3 consecutive same-type events (`deaminase_segment_min_adjacent_events`). Deliberately **not** in raw — the walk is reproducible at preprocess time from the raw ragged row, and the threshold is an analysis parameter the user expects to revise. Gated off for `direct`. |
| EGL-20a | **Complete** — core `62ff96e` (PR #542), reorder `f5e5823` (PR #543), wiring `bbb4535` (unmerged). Measured on `251105`: **zero chimera calls changed** (calls 330,814 -> 330,807, no_call 78,306 -> 78,299). The pilot has median 1 deamination segment per molecule, so local chemistry equals read-level chemistry almost everywhere — insurance for data containing switches, not an improvement here. Measuring found two defects testing had not: only one strand's catalog was built (making the lane a silent no-op), and the callable-fraction denominator inflated ~27%. Note the candidate-site exclusion the ordering was designed around changed nothing on `251105` (18 chimeras either way, 0.5% of observations dropped) — kept as insurance for a denser reference pair, not as a demonstrated fix. | Deaminase variant calling takes per-position acceptance from the covering deamination segment, not a per-read rule: in deaminase the applicable chemistry varies *within* a read. Requires excluding reference-derived SNP sites from deamination evidence first (20 of 22 pilot sites would otherwise be miscounted as deamination events); resolved by an acyclic ordering since the candidate site list is reference-only. Inverts the lane order for deaminase: `EGL-18b` depends on `EGL-19`. |
| EGL-20 | **Merged** — PR #545 (`38083c3`; main `cbb8399`) | Measured on a 25k-read deaminase run: variant 233 + scalar 18 + segment 18 -> union **266**, exactly 3 overlapping. Only **one** additional read leaves the analysed population (2,999 -> 2,998), since chimeric reads mostly fail QC or dedup upstream. | `omit_chimeric_reads` omits a read chimeric by **either** method (union of the two columns; decided 2026-08-18). Test fixture: a 25,098-read natively-barcoded deaminase run, same reference pair as the conversion pilot, 11 chimeras under the current scalar method. Two separate chimera classes: deaminase-detected (both `C->T` and `G->A` segments in one molecule) and SNP-resolvable (`has_other_reference_segment` on `EGL-18`-corrected calls). Supersedes the scalar `deaminase_chimera_mask`, kept in parallel for one release for comparison. |
| EGL-24 | **Done** — merged 2026-08-21 (PR #549) | Cap reads for spatial position-correlation matrices on the **CLI path**, subsampling to **1,000 molecules** by default; `compute_positionwise_statistics` keeps no cap so library callers are unaffected. Clustermaps already cap at `clustermap_max_reads_per_plot: 5000` with deterministic selection; the correlation path has **no cap at all** and is the asymmetry. Both pilots set `spatial_generate_position_matrices: FALSE` against a default of `True`, which suggests the cost is already being avoided rather than paid. |
| EGL-25 | **Done** — merged 2026-08-21 (PR #547) | `bypass_deamination_segmentation` to skip PELT chimera detection, matching the fourteen existing `bypass_*` flags. Distinct from `deamination_reporting_enabled`, which gates on the modality having no chemistry at all. Must leave the scalar `deaminase_PCR_chimera` column computed, and `omit_chimeric_reads` must degrade to the scalar alone rather than treating an absent column as "not chimeric". |
| EGL-26 | **Done** — merged 2026-08-21 (PR #548) | Per-read mismatch clustermaps during preprocess, from the pre-partition `variant` CLI (`plot_sequence_integer_encoding_clustermaps`, `exclude_mod_sites=True`). Renderer exists; the layer does not — `mismatch_integer_encoding` is absent from the partitioned store and must be rasterized from the raw ragged `MISMATCH`/`SEQUENCE` arrays, the `EGL-17` pattern. Plots the dedup-passing population (decided 2026-08-20), matching `EGL-17` and the old `deduplicated/` output. Must register in the plot catalog and honour `index_col_suffix`, both of which `EGL-17`/`EGL-23` had to fix after the fact. |
| EGL-27 | **Done** — merged 2026-08-21 (PR #547) | Raise `clustermap_max_reads_per_plot` 5,000 -> **10,000** in dataclass *and* `default.yaml` (the YAML wins on the CLI path, `F18`), and make selection seeded-random everywhere. `EGL-17`'s variant segment panels currently take first-N by read id rather than sampling — my error, and a biased subset. |
| EGL-28 | **Merged** — `28a` (PR #550), resolutions (PR #552), `28b`+`28c` (PR #553), `28d` (PR #554), all 2026-08-21 | Per-DR-strategy KNN + Leiden (today Leiden is computed once from UMAP's internal graph, not per strategy), hierarchical ordering within clusters, clustermap of the non-DR layer binned by Leiden, and stacked composition barplots. Spec is the `Nkg2a_DAFseq_merged/claude_outputs` reference analysis. Four sub-parts and four open questions — see the lane entry. |
| EGL-21 | **Merged** — PR #546 (`68d49d0`) | Deamination segment clustermaps, reusing the variant renderer (state encodings coincide; transitions synthesized between differing segments). 32 panels / 32 catalog rows on `251105`. Also parameterizes the renderer's filename suffix — panels were being written as `..._variant_segments.png` inside `deamination_segments/`. | Deamination segment clustermaps, the `EGL-17` pattern applied to deamination state; new `deamination_segments` plot category. |
| EGL-22 | **Merged** — PR #539 (`709c35d`; main `8fb8da1`) | Clustermaps crop to the read span by default. On `241213` ~40% of the plotted axis carried no read. Per reference rather than per barcode: barcode spans differ from the union by <2% of positions, which does not justify losing a common x-axis. Flipped in the dataclass **and** `default.yaml` — the YAML sits above the dataclass on the real config path, so the dataclass alone would have changed nothing (`F18`). |
| EGL-23 | **Merged** — PR #539 (`709c35d`; main `8fb8da1`) — `F17` | Coordinate projection made consistent and its legacy flag made loud. `invert_adata` now raises with a message naming `reindexing_invert`, firing only on `True`. Variant segment clustermaps project through `index_col_suffix` like HMM and spatial. Eight EMseq configs migrated off the dead flag (all ten DAFseq runs had been migrated in 2.12.0; the EMseq ones never were, so their figures were silently uninverted for months). Coordinate model documented in `experiment_config.md`, which previously mentioned neither parameter. |
| EGL-16 | **Merged** — PR #536 (`7a61a61`; main `6705318`) — `F14` | `variant_chimera_min_adjacent_sites` (default 2, settable from the config sheet) requires a run of N consecutive informative sites calling the other reference. Threaded through both segmenters: `segment_sparse_variant_calls` (partitioned pipeline, which produced the pilot) and `segment_variant_calls` (legacy CLI). Support is counted in *sites*, not bases — variant sites are sparse, so a base-length floor would measure interpolation distance rather than evidence. Library defaults stay at 1 so direct callers are unchanged and the old behavior stays reachable for comparison; only the config default is 2. Support gates membership, never geometry: a qualifying stretch is kept whole, because trimming it to its supporting sites re-types a run-to-the-end segment from `right_segment_mismatch` to `middle_segment_mismatch` — the first version did exactly that and the existing reference-contract test caught it. Counts stay raw, since `variant_other_base_count` is what diagnosed `F14`. 17 focused tests across both segmenters. **Threshold chosen from the data, not assumed** — see the run-length distribution under `F14`. **Open:** no generation is rewritten, so the pilot must be re-run before its chimera numbers mean anything. |
| EGL-29 | **Merged** — PR #555 (`e5868e4` + `69943cd`), 2026-08-21 | Pre-basecalled FASTQ input as a first-class source: decouple *barcode authority* from *end-reason re-derivation* so an already-demuxed FASTQ tree can keep its directory-assigned barcode while `demux_type` is re-derived from sequence, and **report disagreements** between the two assignments. Plus a `sequencing_summary.txt` reader for the case where dorado left one. Motivated by a pre-basecalled FASTQ run. |

**Test-invocation note (2026-08-14).** Run the unit suite as
`venvs/venv-all/bin/python -m pytest -m unit -q`, not
`venvs/venv-all/bin/pytest`. The bare launcher leaves the repo root off
`sys.path` in *spawned* workers, so the four process-pool tests
(`test_memory_guard.py` ×3, `test_raw_adata.py::test_map_references_parallel_sequential_matches_process_pool`)
fail with `ModuleNotFoundError: No module named 'tests'` inside the child. It is
a launcher artifact, not a regression — verified by re-running those exact files
under `python -m pytest` (69 passed) and then the whole suite at `main`'s content
(`a2e7e0f`): **1858 passed, 8 skipped, 7 xfailed, 0 failed** in 4m12s. That is
the baseline to compare against from now on. Worth recording because the failure looks
like a real parallelism bug and the previous baseline in this document cited a
different set of four failures for an unrelated reason.

**What EGL-02 confirmed for EGL-01:** all four layouts use identical directory
and file naming, differing only in manifest payload and in whether the pointer
carries `manifest_sha256` (latent omits it). A shared helper is therefore a
genuine consolidation, not a forced abstraction — discovery needed no per-kind
branching at all.

**What EGL-01 then confirmed, against real runs:** the six-step wiring spec below
was right about steps 1–6 but understated step 1. Correct run-root recovery is
necessary but *not sufficient* — the spine's `uns` pointers are run-root-relative,
so they faithfully recorded `.staging/<id>/…` and dangled the moment the tree
moved. Six leaked pointers on hmm's first real run, ten rebound on spatial's.
Predicting the hazard did not prevent it; only running it caught it. Any further
kind adopting the helper (basecalls, `chimeric`) must call
`rebind_staged_spine_pointers` and be validated on a real run, not just unit tests.

The same work exposed a **pre-existing** defect worth carrying forward: `hmm` and
`spatial` called `partitioned_stage_is_complete` without `source_path`, which
`preprocess` already passed — so neither noticed its source had changed
(`experiment plan` said `stale_input` while `experiment hmm` skipped). Fixed in
`a2e7e0f`. It mattered *now* because `NKG-03`'s bulk regeneration drives stages
directly rather than through `experiment full`, which does honour the planner.
Target: `chore/start-2.21.0-dev` lineage
Prefixes: `EGL-` (smftools code), `NKG-` (local project rollout)

## Objective

Make an experiment's *identity* and its *generations* durable enough that a
publication can cite a specific computation, and that upgrading smftools or
re-running basecalling produces a **new** addressable result rather than
silently replacing the cited one.

Two halves, deliberately in one document because the second is what forces the
first:

- **Part A (`EGL-*`)** — smftools code changes.
- **Part B (`NKG-*`)** — local organization for the `nkg2a_final` project,
  which is the first real consumer.

## Relationship to `selective_pod5_rebasecalling` (SRB)

`dev/in-progress/selective_pod5_rebasecalling_*` already owns re-basecalling:
selection, checksum-first source resolution, immutable model identity,
descendant raw generations that declare their origin, lineage-aware project
registration, and explicit promotion/rollback. **This program does not
re-specify any of that.**

It covers what SRB assumes but does not provide:

| Concern | Owner |
|---|---|
| Re-basecalling → descendant lineage | SRB |
| Lineage promotion / rollback | SRB |
| Generations for stages *after* preprocess | **EGL** |
| Listing / retention / pruning of generations | **EGL** (SRB non-goal: "Deleting … historical lineages") |
| Experiment identity and naming discipline | **EGL** |
| Analysis invalidation on a code upgrade | **EGL** |

`EGL-01` and `EGL-04` are prerequisites for SRB landing cleanly: SRB's
`lineages/<id>/run/{spatial,hmm,latent}_adata_outputs/` presumes those stages
have a generation concept, and its project fan-out presumes stable experiment
ids.

---

## Reconciliation with SRB: where basecall outputs live

SRB and EGL currently imply two different homes for a basecall. This section is
the reconciliation; `SRB-04`/`SRB-05` and `EGL-01` should both be read against it.

### The conflict

SRB's artifact layout nests the basecall *and* a full pipeline run inside each
lineage:

```text
rebasecall_outputs/lineages/<lineage_id>/
  basecall/{basecall_manifest.json, calls.bam, sequencing_summary.tsv}
  run/{experiment_manifest.json, raw_outputs/, preprocess_adata_outputs/, ...}
```

Local practice (and `NKG-02`, already executed) puts basecalls at **run** level,
shared by sibling experiments:

```text
analyses/runs/<run_name>/basecalls/{hac,sup}_canonical_basecalls.bam
```

Both cannot be the home. The evidence favours run level:

- A basecall is a property of `(run x model)`, not `(run x modality)`.
  `260309` has **one** SUP basecall serving two experiments (conversion barcodes
  1-5, deaminase barcodes 6+); `251216` has **two** basecalls (HAC, SUP) serving
  two experiments. Neither relation fits inside an experiment or a lineage.
- The two modalities align to different reference sets — conversion builds
  `*_converted_*.fasta` and maps to `6B6_unconverted_top`, deaminase has an empty
  `fasta_outputs/` and maps to `6B6_top` — so the *aligned* BAM must stay
  per-experiment while the *basecalled* BAM is shared.
- `IntermediateSpec.compatibility_payload()`
  (`informatics/raw_intermediate_manifest.py:193`) keys the `dorado-basecalling`
  intermediate on operation + pod5 tree checksum + `operation_config`
  (model, barcode_kit, device, trim, modifications) + tool-version policy —
  **nothing experiment-specific**. Two experiments over one pod5 with one model
  already compute an identical `compatibility_key`; only the *lookup path*
  (`_revision_manifests`, line 225) is scoped to the requesting experiment's
  `output_directory`.

SRB's own text anticipates the fix: the lineage container exists *"even where an
individual stage does not yet expose multiple generations in one run root"* — a
stated workaround for the gap `EGL-01` closes.

### The reconciled layout

A basecall becomes a **fifth generation kind**, using the same on-disk vocabulary
`EGL-01` consolidates:

```text
analyses/runs/<run_name>/
├── basecalls/
│   ├── current.json                       # default basecall for new experiments
│   ├── generations/<basecall_id>/
│   │   ├── generation_manifest.json       # model identity, tool_versions,
│   │   │                                  #   compatibility_key, pod5 selection +
│   │   │                                  #   checksums, generation_kind, read count
│   │   ├── calls.bam (+ .bai)
│   │   ├── sequencing_summary.tsv
│   │   ├── read_to_pod5_origin.csv        # describes the basecall, so it lives here
│   │   └── selection/                     # SRB: which pod5/read IDs went in
│   └── .staging/
├── <experiment_id>/                       # one per (modality x basecall)
│   ├── experiment_config.csv
│   └── store/{raw_outputs, preprocess_adata_outputs, ...}
└── <ref>.fasta
```

Consequences, all of which reduce total work:

1. **SRB's lineage references a basecall generation id instead of embedding the
   BAM.** SRB already carries `resolved basecall result ID` as a lineage identity
   component, so this is a storage change, not a model change. Re-basecalling
   publishes a new basecall generation at run level; the lineage records which.
2. **`smftools experiment generations` (EGL-02, merged) lists basecalls for
   free** once `basecalls/` uses the shared vocabulary — no new code.
3. **Retention (`EGL-03`) covers basecalls**, where it matters most: a basecall is
   the only artifact here that cannot be regenerated without pod5 and a GPU, so it
   should outrank every derived generation in the pruning order.
4. **`EGL-09` becomes nearly free** — with basecalls at run level and the
   compatibility key already experiment-independent, sharing is a search-path
   change, not a cache design.
5. **As `EGL-01` gives spatial/hmm generations, SRB's `lineages/<id>/run/`
   nesting can collapse** into references to per-stage generation ids, which SRB
   states it would prefer.

### Decisions taken (2026-08-14)

Both were raised as open items by this reconciliation and are now settled. `SRB`
should adopt them rather than re-specify; where they change SRB text, the change
is noted per decision.

#### D1 — Two selector layers, and `current.json` is the only new one

**Decision.** Exactly two mechanisms, at different scopes, plus one deletion:

| Scope | Mechanism | Owner | Answers |
|---|---|---|---|
| Within one kind, at one location | `current.json` | EGL | "which basecall / raw / hmm / spatial generation is the default here" |
| Across stages, for one experiment | named lineage in the project registry (`active_lineage`) | SRB | "which coordinated *set* of stage generations does this experiment resolve to" |

SRB's optional experiment-local `active.json` (artifact layout, plan line 409) is
**dropped**. It is the third mechanism and the actual duplication risk: a lineage
is already an experiment-scoped selector, and having a second one inside the
lineage container invites the two to disagree with no rule for which wins. An
experiment consulted outside any project resolves `original`.

**Why this shape.** A lineage *is* a map `stage → generation id`. The `original`
lineage is that map defined implicitly as "each stage's `current.json`". So the
two layers compose rather than compete, and `EGL-01` ships **no** new selector —
which is what the lane text below worried about ("ship a narrower experiment-local
one that SRB can subsume"). There is nothing to subsume.

**What this obliges the code to do** (`EGL-01b`). One resolver,
`resolve_stage_generation(stage_dir, lineage=None)`: with no lineage it reads
`current.json`; with a lineage it takes the pinned id. `write_experiment_spine`
then composes over *resolved generations* instead of stage-root paths.

Before PR #505 it did neither, and that was a **live defect**, not a future concern.
`experiment_spine.py:110` unions each stage's `uns` in a fixed order —
`raw, preprocess, spatial, hmm, latent` — with later stages overwriting earlier
ones, and only `raw` resolves a generation at all (line 112). Because
`hmm_spine = spine.copy()` carries a *copy* of spatial's `uns`, and `hmm` sorts
after `spatial`, an hmm run that predates the current spatial generation
overwrites spatial's fresh pointers with its stale copies. Observed directly last
session: the canonical spatial spine was correct while the experiment spine,
written one second later, held the old pointer.

The composition rule that fixes it, and that a generation set requires anyway:
each stage contributes only the keys it **owns** — its `<stage>_*` prefix — taken
from its resolved generation's spine; shared non-prefixed keys come from the
earliest stage in the order. Verified as workable: every key `hmm` and `spatial`
write is already cleanly prefixed (`partitioned_hmm.py:1344-1358`,
`partitioned_spatial.py:1807-1834`). Confirm the prefix rule against `raw`,
`preprocess`, and `latent` key sets before implementing — those were not audited.

#### D2 — `generation_kind` lives on the **basecall** generation

**Decision.** `full_source | parent_universe | selected_cohort` is stamped on the
basecall generation manifest. The descendant raw generation records
`basecall_generation_id` and *derives* `generation_kind` from it. Raw may mirror
the value for read convenience, but the basecall manifest is authoritative and
validation fails on disagreement. This amends SRB's "Descendant raw generations
declare their origin" contract (SRB plan line 293).

**Why.** Three reasons, in order of force:

1. Selection determines the *content* of the BAM — the basecall generation
   contains exactly the selected reads. Raw is a deterministic function of
   (basecall, reference, config) and has no independent notion of a cohort.
2. Under the reconciled layout, basecalls sit at run level and are **shared**:
   `260309` has one SUP basecall serving two experiments. On raw, the same fact
   would be restated once per descendant raw generation, and two restatements can
   disagree. On the basecall, it is stated once where it is true.
3. SRB already treats selection as basecall identity, not downstream identity —
   "Basecall configuration is separate from downstream configuration" (line 280),
   and `selection result ID` / `source-signal resolution ID` are already basecall
   identity components. `generation_kind` is a one-word summary of the selection,
   so it belongs with the selection.

**What raw keeps**, because it is genuinely raw's and not the basecall's: `origin
experiment_uid`, `parent raw generation ID`, `parent preprocess generation ID`,
`selection result ID`, and the origin-to-descendant identity map. Only
`generation_kind` moves.

**Legacy default.** A flat `<run>/basecalls/*.bam` with no manifest reports
`generation_kind: full_source` — no selection exists, so the whole source was
called. This keeps the migration bullet below a read-side rule with no rewrite.

#### D3 — Infer legacy names, validate every modern identity source

**Decision.** `experiment_id` is the canonical human label. Existing configs
that supply only `experiment_name` promote that value without changing output
names; a config with neither field warns and derives the output-directory name.
The time-dependent `YYMMDD_SMF_experiment` default is removed. If both config
fields are supplied, they must match.

For a modern `project add`, manifest identity, experiment-directory name, and
explicit `--id` must match before the registry is changed. A directory-only
fallback remains available with a warning for old stores without identity
metadata. Legacy monolithic H5AD files retain the explicit `--id` override used
for migration. Naming disagreement is an error, not a warning.

**Delivery split.** EGL-04a owns derivation, persistence, and read-side
validation. EGL-04b owns transactional `rename-id` across the manifest and
project registries, keyed by immutable `experiment_uid`.

### Open items this creates

- **Migration.** `NKG-02` already placed loose BAMs at `<run>/basecalls/*.bam`
  with no `generations/` wrapper. Readers must accept that flat form as a legacy
  basecall (reported like `legacy_in_place`) rather than requiring a rewrite.
  Settled in principle by `D2`'s legacy default; still needs implementing
  wherever basecalls become the fifth generation kind.

---

# Part A — smftools changes (`EGL-*`)

## Verified findings

**F1 — Four result kinds have immutable generations; `spatial`, `hmm`, and
`chimeric` do not.**

| Has `generations/` + `current.json` | Module |
|---|---|
| raw | `informatics/raw_generation.py` |
| preprocess | `preprocessing/preprocess_generation.py` |
| latent | `tools/partitioned_latent.py` (`LATENT_GENERATIONS_SUBDIR`) |
| project embeddings | `project/embedding_store.py` (`GENERATIONS_DIRNAME`) |

`variant` is not a gap — it was folded into preprocess (commit `6542ec1`, and
the CLI now calls `variant` a "deprecated alias for integrated preprocess
variant"), so it inherits preprocess generations. The real gaps are **`spatial`,
`hmm`, and `chimeric`**, which publish **in place**: re-running `hmm` overwrites
the exact artifact a paper cited, with no way to address the prior result.

They do *not* mutate the preprocess store — `tools/partitioned_hmm.py` and
`tools/partitioned_spatial.py` write their own zarr stores under their own
output directories and record only a relative pointer back
(`hmm_source_spine` / `spatial_source_spine`). So this is a missing capability,
not a correctness bug in the existing generations.

**Why the split exists (evidence, not inference).** Each completed program built
generations for the stage *it* needed: raw generations came from the input
ingestion/alignment program (IAR), preprocess generations from the semantic-DAG
variant program (SDV — whose plan states "Adding a missing analysis means:
validate the current source generation; plan compatible, missing, stale, and
blocked nodes"), latent and embeddings from the project/latent program (PL). The
governing principle is stated most clearly in `embedding_store.py`: *"Each fit or
extension is published as an immutable generation beneath that definition."*
Generations exist where a result is **fit or extended** rather than recomputed
wholesale. Nothing documents a decision that `spatial`/`hmm` should be excluded —
they simply have not had a program that needed it.

**F2 — Nothing lists, retains, or prunes generations.**
No prune/GC/retention command exists; every `shutil.rmtree` in
`informatics/`, `pipeline/`, `cli/` is staging or tempdir cleanup. Generations
accumulate indefinitely, and there is no way to enumerate what exists, how large
it is, or what is safe to delete.

**F3 — Experiment identity is derived by fallback, never validated.**
`project/registry.py:433`:
```python
exp_id = str(experiment_id or meta.get("experiment") or run_root.name)
```
`meta["experiment"]` is the config's `experiment_name`, which itself defaults to
`f"{date_str}_SMF_experiment"` (`config/experiment_config.py:1511`). Nothing
checks that run-directory name, `experiment_name`, and registry id agree.
Observed in the live tree: the DAFseq runs agree (directory name = manifest
`experiment`), while the EMseq set does not — several have a bare date as the
directory and a longer descriptive `experiment_name`, or vice versa. Registering by directory vs. by config silently yields different
ids for the same data.

**F4 — Project analysis caches carry no code identity.**
`project/sample_analysis.py:59`:
```python
def _definition_hash(definition: dict) -> str:
    encoded = json.dumps(definition, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
```
The definition is analysis parameters only (layer, start, end, method, kwargs).
Cached at `analyses/<analysis_name>/<definition_hash>/`. If an algorithm changes
but the parameters do not, the stale cache is silently reused.
`project/embedding_store.py` is marginally better — its definition includes
`molecule_identity_schema_version` — but sample analyses include nothing.

**F5 — `SEMANTIC_GRAPH_DEFINITION_VERSION = 1` is a single global integer**
(`constants.py:23`), fed to `CompatibilityPlanner` (`pipeline/compatibility.py:220`).
It is the only coarse lever: bumping it invalidates every node in every
experiment indiscriminately, so in practice it will not be bumped, so
`STALE_ALGORITHM` under-reports.

**F6 — `experiment_manifest.json` records no smftools version.** Version lands
only in `software_versions.json` (written on the `experiment run` path only) and
in `adata.uns["smftools"]`. A store produced by `experiment full` carries no
top-level record of the code that made it.

**F7 — Stage compatibility hashes the whole config, so any field invalidates
every stage.** Observed live: adding `spatial_generate_position_matrices` — a
spatial-only plotting toggle — to a config whose raw and preprocess generations
were already published moved `experiment.raw.complete` to `stale_config`
("declared semantic configuration changed") and preprocess to
`dependent_recompute`. Changing a plot flag should not invalidate basecall
ingestion. This is the same over-broad-invalidation family as F5 and makes
iterating on a large run expensive: on `260420` (1.57M reads, 10 GB BAM) a
plotting tweak would force a full raw recompute.

**F8 — The basecalling intermediate key is already experiment-independent.**
`IntermediateSpec.compatibility_payload()`
(`informatics/raw_intermediate_manifest.py:193`) covers operation, input artifact
checksums, `operation_config`, and tool-version policy — no experiment name,
output directory, or modality. Only `_revision_manifests` (line 225) scopes the
*lookup* to the requesting experiment's `output_directory`. See EGL-09 and the
SRB reconciliation section.

**F9 — `position_valid` is computed against the unfiltered read population,
before read/modification QC and dedup exist.** Found 2026-08-17 while running
the `241213` pilot under current `main`; it is why that run's `latent` stage
failed and why `position_valid` is `False` for all 18,746 positions.

`reduce_partial_coverage` (`preprocessing/partitioned_executor.py:412`) computes

```python
valid_count_partial = np.sum(~np.isnan(coverage_matrix), axis=0)   # line 292, every read
valid_fraction      = valid_count / reference_plans[ref]["n_reads"]  # line 469, planned reads
position_valid      = valid_fraction >= 1 - cfg.position_max_nan_threshold
```

Both sides are the *unfiltered* read set. It runs at line 1152, before
`append_modification_qc_mask` (1159) and `reduce_duplicate_reads`, so it
structurally cannot filter — the QC columns do not exist yet.

Measured on `241213`, positions clearing the 0.8 threshold
(`position_max_nan_threshold = 0.2`):

| reference | read set | n | positions >= 0.8 | max fraction |
|---|---|---|---|---|
| `6BALB_cJ_top` | all reads *(what the code uses)* | 9,253 | **0** | 0.670 |
| | `passes_read_qc` | 4,172 | 1,341 | 0.842 |
| | `passes_dedup` | 1,929 | **1,361** | 1.000 |
| `6B6_top` | all reads *(what the code uses)* | 9,885 | **0** | 0.569 |
| | `passes_read_qc` | 4,566 | 0 | 0.676 |
| | `passes_dedup` | 1,932 | **1,364** | 1.000 |

About 1,360 positions per reference — ~29% of each ~4,690-position reference —
are measured in >=80% of the reads that survive QC and dedup. Against the
unfiltered population, zero are. `X` is 83.5% NaN because most reads cover only
part of the amplicon, and the reads covering least are largely the ones QC
discards: they dilute every position and are then thrown away.

**Why latent surfaced it and other stages did not.**
`partition_read.py:777` copies `position_valid` into `var` as
`position_in_<reference>`. Latent is the stage that *fails loudly* on it, being
the only one needing a *shared dense* position set for PCA/NMF/CP;
`partitioned_spatial.py:1570,1621` and `partitioned_hmm.py:1063` pass
`min_position_valid_fraction=None` explicitly and work from per-read measured
values, which is why clustermaps look correct.

> **Correction (2026-08-17).** An earlier revision of this finding claimed
> latent was the *only* consumer. That is wrong, and the error understated the
> severity. `position_in_<reference>` is read by
> `preprocessing/duplicate_detection_dispatch.py:128`,
> `cli/chimeric_adata.py` (8 sites), `plotting/variant_plotting.py` (4),
> `machine_learning/data/partition_dataset.py`, and
> `preprocessing/preprocess_adata.py:732`. Every one of them ANDs it with a
> site-type mask, i.e. asks a *membership* question of a column holding a
> *coverage* answer. The consequence is `F11`.

**Not a tuning question.** Loosening `position_max_nan_threshold` would let
latent run, but compensates for the wrong denominator and simultaneously
loosens `tools/position_stats.py`, which uses the same knob for a genuinely
different purpose. Note also that `6B6_top` still yields zero positions after
`passes_read_qc` alone and only clears under `passes_qc`/`passes_dedup`, so a
fix must use the modification-QC/dedup population, not merely read QC.

**F10 — Directory checksums counted operating-system metadata, so opening a
published generation in Finder invalidated it.** Found 2026-08-17: the `241213`
re-run aborted with `preprocess generation artifact is missing or corrupt:
plots` before reaching any pipeline work.

The `plots` directory's recorded digest was
`e89002a7…7ea073a0`; live it hashed `0456c7a3…b95871ea`. The cause was one file:

```
plots/.DS_Store          2026-08-17 09:32:44   <- macOS Finder, 25 min earlier
plots/*  (120 files)     2026-08-14 17:49:17   <- publication time, untouched
```

Nothing scientific had changed. `_sha256` hashed every file under the directory,
so a dotfile the OS wrote while someone browsed to look at clustermaps made an
immutable artifact fail validation. Deleting it restored the recorded digest
byte-for-byte, which is itself the proof that the artifact was never damaged.

**Severity is about frequency, not blast radius.** These projects live on a Mac
desktop; `find runs -name .DS_Store` returned 239 hits across the tree. `NKG-03`
regenerates 20 runs and every published generation is validated on read, so this
would have recurred at unpredictable moments, presenting each time as data
corruption.

**Fixed** by `constants.is_os_metadata`, applied to every directory content hash
that guards artifact validity: `experiment_manifest._sha256` (generations),
`raw_intermediate_manifest.artifact_checksum` (raw intermediates *and* Dorado
model bundles, so `SRB-04a` model identity is covered), and both
`cli/workflow_contract` sites. A digest must describe the artifact, not which
platform last browsed it.

**Migration risk, checked rather than assumed.** Ignoring a file changes the
digest of any tree that contained one *at publication time*. No published
generation in this project contains OS metadata (verified: zero hits under
`runs/*/store/*/generations`), so nothing already recorded is invalidated here.
A tree published elsewhere with a `.DS_Store` inside it would flip once, and
that is the correct trade.

**Still open — diagnosability.** The failure said `missing or corrupt: plots`
and named no file, which is why establishing "one OS dotfile" took a checksum
comparison and an `mtime` sweep. Reporting which path differs would make the
next such failure self-explanatory.

**F11 — `position_in_<reference>` promised structural membership and delivered
coverage density, silently disabling duplicate detection.** Found 2026-08-17
while asking why the `241213` complexity analysis reported implausibly high
library complexity. This is the same defect as `F9` seen from its quiet side,
and it is the more serious half: `F9` crashed, `F11` produced a wrong number.

Duplicate detection reported **0 duplicates in 19,328 reads** — every cluster a
singleton, `passes_dedup == passes_qc == 3,861`. Not a threshold being too
strict: no comparison ever ran.
`duplicate_detection_dispatch._build_duplicate_detection_context_mask` ORs the
configured site-type columns, then ANDs `position_in_<reference>`. That column
held `position_valid`, which was `False` everywhere (`F9`), so the mask was
empty, the function returned `None` for every chunk, and each read was recorded
unique. No error, no warning — the run looked clean and the complexity estimate
was inflated.

The signal was there the whole time: 154 candidate sites, a median 113 measured
sites per read, and **100% of 63,190 candidate pairs** clearing the
`min_overlap_positions >= 20` gate.

**The design error.** Every consumer ANDs `position_in_<reference>` with a
site-type mask, which is a structural question — *does this column belong to
this reference*. The column was assigned a statistic answering a different
question — *is this column densely covered*. The name described the first, the
value carried the second, and the two agree only when coverage happens to be
good. `F9`'s fix (`EGL-11`) corrected the *denominator* but kept the conflation,
so it would have left `F11` in place.

**Why a global gate was the wrong instrument for dedup anyway.** Sparsity in
read-pair comparison is already handled where it belongs — per *pair*, by
`min_overlap_positions`, which asks whether these two reads share enough
measured sites to be compared. A position-level density gate answers a
population question and, when it fails, fails closed and silently.

**Fixed** by `EGL-13` (below), which separates the two meanings rather than
retuning either. `EGL-13` was necessary but *not sufficient*: dedup stayed at
zero afterwards for the independent reason recorded as `F12`.

**F12 — A staging spine pointed at artifacts that did not exist yet, so
duplicate detection silently compared nothing.** Found 2026-08-17 by re-running
the `241213` pilot after `EGL-13` merged and seeing `is_duplicate` still 0 for
all 19,328 reads — byte-identical to the run before the fix.

`staged_generation` builds a preprocess generation in `.staging/<id>/` and only
moves it to `generations/<id>/` at publish.
`execute_partitioned_preprocessing` stamped the *publication* paths into the
spine's uns pointers before that move, then handed that spine to
`reduce_duplicate_reads`. Every pointer named a directory that did not exist
yet. `_overlay_preprocess_var` returned quietly on a missing catalog, so the
dedup slice arrived with no site-type columns and no `position_in_<reference>`,
the comparison mask was empty, every chunk returned `None`, and every read was
recorded unique.

The duplicates were real and abundant. On `6BALB_cJ_top`'s largest sample group,
657,886 read pairs share at least the required 20 comparison positions and 10.7%
of a 20,000-pair sample fall under the configured 0.07 threshold, minimum
distance 0.0 (exact matches).

This is the same hazard class as the `EGL-01` pointer leak, inverted: there the
staging path leaked into a published manifest; here the publication path leaked
into a spine still being built. Both come from writing a pointer at a moment
when it does not yet describe reality.

**Fixed** by `EGL-14`. Validated on real data: `is_duplicate` 0 → 2,350,
`passes_dedup` 3,861 → 1,511, 2,983 clusters of size > 1 (max 39).

**F13 — `bool("False")` is `True`, so every HMM clustermap plotted nothing.**
Found 2026-08-17 while checking the pilot's outputs; every HMM generation on
disk back to 2026-08-14 has an empty `plots/clustermaps` tree.

`chimeric_variant_sites` round-trips through parquet as a *category of the
strings* `"True"`/`"False"`. The clustermap filters negated it with
`~s.astype(bool)`; a non-empty string is truthy, so every read was treated as
chimeric, every group logged `No reads for <barcode> after filtering`, and the
output directories were created but left empty. Live because
`omit_chimeric_reads: TRUE` is set in this experiment's config.

Plotting-only: no analysis result, QC decision, or published array was
affected. **Fixed** by `EGL-15`, verified 0 → 15 PNGs on a direct call.

**F14 — Chimera calling flags a read as reference-switching on two or three
discordant bases, with no minimum segment support.** Found 2026-08-17: `EGL-15`
fixed the truthiness bug but the pilot's clustermaps stayed nearly empty, so the
filter itself was examined rather than assumed correct.

`chimeric_variant_sites` is `has_other_reference_segment` from
`preprocessing/variant_evidence.segment_variant_calls` — the *variant caller's*
reference-switching chimera (a `6B6_top` read carrying a `6BALB_cJ_top`
segment), not a deaminase-style chimera. `segment_variant_calls` interpolates a
segment class between consecutive informative sites and takes no minimum-support
threshold, so one isolated discordant base opens a segment of the other
reference.

Measured on the pilot, among the 3,861 reads passing modification QC:

| quantity | value |
|---|---|
| `chimeric_variant_sites_type` = `multi_segment_mismatch` | 3,786 |
| `variant_self_base_count` | ~1,888–2,618 |
| `variant_other_base_count` | **2 or 3** for 3,586 of them |
| `variant_breakpoint_count` | 4 or 6 for 3,609 of them |

So a read is called chimeric on the strength of 2–3 bases out of ~2,300 — about
0.13% — which is indistinguishable from basecall error at this depth. The
consequence is that **100.0% of QC-passing reads (3,860 / 3,861) are flagged
chimeric**, versus 61.0% of all reads and 78.9% of read-QC-passing reads. Any
consumer honouring `omit_chimeric_reads` therefore sees an empty dataset.

The pre-partition `variant` CLI's segment clustermaps show what the correct
distribution looks like: the large majority of reads are `no_segment_mismatch`,
with a handful showing genuine multi-hundred-base segment switches. That is the
shape a fix has to restore.

**Where the floor belongs, measured rather than guessed.** Per read, the longest
run of *adjacent* informative sites calling the other reference:

| longest run | reads |
|---|---|
| 0 sites | 5,955 |
| 1 site | **10,355** ← noise |
| 2 sites | 499 ← cliff |
| 3 sites | 424 |
| 4+ sites | 459 |

A hard edge between 1 and 2, which is what single-site noise looks like. Effect
on the QC-passing chimera rate: 100.0% at a threshold of 1, **3.6% at 2**, 2.8%
at 3, 1.6% at 5. Two adjacent sites is both the user's requirement and the
empirical cut.

**Fixed** by `EGL-16` — `variant_chimera_min_adjacent_sites`, default 2. Not yet
reflected in any generation: the flag is recomputed at preprocess time, so the
pilot must be re-run before its chimera numbers mean anything.

**F16 — The partitioned variant caller lost conversion awareness, so
C/T-distinguishing sites miscall at up to 71%.** Found 2026-08-18 by comparing
the new variant segment clustermaps against the pre-partition ones in
`projects/Nkg2a_EMseq_sup_merged/241213/outputs/variant_adata_outputs`.

**Not the FASTA.** The reference files are byte-identical between the project
and the run (`6B6` 4690, `6BALB_cJ` 4683). The old log's "4674 / 4669 bases" are
aligned base-column lengths, not the reference.

`resolve_variant_reference_set` (`variant_reporting.py:118`) builds the set with
`sequence_sources` only -- never `converted_sequence_sources`, leaving
`conversion_semantics="none"` -- on an experiment whose modality is
`conversion` with `conversion_types: [5mC]`. Nothing in the tree passes that
argument; the parameter exists and has no caller. The legacy CLI did pass it
(`cli/variant_adata.py:322`, "Using converted columns for variant calling").

The informative-site policy is `disjoint_accepted_bases_substitutions_only`.
With conversion awareness a `6B6=C` / `6BALB=T` site has accepted bases
`{C,T}` vs `{T}` -- not disjoint, so the site is *excluded*. Without it the
comparison is `{C}` vs `{T}`, the site is admitted, and every read whose C was
converted reads as T and is called as the other reference.

Measured on `241213`, reads aligned to `6B6_top`, share called `6BALB_cJ_top`:

| site | 6B6 | BALB | % called BALB |
|---|---|---|---|
| 1344 | C | T | **46.6%** |
| 2054 | C | T | **70.2%** |
| 2519 | C | T | **70.7%** |
| 2047 | T | C | 0.7% |
| 2646 | T | C | 0.6% |
| 3435 | T | C | 1.0% |
| 15 others | -- | -- | 1.5-8.7% |

The asymmetry is the proof. `C->T` conversion can only manufacture a *false
other-reference* call where the read's own reference has the C; the reverse
orientation (`6B6=T`, `6BALB=C`) is protected and sits *below* baseline. G/A
pairs sit at baseline too, confirming top-strand reads are unaffected by them.

Old versus new, same experiment:

| | pre-partition | partitioned |
|---|---|---|
| informative sites with calls | 15 | 22 |
| **C/T pairs among them** | **0** | **6** |
| max spurious other-reference rate | 5.1% | **70.7%** |

Zero C/T sites in the old run is the conversion policy working: it excluded
exactly the sites that now dominate.

**This is the root cause of `F14`**, which `EGL-16` treated at the symptom. The
100%-chimeric rate came from three sites flipping most reads, and the
min-adjacent-sites floor suppressed the *consequence*. Both fixes are wanted --
the floor is still right for genuine single-site noise -- but once conversion
awareness is restored those six sites leave the catalog and the chimera rate
should be reconsidered from scratch rather than assumed to be the 3.6% measured
under `EGL-16`.

**The legacy behavior is not a sufficient target (checked 2026-08-18, before
implementing).** Restoring what the pre-partition CLI did would fix C/T and
leave the mirror case open: it modelled *only* `C->T`. In the old run's var,
accepted bases are `C,T` at C positions but plain `G` at G positions, and the
consequence is visible in its own output -- 196 C/T positions, 0 informative;
213 G/A positions, **8 informative**.

Both directions are live in this very experiment. `Read_mismatch_trend` is
`G->A` for 8,189 reads against `C->T` for 1,290, and spurious other-reference
calls at G/A sites track it: 5.8% for `G->A`-trend reads against 0.3% for
`none`, a 19x enrichment. Small in absolute terms here only because the library
resolves predominantly one way; in a deaminase library, where a molecule can
carry `C->T` on one strand and `G->A` on the other -- and can switch mid-read,
which is why `strand_switch_position` and `strand_segment_purity` exist -- it
would dominate.

The full conversion map already exists in
`informatics/fasta_functions._convert_FASTA_record` and covers all four cases:

| modification | top | bottom |
|---|---|---|
| 5mC | `C->T` | `G->A` |
| 6mA | `A->G` | `T->C` |

Cost of each policy, measured on the pilot's 22 sites:

| policy | sites kept | dropped |
|---|---|---|
| no conversion awareness (current) | 22 | 0 (6 of them miscalling at 47-71%) |
| legacy, 5mC top only | 16 | 6 |
| **both 5mC directions** | **7** | 15 |
| 5mC + 6mA, both strands | 7 | 15 |

So correctness costs two thirds of the discriminating sites, and 7 sites across
~2,800 bp makes for coarse segments. That is a real trade, not a free fix, and
it is the reason to decide the policy deliberately rather than inherit it.

Two designs, both defensible:

- **Site-level union** -- exclude any site ambiguous under any conversion the
  experiment can produce. Simple, safe, and the honest default for deaminase,
  where a single read can carry both directions. Costs the 15 sites above.
- **Per-read strand-aware** -- choose the applicable conversion from the read's
  own strand, so a top-strand read only forfeits C/T and a bottom-strand read
  only forfeits G/A. Keeps closer to 16 sites. The per-read signal already
  exists (`Strand`, `Read_mapping_direction`, `ct_event_count`,
  `ga_event_count`), but reads that switch strand mid-molecule need per-segment
  acceptance, and those are precisely the reads chimera calling cares about.

Recommendation: derive the conversion set from `cfg` (modality,
`conversion_types`, converted strands) and apply the union by default, with
per-read refinement as a later, opt-in lane. Seven trustworthy sites beat
twenty-two with six broken ones.

**Open.** The raw spine stores only unconverted sequences
(`<ref>_FASTA_sequence`), so a fix has to derive the converted forms from
`cfg.conversion_types` as the legacy path did, and pass them plus
`conversion_semantics` into `variant_reference_set_from_legacy`. Until then no
variant, chimera, or segment result from the partitioned pipeline is
trustworthy for a conversion-modality experiment.

**F15 — There is no supported way to re-run a stage after a code fix.** Found
2026-08-17 when `experiment full` exited in one second having done nothing after
`EGL-14`/`EGL-15`: `experiment plan` reported all five stages `compatible`.

Stage compatibility hashes the *config*. A code change does not invalidate it,
`EGL-06`'s `smftools_version` stamp records identity without gating on it, and
neither `experiment full` nor `experiment preprocess` exposes a force flag —
`generations` offers only pin/unpin/prune. The `EGL-13` re-run only recomputed
by accident, because removing the `position_analysis_max_nan_threshold` knob
happened to change the config hash.

Worked around for validation by running into a fresh `output_directory`
(`store_f12f13_validation/`), which forces a full recompute and leaves the
pilot store intact for comparison. That is a workaround, not a fix: it
recomputes stages that did not change and duplicates the store on disk. This is
`F7`/`EGL-10` territory and should be folded into that lane.

**What already works and must not be rebuilt:** `PlanState`
(`COMPATIBLE`/`STALE_CONFIG`/`STALE_ALGORITHM`/`STALE_INPUT`/`DEPENDENT_RECOMPUTE`/…),
the read-only `experiment plan` and `project plan` commands, per-channel
`schema_version`, `compatibility_key`, and `artifact_record`/`resolve_artifact_record`.

## Delivery lanes

### EGL-01 — Consolidate the generation pattern, then extend it to `hmm` (and `spatial`)

> **Status: extension merged (PRs #502–#504); consolidation deferred to `EGL-01b`.**
> The order below was deliberately inverted at implementation time. The shared
> helper was *added* rather than retrofitted: the four existing implementations
> are working, tested, and load-bearing for already-published data, and folding
> them in changes no on-disk layout — so it can follow separately instead of
> blocking the kinds that needed generations now (`1825fe6`'s commit message
> records this). Steps 1–6 below are the as-built spec, with the step-1 caveat
> noted in the status section at the top of this document.

Four independent implementations of the same pattern now exist (F1), each with
its own constants and publish/resolve pair. First factor them into one shared
helper (`informatics/generation.py`: `publish_generation`,
`resolve_current_generation`, `validate_generation`) with the existing on-disk
layouts and schema versions preserved byte-for-byte — a refactor, not a format
change.

Then extend it, in priority order:

1. **`hmm` — strongest case.** Its state calls are the cited biological result,
   and it carries fitted state (`hmm_model_store` in the spine `uns`), so it
   satisfies the "fit or extended" principle directly.
2. **`spatial` — good case.** Autocorrelation/Lomb-Scargle `obsm` feeds figures;
   overwriting destroys the cited artifact.
3. **`chimeric` — defer.** Annotation-shaped and cheap; no evidence it needs
   addressable history.

Legacy in-place stage outputs must keep loading, reported as `legacy_in_place`
rather than failing.

#### Implementation spec for wiring a stage (derived by tracing `hmm`, verified)

The stage executor keeps writing to a single `output_dir`; the caller hands it a
staging directory instead of the stage root. Six steps, of which 1 and 2 are the
non-obvious ones:

1. **Generation-aware run-root recovery.** `tools/partitioned_hmm.py:1335` does
   `run_root = output_dir.parent`, then encodes six `uns` pointers against it
   (`hmm_source_spine`, `hmm_catalog`, `hmm_store`, `hmm_model_store`,
   `hmm_fit_catalog`, `hmm_fit_selection`, `hmm_read_index`). Under a generation
   the stage output is *three* levels below the run root, not two, so every one
   of those would encode a `.staging/<id>/…` path. The reader side is already
   correct — `partition_read._run_root_from_spine_path` (line 96) and
   `_run_root_from_source_base` (line 109) both already special-case
   `{"generations", ".staging"}`. Use the latter; do not reinvent it.
   `partitioned_spatial.py:1831` has the same shape.
2. **Move `write_experiment_spine` out of the executor.** Generation-bearing
   stages already call it *after* publication — latent at
   `cli/latent_adata.py:346`, raw at `cli/raw_adata.py:1917`, preprocess at
   `preprocess_generation.py:668`. Non-generation stages call it from inside the
   executor (`partitioned_hmm.py:1352`, `partitioned_spatial.py:1831`). Left in
   place, a generation that later fails validation would already have been
   unioned into the superset experiment spine.
3. Wrap the executor call in `staged_generation(...)`, passing
   `staged.staging_dir` as its `output_dir`.
4. Publish a canonical `spine.h5ad` at the stage root after the move, as latent
   does via `_atomic_publish_spine`, so existing readers of `paths.hmm_spine`
   keep working. That copy sits two levels below the run root, so its
   run-root-relative `uns` pointers resolve unchanged.
5. Add `generation`, `current`, `generation_manifest`, and `generation_spine` to
   `PARTITIONED_STAGE_REQUIRED_ARTIFACTS[<stage>]` in `constants.py` — these are
   exactly the four keys that distinguish raw/preprocess/latent from
   spatial/hmm today.
6. Remap the executor's returned `outputs` paths from the staging directory to
   the published generation directory before handing them to
   `publish_stage_outputs`.

Validators should resolve everything **relative to the generation directory**
(latent's `_validate_latent_generation` is the model: it rejects absolute or
escaping paths and checks content hashes), which is what makes the tree
survive `os.replace` intact.

**Known complexity this introduces, which must be designed for, not discovered:**
`spatial` and `hmm` are *sibling* branches off preprocess, which is already why
the generated `experiment_spine_outputs/spine.h5ad` superset cache exists (it
unions every stage spine's `uns` because no single lineage spine can resolve
both). Once those siblings each have multiple generations, the superset spine
must decide *which* generation of each to compose — i.e. a per-experiment
generation-set selector.

**Resolved by `D1`:** adopt SRB's lineage as the generation set, ship no new
selector, and make the composition read resolved generations with per-stage key
ownership. The concern was well-founded but understated — the union is not merely
unable to pick a generation, it currently propagates *stale* sibling pointers
between spatial and hmm. See `D1` for the mechanism and the fix. This work is
`EGL-01b`, not a blocker for what shipped.

### EGL-01b — Fold the legacy four onto the helper, and fix spine composition
Two pieces, in this order:

1. **Composition first — merged in PR #505** (it fixes a live defect, per `D1`). Add
   `resolve_stage_generation(stage_dir, lineage=None)` and rewrite
   `write_experiment_spine` to compose over resolved generations with per-stage
   key ownership. Acceptance: an experiment whose spatial generation is newer
   than its hmm generation resolves spatial's *current* pointers in the
   experiment spine — the case that fails today.
2. **Consolidation second — merged in PR #506.** Retrofit `raw`, `preprocess`, `latent`, and project
   embeddings onto `informatics/generation.py`, preserving each on-disk layout
   byte-for-byte, including latent's omission of `manifest_sha256` from the
   pointer. Pure refactor: no format change, no migration, and the existing
   tests for each kind are the regression gate. Do it after (1) so a composition
   bug and a refactor bug cannot be confused for one another.

### EGL-02 — `smftools experiment generations list`
Read-only enumeration, no writes: stage, generation id, created-at, source kind
(`pod5`/`bam`/`fastq`), input manifest digest, label, size on disk, artifact
count, `is_current`, and validation state. `--json` emits the stable schema.
Add `project generations list` fanning out across registered experiments.
This alone converts retention from aspiration to policy.

### EGL-03 — Retention, pinning, and pruning
Published generation manifests are immutable and may be checksum-bound by
`current.json`, so retention must **not** be written into them. Add an atomic
`retention.json` registry beside each stage's `current.json`, recording one or
more reasons a generation must survive (DOI, SRA accession, "paper fig 3").

Split delivery into two phases:

1. **EGL-03a — retention and planning.** Add reasoned pin/unpin, surface pin
   state in generation inventory schema v2, and make
   `smftools experiment generations prune` a read-only policy planner.
2. **EGL-03b — deletion.** Add an explicit apply path only after basecall/input
   provenance and code identity can prove whether a generation is
   byte-reproducible. Keep force behavior out of EGL-03a.

The planner:
- refuses to touch `current` or any pinned generation;
- supports `--keep-last N` and `--older-than`, and is always dry-run in EGL-03a;
- refuses any generation not byte-reproducible from retained inputs unless
  the future EGL-03b explicitly permits it — specifically, a pod5-sourced
  generation is *not* reproducible from a retained BAM and must outrank it;
- prints policy-candidate bytes and zero safely reclaimable bytes while deletion
  remains blocked.

Deletion stays unavailable in EGL-03a; the default posture remains "keep".

### EGL-04 — Canonical experiment identity
Introduce a single derivation with validation instead of a silent fallback chain:
- `experiment_id` is explicit in config (new field) or `--id`; the
  directory-name fallback warns loudly.
- `experiment_name` must equal `experiment_id` or the config is rejected with a
  message naming both.
- Drop the `f"{date_str}_SMF_experiment"` auto-default, or make it a hard error
  when a run is registered into a project.
- `project add` errors on a mismatch between the manifest `experiment`, the
  directory name, and `--id`, rather than picking one.
- Provide `smftools experiment rename-id` that rewrites the manifest and any
  registry entries consistently, keyed on the durable `experiment_uid`.
Constraint: `experiment_uid` remains the identity of record; `experiment_id` is
the human label. Renaming must never mint a new uid.

Delivery is split: EGL-04a establishes the canonical config/manifest/registry
read and validation contract; EGL-04b adds transactional `rename-id`. EGL-04a
merged in PR #511 (`d0ffaa3`; main `3434bb3`). EGL-04b is implemented on
`feature/transactional-experiment-rename`: it preflights every target before
writing, records a durable prepared/committed recovery journal, rolls back exact
file bytes and directory moves on failure, preserves `experiment_uid`, records
rename history, and updates explicitly supplied project registries, list sets,
and per-sample state. Query-set SQL and immutable generations remain historical.
Validation: 6 focused transaction tests; 88 combined identity/registry/project
tests; full unit suite 1,923 passed, 8 skipped, 178 deselected, 7 xfailed; Ruff
check/format and warning-strict Sphinx build clean.

### EGL-05 — Code identity in analysis cache keys
Extend `_definition_hash` inputs in `project/sample_analysis.py` (and audit
`embedding_store.py`, `set_store.py`) to include an explicit
`algorithm_version` per analysis plus `SEMANTIC_GRAPH_DEFINITION_VERSION`.
Bumping one analysis's `algorithm_version` must invalidate only that analysis.
Migration: existing cache directories are keyed by the old hash and simply miss;
document that the first run after upgrade recomputes, and provide
`project analyses list --stale` so the cost is visible in advance.

### EGL-06 — Stamp code identity into the experiment manifest
Add `smftools_version`, `graph_definition_version`, and `git_commit` (when
available) to `experiment_manifest.json` at every stage completion, not just on
the `experiment run` path. `update_experiment_manifest` already merges arbitrary
top-level keys atomically, so this is additive and schema-safe.

Implemented on `feature/experiment-manifest-code-identity` through the central
`record_stage_state(..., state="complete")` path. The identity remains that of
the latest successful publication across planned, running, failed, and
superseded attempts. Validation: 28 focused manifest tests; 86 combined
manifest/workflow/graph tests; full unit suite 1,905 passed, 8 skipped, 178
deselected, 7 xfailed; Ruff check/format and warning-strict Sphinx build clean.

### EGL-07 — Upgrade-impact reporting
Extend `experiment plan` / `project plan` with `--against-installed` (or
`--upgrade-impact`), answering "if I run under the current code, what changes?"
Group output by `PlanState`, name the triggering node and the reason, and report
estimated recompute cost. Purely read-only, built on the existing
`CompatibilityPlanner` — no new invalidation logic, just surfacing it.

Implemented on `feature/upgrade-impact-reporting` with `--upgrade-impact` as a
schema-versioned projection over the existing semantic plan. Human and JSON
reports group every decision by `PlanState`, distinguish trigger, dependent,
blocked, and compatible roles, preserve `invalidated_by`, and report timing
coverage per node. Experiment estimates sum valid nonnegative historical
`elapsed_seconds` observations from the manifest, including the prior complete
record behind an interrupted attempt. Missing or malformed observations remain
unknown. Project materializations are task-local and project caches have
definitions more detailed than the coarse plan request; treating them as
generic current results would add invalidation semantics, so their cost remains
explicitly unavailable in this slice. Validation: 59 focused graph/CLI tests;
85 pipeline/workflow tests; full unit suite 1,927 passed, 8 skipped, 178
deselected, 7 xfailed; Ruff check/format and warning-strict Sphinx build clean.

### EGL-08 — Documentation and acceptance
Extend `docs/source/tutorials/directory_organization.md` with a generations and
retention section, add a "managing analyses across smftools versions" tutorial,
and record acceptance criteria in `tests/acceptance/` in the established
JSON-criteria form. Acceptance must include: a post-preprocess stage re-run
producing a second generation with the first still resolvable; a prune dry-run
refusing a pinned generation; a naming mismatch rejected by `project add`; and
an `algorithm_version` bump invalidating exactly one analysis.

### EGL-09 — Shared run-level basecall reuse *(candidate; not yet agreed)*
Per F8 the key already matches across experiments; only the lookup is
experiment-scoped. Add an ordered list of read-only shared intermediate roots
(default `<run>/intermediates/`, beside `basecalls/`), searched on miss, with the
hit's origin recorded in the consuming experiment's provenance.
`validate_intermediate_commit` already re-verifies operation, key, full
compatibility payload, state, and artifact checksums, so a foreign or stale hit
is rejected by existing code — this adds discovery, not trust.

The payoff is provenance, not just compute. Pointing a second experiment at the
first's BAM works today but gives it `input_type=bam`, forfeiting every `pod5_*`
column and the `dorado-basecalling` commit record — exactly the SRA metadata that
then has to be reconstructed by hand. With reuse, the second experiment declares
**pod5** as its input, keeps full provenance identity, and skips only the compute.

Boundary with SRB: SRB owns *re-basecalling* (selective, from pod5, producing
descendant lineages). EGL-09 owns *not* re-basecalling — reusing an existing
basecall across sibling experiments. Depends on the reconciled layout above.

### EGL-11 — Position validity against the analysed read population *(F9)*

> **Superseded by EGL-13 and reverted 2026-08-17.** EGL-11 was implemented as
> written below and then rolled back, because `F11` showed its premise was
> wrong. It fixed the *denominator* of a statistic while leaving that statistic
> wired into a column whose name promises structural membership — so duplicate
> detection would have stayed silently disabled. It also added a second
> precomputed, published, fallback-requiring column, which is cost paid for the
> architecture that caused the defect. Kept here because the reasoning about
> ordering (below) remains correct and constrains any future work.
>
> Removed in the revert: `reduce_analysis_coverage`, `analysed_read_population`,
> `ANALYSIS_COVERAGE_COLUMNS`, the `position_analysis_max_nan_threshold` knob,
> the `position_valid_analysis` / `*_analysis` var columns, and
> `tests/unit/test_position_validity_analysis.py`.

Fixes the defect that made the `241213` pilot's `latent` stage fail. `latent` is
the only consumer, but the quantity it consumes is wrong for every run, so this
is a correctness lane rather than a latent workaround.

**The constraint that shapes the fix.** The obvious repair — move
`reduce_partial_coverage` after dedup — is circular. `reduce_duplicate_reads`
(line 1234) needs a real spine on disk, that spine is built from the var catalog
(line 1152), and the var catalog is what we would be trying to compute after
dedup. The existing staging-spine dance at 1218-1233 exists precisely because
this ordering is already delicate, and it is guarding a real production incident
(`260420`'s dedup never finished, its pre-dedup spine was left on disk, and
every restart silently analysed non-dedup'd data). Do not disturb it.

**Recommended shape: a second pass writing a *new* column.**

1. Keep `reduce_partial_coverage` exactly as it is. Its output still feeds spine
   construction, and `position_valid` keeps its current meaning — "measured in
   >=(1-`position_max_nan_threshold`) of the reads the assay produced" — which is
   the right question for `tools/position_stats.py` and is what every existing
   generation already recorded.
2. After `reduce_duplicate_reads` returns, recompute coverage restricted to the
   analysed population and write `position_valid_analysis` (plus
   `valid_count_analysis` / `valid_fraction_analysis`) into the same var
   catalog. The per-read data needed is already on disk: task stores persist a
   `covered_base_mask` layer, so this is a masked re-sum over existing
   artifacts, not a recomputation of the stage.
3. Population: `passes_dedup` where present, else `passes_qc`. **Not**
   `passes_read_qc` — measured on `241213`, `6B6_top` still yields zero
   positions at 0.8 after read QC alone and only clears under `passes_qc`.
4. Threshold: a distinct knob, `position_analysis_max_nan_threshold`, defaulting
   to `position_max_nan_threshold`. `F9` is partly a story about one knob
   serving two meanings; fixing the denominator while keeping the conflated knob
   would leave half the defect in place.
5. Point `_build_reference_position_mask` (`cli/latent_adata.py:185`) and the
   mask builders it feeds at `position_valid_analysis`, falling back to
   `position_valid` when the column is absent.

**Why a new column rather than redefining the old one.** `position_valid` is
recorded in every published preprocess generation. Silently changing what it
means would make old and new generations incomparable while both claim the same
schema, which is exactly the failure this program exists to prevent. A new
column is additive: old generations lack it and say so.

**Migration.** A pre-EGL-11 generation has no `position_valid_analysis`. Latent
must fall back and say which quantity it used, rather than crashing or silently
using the diluted one. The `241213` pilot is the regression case: it must go
from `RuntimeError: no units meeting latent_min_reads` to a completed latent
generation over ~1,360 positions per reference.

**Tests**

- A synthetic frame where failing reads dilute a position below threshold while
  the passing reads clear it: `position_valid` False, `position_valid_analysis`
  True. This is the defect, stated as a test.
- Population selection: `passes_dedup` preferred, `passes_qc` fallback, and
  `passes_read_qc` explicitly *not* sufficient.
- A generation lacking the new column still runs latent, on the documented
  fallback, with the choice reported.
- Real-data check on `241213` before/after, since this is a defect only real
  coverage profiles exposed.

**Exit gate**

`241213`'s latent stage completes, and a position whose apparent validity is
decided entirely by reads that QC discards is reported as valid for analysis and
invalid for the assay — each under its own name.

### EGL-13 — Separate structural membership from coverage density *(F9, F11)*

> **Merged 2026-08-17** — PR #535, commit `4fc0e60`, `main` `d862e3a`. The exit
> gate below is *not* met: the `241213` re-run has not been done. Merging the
> fix and confirming the fix are different claims, and only the first is true.

Replaces EGL-11. The defect is not that the coverage statistic was computed over
the wrong reads; it is that a coverage statistic was stored in a column every
consumer reads as a membership fact. Fixing the denominator leaves that intact.

**The principle.** *Membership is a property of the reference. Density is a
property of the read subset you are looking at.* Membership is stable, so it is
precomputed and published. Density is not — it changes with every filter — so it
is computed at the point of use, by the consumer that has the reads in hand.
Precomputing density is what forced a published column, a schema migration, a
fallback path, and a second knob in EGL-11, all to describe a quantity that is
only meaningful relative to a read set the preprocess stage cannot know.

**Changes**

1. `partition_read._overlay_preprocess_var` — `position_in_<reference>` is now
   `positions.isin(frame.index)`: does the reduction have a row for this
   position under this reference. `N_Reference_strand_with_position` follows it.
   The coverage flag survives under its own name, `<reference>_position_valid`,
   alongside the existing `<reference>_valid_count` / `_valid_fraction`.
2. `duplicate_detection_dispatch._build_duplicate_detection_context_mask` — drops
   the density intersection entirely. `min_overlap_positions` is the correct
   guard and acts per pair. Site-type restriction and reference membership are
   unchanged, so the mask stays a real filter.
3. `cli/latent_adata._build_reference_position_mask` — takes
   `minimum_valid_fraction` and measures density from the matrix it is about to
   factorize (`np.isfinite(adata.X).mean(axis=0)`), defaulting to `0.0`
   (membership only) for the four callers that want no density requirement.
   `tools/partitioned_latent.py` passes `1 - position_max_nan_threshold`.
4. EGL-11's machinery removed (see the note under EGL-11).

**Why latent gets correct answers now without a published column.** Latent
filters to its analysis population *before* factorizing, so the matrix in hand is
already the analysed read set — the exact population EGL-11 was trying to
precompute. Measuring there is both simpler and strictly more accurate: it
follows every subset (per reference, per barcode, per unit), which a single
published column cannot.

**Migration: none, and that is checked rather than assumed.** No published
artifact changes. `position_in_<reference>` is written during slice materialization
from the reduction parquet, not stored in it, so existing generations take the
new meaning on read with no regeneration. `position_valid` remains in the
catalog, unchanged, as the descriptive statistic it always was — it is simply no
longer a gate anywhere.

**Tests** — `tests/unit/test_position_membership_and_coverage.py`

- Membership survives when *no* position clears the density threshold. This is
  the defect stated as a test, built from the real overlay rather than a
  hand-written `var`, because the defect was in what the overlay wrote.
- Membership still excludes positions the reference never covered, and dedup
  still restricts to configured site types — the mask must not become a
  pass-through.
- The coverage statistic survives under its reference-qualified name.
- Latent density is measured over the reads being factorized, and follows a
  subset rather than a global flag.
- Verified to fail against `main`: 6 of 9 fail before the change; the 3 that
  pass are the guards against over-correcting.

**Exit gate**

`241213` re-run reports a non-zero duplicate count with a defensible complexity
estimate, and latent completes. Both were previously wrong in opposite
directions from the same cause — and only one of them said so.

### EGL-10 — Scope stage compatibility to fields that stage consumes *(candidate)*
Per F7, hash each stage against the config fields it actually reads rather than
the whole config. Likely a declared field set per semantic node, validated so a
field consumed but undeclared is a hard error rather than a silent
under-invalidation. Higher risk than the rest of the program — under-invalidation
is a correctness bug, not an inconvenience — so it needs its own acceptance
matrix and should not be attempted alongside EGL-01.

## Dependency order

```
EGL-02 (merged) ─> EGL-03
EGL-01 (merged)  ─> EGL-01b ─> EGL-03
EGL-04 ─┬─> EGL-07 ─> EGL-08
EGL-05 ─┤
EGL-06 ─┘
```
`EGL-01` and `EGL-04` should land before SRB implementation starts. `EGL-01` now
has; `EGL-01b` is not an SRB blocker, since `D1` fixes the selector *contract*
that SRB depends on independently of when the composition code lands.

**Corrected after EGL-02 shipped:** the original graph had `EGL-01 → EGL-02`.
That was wrong — listing needed no shared helper, because the four layouts had
already converged on identical naming. EGL-02 shipped standalone. The real
constraint is the reverse: EGL-02's findings now inform EGL-01's design.

**Practical gate, not a code dependency:** the `241213` pilot now supplies a
narrow real generation tree for resolver and composition checks. Broader
retention behavior still needs `NKG-03` to regenerate more experiments under
2.21-dev; designing project-wide pruning from the pilot alone would remain
speculative.

## Non-goals

- Re-basecalling, lineage promotion, model identity — owned by SRB.
- Changing default QC thresholds, dedup, or HMM semantics.
- Automatic deletion of anything. Pruning is always explicit and dry-run first.
- Rewriting existing raw/preprocess generation formats.
- A general provenance database. The manifest plus the registry remain the
  record.

## Decision gates
## Sequencing

`NKG-01` → `NKG-02` → `NKG-06` (get it under git before bulk changes) →
`NKG-03` interleaved with legacy deletion → `NKG-04` → `NKG-05`.

Part A can proceed independently; `NKG-04` is written so that adopting
`EGL-02`/`EGL-03` later is a drop-in replacement rather than a migration.

## Deamination-aware chimera detection and strand-aware SNP calling (planned 2026-08-18)

Four lanes, `EGL-18`..`EGL-21`, arising from `F16`. They share one idea: a
mismatch against the reference is only evidence of *anything* once you know
which chemistry could have produced it, and that is a property of the read's
strand -- and, for deaminase, of the position's neighbourhood within the read.

### Where this belongs: preprocess, not raw

`ragged_store._read_record` (raw) already walks the CIGAR against the reference
and computes per-position deamination votes -- `strand_vote_signs` (+1 `C->T`,
-1 `G->A`) and `strand_vote_positions` -- then discards them, keeping only the
scalars `ct_event_count`, `ga_event_count`, `strand_segment_purity`,
`strand_switch_position` in raw obs.

It is tempting to capture the per-position arrays there. Do not:

1. **It is not needed.** `partitioned_variant._observed_bases` shows the walk is
   fully reproducible at preprocess time from the raw ragged row alone
   (`SEQUENCE` + `CIGAR` + `REFERENCE_START`) plus reference sequences in the
   raw spine's `uns['References']`. The same walk yields deamination state.
2. **Raw is immutable and expensive.** A raw schema change forces re-ingestion
   of every existing raw generation for information that is derivable. None of
   the `241213` work would carry it without a full re-run.
3. **Segmentation is an analysis parameter.** The 3-in-a-row rule is explicitly
   a first pass the user expects to replace. A rule baked into an immutable
   tier cannot be revised without regenerating data; in preprocess it is a
   config knob and a recompute.

So the deamination lane mirrors the variant lane exactly, one tier down from
raw: sparse per-position evidence, then segments, then per-molecule summary,
written as a preprocess sidecar next to `variant/`. Raw is untouched, and the
existing raw generations are sufficient input.

### EGL-18 — Strand-aware conversion site typing *(closes `F16`)*

Per the decision of 2026-08-18: acceptance is chosen by the read's **mapped
strand**, not by a union over all chemistries.

- `top` -> `C->T`, so a reference `C` accepts `{C, T}`
- `bottom` -> `G->A`, so a reference `G` accepts `{G, A}`

The conversion map already exists as
`informatics/fasta_functions._convert_FASTA_record.conversion_maps` and covers
`5mC` (`C->T` / `G->A`) and `6mA` (`A->G` / `T->C`); reuse it rather than
restating the pairs.

Consequence to design around: informative-site status becomes **read-dependent**,
so it can no longer be a single precomputed catalog column. Two sub-parts:

- `EGL-18a` — thread `converted_sequence_sources` and `conversion_semantics`
  into `variant_reporting.resolve_variant_reference_set`
  (`variant_reporting.py:118`), derived from `cfg.smf_modality` and
  `cfg.conversion_types`. Members already carry orientation via
  `_orientation(source_id)`, so a top member and a bottom member get different
  accepted sets from the same catalog build.
- `EGL-18b` — make the per-read call in `call_read_variant_sites` select the
  member acceptance matching that read's `Strand`. A site ambiguous for a
  top-strand read stays usable for a bottom-strand read, which is the whole
  point of choosing per-read over the union.

**Expected effect on the pilot**, from the measurement under `F16`: the three
C/T sites miscalling at 46.6% / 70.2% / 70.7% become non-informative for
top-strand reads. 16 of 22 sites survive for a top-strand read versus 7 under
the union. The chimera rate must be re-derived afterwards -- the 3.6% measured
under `EGL-16` was computed on corrupted calls.

### EGL-19 — Per-molecule deamination segmentation

New preprocess sidecar `deamination/`, structured like `variant/`.

**Per-position state**, from the same CIGAR walk (reference base -> query base):

| reference | query | state |
|---|---|---|
| C | T | `ct_event` (top-strand deamination) |
| C | C | `c_retained` |
| G | A | `ga_event` (bottom-strand deamination) |
| G | G | `g_retained` |
| anything else | -- | uninformative, not recorded |

Only C and G reference positions carry information; everything else is skipped,
which keeps the sparse evidence roughly the size of the variant calls table.

**Segmentation.** Runs of >= `deaminase_segment_min_adjacent_events` (new config
knob, default **3**) consecutive same-type *events* form a segment, counted over
the informative-position sequence rather than over genomic distance -- the same
choice, for the same reason, as `variant_chimera_min_adjacent_sites` in
`EGL-16`. Below-threshold runs stay in the evidence table and are simply not
promoted to segments, so the threshold can be re-derived without recompute and
a future algorithm can consume the raw events.

**Artifacts**, mirroring `variant/task_store/{calls,events}.parquet`:

- `deamination/task_store/**/events.parquet` — `read_id`, `position`, `state`
- `deamination/task_store/**/segments.parquet` — `read_id`, `start`, `end`,
  `state`, `n_events`
- `deamination/deamination_obs/**/*.parquet` — per molecule: segment count per
  state, dominant state, switch positions, purity, and the `EGL-20` chimera call

**Skip for `direct`.** Direct modality has no deamination chemistry and, as the
user notes, no chimeras in general. Gate the whole lane on
`cfg.smf_modality in {"conversion", "deaminase"}` so direct experiments pay
nothing.

### EGL-20 — Chimera classification from segments

Two independent chimera classes, deliberately kept separate because they are
different evidence about the same molecule and either can be absent:

- **Deaminase-detected**: a molecule carrying >= 1 qualifying `ct_event`
  segment *and* >= 1 qualifying `ga_event` segment. Both strands were deaminated
  within one molecule, which a single template cannot produce.
- **SNP-resolvable**: the existing `has_other_reference_segment`, recomputed on
  `EGL-18`-corrected calls.

This supersedes the scalar `label_deaminase_pcr_chimeras.deaminase_chimera_mask`
(`ct_event_count` / `ga_event_count` / `strand_segment_purity`), which infers a
switch from per-read totals and a best two-segment purity rather than from
located segments. Keep the old column for one release, computed alongside, so
the two can be compared on real data before the scalar version is retired --
its `min_events_per_span = 3` default is the same rule this lane implements
properly, so agreement is expected and disagreement is informative.

### EGL-21 — Deamination segment clustermaps

The same figure as the variant segment clustermaps, for deamination state.
`EGL-17` established the pattern: rasterize sparse segments onto a
read x position grid and hand it to the existing renderer.

Reuse `plotting.variant_plotting._plot_variant_segment_one_group` if its
palette and legend can be parameterized cleanly; fork it only if the colour
semantics genuinely differ (four states rather than seq1/seq2/transition/none).
The row annotation strip becomes the deamination chimera class, mirroring
`chimeric_variant_sites_type`. New plot category `deamination_segments`
alongside `variant_segments` in `STAGE_PLOT_CATEGORIES["preprocess"]`.

### Sequencing and risk

`EGL-18` first: it is self-contained, closes a known-wrong result, and its
output is the input to the SNP half of `EGL-20`. `EGL-19` next, then `EGL-20`
which needs both, then `EGL-21` which needs `EGL-19`.

The main risk is that `EGL-18` and `EGL-19` both change what counts as evidence,
so every chimera and variant number measured to date -- including the 60.9%
duplicate rate, which is computed on QC-passing reads and therefore downstream
of these masks -- may move. Re-measure rather than carry forward. `F15` still
applies: none of this invalidates stage compatibility, so validating each lane
means a fresh `output_directory` until that is fixed.

### EGL-20a — Deaminase variant calling must be segment-aware (decided 2026-08-18)

**Yes, and it is a correctness requirement rather than an optimization.**

In conversion SMF the applicable chemistry is fixed for a whole read by its
mapped strand, which is what `EGL-18` exploits. In deaminase it is *positional*:
a molecule can carry `C->T` over one stretch and `G->A` over another -- that is
precisely what makes it a chimera -- so a single per-read acceptance rule is
wrong by construction. Acceptance at position `p` for read `r` must come from
the deamination segment covering `p`:

| covering segment | reference base | accepts | consequence |
|---|---|---|---|
| `ct_event` | C | `{C, T}` | C/T SNP sites ambiguous here |
| `ga_event` | G | `{G, A}` | G/A SNP sites ambiguous here |
| none | -- | canonical only | site fully informative |

The third row is the payoff. A blanket union discards C/T *and* G/A sites
everywhere (7 of 22 survived on the pilot); segment-aware acceptance only
discards them where the corresponding deamination actually happened, and
recovers them everywhere else.

**The confound that must be handled first.** At a C/T SNP site, a genuine
`6B6(C)` -> `6BALB(T)` reference difference is *indistinguishable* from a `C->T`
deamination event at that position. Measured on the pilot: **20 of 22** SNP
informative sites involve a C or G on one reference, so all 20 would be counted
as deamination events if nothing excluded them. Left alone, the two lanes
corrupt each other -- reference identity masquerades as deamination, and
deamination masquerades as reference identity.

**Resolution: an acyclic ordering.** The candidate site list is reference-only
-- `variant_reference.calculate_variant_informative_sites(reference_set)` takes
the reference set and no reads -- which makes this work without iteration:

1. Compute candidate SNP sites from the two references alone. No read data.
2. Compute deamination events **excluding those positions**, so a reference
   difference can never be counted as a deamination event.
3. Segment deamination (`EGL-19`).
4. Call variants at the candidate sites, taking acceptance from the deamination
   segment covering each position.

Step 2 depends only on the reference-derived list from step 1, never on
read-level variant calls, so there is no cycle.

**Why this is not circular for chimera detection.** Strand chimerism and
reference chimerism are orthogonal axes: a molecule can join two strands of the
same allele, or two alleles of the same strand, or both. Using strand chemistry
to remove a confound from reference identity is not double-counting the same
evidence -- it is removing an alternative explanation before attributing a
mismatch to allele identity.

**Residual risk, quantified.** Only 1.2% of the 1,660 deamination-candidate
positions (847 C + 813 G) are SNP sites, so aggregate contamination is small.
The dangerous case is local rather than aggregate: a real reference switch flips
*several* SNP sites at once, which is exactly the clustering that could
manufacture a spurious 3-event deamination segment. On this reference pair no
three consecutive C/G positions are all SNP sites, so it cannot happen here; a
denser pair could, which is the reason step 2 excludes them by rule rather than
relying on it being rare.

**Sequencing correction.** This inverts part of the order given above. For
*conversion*, `EGL-18` remains independent and goes first. For *deaminase*,
`EGL-18b` depends on `EGL-19`, since there are no segments to consult until the
deamination lane exists. Revised order: `EGL-18a` (catalog acceptance) ->
`EGL-18b` for conversion -> `EGL-19` -> `EGL-20a` (deaminase acceptance from
segments) -> `EGL-20` (both chimera classes) -> `EGL-21`.

### Plan readiness audit (2026-08-18)

**Resolved: which signal carries "strand".** The plan said "mapped strand",
which was ambiguous between three different obs columns. Settled empirically:

| candidate | verdict |
|---|---|
| `Strand` | **use this.** Exactly the `Reference_strand` suffix (19,138 top / 190 bottom, 1:1) |
| `Reference_strand` | equivalent; `Strand` is its suffix |
| `Read_mapping_direction` | **not this.** Independent of `Strand` (top reads split 7,227 fwd / 11,911 rev) |

The discriminating evidence: split by `Strand`, C/T sites corrupt at **31.7%**
and G/A at 3.1% -- clean separation, so the rule works. Split by
`Read_mapping_direction`, C/T corrupts at 21.6% (fwd) *and* 38.5% (rev) -- both
directions affected, so it does not discriminate.

This matters because the two are not interchangeable: `ragged_store` derives a
raw `strand` from `read.is_reverse`, but the `Strand` that survives into
preprocess obs reflects the *reference-strand assignment* -- which converted
reference the read aligned to. Conversion direction follows the assignment, not
the BAM flag. `EGL-18` keys on `Strand`.

**Remaining gaps, in rough order of how much they could change the design:**

1. **Catalog serialization under read-dependent acceptance.** If acceptance
   varies by read, `informative_site_catalog.informative_sites[].accepted_bases`
   can no longer be one list. Decide before writing `EGL-18a`: emit one catalog
   per member orientation, or store canonical bases only and resolve acceptance
   at call time. The second keeps a single catalog id and avoids doubling the
   artifact, but moves work into the per-read hot path.
2. **No deaminase data on hand.** `241213` is conversion. `EGL-19`/`20a`/`21`
   have no dataset here to validate against, so they would ship on unit tests
   plus a synthetic fixture until a deaminase run exists. Worth finding a real
   one before building, since the whole lane is motivated by molecules this
   experiment does not contain.
3. **`omit_chimeric_reads` becomes ambiguous.** With two chimera classes, the
   existing flag -- which today gates HMM/spatial clustermaps and the analysed
   population -- has no defined meaning. Needs an explicit decision: honour
   either, both, or a named choice. This is user-facing behaviour, not an
   internal detail.
4. **CLOSED 2026-08-18 — `variant_chimera_min_adjacent_sites` stays at 2.** Measured after `EGL-18`, the floor still does real work: 13.9% at threshold 1 versus 6.9% at 2, so it is suppressing genuine single-site noise rather than the conversion artifact `EGL-18` removed. Threshold 3 gives 4.2%; choosing between 2 and 3 is now a judgment on trustworthy calls rather than noise suppression, and is left open deliberately. Original note: **`variant_chimera_min_adjacent_sites` after `EGL-18`.** `EGL-16` set it to 2
   to suppress a symptom whose cause `EGL-18` removes. The knob is still right
   for genuine single-site noise, but its default must be re-derived on
   corrected calls rather than carried over.
5. **6mA generalization.** `EGL-19`'s state table is 5mC-specific (C/G
   positions). For 6mA the analogous states are `A->G` / `T->C`, so the table
   and the "deamination-candidate position" definition need to come from
   `conversion_maps` rather than being hard-coded to C and G.
6. **Indels.** The variant lane has an explicit `per_read_indel_policy`; the
   deamination walk has no stated position on insertions or deletions
   interrupting a run of same-type events.
7. **Bottom-reference reads.** Only 190 of 19,328 here, but the segment and
   acceptance semantics for reads assigned to a `_bottom` reference should be
   stated rather than left to fall out of the orientation handling.

None of these blocks starting `EGL-18a`; (1) must be decided within it, and (3)
and (4) must be decided before `EGL-20` changes what downstream stages see.

### Decisions closing gaps 2 and 3 (2026-08-18, user)

**`omit_chimeric_reads` is the union.** A read is omitted if it is chimeric by
*either* the deaminase-segment method or the variant/SNP method. This keeps one
user-facing flag with one meaning and does not require callers to choose a
method. Implementation note: the two classes stay as separate obs columns
(`EGL-20`) and the flag consumes their logical OR, so the composite can always
be decomposed when a result looks surprising.

**Test dataset: the smallest native-barcode deaminase run.** Chosen over the six other
native-barcode deaminase runs because it is by far the smallest (121 MB input,
25,098 reads) *and* uses the same `BALBC_B6_NKG2A_mNanog.fasta` reference pair
as the conversion pilot, so the SNP informative-site catalog is
directly comparable across modalities.

It carries the signal these lanes need:

| quantity | value |
|---|---|
| reads | 25,098 |
| `ct_event_count` | median **245** (conversion pilot: 0) |
| `ga_event_count` | median 5, max 342 |
| `Read_mismatch_trend` | `C->T` 23,082 / `G->A` 1,270 / none 689 |
| `Strand` | top 23,828 / bottom 1,270 |
| bottom-*reference* reads | 1,127 (`6B6_bottom` 1,085 + `6BALB_cJ_bottom` 42) |
| `deaminase_PCR_chimera` (current scalar method) | **11** |
| `strand_switch_position` | median -1, max 4,681 |

Three things make it a good fixture beyond size. `Read_mismatch_trend == G->A`
(1,270) equals `Strand == bottom` (1,270) exactly, so the strand-to-direction
correspondence `EGL-18` depends on can be checked against an independent signal
rather than assumed. Bottom-reference reads are present, so gap 7 is testable
rather than theoretical. And it has both a legacy
`outputs/variant_adata_outputs` and a partitioned `outputs_v2.15`, so the same
old-versus-new comparison that diagnosed `F16` is available here.

**Caveat on what it can and cannot show.** The 11 chimeras give a precision
test, not a recall test: with 23,082 reads single-direction `C->T` and only
1,270 `G->A`, genuine within-read switches are rare. `EGL-20` can check that
segment-based detection finds those 11 and can explain any it adds or drops,
but establishing sensitivity needs a larger deaminase run -- one of the 2.7 GB
or 10 GB sets -- once the method is settled. Note the 10 GB run is the one whose
dedup pass over ~1.3M reads
never finished, per the incident recorded in `partitioned_executor`, so budget
accordingly.

**Gap status:** 2 and 3 closed. Remaining open: 1 (catalog serialization,
decide inside `EGL-18a`), 4 (`variant_chimera_min_adjacent_sites` default),
5 (6mA generalization -- note this run sets `conversion_types: [5mC]` despite
being deaminase, so the state table cannot key off modality alone), 6 (indels),
7 (bottom-reference semantics, now testable on this run).

## What is outstanding (2026-08-18)

**No merged lane is waiting on more code.** Everything from `F12` through `F18`
is on `main`. What remains splits cleanly into validation, decisions, and one
planned program.

### 1. Validation — the immediate work, and it is running, not coding

Nine changes have landed since the last pilot outputs were produced, and **not
one of them is reflected in any published generation.** Five change what the
pipeline computes (`EGL-14`, `16`, `18` -- duplicates, chimeras, variant calls)
and four change what it plots (`EGL-15`, `17`, `22`, `23`). Every number quoted
in this document below the `F16` section was measured in a scratch validation
store, never in a canonical one.

Two pilots, deliberately different:

- **`241213` (EMseq / conversion)** into the canonical `store/`. Exercises
  `EGL-18`'s conversion path, the dedup fix, read-span cropping, and -- for the
  first time -- inversion, since `EGL-23` migrated its config off the dead
  `invert_adata`. Expect the figures to look different: the axis will be
  inverted and cropped where it never was before. That is the fix, not a
  regression.
- **`251105` (DAFseq / deaminase)** -- 25,098 reads, native barcoding, same
  reference pair. The only run here that exercises `EGL-18`'s *bottom*-strand
  branch in anger and the `deaminase` modality sheet. It has never been run
  under any of this. It is also the fixture `EGL-19` will need, so establishing
  a clean baseline now pays twice.

Both need a fresh `output_directory` or a config change to recompute (`F15`).
The `241213` config changed under `EGL-23`, so it will recompute on its own;
`251105`'s did not.

### 2. Numbers that need a human, not a run

Now computed correctly for the first time, and none of them a code question:

| quantity | current value | note |
|---|---|---|
| duplicate rate | 60.9% | was silently 0% |
| chimera rate (QC-passing) | 3.1% | was 100% |
| `variant_chimera_min_adjacent_sites` | 2 | 3 gives 4.2%; both defensible on corrected calls |

### 3. Open decisions and known gaps

- **`F18`** -- `from_csv` and `load_experiment_config` resolve config
  differently. Every ad-hoc measurement in this session used the former. None of
  the conclusions depended on a field where they disagree, but that was luck.
  Fix is to default `from_csv` to the packaged `defaults_dir`.
- **`F15`** -- no supported re-run after a code fix. Has now cost five scratch
  stores. Belongs to `EGL-10`.
- **`EGL-03b`** -- destructive pruning, still blocked on byte-reproducibility
  evidence. Unchanged and not urgent.
- Gaps 5-7 from the `EGL-19` readiness audit: 6mA generalization, indel
  handling, bottom-reference semantics. All testable on `251105`.

### 4. `EGL-19`/`20`/`20a`/`21` — the deaminase program

Specced, sequenced, and with a fixture chosen, but **not started**. This is the
only remaining body of new code, and it is substantial: per-molecule deamination
segmentation, two chimera classes, and a fourth clustermap family.

Recommended order: validate on both pilots first. `EGL-20a` consumes
`EGL-18`'s output, so building on top of an unvalidated `EGL-18` would stack a
large speculative lane on an unconfirmed base -- which is exactly the mistake
`EGL-16` made against `F16`, where a threshold was tuned to mask a defect whose
cause had not yet been found.

## EGL-31 — Barcode contamination QC from an unbarcoded spike-in

**Requested 2026-08-23.** Barcodes are ligated to read ends before libraries are
pooled, so a barcode from one library can ligate to a molecule from another. The
resulting read is misassigned and looks entirely normal downstream.

The run carries an unbarcoded **CTCF mNanog** amplicon that makes this
measurable. Pooled without a barcode of its own, its true assignment is known
for every read, so any barcode observed on a spike-in read is a mis-barcoding
event **by construction** -- a direct count of known errors rather than a rate
inferred from a model. It also scavenges free adapter that would otherwise land
on real molecules, which is why one amplicon serves both roles. Those roles are
in mild tension: the better it scavenges, the more it over-reports relative to a
real molecule that already carries its own barcode, so the spike-in figure is an
upper bound on the per-molecule rate.

**Shipped 2026-08-23** (`c481ac8`) as `preprocessing/barcode_contamination_qc`,
run from the preprocess executor. Three measures that fail independently:

| measure | what it answers |
|---|---|
| per-barcode enrichment | which barcodes contaminate *more than their library share predicts* -- raw counts just rank library size |
| single- vs double-ended mislabeling | how much trust a double-ended assignment earns, measured on known errors |
| end disagreement | `barcode_front` != `barcode_rear`: mis-ligation caught in the act, across the whole run |

The third needs no spike-in and is the one that scales -- it applies to every
read with both ends called rather than to the 0.1% that is spike-in. It is a
*floor*: two ends that mis-ligate the same wrong barcode agree with each other
and stay invisible, which is precisely the case the spike-in does see. The two
are complementary, not redundant.

**Poisson intervals throughout.** A per-barcode cell holds tens of reads, where
an enrichment of 1.3 against 1.0 is noise. Byar's approximation is used rather
than a normal one because a Wald interval goes negative at these counts and
stops meaning anything.

**The denominator is the whole spike-in population**, including reads correctly
left unassigned. Counting only barcoded spike-in reads would report 100%
contamination on any input whatsoever. This is why the unclassified population
must be ingested -- and it already was: 17 of the 575 resolved input files, 9.3%
of reads.

**It refuses rather than degrades.** A store written before `F35` has only the
collapsed barcode column; falling back to it would reproduce the vacuous
self-comparison that `F35` removed. The QC raises `ContaminationQCError` naming
the rebuild instead. Verified against the real `260820` store, which correctly
skips both spike-in sections and says why.

**Not yet run on real data.** Every current generation predates `F35`, so the
columns do not exist. The numbers arrive with the raw rebuild.

**What is known so far**, from `BM` alone (independent of barcode assignment):
of 1,270 spike-in reads, **1,268 single-ended and 2 double-ended**. Every one is
a mis-barcoding event, so double-ended assignment looks roughly three orders of
magnitude cleaner -- but the rate needs the rebuild before it can be stated
properly.


