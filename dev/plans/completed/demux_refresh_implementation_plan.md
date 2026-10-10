# Demux refresh without re-extraction (`DRF`)

**Status:** completed -- `DRF-01`..`DRF-03` merged (PR #725, `0523523`,
`2b0b26c`); `DRF-04` is an open question, recorded here rather than in a new
plan because it is one decision.

**Predecessors:** `EGL-29` (sequencing-summary demux status, in
`completed/generation_lifecycle_and_naming_implementation_plan.md`) and `F34`
(raw reassembly from existing shards).

## Problem

Runs demultiplexed upstream (BAM `BC` tags or MinKNOW per-barcode
directories) carry no barcode-*end* evidence: every read got
`demux_type = unclassified`. A double-barcoded-only selection then keeps
nothing. The basecaller's sequencing summary holds per-read front / rear
barcode scores for nearly every read (checked: 99.9-100 % of stored reads on
four such deaminase runs, including one whose reads were trimmed at
basecalling, where sequence-based re-derivation found both ends in only 7 %),
but applying it meant regenerating the run from the raw stage -- about an
hour of extraction per run, then preprocess, spatial and HMM again.

## What landed

| item | status | what |
|---|---|---|
| `DRF-01` summary fills `unclassified` | merged (`0523523`) | `attach_demux_status` treated only empty / `unknown` as no call; `unclassified` from end-less BAM tags blocked the summary. It now fills; classifier calls (double, single, mismatch) are still kept. |
| `DRF-02` summary in raw reassembly; annotations reach every copy | merged (`2b0b26c`) | `reassemble-raw` applies the configured (or found) sequencing summary. Reassembly rewrote obs, molecules and the spine but **hardlinked the segment catalog and both pointer indexes**, which also carry the read annotations -- and the molecule index is the ML reader's identity source, so re-derived `demux_type` never reached readers. The annotation columns (`demux_type`, `demux_type_source`, `demux_type_confidence`, `barcode_agreement`) are now rewritten, or added, in all three. |
| `DRF-03` `experiment refresh-demux` | merged (`2b0b26c`) | reassemble raw with the summary, then publish a sibling preprocess generation (`preprocessing/demux_refresh.py`): demux columns from the current raw generation, duplicate keepers re-chosen over the **existing** clusters with `_select_duplicate_keeper` (`--prefer`), every other artifact hardlinked, manifest checksums updated, provenance in an `annotation_refresh` block. Matrices, spatial and HMM generations are reused (they cover every read). |

**Verification.** With unchanged demux calls the keeper recomputation
reproduces the stored `passes_dedup` exactly (two deaminase runs, 0.58M and
1.57M reads). On a 0.58M-read run: about 4 minutes end to end (vs hours for a
regeneration); raw 403k double / 175k single; 577 duplicate clusters switched
to a double-barcoded keeper (double-barcoded kept molecules 4,294 -> 4,871 of
5,836); both new generations pass `validate_raw_generation` /
`validate_preprocess_generation`; ML reads with HMM layers succeed. Applied
across a 23-experiment project; tests in `tests/unit/test_demux_refresh.py`,
`test_raw_reassembly.py`, `test_sequencing_summary_demux.py`.

**Known consequence.** A refreshed run's raw generation changed, so the next
ordinary pipeline run recomputes downstream stages (their recorded inputs no
longer match). The `annotation_refresh` block records why; downstream
provenance is deliberately not rewritten.

## `DRF-04` -- the default keeper preference (open)

`duplicate_detection_demux_types_to_use` defaults to
`["single", "double", "already"]` -- any barcoded read -- so even runs with
end calls choose duplicate keepers without preferring double-barcoded reads.
Under a double-only selection that drops whole clusters: refreshing a project
with `--prefer double` raised double-barcoded kept molecules by 10-40 % on
most runs and by up to ~2x on some. Options: default to `["double"]` with
fallback to the old order (keeps a keeper in every cluster), or leave the
default and document the trade-off. Not decided.
