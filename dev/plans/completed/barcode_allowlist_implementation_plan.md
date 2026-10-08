# Barcode allowlist for multi-experiment runs (`BAL`)

**Status:** completed. `BAL-01` merged (#639); `BAL-02` qualified.

## Problem

One sequencing run can carry more than one experiment. The case that forced
this: EMseq (conversion) and DAFseq (deaminase) libraries barcoded onto one flow
cell, e.g. barcodes 1-5 conversion and 6-13 deaminase, sharing one basecall.
The generation-lifecycle plan (`completed/generation_lifecycle_and_naming_implementation_plan.md`)
already settled the layout: basecalls at run level, one experiment directory per
(modality x basecall). What was missing is any way to keep an experiment to its
own barcodes. Without it each experiment processes every read under its own
modality, which is not merely wasteful: HMM training and latent fits pool reads
across samples, so the other assay's reads would contaminate the models, not
just sit in a separable bucket.

Pre-splitting the basecalled BAM by barcode outside smftools would work but
starts the provenance chain at a hand-made file; filtering afterwards by sample
metadata is too late for the pooled model fits.

## Work items

| item | status | evidence |
|---|---|---|
| `BAL-01` `barcodes_to_include`: drop non-listed barcodes before raw extraction | merged (#639) | `414d7fe`; `test_allowed_read_ids_keeps_only_listed_barcodes`, `test_unset_barcode_allowlist_leaves_raw_fingerprint_unchanged` |
| `BAL-02` real-data qualification on a dual-modality run | qualified | see `BAL-02` below |

### `BAL-01` — as implemented

- Config key `barcodes_to_include` (list; null keeps every barcode).
- Raw extraction planning (`_allowed_read_ids`, `_keep_allowed` in
  `cli/raw_adata.py`) drops reads whose resolved barcode -- from the canonical
  identity sidecar, i.e. after `F58`'s precedence -- is not listed, before
  bucketing. Excluded reads are never extracted, so no downstream stage sees
  them. Both convertible and direct paths. Set without a sidecar, it raises.
- `barcode_number_key` compares by barcode number across spellings (`4`,
  `NB04`, `barcode04`, `SQK-NBD114-24_barcode04`); `barcode_key` alone does not
  reduce a kit-qualified label. Values without a barcode token (`unclassified`,
  a read-group ID) only match if listed literally.
- **Fingerprint.** The key joins raw's semantic config only when set
  (`_OMIT_WHEN_UNSET_CONFIG_KEYS`). Unset is the behaviour every existing raw
  store was built with; including it unconditionally would have marked every
  stored raw generation `stale_config` for no change in output. Verified with
  `experiment plan` on an existing run: raw stays `compatible`.

### `BAL-02` — qualification

Run both experiments of a dual-modality run from one shared basecall, each with
its own allowlist; confirm each published store holds only its listed barcodes
and that the shared control barcode appears in both.

**Qualified 2026-10-03** on a dual-modality run (one SUP basecall, 811,666 primary
segments; conversion barcodes 1-4, deaminase 6-13, no-enzyme control 5 in both),
`experiment batch full` over both configs, 2/2 completed:

| experiment | allowlist | kept at raw | barcodes in published preprocess store |
|---|---|---|---|
| conversion | 1-5 | 441,502 of 811,666 | exactly 1-5 (156,228 reads) |
| deaminase | 5-13 | 172,135 of 811,666 | exactly 5-13 (94,343 reads) |

Nothing outside either allowlist reached a store; the shared control barcode
is present in both.
