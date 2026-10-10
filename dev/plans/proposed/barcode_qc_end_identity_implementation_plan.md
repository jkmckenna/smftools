# Barcode QC: per-end identities and a free-adapter tagging model (`BQC`)

**Status:** proposed. No implementation branch yet.

**Predecessor:** `EGL-31` (barcode contamination QC from an unbarcoded
spike-in, `preprocessing/barcode_contamination_qc.py`; in
`completed/generation_lifecycle_and_naming_implementation_plan.md`).

## Problem

`EGL-31` counts barcodes on an unbarcoded spike-in amplicon (pooled after
barcoding), split by `demux_type`, and reports end disagreement from
`barcode_front` / `barcode_rear`. In practice:

- it never ran on a real project: `spike_in_references` is unset by default
  ("no reads matched");
- per-end barcode **names** exist only for runs demultiplexed by the native
  extractor (`B5` / `B6`); for runs demultiplexed by dorado or MinKNOW the
  end fields are empty, and **mismatch** reads (two different barcodes) are
  dropped or left unclassified, so they never reach the store;
- `EGL-31`'s framing assumed unbarcoded spike-in reads as a denominator. A
  read is only sequenced if at least one barcode is ligated (the sequencing
  adapter attaches to the barcode), so every observed spike-in read carries
  >= 1 barcode; the informative split is **single / double-same / mismatch**.

Measured on a project (before any filtering): barcoded spike-in reads are
overwhelmingly single-ended (e.g. 12,224 single vs 212 double-same on a
0.58M-read deaminase run; ~10.5k vs ~60 on a dual-assay run); only one run
(native demux) has the mismatch class (673 spike-in mismatch reads).

## Model

Each end of a free-ended molecule is tagged independently with probability
lambda, by barcode b with probability pi_b (the free-adapter pool's
composition). Among observed reads: single 2 lambda (1 - lambda); double-same
lambda^2 sum pi_b^2; mismatch lambda^2 (1 - sum pi_b^2). From spike-in counts:
lambda (per-end tagging efficiency), pi_b (free adapter per barcode, against
library share), an independence test (observed double-same share vs
sum pi_b^2: an excess means a same-barcode mechanism -- adapter dimers,
carryover), and per-barcode double-same excess. For real molecules: a
properly ligated molecule cannot be re-tagged; one free end yields a mismatch
or single read; only molecules with both ends free can appear as a foreign
double-same read (lambda^2 pi_b^2 each) -- the contamination a double-only
selection still admits, reported as a curve over the unknown number of such
molecules, bounded by the real reads' single / mismatch rates. Real mismatch
reads (own + foreign barcode) give a barcode-pair matrix directly.

## Work items

| item | status | what |
|---|---|---|
| `BQC-01` per-end identities for every read | proposed | a sidecar of read-start / read-end barcode names (native extractor) over the **full** basecall BAM, unclassified reads included, joinable by read id; only for runs whose reads kept their barcodes (basecalled without trimming) |
| `BQC-02` tagging model in the contamination QC | proposed | single / double-same / mismatch counts per barcode; lambda, pi_b vs library share, the independence test, per-barcode excess, the double-only contamination curve; Poisson / bootstrap intervals (per-barcode cells hold tens of reads) |
| `BQC-03` standalone recompute | proposed | compute the QC for an existing generation (no regeneration), writing beside it -- as `refresh-demux` does for demux columns |
| `BQC-04` spike-in configuration | proposed | document `spike_in_references`; consider a reference-name convention so it need not be set per experiment |

## Open questions

- Is end tagging independent in practice? `BQC-02`'s test answers it; if
  not, the contamination curve needs the dimer / carryover term.
- Where real mismatch reads are unavailable (upstream demux), is the
  spike-in alone enough to bound double-same contamination? The bound is
  loose without the real reads' end structure.
