# Barcode QC: per-end evidence, a ligation / tagging model, and trimming before alignment (`BQC`)

**Status:** proposed. No implementation branch yet. A project-side prototype
exists (one 1.3M-read native-demux deaminase run) and its findings shape the
items below.

**Predecessor:** `EGL-31` (barcode contamination QC from an unbarcoded
spike-in, `preprocessing/barcode_contamination_qc.py`; in
`completed/generation_lifecycle_and_naming_implementation_plan.md`).

## Problem

`EGL-31` counts barcodes on an unbarcoded spike-in amplicon (pooled after
barcoding), split by `demux_type`. In practice:

- it never ran on a real project: `spike_in_references` is unset by default;
- per-end barcode **names** exist only for runs demultiplexed by the native
  extractor (`B5` / `B6`); with dorado or MinKNOW demux the end fields are
  empty and **mismatch** reads (two different barcodes) are dropped or left
  unclassified;
- a read is only sequenced if at least one barcode is ligated (the sequencing
  adapter attaches to the barcode), so there is no "unbarcoded" denominator;
  the informative split is **single / double-same / mismatch**;
- and, from the prototype, **the raw classes do not mean what they appear
  to.** Three effects must be separated before any rate is read:
  1. **Miscalls.** On a pool of 32 barcodes, 5.2 % of rear-end calls on real
     reads name a barcode outside the pool, so are miscalls by construction;
     a few out-of-pool barcodes absorb most of them (sequence confusion), and
     on the spike-in 80 % of rear calls are out of pool (its end resembles two
     barcodes). Per-end **edit distance separates them**: out-of-pool calls
     are >= 6 edits 99 % of the time, own-barcode calls <= 4 edits 94 %. Only
     ~6-19 % of in-pool "foreign" rear calls look like real tags.
  2. **Rear end not sequenced.** Among reads covering their whole insert, 92 %
     of reads with only one end's worth of unaligned sequence are
     single-barcoded (the read ends at the insert: 3'-end loss), against 9 %
     when the rear region was sequenced. Single vs double is mostly a
     sequencing property, not a ligation one.
  3. **Truncation inside the insert** (amplicon: alignment short of the
     reference end; genome-wide: no flanked rear clip).
- With miscalls removed and only rear-sequenced reads kept, a ligation /
  tagging model fits: native ligation ~99 % per end, free-adapter tagging of
  a free end ~2-3 % (spike-in, detection-corrected), and foreign double-same
  reads ~1e-7 per barcode. A naive spike-in bound (miscalls included) was
  ~4,000x higher.

## Model

Per sample s and end: native ligation in the sample's own reaction with
probability q_s; in the pool a still-free end is tagged with probability
lambda, by barcode c with probability pi_c (free-adapter composition); a
present rear barcode is read with probability d (end detection). Reads whose
rear end was sequenced then fall into:

- double-same s: q^2 d (+ small pool terms);
- single s: rear barcode present but undetected, or a free rear end left
  untagged -- 2q(1-q)(1-lambda) and q^2 (1-d)-type terms;
- mismatch (s, c): 2q(1-q) lambda pi_c d;
- foreign (both ends free): single / double-same / mismatch of pool barcodes,
  weight (1-q)^2.

The spike-in has q = 0: it fixes lambda and pi. Each barcode's single:double
ratio and its **mismatch participation** together separate d from q (free ends
would produce mismatches; undetected rear barcodes do not). Outputs: q_s
(native ligation efficiency per sample), d, lambda, pi_b vs library share, an
independence test (double-same excess = dimers / carryover), predicted vs
observed mismatch participation per barcode (goodness of fit), an **ownership
posterior** for every mismatch read (b, c), and a **point estimate** of foreign
double-same reads per barcode (the contamination a double-only selection
keeps), with intervals.

## Work items

| item | status | what |
|---|---|---|
| `BQC-01` per-end evidence for every read | proposed | per read, in read (sequencing) orientation, unclassified reads included: front / rear barcode name, **edit distance**, **barcode / insert boundary position**, adapter flank found (y/n); **front / rear clip lengths** from the CIGAR (leading / trailing soft or hard clips only, flipped for reverse-strand records; internal insertions never count); supplementary-alignment flag (an `SA` clip is genomic, not barcode); end reason where a sequencing summary exists. Mode-agnostic (amplicon or genome-wide). Carried into the molecule tables, not only the sidecar |
| `BQC-02` ligation / detection / tagging model | proposed | the model above, on confident calls (edit <= threshold; out-of-pool calls give the miscall profile) and reads whose rear end was sequenced (`BQC-01` boundary / clip / flank); miscall mixture per end; per-barcode q, d, lambda, pi, ownership posteriors, contamination estimate; Poisson / bootstrap intervals; the naive spike-in bound kept as a fallback for runs without per-end evidence |
| `BQC-03` standalone recompute | proposed | compute `BQC-02` for an existing generation without regenerating, writing beside it -- as `refresh-demux` does for demux columns |
| `BQC-04` spike-in configuration | proposed | document `spike_in_references`; a reference-name convention so it need not be set per experiment; genome-wide runs have no spike-in amplicon: the fit then leans on mismatch participation alone (or a declared unbarcoded control) |
| `BQC-05` extract, record, trim, then align | proposed | basecall untrimmed (`--no-trim`) and keep that BAM as the archive; extract per-end barcodes and boundaries (`BQC-01`; dorado summaries' positions for dorado-demultiplexed runs); trim adapter + barcode up to the boundary in the reads streamed to the aligner (no second BAM: untrimmed BAM + trim table); for direct-modification runs the trim must update MM / ML and move tables as dorado's does, for conversion / deamination (canonical) runs sequence and quality only. Removes aligner extension into flanks (spurious end mismatches, apparent conversions at read edges), makes read-length QC ratios meaningful, and leaves soft clips meaning real non-reference sequence. Opt-in (`barcode_trim_before_alignment`); it changes the alignment input, so it invalidates downstream generations; **parity check** (trimmed vs untrimmed on one run: end mismatch and conversion rates, QC pass rates, demux agreement) before any default change |

## Open questions

- Is per-end detection d the same at both ends? The front end anchors the
  read and is nearly always read; `BQC-02` should fit d per end.
- Dorado-demultiplexed runs: summaries carry front / rear scores and positions
  for the assigned barcode only. Enough for detection and trim boundaries, not
  for mismatch identities -- those runs need the native extractor on the
  untrimmed BAM where it still exists (not recoverable once reads were
  trimmed at basecalling).
- Genome-wide: shorter fragments mean more 3'-end loss and more chimeras; the
  supplementary flag and flank requirement matter more than in amplicons.
