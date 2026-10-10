# Barcode QC: per-end evidence, a ligation / tagging model, and trimming before alignment (`BQC`)

**Status:** in progress. `BQC-01` merged; `BQC-02` implemented. A project-side prototype
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
  2. **Rear end without a barcode.** Among reads covering their whole insert,
     ~18 % carry no sequence at all beyond the rear of the insert. Since the
     barcode is part of the molecule, those rear ends were never barcoded;
     single vs double is mostly barcode ligation efficiency, plus undetected
     rear barcodes (~11 % of barcoded rear ends at <= 4 edits).
  3. **Truncation inside the insert** (amplicon: alignment short of the
     reference end; genome-wide: no flanked rear clip).
- With miscalls removed and end structure read from clips, a ligation /
  tagging model fits (see Model): native barcode ligation 0.75-0.86 per end,
  free-end tagging ~0.001, foreign double-same reads ~1e-9 per barcode. A
  naive spike-in bound (miscalls included) was ~1e6x higher. The prototype's
  spike-in had one end that almost never ligates, so it fixed neither lambda
  nor its denominator; real reads did.

## Model

**Scope: two-step ligation chemistry** (native barcoding): barcodes are
ligated per sample, samples are pooled, then the sequencing adapter is ligated
-- and joins only barcoded ends. The adapter is needed to load a molecule into
a pore, and a molecule is sequenced at most once, starting at an adapter end.
The rates and the model below are specific to this chemistry; rapid
(transposase) and PCR barcoding attach barcode and adapter differently and
need their own (`BQC-02` must take the chemistry from the kit config and fall
back to the naive bound for others). Consequences:

- the rear barcode is part of the molecule, so it is read whether or not the
  rear adapter joined; an end with **no sequence beyond the insert has no
  barcode** (prototype, 1.3M reads: rear clips 0-5 bp vs ~51 bp, cleanly
  bimodal; a ~35 bp shoulder may be barcode without the adapter arm);
- with adapter efficiency alpha per barcoded end, reads with / without a rear
  barcode are in ratio a (2 - alpha) / (2 (1 - a)) (a = barcoded share per
  end), so q is bounded by alpha = 1 and alpha -> 0 unless alpha is measured
  (adapter arm present beyond the rear barcode);
- free-end tagging happens in the pool (carried-over barcode during adapter
  ligation); alpha cancels from contamination ratios, since own and foreign
  molecules need the same adapters.

Prototype numbers revised on this basis: q 0.75-0.86 per end (not ~99 %),
free-end tagging lambda ~0.001, foreign double-same ~1e-9 per barcode
(max ~3e-8), foreign single ~2e-4 (max ~7e-3 for a small library with excess
free barcode).

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

## What can be estimated, with and without a spike-in (ligation chemistry)

A spike-in here is an unbarcoded DNA added to the pool after barcoding and
before adapter ligation. Its molecules have q = 0, so every spike-in read is
a free-end tagging event. Everything else comes from the real reads, given
per-end evidence (`BQC-01`: per-end names, edit distances, boundaries and
clips from the native extractor on untrimmed reads).

| estimate | without a spike-in | what a spike-in adds |
|---|---|---|
| per-end miscall rate | confident calls naming kit barcodes outside the pool; needs the pool to leave some kit barcodes unused, otherwise only the edit-distance profile | spike-in ends cannot carry an own barcode, so its in-pool calls are tags or miscalls only -- a second miscall profile |
| per-end detection d | confident-call share of barcoded ends (sequence present beyond the insert) | -- |
| barcode ligation q per library | amplicon: rear-barcoded vs rear-free full-length reads, bounded by adapter efficiency alpha (measurable from the adapter arm beyond the rear barcode). Genome-wide: **not identifiable** -- a read ending with no clip may be an unbarcoded end or a truncated read; reads with a rear barcode give only a lower bound | a spike-in amplicon has defined ends, so its full-length reads separate truncation from unbarcoded ends in genome-wide runs too (for its own ends) |
| free-end tagging lambda | from confident mismatch reads: phi = (1 - q) lambda / a, so lambda needs q (amplicon) and assumes mismatches are free-end tags, not chimeras or miscalls | **direct**: double / single ratio of spike-in reads (q = 0, both ends free) -- an estimate independent of q and of the real-read mismatch model; with a known spike-in molar input, an absolute rate |
| free-barcode composition pi_b | mismatch ends less their own-barcode half (library share) | spike-in end calls: pi_b with no own-barcode term to subtract |
| contamination of double / single reads | with psi = phi / (1 - phi) = (1 - q) lambda / q: psi^2 pi_b^2 (N - N_b) / N_b and psi pi_b / share_b, exactly -- **q is not needed**, so genome-wide runs get these too | a check: foreign double-same predicted vs observed on the spike-in (independence of the two ends; an excess means dimers, carryover or chimeric ligation) |
| ownership of mismatch reads | posterior from library share and pi_b (weak when pi_b tracks share) | -- |
| failed / empty libraries | barcodes with free barcode (mismatch / single ends) but no double reads | -- |
| read-through / concatemers | "long" rear clips | -- |

So without a spike-in, an amplicon run still gives every rate, but lambda
and the contamination estimates rest on one assumption (mismatch reads are
free-end tags) that nothing in the data tests; a genome-wide run loses q
and lambda but keeps phi, pi_b and so both contamination estimates (they
depend on q only through phi). The spike-in's value is an **independent, model-free
measurement of free-end tagging**, which validates or replaces the
real-read estimate.

**Spike-in design (lesson from the prototype).** Both spike-in ends must be
ligatable to a barcode (end-prepped / dA-tailed like the libraries) and the
amplicon must be fully covered by the reference. The prototype's spike-in was
a ~1.75 kb molecule with a 213-bp reference at one end, and that end almost
never ligated (3 of 269 short-form reads), so the spike-in measured one end
with no denominator. Declare it via `spike_in_references` (`BQC-04`).

## Work items

| item | status | what |
|---|---|---|
| `BQC-01` per-end evidence for every read | implemented, not merged | per read, in read (sequencing) orientation, unclassified reads included: front / rear barcode name, **edit distance**, **barcode / insert boundary position**, adapter flank found (y/n); **front / rear clip lengths** from the CIGAR (leading / trailing soft or hard clips only, flipped for reverse-strand records; internal insertions never count); supplementary-alignment flag (an `SA` clip is genomic, not barcode); end reason where a sequencing summary exists. Mode-agnostic (amplicon or genome-wide). Carried into the molecule tables, not only the sidecar. **As built:** alignment records carry `front_clip_length` / `rear_clip_length` (`ragged_store.cigar_end_clips`) and `has_supplementary_alignment`; the extractor writes `B7` / `B8` (bases from each read end to the barcode / insert boundary, whenever a barcode was extracted, matched or not); the identity sidecar names them `barcode_{front,rear}_edit_distance` / `barcode_{front,rear}_insert_offset`; all reach the raw-store obs. The flank flags (`B3` / `B4` present) are not "end sequenced": the amplicon-side flank is found whether or not a barcode follows -- use the clips. End reason moved to `BQC-01b` |
| `BQC-01b` end reason | proposed | per-read end reason (and the adapter arm beyond the rear barcode, for alpha) from the sequencing summary / POD5 where present | `BQC-01` |
| `BQC-02` ligation / detection / tagging model | implemented, not merged | the model above, on confident calls (edit <= threshold; out-of-pool calls give the miscall profile) and reads whose rear end was sequenced (`BQC-01` boundary / clip / flank); miscall mixture per end; per-barcode q, d, lambda, pi, ownership posteriors, contamination estimate; Poisson / bootstrap intervals; the naive spike-in bound kept as a fallback for runs without per-end evidence. **As built** (`preprocessing/barcode_tagging_model.py`, run after the `EGL-31` QC in preprocess, writing `tagging_by_barcode.parquet` / `tagging_summary.json` beside it): chemistry from the kit name (`SQK-NBD`/`EXP-NBD` -> native ligation; rapid / PCR kits skipped) rather than a kit-YAML key, since editing the kit YAMLs changes their checksum and would re-extract every run's barcodes; pool declared or inferred (barcodes with confident double-same reads >= 1 % of the median pool barcode's -- so a pooled library that failed reads as absent); miscall rate from out-of-pool confident calls when the kit has unused barcodes; phi from confident mismatch reads (both ends barcoded); contamination from phi (no q); q per library at alpha = 1 and alpha -> 0 when >= 1,000 reads cover their whole reference; lambda from phi and q; ownership summary; spike-in lambda only from spike-in reads covering the reference end to end with both reference ends starting reads (else the reason is recorded). Poisson bootstrap intervals on contamination. With `skip_unclassified` mismatch reads are gone and phi is flagged as a floor. Simulation tests recover q, lambda, the miscall rate and single-read contamination; on the prototype run it reproduces the project fit (phi 1.9e-4, q 0.86 / 0.76, double 1.4e-9, single 2.1e-4) |
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
