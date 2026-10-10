# Chimera rates: detected, estimated and blind (`CHS`)

**Status:** proposed. No implementation branch yet.

**Related:** `preprocessing/chimera_classes.py` (`is_chimeric_any`, the union
of `chimeric_variant_sites`, `deaminase_PCR_chimera`,
`deaminase_segment_chimera`).

## Problem

The detectors flag chimeras, but each sees only chimeras that join
*different* templates: the SNP detector those joining different alleles, the
deaminase detectors those joining top- and bottom-strand templates
(deaminase runs). Chimeras between two templates of the same allele, or the
same strand, are invisible ("blind"). The flagged rate therefore
underestimates the chimera rate, by an amount that depends on each barcode's
allele and strand composition and on the detector's sensitivity -- and there
is no per-barcode report of either.

## Model

Under random template switching, a chimera joins two templates drawn from the
barcode's pool, so the detectable fraction is 2pq for allele frequencies p, q
(SNP; at most 1/2) and 2 f_t f_b for top / bottom fractions (strand). With D
the detected rate among callable reads and s the detector's sensitivity for a
cross-template chimera: estimated total C = D / (2pq s_snp) =
D / (2 f_t f_b s_strand); blind = C - D. The two detectors are independent
axes, so reads flagged by both test the random-switching assumption
(expected C 2pq s_snp 2 f_t f_b s_strand) and the two estimates of C should
agree (capture-recapture in spirit).

## Work items

| item | status | what |
|---|---|---|
| `CHS-01` detectors on synthetic reads | proposed | expose the SNP and deaminase detectors as functions over a read's calls, so they run on spliced reads outside the pipeline |
| `CHS-02` in-silico sensitivity | proposed | splice real non-chimeric reads of two templates (alleles; strands) at breakpoints along the aligned span, run the detectors: sensitivity by breakpoint position, SNP / deamination density, read length; an analytical SNP check (P[both segments hold >= k informative SNPs]) |
| `CHS-03` per-barcode report | proposed | per barcode and reference: reads, callable reads, detected rates (SNP, strand, both, union), p / q, f_t / f_b, sensitivity, estimated total and blind rates with intervals; overlap against the random-switching expectation |
| `CHS-04` deaminase detector agreement | proposed | where both deaminase methods ran, their agreement (it was low on a pilot), reported beside the rates |

## Caveats

Switching may not be random (hot spots, length dependence, allele-biased
amplification). Single-allele samples or references give no SNP estimate;
low deamination makes strand calls insensitive (sensitivity per enzyme and
dose); conversion stores holding one strand give no strand estimate.
