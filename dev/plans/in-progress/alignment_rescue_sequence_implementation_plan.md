# Rescued reads keep their sequence (`ARS`)

**Status:** in progress. `ARS-01`–`ARS-03` implemented on
`fix/rescue-carry-sequence`; not yet merged.

## Problem (`F64`)

`rescue_secondary_alignments` corrects reads whose primary alignment lost to a
better-covering secondary on another reference (nested references: an
enhancer deletion inside its parent allele, B6 vs BALB/c). It did so by
flipping FLAG bits only. minimap2 writes secondary records without SEQ or
QUAL, so every promoted record came out sequence-less, and extraction skips
sequence-less primaries. Every rescued read was therefore silently dropped --
exactly the reads rescue exists to keep -- while the summary reported them
rescued.

Two further defects in the same pass:

- The winning record was matched on (reference, start) only. A chimeric read
  can carry several alignments at one start with different extents; all were
  promoted, leaving one read with several primaries.
- Validation of a freshly made alignment counts sequence-less primaries and
  fails above 1 %, so a run where rescue moved more than 1 % of reads failed
  raw outright instead of losing them quietly.

Measured on existing alignments: rescued (dropped) reads were 0.1–1.8 % of
primaries per run, about 32,000 across 22 experiments, concentrated on
enhancer-deletion and B6/BALB-reassigned molecules. One run with 1.8 % failed
validation; one dual-allele run had 3,246 reads with more than one primary.

## Work items

| item | status | evidence |
|---|---|---|
| `ARS-01` rescue carries SEQ/QUAL (and `MM`/`ML`/`MN`) from the demoted record onto the promoted one; reverse-complements SEQ and reverses QUAL when the two align to opposite strands; promotes exactly one record per read | implemented, not merged | `test_rescued_read_keeps_its_sequence[*]`, `test_rescue_promotes_one_record_when_a_chimeric_read_shares_a_start` |
| `ARS-02` repair of already-committed alignments: `restore_primary_sequences` keeps the longest-aligned primary per read, gives it SEQ from a sibling record, demotes any other primaries; run as its own raw intermediate after alignment | implemented, not merged | `test_restore_primary_sequences_repairs_an_already_rescued_bam`, `test_restore_demotes_extra_primaries_left_by_old_rescue`, `test_restore_refuses_a_donor_that_does_not_fit_the_cigar`, `test_restore_leaves_paired_mates_as_two_primaries` |
| `ARS-03` raw algorithm version `4` → `5`, so stored raw generations re-extract with the recovered reads | implemented, not merged | `_STAGE_ALGORITHM_VERSIONS["raw"]` |
| `ARS-04` real-data qualification | qualified (repair) | see below |

### Design notes

- **Donor.** Any non-supplementary record of the same read that has SEQ.
  Supplementary records may be hard-clipped and are never donors. The donor's
  SEQ must match the target's CIGAR query length or nothing is written.
- **Read identity.** The repair and donor lookup key a read by name plus
  mate (`read1`/`read2`): paired mates share a name but are separate reads,
  each rightly primary. Rescue's own candidate grouping is still by name, as
  before this plan; it is not used on paired data today.
- **Base-modification tags.** `MM`/`ML` positions are relative to the read as
  sequenced, so they copy unchanged across strands.
- **Why a separate intermediate for the repair.** The aligned BAM is a
  committed, checksum-validated intermediate, and rescue is skipped when its
  summary already exists. Editing it in place would invalidate the commit. The
  repair is keyed on the aligned BAM's checksum, writes a new BAM only when
  something needs fixing, and is idempotent; on an already-correct alignment
  it costs one read pass, once.

### `ARS-04` — qualification

Repair run on the committed alignments of two runs, then the repaired BAM
re-validated:

| run | primaries | SEQ-less before → after | reads with >1 primary before → after | validation |
|---|---|---|---|---|
| conversion, one allele + enh-del (previously failed raw at 1.8 %) | 39,019 | 2,929 → 0 | 17 → 0 | passed |
| deaminase, two alleles + both enh-dels | 1,405,826 | 14,135 → 0 | 3,246 → 0 | passed |

Open: end-to-end `experiment full` on both, comparing per-reference read
counts before and after.
