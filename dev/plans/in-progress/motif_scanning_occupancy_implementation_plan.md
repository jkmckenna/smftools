# Motif scanning and per-molecule motif occupancy (`MOT`)

**Status:** in progress. `MOT-01` merged, `MOT-02` implemented. One PR per item, in order.

## Question

Which transcription-factor motifs on a reference does each molecule leave
bound by something TF-sized, cover with a nucleosome, or leave accessible --
per sample, per condition, and jointly with other motifs on the same molecule?
Bulk assays give a population average per position; single-molecule footprints
give a state per molecule per motif instance, and co-occupancy.

## What exists

- **Nothing motif-aware in smftools**: no motif-file reading, scanning or
  interval labelling. Biopython is a dependency but unused for motifs.
- **HMM occupancy classes** (`hmm_feature_sets`): per position and read,
  footprints `small_bound_stretch` (6-40 bp), `medium_bound_stretch` (40-100),
  `putative_nucleosome` (100-200), `large_bound_stretch` (200+); accessible
  `small/mid/large_accessible_patch` and `nucleosome_depleted_region` (110+);
  per emission variant (`HCE-06`).
- **Selection and grouping**: plan datasets bound with `bind_ml_dataset`
  (molecules, QC/dedup filters, groups, masks, coordinate frames), as
  `context-bias` (`SCB`) and `periodicity` (`RPG`) use; cached runs keyed by
  referenced files' content (`tools/analysis_cache`, `F72`).
- **Strand handling**: `site_context_bias.strand_of`, `strand_window`;
  reference sequences from spines (`sequences_from_uns`, padding trimmed).
- **Earlier project scripts** (not in smftools): FIMO over the NKG2A
  references with a hard-coded motif file and hand-written TSS/deletion
  coordinate conversion; mean small-footprint signal summed over motif
  intervals per cell type (population level); per-read TetO/ZF site occupancy
  (bound vs accessible vs observed) on one fixed reference.

## Engines: a prototype (2026-10-07)

On the 6B6 reference (4.7 kb) with 637 archetype motifs (MEME v4), at
p < 1e-4, both strands:

- **Biopython is not usable as the engine.** `Bio.motifs.parse(..., "minimal")`
  rounds letter probabilities to integer counts (0.677 x nsites 20 -> 14, read
  back as 0.70), and `pssm.search` + `pssm.distribution` took > 6 min where
  FIMO takes < 1 s.
- **A numpy engine is fast**: exact MEME parsing, vectorized log-odds scoring
  of every window on both strands, exact p-values by dynamic programming over
  the integer-scaled score distribution (FIMO's method): 0.5 s for all motifs.
- **FIMO's defaults differ from the motif file**: without `--bfile` FIMO uses
  NRDB background frequencies, not the file's (uniform) ones -- 12.90 vs 14.36
  for the same hit. With `--bfile --uniform--` it agrees with the `MOT-01`
  scanner on 964 of 965 hits (identical coordinates, strand, score, matched
  sequence and p-value; the odd pair sits on the 1e-4 cutoff). The prototype's
  apparent disagreement was its own minus-strand mapping. Low-information
  motifs (long C2H2 zinc fingers) are significant at low or negative log-odds
  scores -- in both tools. Further agreement checks: `MOT-06`.

## Design

### Motif files are always user-supplied

No motif file ships with smftools or is assumed: every command takes
`--motifs PATH` (MEME minimal format first; JASPAR / TRANSFAC readable later
through the same parser interface). Motif identity in every output is the
file's content hash plus the motif ID, so cached scans follow the file.

### Two engines, one output

- `engine: builtin` (default): the numpy scanner -- no external tools.
  Background: uniform (default), the motif file's, or the scanned sequences'
  base composition; pseudocount 0.1 (FIMO's default) distributed by
  background; p-values exact for the integer-scaled matrix.
- `engine: fimo`: runs FIMO (MEME suite) when `fimo` is on `PATH` (or
  `--fimo PATH`) with the same threshold and an explicit `--bfile` matching
  the chosen background, so the two engines answer the same question; a clear
  error when FIMO is requested but not found.
- Both write the same interval table: `motif_id`, `motif_name`, `family`
  (parsed from IDs such as `AC0395:SOX:Sox` when present), `reference`
  (physical, strand-suffixed as smftools stores it), `start`, `end`
  (0-based, half-open, reference coordinates), `motif_strand`, `score`,
  `pvalue`, `engine`, `motif_file_sha256`. Windows touching `N` are skipped;
  case is ignored.

### Bulk feature-class tracks with motif lanes

The first look, before any per-molecule classification: per group (sample /
barcode, or any grouping) and reference, the fraction of reads in each HMM
feature class at every position -- TF-sized footprint (`small_bound_stretch`),
medium footprint, nucleosome (`putative_nucleosome` + `large_bound_stretch`),
accessible (any accessible feature) -- over reads that span the position
(Wilson band optional). Below the axis, the motif instances from the scan,
packed into non-overlapping lanes, coloured by family; filtered by p-value,
family list or top-N per region, labels for the strongest. Display
coordinates as `RPF` (origin and orientation, e.g. TSS-relative, upstream
left); a whole-locus figure plus zooms on named regions; one panel per group
stacked on a shared x axis, or groups overlaid per class.

To make "overlaps cleanly" measurable, a per-instance table: for each motif
instance and group, the mean class fraction inside the motif vs in flanks of
the same width either side (`contrast = inside - flanks`, and the share of
reads spanning it), ranked -- instances where a TF-sized footprint sits on
the motif and not around it rise to the top, and are outlined in the figure.
Default and learned HMM layers can be drawn side by side (layer prefix).

### Occupancy states per molecule and motif instance

For each read covering a motif instance (with `flank` bp either side),
classify by the HMM layers over the instance:

| state | rule (default) |
|---|---|
| `tf_bound` | a `small_bound_stretch` covers the motif core |
| `medium_bound` | a `medium_bound_stretch` covers it |
| `nucleosome` | `putative_nucleosome` or `large_bound_stretch` covers it |
| `accessible` | an accessible feature covers it |
| `uninformative` | fewer than `min_sites` observed sites of the model in motif +/- flank, or the read does not span it |

Precedence and coverage rule (any overlap vs a covered fraction) are
parameters; the informative-site rule is not optional -- a motif with no
observable site would otherwise read as bound. The HMM layer prefix (model and
variant, e.g. `C_` / `C_learned_`) is a parameter, so variants can be
compared on the same motifs.

### Outputs

- Per read x motif instance states (sparse, parquet) -- the base table.
- Per group (sample, condition, ...) x motif instance: state fractions with
  Wilson intervals, n informative reads.
- Per motif (family): aggregates over instances.
- Co-occupancy: for instance pairs within `max_distance`, the 2x2 table of
  bound/not bound on molecules informative for both, with log odds ratio.
- Group comparisons: difference in bound fraction with a test per instance
  (Fisher / chi-square), BH-adjusted.
- Figures: locus track (state fractions per group along the reference, motif
  lanes coloured by family, as the earlier enhancer plot), per-instance
  state bars per group, co-occupancy heatmap, comparison volcano.

### An analysis, not a stage

Results depend on a user motif file, thresholds and the motifs of interest,
and are wanted for some experiments, not every run; as a stage they would
enter fingerprints and run everywhere. So: library functions + thin CLI,
reading through plan datasets like `context-bias` and `periodicity`, cached by
`analysis_cache`. Projects keep motif sets and figure layouts in their own
metadata.

## Work items

| item | status | scope |
|---|---|---|
| `MOT-01` motif files and the built-in scanner | merged | MEME parser (exact), numpy scanner, exact p-values, interval table; `smftools motifs scan` |
| `MOT-02` FIMO engine | implemented, not merged | optional `engine: fimo`, background parity, same table |
| `MOT-03` bulk class tracks with motif lanes | proposed | per group x reference: class fractions along the locus, motif lanes below, per-instance inside-vs-flank contrast table; `project|experiment motif-tracks` |
| `MOT-04` per-molecule occupancy | proposed | states per read x instance from HMM layers via a plan dataset; group fractions; `project|experiment motif-occupancy` |
| `MOT-05` co-occupancy, comparisons, figures | proposed | pair tables, group tests, locus track, bars, heatmap, volcano |
| `MOT-06` qualification | proposed | built-in vs FIMO agreement on real references; occupancy vs the earlier per-read TetO script on its data; NKG2A locus run |

### `MOT-01` — motif files and the built-in scanner

`analysis/compute/motifs.py`: `read_motifs(path)` (MEME minimal; exact
probabilities, nsites, IDs), `log_odds(matrix, background, pseudocount)`,
`score_distribution` / `pvalues` (integer-scaled DP), `scan(sequences, motifs,
threshold, background)` -> interval table. `tools/motifs.py`: references from
spines (or `--fasta`), physical strand references mapped as `strand_of`
does, caching by motif-file and sequence hashes. CLI `smftools motifs scan
--motifs PATH (--experiment-dir | --project-dir | --fasta) [--threshold 1e-4]
[--background uniform|motif|sequence] -o DIR`.

Tests: parser keeps probabilities exactly; scores equal a hand computation;
p-values equal brute-force enumeration for short motifs; reverse-strand hits
equal a forward scan of the reverse complement; `N` windows skipped;
threshold monotone.

As built: `analysis/compute/motifs.py` (`read_motifs`, `log_odds`,
`score_model`, `strand_models`, `scan_sequence`, `scan`) and `tools/motifs.py`
(`fasta_sequences`, `experiment_sequences`, `project_sequences`,
`scan_references`); CLI `smftools motifs scan --motifs PATH
(--experiment-dir | --project-dir [--experiment ID ...] | --fasta)
-o DIR [--max-pvalue] [--background uniform|motif|sequence] [--pseudocount]
[--motif ID ...] [--reference NAME ...] [--refresh]`.

- `reference` in the hit table is the sequence name as the spines record it
  (strand suffix removed: `6B6`, `6B6_enh_del`), since motif instances are
  sequence features on the forward strand; `MOT-03` maps physical references
  (`6B6_top`, `6B6_bottom`) to it as `strand_of` does.
- Minus-strand hits score the reverse-complement matrix on the forward
  sequence, with its own p-value model (a strand-asymmetric background
  changes the distribution); `start`/`end` are forward, `matched_sequence`
  is read 5'->3' on the motif's strand.
- Scores use 10,000 integer bins; p-values are exact for the binned matrix.
- Cached as `motif_hits.parquet` + `run.json`, keyed by the motif file's and
  every sequence's SHA-256 and the settings.

On the 260923 project references with the 637 archetype motifs at p < 1e-4:
3,668 instances (B6 965, BALB 960, enh-del 853 / 855, ctcf_mNanog 35) in
12 s, most of it reading spines; the spine's B6 sequence is the earlier
project's FASTA byte for byte.

### `MOT-02` — FIMO engine

Run FIMO with `--text --thresh --bfile` (background written from the chosen
one) and `--max-stored-scores` high enough; parse into the same table.
Tests (skipped without FIMO): same columns; on a fixture, hits equal the
built-in engine within a stated tolerance.

As built: `tools/motifs.py` `find_fimo`, `fimo_version`, `parse_fimo_text`,
`scan_with_fimo`; `scan_references(engine="fimo", fimo=PATH)`; CLI
`smftools motifs scan --engine fimo [--fimo PATH]`. Backgrounds map to
`--bfile --uniform--`, `--bfile --motif--`, or a written order-0 file;
`--motif-pseudo` carries the pseudocount; the FIMO version joins the cache
key. Requesting FIMO when it is not installed is an error naming
`--engine builtin`.

The `sequence` background is now strand-symmetric (both strands counted, as
FIMO makes it): with forward-strand counts the engines disagreed on ~6 % of
hits; with both strands they agree.

On the 260923 references (637 archetype motifs, p < 1e-4), builtin vs FIMO 5.5.9:

| background | builtin | FIMO | shared | max abs score diff | max abs log10 p ratio |
|---|---|---|---|---|---|
| uniform | 3,668 | 3,666 | 3,662 | 0.045 | 0.064 |
| motif | 3,668 | 3,666 | 3,662 | 0.045 | 0.064 |
| sequence | 3,808 | 3,805 | 3,803 | 0.050 | 0.047 |

Every unshared hit has p between 9.9e-5 and 1e-4 (binning at the cutoff);
matched sequences are identical; both take 2-3 s.

### `MOT-03` — bulk class tracks with motif lanes

`analysis/compute/motif_tracks.py`: per group, position and class, reads in
the class and reads spanning (from the HMM class layers through a plan
dataset, streamed in blocks); per instance and group, inside / flank means
and `contrast`. `analysis/plot/motif_tracks.py`: tracks (one panel per group
or groups overlaid per class), motif lanes (greedy interval packing, family
colours, labels), outlined high-contrast instances, display coordinates and
region zooms. CLI `smftools project|experiment motif-tracks --plan --dataset
--motif-hits PARQUET --group-by ... [--layer-prefix C_] [--region NAME|START-END]
[--max-pvalue] [--families] [--coordinate-origin --coordinate-reverse]`.

Tests: class fractions equal a direct count; reads not spanning a position
not counted; contrast on a constructed footprint over a motif; lanes never
overlap; coordinates and orientation as `RPF`; figure per group x reference.

### `MOT-04` — per-molecule occupancy

`analysis/compute/motif_occupancy.py`: given read x position layers (HMM
class layers, observed sites) and instances, the state per read x instance.
`tools/motif_occupancy.py`: binds a plan dataset (channels: the HMM class
layers at all positions, the model's site calls), groups, writes the base
table and group fractions. CLI `smftools project|experiment motif-occupancy
--plan --dataset --motif-hits PARQUET [--layer-prefix C_] [--flank 10]
[--min-sites 2] --group-by ...`.

Tests: each state from a constructed read; uninformative when no sites;
precedence; variants by prefix; group fractions equal a direct count.

### `MOT-05` — co-occupancy, comparisons, figures

As in Outputs. Tests: 2x2 tables equal direct counts; only reads
informative for both instances counted; BH adjustment; figures written.

### `MOT-06` — qualification

Built-in vs FIMO on the NKG2A references with matched background (shared
hits, score and p agreement, explained differences); the per-read
TetO/ZF-occupancy numbers reproduced from the same layers; a full-locus run
on the enzyme panel and 260820 sets (WT vs enh-del, cell type, spermidine)
with run time.

## Out of scope

- Shipping or downloading motif databases.
- Joint thermodynamic inference of TF and nucleosome binding (e.g.
  HiddenFoot); this plan classifies from the HMM's states. A later
  comparison against such a model is possible on the same instances.
- Genome-wide k-mer / motif enrichment in accessible stretches: on a
  single amplicon it mostly reflects which few hundred bp are open and
  partly the enzyme's context bias (`SCQ-04`); revisit for genome-mode data.
