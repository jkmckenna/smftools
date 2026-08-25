# smftools CLI Audit — `load` and `preprocess`

> **Superseded** by `pipeline_scaling_audit.md`, which states that this document
> describes the pre-partitioned, monolithic `.h5ad.gz` architecture — a materially
> different system that no longer exists, not an earlier version of the current one.
> Kept for its reasoning, not as a description of the code.

Scope: the `smftools load` and `smftools preprocess` stages of the CLI workflow,
their data structures, efficiency, and readiness to scale to millions of reads
across variable CPU/RAM/GPU hardware. Written against branch `0.4.5-layer-audit`.

---

## 1. Architecture overview

### 1.1 Stage model

The CLI (`cli_entry.py`, Click group) exposes one command per pipeline stage:
`load → preprocess → spatial / variant / chimeric / hmm / latent`. Each stage:

- Takes a single **config CSV** path, merged over `default.yaml → MODALITY.yaml →
  user CSV` into a flat `ExperimentConfig`.
- Resolves a canonical set of `.h5ad.gz` artifact paths per stage
  (`cli/helpers.py::get_adata_paths` → `AdataPaths`).
- **Resumes by artifact existence**: if a later-stage `.h5ad` already exists, the
  stage short-circuits and returns paths (`load_adata`, `preprocess_adata`).
- Reads the previous stage's AnnData, mutates/extends it, writes the next.

This is a clean, inspectable, restartable design. The central data object flowing
between stages is a single **AnnData**, which is the right choice: it plugs into
the scanpy/anndata ecosystem and gives obs/var/layers/uns/obsm a stable schema.

### 1.2 The AnnData schema (produced by `load`)

Documented in `AGENTS.md` and built in `informatics/converted_BAM_to_adata.py`
(conversion/deaminase) and `informatics/modkit_extract_to_adata.py` (direct):

| Slot | Contents | dtype / representation |
|------|----------|------------------------|
| `X` | read × position SMF signal (binarized methylation / conversion) | **dense float32** (NaN = no call) |
| `layers[sequence_integer_encoding]` | integer-encoded read bases | dense **int8** |
| `layers[mismatch_integer_encoding]` | read-vs-ref mismatch encoding | dense **int8** |
| `layers[base_quality_scores]` | per-base Phred | dense **int8** (−1 sentinel) |
| `layers[read_span_mask]` | 1 where read spans the position | dense **int8** |
| `var` | per-`Reference_strand` FASTA base per position; consensus | one string column **per reference** |
| `obs` | read-level metadata (barcode, strand, mapping QC, tags) | **category** |
| `uns` | encode/decode maps, reference sequences, bam paths | dicts |

The per-base layers were recently narrowed to int8 (see the inline notes in
`process_single_bam`), which halved layer footprint. Good.

---

## 2. `load` pipeline

### 2.1 Goal

Turn raw sequencing inputs (POD5/FAST5/FASTQ/BAM) into the raw AnnData above:
basecall (dorado) → align/sort (dorado/minimap2) → optional UMI/barcode
extraction → demultiplex → BAM QC → build AnnData → read-level QC annotations →
MultiQC → cleanup.

### 2.2 What's done well

- **Backed / streaming boundaries.** Per-BAM workers write per-sample `.h5ad`,
  which are then merged with `anndata.experimental.concat_on_disk` and returned
  `backed="r"`; the object is only `.to_memory()`-materialized at the last
  possible moment (`converted_BAM_to_adata`). This avoids holding N batch files
  resident during concatenation.
- **Sidecar parquet** for barcode (`BC/BM/bi`) and UMI (`U1/U2/RX/FC`) tags keeps
  the BAM clean and lets tags be joined columnar via `read_name` index. This is
  a scalable pattern.
- **Version-aware demux** (dorado ≥1.3.1 single-pass vs. 2-pass legacy) and an
  alternative `smftools` demux backend.
- **Per-worker thread capping** (`parallel_utils.configure_worker_threads`)
  prevents `n_workers × BLAS_threads` CPU oversubscription and forces the
  matplotlib Agg backend in workers — a real, commonly-missed footgun.
- **Memory guard** (`memory_guard.py`): cgroup v2 `memory.max` over the whole
  process tree on Linux; per-worker RSS watchdog on macOS. Best-effort, fail-open.
- Idempotent sub-steps gated on file/dir existence make partial reruns cheap.

### 2.3 Concerns / scaling risks

**(a) `extract_base_identities` is the load bottleneck (CPU + memory).**
`informatics/bam_functions.py:3991`. For each reference record it:
- Builds Python `defaultdict`s of **full-length dense numpy arrays per read**
  (fwd bases, rev bases, mismatch, quality, span) held **entirely in memory** for
  all reads of that record at once.
- Iterates `get_aligned_pairs(matches_only=True)` and loops **per aligned base in
  pure Python**.

At millions of reads this is both a Python-loop CPU wall and a memory spike
(reads_in_record × max_reference_length × several arrays). This is the single
biggest obstacle to the "millions of reads" goal.

**(b) BAM is re-`fetch`ed once per reference record.** `process_single_bam` loops
records and calls `extract_base_identities(bam, record, …)` each time, re-opening
and re-scanning. With many references this multiplies I/O and decompression. A
single pass that routes each read to its record would scale better.

**(c) Temp-h5ad-as-key-value-store for sequence encoding.** `_write_sequence_batches`
writes `AnnData(X=zeros((1,1)), uns=batch)` files purely to stash a dict of encoded
reads, then `_load_sequence_batches` reads them all back into one dict in the same
worker. Unless the intent is transient memory offload (it isn't — everything is
reloaded immediately), this is avoidable serialization + small-file I/O overhead.

**(d) Dense float32 `X`.** SMF signal is effectively ternary (0 / 1 / NaN), yet
stored as dense float32 = 4 bytes/cell. At 5M reads × 2000 positions that is
~40 GB for `X` alone, plus ~40 GB across the four int8 layers — before any copy.
This dominates memory at scale and is the main data-structure lever (see §5).

**(e) GPU is essentially unused in `load`.** Device is resolved to cuda/mps/cpu
but the hot path (per-base extraction, numpy vstack) is CPU/Python. The `device`
knob implies GPU acceleration that this stage doesn't deliver.

**(f) `manager.dict(record_FASTA_dict)`** proxies every access over IPC. It's only
touched per-record (not per-read) so it's fine today, but it's a latent trap if
future code reads it in inner loops.

**(g) Divergent AnnData-construction code paths.** Conversion and direct modalities
each reimplement encoding/batching/worker-sizing
(`converted_BAM_to_adata.py` vs `modkit_extract_to_adata.py`, 1.4k + 2.9k lines).
The direct path actually has the more sophisticated memory-and-CPU-aware pool
sizing (`_estimate_max_workers` / `_resolve_max_workers`) that the conversion path
lacks. These should converge on shared primitives.

---

## 3. `preprocess` pipeline

### 3.1 Goal

Consume the raw AnnData; add QC metrics; filter reads (length/quality/mapping,
then modification thresholds); positional coverage; base-context annotation;
binarization (direct); NaN-fill layer strategies; duplicate detection + complexity
(conversion/deaminase); optional invert/reindex; QC plots. Emit `pp` and
`pp_dedup` AnnData.

### 3.2 What's done well

- **Explicit peak-memory management.** Heavy layers (`sequence`/`mismatch`) are
  dropped up front and **re-attached from the backed source file** before saving
  (`preprocess_adata_core`), and the full `adata` is freed before the deduplicated
  copy is transformed. The reasoning is documented inline. This shows real
  awareness of the double-materialization problem.
- **Duplicate detection avoids naive O(n²)** (`flag_duplicate_reads.py`): reads are
  sorted and compared in sliding windows (`fwd_hamming_to_next` / `rev_hamming_to_prev`),
  hierarchical clustering runs only on cluster **representatives**, Hamming is
  computed on GPU/torch, and work is parallelized per `(sample, reference)` group
  via `ProcessPoolExecutor`. This is the right shape for scale.
- **Uncompressed-HDF5 write default** (`safe_write_h5ad`) is a deliberate, measured
  choice: gzip'd chunks made scattered boolean-masked row reads ~32 s vs sub-second
  uncompressed on a ~9 GB file. The re-attach-by-mask step depends on this.

### 3.3 Concerns / scaling risks

**(a) Whole-object in-memory model.** `preprocess_adata` loads the full AnnData via
`safe_read_h5ad` and operates on it in RAM. The layer-drop/re-attach dance mitigates
but doesn't eliminate this; read-wise filters and per-group stats could run
chunked/backed. Peak memory is still O(full object) for the base-context and
coverage passes.

**(b) Leftover debug `print(adata.shape)`** at `preprocess_adata.py:341,355` — should
be `logger.debug`.

**(c) Resume-by-directory-existence** treats "output dir exists" as "step complete."
A crashed/partial step leaves a dir that later runs will skip. A small completion
marker (or manifest status, which already exists for load artifacts) would be safer.

**(d) `var` grows one string column per reference** (`{ref}_top_strand_FASTA_base`,
consensus, reindexed copies). Fine for tens of references; with many amplicons/refs
this bloats `var` and every stage rewrite.

---

## 4. Cross-cutting

- **Config**: one flat `ExperimentConfig` with hundreds of fields. Convenient, but
  `threads` is overloaded (process count vs intra-op threads vs plotting workers),
  and there is no first-class memory budget beyond the guard's safety fraction.
  A small structured `resources` block (n_workers, threads_per_worker, mem_budget,
  device) would make scaling behavior predictable and testable.
- **I/O**: `.h5ad.gz` naming is uncompressed HDF5 (intentional, documented) — but the
  `.gz` suffix is misleading to anyone reading the tree. Each stage rewrites the full
  object; with a millions-of-reads object this is repeated multi-GB serialization.
- **Sparsity**: layers are all dense. SMF matrices are numerically dense (mostly
  observed), so CSR won't obviously help `X`; the win is in dtype/masking (§5), not
  sparse formats. `read_span_mask` (mostly 1s within span) and `base_quality` are
  genuinely dense.

---

## 5. Prioritized recommendations toward "millions of reads"

1. **Rework `X`'s representation** (highest leverage). Ternary SMF signal in dense
   float32 dominates memory. Options: store an int8 value layer + a boolean validity
   mask instead of NaN-bearing float32 (halves `X`, removes NaN ambiguity), and/or
   keep `X` backed and never fully materialize it in preprocess. Measure first with a
   representative object.

2. **Make `extract_base_identities` streaming + vectorized.** Avoid holding all
   per-read arrays for a record in memory; write directly into a preallocated
   memmap/backed array, and replace the per-base Python loop with vectorized CIGAR
   walking. This addresses both the CPU wall and the load memory spike.

3. **Single-pass BAM read across records** instead of one `fetch` per reference,
   when a BAM contains many references.

4. **Drop the temp-h5ad KV round-trip** in sequence batching (§2.3c) or make it a
   true spill-to-disk that isn't immediately reloaded.

5. **Unify the two AnnData builders** on shared encoding/batching/worker-sizing
   primitives; promote the direct path's memory-aware `_estimate_max_workers` to both.

6. **Chunk/back the preprocess read-wise passes** (filters, coverage, base context)
   so peak RAM is bounded by chunk size, not dataset size.

7. **Formalize a `resources` config** (workers, threads/worker, mem budget, device)
   and a completion-marker/manifest for step resume; replace leftover `print`s.

8. **Clarify the `device` contract** — either wire GPU into a hot path or scope the
   knob to the stages that actually use it (duplicate-detection Hamming, HMM).

None of these block current use; the recent commit history shows active,
well-reasoned memory work (int8 layers, backed concat, layer drop/re-attach). The
items above are the next tier for order-of-magnitude read counts.
