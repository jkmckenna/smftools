# ML-700 — performance and scalability qualification: benchmark plan

**Work package:** ML-700
**Branch:** `test/ml-scale-qualification` (cut from `f74a42f`, no upstream)
**Plan date:** 2026-08-04
**Parent ledger:** [ml_implementation_ledger.md](ml_implementation_ledger.md)
**Status:** `IN_PROGRESS` — harness, fixtures, and Sweeps A/C/refusal implemented; first
measurements taken; **the originally planned measurement protocol was found invalid and replaced**
(see "Methodology correction"). Sweeps B and D not started.

## The question this package actually answers

Every memory limit in the ML data plane is currently **analytic, not measured**.
`partition_dataset._bytes_per_row` (`partition_dataset.py:293-304`) computes a closed form:

```text
persistent  = positions*channels*(4+1+1) + channels + positions + 8 + (8 if labeled)
bytes_per_row = 2 * persistent                      # 2x transient allowance
materialization_estimate = 3 * n_rows * bytes_per_row
```

Those `2x` and `3x` constants are the entire safety margin between a bounded read and an OOM, and
nothing in the repository has ever compared them to a measured peak. ML-700 exists to calibrate
them. Two failure directions, both real:

- **Under-estimate → OOM.** The preflight approves a read that then exceeds real memory. This is
  the failure the whole `MLMemoryBudgetError` mechanism was built to prevent, and it is currently
  unverified.
- **Over-estimate → false refusal.** The preflight refuses workloads that would have fit
  comfortably, pushing users to raise budgets manually and eroding trust in the guardrail.

The primary published artifact is therefore the measured ratio

```text
headroom = peak_rss_delta / estimated_bytes
```

across the matrix, with the required invariant `headroom <= 1.0` (the estimate is a true upper
bound) and a reported distribution so the constants can be tightened or loosened on evidence.

## Benchmark matrix

Dimensions are the five named in the ledger deliverable. Levels are chosen to straddle the
analytic refusal frontier rather than to sample uniformly.

| Dimension | Levels | Rationale |
| --- | --- | --- |
| Rows | 500; 5,000; 50,000 | 50,000 crosses the materialization budget at every width ≥ 1,000 positions |
| Positions (features) | 500; 1,000; 5,000; 20,000 | 20,000 approximates a full amplicon panel |
| Channels | 1 (deaminase); 2 (conversion: GpC + CpG) | The two registered modality shapes |
| Partitions | 1; 4 experiment sources | Exercises multi-source union-channel projection |
| Backend | sklearn (RF, logistic); plain Torch (residual CNN) | The two merged verticals, ML-301 and ML-303 |
| Device | cpu; mps | This host: Apple Silicon, `torch.backends.mps.is_available() == True`. No CUDA available; the CUDA row is a documented gap, not a silent omission. |

Full cross-product is 384 cells, which is not worth running. The executed matrix is:

- **Sweep A (memory calibration, cpu only):** all rows × positions × channels, batch reads only.
  24 cells. This is the cell set that produces the headroom distribution.
- **Sweep B (backend throughput):** 5,000 rows × {1,000; 5,000} positions × both channels ×
  both backends × both devices. 16 cells.
- **Sweep C (worker scaling):** 50,000 rows × 1,000 positions × 1 channel × `num_workers` ∈
  {1, 2, 4, 8}. 4 cells.
- **Sweep D (explanation chunking):** 500 rows × {1,000; 5,000} positions ×
  `example_batch_size` ∈ {1, 8, 64, 512}. 16 cells.

## Analytic frontier to validate

Computed from the current constants. Sweep A must confirm each refusal boundary is real and each
approved cell actually fits.

| Positions | Channels | bytes/row | Max rows (materialize, 2 GiB) | Max rows/batch (64 MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 1 | 7,034 | 101,766 | 9,540 |
| 500 | 2 | 13,036 | 54,911 | 5,147 |
| 1,000 | 1 | 14,034 | 51,006 | 4,781 |
| 1,000 | 2 | 26,036 | 27,493 | 2,577 |
| 5,000 | 1 | 70,034 | 10,221 | 958 |
| 5,000 | 2 | 130,036 | 5,504 | 516 |
| 20,000 | 1 | 280,034 | 2,556 | 239 |
| 20,000 | 2 | 520,036 | 1,376 | 129 |

**Expected headline finding.** Real SMF experiments run ~10^6 reads (the splenic DAFseq run that
motivated `duplicate_detection_scaling.md` had ~1.3M). At every width ≥ 1,000 positions the
materialization budget refuses well below 10^5 rows. So at production scale `materialize()` is
*always* refused and `iter_batches()` is the only supported path — which in turn means
**non-incremental sklearn fitting has a hard row ceiling that Torch does not**. That asymmetry is
the single most useful thing this package can publish, and it belongs in the ML-701 guidance.

> **Superseded 2026-08-05.** The asymmetry above is wrong: it was inferred from the ledger, not
> from the code. Torch has the same ceiling, and sklearn's incremental path does not avoid it
> either. See "Sweep B" below — the real finding is worse and more important than the predicted
> one.

## Measurement protocol

- **Peak memory:** sample `psutil.Process().memory_info().rss` from a watchdog thread at 20 ms
  while the measured region runs; record `peak - baseline`. Baseline is taken after fixture
  construction and a forced `gc.collect()`, so store-building cost is excluded from the read
  measurement. `psutil` 7.2.2 is present in `venvs/venv-all`.
- **Repeats:** 1 warmup (discarded, pays import/JIT/page-cache cost) + 3 measured. Report median
  and full spread. A cell whose spread exceeds 20% of its median is flagged unstable rather than
  silently averaged.
- **Throughput:** rows/second and MiB/second for partition reads; wall seconds for fit and
  explanation. Timed with `time.perf_counter` around the measured region only.
- **Fixtures:** reuse `_write_source` from `tests/integration/machine_learning/test_partition_dataset.py`,
  generalized into a shared parameterized builder. It calls the real
  `informatics.partition_store.write_experiment_store`, so benchmarks exercise genuine Zarr reads
  rather than synthetic arrays. Deterministic RNG seed per cell, recorded.
- **Environment record:** every result file carries Python, numpy, sklearn, torch versions, device,
  platform, physical memory, CPU count, git commit, and per-cell seed. Recorded state for this
  host: Python 3.12.9, numpy 1.26.4, sklearn 1.9.0, torch 2.13.0, MPS available.

## Leakage guard

The acceptance criterion "no benchmark uses validation/test leakage to improve throughput" is not
a documentation promise — it gets a test. Concretely the harness must:

- build splits through the real `splitting` machinery, never by slicing rows directly;
- fit transforms through the real train-only path and assert the fitted-transform checksum was
  derived from `split == "train"` rows only, reusing the ML-203 fingerprint;
- never cache a materialized array across split roles; and
- measure `apply`/inference on validation and test rows that were never seen by fit.

A benchmark that got faster by pre-materializing all splits together would be measuring a
configuration the package refuses to run, so this guard protects the *validity* of the numbers,
not just scientific hygiene.

## Methodology correction — 2026-08-04

**The in-process RSS protocol specified above is invalid and has been replaced.** The first run
exposed it immediately, so it is recorded here rather than quietly fixed.

Measuring peak RSS in-process across repeats systematically *under*-reports peak allocation,
because CPython's allocator keeps a warm arena between repeats. A repeat can satisfy a
multi-megabyte NumPy allocation with no process RSS growth at all. Measured on
`rows500_pos1000_ch1`, materializing ~12 MB of arrays:

| Warmup passes | Median measured "peak" | Reported headroom |
| --- | ---: | ---: |
| 0 | 1,146,880 | 0.091 |
| 1 | 802,816 | 0.064 |
| 2 | 81,920 | **0.007** |

At two warmups the protocol reports 82 KB for a 12 MB workload. Under-reporting is the dangerous
direction for a memory guardrail: it makes the estimator look far safer than it is.

**Replacement:** memory cells are measured **once, in a fresh subprocess, with no warmup**
(`benchmarks/_isolated.py`, driven by `harness.measure_memory_isolated`). Timing keeps the
in-process warmup path, where warmup is correct and helpful. The two measurements are now separate
concerns rather than two readings of one run.

## RETRACTED — single-shot measurements, 2026-08-04

An earlier revision of this document reported, from one cold process per cell, that the headroom
invariant was violated at 500 positions (headroom 1.486) and that ~1.6 MB of unmodelled fixed
overhead made small workloads unsafe. **Both claims are withdrawn.** They were artifacts of
single-shot noise, which the same revision had flagged as a caveat and which turned out to be
large enough to invent the entire finding. The same cell measured three times gave peaks of
9,404,416, 1,769,472, and 3,014,656 bytes.

Recorded rather than deleted: the lesson is that one cold-process RSS sample cannot support any
claim about this estimator, and the retraction is the most useful thing an ML-702 reviewer can
learn from this section.

## Measurements — cold-process with repeats, 2026-08-05

Each repeat is an independent fresh interpreter. Headroom is reported against the **worst**
observed peak, not the median: a memory bound is a worst-case claim, and an estimator that covers
the typical run but not the worst run is not a bound.

### Critical cell, N=7 (`materialize`, 500 rows / 500 positions / 1 channel)

| Prewarm | Median peak | Worst peak | Headroom (median) | Headroom (worst) |
| --- | ---: | ---: | ---: | ---: |
| off | 2,392,064 | 5,537,792 | 0.378 | 0.875 |
| on | 2,408,448 | 3,768,320 | 0.380 | 0.595 |

Headroom stays below 1.0 in both conditions. The init-separation control (`prewarm`, a discarded
read against a trivial store before the baseline) barely moves the median — 0.378 vs 0.380 — but
substantially tightens the worst case, 0.875 to 0.595. **One-time initialization contributes to
the tail, not to the centre.**

### Row decomposition, N=5, prewarm on (500 positions / 1 channel)

| Train rows | Estimate | Median peak | Worst peak | Headroom (median) | Headroom (worst) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 6,330,600 | 2,342,912 | 3,506,176 | 0.370 | 0.554 |
| 600 | 12,661,200 | 3,571,712 | 13,041,664 | 0.282 | **1.030** |
| 1,200 | 25,322,400 | 10,256,384 | 19,857,408 | 0.405 | 0.784 |
| 2,400 | 50,644,800 | 21,676,032 | 32,751,616 | 0.428 | 0.647 |

Least squares against train rows:

```text
median: peak = -1,233,074 + 9,507 * rows    R^2 = 0.994
worst : peak =  2,720,456 + 12,950 * rows   R^2 = 0.956
estimator per-row (3 * bytes_per_row) = 21,102
```

### Conclusions

1. **No change to `_bytes_per_row` is warranted on this evidence.** The `2x`/`3x` constants hold.
2. **The linear term is genuinely conservative**, by roughly 1.6x against the worst-case slope
   (21,102 assumed vs 12,950 measured) and 2.2x against the median slope. It is over-conservative,
   not unsafe — the opposite of the retracted claim.
3. **There is no meaningful fixed overhead in the median case.** The median fit's intercept is
   *negative* and R² = 0.994: allocation scales cleanly linearly. The worst-case fit does carry a
   ~2.7 MB intercept, consistent with tail initialization cost rather than a per-call floor.
4. **One sample touched headroom 1.030** (600 train rows). In a distribution whose max ran 3.65x
   its median in that same cell, one 3% overshoot in five samples is not evidence of a real
   breach — but it is not proof of safety either. See the open caveat below.

### Bounded batch reads, N=5, prewarm on (500 positions / 1 channel)

`iter_batches` measured against the single-batch estimate as the split grows:

| Train rows | Batch estimate | Median peak | Worst peak | Headroom (worst) |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 450,176 | 1,998,848 | 7,766,016 | 17.3 |
| 600 | 450,176 | 2,310,144 | 7,979,008 | 17.7 |
| 1,200 | 450,176 | 3,063,808 | 15,400,960 | 34.2 |
| 2,400 | 450,176 | 19,120,128 | 21,708,800 | 48.2 |

Taken at face value this says streaming reads are unbounded and blow the batch estimate by up to
48x. **That reading is wrong, and the instrument is why.** Peak RSS is a *high-water mark*: pages
freed between batches are not returned to the OS, so the high-water grows with the number of
distinct allocations even when only one batch is ever live.

The per-batch RSS trajectory settles it. 38 batches, 2,400 train rows, batch size 64:

| After batch | RSS delta |
| ---: | ---: |
| 0 | 0 |
| 9 | 7,340,032 |
| 19 | 9,781,248 |
| 37 | 11,190,272 |

Growth in the first half is 9,781,248 bytes (~515 KB/batch); in the second half 1,409,024
(~78 KB/batch). A genuine per-batch accumulation would grow linearly at roughly the batch size
(~450 KB/batch) and never decelerate. Sharp deceleration toward a plateau is the signature of an
allocator arena reaching steady state.

**Conclusion: `iter_batches` holds bounded live memory. The acceptance criterion is met.**

### Trajectory run to asymptote — 300 batches, 2026-08-05

The 38-batch run above was suggestive but too short: quartiles of ~9 batches are dominated by
page-granularity noise, and a short window cannot distinguish a plateau from the early part of a
linear rise. Re-run at 32,000 rows / 19,200 train rows / 300 batches
(`sweeps.measure_batch_trajectory`):

| Quartile | Growth (bytes/batch) | Fraction of batch estimate |
| --- | ---: | ---: |
| Q1 | 739,937 | 164% |
| Q2 | 108,046 | 24% |
| Q3 | 27,676 | 6.1% |
| Q4 | 17,270 | **3.8%** |

Monotonic decay, 43x from Q1 to Q4, ending at 3.8% of the batch estimate. Accumulation is
excluded: retaining one batch per iteration would hold growth near 450 KB/batch in *every*
quartile and could not decay, because a fresh allocation happens on each pass. **The plateau is
demonstrated, not merely consistent with the data.**

For contrast, the same analysis over 38 batches returned quartiles of 276K / 31K / 55K / 519K --
non-monotonic and verdict "accumulating". The short run was actively misleading; run length is
part of the method, not a convenience.

#### One honest caveat on the tail

Final RSS after 300 batches was 94,240,768 bytes and did not fall after `gc.collect()`. Q4 growth
is small but not zero. Extrapolating the Q4 rate to a production-scale 1.3M-read experiment
(~20,300 batches at batch size 64) gives roughly **350 MB** of arena creep. That is an upper
bound, since the Q4 rate is itself still decaying, and it is affordable against the multi-GB
budgets in play — but it is not nothing, and a long-running streaming job should not be described
as flat-memory. The honest published claim is "sublinear and decaying", not "constant".

Two things follow for the published guidance, and they matter for ML-701:

1. `max_batch_bytes` bounds **one decoded batch**, not process RSS. A process streaming a large
   split will show RSS far above `max_batch_bytes` — here ~11 MB against a 450 KB batch estimate —
   while remaining perfectly bounded. Documentation must not promise that process RSS stays under
   the batch budget; users who watch RSS and see 25x the budget will otherwise file bugs against
   correct behaviour.
2. The right published claim is: **process RSS reaches a plateau that is independent of split
   size.** That is what "bounded" means here and it is what was measured.

## Sweep B — no training path streams. Both backends materialize.

Sweep B was specified as backend throughput across sklearn/Torch and cpu/mps. Reading the training
entry points to wire it up surfaced something that makes most of that matrix unreachable.

### The code

- `training/torch_backend.py` materializes **three times**: `train` (line 403), `validation`
  (line 404), and `test` (line 514). It never calls `iter_batches`.
- `training/sklearn_backend.py` materializes `train` at line 145 — **before** the incremental
  branch. Line 167 transforms the entire materialized array, and only then does line 175 chunk it
  for `partial_fit`.

So the advertised `incremental_fit` capability chunks an array that has already been fully
materialized and fully transformed. It reduces peak memory *inside the estimator*, and nothing at
the data boundary.

### Empirical confirmation

Against a dataset whose `max_materialization_bytes` is exactly one byte below its own train
estimate (1,020,240 bytes), every training path refuses at preflight:

| Backend | Family | Incremental | Result |
| --- | --- | --- | --- |
| sklearn | `bernoulli_nb` | **True** | REFUSED at preflight |
| sklearn | `bernoulli_nb` | False | REFUSED at preflight |
| sklearn | `random_forest` | False | REFUSED at preflight |
| torch | `residual_dilated_cnn` | n/a | REFUSED at preflight |

`incremental=True` and `incremental=False` refuse at the *identical* boundary.

### What this means at production scale

Refusal fires when `3 * train_rows * bytes_per_row > max_materialization_bytes`. With the default
2 GiB budget and a 60% train fraction:

| Positions | Channels | bytes/row | Max total rows | Shortfall vs 1.3M reads |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 1 | 7,034 | 169,611 | 8x |
| 1,000 | 1 | 14,034 | 85,011 | 15x |
| 1,000 | 2 | 26,036 | 45,823 | 28x |
| 5,000 | 1 | 70,034 | 17,035 | 76x |
| 5,000 | 2 | 130,036 | 9,175 | 142x |
| 20,000 | 2 | 520,036 | 2,294 | 567x |

**No training path in the package can consume a production-scale experiment.** The bounded
streaming reader exists, is correct, and — as the trajectory run demonstrates — genuinely holds
memory. No training code uses it.

This is an architectural gap, not a performance number, and it is arguably the most consequential
thing ML-700 has produced. Raising `max_materialization_bytes` is not a fix: at 1.3M reads and
1,000 positions the train split alone estimates ~33 GiB, so the budget is doing its job.

### Consequences for the rest of Sweep B

The originally planned throughput matrix (5,000 rows x {1,000, 5,000} positions x 2 channels x
2 backends x 2 devices) is still runnable — those cells sit under the ceiling — but it measures
the *supported* region only, and must be published as such. Benchmarking throughput at scales the
code refuses is not possible, and quoting a rows/second figure without the ceiling next to it
would be misleading.

Recommendation for the owner: this warrants its own work package (streaming training, or an
explicit documented ceiling with a refusal message that names the streaming alternative), not a
line in a benchmark report. Recorded as an ML-702-visible finding.

## Streaming-fit memory — ML-204 acceptance evidence, 2026-08-05

ML-204's outstanding acceptance criterion was that peak memory during a streaming *fit* is bounded,
verified with the trajectory method rather than a single peak reading. Measured via a new
`streaming_fit` mode that samples RSS as each batch leaves the reader, so sample points sit inside
the fit rather than around it.

| Run | Batches | Plateau | Plateau ÷ batch estimate | Final-quarter growth |
| --- | ---: | ---: | ---: | ---: |
| read (control) | 300 | 121 MB | 281x | 6.46% |
| sklearn streaming fit | 300 | 132 MB | 307x | 1.02% |
| Torch streaming fit (3 epochs) | 700 | 2,172 MB | **12,557x** | 0.01% |

**Both fits are bounded.** The Torch decile trajectory makes it unambiguous: +1,696.78 MB in the
first decile, then +0.00 MB, +0.25 MB across the last two. It plateaus and stays there. sklearn
decays 52x from first to final quartile.

### The finding that matters for published limits

**Process RSS during Torch training is 12,557x the data batch estimate.** The working set is
dominated by model activations, which no budget in the ML data plane models at all:
`partition_dataset` estimates *data* bytes, and nothing estimates activation bytes.

Consequences:

1. `max_batch_bytes` does not control Torch training memory. Tuning it will not stop an OOM whose
   cause is model width or depth.
2. A supported/slow/refused taxonomy for Torch training **cannot be derived from the data
   estimator alone**. Sizing a training job from the data budget under-predicts by four orders of
   magnitude here.
3. sklearn is different: its plateau (307x) is within the same order as reads (281x), because
   `BernoulliNB` holds only per-class feature counts. The gap is a neural-model property, not a
   streaming property.

This is a gap in the estimator's coverage, not a defect in streaming, and it should shape ML-701's
guidance: users size Torch jobs by architecture, not by batch budget.

### Torch memory does not track the data estimate — no fixed multiplier exists

Attempted to derive a sizing rule by varying batch size and sequence length at a fixed model
(`residual_dilated_cnn`, `block_channels` summing to 576 plus a 32-channel stem), 6,000 rows,
2 epochs, one run per cell:

| Batch | Positions | Data batch estimate | Plateau RSS | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 100 | 22,944 | 456 MB | 20,833x |
| 16 | 400 | 90,144 | 1,637 MB | 19,041x |
| 64 | 100 | 91,776 | 1,465 MB | 16,735x |
| 64 | 400 | 360,576 | 1,516 MB | 4,409x |

**No usable scaling law comes out of this, and the grid should not be published as one.** The ratio
spans 4,409x to 20,833x — a 4.7x spread — and the (64, 400) cell breaks the otherwise
roughly-linear trend entirely. Two confounds explain why the grid cannot answer the question as
built:

1. **Batch count is not held constant.** At 6,000 rows the batch-16 cells run ~450 sampled batches
   and the batch-64 cells ~114, so the arena has far more opportunity to grow in the former. The
   cells are not comparable.
2. **Single runs, and RSS variance is large** — established repeatedly above.

What the grid *does* support, and what is actually actionable:

- Torch plateau RSS sits **3-4 orders of magnitude** above the data batch estimate in every cell.
- The ratio is **not constant**, so no multiplier converts a data budget into a Torch memory
  prediction.
- Therefore **the data budget cannot size a Torch training job**, which was the conclusion already
  reached from the 12,557x measurement. This grid strengthens it rather than refining it.

A real activation-memory model needs controlled batch counts, repeats per cell, and variation of
the model config itself (`block_channels`, `stem_channels`, `hidden_dim`) rather than only the data
shape. That is its own piece of work, not a benchmark cell.

### Scoping decision this forces on ML-700

The package's remaining deliverable is "published limits distinguishing supported, slow, and
refused". The plan assumed those derive from the data estimator. For Torch they cannot. Two ways
forward, and this is an owner decision:

1. **Publish limits for reads, materialization, and sklearn only**; document Torch training as
   "size empirically, the data budget does not predict it", with the measured plateaus as
   illustrative rather than as a rule. Cheap, honest, leaves a real gap for users.
2. **Add an activation-memory estimator** so Torch training gets a genuine preflight like the data
   plane has. Substantially more work, arguably its own package, and it would need to model the
   registered architecture rather than data shape.

Recommendation: option 1 for ML-700, and record the estimator as a candidate follow-up. Shipping a
limits document that silently omits the dominant memory term would be worse than shipping one that
names it as unmodelled.

### The verdict flag is threshold-sensitive; do not gate CI on it

`measure_batch_trajectory` returns a `bounded`/`accumulating` verdict. It has now been recalibrated
once — the original criterion compared final-quartile growth to the *data* batch estimate, which
called a genuinely flat Torch trajectory "accumulating" purely because the model is larger than one
data batch. The replacement is plateau-relative.

The replacement is not reliable either. Under it, the **read control** flips to `accumulating` at a
6.46% tail, on a run whose earlier measurement gave 1.4% — pure run-to-run RSS variance, not a
behavioural change. This is the third time in this package that a fixed RSS threshold has flipped a
verdict between runs.

Treat the verdict as a reading aid. The defensible evidence is the trajectory **shape** across
orders of magnitude, and any CI gate needs repeats and a wide margin, not a single run.

### RSS is wrong in both directions

This package has now caught RSS misreporting twice, in opposite directions, and both errors were
large enough to manufacture a false finding:

| Workload shape | RSS error | Magnitude observed |
| --- | --- | --- |
| Repeated in-process | **Under**-reports (warm arena absorbs allocation) | 82 KB reported for a 12 MB workload |
| Streaming many batches | **Over**-reports (high-water accumulates freed pages) | 48x the true per-batch bound |

Neither is a defect in the package under test; both are properties of the instrument. Any future
ML-700 work must state which direction of error applies to the workload shape before quoting a
number, and the same applies to the eventual `thresholds.json`.

### Remaining caveats

1. **RSS has a heavy right tail this method cannot attribute.** Max/median reached 3.65x within a
   single cell. RSS counts pages the process touched, which is not the same as bytes the
   materializer allocated; GC timing and page-fault behaviour land in the signal. Whether the
   1.030 sample was real allocation or measurement artifact is **unresolved**, and resolving it
   needs a different instrument (tracemalloc plus explicit array accounting) rather than more RSS
   samples.
2. **RSS is nonetheless the right question for a guardrail.** The budget exists to keep a process
   inside physical memory, and that is an RSS question, not an allocated-bytes question. The
   instrument is imprecise but not the wrong target.
3. **One host only.** 128 GiB, Apple Silicon, Python 3.12.9, torch 2.13.0, sklearn 1.9.0.
4. **Arena creep is small but nonzero at length.** Q4 growth of 17,270 bytes/batch extrapolates to
   roughly 350 MB over a production-scale streaming run. Bounded and affordable, but "constant
   memory" would be an overclaim.

## Published limits taxonomy

Every cell resolves to exactly one of three states, which is what "distinguish supported, slow, and
refused" requires:

| State | Definition |
| --- | --- |
| **Supported** | Completes within budget, `headroom <= 1.0`, median wall time under the cell's threshold |
| **Slow** | Completes correctly and within budget, but exceeds the wall-time threshold. Published with a recommended alternative (larger batch, fewer channels, Torch instead of sklearn) |
| **Refused** | Raises `MLMemoryBudgetError` at preflight, before allocation. Must be refused *deterministically* and the message must name the estimate, the budget, and a concrete remedy |

The refusal message is itself under test. `partition_dataset.py:478-482` currently says
"use iter_batches() or raise the explicit budget" — Sweep A must confirm every refusal path
carries an equally actionable message, since "failures provide actionable memory/batch
suggestions" is a named acceptance criterion.

## Regression thresholds

Thresholds are committed as data, not as assertions scattered through test bodies, so ML-702 can
review them in one place and CI drift is visible in a diff.

- Memory: `headroom <= 1.0` is a hard failure. `headroom < 0.25` on any cell is a soft warning
  (the estimator is over-conservative there and the constants deserve revisiting).
- Throughput: per-cell median wall time may regress by at most 50% against the committed baseline
  before failing. Chosen loose deliberately — this is a shared laptop/CI signal, not a
  microbenchmark, and a tight bound would produce false failures.
- Refusal boundaries: the row count at which each cell flips to refused must match the analytic
  table above exactly. A change there means someone altered the estimator constants, which must be
  a deliberate, reviewed edit.

Only a small representative subset runs in CI (`integration` marker); the full sweep is an
operator-invoked script.

## File map

- `src/smftools/machine_learning/benchmarks/__init__.py` — new
- `src/smftools/machine_learning/benchmarks/harness.py` — cell definition, RSS watchdog, repeat
  policy, environment capture, JSONL emission
- `src/smftools/machine_learning/benchmarks/fixtures.py` — parameterized real-store builder
  generalized from the integration `_write_source`
- `src/smftools/machine_learning/benchmarks/sweeps.py` — the four sweeps
- `src/smftools/machine_learning/benchmarks/thresholds.json` — committed regression thresholds
- `tests/integration/machine_learning/test_ml_scale_qualification.py` — CI subset
- `docs/source/ml_performance.md` — published limits (drafted here, finalized in ML-701)

Reuses `smftools.perf_log` conventions (JSONL, one object per line, `summarize_perf_logs`-shaped
records) rather than inventing a second performance-record format. It does **not** reuse the
`PerfLogger` ContextVar itself — that is wired to stage logging and would tie benchmarks to a
running pipeline stage.

## Test list

1. Headroom invariant holds on the CI subset (estimate is an upper bound on measured peak).
2. Refusal fires exactly at the analytic boundary, and one row below it succeeds.
3. Refusal happens at preflight — assert no large allocation occurs, by checking peak RSS stays at
   baseline through the raised error.
4. Refusal messages name estimate, budget, and a remedy.
5. Worker shards at `num_workers` ∈ {1,2,4,8} produce identical row sets (reuses the existing
   determinism guarantee) and per-worker peak stays bounded.
6. Transform fingerprint derives from train rows only (leakage guard).
7. Environment record round-trips and is complete.
8. Threshold file parses and every CI cell has a threshold entry.

## Owner decisions — resolved 2026-08-04

1. **CUDA gap:** accepted as a documented limitation. Device axis is cpu + mps.
2. **50,000-row cells:** operator sweep only, excluded from CI.
3. **Threshold provenance:** commit laptop baselines; CI compares ratios, not absolutes.

## Next steps

Done: repeated cold-process measurement (`harness.measure_memory_repeated`), the init-separation
control (`prewarm`), and the estimator verdict — no change warranted.

Done: the 300-batch trajectory demonstrates the plateau (Q1→Q4 decay of 43x to 3.8% of the batch
estimate).

1. Decide how to publish the ~350 MB extrapolated arena creep at production scale — as a
   documented characteristic, or investigate whether it is reclaimable.
2. Resolve the heavy-tail attribution question with a second instrument (tracemalloc plus explicit
   array accounting) rather than more RSS samples. Until then, published limits should quote
   worst-case headroom and say plainly that the tail is not fully attributed.
3. Implement Sweeps B and D.
4. Commit `thresholds.json` and the CI integration test. Threshold on **worst-case** headroom for
   materialization, and on **trajectory deceleration** for streaming — never on streaming
   high-water RSS, which the table above shows is meaningless as a bound.
5. Consider whether the measured 1.6-2.2x over-conservatism is worth reclaiming. Loosening the
   constants would admit larger workloads, but it moves refusal boundaries package-wide and is an
   ML-702-visible change. Recommendation: leave them alone. An over-conservative memory bound
   costs a config override; an under-conservative one costs an OOM mid-run.
