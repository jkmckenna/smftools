# Machine learning: performance and limits

Measured limits for the ML data plane. The exhaustive machine-readable table lives in
`tests/acceptance/ml_scale_thresholds.json`; this page gives the rules behind it, so you can size
a job without looking anything up.

Figures were measured on one developer laptop (Apple Silicon, Python 3.12, torch 2.13). Treat
ratios and shapes as the finding, not absolute seconds.

## Reading a split: bounded, or refused

Two ways to get rows out of a partition store, with different guarantees.

`iter_batches()` holds **bounded** memory regardless of split size. Process RSS climbs while the
allocator arena settles and then plateaus; over a 300-batch read, growth per batch fell from
739,937 bytes in the first quarter to 17,270 in the last.

`materialize()` runs a preflight and **refuses** rather than risking an out-of-memory kill. The
refusal is a closed form:

```text
bytes_per_row = 2 * (n_positions * n_channels * 6 + n_channels + n_positions + 8 + 8)
refused when   3 * split_rows * bytes_per_row > max_materialization_bytes   (default 2 GiB)
```

Worked examples at the default budget:

- **1,000 positions, 1 channel** — 14,034 bytes/row, so a split above roughly **51,000 rows** is
  refused.
- **5,000 positions, 2 channels** — 130,036 bytes/row, so a split above roughly **5,500 rows** is
  refused.

A real SMF experiment runs on the order of 10⁶ reads. **At production scale, materialisation is
refused at every width.** That is the budget working correctly, not a bug to tune around — at
1.3M reads and 1,000 positions the split alone estimates about 33 GiB.

:::{note}
`max_batch_bytes` bounds **one decoded batch**, not process RSS. A process streaming a large split
will sit far above the batch budget while remaining perfectly bounded. Watching RSS and seeing
25× the batch budget is expected behaviour, not a leak.
:::

## Training

| Path | Behaviour |
| --- | --- |
| sklearn, families declaring `incremental_fit` | Streams. No ceiling. `bernoulli_nb` qualifies today. |
| sklearn, other families | Requires materialisation, so bounded by the ceiling above. `logistic_regression` and `random_forest` have no `partial_fit`. |
| plain PyTorch | Streams. No data-side ceiling. |

Refusals name the streaming-capable alternatives rather than leaving you to guess.

Two properties of the streaming Torch path are deliberate and affect results:

- **Shuffling is buffered, not global.** Rows are permuted within a window of decoded batches,
  because random access is exactly what streaming avoids. Fitted weights therefore differ from a
  materialised fit at the same seed. The window is `shuffle_buffer_batches` in
  `TorchTrainingConfig` — part of the persisted training provenance, precisely because it changes
  the model.
- **Every epoch re-reads the store.** In-memory training decodes once for all epochs; streaming
  decodes once per epoch. It trades wall time for the ability to run at all.

## Torch memory is not modelled

The most important limit on this page is one the package does **not** enforce.

Every memory budget in the ML data plane models *data* bytes. During PyTorch training, process
memory is dominated by *model activations*, which nothing estimates. Measured: a streaming Torch
fit plateaued at 2,172 MB — **12,557×** the data batch estimate for the same shape. Across batch
sizes and sequence lengths the ratio ranged from 4,409× to 20,833×, so no fixed multiplier
converts a data budget into a Torch memory prediction.

Practically:

- `max_batch_bytes` will not prevent an out-of-memory kill caused by model width or depth.
- Size PyTorch training runs **empirically**. Measure one, then scale.
- sklearn is unaffected — its streaming fit plateaued at 307× the batch estimate, the same order
  as a plain read at 281×, because the estimator holds only per-class counts.

Closing this gap is scoped as future work, gated on a real production out-of-memory event or a
workflow that needs a training-memory preflight.

## Sharding

Splitting a read across workers costs nothing in arithmetic terms. Per-shard throughput was flat
at 549–567 rows/second across 1, 2, 4, and 8 workers, and shard time scaled as exactly 1/N.

This does **not** mean N workers finish N times sooner. Shards were measured one at a time in a
single process; real parallel workers contend for disk and memory bandwidth, and that contention
has not been measured.

## Choosing an explanation chunk size

`example_batch_size` controls how many observations are attributed at once. Two things about it
are easy to get wrong.

**Wall time has an interior optimum — larger is not faster.** Over 400 rows × 400 positions with
Saliency, a chunk of 8 was fastest; a chunk of 1 took 1.99× longer and a chunk of 512 took 2.76×
longer.

**It is a scientific parameter, not only a performance knob.** Attributions repeat bitwise at a
fixed chunk size but differ by roughly 1.1×10⁻³ relative across chunk sizes, consistent with
batch-size-dependent convolution kernel selection accumulating through the network. Provenance
stays honest — `example_batch_size` is part of the explanation request and therefore its identity,
so two chunk sizes are recorded as different results rather than colliding.

**Pick a chunk size and hold it fixed across runs you intend to compare.** Tuning it for that
2.76× speedup between runs silently perturbs the attributions being compared.

## Known gaps

Recorded rather than omitted:

- **No CUDA measurements.** The measuring host had CPU and MPS only.
- **Explanation memory across chunk sizes is unmeasured.** The sweep runs chunk sizes sequentially
  in one process, and a warm allocator arena under-reports later cells badly enough that the
  numbers would mislead.
