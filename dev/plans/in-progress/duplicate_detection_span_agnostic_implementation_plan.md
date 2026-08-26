# Span-agnostic duplicate detection (`DSA`)

**Status:** in progress. Drafted on `fix/duplicate-detection-span-agnostic`, cut
from `3301793`. Not yet committed, so the evidence column names tests rather
than shas; replace them as items land.

Motivated by `F51` in `logs/pipeline_findings.md`, which reproduces the defect
and dates it. This plan does not restate that investigation -- read it first.

## The defect in one paragraph

`popcount_hamming_windowed` scores only overlapping valid positions, so the
*metric* has always been span-blind. Candidate generation is not: the sort key in
`_process_group` fills unmeasured positions with `-1`, so two reads that agree
perfectly over their overlap but differ in span diverge in the leading key
columns, sort far apart, and never land within `window_size` of each other. The
hierarchical top-up was the only span-blind step, and `e18d593` (2026-07-20)
capped it at 5,000 representatives -- above default chunk sizes -- so it stopped
running on exactly the large, fragmented groups that needed it. Measured recall
on differing-span duplicate pairs: **0.019**, against 1.000 with the pass intact.

## Approach

Add **anchor-window banding**: tile the site axis, and for each tile band only
the reads whose measured extent covers the whole tile, keying on that tile's
columns alone. Every read in such a subset measures every key column, so the
`-1` fill cannot enter the key and span drops out of the ordering entirely.

### The reach condition

Windows start at multiples of the stride, so a pair whose overlap is `V`
positions long shares a window only if some aligned window falls entirely inside
that overlap. In the worst phase that requires:

```text
V >= anchor_window_sites + anchor_window_stride_sites - 1
```

A pair overlapping by less than that is never banded together, however well its
calls agree. `min_overlap_positions` is the user's own statement of the shortest
overlap worth scoring, so the geometry is **derived from it** rather than
guessed: width is half the reach, stride makes up the rest, and every pair the
comparison would accept can reach a window.

The first draft of this plan fixed the width at 100 sites with a stride of 100,
needing **199** positions of overlap against a declared minimum of 20. On two
clean size classes that was invisible -- both reads covered the same wide
windows. Under random fragmentation it was not: see the measurements below.

Three properties make this the right shape:

- **Distances are untouched.** Comparison still runs over each pair's full
  overlap. Only which pairs get compared changes, so nothing about the
  calibrated `distance_threshold` moves.
- **It is near-linear.** A read participates in roughly `span / stride` windows,
  each keyed on `anchor_window_sites` columns -- comparable to the two
  full-width natural-order passes it sits beside, and nothing like the O(n^2)
  pass it replaces.
- **It composes.** A read covering several windows is banded in each, and
  union-find already absorbs merges from any chunk, round or pass in any order.

**Restoring the uncapped hierarchical pass was rejected.** The OOM that motivated
the cap was real, and it would not restore the old *result* anyway: the pass then
ran across a whole group (up to ~46,000 reads), whereas it can now only run
within a chunk, so cross-chunk differing-span pairs would still depend on the
survivor reshuffle. Fixing candidate generation fixes both.

## Work items

| item | status | evidence |
|---|---|---|
| `DSA-01` anchor-window planner and anchored banding passes | drafted | `test_differing_span_duplicates_are_clustered`, `test_plan_anchor_windows_selects_only_covering_reads` |
| `DSA-02` derive the window geometry from `min_overlap_positions` | drafted | `test_random_fragmentation_recovers_every_comparable_pair`, `test_derived_geometry_reaches_the_configured_minimum_overlap` |
| `DSA-03` config surface and semantic fingerprint entries | drafted | `duplicate_detection_span_agnostic_banding` and three siblings in `semantic_upgrade.py` |
| `DSA-04` route the hierarchical-skip notice to the logger | drafted | `test_hierarchical_topup_skipped_above_representative_cap` |
| `DSA-05` qualify on a real fragmented run; revisit the hierarchical cap | measured | see Real-data qualification below; cap revisit still open |

### `DSA-01` — anchored banding

`_plan_anchor_windows` in `flag_duplicate_reads.py` returns
`(anchor_start, anchor_end, row_indices)` per window, selecting reads by the
per-read measured extent captured before the dense array is dropped.
`cluster_pass` gained a `rows` argument so a pass can band a subset; its
`column_order` argument already tolerated an arbitrary column list, so an anchor
window passes only its own columns.

### `DSA-02` — geometry derived from the minimum overlap

The width and stride default to a derivation from `min_overlap_positions` (see
the reach condition above) rather than to fixed site counts. A second, narrower
safeguard remains for the pathological case where even the derived width exceeds
the median measured read span: a window wider than the reads plans nothing and
silently restores the bug, the same class of invisible failure as `F51` itself.

**A wrong version, recorded so it is not retried.** The first attempt narrowed
the width when it exceeded the *median read span*. That is the wrong quantity.
It fixed the case where reads are uniformly short, and did nothing for the case
that actually matters -- a long read and a short read overlapping by less than
the window width, which is most pairs under random fragmentation. What governs
whether a pair can be banded is the length of *their overlap*, never the length
of either read. Keying on read length looked right because the failing fixture
at the time had every read the same length.

### `DSA-03` — configuration

| key | default | meaning |
|---|---|---|
| `duplicate_detection_span_agnostic_banding` | `True` | Enable anchored passes. `False` restores pre-fix candidate generation. |
| `duplicate_detection_anchor_window_sites` | `0` | Window width in comparison columns. `0` derives it from `min_overlap_positions` so the reach condition holds. |
| `duplicate_detection_anchor_window_stride_sites` | `0` | Window start spacing. `0` derives it alongside a derived width, or tiles non-overlapping under an explicit width. |
| `duplicate_detection_max_anchor_windows` | `512` | Window ceiling. The stride widens to fit rather than truncating, so no end of the reference loses its anchored pass. |

Every way of breaking the reach condition is logged: an explicit width too wide
for the minimum overlap, and a ceiling that forces the stride past it. Silent
recall loss on short overlaps is precisely how `F51` survived a month.

All four join the preprocess semantic fingerprint: they change dedup results, so
they must invalidate a preprocess generation rather than silently reusing one
built the other way.

### `DSA-04` — the skip becomes visible

`default.yaml` has always described the hierarchical skip as "logged", but it was
a `warnings.warn` and nothing calls `logging.captureWarnings`, so it never
reached a run log. It is now `logger.warning` and names whether anchored banding
is active. One existing test moved from `pytest.warns` to `caplog`; its
assertion that linkage does not run above the cap is unchanged.

### `DSA-05` — measured; the cap revisit stays open

Cost and real-data behaviour are now measured (see below). What remains open is
whether `duplicate_detection_hierarchical_max_representatives` can drop. The
evidence says it nearly can -- with banding on, capping the top-up at 50 instead
of 5000 changes the result by 0.01 points and one cluster -- but that is one run,
and lowering it would remove the only exact pass. Leave it until a second
library agrees.

## Measurements

### Real-data qualification

A rapid-kit (transposase, single-ended barcode) deaminase library: 14,358 reads
surviving raw ingestion, median read length 585, p10 144, p90 1,960. Four
preprocess generations over one shared raw generation, differing only in the
dedup keys.

| variant | banding | hierarchical cap | duplicates | unique clusters | preprocess stage |
|---|---|---|---|---|---|
| A | off | 5000 | 53.03% | 5,496 | 175.7s |
| B | on | 5000 | **61.47%** | **4,508** | 202.8s |
| C | off | 50 | 52.50% | 5,558 | 204.4s |
| D | on | 50 | 61.46% | 4,509 | 204.9s |

Three things fall out, and the first is a correction.

**The hierarchical pass was never carrying this.** `F51` framed the defect as
the cap disabling the only span-blind step. A against C -- that pass running
versus skipped -- differs by **0.53 points**. The sort-key span sensitivity was
under-calling differing-span duplicates even when the pass ran; the cap made a
pre-existing defect worse rather than creating it. The synthetic fixture showed
the pass recovering 1.000 because uniform-random patterns and identical
truncation make mean-imputed euclidean distance far more separable than real
footprint data is. **A synthetic fixture that agrees with the story you already
have is the one to distrust.**

**The recovered pairs are well supported.** Of the 990 reads newly called
duplicate, 89.4% overlap their best cluster partner by 500+ positions, median
overlap 1,537; exactly one pair sits in the 30-50 position range and none in
20-30. They are not artifacts of the `min_overlap_positions` floor. 30% of the
pairs differ by more than 1.5x in span -- the predicted signature.

**Cost is +15%, not the ~10x below.** The synthetic figure came from a dense
2,000-read x 2,000-site group where anchored passes dominate the work; a real
preprocess stage spends most of its time elsewhere.

Apparent library complexity falls **18%** on this run. That is the defect being
corrected, not a regression, but it changes published numbers and belongs in
release notes.

### Random fragmentation

1,000 molecules over a 2,000-site locus, each observed twice at independent
random spans, hierarchical top-up capped out. Recall is scored only over pairs
whose overlap reaches `min_overlap_positions=20`, since shorter pairs are
correctly uncomparable. Bucketed by overlap length:

| fragment sizes | geometry | overall | 20-49 | 50-99 | 100-199 | 200-499 | 500+ |
|---|---|---|---|---|---|---|---|
| 50-1500 | banding off | 0.360 | 0.24 | 0.08 | 0.21 | 0.28 | 0.52 |
| 50-1500 | fixed 100/100 (first draft) | 0.904 | 0.24 | 0.08 | 0.86 | 1.00 | 1.00 |
| 50-1500 | derived (default) | **1.000** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 50-300 | banding off | 0.711 | 0.23 | 0.71 | 0.87 | 1.00 | -- |
| 50-300 | fixed 100/100 (first draft) | 0.745 | 0.23 | 0.71 | 0.96 | 1.00 | -- |
| 50-300 | derived (default) | **1.000** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The fixed-width draft is *worse than useless* on short fragments -- 0.745
against 0.711 for no banding at all -- because no window fits inside a 50-99
position overlap. The bucketed view is what exposed this; the overall figure on
the wide-size-range row (0.904) looks respectable and hides it entirely.

Zero false merges in every row, and a control of 800 genuinely distinct
randomly-fragmented reads yields 800 clusters.

### Cost

Same 2,000-read group, wall-clock inside `_process_group`:

| geometry | recall | windows planned | seconds |
|---|---|---|---|
| banding off | 0.360 | 0 | 2.9 |
| fixed 100/100 | 0.904 | 64 | 11.3 |
| derived (default) | 1.000 | 181 | 29.8 |

**Correctness is ~10x the cost of the broken behaviour on this synthetic
group.** That figure did not survive contact with a real stage, which measured
+15% -- the group above is dense and wide, so anchored passes dominate its work
in a way they do not dominate a real preprocess run. Kept here because it is the
worst case the geometry can produce, and because the lever if it ever does bite
is `duplicate_detection_anchor_window_sites`, which trades short-overlap recall
for speed and says so in the log.

### Two size classes

Synthetic groups of 400 sites, two reads per molecule (one full-span, one
truncated at both ends), hierarchical top-up capped out to isolate banding,
defaults otherwise. Recall is the fraction of true pairs sharing a cluster;
the cluster count detects recall bought by over-merging.

| molecules | span offset | banding | recall | clusters (expected) |
|---|---|---|---|---|
| 500 | 0 | off | 1.000 | 500 (500) |
| 500 | 0 | on | 1.000 | 500 (500) |
| 500 | 120 | off | 0.366 | 817 (500) |
| 500 | 120 | on | 1.000 | 500 (500) |
| 500 | 200 | off | 0.008 | 996 (500) |
| 500 | 200 | on | 1.000 | 500 (500) |
| 2,000 | 120 | off | 0.018 | 3,963 (2,000) |
| 2,000 | 120 | on | 1.000 | 2,000 (2,000) |
| 2,000 | 200 | off | 0.001 | 3,998 (2,000) |
| 2,000 | 200 | on | 1.000 | 2,000 (2,000) |

Full recall arrives at exactly the expected cluster count in every case, and a
control of 2,000 genuinely distinct reads yields 2,000 clusters with banding
both on and off -- so the recovery is not over-merging.

**What this rests on.** Synthetic data plus ten unit tests, not a side-by-side
production run. The span geometry is one truncation pattern; real fragmented
libraries vary more. `DSA-05` is what converts this into a measured claim.

**A negative control that had to be repaired, recorded so it is not mistaken for
a regression.** `test_natural_order_only_misses_early_divergent_pair` asserts a
pair is *missed* without permutation passes. Anchored banding catches that pair
too -- correctly, since the two reads differ only at column 0 -- which made the
control vacuous. Anchoring is now held off in that fixture so it still isolates
what it was written to isolate. A negative control that silently starts passing
for the right reason is indistinguishable from one that never worked.
