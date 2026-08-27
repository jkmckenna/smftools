# Basecalling as a stage, and choosing a read source (`BCS`)

**Status:** in progress. Phase 1 (`BCS-01`-`BCS-04`) implemented on
`feature/bcs-01-source-selection`. Of Phase 2, `BCS-05` (`basecall` as a
generation-publishing stage) is implemented on
`feature/bcs-05-basecall-stage`, standalone -- see its note for what it
deliberately does not yet integrate with. `BCS-06` (`full` runs it ahead of
raw) is implemented on `feature/bcs-06-full-recipe` -- see its note for the
narrower shape that shipped. `BCS-07` (per-source POD5 identity, recording
only) is implemented on `feature/bcs-07-pod5-identity`. `BCS-10`
(config-free `--input/--output` invocation) is implemented on
`feature/bcs-10-config-free-basecall`. `BCS-11` and all of Phase 3 remain
proposed.

**Repository state reviewed:** `69c24e4` — recorded while writing.

## Problem

An experiment's data directory usually holds more than one representation of the
same reads: `pod5/`, a `fastq_pass/` tree, and sometimes a `basecalls/` directory
of BAMs from whichever model was run. Today that directory cannot be used as
`input_data_path` at all — discovery recurses by default, finds more than one
recognized kind, and `ExperimentConfig.from_var_dict` refuses with
`input_data_path contains mixed recognized input types`. The working practice is
to point at one subdirectory by hand and record the reason in a config comment.

What a user actually wants to express is a *model*, not a path: use the reads for
the configured basecalling model if they already exist, and basecall from signal
only when they do not.

Two things follow, and this plan covers both because neither is coherent alone:

- **Source selection.** Given a directory with several representations, pick the
  one that satisfies the configured model and the experiment's capability needs.
- **Basecalling as a stage.** Selection's "otherwise, basecall" branch is
  currently an unnamed step buried inside raw, so it cannot be run, skipped,
  inventoried, pinned, or archived on its own.

## Current behaviour (verified at `69c24e4`)

**Basecalling is inside raw.** `load_adata_core` (`cli/load_adata.py`) decides
`basecall = cfg.input_type == "pod5"` and runs `canoncall`/`modcall` inline.
There is no `basecall` command; `full` runs `raw -> preprocess -> spatial -> hmm
-> latent` (`cli/recipes.py`).

**Its output is an intermediate, not an artifact.** The result lands under
`<run>/raw_outputs/intermediates/dorado-basecalling/<revision>/`, addressed by an
`IntermediateSpec` whose identity is the model, the options, and
`artifact_checksum(pod5_input)`. Reuse works, but the outputs are not stage
generations: `smftools experiment generations` does not list them, they cannot be
pinned, and nothing treats them as a durable product.

**Reuse requires the signal.** Because the spec's identity checksums the POD5
input, deciding "these basecalls are still valid" re-reads the POD5s. With the
archive detached that check cannot run, so `PSR-01` lets the config load and the
work still cannot proceed.

**The provenance primitives already exist**, spread across three places:

| source | where the model is recorded | already read by |
|---|---|---|
| FASTQ | `basecall_model_version_id=` in the read header comment | nothing |
| BAM | `basecall_model=` inside `@RG` `DS:` | `pipeline/rebasecall_basecall.py` |
| POD5 | run conditions -> resolved model name | `informatics/dorado_model.py` |

`dorado_model.py` already turns a short name (`hac`) into a concrete model for
given run conditions, and `_model_version_key` already orders versions.

## Design

### Selection is a policy, stated once

A source satisfies the config when **all** of:

1. **Model matches.** A bare short name (`hac`) is satisfied by any version of
   that family, newest winning. An explicitly versioned name must match exactly.
   Ordering comes from the existing `_model_version_key`.

   **The Dorado version does not participate.** A basecaller release that leaves
   the model identity unchanged does not invalidate a match: gating on it would
   force a re-basecall of every archived run on each instrument software update,
   for reads the model says are equivalent. The observed Dorado version is
   recorded in the basecall manifest and reported, so a difference stays visible
   and auditable, but it never causes selection to reject a source or a stage to
   recompute. Decided 2026-08-26; previously an open question.
2. **Capability suffices.** Model identity alone is not enough. A canonical FASTQ
   carries no MM/ML, which is fine for `deaminase` and `conversion` but
   disqualifying for `direct` — a rule the config layer already enforces for
   FASTQ input. The resolved input manifest already records
   `modification_capability` per source; selection reads it rather than
   re-deriving it.
3. **The bytes are reachable.** A source whose volume is detached does not
   satisfy anything. This only exists as a state because of `PSR-01`, and it is
   why selection must consult availability instead of picking a path it cannot
   read.

Where several sources qualify, prefer **BAM over FASTQ** (tags, read groups and
any alignment survive) and **pass over fail**. A `fastq_fail` tree is never a
selectable source; sweeping it in is one of the things the current manual
workaround exists to avoid.

If nothing qualifies and POD5 is present, basecall. If nothing qualifies and
POD5 is absent, that is an error naming what was found, which model each source
carries, and which of the three rules it failed — never a bare "no supported
input files".

### `basecall` becomes a stage

It is promoted out of raw and given the same lifecycle every other stage got in
2.21.0: a `basecall_outputs/` directory publishing immutable, checksummed
generations selected by `current.json`.

The division of labour becomes clean, and is worth stating because it is the
whole point of the split:

```text
basecall   signal -> reads        (expensive, GPU, reusable across experiments)
raw        reads  -> ragged store (cheap, CPU, experiment-specific)
```

`full` becomes `basecall -> raw -> preprocess -> spatial -> hmm -> latent`, where
`basecall` skips whenever selection finds a satisfying source — which is every
run that was basecalled on the instrument. A FASTQ-input experiment therefore
sees no behaviour change at all.

**This is deliberately not the re-basecalling program.** `SRB` re-basecalls an
*already-ingested* experiment and publishes the result as a lineage held beside
the original, because changing the reads under a finished analysis is a new
interpretation of it. `BCS` covers the first ingestion, where there is nothing to
hold beside. The two must not become two ways to do one thing: if an experiment
already has a raw generation, changing the model is an `SRB` lineage operation,
and `basecall` should say so rather than quietly producing a second answer.

### When basecalls and signal disagree

A `basecalls/` directory can stop describing the signal beside it, and smftools
is one of the causes: `max_basecall_reads` and `subsample_pod5_for_basecalling`
deliberately basecall a random subset. POD5s also get added by a resumed run,
pruned after archiving, or copied from the wrong run.

"Mismatch" is therefore not one condition. It has three shapes with opposite
correct answers, so the policy classifies before it responds:

| shape | meaning | response |
|---|---|---|
| basecalls ⊃ signal | references POD5s no longer present | **proceed**, silently |
| basecalls ⊂ signal | covers fewer reads than the signal present | **proceed only if the subsetting was recorded as deliberate**, else refuse |
| disjoint | different reads entirely | **refuse** |

The superset case is not a defect -- it is the end state this program exists to
reach, where the signal has been pruned because its derivative is enough. A
blanket refusal on mismatch would break exactly the workflow being built.

The subset case is the dangerous one, and the reason it cannot simply warn: a
run silently analysed at a fraction of its depth reports fewer molecules, which
reads as a biological result rather than a defect. So it is allowed only when the
basecall manifest records that the subsetting was intended -- `max_basecall_reads`
in force at the time, with the sampled count and seed. Absent that record, it
refuses and names the sources it cannot account for.

This turns a judgement call into a recorded fact, the same move that separated
`offline` from `missing` in `PSR-01`: the state that looks ambiguous from the
outside is unambiguous to whoever wrote it, so write it down at the point of
knowledge rather than inferring it later.

**A detection gap has to close first.** The current spec records
`artifact_checksum(cfg.input_data_path)` -- one digest over the whole input path,
which distinguishes *different* from *same* but carries no shape. Classifying
subset against disjoint needs per-source checksums. The resolved input manifest
already records a `sha256` per row, so the primitive exists; the basecall step
simply does not use it yet. `BCS-07` owns that, since it is the same record that
lets a generation validate without re-reading signal.

### Two ways to invoke it

Basecalling a drawer of runs off an archive drive is data preparation, not
experiment work, and demanding a full experiment config per run to do it is
friction that pushes people back to calling Dorado by hand. So `basecall` takes
either form:

```shell
smftools basecall <config.csv>
smftools basecall --input <pod5-dir> --output <dir> --model hac --model-dir <dir> \
    [--kit <barcode-kit>] [--modifications 5mC_5hmC] [--device auto]
```

One command and one core, not two implementations. The config form supplies the
same model parameters the config already carries — `model`, `model_dir`,
`barcode_kit`, `barcode_both_ends`, `trim`, `device`, `emit_moves`, and
`mod_list` for a `direct` experiment — and explicit flags override them where
both are given.

**Both forms publish the same artifact.** The config-free form writes a basecall
generation into `--output` exactly as the config form writes one into the run
root, which is what makes the batch workflow work: basecall a drawer of runs with
no configs in sight, then point experiments at the results and have selection
(`BCS-03`) recognise them like any other source. A config-free run is not a
lesser product.

It is a **top-level command rather than a member of `experiment` or `project`**,
because in its config-free form it is scoped to neither. Per
`src/smftools/cli/AGENTS.md` that choice is the first step of adding a command,
so it is recorded here deliberately.

**Overriding the model against an already-ingested experiment is refused**, with
a pointer to `SRB`. That combination — an existing raw generation plus a
different model — is exactly a re-basecalling lineage, and silently producing a
second set of reads beside a finished analysis is the failure that boundary
exists to prevent.

### Surviving a detached archive

A published basecall generation records the POD5 identity it consumed —
checksums and durable origin identity, which `SRB` already established in 2.21.0
— so validating it never needs to re-read the signal. This is the same shape as
`PSR-01`'s recorded-identity recovery, and for the same reason: an identity that
must be re-derived from the source is an identity that stops working the moment
the source is archived.

Once that holds, a run whose basecalls live in the analysis tree can be
processed end to end with the POD5 archive unplugged.

### Getting basecalls back to the archive

Basecalls are derived and regenerable, so archiving them is an optimisation
rather than a duty. It is a large one: it is what makes the POD5s optional for
everything downstream.

Layout, as a sibling of the signal rather than mixed into it:

```text
data/<run>/
├── pod5/
└── basecalls/
    └── <model>@<version>/
        ├── basecall_manifest.json   # model, dorado version, POD5 identity, checksums
        └── *.bam
```

Keying the directory by model is what lets several models coexist and what lets
selection answer "is there a derivative for this model?" from the directory
listing, before opening anything.

Write-back is an explicit command, never automatic, and is idempotent and
checksum-verified so an interrupted transfer resumes rather than duplicating.

### Batch basecalling must not thrash the archive

Basecalling a batch of experiments off one HDD has a failure mode that has
nothing to do with correctness. Reading the next run's POD5 is a large sequential
read; writing the previous run's basecalls is a large sequential write. On one
spindle, interleaving them turns both into seek storms and can cost more
throughput than the basecalling itself saves.

The policy is therefore structural rather than advisory:

- **Basecalls are always written to the analysis tree first** — local disk or
  SSD — never straight to the archive volume.
- **Write-back is a separate phase**, run after the batch, so the archive sees
  one long read phase and then one long write phase instead of alternating.
  Making it a distinct command rather than a flag inside the batch loop is what
  makes this the default rather than a thing to remember.
- **Overlap is permitted only across devices.** `PSR-08`'s volume identity is
  what makes this checkable: if source and destination resolve to different
  volume ids, concurrent read and write is fine and should not be prevented.
- **Write-back processes one run at a time**, keeping each run's output
  contiguous rather than interleaving several.

Prefetching the next run's POD5 while the GPU works is read-only against the
archive and stays fine.

## Work items

| item | status | evidence |
|---|---|---|
| `BCS-01` accept a mixed-source directory as `input_data_path` | implemented | `tests/unit/config/test_mixed_source_directory.py` |
| `BCS-02` read basecall-model provenance from FASTQ, BAM and POD5 | implemented | `informatics/basecall_provenance.py`; verified on a real run's FASTQ and BAM |
| `BCS-03` model-match and capability policy | implemented | `test_direct_modality_refuses_a_canonical_fastq`, `test_bare_selector_accepts_any_version_of_its_family` |
| `BCS-04` preference ordering, availability-aware | implemented | `test_bam_is_preferred_over_fastq_when_both_qualify`, `test_fastq_fail_is_never_selected` |
| `BCS-05` `basecall` as a generation-publishing stage | implemented | `tests/unit/informatics/test_basecall_generation.py`, `tests/unit/informatics/test_basecall_execution.py`, `tests/unit/test_basecall_cli.py` |
| `BCS-06` `full` runs basecall before raw and skips it when unneeded | implemented | `tests/unit/test_full_recipe.py` (`test_full_flow_runs_raw_preprocess_spatial_hmm_latent_in_order`, `test_full_summary_links_stage_logs_and_outcomes`) |
| `BCS-07` validate a basecall generation without re-reading POD5 | implemented | `tests/unit/test_input_artifact_identities.py` |
| `BCS-08` archive write-back command and layout | proposed | -- |
| `BCS-09` batch I/O policy: no interleaved read and write on one volume | proposed | -- |
| `BCS-10` config-free `basecall --input/--output` invocation | implemented | `tests/unit/test_basecall_cli.py` (`test_run_from_paths_*`, `test_basecall_cli_config_free_form_publishes`) |
| `BCS-11` classify basecall/signal mismatch and respond per shape | proposed | -- |

### Phase 1 — selection (`BCS-01`–`BCS-04`)

Deliverable on its own: a run directory becomes a valid `input_data_path`, and
the config expresses a model instead of a hand-picked subdirectory. No stage
changes, no new commands.

`BCS-02` is the item with hidden work. Three formats record the model three
ways, and FASTQ provenance is not read anywhere today. The reader must be shared
with whatever `BCS-05` writes, so that a basecall smftools produced and one the
instrument produced are interrogated by the same code.

**Tests.** All of the above, in
`tests/unit/informatics/test_basecall_source_selection.py` and
`tests/unit/config/test_mixed_source_directory.py`. Verified against a real run
directory too: 58 FASTQ discovered across `fastq_pass` and `fastq_fail`, 28
excluded, 30 selected, model read as `dna_r10.4.1_e8.2_400bps_hac@v5.0.0` from
both the FASTQ headers and a BAM's `@RG DS`.

**An IAR acceptance criterion was deliberately superseded, and is recorded as
such rather than quietly rewritten.** `scenario.mixed_input_directory` required a
mixed directory to fail before execution listing the conflicting types. That was
right while there was no principled way to choose between representations of one
set of reads, and it is what forced the hand-picked subdirectory. It is now
`withdrawn` in `tests/acceptance/input_alignment_criteria.json` with the
reasoning; `finding.iar_m1` stays automated, because the defect it guards --
silent, priority-based selection -- is still prevented, just by a different
mechanism.

**One behaviour worth knowing.** A directory whose reads record no model at all
falls through to basecalling from signal rather than failing. That is correct,
but it is warned about, naming every rejected candidate, because a user who
pointed at that directory expecting its reads to be used should not learn
otherwise from a GPU bill.

**Docs gap closed after the fact.** Phase 1 shipped with no docs changes at
all (verified: the merge commit touches no file under `docs/`), leaving the
tutorial's own input-contract bullet stating the old behaviour --
"directories must contain one recognized input kind" -- directly
contradicting the code. Fixed in `docs/source/tutorials/experiment_config.md`
(a new "Choosing a read source from a mixed-source directory" subsection,
plus the stale bullet corrected) and `docs/source/cli.md` (a pointer
distinguishing this from `SRB`'s re-basecall planner, which is a different
thing entirely -- picking an existing representation at first ingestion
versus producing a *new* basecall of an already-ingested experiment).

### Phase 2 — the stage (`BCS-05`–`BCS-07`)

- `BCS-05` — `basecall_outputs/` with generations, `current.json`, retention, and
  an entry in `smftools experiment generations`. The existing
  `dorado-basecalling` intermediate becomes the stage's internal execution step
  rather than the product.
- `BCS-06` — `full` gains the stage ahead of raw. A FASTQ-input experiment must
  produce byte-identical results to today, with `basecall` reporting `skipped`.
- `BCS-07` — a published generation records POD5 identity so validation never
  re-reads signal.
- `BCS-10` — the config-free entry point, sharing the core and the published
  artifact with the config form.
- `BCS-11` — mismatch classification, which depends on `BCS-07`'s per-source
  identity record to tell a subset from a disjoint set at all.

**Risk.** Adding a stage touches the workflow contract, `experiment plan`
targets, `full_summary.json`, and the acceptance criteria files. The migration
question is what happens to runs whose basecalls exist only as the old
intermediate: they should be adoptable into a generation rather than recomputed,
and if that proves unreliable, recomputation must be explicit rather than silent.

**`BCS-05` shipped narrower than the risk paragraph above, deliberately.** It
adds a genuinely new, working top-level `smftools basecall CONFIG_PATH`
command that publishes a real `basecall_outputs/` generation -- but it does
**not** touch the workflow contract, `experiment plan` targets,
`full_summary.json`, or `raw`'s own inline basecalling at all. Those are
exactly the things this section's "Risk" paragraph warned would need care,
and `raw`'s inline path has no existing automated test coverage for its own
POD5-input basecalling branch to catch a mistake in -- verified by grepping
for it before starting. Rewriting that path as a side effect of adding a
parallel one was judged a worse trade than shipping a standalone command and
leaving the integration to `BCS-06`, which needs to touch `raw`'s call
ordering anyway and is the natural place to do it with real coverage.

What *is* shared: `informatics.basecall_execution.run_dorado_basecall` builds
the exact same `IntermediateSpec` shape `raw`'s inline code does, and
`prepare_intermediate` keys reuse on that spec regardless of caller --
both write under the fixed `<output_directory>/raw_outputs/intermediates/
dorado-basecalling/` location `IntermediateSpec` itself hardcodes, not a
path either caller chooses. So basecalling a run once, by either path, is
never redone by the other -- pre-basecalling a drawer of runs via `smftools
basecall` and then running `smftools experiment raw` on each reuses the
committed intermediate rather than re-invoking Dorado. `raw` still won't
know a `basecall_outputs/` generation exists or publish to it -- only the
underlying execution is shared, not the artifact model -- but the plan's
core worry (redundant GPU work between the two paths) does not apply even
before `BCS-06` unifies them properly.

**`BCS-06` shipped as a `full`-only pre-step, not a DAG node.** `full_flow`
now calls a new `cli.basecall.basecall_stage(cfg)` before
`execute_experiment_target`, entirely outside `pipeline/experiment_graph.py`'s
semantic DAG -- `EXPERIMENT_STAGES`, `_STAGE_DEPENDENCIES`,
`EXPERIMENT_NODE_IDS`, `experiment plan` targets, and `raw`'s own inline
basecalling call site are all untouched. That is narrower than "needs to
touch `raw`'s call ordering" above: registering basecall as a real DAG
predecessor of `raw` would mean adding it to every one of those per-stage
metadata dicts, plus `experiment plan`/`experiment run --target basecall`,
for a benefit (skippable-by-the-graph, individually targetable) this
increment does not need. The two behaviors the work item actually asked for
both hold without that: a FASTQ/BAM-input experiment's `full` run is
byte-identical to before (`basecall_stage` catches
`BasecallInputError` and returns without calling anything downstream), and
that stage reports `skipped` in `full_summary.json` (a new `"basecall":
BASECALL_DIR` entry in `cli.recipes.FULL_STAGE_DIRECTORIES`, populated by
`basecall_stage`'s own `setup_stage_logging`/`mark_stage_outcome("skipped",
...)` call, read back the same way every other stage's outcome is). For
POD5/FAST5 input, `basecall_stage` runs for real ahead of `raw`, publishing
into `basecall_outputs/` before `raw`'s own inline dorado call reuses the
same `IntermediateSpec` commit BCS-05 already made cache-compatible, so nothing
gets basecalled twice.

The standalone `smftools basecall` command's behavior did not change: it
still raises `BasecallInputError` for non-signal input rather than skipping
quietly, since a direct invocation on the wrong kind of input is more likely
a mistake worth surfacing than a `full` run's ordinary "nothing to do here."
`basecall_core` stays the shared implementation either way; only
`basecall_stage` adds the catch-and-skip behavior and the stage logging.

`workflow_contract.py`'s `run_experiment_workflow` needed no separate change
for `target == "full"` -- it already delegates to `recipes.full_flow` (not
its own DAG call), so it picks up `basecall_stage` for free; verified by
reading its dispatch rather than assumed.

One further simplification, intentional and `BCS-11` territory to remove:
idempotent reruns compare `stage_config_hash(cfg)` with no `stage=` argument
(the coarse, unnarrowed hash) rather than a `basecall`-specific semantic key
registered in `cli/helpers.py`'s `_STAGE_SEMANTIC_CONFIG_KEYS` -- safe (never
falsely reuses) but coarser than necessary (can invalidate on config changes
irrelevant to basecalling), and it does not yet consult the per-source
identity `BCS-07` now records at all: a POD5 pruned after archiving still
changes `stage_config_hash` today (nothing narrows it against
`input_artifact_ids`), so it forces a redundant `dorado-basecalling`
intermediate recompute -- reused by cache key rather than truly skipped, but
not the "nothing to do" outcome the recorded identity should make possible.
That consumption is `BCS-11`'s job, once mismatch classification exists to
decide *how* a changed source set should affect reuse, not just *that* it
changed.

**`BCS-07` shipped as the recording half only.** `cli.helpers.raw_input_artifact_ids`
already built exactly this shape for `raw` -- one `input-manifest:<digest>`
entry plus one `source:<source_id>:<sha256>` per resolved input row, using
the same `resolve_input_manifest_readonly` `raw` uses -- so the fix was to
extract that computation into a new shared `resolved_input_source_identities(cfg)`
and give basecall its own `basecall_input_artifact_ids(cfg)` wrapper around
it, deliberately *not* reusing `raw_input_artifact_ids` itself: that helper
also appends an `alignment-reference-bundle:<digest>` entry, which basecalling
never consults, so including it would couple a basecall generation's identity
to `cfg.fasta` and invalidate reuse on a reference-only config change that
changes nothing about the resulting BAM. `basecall_core` now records this
list instead of `raw_intermediate_manifest.artifact_checksum(cfg.input_data_path)`'s
single directory-wide digest -- the "one hash, no shape" problem this section
opened by naming. `validate_basecall_generation` needed no change: it already
never touched POD5, only re-checksumming the published BAM.

**`BCS-10` shipped the config-free form exactly as specified, and confirmed
"one command and one core" the way `BCS-05`'s note confirmed intermediate
sharing: by reading, not asserting.** `cli.basecall.run_from_paths` discovers
`--input`'s POD5/FAST5 files with `discover_input_files` (the same discovery
`ExperimentConfig` itself uses), builds a new `_ConfigFreeBasecallConfig`
dataclass carrying exactly the attributes `basecall_core` reads, and calls
`basecall_core` with it unchanged -- verified by checking `resolved_stage_config`
falls back to `vars(cfg)` for anything without `to_dict`, so a config-free
run idempotency-hashes the same way a real `ExperimentConfig` does, rather
than assumed compatible. No new `ExperimentConfig` machinery, since basecalling
needs no alignment reference or experiment metadata at all -- building a full
`ExperimentConfig` for it would need a `fasta`/`experiment_id` a config-free
invocation has no reason to have.

`smf_modality` gets repurposed as an internal direct/canonical signal, set to
`"direct"` when `--modifications` is given and `"canonical"` otherwise: only
the literal string `"direct"` is special-cased anywhere in `basecall_core`/
`run_dorado_basecall`, so any other value works identically. One consequence
worth naming rather than silently shipping: `run_dorado_basecall`'s
`IntermediateSpec` cache key includes `modality` verbatim, so a config-free
`--modifications 5mC_5hmC` run and an experiment config whose own
`smf_modality` is `"direct"` share the dorado intermediate (both record
`"direct"`), but a config-free canonical run (`"canonical"`) and an
experiment config using, say, `smf_modality=deaminase` for the *same* POD5s
do not -- two different non-`"direct"` strings are still two different cache
keys. Both still basecall correctly; the only cost is a redundant dorado
invocation in that specific cross-invocation-form scenario, not a wrong
result. A canonical, invocation-agnostic modality string is bigger than this
item and not attempted here.

**Deliberately not built:** the "overriding the model against an
already-ingested experiment is refused" guard this section's prose describes
is not implemented, in either invocation form. Checking it needs reading the
model an experiment's existing raw generation actually used, which
`raw_generation.py`'s manifest does not record today -- it would require
either extending that manifest or reading it back through
`informatics.basecall_provenance` (`BCS-02`)'s BAM-header reader, both
real work beyond "add the config-free invocation." Today, running `smftools basecall --output <dir>` a second time with a
different `--model` against the same `--output` just publishes a new
generation and (`staged_generation`'s `select_current=True` default)
advances `basecall_outputs/current.json` to it -- not a crash, and it does
not touch an already-ingested raw generation at all (`raw` does not consult
`basecall_outputs/` yet), but it is not the named refusal either: nothing
stops it, and nothing warns that the directory's basecall identity just
changed. Worth its own follow-up if it turns out to matter in practice.

### Phase 3 — the archive round trip (`BCS-08`–`BCS-09`)

Depends on `PSR-08` volume identity for the cross-device check in `BCS-09`.
Without it, `BCS-09` degrades to "always defer write-back", which is the safe
behaviour anyway and can ship first.

## Decided

- **A Dorado version difference does not invalidate a model match** (2026-08-26).
  Recorded and reported, never gating. See the model-match rule above.
- **`basecall` takes either an experiment config or plain input/output paths plus
  model parameters** (2026-08-26). One command, one core, the same published
  artifact either way. See "Two ways to invoke it".
- **A basecalls/signal mismatch is classified, not judged** (2026-08-26). Superset
  proceeds, disjoint refuses, subset proceeds only against a recorded deliberate
  subsample. See "When basecalls and signal disagree".

## Open questions

- **What identifies a config-free basecall's output as belonging to a run?** The
  config form has the run root and the experiment identity. The config-free form
  has only `--output`, so the manifest's link back to the signal it consumed is
  the only provenance it carries. Whether that is enough for an experiment to
  adopt it later without ambiguity is unresolved, and is the main thing `BCS-10`
  has to get right.
