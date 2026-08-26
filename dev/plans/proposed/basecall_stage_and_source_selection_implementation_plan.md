# Basecalling as a stage, and choosing a read source (`BCS`)

**Status:** proposed. No implementation branch.

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
| `BCS-01` accept a mixed-source directory as `input_data_path` | proposed | -- |
| `BCS-02` read basecall-model provenance from FASTQ, BAM and POD5 | proposed | -- |
| `BCS-03` model-match and capability policy | proposed | -- |
| `BCS-04` preference ordering, availability-aware | proposed | -- |
| `BCS-05` `basecall` as a generation-publishing stage | proposed | -- |
| `BCS-06` `full` runs basecall before raw and skips it when unneeded | proposed | -- |
| `BCS-07` validate a basecall generation without re-reading POD5 | proposed | -- |
| `BCS-08` archive write-back command and layout | proposed | -- |
| `BCS-09` batch I/O policy: no interleaved read and write on one volume | proposed | -- |

### Phase 1 — selection (`BCS-01`–`BCS-04`)

Deliverable on its own: a run directory becomes a valid `input_data_path`, and
the config expresses a model instead of a hand-picked subdirectory. No stage
changes, no new commands.

`BCS-02` is the item with hidden work. Three formats record the model three
ways, and FASTQ provenance is not read anywhere today. The reader must be shared
with whatever `BCS-05` writes, so that a basecall smftools produced and one the
instrument produced are interrogated by the same code.

**Tests.** A directory holding POD5, `fastq_pass` and BAMs resolves to the BAM
when its model matches; to the FASTQ when only that matches; to basecalling when
neither does. `fastq_fail` is never selected. A `direct` experiment refuses a
canonical FASTQ even when the model matches. A source on a detached volume does
not satisfy selection while an equivalent local one does.

### Phase 2 — the stage (`BCS-05`–`BCS-07`)

- `BCS-05` — `basecall_outputs/` with generations, `current.json`, retention, and
  an entry in `smftools experiment generations`. The existing
  `dorado-basecalling` intermediate becomes the stage's internal execution step
  rather than the product.
- `BCS-06` — `full` gains the stage ahead of raw. A FASTQ-input experiment must
  produce byte-identical results to today, with `basecall` reporting `skipped`.
- `BCS-07` — a published generation records POD5 identity so validation never
  re-reads signal.

**Risk.** Adding a stage touches the workflow contract, `experiment plan`
targets, `full_summary.json`, and the acceptance criteria files. The migration
question is what happens to runs whose basecalls exist only as the old
intermediate: they should be adoptable into a generation rather than recomputed,
and if that proves unreliable, recomputation must be explicit rather than silent.

### Phase 3 — the archive round trip (`BCS-08`–`BCS-09`)

Depends on `PSR-08` volume identity for the cross-device check in `BCS-09`.
Without it, `BCS-09` degrades to "always defer write-back", which is the safe
behaviour anyway and can ship first.

## Open questions

- **Does a matching model with different Dorado versions count as a match?** The
  model name and the basecaller version are separate identities, and MinKNOW
  output records both. Treating a Dorado version change as invalidating would
  force a re-basecall on every instrument software update; ignoring it entirely
  hides a real difference. Leaning toward recording it and not gating on it, but
  this is unresolved.
- **Should `basecall` be runnable without an experiment config?** Batch
  basecalling a drawer of runs is a data-preparation task, not an experiment
  task, and forcing a config per run to do it is friction. A `smftools data`
  subcommand may fit better than `smftools experiment basecall`. Deferred until
  Phase 2 shows how much of the stage machinery a config-free path would lose.
- **What happens when the selected source and the POD5s disagree?** A
  `basecalls/` directory can be stale relative to the signal beside it. Checksums
  detect it, but the right response — refuse, warn, or prefer the signal — is
  not obvious and depends on whether the user is re-analysing or re-basecalling.
