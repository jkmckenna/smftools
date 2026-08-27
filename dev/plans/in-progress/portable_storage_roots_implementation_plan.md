# Portable storage roots, volume identity, and offline raw data (`PSR`)

**Status:** every numbered work item (`PSR-01`-`PSR-20`) is implemented, with
one deliberate exception: `PSR-19`'s in-band catalog updates on publish were
not built (see its note) -- `data scan` is a full substitute in the meantime,
and this is the one piece of the plan still worth picking up as a follow-on.
Nothing else in this document describes unbuilt behavior.

**Repository state reviewed:** `bf6e9b1` — recorded while writing. The
"Current behaviour" section below is a direct investigation of the code at that
commit rather than a separate audit document; treat its claims the way an
audit's are treated, and re-verify before relying on them.

## Problem

smftools already prescribes a two-part layout (`docs/source/tutorials/directory_organization.md`):
immutable instrument output under `data/`, regenerable pipeline output under
`analyses/`. That split is right, but the code assumes both halves are always
mounted at the absolute paths recorded when the experiment was first configured.
Real long-term storage does not work that way:

- **Raw data gets archived, and to more than one drive.** The HDD used for the
  original run is not necessarily the one attached later; a dataset may exist on
  an original archive drive, a backup drive, and institutional storage
  simultaneously. There is no way today to say "this run's raw data is on any of
  these three volumes" and let smftools use whichever is attached.
- **Drives move between analysis machines.** A volume mounts at `/Volumes/<label>`
  on macOS and `/media/<user>/<label>` or `/mnt/<label>` on Linux, and the label
  itself can differ. Any address that encodes a mount path is machine-specific by
  construction.
- **Analysis storage moves too.** `analyses/` may live on an internal disk on one
  machine and an external SSD on another, and the same SSD may be driven from
  several compute machines.
- **New users must not need any of this.** The default experience has to stay
  "put in absolute paths and it works"; the portable layer has to be opt-in and
  reachable in one command.

## Current behaviour (verified at `bf6e9b1`)

### What already relocates correctly

These are not problems and must not be disturbed by this work:

| mechanism | where | encoding |
|---|---|---|
| project registry entries | `src/smftools/project/registry.py:209-231` | path relative to the project dir, absolute accepted for legacy |
| stage spine cross-references | `src/smftools/informatics/artifact_paths.py` | relative to the run root, recovered structurally from the spine's own location |
| experiment manifest artifacts | `src/smftools/informatics/experiment_manifest.py:95-151` | explicit `path_kind: relative / absolute`, with `sha256` |
| workflow result pointers | `src/smftools/cli/workflow_contract.py:1696` | `run_root`/`output_root` must be relocation-safe `"."` |

Moving an entire `analyses/` tree to a new machine or mount point already works
without editing anything.

### F-PSR-01 — An offline input volume breaks every stage, not just ingestion

`ExperimentConfig.from_var_dict` eagerly discovers input files whenever
`input_data_path` is set (`src/smftools/config/experiment_config.py:1725`), and
`discover_input_files` raises when the path is absent
(`src/smftools/config/discover_input_files.py:99`). Every stage command routes
through `load_experiment_config` (`src/smftools/cli/helpers.py:860`), so the
failure is universal.

Reproduced at `bf6e9b1` with a minimal config naming an unattached volume:

```text
FileNotFoundError: input_data_path does not exist: /Volumes/<drive>/data/<run>/pod5
```

`smftools experiment hmm`, `spatial`, `plot-current` and `export-bundle` read
nothing from `data/`, and all of them fail this way once the archive drive is
unplugged. This is the single change that most improves archival workflows, and
it needs none of the rest of this plan.

**Why it hid.** The documented portability guidance says configs hold absolute
paths that must be edited when data moves — which reads as "stale paths need
fixing", not "a correct, deliberately archived path halts unrelated stages".
The two cases are indistinguishable in the current error.

### F-PSR-02 — Config paths have no indirection of any kind

There is no environment or variable expansion in config loading: `Path(...)` is
applied to the raw string (`experiment_config.py:1725`, `:1801`), so `${VAR}`
would be taken literally, and a bare relative path resolves against the current
working directory rather than the config file's directory. The user-supplied
path fields are:

`input_data_path`, `input_manifest_path`, `output_directory`, `fasta`,
`alignment_regions_bed`, `analysis_regions_bed`, `plot_regions_bed`,
`fasta_regions_of_interest`, `sample_sheet_path`, `sequencing_summary_path`,
`model_dir`, `custom_barcode_yaml`, `umi_yaml`.

The rest of the path-typed fields on `ExperimentConfig` are derived from
`output_directory` via `constants.py` and are not user-facing.

### F-PSR-03 — Cross-volume registry paths encode the mount name

`serialize_artifact_path` (`src/smftools/informatics/artifact_paths.py:16-21`)
falls back to an absolute path only on `ValueError`, which POSIX never raises —
it is a Windows drive-letter guard. On macOS or Linux, a project on the internal
disk registering a run on an external SSD gets a relative path that walks up out
of the project and back down through the mount root. That value is valid but
encodes both the volume's mount name and the project's directory depth: exactly
the two things this plan is trying to stop encoding.

## Design

Three layers, each independently useful and each optional. A user who adopts
none of them sees today's behaviour.

### Layer 1 — input presence becomes a state, not a precondition

Resolution of `input_data_path`/`input_manifest_path` yields one of:

- `present` — files discovered, as today.
- `offline` — the path's volume is known but not attached.
- `missing` — the volume is attached and the path is genuinely gone.

`offline` is an expected, non-fatal state. Only stages that consume raw input
(`raw`, and re-basecalling) demand `present`, and they fail with an error naming
what to attach. `missing` stays an error everywhere it is reached.

Distinguishing `offline` from `missing` without volume identity is possible by
approximation — walk up to the nearest existing ancestor and check whether it is
a platform mount root (`/Volumes`, `/media/<user>`, `/mnt`) — and becomes exact
once Layer 3 exists. Ship the approximation first; it is correct for the case
that matters and never turns a real error into a silent skip.

### Layer 2 — named roots

A **root** is a named logical location. Configs reference `${data}`, `${analyses}`,
or any user-defined name; resolution order is:

1. environment variable `SMFTOOLS_ROOT_<NAME>`
2. the machine-local roots file (`~/.config/smftools/roots.toml`, or
   `$SMFTOOLS_CONFIG_DIR`)
3. a `roots.toml` found by walking up from the config file
4. literal (unexpanded values that happen to contain `$` are an error, not a
   silent literal — a typo'd root name must not become a directory name)

Bare relative paths in a config resolve against **the config file's directory**,
not the working directory. That one change alone makes an experiment directory
self-describing for everything that already lives inside it.

Roots are machine-local by design and are never written into stored artifacts.
Internal artifact pointers stay exactly as they are — relative to their anchor.

### Layer 3 — volume identity and replicas

**A stamp file travels with the drive.** `smftools data init-volume <mount>`
writes `.smftools-volume.json` at the volume root:

```json
{"volume_id": "<uuid4>", "label": "archive-01", "kind": "archive", "created": "<iso8601>"}
```

Because the stamp is on the drive, plugging it into any machine identifies it
with no per-machine configuration. This is what makes drives portable across
analysis machines rather than merely re-pathable.

**A location is `(volume_id, path_within_volume)`** — independent of mount point,
mount name, and platform. Discovery scans the platform's mount roots plus any
configured extra search paths for stamp files, producing a live
`volume_id → mount path` map.

**A dataset may have many locations.** The catalog is keyed by *dataset
identity*, not by path: reuse the existing input-manifest digest and per-file
`sha256` machinery (`src/smftools/informatics/input_manifest.py:509`,
`:544`) rather than inventing a second identity scheme. One dataset id maps to a
list of `(volume_id, path)` replicas; resolution picks the first attached one, in
a configurable preference order (working SSD before archive HDD).

**The catalog is a plain JSON file that can be copied or synced** between the
user's machines, and is rebuildable from attached drives by `smftools data scan`.
It is never required: a missing catalog degrades to Layer 2, and a missing
roots file degrades to Layer 1.

### Layer 4 — a root is a set of locations, and locality is tracked per run

Layers 2 and 3 assume one root binds to one path. Real trees are not shaped that
way: `data/` is simultaneously where new collection lands *and* where archive
drives mount, and `analyses/` holds some runs locally and others on an external
SSD. A root is therefore an **ordered set of locations**, and resolution searches
them in priority order rather than expanding to a single path.

**The unit of placement is one run directory.** Never split a run's analysis tree
across volumes. Everything inside a run already resolves by relative path and
works today; keeping the unit at the run preserves that while making locality a
per-run property — coarse enough to reason about, fine enough to be useful.

**Raw replicas and analysis copies are not the same thing.** Two raw replicas are
interchangeable: identical by checksum, so any attached one will do. Two copies
of a run's *analysis* tree can legitimately differ, because each may hold
different generations. Reusing the replica model on the analysis side would let a
stale local copy silently shadow a newer one on an SSD. The analysis side needs
one authoritative location per run, with duplicates detected and reported.

#### "Newer" means generations, never timestamps

Every stage directory publishes `generations/<id>/` selected by a `current.json`
recording `generation_id` and `manifest_sha256`. That is the comparison basis.
Modification times are not: `cp` does not preserve them, exFAT rounds to two
seconds, and clocks drift between machines, so an mtime rule would silently
prefer the wrong copy exactly when two machines have both been working.

Comparing one run's stage between two locations gives four states:

| state | meaning | response |
|---|---|---|
| identical | same generation set, same pointer | nothing to do |
| ahead / behind | one side's generation set contains the other's | the superset wins; sync is additive |
| diverged | each side holds generations the other lacks | **classify, do not resolve** |
| pointer conflict | same generations, different `current.json` | ask; a pointer is a decision, not a copy |

Divergence is ordinary — analyse locally, analyse again elsewhere — and it is
never resolved by picking a side. This is the same rule `BCS-11` applies to
basecalls against signal, for the same reason: a state that looks ambiguous from
outside is unambiguous to whoever created it, so it must be reported rather than
guessed.

#### Sync is additive because generations are immutable

Generations are content-addressed and never edited after publication, so copying
one that a destination lacks cannot corrupt anything and can resume after an
interruption. Sync therefore copies missing generations freely in either
direction and **never** moves a `current.json` on its own. Advancing a pointer is
a separate, explicit act.

#### State updates should not depend on remembering

The catalog is updated **in band** by the work itself: a stage that publishes a
generation records its location as it writes. That covers collecting new data and
running new analyses, which is most of what happens.

`smftools data scan` exists for everything smftools did not do itself — a manual
`rsync` to an SSD, a drive that was worked on from another machine, an archive
copy made with the Finder. It is a reconciliation tool, not the primary
mechanism, and it must be incremental so that running it over an attached archive
is cheap enough to do reflexively.

#### Project operations become locality-aware

`PSR-01` gave raw input a `present`/`offline`/`missing` state and project
operations got none of it. Verified at `d0a6b5e`: `list_experiments` resolves
registry paths with no reachability check, and the one existence test in
`project/catalog.py` silently `continue`s past a missing index.

So with an SSD detached, `project list` reports experiments as though present,
and a cross-experiment `materialize` either fails on a raw path or returns a
pooled result quietly missing whichever experiments were unreachable. **The
silent partial answer is the serious one** — a pooled result short several
experiments reads as a biological result, not a defect. A partial selection must
be refused by default, and labelled in the result when explicitly allowed.

### CLI surface

`smftools data` — a third top-level group alongside `experiment` and `project`.
Neither existing group fits: this is machine- and volume-scoped, below any single
experiment and across all projects. Per `src/smftools/cli/AGENTS.md`, that
decision is the first step of adding a command and is recorded here deliberately.

| command | purpose |
|---|---|
| `data init-volume <mount> --label --kind` | stamp a drive |
| `data volumes` | known volumes and attached status |
| `data scan [<mount>...]` | index runs on attached volumes into the catalog |
| `data roots [list/set]` | inspect or set machine-local root bindings |
| `data locate <experiment/dataset>` | replicas, and which are attached |
| `data verify <dataset>` | checksum a replica against its manifest |
| `data localize <config>` | copy small referenced inputs into the run directory |
| `data status` | where every run's data and analyses are, what is attached, and what is ahead, behind or diverged |
| `data sync <run>` | copy missing generations between two locations; never moves a pointer |

`data localize` is the cheapest adoption win in the plan: copying `fasta`, the
bed files, the sample sheet, and any barcode/UMI YAML into the experiment
directory makes the whole `analyses/` tree self-contained, with no roots, no
volumes, and no catalog.

## Work items

| item | status | evidence |
|---|---|---|
| `PSR-01` input resolution state (`present`/`offline`/`missing`) | implemented | `tests/unit/config/test_input_availability.py` |
| `PSR-02` skip discovery when offline; recover identity from the run | implemented | `test_offline_run_hashes_identically_to_the_attached_run` |
| `PSR-03` actionable offline error naming the volume | implemented | `test_stage_requiring_raw_input_refuses_while_offline` |
| `PSR-04` root variable expansion in config values | implemented | `tests/unit/config/test_named_roots.py` |
| `PSR-05` config-relative resolution of bare relative paths | implemented | `test_relative_paths_anchor_to_the_config_not_the_cwd`, `test_working_directory_fallback_keeps_pre_psr05_configs_working` |
| `PSR-06` machine-local roots file + `SMFTOOLS_ROOT_<NAME>` | implemented | `test_environment_binding_wins`, `test_nearest_roots_file_wins` |
| `PSR-07` root-qualified artifact/registry pointers | implemented | `test_path_under_a_bound_root_is_qualified`, `test_all_three_encodings_resolve` |
| `PSR-08` volume stamp file + `data init-volume` | implemented | `tests/unit/data/test_volume_stamp.py`, `tests/unit/test_data_cli.py` |
| `PSR-09` mount discovery, macOS + Linux | implemented | `tests/unit/data/test_volume_discovery.py`, `tests/unit/config/test_volume_search_paths.py`, `tests/unit/test_data_cli.py` |
| `PSR-10` replica catalog keyed by dataset digest | implemented | `tests/unit/data/test_replica_catalog.py` |
| `PSR-11` `data scan` / `locate` / `verify` | implemented | `tests/unit/data/test_volume_scan.py`, `tests/unit/data/test_volume_verify.py`, `tests/unit/test_data_cli.py` |
| `PSR-12` exact `offline` vs `missing` via volume identity | implemented | `tests/unit/config/test_input_availability.py`, `tests/unit/config/test_offline_input_config.py` |
| `PSR-13` `data localize` | implemented | `tests/unit/data/test_localize.py`, `tests/unit/test_data_cli.py` |
| `PSR-14` `data init` scaffold for a new lab tree | implemented | `tests/unit/data/test_lab_init.py`, `tests/unit/test_data_cli.py` |
| `PSR-15` docs + migration of existing absolute configs | implemented | `docs/source/tutorials/directory_organization.md`'s Portability section |
| `PSR-16` a root resolves over an ordered set of locations | implemented | `tests/unit/config/test_root_location_sets.py` |
| `PSR-17` per-run analysis locality, with duplicate detection | implemented | `tests/unit/data/test_run_locality.py` |
| `PSR-18` locality state in `project list`/`materialize`; refuse silent partial answers | implemented | `tests/unit/project/test_project_unreachable_selection.py` |
| `PSR-19` in-band catalog updates on publish; incremental `data scan`; `data status` | partial | `tests/unit/data/test_analysis_catalog.py`, `tests/unit/data/test_volume_scan.py`, `tests/unit/test_data_cli.py` -- `data status` and scan-based population implemented; in-band publish hooks not built (see note) |
| `PSR-20` `data sync`: additive generation copy, divergence classified not resolved | implemented | `tests/unit/data/test_run_sync.py`, `tests/unit/test_data_cli.py` |

### Phase 1 — offline tolerance (`PSR-01`–`PSR-03`)

Self-contained, no new user-facing concepts, no config format change. Delivers
the archival workflow on its own.

- `PSR-01` — introduce the resolved-input state on `ExperimentConfig`. Keep
  `input_files`/`input_type` semantics unchanged for `present`.
- `PSR-02` — move discovery out of `from_var_dict` into the stages that consume
  input. Ingestion behaviour must be byte-identical when the data is attached.
- `PSR-03` — the error a consuming stage raises names the path, the volume label
  where one is known, and the fact that the volume appears unattached rather than
  the path deleted.

**Tests.** A config naming an unattached path parses; `raw` fails with the
actionable message; a genuinely deleted path under an attached volume still
fails as `missing`; and an offline load hashes identically to an attached one.

**`plot-current` was wrong in this list.** It reads POD5 signal directly from
`input_data_path` (`cli/plot_current.py`), so it is a *consuming* stage and
refuses while the input is archived, alongside `raw` and `load`. The stages that
run offline are preprocess, variant, chimeric, spatial, hmm, latent,
export-bundle and export-fastq. Re-basecalling is out of scope here: it resolves
POD5 through its own durable-origin identity rather than `cfg.input_data_path`.

**What discovery actually needed.** The plan said to move discovery out of
`from_var_dict` entirely. Implementation kept it there for the `present` case and
skipped it only when the volume is detached. Every structural check discovery
feeds -- mixed types, unsupported SAM, CRAM alignment mode, direct-versus-FASTQ
-- is worth keeping at config load when the data is reachable, and `missing`
still raises there, so a mistyped path is caught exactly as early as before.
Only the offline case defers.

**The hash was the real blocker, and the plan did not anticipate it.**
`input_type` and `input_files` feed each stage's config hash. Unset while the
volume is detached, they moved raw's hash from `c40b924b8b11b4af` to
`0dd065eb8fba68cc`, so a finished raw generation read as incompatible and every
downstream stage -- all of which re-enter raw's gate through the dependency walk
-- tried to re-ingest data it could not reach. Parsing succeeding was therefore
not sufficient on its own.

Two fixes were considered. Excluding the derived fields from the hash is more
principled (they are a function of `input_data_path`, which stays hashed, and
content identity is separately covered by `input_artifact_ids` and the input
manifest's sha256) but removing keys changes the hash for everyone and would
invalidate **every existing raw generation**, forcing a rebuild from
FASTQ/POD5. Instead the offline branch recovers `input_type`/`input_files` from
the run's own `resolved_input_manifest.json`, which reproduces the attached hash
exactly and invalidates nothing.

That recovery reads the resolved input manifest, **not** the experiment
manifest's config snapshot: the snapshot records `input_type` after the
FASTQ-to-BAM rewrite has replaced it, so it would not reproduce the hash. The
three new availability fields are excluded from the stage hash for the same
reason the recovery exists -- whether a drive is plugged in is not a property of
the experiment.

**Risk.** Deferring discovery moves the point where a real typo in
`input_data_path` is caught. `PSR-03`'s `missing` classification keeps that from
being a regression: a path absent while its volume is attached still raises at
config load.

### Phase 2 — named roots (`PSR-04`–`PSR-07`)

- `PSR-04` — expansion applied to the thirteen user-supplied path fields listed
  in `F-PSR-02` and to nothing else. An unresolvable root name is an error.
- `PSR-05` — bare relative paths resolve against the config file's directory.
  This changes behaviour for any existing config that relies on CWD resolution;
  gate it and state the migration in `PSR-15`.
- `PSR-06` — resolution order as designed above. `known_roots()` reports which
  layer each binding came from (`RootBinding.source`), exercised directly by
  `tests/unit/config/test_named_roots.py`. `data roots list` (a `data roots`
  subgroup, with room left for a future `set`) was wired to it as a follow-up
  after being noted-not-fixed while writing `PSR-15`'s docs -- a small,
  contained gap with no open design questions once identified, so it did not
  need to wait for a dedicated Phase.
- `PSR-07` — implemented one level lower than planned, in
  `serialize_artifact_path`/`resolve_artifact_path` rather than in the registry
  alone, so every artifact pointer gains the encoding rather than just registry
  entries. The trigger is also simpler than "a different volume", which cannot be
  answered until `PSR-08`: a path inside the anchor stays plain relative, a path
  under a bound root is qualified, and everything else keeps the old relative
  walk and absolute fallback. The most specific root wins. Readers accept all
  three encodings.

**The gate this plan asked for was needed, and skipping it broke fifteen tests.**
`PSR-05` was written as "gate it and state the migration", which I first read as
a documentation task. It is not: test configs carried relative paths resolved
against the working directory, and anchoring them to the config's directory
repointed every one. The gate is now behavioural — the pre-`PSR-05` reading is
honoured, with a warning, when the config-relative path names nothing and the
working-directory one names something real. Where both exist the config-relative
reading wins, which is the only case that still changes meaning.

### Phase 3 — volume identity (`PSR-08`–`PSR-12`)

- `PSR-08` — stamp format, written once, never rewritten; a stamped volume that
  reappears with a changed label keeps its `volume_id`.
- `PSR-09` — discovery across macOS and Linux mount conventions, plus configured
  extra search paths for network mounts.
- `PSR-10` — catalog schema, keyed by the existing input-manifest digest. A
  replica record carries `volume_id`, path within the volume, the digest it was
  observed with, and when it was last verified.
- `PSR-11` — `scan` walks an attached volume for experiment manifests and input
  manifests; `locate` answers while the drive is *unplugged*, which is the point;
  `verify` re-checksums.
- `PSR-12` — replaces the Phase 1 approximation with an exact answer.

**`PSR-08`'s "never rewritten" is literal, not just for `volume_id`.** The whole
`.smftools-volume.json` is immutable once written, including `label` and `kind`:
`data init-volume` on an already-stamped mount is idempotent and returns the
existing stamp untouched, warning if the requested `--label`/`--kind` differ
rather than applying them. "A stamped volume that reappears with a changed
label keeps its `volume_id`" falls out of this for free -- discovery (`PSR-09`)
will only ever read `volume_id` back out of the stamp, never derive it from the
OS-reported volume name, so an OS-level rename can't touch identity regardless
of whether `init-volume` is ever re-run. `kind` is constrained to
`{working, archive, backup}`, matching the working/archive preference ordering
Layer 3 already commits to for replica resolution -- there's nowhere else in
the plan a fourth kind is used, so this list is not meant to be treated as
closed if one turns up.

**`PSR-09` shipped as a library function plus a thin `data volumes` listing
command, not the full catalog-aware command the CLI surface table describes.**
That table's `data volumes` ("known volumes and attached status") needs
`PSR-10`'s catalog to say anything about a volume that is *not* currently
attached; until then, `smftools data volumes` reports exactly what
`discover_volumes()` finds attached right now and nothing about history. The
platform mount-root table was promoted out of
`smftools.config.input_availability` (`MOUNT_ROOTS`, was `_MOUNT_ROOTS`) rather
than duplicated, since Phase 1's structural approximation and Phase 3's real
discovery need to agree on the same mount conventions or `PSR-12` cannot
cleanly replace one with the other. Extra search paths for network mounts live
in the same `roots.toml` as named roots, under a new `[volumes]
extra_search_paths` table, resolved through
`smftools.config.roots.extra_volume_search_paths` -- unlike a named root's
"nearest layer wins", every configured source is unioned, since a search path
is additive by nature (more places to look), and only the
`SMFTOOLS_VOLUME_SEARCH_PATHS` environment override replaces rather than adds.
A candidate directory with a corrupt stamp is skipped with a warning rather
than raised, so one bad mount can't fail discovery of every other attached
volume; the same is true of two mounts reporting the same `volume_id`, which
`PSR-08`'s "written once, never rewritten" guarantee should make impossible in
practice but discovery does not trust that from outside.

**`PSR-10` is schema and API only, with no CLI and nothing populating it
yet -- that is `PSR-11`'s `scan`/`locate`/`verify` and `PSR-19`'s in-band
updates.** `smftools.data.replica_catalog` stores `{dataset_digest: [replica,
...]}` as a plain JSON file (`replica_catalog.json`) next to `roots.toml`
(same `SMFTOOLS_CONFIG_DIR`, no second override), with `load_catalog`/
`save_catalog`, an `add_replica` that updates a replica already recorded at
the same `(volume_id, path)` in place instead of duplicating it, and a
`resolve_replica` that -- given a `discover_volumes()` result -- picks the
first attached replica by kind preference (`working` before `archive` before
`backup`, matching the design's "working SSD before archive HDD"; an
unrecognized kind sorts last rather than being dropped). A replica's `digest`
is stored per-replica rather than assumed equal to the catalog key, so a
replica that has drifted from its dataset (partial copy, bit rot) is
something `PSR-11`'s `verify` can actually detect instead of taken on faith.

**`PSR-11` reads "walks an attached volume for experiment manifests and input
manifests" as one walk, not two.** A run root's identity for this purpose
*is* its resolved input manifest -- finding that file is finding the run, so
`scan` never separately looks for `experiment_manifest.py`'s manifest; nothing
in `PSR-10`'s dataset-keyed catalog needs it. What `scan` actually registers
per run is deliberately narrower than the design's original Layer-3 picture of
"a raw archive drive is scanned for its raw bytes": that would require
recomputing checksums for arbitrary files with no manifest to check them
against, which is exactly what `verify` exists to do once a manifest *is*
known. Instead `scan` registers **wherever a run's resolved input manifest
itself lives** as a replica of that run's dataset -- typically an analyses
tree, on the volume it happens to be stamped on, not necessarily where the
raw bytes are. This is enough to make `locate`/`verify` useful today, and nothing about it needs revisiting when Phase 5 draws the raw/analysis distinction
formally: a replica here already means "a location with a checksummed record
of this dataset," which is a strict subset of what a Phase-5 raw replica would
mean, not a conflicting definition of it.

The walk is pruned to stay affordable on a real archive drive with millions of
files: it never descends into a directory named `generations`, and once
inside `raw_outputs` it only descends into `input_manifest` -- both proven by
a test that buries an otherwise-valid (if deliberately corrupt) manifest where
pruning should prevent it from ever being reached, rather than merely
asserting "not found."

`verify` re-checksums each declared source directly, **bypassing**
`smftools.informatics.input_manifest`'s stat-signature cache on purpose: that
cache exists to make repeat ingestion cheap by trusting an unchanged
mtime/inode/size signature, which is precisely the shortcut a corruption check
must not take. A declared source not currently reachable on disk is reported
as `unreachable`, distinct from `mismatch` and not a verification failure --
raw input being archived elsewhere is `PSR-01`'s ordinary case, and `verify`
checks whatever it *can* reach, exactly like `scan` and `locate` do.

**What the stamp is not.** It is an identifier, not an integrity guarantee.
Nothing may treat a matching `volume_id` as evidence that the data is intact;
that is what the checksums are for.

**`PSR-12` turned out to need two things the phrase "exact answer" doesn't
say by itself: what specifically the approximation gets wrong, and a decision
about what "exact" does with a path once it knows it moved.**

*What the approximation actually gets wrong* is not "offline vs. missing" in
general -- it is two narrower failures. First, a path under a mount
convention `MOUNT_ROOTS` does not recognize (a custom network share, an
unconventional layout) is misclassified `missing`, a hard error, when it may
simply be an unattached volume the structural test cannot see as one.
Second, and the case the whole "volume identity" half of this plan exists
for: the same physical drive, reattached under a *different* mount name,
reads as `missing` even though the data is right there, because nothing
about `input_data_path` as a bare path can say "this is the same volume I
saw before." Both require actually knowing which volume a path's data used
to be on -- which a bare path never encodes -- so `PSR-12` could not be
built as a smarter version of `detached_volume_for`; it needed the
identity layer underneath it.

*What "exact" does once it has that identity* -- `resolve_input_availability`
gained an optional `output_directory` parameter (default `None`, so every
caller who does not pass it sees byte-identical behavior to before this
landed). When given, it reads the run's own published input manifest to get
the manifest's dataset digest and `base_directory`, consults the replica
catalog (`PSR-10`) for that digest, and asks `discover_volumes()` (`PSR-09`)
whether any known replica is attached *right now, under whatever name it
currently has*:

- No catalogued replica attached at all -- confident `offline`, regardless of
  whether the path matches a recognized mount convention. This is the fix for
  the network-share case above, and required no path remapping at all: the
  literal path already failed to resolve (that is why this function is being
  asked), so "nothing catalogued is attached" is sufficient on its own.
- A replica *is* attached -- remap the queried path from the manifest's
  recorded `base_directory` onto the replica's current, possibly-relocated
  location, and check whether *that* resolves. Only then is it `present`;
  an attached replica whose remapped path still does not exist defers to the
  structural guess rather than asserting `missing` on partial information.
- Any prerequisite missing -- no `output_directory`, no published manifest
  yet, no catalog, nothing catalogued for this digest -- returns `None` and
  the structural approximation runs exactly as it did before `PSR-12`. A user
  who has never run `data scan` is entirely unaffected.

Reporting `present` from a relocated path was not the end of the change by
itself: `ExperimentConfig.from_var_dict`'s `has_input_path` branch calls
`discover_input_files(input_data_path, ...)` on the *original* variable
after checking availability, not on `InputAvailability.path`. Without also
substituting `input_data_path = input_availability_state.path` when it
differs, an exact `present` classification would have made discovery run
against the same stale, nonexistent path it just proved was gone elsewhere --
turning a correct classification into a crash instead of a silent no-op. This
was caught by an end-to-end test through `from_var_dict`, not the
`resolve_input_availability` unit tests alone, which is why one exists
(`test_config_load_relocates_input_through_the_replica_catalog`).

`InputAvailability` gained a `volume_id` field alongside the existing
`volume` (a mount *path*): the exact-offline case has no mount path to name
(nothing is attached), only a catalog-known `volume_id`, and forcing that
into a `Path`-typed field would have been a type lie. Every place that used
to print `.volume` in a message now prefers it and falls back to `.volume_id`
so an exact-offline message never reads "volume None".

### Phase 4 — adoption (`PSR-13`–`PSR-15`)

- `PSR-13` — `data localize`, including a dry-run listing what would be copied
  and its size.
- `PSR-14` — `data init` scaffolds `data/` + `analyses/runs/` +
  `analyses/projects/` and offers to stamp the volume it is on, mirroring what
  `project init` does for a project directory.
- `PSR-15` — rewrite the Portability section of
  `docs/source/tutorials/directory_organization.md` around the three layers, and
  ship a migration note for `PSR-05`.

**`PSR-13`'s riskiest design decision was whether `--apply` may edit the
user's config file in place, and it was decided no.** Every other write this
plan makes is either additive (a stamp, a catalog entry) or explicitly
idempotent; rewriting a hand-maintained config in place would be neither --
and a mistaken field mapping would be much harder to notice in an edited
file than in a fresh one sitting next to it. `--apply` therefore always
writes a **new** file (`<config>.localized<suffix>` by default, `--out`
otherwise) and never touches the original, which also means a dry run and an
applied run share one code path (`build_localize_plan` is unconditional;
`apply_localize_plan` is the only part that writes) rather than needing two
independently-maintained implementations that could drift apart.

The localizable field set (`fasta`, the three BED fields, `sample_sheet_path`,
`custom_barcode_yaml`, `umi_yaml`) is a deliberate subset of
`USER_SUPPLIED_PATH_FIELDS` in `smftools.config.experiment_config`, not that
whole list: `input_data_path`/`input_manifest_path` are the large data this
entire plan exists to leave archived, not duplicate; `sequencing_summary_path`
and `model_dir` can themselves be large (a MinKNOW run's summary, a
basecalling model directory); and `fasta_regions_of_interest` is deprecated in
favor of `alignment_regions_bed`, so localizing it would just entrench a
field already being phased out.

A repeat `--apply` is deliberately not an error: a destination already
holding byte-identical content is treated as "already localized" and skipped,
matching this plan's broader pattern of idempotent re-application (`PSR-08`'s
stamp, `PSR-10`'s replica de-duplication). Only a destination with *different*
content raises, since silently overwriting it would either corrupt a prior,
unrelated localization or mask the fact that the source file itself changed
since the last `--apply`.

**`PSR-14` skipped the interactive "offers to stamp" the CLI table's wording
implies.** No command anywhere in `smftools` prompts interactively (`grep` for
`click.confirm`/`click.prompt` across `src/` turns up nothing); adding the
first one for this alone would make `data init` hang under any non-interactive
caller (CI, a script, an agent) unless every such caller remembered a
`--yes`-style escape hatch that doesn't exist yet either. `--stamp-volume` is
the flag-driven equivalent instead, off by default, with a printed hint
pointing at it (or at `data init-volume` directly) when it is not passed --
"offers" in the sense of telling the user the option exists, not blocking on
an answer.

`--stamp-volume` stamps `LAB_ROOT` itself, not any enclosing directory --
matching `data init-volume`'s own contract of stamping exactly the path it is
given, with no attempt to walk up to find "the real" OS mount point (there is
no portable, dependency-free way to ask an OS for that, and none of `PSR-08`/
`PSR-09` needed one). The consequence, stated in both the CLI help and the
docs rather than hidden: the stamp is only found by `data volumes`/`data scan`
elsewhere if `LAB_ROOT` **is** the volume's own mount point, since discovery
(`PSR-09`) only looks one level below a platform mount root. A lab root nested
inside a larger drive can still be stamped, it just will not be discoverable
that way -- an honest limitation of the discovery mechanism, not something
`PSR-14` could paper over on its own.

**`PSR-15`'s migration note was already done.** `PSR-05`'s own implementation
note (above) already added the exact gate-and-warn behaviour the plan asked
for, and the tutorial's "Relative paths anchor to the config" subsection
already documented it in full, including the migration text, before this item
was picked up. What `PSR-15` actually added was the missing half: the
Portability section covered relative pointers (pre-existing) and named roots
(`PSR-04`-`PSR-07`) but never mentioned offline/missing input classification
(`PSR-01`-`PSR-03`) or volume identity (`PSR-08`-`PSR-12`) at all -- a `grep`
for "offline"/"volume_id"/any `PSR-0[89]`/`PSR-1[0-4]` reference against the
file came back empty before this. Two new subsections close that gap:
"Archived raw input is not an error" (Layer 1) and "Volume identity for
removable drives" (Layer 3, plus a closing pointer to `data localize`/
`data init` as the lighter-weight alternative). `docs/source/cli.md` already
carried the command-level reference for every `data` subcommand as each PSR
landed; this tutorial is the conceptual walkthrough the CLI reference was
never meant to replace.

**One stale claim in `PSR-06`'s own note (above) was found while writing
these docs, and later fixed rather than left corrected-in-prose**: it said
`data roots list` shipped, and at the time no such command existed
(`known_roots()`, the function it would call, did -- just never wired to a
CLI command). That gap was small and well-scoped enough to close directly as
a follow-up rather than stay a documented limitation; see `PSR-06`'s note.

### Phase 5 — many locations per root (`PSR-16`–`PSR-20`)

Depends on `PSR-08` volume identity to say *where* a copy is in a way that
survives remounting, and on `PSR-10`'s catalog to hold it.

- `PSR-16` — a root resolves over an ordered set of locations, not one.

  **This was already three-quarters built during Phase 2, deliberately.**
  `config/roots.py`'s `_binding_path` already accepted a list-valued binding
  and picked "first that exists, falling back to first" -- exactly `PSR-16`'s
  resolution rule -- because its own docstring said so at the time: "so that
  when `PSR-16` makes a root an ordered set of locations the file format does
  not have to change under anyone." What was missing was not resolution; it
  was the write-back direction. `qualify_with_root` (used when serializing an
  artifact pointer, `PSR-07`) checked containment against only `RootBinding`'s
  single winning `.path`, not every candidate a list binding names. A path
  living under a *second* candidate location -- e.g. a run whose analysis
  tree sits on an SSD that happens to be detached right now, while the root's
  first-listed candidate exists but is irrelevant -- would silently fail to
  qualify and fall back to an absolute or relative-walk pointer, defeating the
  entire point of naming that root. `RootBinding` gained `all_paths` (every
  candidate, in order) alongside the existing `path` (the winner), and
  `qualify_with_root` now checks every candidate of every root rather than
  just the winning one. `resolve_root`/`expand_roots` (the read direction)
  needed no change: picking one winning path for substitution was already
  correct, since a run's own tree still always resolves under a single,
  consistent location -- this item never required splitting one run across
  volumes, only allowing *different* runs under the same root name to live in
  different places.

- `PSR-17` — per-run analysis locality, with duplicate detection.

  **Duplicate detection needed an identity this plan hadn't named yet, and one
  already existed.** "Are these two directories copies of the same run" can't
  be answered from a path or the human-chosen `experiment_id` label -- neither
  is stable across a rename or a machine. `experiment_uid`
  (`smftools.informatics.molecule_identity.new_experiment_uid`, a UUID4
  minted once at raw ingestion and persisted in `experiment_manifest.json`)
  already exists for an unrelated reason (molecule/segment UID namespacing)
  and turned out to be exactly the right primitive: durable, content-
  independent, and already written by every modern run. `are_duplicates`
  reuses it rather than inventing a second identity scheme -- the same
  choice `PSR-10`'s catalog made about the input-manifest digest. Two
  locations where *either* side's identity is unknown (no manifest yet, or a
  manifest that predates the identity system) are **not** treated as
  duplicates; "no proof" is not "proof they differ", but neither is grounds
  to compare them as the same run.

  **The four-state table was implementable directly on top of
  `informatics.generation_listing.list_experiment_generations`**, already
  built for `smftools experiment generations` inventory: group its records by
  stage (`kind`) at each location, diff the two `state=ok` generation-id sets,
  and the four states fall out of set membership alone -- both-empty
  differences is `diverged`, one-sided is `ahead`/`behind`, no difference but
  a different `is_current` flag is `pointer_conflict`, otherwise `identical`.
  No new generation-reading code, no mtime anywhere, matching the design's own
  rule at the top of this phase. A stage neither location has published
  anything for is skipped rather than forced into one of the four states --
  there is nothing to disagree about, not a fifth kind of disagreement.

  **Scope stops at classification, deliberately.** No CLI command surfaces
  this yet (`PSR-19`'s `data status` is where locality across every run
  becomes visible) and nothing here acts on a diverged or conflicting
  result -- `PSR-20`'s `data sync` is additive-only and never resolves
  divergence by picking a side, so there is no "fix it" action for this
  module to trigger even in principle. `compare_run_locations` also doesn't
  call `are_duplicates` itself: comparing two unrelated runs' generation sets
  is meaningless but harmless, and forcing the check inside would make the
  function's cost depend on manifest I/O a caller who already knows the
  answer shouldn't have to pay for.

- `PSR-18` is separable and came first, as planned. Implemented ahead of the rest
  of Phase 5, needing none of the union-root machinery.

  **The characterisation in this plan was half wrong, and the code was worse in
  one place and better in another.** Pooling did *not* silently drop experiments:
  `resolve_experiment_spine` never checked existence, so an unreachable
  experiment stayed a member and `iter_set_parts` failed inside `materialize` —
  mid-stream, after earlier parts had been yielded and, for
  `export_project_partitions`, written. The genuinely silent loss was in
  `ProjectCatalog`'s three union methods, which dropped an absent path and
  returned the union of what remained.

  Both are fixed: unreachable experiments are classified before any read,
  `resolve_set_members` refuses by default, `allow_unreachable=True` *excludes*
  them and says so rather than deferring the failure, the union methods name what
  they omit, and `project list` reports `locality` per experiment.
- `PSR-19` is what makes the rest usable rather than a thing to remember.
  In-band updates cover the common cases; `scan` reconciles what happened outside
  smftools.

  **Shipped: the analysis-location catalog, `data status`, and scan-based
  population. Deliberately not shipped: in-band updates.** A second catalog
  (`smftools.data.analysis_catalog`, `analysis_catalog.json` next to
  `roots.toml`) tracks where copies of a run's *analysis* tree are, keyed by
  `experiment_uid` -- distinct from the replica catalog's dataset-digest key,
  since analysis copies are not interchangeable the way raw replicas are
  (`PSR-17`). `data scan`'s existing walk (`PSR-11`) was extended to
  populate it: it already found every run root to check for a raw input
  manifest, and `experiment_manifest.json` -- present at the root of every
  modern run regardless of which stages completed -- turned out to be an
  *easier* marker to walk for than the raw manifest, since nothing lives
  below a run root that this scan needs once it's found (no
  `raw_outputs`-specific descent rule was needed here at all). `data status`
  combines both catalogs with `discover_volumes()` and `PSR-17`'s
  `compare_run_locations`, run pairwise against the first attached location,
  to report locations, attached status, per-stage locality, and (when at
  least one location is reachable) the run's raw dataset and its replicas.

  In-band updates -- a stage recording its own location as it publishes,
  which the design calls the *primary* mechanism, not `scan` -- were not
  built. Doing this honestly means threading a location-recording call
  through every generation publisher (raw, preprocess, latent, project
  embeddings -- the same four subsystems `generation_listing.py`'s own
  docstring names as having "converged independently" on the shared
  generations vocabulary), which is a materially larger and riskier change
  than anything else in this plan: many call sites across files this session
  never touched, each needing the same care given to the scan/catalog code
  that did get built. `data scan` is a complete substitute in the meantime --
  the design already treats it as a legitimate, if secondary, mechanism
  ("a reconciliation tool, not the primary mechanism") -- so `data status`
  is fully functional today, just not automatically current without a scan.
  This gap is stated in `docs/source/cli.md`, not hidden.

  **"Incremental" was already satisfied, not a separate task.** The design's
  reason for `scan` needing to be incremental was cost: cheap enough to run
  reflexively over an attached archive. Neither walk (`_iter_resolved_manifests`,
  `_iter_run_roots`) hashes anything -- both just parse small JSON files they
  find, and `add_replica`/`add_location` are idempotent updates-in-place -- so
  a repeat scan was already cheap by construction from `PSR-11` onward. No
  separate caching or change-tracking was added.

- `PSR-20` must not gain a `--force-newer` flag that resolves divergence by
  timestamp. If that appears, the generation-set comparison has been abandoned
  and the tool is guessing again.

  **Shipped exactly as scoped, with no `--force-newer` and no third option
  beyond copy-or-refuse.** `smftools.data.run_sync.sync_run_locations`
  reuses `PSR-17`'s `compare_run_locations` directly rather than
  recomputing anything: `ahead`/`behind` copies the missing generation
  directories (`shutil.copytree` into a `.{id}.syncing-<uuid>` staging
  name, then `rename` into place -- the same stage-then-rename publish
  pattern generation writers themselves already use, so a destination only
  ever exists once fully copied); `diverged`/`pointer_conflict` copy
  nothing and carry a human-readable `skipped_reason` instead of an
  exception, since a partially-successful sync across several stages is the
  normal case, not a failure. `current.json` is never written by this
  module at all -- there is no code path that could move a pointer, not
  merely a check that declines to. The CLI exits non-zero exactly when at
  least one stage was left unresolved, so a script can tell "fully synced"
  from "some stages need a human" without parsing output.

  `--from`/`--to` (by `volume_id`) exist only to disambiguate when more than
  two of a run's catalogued locations are attached at once; with exactly two
  attached, sync runs on that pair with no flags needed. This is the only
  addition beyond the CLI surface table's original `data sync <run>`, and it
  is a refusal path (an explicit ambiguity error), not a way to skip
  classification.

**Tests.** Two locations of one run, one holding a generation the other lacks,
classify as ahead/behind and sync additively; each holding a distinct generation
classifies as diverged and refuses; equal generation sets with different
`current.json` classify as a pointer conflict. A detached experiment makes
`project materialize` refuse rather than pool a subset, and `project list` shows
it as unreachable rather than present.

## Rejected alternatives

**A symlink farm** — `data/<run>` as a symlink into the current mount. Low-tech
and needs no code, and it is what a user would reach for first. Rejected as the
mechanism because it fails exactly where this plan has to work: a dangling
symlink is indistinguishable from deleted data, the link target encodes the mount
name so it breaks when the drive moves to another machine, and one link cannot
express several replicas. It remains a fine thing for a user to do on top of
Layer 2.

**Environment variables only** — no roots file, no catalog, `$SMF_DATA_ROOT` and
done. Genuinely simple, and it is retained as the highest-priority resolution
layer in Layer 2. Rejected as the whole answer because a single variable cannot
name several archive drives, and per-shell state that silently changes meaning
between terminals is a bad place for the only pointer to where the data is.

**SQLite for the catalog** — better at scale, worse at everything this catalog
needs to be: diffable, hand-inspectable, copyable between machines, and
rebuildable from the drives themselves. Revisit only if a real user's catalog
outgrows JSON.

**Resolving divergence by modification time.** The obvious answer to "which copy
is newer", and it is what a user would reach for. Rejected because the signal is
not trustworthy where it matters most: `cp` does not preserve mtimes, exFAT
rounds to two seconds, and clocks drift between machines — so it is least
reliable in exactly the two-machine case that produces divergence. Generation
sets are an exact answer to the same question and are already on disk.

**Treating analysis copies as replicas.** Reusing Layer 3's replica model on the
analysis side would be less code and is tempting for symmetry. Rejected because
replicas are interchangeable by checksum and analysis trees are not: two copies
can hold different generations, and "any attached one will do" would let a stale
local copy shadow a newer SSD one.

**Recording replicas per experiment instead of centrally** — the natural first
instinct, since the experiment already has a manifest. Rejected because
"I copied this to a second backup drive" is learned long after the experiment
finished, and is knowledge about a dataset, not about a run. Writing it into a
published, checksummed generation manifest would also violate the immutability
that `docs/source/tutorials/directory_organization.md` already establishes.
