# Portable storage roots, volume identity, and offline raw data (`PSR`)

**Status:** in progress. Phase 1 (`PSR-01`-`PSR-03`) implemented on
`feature/psr-01-offline-input-state`; Phases 2-4 remain proposed.

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
| `PSR-04` root variable expansion in config values | proposed | — |
| `PSR-05` config-relative resolution of bare relative paths | proposed | — |
| `PSR-06` machine-local roots file + `SMFTOOLS_ROOT_<NAME>` | proposed | — |
| `PSR-07` root-qualified registry paths across volumes | proposed | — |
| `PSR-08` volume stamp file + `data init-volume` | proposed | — |
| `PSR-09` mount discovery, macOS + Linux | proposed | — |
| `PSR-10` replica catalog keyed by dataset digest | proposed | — |
| `PSR-11` `data scan` / `locate` / `verify` | proposed | — |
| `PSR-12` exact `offline` vs `missing` via volume identity | proposed | — |
| `PSR-13` `data localize` | proposed | — |
| `PSR-14` `data init` scaffold for a new lab tree | proposed | — |
| `PSR-15` docs + migration of existing absolute configs | proposed | — |

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
- `PSR-06` — resolution order as designed above, with `data roots list` printing
  which layer each binding came from.
- `PSR-07` — registry entries store `${root}/relative` when the run is on a
  different volume from the project, keeping the existing plain-relative encoding
  when they share one. Readers accept all three encodings (plain relative,
  legacy absolute, root-qualified).

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

**What the stamp is not.** It is an identifier, not an integrity guarantee.
Nothing may treat a matching `volume_id` as evidence that the data is intact;
that is what the checksums are for.

### Phase 4 — adoption (`PSR-13`–`PSR-15`)

- `PSR-13` — `data localize`, including a dry-run listing what would be copied
  and its size.
- `PSR-14` — `data init` scaffolds `data/` + `analyses/runs/` +
  `analyses/projects/` and offers to stamp the volume it is on, mirroring what
  `project init` does for a project directory.
- `PSR-15` — rewrite the Portability section of
  `docs/source/tutorials/directory_organization.md` around the three layers, and
  ship a migration note for `PSR-05`.

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

**Recording replicas per experiment instead of centrally** — the natural first
instinct, since the experiment already has a manifest. Rejected because
"I copied this to a second backup drive" is learned long after the experiment
finished, and is knowledge about a dataset, not about a run. Writing it into a
published, checksummed generation manifest would also violate the immutability
that `docs/source/tutorials/directory_organization.md` already establishes.
