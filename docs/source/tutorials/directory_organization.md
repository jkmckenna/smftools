# Organizing data, experiments, and projects

smftools separates three things that change at different rates and should live in
different places: raw instrument output (never changes), one experiment's pipeline
outputs (regenerable from raw + config), and a project that spans multiple
experiments (a set of pointers, not a copy). This page lays out a directory
convention for that split, walks through the typical `experiment` → `project`
workflow, and covers what it takes to move or share that directory tree.

## Recommended layout

```text
lab_root/
├── data/
│   └── <run_name>/                  # Raw pod5/fast5/fastq, straight off the
│                                     # instrument. Read-only, never modified.
└── analyses/
    ├── runs/
    │   └── <run_name>/               # One sequencing run; one or more experiments
    └── projects/
        └── <project_name>/           # Cross-experiment registries + comparisons
```

The split matters because the two halves have different lifecycles: `data/` is
expensive to regenerate and should be treated as immutable, while everything
under `analyses/` is derived and can be deleted and rebuilt from `data/` + an
`experiment_config.csv` if you ever need to. Keeping them apart also makes the
portability story in the next section simpler, since only `analyses/` needs to
move when you share or migrate a project.

### `analyses/runs/<run_name>/`

One directory per sequencing run, with one canonical directory per experiment.
A run may produce more than one experiment (for example, separate modalities):

```text
<run_name>/
├── <experiment_id>/
│   ├── experiment_config.csv      # experiment_id/name -> <experiment_id>
│   │                               # output_directory -> this directory
│   ├── <ref>.fasta                # Reference FASTA used for this experiment
│   ├── full_summary.json          # Linked outcomes/logs from experiment full
│   ├── experiment_manifest.json   # Lifecycle, provenance, identity, code version
│   ├── raw_outputs/               # smftools experiment raw
│   ├── preprocess_adata_outputs/  # smftools experiment preprocess
│   ├── spatial_adata_outputs/     # smftools experiment spatial
│   ├── hmm_adata_outputs/         # smftools experiment hmm
│   ├── load_adata_outputs/        # smftools experiment load (optional dense cache)
│   └── ...                        # latent_adata_outputs/, chimeric_adata_outputs/
└── README.md                      # What this run is, who ran it, what it found
```

Each of the four standard stage directories contains a `logs/` directory with a human log and a
JSONL performance log for every invocation, including explicit skipped and failed outcomes. The
top-level `full_summary.json` uses paths relative to `<experiment_id>/`, so those links remain valid
when a completed experiment tree is moved.

Every stage directory under `<experiment_id>/` is a sibling of the others — that's
not just cosmetic, it's what lets a later stage's spine find an earlier stage's
data by relative path (see [Portability](#portability) below). Run folder names
are typically `YYMMDD_<short_description>` (sequencing date, not analysis date),
matching the `data/<run_name>/` folder it reads from.
The experiment directory, config `experiment_id`, compatibility
`experiment_name`, manifest identity, and `project add --id` must all agree.
This prevents the same run from acquiring different labels depending on which
fallback a command happened to use.

### Immutable generations and retention

Generation-aware stage directories keep published outputs immutable and select
one default with an atomic pointer. A raw stage, for example, has this shape:

```text
raw_outputs/
├── current.json
├── retention.json                     # Optional mutable pin registry
└── generations/
    ├── <generation_id>/
    │   ├── generation_manifest.json
    │   └── ...                         # Generation-owned artifacts
    └── <older_generation_id>/
        ├── generation_manifest.json
        └── ...
```

`current.json` identifies the default generation consumed by ordinary readers.
`generation_manifest.json` and its generation directory are published once and
must not be edited afterward. Retention reasons therefore live separately in
`retention.json`, beside `current.json`, and can be added or removed without
invalidating a published manifest or its checksum.

Inventory and pin generations through the experiment CLI:

```shell
smftools experiment generations <experiment_id> --size
smftools experiment generations <experiment_id> pin raw <generation_id> \
  --reason "paper figure 3"
```

`smftools experiment generations <experiment_id> prune --keep-last 2` is
currently a dry-run planner only. It protects current, pinned, unreadable,
recent, and newest generations. Even older policy matches remain blocked until
their byte-level reproducibility from retained inputs is represented
authoritatively; this phase does not expose deletion or force behavior.

### Analysis caches across smftools versions

Project periodicity and embedding caches include two forms of code identity in
their definition hashes: an independently bumpable version for that analysis's
algorithm and the shared semantic graph definition version. A change confined
to periodicity therefore creates a new periodicity cache key without invalidating
embedding caches, and the reverse is also true. A semantic graph change creates
new keys for both.

Existing cache directories are never rewritten or deleted during an upgrade.
They simply stop matching the current definition, and the next requested
analysis computes under a new key. This preserves old results for inspection
while preventing changed code from silently serving them as current output.

Inventory the retained definitions before recomputing with:

```shell
smftools project analyses list PROJECT_DIR --stale
```

The read-only listing reports stale and structurally invalid caches, their
project-relative paths and sizes, and why their stored code identity does not
match the installed version. Add `--json` for the stable schema. The inventory
does not read analysis result tables or unpickle embedding models.

### `analyses/projects/<project_name>/`

A project **references** runs — it never copies or merges their data.
`smftools project init` scaffolds the directory with both the machine-managed
registry and a set of starter docs/working directories (skipping anything that
already exists, so it's safe to re-run):

```text
<project_name>/
├── registry.json          # Which experiments belong to this project, and where
├── sets/                  # Named experiment subsets
├── runs/                  # Symlinks only -- no data
│   └── <experiment_id> -> ../../../runs/<run_name>/<experiment_id>
├── project_scripts/       # Project-specific drivers/constants (importable package)
├── project_outputs/       # Materialized/derived outputs (project materialize -o, figures)
├── project.yaml           # Human-curated run/reference manifest (not read by smftools)
├── README.md
├── AGENTS.md              # Working context for coding agents
├── CLAUDE.md              # Points Claude Code at AGENTS.md
└── PLAN.md                # Current objective / status / next steps
```

The `registry.json`/`sets/`/`runs/` pieces are the only ones smftools itself
reads back — `project_scripts/`, `project_outputs/`, `project.yaml`, and the
README/AGENTS/CLAUDE/PLAN docs are starting points for you (or a coding agent)
to fill in as the project develops.

The symlink points at the canonical `<experiment_id>/` directory — registration
discovers *every* pipeline stage under it (`raw_outputs/`, `preprocess_adata_outputs/`,
`spatial_adata_outputs/`, `hmm_adata_outputs/`, ...), not just one, so a project
query can pull from whichever stage it needs per experiment (see
[Every stage is visible to the project](#every-stage-is-visible-to-the-project)
below). Pointing the symlink at one specific stage dir (e.g. `raw_outputs/`
directly) also works — its siblings are still discovered — so either convention
is fine. Use a *relative* symlink so it survives the project being copied
elsewhere:

```shell
cd analyses/projects/<project_name>/runs
ln -s ../../../runs/<run_name>/<experiment_id> <experiment_id>
```

## Typical workflow

1. **Run the pipeline for one experiment.** Either stage by stage:

   ```shell
   smftools experiment raw analyses/runs/<run_name>/<experiment_id>/experiment_config.csv
   smftools experiment preprocess analyses/runs/<run_name>/<experiment_id>/experiment_config.csv
   smftools experiment spatial analyses/runs/<run_name>/<experiment_id>/experiment_config.csv
   smftools experiment hmm analyses/runs/<run_name>/<experiment_id>/experiment_config.csv
   ```

   or as one wrapped call that respects each stage's normal skip/restart behavior:

   ```shell
   smftools experiment full analyses/runs/<run_name>/<experiment_id>/experiment_config.csv
   ```

   Processing many runs the same way is one `batch` call instead of a shell loop:

   ```shell
   smftools experiment batch full analyses/runs/config_paths.csv
   ```

2. **Register the experiment into a project.** Create the project once, then add
   experiments to it as they finish (append-only — nothing is copied). Every
   pipeline stage that has run gets recorded, whether you point at the run's
   top-level output directory or at one stage dir specifically:

   ```shell
   smftools project init analyses/projects/<project_name>
   smftools project add analyses/projects/<project_name> \
       analyses/runs/<run_name>/<experiment_id> --id <experiment_id>
   ```

3. **Query and combine across the project.** `project list` shows registered
   experiments (including which stages each has reached), and the reference
   names smftools has harmonized across them (by sequence identity, so the same
   locus can be called different things in different experiments' FASTAs);
   `project materialize` resolves one canonical reference back to each matching
   experiment's own name and concatenates the slices (never a global merge):

   ```shell
   smftools project list analyses/projects/<project_name>
   smftools project materialize analyses/projects/<project_name> my_canonical_reference \
       -o analyses/projects/<project_name>/outputs/my_canonical_reference.h5ad.gz
   smftools project materialize analyses/projects/<project_name> my_canonical_reference \
       -o analyses/projects/<project_name>/outputs/my_canonical_reference_parts --partitioned
   ```

   By default this pulls each experiment's most-derived available stage (HMM >
   spatial > preprocess > raw); pass `--stage preprocess` (or any other stage
   name) to pin all experiments to one specific stage instead, skipping any
   that haven't reached it yet. `--read-metrics` additionally attaches
   spatial's per-read outputs (autocorrelation, Lomb-Scargle) where available. Use the partitioned
   form for selections that should remain independently readable without a final pooled AnnData.

4. **Export raw reads across a project**, e.g. for a re-analysis pipeline outside
   smftools, the same way you would for one experiment:

   ```shell
   smftools project export-fastq analyses/projects/<project_name> --outdir ./fastqs
   ```

## Registering legacy (pre-partitioned-store) runs

Older smftools runs produced a single monolithic `.h5ad`/`.h5ad.gz` per stage
(e.g. `<experiment>_preprocessed_duplicates_removed.h5ad.gz`) instead of the
partitioned spine + task-store layout described above. There's no need to
convert these before joining a project — `project add` accepts a file path
directly, and `materialize()` detects a legacy spine (no `uns["is_spine"]`)
and reads it directly instead of through the partition machinery, so every
later `project` query treats it the same as a modern run:

```shell
smftools project add analyses/projects/<project_name> \
    /path/to/<experiment>_preprocessed_duplicates_removed.h5ad.gz \
    --id <experiment> --stage preprocess
```

`--stage` names which pipeline stage the file represents; omit it and
smftools guesses from the filename (`_preprocessed` → preprocess, `_spatial`
→ spatial, `_hmm` → hmm, ..., defaulting to `raw`). Register each stage file
for the same experiment with repeated calls (same `--id`) — stages accumulate
onto the same registry entry rather than replacing each other, so registering
`_hmm.h5ad.gz` after `_preprocessed.h5ad.gz` doesn't lose the earlier one.

The legacy file itself is only ever **read**, never modified: reference
identity for cross-experiment harmonization is computed on the fly from
`uns["References"]` at registration time (falling back to it only when the
file predates `uns["reference_uids"]`) rather than being cached back into the
source. This keeps the original monolithic file byte-for-byte untouched, so
it's safe to register into a project without disturbing whatever else still
depends on it.

## Every stage is visible to the project

Within one experiment, later pipeline stages don't lose access to earlier
stages' output — but the mechanism differs by what kind of data it is:

- **Per-read metadata** (obs columns: QC flags, dedup status, ...) is fully
  cumulative. Each stage's derived spine is a copy of whatever spine it started
  from, so everything an earlier stage added is still there in a later stage's
  spine.
- **Per-position derived layers** (binarized methylation, HMM state calls) live
  in each stage's own task store, addressed by a pointer a later stage's spine
  carries forward. `materialize()` resolves these transparently — an HMM task
  loads preprocess's derived layers as its own model input without them ever
  being duplicated into HMM's own store.
- **Spatial's per-read outputs** (autocorrelation curves, Lomb-Scargle
  periodograms) are a different shape (read × lag, not read × position), so
  they're opt-in rather than loaded automatically: pass `read_metrics=True` (or
  a specific name subset) to `materialize()` to attach them.

The project registry builds directly on this: registering an experiment
records every stage spine found for it, and `project materialize`'s default
stage fallback (most-derived first) means pointing at whichever stage happens
to be furthest along per experiment already exposes everything upstream of it
— you don't need to separately register or query each stage.

## Portability

As of smftools 2.1, everything a project or a later pipeline stage stores about
*where its own artifacts live* is written as a path relative to a stable anchor
(a run's `output_directory`, or a project's own directory) rather than an
absolute, machine-specific string. Concretely:

- A stage spine's cross-stage pointers (e.g. a spatial spine's pointer back to
  its source preprocess catalog) resolve relative to the run's
  `output_directory`, recovered structurally from wherever that spine file
  currently lives — not from a value baked in when it was originally written.
- `obs["bam_path"]` on the raw spine resolves the same way, since the aligned
  BAM lives alongside the raw store under `raw_outputs/bam_outputs/`.
- A project's `registry.json` stores each experiment's path relative to the
  project directory, the same way its `runs/` symlinks already do.

Old spines/registries written before this (absolute strings) still work — the
reader accepts both — so you don't need to regenerate anything already on disk.

**What this means in practice:** you can `rsync`/copy the *whole* `analyses/`
tree (or `data/` + `analyses/` together) to a different machine or mount point,
at a different absolute path than the original, and every relative pointer
(spine cross-references, project registry, `runs/` symlinks) resolves correctly
without editing or re-running `project add`.

### Archived raw input is not an error

Once an experiment's pipeline outputs exist, its raw `data/<run_name>/`
input is routinely moved to archival storage — there's no reason to keep
terabytes of pod5 attached once `raw` has ingested it. Config loading treats
that as one of three expected states rather than a single failure:

- **present** — the input resolves normally.
- **offline** — the path lies on a volume that isn't currently attached.
  Expected, not an error.
- **missing** — the volume is attached (or was never removable) and the path
  is genuinely absent. A real error, caught at config load exactly as before.

Only the stages that actually read raw input (`raw`, and re-basecalling)
refuse while it's offline, naming the volume to attach. Every other stage —
`preprocess`, `variant`, `chimeric`, `spatial`, `hmm`, `latent`,
`export-bundle`, `export-fastq` — runs normally, since none of them touch a
byte of it:

```shell
smftools experiment hmm my_run/experiment_config.csv   # works, archive drive unplugged
smftools experiment raw my_run/experiment_config.csv   # refuses, names the volume to attach
```

### Named roots in a config

A config no longer has to name absolute paths. `${data}/<run>/pod5` resolves
through a **root** bound on the machine, so the config is portable and only the
binding is local:

```toml
# ~/.config/smftools/roots.toml, or a roots.toml beside (or above) the config
[roots]
data = "/Volumes/ArchiveDrive01"
analyses = "/Volumes/WorkSSD/analyses"
```

Resolution takes the first match of: the environment variable
`SMFTOOLS_ROOT_<NAME>`, the user roots file (`$SMFTOOLS_CONFIG_DIR/roots.toml`
or `~/.config/smftools/roots.toml`), then any `roots.toml` found walking up from
the config's own directory. A root with no binding is an error, never a literal
— a typo'd name must not become a directory name.

Expansion applies to the path values you write by hand (`input_data_path`,
`input_manifest_path`, `output_directory`, `fasta`, the bed files,
`sample_sheet_path`, `sequencing_summary_path`, `model_dir`,
`custom_barcode_yaml`, `umi_yaml`) and to nothing else; every other path is
derived from `output_directory`.

**A binding can also be an ordered list of locations**, for a root that
doesn't live in exactly one place — `analyses` might hold some runs on an
internal disk and others on an external SSD:

```toml
[roots]
analyses = ["/Users/you/analyses", "/Volumes/WorkSSD/analyses"]
```

`${analyses}` expands to whichever location currently exists, checked in the
listed order (falling back to the first when creating a new run, since
nothing exists yet). One run's own tree always resolves under a single
location — never split across two — since everything it names is checked
together as it's substituted.

### Relative paths anchor to the config

A bare relative path in a config now resolves against **the config file's own
directory**, not the working directory, so an experiment directory is
self-describing and means the same thing wherever you run smftools from:

```text
fasta,my_reference.fasta
output_directory,store
```

**Migration.** Configs written before this resolved relative paths against the
working directory. If the config-relative path does not exist and the old
working-directory one does, the old reading is still honoured and a warning says
so — nothing breaks on upgrade. Make such paths absolute, root-qualified, or
genuinely relative to the config to stop depending on where the command is run.

Where both readings exist, the config-relative one wins.

### Volume identity for removable drives

Named roots solve *"where does `${data}` point on this machine"* — one binding
per machine. They don't solve a narrower, common problem: raw sequencing
output is routinely archived to more than one removable drive (the original,
a backup, institutional storage), a drive's mount point and OS-reported name
change depending on which machine it's plugged into and what else is already
attached, and none of that has anything to do with `${data}`'s binding on any
one machine. This layer exists for exactly that case, and — like named roots
— is entirely opt-in; if you never touch it, nothing here changes.

**Stamp a drive once, permanently:**

```shell
smftools data init-volume /Volumes/ArchiveDrive01 --label archive-01 --kind archive
```

This writes a small `.smftools-volume.json` identity file at the drive's own
root. The stamp is written once and never rewritten, so the drive keeps its
identity even if it's later relabeled at the OS level or reattached under a
different mount point elsewhere — nothing about identifying it depends on its
current name.

**See what's attached right now:**

```shell
smftools data volumes
```

Scans the platform's usual mount locations (`/Volumes` on macOS; `/mnt`,
`/media/<user>`, `/run/media/<user>` on Linux) plus anything you've added to
`[volumes] extra_search_paths` in `roots.toml`, for network shares that live
outside those conventions.

**Index what a drive holds into the replica catalog:**

```shell
smftools data scan /Volumes/ArchiveDrive01
```

Walks the drive for published input manifests and records, per dataset
(identified by the same relocation-invariant digest the manifest already
computes — not by path), which stamped volume holds a copy and where. The
catalog itself is a plain JSON file next to `roots.toml`, copyable between
your own machines and rebuildable from attached drives by re-running `scan`.

**Find a dataset without plugging anything in:**

```shell
smftools data locate analyses/runs/240101_run/experiment_1
smftools data verify analyses/runs/240101_run/experiment_1
```

`locate` reports every catalogued replica and which are currently attached —
this is the point of a catalog: it answers while the drive is unplugged.
`verify` re-checksums a replica's declared sources directly against the
files, catching corruption a matching `volume_id` alone would not — a stamp
is an identifier, not an integrity guarantee.

**What this buys you day to day:** once a run has been scanned at least once,
`smftools experiment <stage>` on it stops guessing. If the archive drive
simply isn't attached, that's a confident, expected "offline" — even for a
network share that doesn't match any recognized mount convention, which
would otherwise read as a hard error. If the drive *is* attached, just under
a different mount point or name than when the config was written, the run
resolves it transparently and proceeds — no config edit required.

**The cheapest alternative, if this is more than you need:** for a single
experiment, `smftools data localize CONFIG_PATH --apply` copies just the
small, hand-edited inputs (`fasta`, the BED region files, the sample sheet,
barcode/UMI YAML — never the raw data itself) into the run's own output
directory and writes a new, self-contained config. No roots, no volumes, no
catalog required to read the result elsewhere. `smftools data init LAB_ROOT`
scaffolds a fresh `data/` + `analyses/{runs,projects}/` tree for a new lab
root the same way `project init` does for a single project, and can
optionally stamp the drive it's given at the same time with `--stamp-volume`.

### Sharing a project with a collaborator

Split the same way the directory layout already splits code from data:

- **Version-control the small, text artifacts**: `registry.json`, `sets/`,
  `README.md`, and any analysis scripts or manifests you've layered on top
  (e.g. a `project.yaml`/`samples.csv` if you maintain your own per-sample
  metadata alongside the registry). These are cheap, diff-friendly, and this is
  exactly what git is for.
- **Sync the data separately**: the referenced `analyses/runs/<run_name>/`
  directories (and `data/<run_name>/` if your collaborator needs the raw
  instrument files too) are too large for git — use shared storage, rsync, or
  institutional data transfer instead.
- Because paths are relative now, your collaborator can put the synced run
  directories anywhere on their machine, recreate the project's `runs/`
  symlinks (or just re-run `project add`, which is idempotent), and everything
  resolves from there — no absolute-path coordination required between you.
