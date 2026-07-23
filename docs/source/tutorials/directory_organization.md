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
    │   └── <run_name>/               # One smftools experiment per sequencing run
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

One directory per sequencing run, holding the experiment config and everything
`smftools experiment` produces for it:

```text
<run_name>/
├── experiment_config.csv        # input_data_path -> ../../../data/<run_name>/...
│                                 # fasta           -> ./<ref>.fasta
│                                 # output_directory -> ./<date>_outputs/
├── <ref>.fasta                  # Reference FASTA used for this run
├── <date>_outputs/
│   ├── full_summary.json          # Linked outcomes/logs from experiment full
│   ├── raw_outputs/             # smftools experiment raw
│   ├── preprocess_adata_outputs/  # smftools experiment preprocess
│   ├── spatial_adata_outputs/     # smftools experiment spatial
│   ├── hmm_adata_outputs/         # smftools experiment hmm
│   ├── load_adata_outputs/        # smftools experiment load (optional dense cache)
│   └── ...                        # variant_adata_outputs/, latent_adata_outputs/,
│                                   # chimeric_adata_outputs/ if those stages run
└── README.md                     # What this run is, who ran it, what it found
```

Each of the four standard stage directories contains a `logs/` directory with a human log and a
JSONL performance log for every invocation, including explicit skipped and failed outcomes. The
top-level `full_summary.json` uses paths relative to `<date>_outputs/`, so those links remain valid
when a completed experiment tree is moved.

Every stage directory under `<date>_outputs/` is a sibling of the others — that's
not just cosmetic, it's what lets a later stage's spine find an earlier stage's
data by relative path (see [Portability](#portability) below). Run folder names
are typically `YYMMDD_<short_description>` (sequencing date, not analysis date),
matching the `data/<run_name>/` folder it reads from.

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
│   └── <run_name> -> ../../../runs/<run_name>/<date>_outputs
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

The symlink points at the run's `<date>_outputs/` directory — registration
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
ln -s ../../../runs/<run_name>/<date>_outputs <run_name>
```

## Typical workflow

1. **Run the pipeline for one experiment.** Either stage by stage:

   ```shell
   smftools experiment raw analyses/runs/<run_name>/experiment_config.csv
   smftools experiment preprocess analyses/runs/<run_name>/experiment_config.csv
   smftools experiment spatial analyses/runs/<run_name>/experiment_config.csv
   smftools experiment hmm analyses/runs/<run_name>/experiment_config.csv
   ```

   or as one wrapped call that respects each stage's normal skip/restart behavior:

   ```shell
   smftools experiment full analyses/runs/<run_name>/experiment_config.csv
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
       analyses/runs/<run_name>/<date>_outputs
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

**What it does *not* cover:** `experiment_config.csv` itself still holds
absolute paths (`input_data_path`, `fasta`, `output_directory`) — those are
user-supplied, not internal bookkeeping, so smftools has no way to know they
moved. Two ways to handle that:

- **Keep the same absolute mount point on every machine** (a shared lab
  server/NFS mount, or a consistently-named external drive) — configs never
  need editing, and the whole tree is portable with zero manual steps.
- **Edit the config's paths per machine** when you genuinely relocate the data
  to a new absolute location. You only need to do this for
  `experiment_config.csv`; the pipeline's own outputs and the project registry
  don't need touching.

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
