# AGENTS.md

This file tells coding agents (including OpenAI Codex and Claude Code) how to work in this repo.

Coding agents can only read from AGENTS.md or Claude.md files.
Agents can not edit AGENTS.md or Claude.md files.

## Goals
- Make minimal, correct changes.
- Prefer small PRs / diffs.
- Keep behavior stable unless the task explicitly requests changes.
- Generate production grade, scalable code.

## Prompt interface
- When asked about a problem or task, first describe the plan to handle the task.
- Keep taking prompts until the plan is validated.
- Implement code after being told to proceed.

## Repo orientation
- Read existing patterns before inventing new ones.
- Don’t refactor broadly unless asked.
- If you’re unsure about intended behavior, look for tests/docs first.
- If behavior is not clear after tests/docs, look at the Click commands section in this file.
- Ignore all files in any directory named "archived".
- User defined parameters exist within src/smftools/config.
- Parameters are herited from default.yaml -> MODALITY.yaml -> user_defined_config.csv
- Frequently used non user defined variables should exist within src/smftools/constants.py
- Logging functionality is defined within src/smftools/logging_utils.py
- Optional dependency handling is defined within src/smftools/optional_imports.py
- Frequently used I/O functionality is defined within src/smftools/readwrite.py
- CLI functionality is provided through click and is defined within:
  - src/smftools/cli_entry.py
  - Modules of the src/smtools/cli subpackage
- RTD documentation organization through smftools/docs
- Pytest testing within smftools/tests

## Project dependencies
- A core set of dependencies is required for the project.
- Various optional dependencies are provided for:
    - Optional functional modules of the package (ont, plotting, ml-base, ml-extended, umap, qc)
    - If available, a Python version of a CLI tool is preferred (Such as for Samtools, Bedtools, BedGraphToBigWig).
    - torch is listed as an extra dependency, but is currently required.
    - All dependencies can be installed with `pip install -e ".[all]"`
- Certain command line tools are currently needed for certain functionalities within smftools load:
  - dorado: Used for nanopore basecalling from POD5/FAST5 files to BAM.
  - dorado/minimap2: Used for alignment of reads to reference.
  - dorado: Used for demultiplexing of nanopore derived BAMs.
  - modkit: Used for extracting modification probabilities from MM/ML BAM tags for native smf modality.

## Setup
- Use current environment if the core dependencies are installed.
- If dependencies are not found, create a venv in smftools/venvs/ directory:
  - `python3 -m venv .temp-venv && source .temp-venv/bin/activate`
- Install the core dependencies and development dependencies for testing/formatting/linting:
  - `pip install -e ".[dev,torch]"`
- If code is raising dependencies errors and they are in the optional dependencies:
  - `pip install -e ".[EXTRA_DEPENDENCY_NAME]"`

## How to run checks
- Smoke tests: `pytest -m smoke -q`
- Unit tests: `pytest -m unit -q`
- Integration tests: `pytest -m integration -q`
- E2E tests: `pytest -m e2e -q`
- Coverage (if configured): `pytest --cov`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type-check (if configured): `mypy .`

## Coding conventions
- Follow existing style and module layout.
- Prefer clear, explicit code over cleverness.
- Prefer modular functionality to facilitate testing and future development.
- Do not over-parametize functions when possible.
- For function parameters that a user may want to tune, use the config management strategy.
- Use constants.py when appropriate.
- Annotate code blocks to describe functionality.
- Add/adjust tests for bug fixes and new behavior.
- Keep public APIs backward compatible unless explicitly changing them.
- Python:
  - Use type hints for new/modified functions where reasonable.
  - Use Google style docstring format.
  - Avoid heavy dependencies unless necessary.
  - Use typing.TYPE_CHECKING and annotations.
  - In docstring of new functions, define the purpose of the function and what it does.

## Testing expectations
- New functionality must include tests.
- If tests are flaky or slow, note it and scope the change.

## Logging & secrets
- Don’t log secrets, tokens, or PII.
- Never hardcode credentials.
- If sample keys are needed, use obvious placeholders like `YOUR_API_KEY_HERE`.

## Git / PR hygiene
- Keep commits focused.
- Update docs/changelog if behavior or user-facing CLI changes.
- If you change a CLI flag or config schema, add a migration note.

## If something fails
- If a command fails, paste the full error and summarize likely causes.
- Don’t “fix” by deleting tests or weakening assertions unless explicitly instructed.

## Click commands and their primary intent. Look in docs first, and underneath if the task is still not clear.
- smftools load:
  - Take a variety of raw sequencing input options (FASTQs, POD5s, BAMs) from a single molecule footprinting experiment.
  - Determine the smf modality specified by the user (conversion, deaminase, native).
  - Handle FASTA inputs
  - Basecall the files using dorado if needed.
  - Align the reads using dorado or minimap2.
  - Sort/Index/Demultiplex BAMs.
  - BAM QC.
  - Extract Base modification probabilities for native smf modality
  - Load an AnnData object containing:
    - adata.X with a read X position matrix of SMF data.
    - adata.layers with:
      - integer encoded DNA sequences of each read.
      - mismatch encodings of DNA sequence vs reference for each read.
      - Base Q-scores for each read.
      - Read span masks indicating where the read aligned.
    - adata.var with per Reference_strand FASTA bases across positions.
    - adata.var_names being positional indexes within each read.
    - adata.obs_names being read names.
    - adata.obs with read level metadata
    - adata.uns with various unstructured data metrics.
  - Run multiqc on the BAM qc files.
  - Directory temp file cleanup.
  - Write out the adata, it's backup accessory data, and csv files of obs, var, and keys.
- smftools preprocess:
  - Requires the adata produced by smftools load.
  - Adds various QC metrics and performs data preprocessing and filtering.
    - Read length, quality, and mapping based QC.
    - Per reference position level QC.
    - Appending base context for each reference.
    - Binarization of SMF probabilities for the native smf modality
    - NaN filling strategies in adata.layers.
    - Read level modification QC and filtering.
    - Duplicate detection and complexity analysis for conversion/deaminase modalities.
    - Visualizing read spans and base quality clustermaps.
  - Optionally inverts the adata along the var-axis.
  - Optionally reindexes var.
- smftools variant:
  - Requires at least a preprocessed adata object.
  - Calculates per position mismatch frequencies/types for each reference/sample.
  - Optional variant site labeling if comparing two references.
  - Visualized sequence encodings and mismatch encodings with clustermaps.
- smftools chimeric:
  - Requires at least a preprocessed adata object.
  - Meant to detect putative PCR chimeras.
- smftools spatial:
  - Requires at least a preprocessed adata object.
  - Basic spatial signal analyses.
  - Clustermaps to visualize smf signal per reference/sample.
  - Spatial autocorrelation.
  - Position x position correlation matrices (Pearson, Binary covariance, chi2, relative risk)
- smftools hmm:
  - Requires at least a preprocessed adata object.
  - Fits/saves/applies HMM to adata to label putative molecular features.
  - Creates adata.layers that hold binary masks of each feature class/subclass.
  - Creates adata.layers that hold HMM emission probabilities.
  - Visualizes HMM layers with clustermaps.
  - Performs peak calling on HMM layers and labels reads with the features in obs.
- smftools latent:
  - Requires at least a preprocessed adata object.
  - Generates latent representations of the smf data.
  - PCA/KNN/UMAP/NMF/CP decomposition strategies.
  - Represents full sequences.
  - Represents modified sites only.
  - Represents non-modified sites only.