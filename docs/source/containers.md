# Production CPU container

The production image in `containers/cpu/Dockerfile` is the supported isolated
runtime for the initial CPU, BAM-entry workflow profile. It is separate from
the development container and is built as a non-editable wheel from one exact
source checkout.

## Build an immutable image

Build from a clean checkout and identify both the source revision and image
tag. Use a release version or complete commit SHA; do not publish or consume a
`latest` tag.

```shell
revision="$(git rev-parse HEAD)"
docker build \
  --file containers/cpu/Dockerfile \
  --build-arg "SMFTOOLS_REVISION=${revision}" \
  --build-arg "SMFTOOLS_CONTAINER_TAG=sha-${revision}" \
  --tag "smftools-cpu:sha-${revision}" \
  .
```

The Python base is pinned by multi-platform digest. The build creates a wheel
in a builder stage, installs CPU-only PyTorch and that wheel into the runtime
stage, and runs `pip check`. The runtime image contains `/bin/bash`, `ps`,
`minimap2`, and `samtools`. It runs as UID/GID 10001 and writes beneath
`/work`.

The repository container workflow builds the image without publishing it,
runs the checked-in modification-bearing BAM smoke profile, validates a
compatible restart and relocated result, emits an SPDX JSON SBOM, and scans
for fixable critical vulnerabilities. Publishing an image remains an explicit
release action.

## Run the BAM-entry profile

Mount input files read-only and provide a writable task directory:

```shell
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,mode=1777 \
  --mount type=bind,src="${PWD}/inputs",dst=/inputs,readonly \
  --mount type=bind,src="${PWD}/task-output",dst=/work \
  --env SMFTOOLS_CONTAINER_DIGEST="sha256:IMAGE_DIGEST" \
  "smftools-cpu:sha-COMPLETE_COMMIT_SHA" \
  experiment run /inputs/experiment.csv \
    --target full \
    --output-root /work/output \
    --input /inputs/reads.bam \
    --fasta /inputs/reference.fasta \
    --cpus 8 \
    --memory-gb 32 \
    --accelerator cpu \
    --strict
```

The checked-in direct-modification smoke profile uses `alignment_mode:
existing` because its BAM is already aligned and modification-tagged; this
avoids a lossy BAM-to-FASTQ realignment. For direct-modification workflows that
should not require modkit, set `direct_signal_backend` to `pysam`. Generated
minimap2 alignments require minimap2 2.24.0 or newer. BWA-MEM2 and Bowtie2
adapters require separately supplied executables because the production CPU
image does not package them. Set `samtools_backend` to `python` when the
portable Python backend is desired; the packaged `samtools` executable remains
available for explicitly selected CLI behavior and must be samtools 1.10.0 or
newer.

Validate the result using the same immutable image:

```shell
docker run --rm \
  --read-only \
  --mount type=bind,src="${PWD}/task-output",dst=/work,readonly \
  "smftools-cpu:sha-COMPLETE_COMMIT_SHA" \
  experiment validate /work/output --json
```

When `SMFTOOLS_CONTAINER_DIGEST` is supplied by the workflow engine,
`software_versions.json` records the image name, tag, digest, source revision,
and `cpu-bam` profile. The digest should be the registry manifest digest used
to launch the task, not a mutable tag or local image ID.

## Apptainer-compatible execution

The image does not require root at runtime and the CLI works when an
Apptainer-style arbitrary UID has no matching passwd entry. Convert or pull a
pinned OCI image, bind staged inputs read-only, and bind one writable task
directory:

```shell
apptainer exec --cleanenv \
  --bind "${PWD}/inputs:/inputs:ro" \
  --bind "${PWD}/task-output:/work" \
  smftools-cpu.sif \
  smftools versions --json
```

Scheduler or workflow wrappers should set `HOME`, `MPLCONFIGDIR`, and a
writable temporary directory when the container root filesystem is read-only.
For `docker run`, the image `ENTRYPOINT` is `smftools`, so commands after the
image name begin with `experiment`, `project`, or `versions`; do not repeat
`smftools`. `apptainer exec` executes the command supplied to it directly, so
its example includes the executable.

## Included and excluded tools

The image is intentionally limited to the CPU BAM-entry profile:

- `minimap2` and `samtools` are installed from the pinned Debian base
  distribution. Their packaged copyright and license texts remain under
  `/usr/share/doc`.
- Python runtime packages and their licenses are enumerated in the generated
  SBOM.
- Dorado, CUDA, GPU libraries, vendor model assets, modkit, bedtools,
  BedGraphToBigWig, POD5 tooling, and MultiQC are not bundled.

Workflows that select an excluded executable must supply a separately
reviewed image or process. Review the tool and model licenses independently;
the smftools MIT license does not grant rights to third-party executables or
model assets.
