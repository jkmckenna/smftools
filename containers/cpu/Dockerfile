# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

FROM ${PYTHON_IMAGE} AS wheel-builder

WORKDIR /src

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade "pip==25.1.1" "build==1.2.2.post1" \
    && python -m build --wheel --outdir /wheels


FROM ${PYTHON_IMAGE} AS runtime

ARG SMFTOOLS_REVISION
ARG SMFTOOLS_CONTAINER_IMAGE=ghcr.io/jkmckenna/smftools
ARG SMFTOOLS_CONTAINER_TAG
ARG TORCH_VERSION=2.8.0

LABEL org.opencontainers.image.title="smftools CPU" \
      org.opencontainers.image.description="Pinned CPU runtime for BAM-entry smftools workflows" \
      org.opencontainers.image.source="https://github.com/jkmckenna/smftools" \
      org.opencontainers.image.revision="${SMFTOOLS_REVISION}" \
      org.opencontainers.image.version="${SMFTOOLS_CONTAINER_TAG}" \
      org.opencontainers.image.licenses="MIT" \
      io.smftools.profile="cpu-bam" \
      io.smftools.base.name="python:3.12.13-slim-bookworm" \
      io.smftools.base.digest="sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"

RUN test -n "${SMFTOOLS_REVISION}" \
    && test -n "${SMFTOOLS_CONTAINER_TAG}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        gzip \
        libgomp1 \
        minimap2 \
        procps \
        samtools \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheel-builder /wheels /wheels

# Install the CPU wheel first so the smftools dependency resolver cannot select
# a CUDA-enabled PyTorch distribution from the default package index.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
    && python -m pip install --no-cache-dir /wheels/smftools-*.whl \
    && python -m pip check \
    && rm -rf /wheels

RUN groupadd --gid 10001 smftools \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash smftools \
    && install -d -o smftools -g smftools /work

ENV HOME=/tmp/home \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    SMFTOOLS_CONTAINER_IMAGE="${SMFTOOLS_CONTAINER_IMAGE}" \
    SMFTOOLS_CONTAINER_PROFILE=cpu-bam \
    SMFTOOLS_CONTAINER_REVISION="${SMFTOOLS_REVISION}" \
    SMFTOOLS_CONTAINER_TAG="${SMFTOOLS_CONTAINER_TAG}"

WORKDIR /work
USER 10001:10001

ENTRYPOINT ["smftools"]
CMD ["--help"]
