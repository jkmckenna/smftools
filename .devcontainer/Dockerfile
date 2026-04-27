FROM python:3.12-slim-bookworm

# System deps needed by pysam, pybedtools, htslib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libdeflate-dev \
    samtools \
    bedtools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set up a venv so the editable install is isolated
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"

# Pre-install heavy deps so rebuilds are faster
COPY pyproject.toml README.md LICENSE /tmp/smftools/
RUN pip install --no-cache-dir \
    "numpy>=1.22.0,<2" \
    "scipy>=1.7.3" \
    "torch>=1.9.0" \
    "anndata>=0.10.0" \
    "pysam>=0.19.1"

USER $USERNAME
