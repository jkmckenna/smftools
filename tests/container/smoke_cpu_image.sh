#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: smoke_cpu_image.sh IMAGE}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
smoke_root="$(mktemp -d)"
trap 'chmod -R u+rwX "${smoke_root}" 2>/dev/null || true; rm -rf "${smoke_root}"' EXIT
chmod 0777 "${smoke_root}"

image_id="$(docker image inspect --format '{{.Id}}' "${image}")"

docker run --rm \
  --read-only \
  --entrypoint /bin/bash \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m,mode=1777 \
  --mount "type=bind,src=${smoke_root},dst=/work" \
  --mount "type=bind,src=${repo_root}/tests/container/fixtures,dst=/fixtures,readonly" \
  --mount "type=bind,src=${repo_root}/tests/_test_inputs/parallel_dispatch,dst=/inputs,readonly" \
  --env HOME=/tmp/home \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env "SMFTOOLS_CONTAINER_DIGEST=${image_id}" \
  "${image}" \
  -euxo pipefail -c '
    test "$(id -u)" = "10001"
    command -v ps
    command -v minimap2
    command -v samtools
    smftools --version
    smftools versions --json > /work/versions-before-run.json
    smftools experiment plan /fixtures/bam_full_config.csv --target full --json \
      > /work/plan-before-run.json
    smftools experiment run /fixtures/bam_full_config.csv \
      --target full \
      --output-root /work/output \
      --input /inputs/sample.bam \
      --fasta /inputs/sample.fasta \
      --cpus 2 \
      --memory-gb 4 \
      --accelerator cpu \
      --strict
    smftools experiment validate /work/output --json > /work/validation.json
    smftools experiment run /fixtures/bam_full_config.csv \
      --target full \
      --output-root /work/output \
      --input /inputs/sample.bam \
      --fasta /inputs/sample.fasta \
      --cpus 2 \
      --memory-gb 4 \
      --accelerator cpu \
      --strict
    python -c "import json; p=json.load(open(\"/work/output/workflow_result.json\")); assert p[\"outcome\"] == \"compatible_skip\"; assert p[\"target\"] == \"full\""
    python -c "import json; p=json.load(open(\"/work/output/software_versions.json\")); assert p[\"container\"][\"profile\"] == \"cpu-bam\"; assert p[\"container\"][\"digest\"].startswith(\"sha256:\")"
    cp -a /work/output /work/relocated
    mv /work/output /work/original-moved
    smftools experiment validate /work/relocated --json > /work/relocated-validation.json
  '

# Apptainer commonly exposes the invoking host UID without adding it to the
# image passwd database. Confirm the CLI remains usable under that model.
docker run --rm \
  --read-only \
  --entrypoint /bin/bash \
  --user 12345:12345 \
  --tmpfs /tmp:rw,nosuid,nodev,size=128m,mode=1777 \
  --env HOME=/tmp \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  "${image}" \
  -euxo pipefail -c '
    test "$(id -u)" = "12345"
    ps -o pid= -p "$$"
    smftools versions --json
  '
