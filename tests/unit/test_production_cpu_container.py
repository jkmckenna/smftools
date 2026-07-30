from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY_ROOT / "containers" / "cpu" / "Dockerfile"
CONTAINER_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "container.yml"
SMOKE_SCRIPT = REPOSITORY_ROOT / "tests" / "container" / "smoke_cpu_image.sh"


@pytest.mark.unit
def test_production_image_is_pinned_nonroot_and_wheel_based():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "python:3.12.13-slim-bookworm@sha256:" in dockerfile
    assert "python -m build --wheel" in dockerfile
    assert "pip install --no-cache-dir /wheels/smftools-*.whl" in dockerfile
    assert "pip install -e" not in dockerfile
    assert 'test -n "${SMFTOOLS_REVISION}"' in dockerfile
    assert 'test -n "${SMFTOOLS_CONTAINER_TAG}"' in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert 'ENTRYPOINT ["smftools"]' in dockerfile
    assert "MPLCONFIGDIR=/tmp/matplotlib" in dockerfile
    for package in ("bash", "minimap2", "procps", "samtools"):
        assert package in dockerfile


@pytest.mark.unit
def test_container_acceptance_builds_without_publishing_and_scans_image():
    workflow = CONTAINER_WORKFLOW.read_text(encoding="utf-8")
    smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "docker build" in workflow
    assert "docker push" not in workflow
    assert "anchore/sbom-action@v0.24.0" in workflow
    assert "aquasecurity/trivy-action@v0.36.0" in workflow
    assert "experiment run" in smoke
    assert "--target full" in smoke
    assert "compatible_skip" in smoke
    assert "experiment validate /work/relocated" in smoke
    assert "--user 12345:12345" in smoke
    assert "trap cleanup EXIT" in smoke
    assert "--user 0:0" in smoke
    assert "/cleanup -mindepth 1 -delete" in smoke
