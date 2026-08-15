from __future__ import annotations

from pathlib import Path

import pytest

from smftools.constants import SEMANTIC_GRAPH_DEFINITION_VERSION
from smftools.project import embedding_store, sample_analysis

pytestmark = pytest.mark.unit


def _periodicity_dir(project_dir: Path) -> Path:
    definition = sample_analysis._periodicity_definition(
        layer=None,
        start=None,
        end=None,
        method="direct",
        kwargs={},
    )
    return sample_analysis._analysis_dir(
        project_dir,
        "experiment-a",
        "reference_top",
        "bc01",
        sample_analysis.PERIODICITY_ANALYSIS_NAME,
        sample_analysis._definition_hash(definition),
    )


def _embedding_dir(project_dir: Path) -> Path:
    return embedding_store.embedding_dir(project_dir, "canonical-reference")


def test_code_identity_is_explicit_in_analysis_definitions() -> None:
    periodicity = sample_analysis._periodicity_definition(
        layer=None,
        start=None,
        end=None,
        method="direct",
        kwargs={},
    )
    embedding = embedding_store._embedding_definition(
        canonical_reference="canonical-reference",
        set_name=None,
        modality=None,
        experiments=None,
        stage=None,
        layer=None,
        start=None,
        end=None,
        feature_kind="raw",
        leiden_resolution=0.5,
        n_neighbors=15,
        min_reads=10,
        random_state=42,
    )

    assert periodicity["algorithm_version"] == sample_analysis.PERIODICITY_ALGORITHM_VERSION
    assert embedding["algorithm_version"] == embedding_store.EMBEDDING_ALGORITHM_VERSION
    assert periodicity["graph_definition_version"] == SEMANTIC_GRAPH_DEFINITION_VERSION
    assert embedding["graph_definition_version"] == SEMANTIC_GRAPH_DEFINITION_VERSION


def test_periodicity_algorithm_bump_invalidates_only_periodicity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    old_periodicity = _periodicity_dir(project_dir)
    old_embedding = _embedding_dir(project_dir)
    old_periodicity.mkdir(parents=True)
    old_embedding.mkdir(parents=True)
    periodicity_marker = old_periodicity / "existing-cache"
    embedding_marker = old_embedding / "existing-cache"
    periodicity_marker.write_text("preserved", encoding="utf-8")
    embedding_marker.write_text("preserved", encoding="utf-8")

    monkeypatch.setattr(sample_analysis, "PERIODICITY_ALGORITHM_VERSION", "2")

    assert _periodicity_dir(project_dir) != old_periodicity
    assert _embedding_dir(project_dir) == old_embedding
    assert periodicity_marker.read_text(encoding="utf-8") == "preserved"
    assert embedding_marker.read_text(encoding="utf-8") == "preserved"


def test_embedding_algorithm_bump_invalidates_only_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    old_periodicity = _periodicity_dir(project_dir)
    old_embedding = _embedding_dir(project_dir)

    monkeypatch.setattr(embedding_store, "EMBEDDING_ALGORITHM_VERSION", "2")

    assert _periodicity_dir(project_dir) == old_periodicity
    assert _embedding_dir(project_dir) != old_embedding


def test_graph_definition_bump_invalidates_both_analysis_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    old_periodicity = _periodicity_dir(project_dir)
    old_embedding = _embedding_dir(project_dir)
    bumped = SEMANTIC_GRAPH_DEFINITION_VERSION + 1

    monkeypatch.setattr(sample_analysis, "SEMANTIC_GRAPH_DEFINITION_VERSION", bumped)
    monkeypatch.setattr(embedding_store, "SEMANTIC_GRAPH_DEFINITION_VERSION", bumped)

    assert _periodicity_dir(project_dir) != old_periodicity
    assert _embedding_dir(project_dir) != old_embedding
