import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.informatics.derived_read_index import write_latent_read_index
from smftools.informatics.experiment_manifest import artifact_record
from smftools.informatics.experiment_spine import write_experiment_spine
from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.project import registry as reg
from smftools.project.catalog import (
    ProjectCatalog,
    export_project_partitions,
    project_adata,
)
from smftools.project.latent_store import export_latent_parts
from smftools.readwrite import safe_read_h5ad, safe_read_zarr, safe_write_h5ad, safe_write_zarr

SEQUENCE = "ACGTAC"


def _project_with_latent_owners(
    tmp_path,
    *,
    project=None,
    experiment_id="experiment",
    run_name="experiment",
):
    run_root = tmp_path / run_name
    rows = pd.DataFrame(
        [
            {
                "read_id": f"read-{index}",
                "reference": "gene",
                "Reference_strand": "gene_top",
                "sample": "bc01",
                "barcode": "bc01",
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": "6M",
                "aligned_length": 6,
                "sequence": [index % 4] * 6,
                "quality": [30] * 6,
                "mismatch": [4] * 6,
                "modification_signal": [float(index % 2)] * 6,
            }
            for index in range(3)
        ]
    )
    reference_identity = reference_uid(SEQUENCE, len(SEQUENCE))
    raw = write_raw_store(
        rows,
        run_root / "raw_outputs",
        reference_lengths={"gene_top": len(SEQUENCE)},
        extra_uns={
            "reference_uids": {"gene_top": reference_identity},
            "modality": "direct",
            "experiment": experiment_id,
        },
    )
    raw_spine, _ = safe_read_h5ad(raw["spine"], verbose=False)
    generation_id = "generation-a"
    generation = run_root / "latent_adata_outputs" / "generations" / generation_id
    owners = (
        ("core-a", 0, 4, ["read-0", "read-1"]),
        ("core-b", 2, 6, ["read-1", "read-2"]),
    )
    for owner_index, (core_id, start, end, read_ids) in enumerate(owners):
        obs = raw_spine.obs.loc[read_ids].copy()
        obs["leiden_signal"] = pd.Categorical([str(owner_index)] * len(obs))
        obs["unused_label"] = "unused"
        result = ad.AnnData(
            obs=obs,
            var=pd.DataFrame(index=[str(position) for position in range(start, end)]),
        )
        result.obsm["X_pca_signal"] = np.full((len(obs), 2), owner_index, dtype=np.float32)
        result.obsm["X_umap_signal"] = np.full((len(obs), 2), owner_index + 1, dtype=np.float32)
        result.varm["PCs_signal"] = np.full((result.n_vars, 2), owner_index, dtype=np.float32)
        group_relative = (
            Path("latent_adata_outputs")
            / "generations"
            / generation_id
            / "store"
            / f"{core_id}.zarr"
        )
        group_path = run_root / group_relative
        safe_write_zarr(result, group_path, backup=False, verbose=False, zarr_format=3)
        checksum = str(artifact_record(group_path, group_path.parent, checksum=True)["sha256"])
        write_latent_read_index(
            generation,
            obs=obs,
            record={
                "reference": "gene_top",
                "core_start": start,
                "core_end": end,
                "analysis_core_id": core_id,
                "group_sha256": checksum,
                "model_id": f"model-{core_id}",
                "model_checksum": str(owner_index) * 64,
                "obsm_keys": list(result.obsm),
                "varm_keys": list(result.varm),
                "obs_columns": list(result.obs),
            },
            generation_id=generation_id,
            group_path=group_relative.as_posix(),
            reference_uid=reference_identity,
            stage_schema_version=3,
        )

    latent_dir = run_root / "latent_adata_outputs"
    latent_spine = raw_spine.copy()
    latent_spine.uns["latent_read_index"] = (
        f"latent_adata_outputs/generations/{generation_id}/read_index"
    )
    safe_write_h5ad(latent_spine, latent_dir / "spine.h5ad", backup=False, verbose=False)
    write_experiment_spine(run_root)

    project = Path(project) if project is not None else tmp_path / "project"
    reg.init_project(project)
    reg.add_experiment(project, run_root, experiment_id=experiment_id)
    molecule_uids = dict(
        zip(
            raw_spine.obs["read_id"].astype(str),
            raw_spine.obs["molecule_uid"].astype(str),
            strict=True,
        )
    )
    return project, reference_identity, molecule_uids


def test_generic_latent_stage_rejects_and_default_genomic_materialization_is_unchanged(
    tmp_path,
):
    project, reference_identity, _ = _project_with_latent_owners(tmp_path)

    with pytest.raises(ValueError, match="iter_latent_parts.*export-latent"):
        project_adata(project, reference_identity, stage="latent", layers=[])
    output = tmp_path / "generic-latent"
    with pytest.raises(ValueError, match="independent coordinate systems"):
        export_project_partitions(
            project,
            reference_identity,
            output,
            stage="latent",
            layers=[],
        )
    assert not output.exists()

    genomic = project_adata(project, reference_identity, layers=[])
    assert genomic.n_obs == 3
    assert not genomic.obsm
    assert not genomic.varm


def test_scoped_reader_projects_rows_and_fields_before_task_materialization(tmp_path, monkeypatch):
    project, reference_identity, molecule_uids = _project_with_latent_owners(tmp_path)
    opened = []
    materialized = []
    original_read_lazy = ad.experimental.read_lazy
    original_to_memory = ad.AnnData.to_memory

    def tracked_read_lazy(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original_read_lazy(path, *args, **kwargs)

    def tracked_to_memory(self, *args, **kwargs):
        materialized.append(
            {
                "n_obs": self.n_obs,
                "obsm": set(self.obsm),
                "varm": set(self.varm),
                "obs": set(self.obs),
            }
        )
        return original_to_memory(self, *args, **kwargs)

    monkeypatch.setattr(ad.experimental, "read_lazy", tracked_read_lazy)
    monkeypatch.setattr(ad.AnnData, "to_memory", tracked_to_memory)
    catalog = ProjectCatalog.open(project)
    parts = list(
        catalog.iter_latent_parts(
            canonical_reference=reference_identity,
            molecule_uids=[molecule_uids["read-0"]],
            representations=["X_pca_signal"],
            labels=["leiden_signal"],
        )
    )

    assert opened == ["core-a.zarr"]
    assert len(parts) == 1
    part = parts[0]
    assert part.scope.analysis_core_id == "core-a"
    assert part.adata.n_obs == 1
    assert set(part.adata.obsm) == {"X_pca_signal"}
    assert set(part.adata.varm) == {"PCs_signal"}
    assert set(part.adata.obs) == {
        "read_id",
        "experiment_uid",
        "molecule_uid",
        "leiden_signal",
    }
    assert materialized[-1] == {
        "n_obs": 1,
        "obsm": {"X_pca_signal"},
        "varm": {"PCs_signal"},
        "obs": {
            "read_id",
            "experiment_uid",
            "molecule_uid",
            "leiden_signal",
        },
    }


def test_scoped_reader_yields_independent_owners_without_combining_coordinates(tmp_path):
    project, reference_identity, _ = _project_with_latent_owners(tmp_path)

    parts = list(
        ProjectCatalog.open(project).iter_latent_parts(
            canonical_reference=reference_identity,
            representations=["X_pca_signal"],
        )
    )

    assert [part.scope.analysis_core_id for part in parts] == ["core-a", "core-b"]
    assert [part.adata.n_obs for part in parts] == [2, 2]
    assert all(set(part.adata.obsm) == {"X_pca_signal"} for part in parts)
    assert all(
        part.adata.uns["latent_coordinate_scope"]["analysis_core_id"] == part.scope.analysis_core_id
        for part in parts
    )


def test_latent_export_is_transactional_relocatable_and_rejects_existing_target(
    tmp_path, monkeypatch
):
    project, reference_identity, _ = _project_with_latent_owners(tmp_path)
    catalog = ProjectCatalog.open(project)
    output = tmp_path / "latent-export"

    export_latent_parts(
        catalog,
        output,
        canonical_reference=reference_identity,
        representations=["X_pca_signal"],
        labels=["leiden_signal"],
    )

    export_catalog = pd.read_parquet(output / "catalog.parquet")
    assert len(export_catalog) == 2
    assert export_catalog["analysis_core_id"].is_unique
    assert set(export_catalog["representations"].map(tuple)) == {("X_pca_signal",)}
    assert export_catalog["path"].map(lambda value: not Path(value).is_absolute()).all()
    with pytest.raises(FileExistsError, match="already exists"):
        export_latent_parts(catalog, output)

    relocated = tmp_path / "relocated-export"
    shutil.copytree(output, relocated)
    shutil.rmtree(output)
    moved_catalog = pd.read_parquet(relocated / "catalog.parquet")
    for relative in moved_catalog["path"]:
        part, _ = safe_read_zarr(relocated / relative, verbose=False)
        assert set(part.obsm) == {"X_pca_signal"}

    failed = tmp_path / "failed-export"
    original_write = safe_write_zarr
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected latent export failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr("smftools.readwrite.safe_write_zarr", fail_second)
    with pytest.raises(RuntimeError, match="injected latent export failure"):
        export_latent_parts(catalog, failed)
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed-export.*"))


def test_project_export_latent_cli_writes_scoped_artifacts(tmp_path):
    project, reference_identity, _ = _project_with_latent_owners(tmp_path)
    output = tmp_path / "cli-latent-export"

    result = CliRunner().invoke(
        cli_entry.cli,
        [
            "project",
            "export-latent",
            str(project),
            str(output),
            "--canonical-reference",
            reference_identity,
            "--representations",
            "X_pca_signal",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(pd.read_parquet(output / "catalog.parquet")) == 2


def test_legacy_latent_requires_partitioned_migration(tmp_path):
    legacy = ad.AnnData(
        X=np.zeros((1, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"Reference_strand": ["gene_top"]},
            index=["legacy-read"],
        ),
    )
    legacy.uns["modality"] = "direct"
    legacy.uns["experiment"] = "legacy"
    legacy.uns["References"] = {"gene_FASTA_sequence": SEQUENCE}
    legacy_path = tmp_path / "legacy_latent.h5ad"
    safe_write_h5ad(legacy, legacy_path, backup=False, verbose=False)
    project = tmp_path / "legacy-project"
    reg.init_project(project)
    reg.add_experiment(project, legacy_path, experiment_id="legacy", stage="latent")

    with pytest.raises(RuntimeError, match="Re-run latent analysis in partitioned mode"):
        list(ProjectCatalog.open(project).iter_latent_parts())


def test_relocated_multi_experiment_latent_access_keeps_duplicate_read_identity(
    tmp_path,
):
    original = tmp_path / "original"
    project = original / "project"
    _, reference_identity, first_molecules = _project_with_latent_owners(
        original,
        project=project,
        experiment_id="experiment-a",
        run_name="experiment-a",
    )
    _, _, second_molecules = _project_with_latent_owners(
        original,
        project=project,
        experiment_id="experiment-b",
        run_name="experiment-b",
    )
    assert first_molecules["read-1"] != second_molecules["read-1"]

    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    parts = list(
        ProjectCatalog.open(relocated / "project").iter_latent_parts(
            canonical_reference=reference_identity,
            molecule_uids=[
                first_molecules["read-1"],
                second_molecules["read-1"],
            ],
            representations=["X_pca_signal"],
        )
    )

    assert len(parts) == 4
    assert {part.scope.experiment for part in parts} == {
        "experiment-a",
        "experiment-b",
    }
    assert {str(part.adata.obs_names[0]) for part in parts} == {"read-1"}
    assert {str(part.adata.obs["molecule_uid"].iloc[0]) for part in parts} == {
        first_molecules["read-1"],
        second_molecules["read-1"],
    }
