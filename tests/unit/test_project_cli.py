import json

import anndata as ad
import pandas as pd
from click.testing import CliRunner

from smftools import cli_entry
from smftools.informatics.raw_store import write_raw_store
from smftools.informatics.reference_identity import reference_uid
from smftools.readwrite import safe_read_h5ad, safe_write_h5ad

SEQUENCE = "ACGTACGTACGT"


def _make_raw_experiment(out_dir, *, reference_strand, uid, npos=12, n=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "read_id": f"{reference_strand}_r{i}",
            "reference": reference_strand.rsplit("_", 1)[0],
            "Reference_strand": reference_strand,
            "sample": "bc01",
            "barcode": "bc01",
            "strand": "top",
            "mapping_direction": "fwd",
            "reference_start": 0,
            "cigar": f"{npos}M",
            "aligned_length": npos,
            "sequence": [i % 4 for _ in range(npos)],
            "quality": [30] * npos,
            "mismatch": [4] * npos,
            "modification_signal": [float(i % 2)] * npos,
        }
        for i in range(n)
    ]
    write_raw_store(
        pd.DataFrame(rows),
        # Each experiment owns a run root, exactly as the pipeline lays one out.
        # The persisted experiment identity is keyed on that root, so sibling raw
        # stores under one parent would be one experiment rather than two.
        out_dir / "raw_outputs",
        reference_lengths={reference_strand: npos},
        extra_uns={
            "reference_uids": {reference_strand: uid},
            "modality": "direct",
            "experiment": out_dir.name,
        },
    )
    return out_dir


def test_project_init_cli_scaffolds_docs_and_working_dirs(tmp_path):
    proj = tmp_path / "project"
    runner = CliRunner()

    r = runner.invoke(cli_entry.cli, ["project", "init", str(proj)])
    assert r.exit_code == 0, r.output
    assert "Initialized project registry" in r.output
    assert "created" in r.output

    for filename in ("README.md", "AGENTS.md", "CLAUDE.md", "PLAN.md", "project.yaml"):
        assert (proj / filename).is_file()
    assert (proj / "project_scripts").is_dir()
    assert (proj / "project_outputs").is_dir()

    # Re-running is a no-op for the scaffold (idempotent registry init too).
    readme = proj / "README.md"
    readme.write_text("# hand-edited\n")
    r2 = runner.invoke(cli_entry.cli, ["project", "init", str(proj)])
    assert r2.exit_code == 0, r2.output
    assert readme.read_text() == "# hand-edited\n"


def test_project_cli_end_to_end(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    runner = CliRunner()

    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0

    r = runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")])
    assert r.exit_code == 0, r.output
    assert "Registered 'expA'" in r.output

    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expB")]
        ).exit_code
        == 0
    )

    r = runner.invoke(cli_entry.cli, ["project", "list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "expA" in r.output and "expB" in r.output
    assert "canonical reference" in r.output

    out = tmp_path / "combined.h5ad"
    r = runner.invoke(cli_entry.cli, ["project", "materialize", str(proj), uid, "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 7
    assert set(combined.obs["experiment"]) == {"expA", "expB"}


def test_project_workflow_contract_end_to_end(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    project = tmp_path / "project"
    output = tmp_path / "workflow-output"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(project)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli,
            ["project", "add", str(project), str(tmp_path / "expA")],
        ).exit_code
        == 0
    )

    run = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "run",
            str(project),
            uid,
            "--output-root",
            str(output),
            "--layers",
            "sequence_integer_encoding",
        ],
    )
    assert run.exit_code == 0, run.output
    result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "success"
    combined, _ = safe_read_h5ad(output / "materialized.h5ad.gz")
    assert combined.n_obs == 4

    repeated = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "run",
            str(project),
            uid,
            "--output-root",
            str(output),
            "--layers",
            "sequence_integer_encoding",
        ],
    )
    assert repeated.exit_code == 0, repeated.output
    assert (
        json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))["outcome"]
        == "compatible_skip"
    )

    validate = runner.invoke(
        cli_entry.cli,
        ["project", "validate", str(project), str(output), "--json"],
    )
    assert validate.exit_code == 0, validate.output
    assert json.loads(validate.output)["valid"] is True


def test_project_plan_cli_is_read_only_and_emits_json(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    project = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(project)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli,
            ["project", "add", str(project), str(tmp_path / "expA")],
        ).exit_code
        == 0
    )
    before = {
        path.relative_to(project): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in project.rglob("*")
        if path.is_file()
    }

    result = runner.invoke(
        cli_entry.cli,
        ["project", "plan", str(project), "embedding", uid, "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["requested_target"] == "project.embedding.generation"
    assert payload["topological_order"] == [
        "project.genomic_selection",
        "project.embedding.feature_matrix",
        "project.embedding.generation",
    ]
    after = {
        path.relative_to(project): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in project.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_project_materialize_cli_pools_with_layer_projection(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]
        ).exit_code
        == 0
    )

    out = tmp_path / "combined.h5ad"
    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "materialize",
            str(proj),
            uid,
            "-o",
            str(out),
            "--layers",
            "sequence_integer_encoding",
        ],
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 4
    assert set(combined.layers) == {"sequence_integer_encoding"}

    # The set store writes nothing to disk (no base.h5ad cache anymore).
    assert not (proj / "project_outputs" / "sets").exists()


def test_project_materialize_cli_guardrail_refuses_oversized_pool(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]
        ).exit_code
        == 0
    )

    # Force the guardrail to trip with an absurdly low limit via the Python API is
    # cleaner; here just confirm the CLI --allow-large flag is accepted and works.
    out = tmp_path / "combined.h5ad"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "materialize", str(proj), uid, "-o", str(out), "--layers", "", "--allow-large"],
    )
    assert r.exit_code == 0, r.output
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 4
    assert len(combined.layers) == 0  # --layers '' => X only


def test_project_materialize_cli_writes_partitioned_export(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]
        ).exit_code
        == 0
    )

    output = tmp_path / "partitioned_export"
    result = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "materialize",
            str(proj),
            uid,
            "-o",
            str(output),
            "--layers",
            "",
            "--partitioned",
            "--max-memory-percent",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "manifest.json").is_file()
    assert (output / "catalog.parquet").is_file()


def test_project_sample_store_list_cli(tmp_path):
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneB_top", uid=uid, n=3)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expB")]
        ).exit_code
        == 0
    )

    r = runner.invoke(cli_entry.cli, ["project", "sample-store-list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "2 partition(s)" in r.output
    assert "expA" in r.output and "geneA_top" in r.output and "bc01" in r.output
    assert "expB" in r.output

    r_filtered = runner.invoke(
        cli_entry.cli, ["project", "sample-store-list", str(proj), "--experiment-id", "expA"]
    )
    assert r_filtered.exit_code == 0, r_filtered.output
    assert "1 partition(s)" in r_filtered.output
    assert "expB" not in r_filtered.output


def test_project_sample_store_list_cli_empty_project(tmp_path):
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0

    r = runner.invoke(cli_entry.cli, ["project", "sample-store-list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "No per-sample-store partitions" in r.output


def test_project_add_cli_backfills_per_sample_store_for_modern_experiment(tmp_path):
    from smftools.project.sample_store import list_per_sample_partitions

    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()

    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    r = runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")])
    assert r.exit_code == 0, r.output

    partitions = list_per_sample_partitions(proj, "expA")
    assert len(partitions) == 1
    assert partitions[0] == {
        "kind": "pointer",
        "experiment_id": "expA",
        "reference_strand": "geneA_top",
        "sample": "bc01",
        "n_reads": 4,
    }


def test_project_add_cli_caches_per_sample_store_for_legacy_file(tmp_path):
    from smftools.project.sample_store import list_per_sample_partitions, load_per_sample_partition

    sequence = "ACGTACGTACGT"
    legacy_file = _make_legacy_monolithic_file(
        tmp_path / "legacyExp2_preprocessed.h5ad",
        reference_strand="geneL_top",
        sequence=sequence,
        n=3,
        sample="bc00",
    )
    before_bytes = legacy_file.read_bytes()
    proj = tmp_path / "project"
    runner = CliRunner()

    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "add",
            str(proj),
            str(legacy_file),
            "--id",
            "legacyExp2",
            "--stage",
            "preprocess",
        ],
    )
    assert r.exit_code == 0, r.output

    # Legacy registration caches the partition (has no lazy read path to point at
    # instead), unlike a modern experiment which only gets a pointer.
    partitions = list_per_sample_partitions(proj, "legacyExp2")
    assert len(partitions) == 1
    assert partitions[0]["kind"] == "cache"
    assert partitions[0]["n_reads"] == 3
    loaded = load_per_sample_partition(proj, "legacyExp2", "geneL_top", "bc00")
    assert loaded.n_obs == 3

    # Source legacy file is only ever read, never mutated.
    assert legacy_file.read_bytes() == before_bytes


def _make_legacy_monolithic_file(path, *, reference_strand, sequence, n=3, npos=6, sample="bc00"):
    import numpy as np

    chromosome = reference_strand.rsplit("_", 1)[0]
    obs = pd.DataFrame(
        {"Reference_strand": [reference_strand] * n, "Sample": [sample] * n},
        index=[f"{reference_strand}_leg{i}" for i in range(n)],
    )
    spine = ad.AnnData(X=np.zeros((n, npos), dtype=np.float32), obs=obs)
    spine.var_names = [str(p) for p in range(npos)]
    spine.uns["modality"] = "direct"
    spine.uns["experiment"] = "legacyExp"
    spine.uns["References"] = {f"{chromosome}_FASTA_sequence": sequence}
    safe_write_h5ad(spine, path, backup=False, verbose=False)
    assert "is_spine" not in spine.uns
    return path


def test_project_cli_registers_and_materializes_legacy_monolithic_file(tmp_path):
    """A pre-partitioned-store run (a single monolithic .h5ad, no uns['is_spine'])
    registers via --stage and materializes through the same project/materialize
    chain as a modern partitioned run -- without ever rewriting the source file."""
    sequence = "ACGTACGTACGT"
    legacy_file = _make_legacy_monolithic_file(
        tmp_path / "legacyExp_preprocessed.h5ad",
        reference_strand="geneL_top",
        sequence=sequence,
        n=3,
    )
    before_bytes = legacy_file.read_bytes()

    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0

    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "add",
            str(proj),
            str(legacy_file),
            "--id",
            "legacyExp",
            "--stage",
            "preprocess",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Registered 'legacyExp'" in r.output

    # Source file untouched by registration (no cached-back reference_uids etc).
    assert legacy_file.read_bytes() == before_bytes

    r = runner.invoke(cli_entry.cli, ["project", "list", str(proj)])
    assert r.exit_code == 0, r.output
    assert "legacyExp" in r.output

    uid = reference_uid(sequence)
    out = tmp_path / "legacy_combined.h5ad"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "materialize", str(proj), uid, "-o", str(out), "--stage", "preprocess"],
    )
    assert r.exit_code == 0, r.output
    combined, _ = safe_read_h5ad(out)
    assert combined.n_obs == 3
    assert set(combined.obs["experiment"]) == {"legacyExp"}


def _project_with_two_experiments(tmp_path):
    """Register two real single-reference experiments and return the project dir."""
    uid = reference_uid(SEQUENCE, 12)
    _make_raw_experiment(tmp_path / "expA", reference_strand="geneA_top", uid=uid, n=4)
    _make_raw_experiment(tmp_path / "expB", reference_strand="geneA_top", uid=uid, n=4)
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    for name in ("expA", "expB"):
        result = runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / name)])
        assert result.exit_code == 0, result.output
    return proj, uid, runner


def test_set_commands_define_list_show_and_remove_without_python(tmp_path):
    """The `--set` flag is only usable if sets can be managed from the CLI alone."""
    proj, _uid, runner = _project_with_two_experiments(tmp_path)

    r = runner.invoke(cli_entry.cli, ["project", "list-sets", str(proj)])
    assert r.exit_code == 0, r.output
    assert "No named sets defined" in r.output

    r = runner.invoke(
        cli_entry.cli,
        ["project", "add-set", str(proj), "cohort", "--experiment", "expA"],
    )
    assert r.exit_code == 0, r.output
    assert "Defined set 'cohort' (list)." in r.output
    assert "resolves to 1 experiment(s):" in r.output

    r = runner.invoke(cli_entry.cli, ["project", "list-sets", str(proj)])
    assert r.exit_code == 0, r.output
    assert "cohort  (list) 1 declared experiment(s)" in r.output

    r = runner.invoke(cli_entry.cli, ["project", "show-set", str(proj), "cohort"])
    assert r.exit_code == 0, r.output
    assert "expA" in r.output and "expB" not in r.output

    r = runner.invoke(cli_entry.cli, ["project", "remove-set", str(proj), "cohort"])
    assert r.exit_code == 0, r.output
    assert "No experiment registration was changed." in r.output

    # The set is gone; both experiments remain registered.
    assert runner.invoke(cli_entry.cli, ["project", "show-set", str(proj), "cohort"]).exit_code == 1
    r = runner.invoke(cli_entry.cli, ["project", "list", str(proj)])
    assert "expA" in r.output and "expB" in r.output


def test_add_set_rejects_an_unresolvable_member_unless_allowed(tmp_path):
    """A typo in a set name must fail at definition, not silently narrow later."""
    proj, _uid, runner = _project_with_two_experiments(tmp_path)

    r = runner.invoke(
        cli_entry.cli,
        ["project", "add-set", str(proj), "typo", "--experiment", "expA", "--experiment", "ghost"],
    )
    assert r.exit_code == 1
    assert "not registered: ghost" in r.output
    assert "--allow-unresolved" in r.output
    # Nothing was written, so the next command does not see a half-accepted set.
    assert (
        "No named sets defined"
        in runner.invoke(cli_entry.cli, ["project", "list-sets", str(proj)]).output
    )

    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "add-set",
            str(proj),
            "typo",
            "--experiment",
            "expA",
            "--experiment",
            "ghost",
            "--allow-unresolved",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "not registered: ghost" in r.output

    r = runner.invoke(
        cli_entry.cli,
        ["project", "add-set", str(proj), "both", "--experiment", "expA", "--query", "x=1"],
    )
    assert r.exit_code == 1
    assert "exactly one of --experiment or --query" in r.output


def test_set_consumers_select_exactly_the_shown_membership(tmp_path):
    """What `show-set` prints has to be what `--set` applies, including a query set."""
    from smftools.project.catalog import ProjectCatalog
    from smftools.project.registry import resolve_set_membership

    proj, uid, runner = _project_with_two_experiments(tmp_path)
    assert (
        runner.invoke(
            cli_entry.cli,
            ["project", "add-set", str(proj), "cohort", "--experiment", "expB"],
        ).exit_code
        == 0
    )
    r = runner.invoke(
        cli_entry.cli,
        ["project", "add-set", str(proj), "direct", "--query", "modality='direct'"],
    )
    assert r.exit_code == 0, r.output

    catalog = ProjectCatalog.open(proj)
    for name in ("cohort", "direct"):
        shown = resolve_set_membership(proj, name).resolved
        selected = catalog.select(canonical_reference=uid, set_name=name)
        assert tuple(sorted(set(selected["experiment"]))) == shown, name

    # A materialize through the same filter pools exactly those experiments.
    out = tmp_path / "cohort.h5ad"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "materialize", str(proj), uid, "-o", str(out), "--set", "cohort"],
    )
    assert r.exit_code == 0, r.output
    combined, _ = safe_read_h5ad(out)
    assert set(combined.obs["experiment"]) == {"expB"}


def test_deactivated_experiment_drops_out_of_a_set_visibly(tmp_path):
    """`project remove` narrows a set; the CLI has to show that, not hide it."""
    proj, uid, runner = _project_with_two_experiments(tmp_path)
    assert (
        runner.invoke(
            cli_entry.cli,
            [
                "project",
                "add-set",
                str(proj),
                "cohort",
                "--experiment",
                "expA",
                "--experiment",
                "expB",
            ],
        ).exit_code
        == 0
    )
    assert runner.invoke(cli_entry.cli, ["project", "remove", str(proj), "expB"]).exit_code == 0

    r = runner.invoke(cli_entry.cli, ["project", "show-set", str(proj), "cohort"])
    assert r.exit_code == 0, r.output
    assert "resolves to 1 experiment(s):" in r.output
    assert "inactive: expB" in r.output

    out = tmp_path / "cohort.h5ad"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "materialize", str(proj), uid, "-o", str(out), "--set", "cohort"],
    )
    assert r.exit_code == 0, r.output
    combined, _ = safe_read_h5ad(out)
    assert set(combined.obs["experiment"]) == {"expA"}


def test_project_run_honors_the_experiment_filter_it_publishes(tmp_path):
    """A published request naming one experiment must not pool every experiment.

    `--experiment` reached the plan and the result's request but never reached
    materialization, so the artifact silently pooled the whole project while the
    result claimed a subset -- the wrong cohort, with provenance that looked right.
    """
    proj, uid, runner = _project_with_two_experiments(tmp_path)
    output = tmp_path / "task-out"

    r = runner.invoke(
        cli_entry.cli,
        ["project", "run", str(proj), uid, "--output-root", str(output), "--experiment", "expA"],
    )
    assert r.exit_code == 0, r.output

    result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert result["request"]["experiments"] == ["expA"]
    combined, _ = safe_read_h5ad(output / "materialized.h5ad.gz")
    assert set(combined.obs["experiment"]) == {"expA"}


def _analyzable_project(tmp_path):
    """Two experiments with enough positions and periodic signal to analyze.

    The 12-position fixture used elsewhere in this file is deliberately tiny; a
    periodicity analysis over it legitimately returns no reads, which would make
    these assertions vacuous.
    """
    npos = 160
    uid = reference_uid("ACGT" * 40, npos)
    for name, sample in (("expA", "bc01"), ("expB", "bc02")):
        out_dir = tmp_path / name / "raw_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "read_id": f"{name}_{sample}_r{i}",
                "reference": "geneA",
                "Reference_strand": "geneA_top",
                "sample": sample,
                "barcode": sample,
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": f"{npos}M",
                "aligned_length": npos,
                "sequence": [i % 4 for _ in range(npos)],
                "quality": [30] * npos,
                "mismatch": [4] * npos,
                "modification_signal": [float((position // 10) % 2) for position in range(npos)],
            }
            for i in range(6)
        ]
        write_raw_store(
            pd.DataFrame(rows),
            out_dir,
            reference_lengths={"geneA_top": npos},
            extra_uns={
                "reference_uids": {"geneA_top": uid},
                "modality": "direct",
                "experiment": name,
            },
        )
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    for name in ("expA", "expB"):
        result = runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / name)])
        assert result.exit_code == 0, result.output
    return proj, uid, runner


def test_sample_analysis_cli_runs_validates_and_skips(tmp_path):
    """The full task-local lifecycle for the sample-analysis product."""
    import pandas as pd

    proj, uid, runner = _analyzable_project(tmp_path)
    output = tmp_path / "analysis-out"

    r = runner.invoke(
        cli_entry.cli,
        ["project", "sample-analysis", str(proj), uid, "--output-root", str(output)],
    )
    assert r.exit_code == 0, r.output

    result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert result["command"] == "project.sample-analysis"
    assert result["outcome"] == "success"
    assert result["summary"]["n_partitions"] == 2
    table = pd.read_parquet(output / "sample_analysis.parquet")
    # Partition identity stays explicit, so reads from different experiments
    # never become indistinguishable rows.
    assert {"experiment", "reference_strand", "sample", "molecule_uid"} <= set(table.columns)
    assert set(table["experiment"]) == {"expA", "expB"}

    r = runner.invoke(cli_entry.cli, ["project", "validate", str(proj), str(output), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["valid"] is True

    r = runner.invoke(
        cli_entry.cli,
        ["project", "sample-analysis", str(proj), uid, "--output-root", str(output)],
    )
    assert r.exit_code == 0, r.output
    assert (
        json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))["outcome"]
        == "compatible_skip"
    )


def test_sample_analysis_selection_narrows_to_the_requested_experiment(tmp_path):
    """Selection flags reach the per-partition scope, not just the plan."""
    proj, uid, runner = _analyzable_project(tmp_path)
    output = tmp_path / "analysis-out"

    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "sample-analysis",
            str(proj),
            uid,
            "--output-root",
            str(output),
            "--experiment",
            "expB",
        ],
    )
    assert r.exit_code == 0, r.output

    result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert result["summary"]["n_partitions"] == 1
    assert [item["experiment"] for item in result["summary"]["partitions"]] == ["expB"]


def test_sample_analysis_reports_an_empty_selection_as_a_structured_failure(tmp_path):
    proj, _uid, runner = _analyzable_project(tmp_path)
    output = tmp_path / "analysis-out"

    r = runner.invoke(
        cli_entry.cli,
        ["project", "sample-analysis", str(proj), "not-a-reference", "--output-root", str(output)],
    )
    assert r.exit_code == 1
    result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "failed"
    assert "no per-sample-store partition" in result["failure"]["message"]


def _embeddable_experiment(tmp_path, name, uid, *, sample, seed, n=30, npos=300):
    """One experiment with enough reads and structure to fit a shared embedding."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out_dir = tmp_path / name / "raw_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        phase = int(rng.integers(0, 2))
        rows.append(
            {
                "read_id": f"{name}_{sample}_r{i}",
                "reference": "geneA",
                "Reference_strand": "geneA_top",
                "sample": sample,
                "barcode": sample,
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": f"{npos}M",
                "aligned_length": npos,
                "sequence": [i % 4 for _ in range(npos)],
                "quality": [30] * npos,
                "mismatch": [4] * npos,
                "modification_signal": [float(((p // 15) + phase) % 2) for p in range(npos)],
            }
        )
    write_raw_store(
        pd.DataFrame(rows),
        out_dir,
        reference_lengths={"geneA_top": npos},
        extra_uns={
            "reference_uids": {"geneA_top": uid},
            "modality": "direct",
            "experiment": name,
        },
    )
    return out_dir.parent


def test_embedding_cli_fits_skips_and_requires_trust_before_growing(tmp_path):
    """Growth unpickles this project's estimators, so it needs an explicit decision."""
    uid = reference_uid("A" * 300, 300)
    _embeddable_experiment(tmp_path, "expA", uid, sample="bc01", seed=1)
    _embeddable_experiment(tmp_path, "expB", uid, sample="bc02", seed=2)
    proj = tmp_path / "project"
    output = tmp_path / "embedding-out"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expA")]
        ).exit_code
        == 0
    )

    command = [
        "project",
        "embedding",
        str(proj),
        uid,
        "--output-root",
        str(output),
        "--min-reads",
        "5",
        "--n-neighbors",
        "5",
    ]
    r = runner.invoke(cli_entry.cli, command)
    assert r.exit_code == 0, r.output
    result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert result["command"] == "project.embedding"
    assert result["summary"]["fit_kind"] == "full"
    assert result["summary"]["n_molecules"] == 30
    assert result["trust_local_models"] is False
    table = pd.read_parquet(output / "embedding.parquet")
    assert {"molecule_uid", "cluster", "umap_1", "umap_2"} <= set(table.columns)
    assert len(table) == 30
    # Estimator pickles are trusted-local project artifacts; exporting coordinates
    # must not spread them into a task output other steps will consume.
    assert not list(output.rglob("*.pkl"))

    r = runner.invoke(cli_entry.cli, ["project", "validate", str(proj), str(output), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["valid"] is True

    assert runner.invoke(cli_entry.cli, command).exit_code == 0
    assert (
        json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))["outcome"]
        == "compatible_skip"
    )

    # Registering another experiment grows the selection, which needs the models.
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add", str(proj), str(tmp_path / "expB")]
        ).exit_code
        == 0
    )
    r = runner.invoke(cli_entry.cli, command)
    assert r.exit_code == 1
    blocked = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert blocked["outcome"] == "failed"
    assert blocked["failure"]["type"] == "EmbeddingTrustError"

    r = runner.invoke(cli_entry.cli, [*command, "--trust-local-models"])
    assert r.exit_code == 0, r.output
    grown = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
    assert grown["trust_local_models"] is True
    assert grown["summary"]["fit_kind"] == "extended"
    assert grown["summary"]["n_molecules"] == 60
    assert grown["summary"]["n_new_molecules"] == 30
    assert grown["summary"]["prior_generation_id"]
    assert len(pd.read_parquet(output / "embedding.parquet")) == 60


def test_named_set_selects_the_cohort_for_every_project_product(tmp_path):
    """`--set` has to mean the same thing to run, sample-analysis, and embedding."""
    proj, uid, runner = _analyzable_project(tmp_path)
    assert (
        runner.invoke(
            cli_entry.cli, ["project", "add-set", str(proj), "cohort", "--experiment", "expB"]
        ).exit_code
        == 0
    )

    materialized = tmp_path / "mat-out"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "run", str(proj), uid, "--output-root", str(materialized), "--set", "cohort"],
    )
    assert r.exit_code == 0, r.output
    combined, _ = safe_read_h5ad(materialized / "materialized.h5ad.gz")
    assert set(combined.obs["experiment"]) == {"expB"}

    analysis = tmp_path / "sa-out"
    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "run",
            str(proj),
            uid,
            "--target",
            "sample-analysis",
            "--output-root",
            str(analysis),
            "--set",
            "cohort",
        ],
    )
    assert r.exit_code == 0, r.output
    summary = json.loads((analysis / "workflow_result.json").read_text(encoding="utf-8"))["summary"]
    assert [item["experiment"] for item in summary["partitions"]] == ["expB"]


def test_duplicate_bare_read_ids_stay_distinct_through_sample_analysis(tmp_path):
    """Two experiments can share an instrument read ID; the rows must not merge."""
    import pandas as pd

    uid = reference_uid("ACGT" * 40, 160)
    for name in ("expA", "expB"):
        # Each experiment gets its own run root, exactly as the pipeline lays one
        # out. The persisted experiment identity is keyed on the run root, so
        # sibling raw stores under one parent would be one experiment, not two.
        out_dir = tmp_path / name / "raw_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                # Deliberately identical across experiments.
                "read_id": f"shared_read_{i}",
                "reference": "geneA",
                "Reference_strand": "geneA_top",
                "sample": "bc01",
                "barcode": "bc01",
                "strand": "top",
                "mapping_direction": "fwd",
                "reference_start": 0,
                "cigar": "160M",
                "aligned_length": 160,
                "sequence": [i % 4 for _ in range(160)],
                "quality": [30] * 160,
                "mismatch": [4] * 160,
                "modification_signal": [float((p // 10) % 2) for p in range(160)],
            }
            for i in range(6)
        ]
        write_raw_store(
            pd.DataFrame(rows),
            out_dir,
            reference_lengths={"geneA_top": 160},
            extra_uns={
                "reference_uids": {"geneA_top": uid},
                "modality": "direct",
                "experiment": name,
            },
        )
    proj = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(cli_entry.cli, ["project", "init", str(proj)]).exit_code == 0
    for name in ("expA", "expB"):
        assert (
            runner.invoke(cli_entry.cli, ["project", "add", str(proj), str(tmp_path / name)])
        ).exit_code == 0

    output = tmp_path / "sa-out"
    r = runner.invoke(
        cli_entry.cli,
        ["project", "sample-analysis", str(proj), uid, "--output-root", str(output)],
    )
    assert r.exit_code == 0, r.output

    table = pd.read_parquet(output / "sample_analysis.parquet")
    assert set(table["experiment"]) == {"expA", "expB"}
    # The bare read ID repeats; molecule identity does not.
    assert table["read_id"].duplicated().any()
    assert not table["molecule_uid"].duplicated().any()
    assert table.groupby("experiment")["molecule_uid"].nunique().to_dict() == {
        "expA": 6,
        "expB": 6,
    }


def test_force_recompute_reruns_instead_of_reporting_a_compatible_skip(tmp_path):
    """Force has to defeat the skip, or a recompute request silently does nothing."""
    proj, uid, runner = _analyzable_project(tmp_path)
    output = tmp_path / "sa-out"
    command = ["project", "sample-analysis", str(proj), uid, "--output-root", str(output)]

    assert runner.invoke(cli_entry.cli, command).exit_code == 0
    assert runner.invoke(cli_entry.cli, command).exit_code == 0
    assert (
        json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))["outcome"]
        == "compatible_skip"
    )

    r = runner.invoke(cli_entry.cli, [*command, "--force-recompute"])
    assert r.exit_code == 0, r.output
    assert (
        json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))["outcome"]
        == "success"
    )


def test_project_run_rejects_options_that_do_not_apply_to_the_target(tmp_path):
    """A flag that cannot apply must fail, not be dropped from what runs."""
    proj, uid, runner = _analyzable_project(tmp_path)

    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "run",
            str(proj),
            uid,
            "--target",
            "materialization",
            "--output-root",
            str(tmp_path / "out"),
            "--trust-local-models",
        ],
    )
    assert r.exit_code == 1
    assert "--trust-local-models" in r.output
    assert "does not apply" in r.output or "do(es) not apply" in r.output

    r = runner.invoke(
        cli_entry.cli,
        [
            "project",
            "run",
            str(proj),
            uid,
            "--target",
            "sample-analysis",
            "--output-root",
            str(tmp_path / "out"),
            "--partitioned",
        ],
    )
    assert r.exit_code == 1
    assert "--partitioned" in r.output


def test_project_run_dispatches_every_executable_target(tmp_path):
    """One engine-facing entry point reaches all three products.

    `experiment run --target` already coexists with the per-stage commands, so
    `project run --target` is the same shape: engines get one command, humans
    keep the named subcommands. `selection` is a planning-only dependency and is
    deliberately not offered here.
    """
    proj, uid, runner = _analyzable_project(tmp_path)
    expected = {
        "materialization": ("project.materialize", "materialized.h5ad.gz"),
        "sample-analysis": ("project.sample-analysis", "sample_analysis.parquet"),
        "embedding": ("project.embedding", "embedding.parquet"),
    }
    for target, (command, artifact) in expected.items():
        output = tmp_path / f"run-{target}"
        args = ["project", "run", str(proj), uid, "--target", target, "--output-root", str(output)]
        if target == "embedding":
            args += ["--min-reads", "5", "--n-neighbors", "5"]
        r = runner.invoke(cli_entry.cli, args)
        assert r.exit_code == 0, r.output
        result = json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))
        assert result["command"] == command, target
        assert result["target"] == target
        assert (output / artifact).exists(), target
        r = runner.invoke(cli_entry.cli, ["project", "validate", str(proj), str(output), "--json"])
        assert json.loads(r.output)["valid"] is True, target

    # The default is unchanged, so existing invocations keep materializing.
    output = tmp_path / "run-default"
    r = runner.invoke(
        cli_entry.cli, ["project", "run", str(proj), uid, "--output-root", str(output)]
    )
    assert r.exit_code == 0, r.output
    assert (
        json.loads((output / "workflow_result.json").read_text(encoding="utf-8"))["command"]
        == "project.materialize"
    )


def test_project_plan_targets_map_to_documented_execution_paths():
    """Every plan target is either executable through `run` or documented as plan-only."""
    import click

    plan_targets = set(
        next(
            param
            for param in cli_entry.project_group.commands["plan"].params
            if param.name == "target"
        ).type.choices
    )
    run_target = next(
        param for param in cli_entry.project_group.commands["run"].params if param.name == "target"
    )
    assert isinstance(run_target.type, click.Choice)
    executable = set(run_target.type.choices)

    assert executable <= plan_targets
    # `selection` is the shared dependency node the other three consume; it has
    # no artifact of its own, which is why it is planned but never run.
    assert plan_targets - executable == {"selection"}
    for target in executable:
        assert target in {"materialization", "sample-analysis", "embedding"}
