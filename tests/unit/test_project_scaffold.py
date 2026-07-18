import yaml

from smftools.project.scaffold import scaffold_project

EXPECTED_FILES = {"README.md", "AGENTS.md", "CLAUDE.md", "PLAN.md", "project.yaml"}
EXPECTED_DIRS = {"project_scripts", "project_outputs"}


def test_scaffold_project_creates_all_expected_files_and_dirs(tmp_path):
    proj = tmp_path / "my_project"
    written = scaffold_project(proj)

    for filename in EXPECTED_FILES:
        assert (proj / filename).is_file()
    for dirname in EXPECTED_DIRS:
        assert (proj / dirname).is_dir()
    assert (proj / "project_scripts" / "__init__.py").is_file()

    # Every created path is reported back.
    created_names = {p.relative_to(proj).as_posix() for p in written}
    assert created_names == EXPECTED_FILES | EXPECTED_DIRS | {"project_scripts/__init__.py"}


def test_scaffold_project_never_overwrites_existing_files(tmp_path):
    proj = tmp_path / "my_project"
    scaffold_project(proj)
    readme = proj / "README.md"
    readme.write_text("# My custom notes\n")

    written = scaffold_project(proj)

    assert readme.read_text() == "# My custom notes\n"
    assert readme not in written


def test_scaffold_project_is_idempotent_second_call_creates_nothing(tmp_path):
    proj = tmp_path / "my_project"
    scaffold_project(proj)
    second = scaffold_project(proj)
    assert second == []


def test_scaffold_project_default_name_from_directory(tmp_path):
    proj = tmp_path / "cool_project_name"
    scaffold_project(proj)
    assert "# cool_project_name" in (proj / "README.md").read_text()
    manifest = yaml.safe_load((proj / "project.yaml").read_text())
    assert manifest["name"] == "cool_project_name"


def test_scaffold_project_uses_explicit_name(tmp_path):
    proj = tmp_path / "dirname_not_used"
    scaffold_project(proj, name="Friendly Name")
    assert "# Friendly Name" in (proj / "README.md").read_text()
    manifest = yaml.safe_load((proj / "project.yaml").read_text())
    assert manifest["name"] == "Friendly Name"


def test_project_yaml_is_valid_and_has_expected_shape(tmp_path):
    proj = tmp_path / "my_project"
    scaffold_project(proj)
    manifest = yaml.safe_load((proj / "project.yaml").read_text())
    assert set(manifest) == {"name", "description", "references", "runs"}
    assert manifest["references"] == []
    assert manifest["runs"] == []


def test_claude_md_points_at_project_local_agents_md(tmp_path):
    proj = tmp_path / "my_project"
    scaffold_project(proj)
    assert "AGENTS.md" in (proj / "CLAUDE.md").read_text()
