from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smftools.data.localize import (
    LOCALIZED_SUBDIR,
    apply_localize_plan,
    build_localize_plan,
)

pytestmark = pytest.mark.unit


def _config(tmp_path: Path, **fields: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "experiment_config.csv"
    rows = "\n".join(f"{name},{value}" for name, value in fields.items())
    config.write_text(f"variable,value\n{rows}\n", encoding="utf-8")
    return config


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_plan_finds_localizable_fields_and_their_sizes(tmp_path: Path) -> None:
    fasta = _write(tmp_path / "refs" / "ref.fasta", b">ref\nACGT\n")
    bed = _write(tmp_path / "regions" / "align.bed", b"chr1\t0\t10\n")
    config = _config(
        tmp_path,
        output_directory=str(tmp_path / "out"),
        fasta=str(fasta),
        alignment_regions_bed=str(bed),
    )

    plan = build_localize_plan(config)

    fields = {item.field: item for item in plan.items}
    assert set(fields) == {"fasta", "alignment_regions_bed"}
    assert fields["fasta"].size_bytes == fasta.stat().st_size
    assert plan.total_bytes == fasta.stat().st_size + bed.stat().st_size
    assert plan.output_directory == tmp_path / "out"


def test_plan_excludes_raw_input_fields(tmp_path: Path) -> None:
    raw = _write(tmp_path / "pod5" / "signal.pod5", b"x" * 1000)
    config = _config(
        tmp_path,
        output_directory=str(tmp_path / "out"),
        input_data_path=str(raw.parent),
    )

    plan = build_localize_plan(config)

    assert plan.items == ()


def test_plan_requires_output_directory(tmp_path: Path) -> None:
    config = _config(tmp_path, fasta=str(_write(tmp_path / "ref.fasta")))

    with pytest.raises(ValueError, match="output_directory"):
        build_localize_plan(config)


def test_plan_rejects_a_missing_declared_file(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        output_directory=str(tmp_path / "out"),
        fasta=str(tmp_path / "not-there.fasta"),
    )

    with pytest.raises(ValueError, match="does not exist"):
        build_localize_plan(config)


def test_plan_ignores_unset_fields(tmp_path: Path) -> None:
    config = _config(tmp_path, output_directory=str(tmp_path / "out"))

    plan = build_localize_plan(config)

    assert plan.items == ()


def test_apply_copies_files_and_writes_a_new_config(tmp_path: Path) -> None:
    fasta = _write(tmp_path / "refs" / "ref.fasta", b">ref\nACGT\n")
    config = _config(
        tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta), experiment_id="x"
    )
    plan = build_localize_plan(config)
    original_config_bytes = config.read_bytes()

    new_config_path, copied = apply_localize_plan(plan)

    assert [item.field for item in copied] == ["fasta"]
    localized_fasta = tmp_path / "out" / LOCALIZED_SUBDIR / "ref.fasta"
    assert localized_fasta.read_bytes() == fasta.read_bytes()
    assert config.read_bytes() == original_config_bytes  # original untouched
    assert new_config_path == config.with_suffix(".localized.csv")

    new_df = pd.read_csv(new_config_path, dtype=str)
    new_fasta_value = new_df.loc[new_df["variable"] == "fasta", "value"].iloc[0]
    assert new_fasta_value == str(localized_fasta)
    experiment_id_value = new_df.loc[new_df["variable"] == "experiment_id", "value"].iloc[0]
    assert experiment_id_value == "x"  # untouched fields survive the rewrite


def test_apply_honors_an_explicit_out_path(tmp_path: Path) -> None:
    fasta = _write(tmp_path / "ref.fasta")
    config = _config(tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta))
    plan = build_localize_plan(config)
    out_path = tmp_path / "custom_name.csv"

    new_config_path, _ = apply_localize_plan(plan, out_config_path=out_path)

    assert new_config_path == out_path
    assert out_path.is_file()


def test_apply_is_idempotent_on_a_rerun(tmp_path: Path) -> None:
    fasta = _write(tmp_path / "ref.fasta", b">ref\nACGT\n")
    config = _config(tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta))
    plan = build_localize_plan(config)
    apply_localize_plan(plan)

    second_plan = build_localize_plan(config)  # source is unchanged; still finds the field
    _, copied_again = apply_localize_plan(second_plan)

    assert copied_again == []  # already localized with identical content; nothing to copy


def test_apply_refuses_to_overwrite_different_content(tmp_path: Path) -> None:
    fasta = _write(tmp_path / "ref.fasta", b">ref\nACGT\n")
    config = _config(tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta))
    plan = build_localize_plan(config)
    apply_localize_plan(plan)

    # A different file now wants to localize to the same destination name.
    other = _write(tmp_path / "other" / "ref.fasta", b">different\nTTTT\n")
    config2 = _config(
        tmp_path / "other-cfg", output_directory=str(tmp_path / "out"), fasta=str(other)
    )
    plan2 = build_localize_plan(config2)

    with pytest.raises(FileExistsError, match="content different from"):
        apply_localize_plan(plan2)


def test_second_localize_pass_skips_an_already_localized_field(tmp_path: Path) -> None:
    fasta = _write(tmp_path / "ref.fasta", b">ref\nACGT\n")
    config = _config(tmp_path, output_directory=str(tmp_path / "out"), fasta=str(fasta))
    plan = build_localize_plan(config)
    new_config_path, _ = apply_localize_plan(plan)

    # Planning again from the *localized* config should see fasta already in
    # place and skip it, rather than re-copying a copy of itself.
    replan = build_localize_plan(new_config_path)

    assert replan.items == ()
