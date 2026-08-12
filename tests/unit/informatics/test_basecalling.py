from types import SimpleNamespace

import pytest

from smftools.informatics import basecalling


def _fake_run(returncode, stdout_bytes=b""):
    """Build a subprocess.run stand-in that writes bytes to the stdout handle."""

    def run(command, stdout=None, **kwargs):
        assert isinstance(command, list)
        if stdout_bytes:
            stdout.write(stdout_bytes)
        return SimpleNamespace(returncode=returncode)

    return run


def _canoncall(tmp_path):
    basecalling.canoncall(
        str(tmp_path / "models"),
        "hac",
        str(tmp_path / "reads.pod5"),
        None,
        str(tmp_path / "basecalls"),
        ".bam",
    )


def _modcall(tmp_path):
    basecalling.modcall(
        str(tmp_path / "models"),
        "hac",
        str(tmp_path / "reads.pod5"),
        None,
        ["5mCG_5hmCG"],
        str(tmp_path / "basecalls"),
        ".bam",
    )


@pytest.mark.parametrize("call", [_canoncall, _modcall])
def test_nonzero_exit_raises_and_leaves_no_output(tmp_path, monkeypatch, call):
    """dorado exits non-zero without writing a BAM; that must not pass silently."""
    monkeypatch.setattr(basecalling.subprocess, "run", _fake_run(1))

    with pytest.raises(RuntimeError, match="exit 1"):
        call(tmp_path)

    assert not (tmp_path / "basecalls.bam").exists()


@pytest.mark.parametrize("call", [_canoncall, _modcall])
def test_empty_output_on_success_raises_and_leaves_no_output(tmp_path, monkeypatch, call):
    """A zero-exit run that produced no basecalls is still a failure."""
    monkeypatch.setattr(basecalling.subprocess, "run", _fake_run(0))

    with pytest.raises(RuntimeError, match="empty BAM"):
        call(tmp_path)

    assert not (tmp_path / "basecalls.bam").exists()


@pytest.mark.parametrize("call", [_canoncall, _modcall])
def test_successful_run_keeps_basecalls(tmp_path, monkeypatch, call):
    monkeypatch.setattr(basecalling.subprocess, "run", _fake_run(0, b"BAM\x01payload"))

    call(tmp_path)

    assert (tmp_path / "basecalls.bam").read_bytes() == b"BAM\x01payload"
