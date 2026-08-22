"""Latent plot artifacts must carry their task's model provenance.

`_validate_latent_generation` requires every plot-catalog row to record the
`model_id`/`model_checksum` of the task that produced it, and rejects the whole
generation otherwise:

    RuntimeError: latent plot model provenance does not match its task

`EGL-28c`/`EGL-28d` registered their clustermaps and barplots without those
fields. Every plot rendered fine and the failure surfaced only at the publish
step, after the stage had done all its work -- the staged generation was then
discarded, costing hours. The unit tests passed throughout because they
exercised rendering, not registration.

These pin the registration call itself.
"""

from __future__ import annotations

import inspect

import pytest

from smftools.tools import latent_clustermaps, partitioned_latent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "function",
    [latent_clustermaps.render_unit_clustermaps, latent_clustermaps.render_unit_composition],
)
def test_render_functions_accept_model_provenance(function):
    parameters = inspect.signature(function).parameters
    assert "model_id" in parameters
    assert "model_checksum" in parameters


@pytest.mark.parametrize(
    "function",
    [latent_clustermaps.render_unit_clustermaps, latent_clustermaps.render_unit_composition],
)
def test_every_register_call_forwards_model_provenance(function):
    """An omitted field here fails the generation, not the plot."""
    source = inspect.getsource(function)
    registrations = source.count("register_plot_artifact(")
    assert registrations >= 1
    assert source.count("model_id=model_id") == registrations
    assert source.count("model_checksum=model_checksum") == registrations


@pytest.mark.parametrize(
    "caller",
    [partitioned_latent._plot_latent_clustermaps, partitioned_latent._plot_latent_composition],
)
def test_stage_callers_supply_provenance_from_the_task_record(caller):
    """The record is the authority; the plot must inherit it, not invent it."""
    source = inspect.getsource(caller)
    assert 'model_id=str(record["model_id"])' in source
    assert 'model_checksum=str(record["model_checksum"])' in source


def test_the_validator_still_demands_provenance():
    """If this check is ever relaxed, the tests above become theatre."""
    source = inspect.getsource(partitioned_latent._validate_latent_generation)
    assert "latent plot model provenance does not match its task" in source
    assert 'record.get("model_id"' in source
