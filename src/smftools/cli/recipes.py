from __future__ import annotations


def raw_adata(config_path: str):
    from ..cli.raw_adata import raw_adata as _raw_adata

    return _raw_adata(config_path)


def preprocess_adata(config_path: str):
    from ..cli.preprocess_adata import preprocess_adata as _preprocess_adata

    return _preprocess_adata(config_path)


def spatial_adata(config_path: str):
    from ..cli.spatial_adata import spatial_adata as _spatial_adata

    return _spatial_adata(config_path)


def hmm_adata(config_path: str):
    from ..cli.hmm_adata import hmm_adata as _hmm_adata

    return _hmm_adata(config_path)


def full_flow(config_path: str):
    """Run the standard raw-to-HMM workflow with stage-level restart semantics."""
    from pathlib import Path

    from smftools.constants import PARTITIONED_STAGE_REQUIRED_ARTIFACTS

    from .helpers import (
        get_adata_paths,
        load_experiment_config,
        partitioned_stage_is_complete,
        publish_stage_outputs,
        stage_lifecycle,
    )

    cfg = load_experiment_config(config_path)
    with stage_lifecycle(cfg, "full") as lifecycle:
        raw_adata(config_path)
        preprocess_adata(config_path)
        spatial_adata(config_path)
        result = hmm_adata(config_path)

        outputs = {}
        required = ()
        paths = get_adata_paths(cfg)
        result_path = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if result_path is not None and paths.hmm_spine is not None:
            result_path = Path(result_path)
            if result_path == Path(paths.hmm_spine):
                incomplete = [
                    stage
                    for stage, stage_required in PARTITIONED_STAGE_REQUIRED_ARTIFACTS.items()
                    if not partitioned_stage_is_complete(
                        cfg,
                        stage,
                        required=stage_required,
                    )
                ]
                if incomplete:
                    raise RuntimeError(
                        "full workflow cannot publish completion; incomplete stage record(s): "
                        f"{incomplete}"
                    )
                outputs = {"hmm_spine": result_path}
                required = ("hmm_spine",)
        publish_stage_outputs(
            lifecycle,
            outputs,
            required=required,
            task_catalog_key=None,
            checksum_keys=(),
            schema_versions={"full_workflow": 1},
            task_count=4,
        )
    return result
