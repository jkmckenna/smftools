"""`data localize`: copy an experiment's small, hand-edited inputs (`PSR-13`).

The cheapest adoption win in the portable-storage-roots plan: copying `fasta`,
the BED region files, the sample sheet, and any barcode/UMI YAML into the
run's own output directory makes the whole `analyses/` tree self-contained --
no named root (`PSR-04`-`PSR-07`), no volume stamp (`PSR-08`), no replica
catalog (`PSR-10`) required to read it on another machine.

Deliberately excluded: `input_data_path`/`input_manifest_path` (the large raw
data this entire plan exists to leave archived, not duplicate),
`sequencing_summary_path` and `model_dir` (can themselves be large), and the
deprecated `fasta_regions_of_interest`. See `USER_SUPPLIED_PATH_FIELDS` in
`smftools.config.experiment_config` for the full set this is a subset of.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config.roots import resolve_config_path

#: Fields worth localizing -- small, hand-edited, and not already covered by
#: the run's own generation manifests.
LOCALIZABLE_FIELDS = (
    "fasta",
    "alignment_regions_bed",
    "analysis_regions_bed",
    "plot_regions_bed",
    "sample_sheet_path",
    "custom_barcode_yaml",
    "umi_yaml",
)

LOCALIZED_SUBDIR = "localized_inputs"

_CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class LocalizeItem:
    """One field's file, and where `--apply` would put a copy of it."""

    field: str
    source: Path
    size_bytes: int
    destination: Path


@dataclass(frozen=True)
class LocalizePlan:
    """What `data localize --apply` would do to `config_path`, computed only."""

    config_path: Path
    output_directory: Path
    items: tuple[LocalizeItem, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)


def _field_value(df: pd.DataFrame, name: str) -> Optional[str]:
    """The last-declared, non-empty string value for `name`, or None.

    Matches `LoadExperimentConfig._parse_df`'s own "later row wins" merge, so
    a plan agrees with what `ExperimentConfig.from_var_dict` would actually
    resolve for this field.
    """
    rows = df[df["variable"].astype(str).str.strip() == name]
    if rows.empty:
        return None
    raw = rows.iloc[-1]["value"]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_localize_plan(config_path: str | Path) -> LocalizePlan:
    """What `--apply` would copy for `config_path`, without touching anything.

    Raises:
        ValueError: `config_path` declares no `output_directory` (nothing to
            localize into), or a localizable field names a file that does not
            exist -- a broken reference is worth catching here just as early
            as everywhere else this plan catches one (`PSR-01`/`PSR-03`).
    """
    from ..config.experiment_config import LoadExperimentConfig

    resolved_config_path = Path(config_path).expanduser().resolve()
    df = LoadExperimentConfig(resolved_config_path).df
    config_dir = resolved_config_path.parent

    output_directory_raw = _field_value(df, "output_directory")
    if output_directory_raw is None:
        raise ValueError(
            f"{resolved_config_path} has no output_directory; nothing to localize into."
        )
    output_directory = Path(
        resolve_config_path(output_directory_raw, config_dir=config_dir, field="output_directory")
    ).expanduser()
    dest_dir = output_directory / LOCALIZED_SUBDIR

    items = []
    for field in LOCALIZABLE_FIELDS:
        raw = _field_value(df, field)
        if raw is None:
            continue
        resolved = Path(resolve_config_path(raw, config_dir=config_dir, field=field)).expanduser()
        if not resolved.is_file():
            raise ValueError(f"{field}={resolved} does not exist; cannot localize a missing file.")
        if resolved.parent == dest_dir:
            continue  # already a localized copy
        items.append(
            LocalizeItem(
                field=field,
                source=resolved,
                size_bytes=resolved.stat().st_size,
                destination=dest_dir / resolved.name,
            )
        )
    return LocalizePlan(
        config_path=resolved_config_path, output_directory=output_directory, items=tuple(items)
    )


def apply_localize_plan(
    plan: LocalizePlan, *, out_config_path: Optional[str | Path] = None
) -> tuple[Path, list[LocalizeItem]]:
    """Copy every item in `plan` and write a new config pointing at the copies.

    The original config is never modified -- a new file is written instead,
    defaulting to `<config>.localized<suffix>` next to it.

    Args:
        plan: A plan from `build_localize_plan`.
        out_config_path: Where to write the localized config.

    Returns:
        `(new_config_path, copied)`. `copied` excludes any item whose
        destination already held byte-identical content, so a repeat
        `--apply` is a safe no-op for those rather than a re-copy.

    Raises:
        FileExistsError: A destination already exists with content different
            from its source -- refuses to silently overwrite an unrelated or
            stale prior localization.
    """
    from ..config.experiment_config import LoadExperimentConfig

    dest_dir = plan.output_directory / LOCALIZED_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[LocalizeItem] = []
    new_values: dict[str, str] = {}
    for item in plan.items:
        if item.destination.exists():
            if item.destination.stat().st_size == item.size_bytes and _sha256_file(
                item.destination
            ) == _sha256_file(item.source):
                new_values[item.field] = str(item.destination)
                continue
            raise FileExistsError(
                f"{item.destination} already exists with content different from "
                f"{item.source}; refusing to overwrite. Remove it first if this is deliberate."
            )
        shutil.copy2(item.source, item.destination)
        copied.append(item)
        new_values[item.field] = str(item.destination)

    df = LoadExperimentConfig(plan.config_path).df.copy()
    for field, new_path in new_values.items():
        mask = df["variable"].astype(str).str.strip() == field
        df.loc[mask, "value"] = new_path

    out_path = (
        Path(out_config_path)
        if out_config_path is not None
        else plan.config_path.with_suffix(".localized" + plan.config_path.suffix)
    )
    df.to_csv(out_path, index=False)
    return out_path, copied
