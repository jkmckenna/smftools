"""Harmonize reference names across experiments by sequence-identity + YAML aliases.

Each experiment stores a name-independent ``reference_uid`` per reference (a hash of
its sequence). Two experiments that named the same sequence differently therefore
share a uid and harmonize automatically. The project ``reference_registry.yaml``
adds (1) friendly canonical names for uids and (2) manual alias groups to unify
references that are *not* byte-identical but are the same locus (trimmed flanks, a
SNP, extra padding). Auto-merge is exact-hash only; near-identical merging is always
explicit, so distinct loci are never silently merged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

REFERENCE_REGISTRY_FILENAME = "reference_registry.yaml"


class ReferenceRegistry:
    """Resolve ``reference_uid -> canonical_reference`` using optional YAML overrides."""

    def __init__(
        self,
        canonical_names: dict | None = None,
        aliases: dict | None = None,
    ) -> None:
        self.canonical_names: dict[str, str] = {
            str(k): str(v) for k, v in (canonical_names or {}).items()
        }
        self.aliases: dict[str, list[str]] = {
            str(name): [str(u) for u in uids] for name, uids in (aliases or {}).items()
        }
        self._alias_uid: dict[str, str] = {}
        for name, uids in self.aliases.items():
            for uid in uids:
                self._alias_uid[uid] = name

    def canonical_reference(self, uid: str) -> str:
        """Canonical name for a uid: explicit name > alias group > the uid itself."""
        uid = str(uid)
        if uid in self.canonical_names:
            return self.canonical_names[uid]
        if uid in self._alias_uid:
            return self._alias_uid[uid]
        return uid

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceRegistry":
        path = Path(path)
        if not path.exists():
            return cls()
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
        return cls(canonical_names=data.get("canonical_names", {}), aliases=data.get("aliases", {}))

    def save(self, path: str | Path) -> Path:
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {"canonical_names": self.canonical_names, "aliases": self.aliases},
                sort_keys=True,
            )
        )
        return path


def build_reference_alias_table(experiments: Iterable[dict], registry: ReferenceRegistry):
    """Build ``(experiment, reference_strand, reference_uid, canonical_reference)`` rows.

    Args:
        experiments: Registry entries with ``id`` and ``references`` ({ref_strand: uid}).
        registry: The :class:`ReferenceRegistry` used to resolve canonical names.

    Returns:
        pandas.DataFrame with the alias mapping (one row per experiment reference).
    """
    import pandas as pd

    rows = []
    for exp in experiments:
        for reference_strand, uid in (exp.get("references") or {}).items():
            rows.append(
                {
                    "experiment": exp["id"],
                    "reference_strand": str(reference_strand),
                    "reference_uid": str(uid),
                    "canonical_reference": registry.canonical_reference(uid),
                }
            )
    return pd.DataFrame(
        rows, columns=["experiment", "reference_strand", "reference_uid", "canonical_reference"]
    )


def detect_reference_conflicts(alias_table) -> list[str]:
    """Flag ambiguous reference names: one name mapping to >1 sequence across experiments."""
    warnings: list[str] = []
    if alias_table.empty:
        return warnings
    per_name = alias_table.groupby("reference_strand")["reference_uid"].nunique()
    for name, n_uids in per_name.items():
        if n_uids > 1:
            warnings.append(
                f"reference name '{name}' maps to {n_uids} distinct sequences across experiments; "
                "assign canonical names in reference_registry.yaml"
            )
    return warnings
