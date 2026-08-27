"""Named storage roots: `${data}/...` instead of a machine-specific absolute path.

An experiment config records where its inputs and outputs live. Written as
absolute paths, those values are correct on exactly one machine with exactly one
set of drives mounted — so moving a tree, or reading it from a second machine,
means editing every config (`F-PSR-02`).

A **root** is a named logical location. `${data}/<run>/pod5` resolves through a
machine-local binding, so the config is portable and only the binding is local.

Resolution order, first match winning:

1. environment variable ``SMFTOOLS_ROOT_<NAME>`` (upper-cased)
2. the user roots file, ``$SMFTOOLS_CONFIG_DIR/roots.toml`` or
   ``~/.config/smftools/roots.toml``
3. a ``roots.toml`` found by walking up from the config file's directory
4. nothing — an unresolved name is an error, never a literal

That last point is deliberate. Treating an unresolved ``${data}`` as a directory
literally named ``${data}`` would turn a typo into a silently wrong path, which
is the failure mode this whole program exists to remove.

Bindings may already be written as a list. Phase 2 takes the first entry that
exists (falling back to the first), so that when `PSR-16` makes a root an ordered
set of locations the file format does not have to change under anyone.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

#: ``${name}`` — the only interpolation form. ``$name`` is deliberately not
#: recognised: shell-style bare names collide with ordinary path text.
_ROOT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

ROOTS_FILENAME = "roots.toml"
ENV_PREFIX = "SMFTOOLS_ROOT_"


class RootResolutionError(ValueError):
    """A config referenced a root with no binding on this machine."""


@dataclass(frozen=True)
class RootBinding:
    """One root name, where it points, and which layer supplied it."""

    name: str
    path: Path
    source: str


def _load_toml(path: Path) -> Mapping[str, object]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("ignoring unreadable roots file %s: %s", path, exc)
        return {}
    return {}


def _binding_path(value: object) -> Optional[Path]:
    """Resolve one binding value, which may already be a list (`PSR-16`)."""
    if isinstance(value, (list, tuple)):
        candidates = [Path(str(item)).expanduser() for item in value if str(item).strip()]
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def user_roots_file() -> Path:
    """Where the machine-local roots file lives."""
    configured = os.environ.get("SMFTOOLS_CONFIG_DIR")
    base = Path(configured).expanduser() if configured else Path.home() / ".config" / "smftools"
    return base / ROOTS_FILENAME


def _walk_up_roots_files(start: Path) -> list[Path]:
    """Every ``roots.toml`` from ``start`` up to the filesystem root, nearest first."""
    found = []
    for directory in [start, *start.parents]:
        candidate = directory / ROOTS_FILENAME
        if candidate.is_file():
            found.append(candidate)
    return found


def resolve_root(name: str, *, config_dir: Path | None = None) -> Optional[RootBinding]:
    """Resolve one root name through the layered order, or None when unbound.

    Args:
        name: The root name, as written inside ``${...}``.
        config_dir: The directory of the config being loaded, for the walk-up
            layer. ``None`` skips that layer.

    Returns:
        RootBinding or None: The binding and which layer supplied it.
    """
    env_value = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
    if env_value and env_value.strip():
        return RootBinding(name, Path(env_value).expanduser(), f"{ENV_PREFIX}{name.upper()}")

    user_file = user_roots_file()
    if user_file.is_file():
        table = _load_toml(user_file).get("roots", {})
        if isinstance(table, Mapping) and name in table:
            path = _binding_path(table[name])
            if path is not None:
                return RootBinding(name, path, str(user_file))

    if config_dir is not None:
        for candidate in _walk_up_roots_files(Path(config_dir)):
            table = _load_toml(candidate).get("roots", {})
            if isinstance(table, Mapping) and name in table:
                path = _binding_path(table[name])
                if path is not None:
                    return RootBinding(name, path, str(candidate))
    return None


def referenced_roots(value: str) -> list[str]:
    """Every root name a config value references."""
    return _ROOT_REFERENCE.findall(str(value))


def expand_roots(value: str, *, config_dir: Path | None = None, field: str = "") -> str:
    """Replace every ``${root}`` in one config value with its bound path.

    Args:
        value: The raw config value.
        config_dir: Directory of the config being loaded, for the walk-up layer.
        field: Field name, used only to make an error actionable.

    Returns:
        str: The value with every reference expanded.

    Raises:
        RootResolutionError: If any referenced root has no binding. Never falls
            back to the literal text: a typo'd root name must not become a
            directory name.
    """
    text = str(value)
    names = referenced_roots(text)
    if not names:
        return text
    for name in dict.fromkeys(names):
        binding = resolve_root(name, config_dir=config_dir)
        if binding is None:
            where = f" for {field}" if field else ""
            raise RootResolutionError(
                f"config{where} references root '${{{name}}}', which is not bound on this "
                f"machine. Bind it with {ENV_PREFIX}{name.upper()}=<path>, or add "
                f'[roots]\\n{name} = "<path>" to {user_roots_file()}.'
            )
        text = text.replace(f"${{{name}}}", str(binding.path))
    return text


def resolve_config_path(value: str, *, config_dir: Path | None = None, field: str = "") -> str:
    """Expand roots and anchor a bare relative path to the config's own directory.

    A relative path in a config used to resolve against the *working directory*,
    so the same config meant different things depending on where it was run from
    (`PSR-05`). Anchoring it to the config file makes an experiment directory
    self-describing.

    Args:
        value: The raw config value.
        config_dir: Directory of the config being loaded. ``None`` leaves a
            relative path untouched, preserving legacy behaviour for callers
            that have no config file.
        field: Field name, for error messages.

    Returns:
        str: The resolved path value.
    """
    expanded = expand_roots(value, config_dir=config_dir, field=field)
    if not expanded:
        return expanded
    path = Path(expanded).expanduser()
    if path.is_absolute() or config_dir is None:
        return str(path) if expanded.startswith("~") else expanded

    anchored = (Path(config_dir) / path).resolve()
    if anchored.exists():
        return str(anchored)

    # Compatibility gate. Configs written before `PSR-05` resolved relative paths
    # against the *working directory*, and silently repointing them would break
    # working setups on upgrade. Defer to the old reading only when the new one
    # names nothing and the old one names something real, and say so, so the
    # config gets fixed rather than depending on where it is run from forever.
    legacy = (Path.cwd() / path).resolve()
    if legacy.exists():
        logger.warning(
            "config value %s=%r resolves against the working directory, not the config "
            "file's directory (%s). That is the pre-PSR-05 behaviour and is kept only "
            "because the config-relative path does not exist. Make it absolute, "
            "root-qualified, or relative to the config to stop depending on where "
            "smftools is run from.",
            field or "path",
            value,
            config_dir,
        )
        return str(legacy)
    return str(anchored)


def list_bindings(names: list[str], *, config_dir: Path | None = None) -> list[RootBinding]:
    """Resolve several roots, skipping unbound ones. For ``data roots list``."""
    bindings = []
    for name in dict.fromkeys(names):
        binding = resolve_root(name, config_dir=config_dir)
        if binding is not None:
            bindings.append(binding)
    return bindings


def known_roots(*, config_dir: Path | None = None) -> dict[str, RootBinding]:
    """Every root bound on this machine, nearest layer winning.

    Used when *writing* a portable pointer: a path under a bound root can be
    stored as ``${root}/relative`` rather than as an absolute string or a
    relative walk that encodes a mount name (`PSR-07`).
    """
    bindings: dict[str, RootBinding] = {}

    def _absorb(table: object, source: str) -> None:
        if not isinstance(table, Mapping):
            return
        for name, value in table.items():
            if name in bindings:
                continue
            path = _binding_path(value)
            if path is not None:
                bindings[str(name)] = RootBinding(str(name), path, source)

    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIX) and value.strip():
            name = key[len(ENV_PREFIX) :].lower()
            bindings.setdefault(name, RootBinding(name, Path(value).expanduser(), key))

    user_file = user_roots_file()
    if user_file.is_file():
        _absorb(_load_toml(user_file).get("roots", {}), str(user_file))
    if config_dir is not None:
        for candidate in _walk_up_roots_files(Path(config_dir)):
            _absorb(_load_toml(candidate).get("roots", {}), str(candidate))
    return bindings


def qualify_with_root(path: Path, *, config_dir: Path | None = None) -> Optional[str]:
    """Express ``path`` as ``${root}/relative`` when a bound root contains it.

    Returns:
        str or None: The qualified form, using the *longest* matching root so the
        most specific binding wins, or None when no root contains the path.
    """
    resolved = Path(path).resolve()
    best: Optional[tuple[int, str]] = None
    for name, binding in known_roots(config_dir=config_dir).items():
        try:
            root_path = binding.path.resolve()
        except OSError:  # pragma: no cover - unresolvable binding
            continue
        if resolved == root_path or root_path in resolved.parents:
            depth = len(root_path.parts)
            if best is None or depth > best[0]:
                relative = resolved.relative_to(root_path).as_posix()
                best = (depth, f"${{{name}}}/{relative}" if relative else f"${{{name}}}")
    return best[1] if best else None
