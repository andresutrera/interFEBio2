"""Configuration helpers for locating the FEBio source tree."""

from __future__ import annotations

import os
from pathlib import Path


def _default_project_root() -> Path:
    """Return the inferred project root directory."""

    return Path(__file__).resolve().parents[2]


def _default_febio_root() -> Path:
    """Return the default FEBio checkout path next to the project."""

    project_root = _default_project_root()
    candidates = [
        project_root / "FEBio",
        project_root.parent / "FEBio",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def get_febio_root() -> Path:
    """Resolve the FEBio repository path, honouring ``FEBIO_ROOT`` when set."""

    env_override = os.environ.get("FEBIO_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return _default_febio_root()


FEBIO_ROOT: Path = get_febio_root()


def resolve_febio_path(*parts: str | os.PathLike[str]) -> Path:
    """Join ``parts`` onto the resolved FEBio root path."""

    return FEBIO_ROOT.joinpath(*parts)
