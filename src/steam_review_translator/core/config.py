"""Config loading, project path resolution, and ``.env`` handling (no extra deps)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PIPELINE_CONFIG = "configs/pipeline.yaml"

# name -> (env var override or None, default path relative to project root)
_BASE_DIRS: dict[str, tuple[str | None, str]] = {
    "data": ("DATA_DIR", "data"),
    "artifacts": ("ARTIFACTS_DIR", "artifacts"),
    "configs": (None, "configs"),
}
# name -> (base dir name, sub path)
_SUB_DIRS: dict[str, tuple[str, str]] = {
    "raw": ("data", "raw"),
    "interim": ("data", "interim"),
    "processed": ("data", "processed"),
    "logs": ("artifacts", "logs"),
    "models": ("artifacts", "models"),
    "runs": ("artifacts", "runs"),
}


def project_root() -> Path:
    """Return the repository root.

    Uses ``SRT_PROJECT_ROOT`` if set; otherwise the first ancestor of the current directory
    (then of this file) that contains both ``pyproject.toml`` and ``configs/``; otherwise
    the current directory.
    """
    env = os.environ.get("SRT_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
                return candidate
    return Path.cwd().resolve()


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path``: absolute as-is, else relative to cwd if it exists, else to the root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return project_root() / p


def get_path(name: str, *, create: bool = False) -> Path:
    """Return a well-known project directory.

    Names: ``data``, ``raw``, ``interim``, ``processed``, ``artifacts``, ``logs``,
    ``models``, ``runs``, ``configs``. ``DATA_DIR`` / ``ARTIFACTS_DIR`` env vars override
    the base dirs (relative values are resolved against the project root).
    """
    if name in _SUB_DIRS:
        base, sub = _SUB_DIRS[name]
        path = get_path(base) / sub
    elif name in _BASE_DIRS:
        env_var, default = _BASE_DIRS[name]
        raw = os.environ.get(env_var) if env_var else None
        path = Path(raw or default).expanduser()
        if not path.is_absolute():
            path = project_root() / path
    else:
        known = sorted({*_BASE_DIRS, *_SUB_DIRS})
        raise KeyError(f"Unknown path name {name!r}; expected one of {known}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Missing file is not an error. Existing environment variables win unless ``override``.
    Returns the parsed key/value pairs.
    """
    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping. An empty file gives ``{}``; a non-mapping top level is an error."""
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved}: expected a mapping at top level, got {type(data).__name__}")
    return data


class ComponentConfig(BaseModel):
    """Which registered implementation a task uses, plus its constructor kwargs."""

    model_config = ConfigDict(extra="forbid")

    impl: str = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Top-level ``configs/pipeline.yaml``."""

    model_config = ConfigDict(extra="forbid")

    sentiment: ComponentConfig = Field(default_factory=ComponentConfig)
    aspect: ComponentConfig = Field(default_factory=ComponentConfig)
    summarization: ComponentConfig = Field(default_factory=ComponentConfig)


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate the pipeline config (default: ``configs/pipeline.yaml``)."""
    return PipelineConfig.model_validate(load_yaml(path or DEFAULT_PIPELINE_CONFIG))
