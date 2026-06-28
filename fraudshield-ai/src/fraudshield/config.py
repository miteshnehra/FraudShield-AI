"""Configuration loading and runtime path management."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.toml"


@dataclass(frozen=True)
class Settings:
    """Validated application settings loaded from YAML."""

    values: dict[str, Any]
    project_root: Path = PROJECT_ROOT

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing or invalid configuration section: {name}")
        return value

    def path(self, name: str) -> Path:
        raw_path = self.section("paths").get(name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise KeyError(f"Missing or invalid path setting: {name}")
        return (self.project_root / raw_path).resolve()


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load TOML configuration and verify the required sections."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as config_file:
        values = tomllib.load(config_file)

    if not isinstance(values, dict):
        raise ValueError("Configuration root must be a TOML mapping.")

    required_sections = {"app", "paths", "data", "model", "risk", "ui"}
    missing = sorted(required_sections.difference(values))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    return Settings(values=values)


def ensure_runtime_directories(settings: Settings) -> None:
    """Create directories used for generated runtime artifacts."""
    for name in ("raw_data", "processed_data", "models", "reports", "logs"):
        settings.path(name).mkdir(parents=True, exist_ok=True)
