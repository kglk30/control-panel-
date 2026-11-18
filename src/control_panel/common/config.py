"""Configuration loading for Control Panel.

Reads TOML config files from %LOCALAPPDATA%\\ControlPanel\\config\\
and provides typed access via Pydantic v2 models.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AppDirs",
    "OrchestratorConfig",
    "SafetyConfig",
    "InferenceConfig",
    "ResourceConfig",
    "AppConfig",
    "get_app_dirs",
    "load_config",
    "get_config",
    "get_api_key",
]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

def _default_data_root() -> Path:
    """Return the root data directory for Control Panel."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "ControlPanel"
    return Path.home() / ".control_panel"


@dataclass
class AppDirs:
    """All filesystem paths used by Control Panel at runtime."""

    data_dir: Path
    config_dir: Path
    database_path: Path
    logs_dir: Path
    models_dir: Path
    templates_dir: Path

    def ensure_all(self) -> None:
        """Create all directories if they don't exist."""
        for p in [self.config_dir, self.logs_dir, self.models_dir, self.templates_dir]:
            p.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


def get_app_dirs(data_root: Optional[Path] = None) -> AppDirs:
    """Return the standard application directory layout."""
    root = data_root or _default_data_root()
    return AppDirs(
        data_dir=root,
        config_dir=root / "config",
        database_path=root / "data" / "control_panel.db",
        logs_dir=root / "logs",
        models_dir=root / "models",
        templates_dir=root / "data" / "templates",
    )


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class OrchestratorConfig(BaseModel):
    """Orchestrator tuning parameters."""

    model_config = ConfigDict(frozen=False)

    timeout_seconds: int = 300
    max_retries: int = 3


class SafetyConfig(BaseModel):
    """Safety Manager parameters and global denylist."""

    model_config = ConfigDict(frozen=False)

    max_ops_per_second: int = 10
    dry_run: bool = False
    denylist: list[str] = Field(default_factory=lambda: [
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
        "C:\\Users\\Default",
    ])


class InferenceConfig(BaseModel):
    """Inference Runtime and backend settings."""

    model_config = ConfigDict(frozen=False)

    preferred_backend: Optional[str] = None  # None / "" = auto-select
    api_base_url: str = "https://api.openai.com/v1"
    api_model: str = "gpt-4o-mini"
    api_timeout_seconds: int = 30
    api_max_retries: int = 2
    probe_on_startup: bool = True


class ResourceConfig(BaseModel):
    """Resource Governor thresholds."""

    model_config = ConfigDict(frozen=False)

    min_free_ram_mb: int = 512
    max_cpu_percent: float = 80.0
    min_free_disk_mb: int = 100
    safety_margin_mb: int = 512


class AppConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(frozen=False)

    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_config: Optional[AppConfig] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load AppConfig from a TOML file, falling back to defaults if missing."""
    if config_path is None:
        config_path = get_app_dirs().config_dir / "control_panel.toml"

    if not config_path.exists():
        return AppConfig()

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    # Parse each section independently so missing sections use defaults
    return AppConfig(
        orchestrator=OrchestratorConfig(**raw.get("orchestrator", {})),
        safety=SafetyConfig(**raw.get("safety", {})),
        inference=InferenceConfig(**{
            k: v for k, v in raw.get("inference", {}).items()
            if v != ""  # treat empty string as unset for optional str fields
        }),
        resources=ResourceConfig(**raw.get("resources", {})),
    )


def get_config() -> AppConfig:
    """Return the cached global config, loading defaults if not yet initialized."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Override the global config (used in testing and CLI --config flag)."""
    global _config
    _config = config


def get_api_key() -> Optional[str]:
    """Read the API key from the CONTROL_PANEL_API_KEY environment variable."""
    return os.environ.get("CONTROL_PANEL_API_KEY") or None
