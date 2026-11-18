"""Model Registry — catalog of known model profiles for the inference runtime.

ModelProfile declares format, runtime type, resource requirements,
supported tasks, backends, and devices. No hardcoded model names anywhere else.

Loaded from config/models.toml at startup.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from control_panel.common.logging import get_logger
from control_panel.common.models import ModelRequirements

__all__ = ["ModelProfile", "ModelRegistry", "get_registry"]

log = get_logger(__name__)


class ModelProfile(BaseModel):
    """Complete profile of one model in the registry."""

    model_config = ConfigDict(frozen=False)

    model_id: str
    model_name: str
    format: str                  # "onnx" | "none"
    runtime_type: str            # "genai" | "structured" | "rule_based"
    quantization: Optional[str] = None   # "int4"|"int8"|"fp16"|"fp32"|None
    file_path: Optional[str] = None

    requirements: ModelRequirements = Field(default_factory=ModelRequirements)

    supported_tasks: list[str] = Field(default_factory=list)
    supported_backends: list[str] = Field(default_factory=list)
    supported_devices: list[str] = Field(default_factory=list)
    context_length: int = 0
    structured_output_support: bool = False

    source: str = "unknown"      # "qualcomm_ai_hub"|"huggingface"|"onnx_zoo"|"builtin"
    version: str = "1.0.0"

    def is_available_on_disk(self) -> bool:
        """Return True if the model file/directory exists on disk.

        Relative paths are resolved against the project root
        (directory containing pyproject.toml), then against cwd.
        """
        if not self.file_path:
            return self.runtime_type == "rule_based"

        p = Path(self.file_path)
        if p.is_absolute():
            return p.exists()

        # Try relative to project root (where pyproject.toml lives)
        project_root = Path(__file__).parent.parent.parent.parent.parent  # src/control_panel/inference → project root
        absolute = project_root / p
        if absolute.exists():
            return True

        # Fall back to cwd
        return (Path.cwd() / p).exists()


class ModelRegistry:
    """Catalog of known ModelProfiles. Queried by the Backend Selection Engine."""

    def __init__(self) -> None:
        self._profiles: dict[str, ModelProfile] = {}

    def register(self, profile: ModelProfile) -> None:
        """Add a model profile to the registry."""
        self._profiles[profile.model_id] = profile
        log.debug("model_registry.registered", model_id=profile.model_id)

    def get(self, model_id: str) -> Optional[ModelProfile]:
        """Return a profile by model_id, or None."""
        return self._profiles.get(model_id)

    def list_models(self) -> list[ModelProfile]:
        """Return all registered model profiles."""
        return list(self._profiles.values())

    def is_model_available(self, model_id: str) -> bool:
        """Return True if the model profile exists AND the file is on disk."""
        profile = self._profiles.get(model_id)
        return profile is not None and profile.is_available_on_disk()

    def find_compatible(
        self,
        task_type: str,
        runtime_type: str,
        backend: Optional[str] = None,
        max_ram_mb: Optional[int] = None,
    ) -> list[ModelProfile]:
        """Return profiles compatible with the given criteria, sorted by preference.

        Preference order: models whose file is available > others;
        lower RAM usage within that group.
        """
        results: list[ModelProfile] = []
        for profile in self._profiles.values():
            if task_type not in profile.supported_tasks:
                continue
            if profile.runtime_type != runtime_type:
                continue
            if backend is not None and backend not in profile.supported_backends:
                continue
            if max_ram_mb is not None:
                if profile.requirements.estimated_ram_mb > max_ram_mb:
                    continue
            results.append(profile)

        # Sort: available-on-disk first, then by RAM (ascending)
        results.sort(
            key=lambda p: (not p.is_available_on_disk(), p.requirements.estimated_ram_mb)
        )
        return results

    def load_from_toml(self, toml_path: Path) -> int:
        """Load model profiles from a models.toml file. Returns count loaded."""
        if not toml_path.exists():
            log.warning("model_registry.toml_not_found", path=str(toml_path))
            return 0

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        loaded = 0
        for raw in data.get("models", []):
            try:
                req_raw = raw.pop("requirements", {})
                req = ModelRequirements(**req_raw)
                profile = ModelProfile(requirements=req, **raw)
                self.register(profile)
                loaded += 1
            except Exception as e:
                log.warning("model_registry.parse_error", error=str(e), raw=raw)

        log.info("model_registry.loaded", count=loaded, path=str(toml_path))
        return loaded


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Return the global ModelRegistry singleton.

    Automatically loads from models.toml on first call.
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
        # Load from default config location
        from control_panel.common.config import get_app_dirs
        toml_path = get_app_dirs().config_dir / "models.toml"
        if not toml_path.exists():
            # Fall back to bundled config/models.toml
            bundled = Path(__file__).parent.parent.parent.parent / "config" / "models.toml"
            if bundled.exists():
                toml_path = bundled
        _registry_instance.load_from_toml(toml_path)
    return _registry_instance
