"""Model catalog — advisory model lookup (Spec §2.9).

Unknown model strings pass through. The catalog is not restrictive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unified_llm.types import ModelInfo

_CATALOG: list[ModelInfo] | None = None
_ALIAS_MAP: dict[str, str] | None = None


def _load_catalog() -> tuple[list[ModelInfo], dict[str, str]]:
    """Load the catalog from the shipped JSON data file."""
    global _CATALOG, _ALIAS_MAP
    if _CATALOG is not None and _ALIAS_MAP is not None:
        return _CATALOG, _ALIAS_MAP

    data_path = Path(__file__).parent / "data" / "models.json"
    raw: list[dict[str, Any]] = json.loads(data_path.read_text())

    models: list[ModelInfo] = []
    aliases: dict[str, str] = {}

    for entry in raw:
        model = ModelInfo(
            id=entry["id"],
            provider=entry["provider"],
            display_name=entry["display_name"],
            context_window=entry["context_window"],
            max_output=entry.get("max_output"),
            supports_tools=entry["supports_tools"],
            supports_vision=entry["supports_vision"],
            supports_reasoning=entry["supports_reasoning"],
            input_cost_per_million=entry.get("input_cost_per_million"),
            output_cost_per_million=entry.get("output_cost_per_million"),
            aliases=entry.get("aliases", []),
        )
        models.append(model)
        for alias in model.aliases:
            aliases[alias] = model.id

    _CATALOG = models
    _ALIAS_MAP = aliases
    return models, aliases


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up a model by ID or alias. Returns None if unknown."""
    models, aliases = _load_catalog()
    # Direct ID match
    for model in models:
        if model.id == model_id:
            return model
    # Alias match
    canonical = aliases.get(model_id)
    if canonical:
        for model in models:
            if model.id == canonical:
                return model
    return None


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """List all known models, optionally filtered by provider."""
    models, _ = _load_catalog()
    if provider is None:
        return list(models)
    return [m for m in models if m.provider == provider]


def get_latest_model(
    provider: str,
    capability: str | None = None,
) -> ModelInfo | None:
    """Get the newest/best model for a provider, optionally filtered by capability."""
    candidates = list_models(provider)
    if capability:
        cap_map = {
            "reasoning": lambda m: m.supports_reasoning,
            "vision": lambda m: m.supports_vision,
            "tools": lambda m: m.supports_tools,
        }
        filter_fn = cap_map.get(capability)
        if filter_fn:
            candidates = [m for m in candidates if filter_fn(m)]
    return candidates[0] if candidates else None
