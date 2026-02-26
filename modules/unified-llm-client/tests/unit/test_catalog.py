"""Tests for unified_llm.catalog — model catalog and lookup functions."""

from unified_llm.catalog import get_latest_model, get_model_info, list_models


class TestGetModelInfo:
    """Spec §2.9 — get_model_info() lookup."""

    def test_known_model(self) -> None:
        info = get_model_info("claude-sonnet-4-20250514")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.supports_tools is True

    def test_unknown_model_returns_none(self) -> None:
        """Spec: Unknown model strings pass through — catalog is advisory."""
        assert get_model_info("nonexistent-model-xyz") is None

    def test_alias_lookup(self) -> None:
        """Spec: Models have aliases for shorthand."""
        # The catalog should have at least one alias defined
        all_models = list_models()
        models_with_aliases = [m for m in all_models if m.aliases]
        if models_with_aliases:
            model = models_with_aliases[0]
            alias = model.aliases[0]
            result = get_model_info(alias)
            assert result is not None
            assert result.id == model.id


class TestListModels:
    """Spec §2.9 — list_models() with optional provider filter."""

    def test_list_all(self) -> None:
        models = list_models()
        assert len(models) > 0

    def test_filter_by_provider(self) -> None:
        anthropic_models = list_models(provider="anthropic")
        assert len(anthropic_models) > 0
        assert all(m.provider == "anthropic" for m in anthropic_models)

    def test_filter_unknown_provider(self) -> None:
        assert list_models(provider="nonexistent") == []


class TestGetLatestModel:
    """Spec §2.9 — get_latest_model() for each provider."""

    def test_latest_anthropic(self) -> None:
        model = get_latest_model("anthropic")
        assert model is not None
        assert model.provider == "anthropic"

    def test_latest_openai(self) -> None:
        model = get_latest_model("openai")
        assert model is not None
        assert model.provider == "openai"

    def test_latest_gemini(self) -> None:
        model = get_latest_model("gemini")
        assert model is not None
        assert model.provider == "gemini"

    def test_latest_unknown_provider(self) -> None:
        assert get_latest_model("nonexistent") is None

    def test_latest_with_capability_filter(self) -> None:
        model = get_latest_model("anthropic", capability="reasoning")
        assert model is not None
        assert model.supports_reasoning is True
