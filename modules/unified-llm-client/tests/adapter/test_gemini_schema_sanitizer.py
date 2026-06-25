"""Regression tests for GeminiAdapter._sanitize_gemini_schema().

Live-found bug: JSON schemas containing keywords unsupported by Gemini's
response_schema (additionalProperties, $schema, $id, patternProperties, etc.)
caused a 400 INVALID_ARGUMENT error from the API.

The sanitizer was added to strip those keywords recursively before forwarding
the schema to the Gemini API.  This test suite locks in that behaviour so a
future refactor cannot silently reintroduce the bug.
"""

from __future__ import annotations

from unittest.mock import patch

from unified_llm.adapters.gemini import GeminiAdapter


def _make_adapter() -> GeminiAdapter:
    with patch("unified_llm.adapters.gemini.genai.Client"):
        return GeminiAdapter(api_key="test-key")


class TestGeminiSchemaSanitizer:
    """_sanitize_gemini_schema() must strip unsupported keys, preserve supported keys."""

    # ------------------------------------------------------------------
    # Basic stripping at the root level
    # ------------------------------------------------------------------

    def test_strips_additional_properties_false(self) -> None:
        """additionalProperties: false is stripped from the root schema."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "additionalProperties" not in result

    def test_strips_schema_keyword(self) -> None:
        """$schema is stripped from the root schema."""
        adapter = _make_adapter()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "$schema" not in result

    def test_strips_id_keyword(self) -> None:
        """$id is stripped from the root schema."""
        adapter = _make_adapter()
        schema = {
            "$id": "https://example.com/schema",
            "type": "object",
            "properties": {},
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "$id" not in result

    def test_strips_pattern_properties(self) -> None:
        """patternProperties is stripped."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "patternProperties": {"^S_": {"type": "string"}},
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "patternProperties" not in result

    def test_strips_all_of(self) -> None:
        """allOf is stripped."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "allOf": [{"required": ["name"]}],
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "allOf" not in result

    def test_strips_not(self) -> None:
        """not is stripped."""
        adapter = _make_adapter()
        schema = {"not": {"type": "null"}}
        result = adapter._sanitize_gemini_schema(schema)
        assert "not" not in result

    def test_strips_if_then_else(self) -> None:
        """if/then/else are stripped."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "if": {"properties": {"flag": {"const": True}}},
            "then": {"required": ["extra"]},
            "else": {"required": ["other"]},
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "if" not in result
        assert "then" not in result
        assert "else" not in result

    # ------------------------------------------------------------------
    # Supported keywords survive
    # ------------------------------------------------------------------

    def test_preserves_type(self) -> None:
        """type is preserved."""
        adapter = _make_adapter()
        schema = {"type": "object", "additionalProperties": False}
        result = adapter._sanitize_gemini_schema(schema)
        assert result["type"] == "object"

    def test_preserves_properties(self) -> None:
        """properties is preserved."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "additionalProperties": False,
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "properties" in result
        assert "name" in result["properties"]
        assert "age" in result["properties"]

    def test_preserves_required(self) -> None:
        """required is preserved."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "$schema": "http://json-schema.org/draft-07/schema#",
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert result["required"] == ["x"]

    def test_preserves_description(self) -> None:
        """description is preserved."""
        adapter = _make_adapter()
        schema = {
            "type": "string",
            "description": "A human name",
            "$id": "irrelevant",
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert result["description"] == "A human name"

    def test_preserves_enum(self) -> None:
        """enum is preserved."""
        adapter = _make_adapter()
        schema = {
            "type": "string",
            "enum": ["red", "green", "blue"],
            "additionalProperties": False,
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert result["enum"] == ["red", "green", "blue"]

    def test_preserves_items(self) -> None:
        """items is preserved for array schemas."""
        adapter = _make_adapter()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "additionalProperties": False,
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert result["items"] == {"type": "string"}

    # ------------------------------------------------------------------
    # Recursive stripping through nested structures
    # ------------------------------------------------------------------

    def test_strips_recursively_in_nested_properties(self) -> None:
        """Unsupported keys inside nested property schemas are stripped recursively."""
        adapter = _make_adapter()
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                    "additionalProperties": False,  # must be stripped
                    "$schema": "nested",  # must be stripped
                    "required": ["street"],  # must be kept
                }
            },
            "additionalProperties": False,
        }
        result = adapter._sanitize_gemini_schema(schema)

        # Root
        assert "additionalProperties" not in result

        # Nested
        address = result["properties"]["address"]
        assert "additionalProperties" not in address
        assert "$schema" not in address
        assert address["required"] == ["street"]
        assert "street" in address["properties"]

    def test_strips_recursively_in_array_items(self) -> None:
        """Unsupported keys inside items schemas are stripped recursively."""
        adapter = _make_adapter()
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "additionalProperties": False,
            },
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "additionalProperties" not in result["items"]
        assert "value" in result["items"]["properties"]

    def test_combined_schema_strips_and_preserves(self) -> None:
        """A realistic schema with $schema + additionalProperties strips bad keys
        while preserving type, properties, required throughout."""
        adapter = _make_adapter()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title"},
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "$id": "tag-item",
                    },
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        }
        result = adapter._sanitize_gemini_schema(schema)

        # Root checks
        assert "$schema" not in result
        assert "additionalProperties" not in result
        assert result["type"] == "object"
        assert result["required"] == ["title"]

        # Properties preserved
        assert "title" in result["properties"]
        assert result["properties"]["title"]["description"] == "Title"

        # tags items cleaned
        assert "$id" not in result["properties"]["tags"]["items"]
        assert result["properties"]["tags"]["items"]["type"] == "string"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_schema_unchanged(self) -> None:
        """An empty schema dict passes through unchanged."""
        adapter = _make_adapter()
        result = adapter._sanitize_gemini_schema({})
        assert result == {}

    def test_schema_with_only_unsupported_keys_becomes_empty(self) -> None:
        """A schema that only has unsupported keys becomes an empty dict."""
        adapter = _make_adapter()
        schema = {
            "$schema": "...",
            "additionalProperties": False,
            "patternProperties": {},
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert result == {}

    def test_any_of_preserved(self) -> None:
        """anyOf is preserved (it IS supported by Gemini)."""
        adapter = _make_adapter()
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }
        result = adapter._sanitize_gemini_schema(schema)
        assert "anyOf" in result
