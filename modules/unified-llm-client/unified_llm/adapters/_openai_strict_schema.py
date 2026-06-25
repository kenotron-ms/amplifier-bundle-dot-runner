"""OpenAI strict-mode JSON Schema transformer.

OpenAI's structured-output strict mode enforces a SUPERSET of JSON Schema.
Every object node in the schema must satisfy:

  1. ``additionalProperties: false``
  2. ``required`` lists **every** key that appears in ``properties``

A standard, legal schema with an optional field (present in ``properties``
but absent from ``required``) causes a hard 400 at runtime:

    "required is required to be an array including every key in properties.
     Missing 'discount_code'."

This module provides :func:`make_openai_strict_schema`, which returns a
**deep copy** of the caller's schema transformed to satisfy the strict-mode
contract **without changing user intent**:

* Every object node gets ``additionalProperties: false``.
* ``required`` is expanded to include all ``properties`` keys.
* Fields that were NOT in the original ``required`` (i.e. optional) become
  *nullable* — their type is widened to ``["original_type", "null"]`` — so
  the model can signal "absent" by returning ``null`` rather than by
  omitting the key.  The caller receives ``null`` back for such fields and
  can treat it as "not provided."
* The caller's original schema is **never** mutated.

Recursion covers all standard schema locations where object sub-schemas
appear: ``properties``, ``items`` (array), ``$defs`` / ``definitions``,
and the combining keywords ``anyOf`` / ``oneOf`` / ``allOf``.

References:
  https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
"""

from __future__ import annotations

import copy
from typing import Any


def make_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *schema* transformed for OpenAI strict mode.

    The caller's *schema* dict is **never** mutated.

    Args:
        schema: User-supplied JSON Schema dict (may contain optional fields,
                missing ``additionalProperties``, etc.).

    Returns:
        A new dict satisfying OpenAI strict-mode requirements:
        ``additionalProperties: false`` and a fully-populated ``required``
        array on every object node, with originally-optional properties made
        nullable so the model can represent "absent" as ``null``.
    """
    result: dict[str, Any] = copy.deepcopy(schema)
    _transform_node(result)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _transform_node(node: Any) -> None:
    """Recursively transform *node* **in-place** for OpenAI strict mode.

    Safe to call on the deep-copy produced by :func:`make_openai_strict_schema`.
    """
    if not isinstance(node, dict):
        return

    # ------------------------------------------------------------------ #
    # Object nodes: must have additionalProperties:false and required=all #
    # ------------------------------------------------------------------ #
    properties = node.get("properties")
    if isinstance(properties, dict) and properties:
        # Capture the original required set BEFORE we expand it.
        orig_required: set[str] = set(node.get("required") or [])

        # Strict mode mandates additionalProperties:false on every object.
        node["additionalProperties"] = False

        # Expand required to ALL property keys (strict mode requirement).
        node["required"] = list(properties.keys())

        # For each originally-optional property, widen its type to include
        # "null" so the model can return null (= absent) instead of omitting.
        for key, prop_schema in properties.items():
            if key not in orig_required:
                _make_nullable(prop_schema)
            # Recurse into every property schema regardless of optionality.
            _transform_node(prop_schema)

    # ------------------------------------------------------------------ #
    # Array nodes: recurse into the items schema                          #
    # ------------------------------------------------------------------ #
    items = node.get("items")
    if isinstance(items, dict):
        _transform_node(items)
    elif isinstance(items, list):
        for item in items:
            _transform_node(item)

    # ------------------------------------------------------------------ #
    # $defs / definitions: recurse into each named definition             #
    # ------------------------------------------------------------------ #
    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if isinstance(defs, dict):
            for def_schema in defs.values():
                _transform_node(def_schema)

    # ------------------------------------------------------------------ #
    # Combining keywords: recurse into each branch                        #
    # ------------------------------------------------------------------ #
    for combo_key in ("anyOf", "oneOf", "allOf"):
        branches = node.get(combo_key)
        if isinstance(branches, list):
            for branch in branches:
                _transform_node(branch)


def _make_nullable(node: Any) -> None:
    """Widen *node*'s type to include ``"null"`` in-place.

    Handles four cases:

    1. ``anyOf`` present → append ``{"type": "null"}`` if not already there.
    2. ``type`` is a string → convert to ``[original, "null"]``.
    3. ``type`` is a list → add ``"null"`` if absent.
    4. No ``type`` / no ``anyOf`` (e.g. bare ``{"enum": [...]}``):
       Wrap the entire node's contents in an ``anyOf`` with a null branch.
    """
    if not isinstance(node, dict):
        return

    # Case 1: anyOf already present — add a null branch if needed.
    if "anyOf" in node:
        branches = node["anyOf"]
        if isinstance(branches, list):
            null_branch: dict[str, Any] = {"type": "null"}
            if null_branch not in branches:
                branches.append(null_branch)
        return

    # Cases 2 & 3: explicit type field.
    if "type" in node:
        t = node["type"]
        if isinstance(t, str):
            if t != "null":
                node["type"] = [t, "null"]
        elif isinstance(t, list):
            if "null" not in t:
                t.append("null")
        return

    # Case 4: no type, no anyOf (e.g. enum-only or $ref-only schema).
    # Preserve all existing keywords in a non-null branch; add null alternative.
    non_null_branch = dict(node)
    node.clear()
    node["anyOf"] = [non_null_branch, {"type": "null"}]
