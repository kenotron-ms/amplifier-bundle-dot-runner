"""CSS-like model stylesheet for assigning LLM config to nodes.

Parses a mini-language where selectors target nodes and declarations
set model-related properties. Applied as a transform during the
INITIALIZE phase (before execution starts).

Spec coverage: STYLE-001-007, Section 8.6

Grammar:
    Stylesheet    ::= Rule+
    Rule          ::= Selector '{' Declaration ( ';' Declaration )* ';'? '}'
    Selector      ::= '*' | ShapeName | '.' ClassName | '#' Identifier
    Declaration   ::= Property ':' PropertyValue
    Property      ::= 'llm_model' | 'llm_provider' | 'reasoning_effort'

Specificity (M-21: added shape-name level):
    *           -> 0 (universal)
    shape_name  -> 1 (bare shape name, e.g. 'box', 'parallelogram')
    .class      -> 2
    #node_id    -> 3

Resolution order (highest wins):
    1. Explicit node attribute
    2. Stylesheet rule by specificity (ID > class > shape > universal)
    3. Graph-level default
    4. System default
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .graph import Graph

# Recognized stylesheet properties (spec Section 8.6)
_RECOGNIZED_PROPERTIES = frozenset({"llm_model", "llm_provider", "reasoning_effort"})

# Regex to match a single rule: selector { declarations }
# M-21: Added bare identifier (shape name) alternative: [A-Za-z][A-Za-z0-9_]*
_RULE_RE = re.compile(
    r"""
    ([*]|[#][A-Za-z_][A-Za-z0-9_]*|[.][a-z0-9-]+|[A-Za-z][A-Za-z0-9_]*)  # selector
    \s*\{                                              # opening brace
    ([^}]*)                                            # declarations
    \}                                                 # closing brace
    """,
    re.VERBOSE,
)

# Regex to match a single declaration: property : value
_DECL_RE = re.compile(
    r"""
    \s*([A-Za-z_][A-Za-z0-9_]*)  # property name
    \s*:\s*                       # colon
    ([^;}]+?)                     # value (up to semicolon or brace)
    \s*(?:;|$)                    # semicolon or end
    """,
    re.VERBOSE,
)


@dataclass
class StyleRule:
    """A single parsed stylesheet rule.

    Attributes:
        selector: The CSS-like selector string (e.g. '*', 'box', '.code', '#node_id').
        specificity: Numeric specificity for precedence ordering.
        properties: Map of recognized property names to their values.
    """

    selector: str
    specificity: int
    properties: dict[str, str] = field(default_factory=dict)


def parse_stylesheet(css: str) -> list[StyleRule]:
    """Parse a CSS-like stylesheet string into a list of StyleRules.

    Unrecognized properties are silently ignored. Empty or whitespace-only
    input returns an empty list. Parsing is safe — no eval.

    M-21: Bare identifiers (e.g. ``box``, ``parallelogram``) are parsed as
    shape-name selectors with specificity 1.

    Args:
        css: The stylesheet text to parse.

    Returns:
        Ordered list of StyleRules (order preserved from source).
    """
    if not css or not css.strip():
        return []

    rules: list[StyleRule] = []
    for match in _RULE_RE.finditer(css):
        selector = match.group(1).strip()
        decl_block = match.group(2)

        # Determine specificity from selector type
        # M-21: universal(0) < shape(1) < class(2) < id(3)
        if selector == "*":
            specificity = 0
        elif selector.startswith("."):
            specificity = 2
        elif selector.startswith("#"):
            specificity = 3
        else:
            # Bare identifier = shape-name selector (M-21)
            specificity = 1

        # Parse declarations
        properties: dict[str, str] = {}
        for decl_match in _DECL_RE.finditer(decl_block):
            prop_name = decl_match.group(1).strip()
            prop_value = decl_match.group(2).strip()
            if prop_name in _RECOGNIZED_PROPERTIES:
                properties[prop_name] = prop_value

        if properties:
            rules.append(
                StyleRule(
                    selector=selector,
                    specificity=specificity,
                    properties=properties,
                )
            )

    return rules


def apply_stylesheet(graph: Graph, rules: list[StyleRule]) -> Graph:
    """Apply stylesheet rules to all nodes in the graph.

    For each recognized property on each node, resolve the value using
    specificity ordering. Explicit node attributes always win (highest
    precedence). Later rules of equal specificity override earlier ones.

    Args:
        graph: The pipeline graph to transform.
        rules: Parsed stylesheet rules from parse_stylesheet().

    Returns:
        The same graph with node attrs updated in place.
    """
    if not rules:
        return graph

    for node in graph.nodes.values():
        # Build resolved properties: for each property, find the
        # highest-specificity matching rule. Among equal specificity,
        # later rules win (we iterate in order, overwriting).
        resolved: dict[str, tuple[int, str]] = {}  # prop -> (specificity, value)

        for rule in rules:
            if not _selector_matches(
                rule, node.id, node.attrs.get("class", ""), node.shape
            ):
                continue
            for prop, value in rule.properties.items():
                prev = resolved.get(prop)
                # Higher specificity wins; equal specificity → later wins
                if prev is None or rule.specificity >= prev[0]:
                    resolved[prop] = (rule.specificity, value)

        # Apply resolved properties, but only if node doesn't already
        # have an explicit attribute (node attrs always override)
        for prop, (_, value) in resolved.items():
            if prop not in node.attrs:
                node.attrs[prop] = value

    return graph


def _selector_matches(
    rule: StyleRule, node_id: str, node_class: str, node_shape: str
) -> bool:
    """Check if a rule's selector matches a node.

    Class selectors support comma-separated multi-class values:
    class="code,critical" matches both .code and .critical selectors.

    M-21: Bare identifiers match against the node's shape attribute.
    """
    sel = rule.selector
    if sel == "*":
        return True
    if sel.startswith("#"):
        return sel[1:] == node_id
    if sel.startswith("."):
        target_class = sel[1:]
        # Split comma-separated classes and strip whitespace
        node_classes = (
            {c.strip() for c in node_class.split(",")} if node_class else set()
        )
        return target_class in node_classes
    # M-21: bare identifier → shape-name selector
    return sel == node_shape
