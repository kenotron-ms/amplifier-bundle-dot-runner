"""Tests for model stylesheet parsing and application.

Spec coverage: STYLE-001–007, Section 8.
"""

from amplifier_module_loop_pipeline.graph import Edge, Graph, Node
from amplifier_module_loop_pipeline.stylesheet import (
    apply_stylesheet,
    parse_stylesheet,
)


# --- Parsing ---


def test_parse_universal_selector():
    """Universal selector '*' has specificity 0."""
    rules = parse_stylesheet("* { llm_model: claude-sonnet-4-5; }")
    assert len(rules) == 1
    assert rules[0].selector == "*"
    assert rules[0].specificity == 0
    assert rules[0].properties == {"llm_model": "claude-sonnet-4-5"}


def test_parse_class_selector():
    """Class selector '.code' has specificity 2 (M-21: universal<shape<class<id)."""
    rules = parse_stylesheet(".code { llm_model: claude-opus-4-6; }")
    assert len(rules) == 1
    assert rules[0].selector == ".code"
    assert rules[0].specificity == 2
    assert rules[0].properties == {"llm_model": "claude-opus-4-6"}


def test_parse_id_selector():
    """ID selector '#critical_review' has specificity 3 (M-21: universal<shape<class<id)."""
    rules = parse_stylesheet(
        "#critical_review { llm_model: gpt-5.2; llm_provider: openai; }"
    )
    assert len(rules) == 1
    assert rules[0].selector == "#critical_review"
    assert rules[0].specificity == 3
    assert rules[0].properties == {
        "llm_model": "gpt-5.2",
        "llm_provider": "openai",
    }


def test_parse_multiple_rules():
    """Multiple rules parsed in order."""
    css = """
        * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
        .code { llm_model: claude-opus-4-6; }
        #critical_review { llm_model: gpt-5.2; llm_provider: openai; reasoning_effort: high; }
    """
    rules = parse_stylesheet(css)
    assert len(rules) == 3
    assert rules[0].selector == "*"
    assert rules[1].selector == ".code"
    assert rules[2].selector == "#critical_review"


def test_parse_reasoning_effort_property():
    """reasoning_effort is a recognized property."""
    rules = parse_stylesheet("* { reasoning_effort: high; }")
    assert rules[0].properties == {"reasoning_effort": "high"}


def test_parse_multiple_declarations():
    """Multiple declarations in one rule."""
    rules = parse_stylesheet(
        ".code { llm_model: claude-opus-4-6; llm_provider: anthropic; reasoning_effort: medium; }"
    )
    assert rules[0].properties == {
        "llm_model": "claude-opus-4-6",
        "llm_provider": "anthropic",
        "reasoning_effort": "medium",
    }


def test_parse_empty_string():
    """Empty stylesheet returns no rules."""
    assert parse_stylesheet("") == []


def test_parse_whitespace_only():
    """Whitespace-only stylesheet returns no rules."""
    assert parse_stylesheet("   \n\t  ") == []


def test_parse_trailing_semicolon_optional():
    """Trailing semicolon in declaration block is optional."""
    rules = parse_stylesheet("* { llm_model: gpt-4 }")
    assert len(rules) == 1
    assert rules[0].properties == {"llm_model": "gpt-4"}


def test_parse_ignores_unrecognized_properties():
    """Unrecognized properties are silently ignored."""
    rules = parse_stylesheet("* { llm_model: gpt-4; unknown_prop: value; }")
    assert rules[0].properties == {"llm_model": "gpt-4"}


# --- Application ---


def _make_graph_with_stylesheet(
    nodes: dict[str, Node],
    edges: list[Edge] | None = None,
    stylesheet: str = "",
) -> Graph:
    return Graph(
        name="test",
        nodes=nodes,
        edges=edges or [],
        model_stylesheet=stylesheet,
    )


def test_apply_universal_rule():
    """Universal rule applies to all nodes."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "plan": Node(id="plan", prompt="Plan"),
            "exit": Node(id="exit", shape="Msquare"),
        },
        stylesheet="* { llm_model: claude-sonnet-4-5; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["plan"].attrs.get("llm_model") == "claude-sonnet-4-5"
    assert result.nodes["start"].attrs.get("llm_model") == "claude-sonnet-4-5"
    assert result.nodes["exit"].attrs.get("llm_model") == "claude-sonnet-4-5"


def test_apply_class_selector():
    """Class selector applies to nodes with matching class."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "plan": Node(id="plan", prompt="Plan", attrs={"class": "planning"}),
            "impl": Node(id="impl", prompt="Implement", attrs={"class": "code"}),
        },
        stylesheet=".code { llm_model: claude-opus-4-6; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("llm_model") == "claude-opus-4-6"
    assert result.nodes["plan"].attrs.get("llm_model") is None


def test_apply_id_selector():
    """ID selector applies only to the node with that ID."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "review": Node(id="review", prompt="Review"),
            "critical_review": Node(id="critical_review", prompt="Critical"),
        },
        stylesheet="#critical_review { llm_model: gpt-5.2; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["critical_review"].attrs.get("llm_model") == "gpt-5.2"
    assert result.nodes["review"].attrs.get("llm_model") is None


def test_specificity_id_overrides_class():
    """ID selector (specificity 3) overrides class (specificity 2)."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "critical_review": Node(
                id="critical_review", prompt="Critical", attrs={"class": "code"}
            ),
        },
        stylesheet="""
            .code { llm_model: claude-opus-4-6; llm_provider: anthropic; }
            #critical_review { llm_model: gpt-5.2; llm_provider: openai; }
        """,
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["critical_review"].attrs.get("llm_model") == "gpt-5.2"
    assert result.nodes["critical_review"].attrs.get("llm_provider") == "openai"


def test_specificity_class_overrides_universal():
    """Class selector (specificity 2) overrides universal (specificity 0)."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(id="impl", prompt="Build", attrs={"class": "code"}),
            "plan": Node(id="plan", prompt="Plan"),
        },
        stylesheet="""
            * { llm_model: claude-sonnet-4-5; }
            .code { llm_model: claude-opus-4-6; }
        """,
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("llm_model") == "claude-opus-4-6"
    assert result.nodes["plan"].attrs.get("llm_model") == "claude-sonnet-4-5"


def test_explicit_node_attrs_override_stylesheet():
    """Explicit node attributes always override stylesheet values."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(
                id="impl",
                prompt="Build",
                attrs={"llm_model": "my-custom-model"},
            ),
        },
        stylesheet="* { llm_model: claude-sonnet-4-5; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    # Explicit node attribute wins
    assert result.nodes["impl"].attrs.get("llm_model") == "my-custom-model"


def test_later_rules_same_specificity_override():
    """Later rules of equal specificity override earlier ones (STYLE-003)."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(id="impl", prompt="Build", attrs={"class": "code"}),
        },
        stylesheet="""
            .code { llm_model: model-a; }
            .code { llm_model: model-b; }
        """,
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("llm_model") == "model-b"


def test_spec_example():
    """Full example from spec Section 8.6."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "exit": Node(id="exit", shape="Msquare"),
            "plan": Node(id="plan", label="Plan", attrs={"class": "planning"}),
            "implement": Node(
                id="implement", label="Implement", attrs={"class": "code"}
            ),
            "critical_review": Node(
                id="critical_review", label="Critical Review", attrs={"class": "code"}
            ),
        },
        stylesheet="""
            * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
            .code { llm_model: claude-opus-4-6; llm_provider: anthropic; }
            #critical_review { llm_model: gpt-5.2; llm_provider: openai; reasoning_effort: high; }
        """,
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)

    # plan gets claude-sonnet-4-5 from the * rule (no class match for .code)
    assert result.nodes["plan"].attrs.get("llm_model") == "claude-sonnet-4-5"
    assert result.nodes["plan"].attrs.get("llm_provider") == "anthropic"

    # implement gets claude-opus-4-6 from the .code rule
    assert result.nodes["implement"].attrs.get("llm_model") == "claude-opus-4-6"
    assert result.nodes["implement"].attrs.get("llm_provider") == "anthropic"

    # critical_review gets gpt-5.2 from the #critical_review rule
    assert result.nodes["critical_review"].attrs.get("llm_model") == "gpt-5.2"
    assert result.nodes["critical_review"].attrs.get("llm_provider") == "openai"
    assert result.nodes["critical_review"].attrs.get("reasoning_effort") == "high"


def test_apply_empty_rules():
    """Empty rules list leaves graph unchanged."""
    graph = _make_graph_with_stylesheet(
        nodes={"plan": Node(id="plan", prompt="Plan")},
    )
    result = apply_stylesheet(graph, [])
    assert result.nodes["plan"].attrs.get("llm_model") is None


def test_partial_property_override():
    """Higher-specificity rule overrides only the properties it declares."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(id="impl", prompt="Build", attrs={"class": "code"}),
        },
        stylesheet="""
            * { llm_model: default-model; llm_provider: default-provider; reasoning_effort: low; }
            .code { llm_model: code-model; }
        """,
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    # .code overrides llm_model only
    assert result.nodes["impl"].attrs.get("llm_model") == "code-model"
    # llm_provider and reasoning_effort still come from *
    assert result.nodes["impl"].attrs.get("llm_provider") == "default-provider"
    assert result.nodes["impl"].attrs.get("reasoning_effort") == "low"


# --- Multi-class matching (1b3) ---


def test_class_selector_matches_comma_separated_classes():
    """`.code` selector must match nodes with class="code,critical"."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(
                id="impl", prompt="Build", attrs={"class": "code,critical"}
            ),
        },
        stylesheet=".code { llm_model: claude-opus-4-6; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("llm_model") == "claude-opus-4-6"


def test_class_selector_matches_second_class_in_list():
    """`.critical` selector must match nodes with class="code,critical"."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(
                id="impl", prompt="Build", attrs={"class": "code,critical"}
            ),
        },
        stylesheet=".critical { reasoning_effort: high; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("reasoning_effort") == "high"


def test_class_selector_no_match_on_multi_class():
    """`.planning` selector must NOT match nodes with class="code,critical"."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(
                id="impl", prompt="Build", attrs={"class": "code,critical"}
            ),
        },
        stylesheet=".planning { llm_model: planning-model; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("llm_model") is None


def test_multi_class_with_spaces_around_commas():
    """class="code, critical" (spaces) still matches .code and .critical."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(
                id="impl", prompt="Build", attrs={"class": "code, critical"}
            ),
        },
        stylesheet=".critical { reasoning_effort: high; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("reasoning_effort") == "high"


def test_single_class_still_works():
    """Single class value still matches normally (regression check)."""
    graph = _make_graph_with_stylesheet(
        nodes={
            "impl": Node(id="impl", prompt="Build", attrs={"class": "code"}),
        },
        stylesheet=".code { llm_model: code-model; }",
    )
    rules = parse_stylesheet(graph.model_stylesheet)
    result = apply_stylesheet(graph, rules)
    assert result.nodes["impl"].attrs.get("llm_model") == "code-model"
