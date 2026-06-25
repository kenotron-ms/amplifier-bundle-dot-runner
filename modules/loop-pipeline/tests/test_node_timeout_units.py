"""Unit tests for node timeout unit conversion (Bug 1).

The DOT parser stores all durations as milliseconds (dot_parser._DURATION_UNITS).
Engine and tool handler must divide node.timeout by 1000 to obtain seconds.

Spec: attractor-spec.md uses timeout_seconds.
Invariant: timeout="900s" in DOT -> 900-second effective asyncio.timeout.
"""

from amplifier_module_loop_pipeline.dot_parser import parse_dot


# ---------------------------------------------------------------------------
# Parser output: node.timeout stores milliseconds
# ---------------------------------------------------------------------------


def test_timeout_900s_stored_as_milliseconds():
    """Parser converts '900s' to 900000 ms stored on node.timeout."""
    graph = parse_dot('digraph { A [timeout="900s"] }')
    node = graph.nodes["A"]
    assert node.timeout == 900_000  # 900 * 1_000


def test_timeout_2m_stored_as_milliseconds():
    """Parser converts '2m' to 120000 ms stored on node.timeout."""
    graph = parse_dot('digraph { A [timeout="2m"] }')
    node = graph.nodes["A"]
    assert node.timeout == 120_000  # 2 * 60 * 1_000


def test_timeout_1h_stored_as_milliseconds():
    """Parser converts '1h' to 3600000 ms stored on node.timeout."""
    graph = parse_dot('digraph { A [timeout="1h"] }')
    node = graph.nodes["A"]
    assert node.timeout == 3_600_000  # 1 * 3600 * 1_000


# ---------------------------------------------------------------------------
# Consumer conversion: divide by 1000 to get effective seconds
# ---------------------------------------------------------------------------


def test_timeout_900s_yields_effective_900_seconds():
    """timeout='900s' in DOT must yield 900 effective seconds after /1000."""
    graph = parse_dot('digraph { A [timeout="900s"] }')
    node = graph.nodes["A"]
    assert node.timeout is not None, "node.timeout must be set"
    # This is the conversion the engine and tool handler now apply:
    timeout_s = float(node.timeout) / 1000.0
    assert timeout_s == 900.0


def test_timeout_2m_yields_effective_120_seconds():
    """timeout='2m' in DOT must yield 120 effective seconds after /1000."""
    graph = parse_dot('digraph { A [timeout="2m"] }')
    node = graph.nodes["A"]
    assert node.timeout is not None, "node.timeout must be set"
    timeout_s = float(node.timeout) / 1000.0
    assert timeout_s == 120.0


def test_timeout_30s_yields_effective_30_seconds():
    """timeout='30s' in DOT must yield 30 effective seconds after /1000."""
    graph = parse_dot('digraph { A [timeout="30s"] }')
    node = graph.nodes["A"]
    assert node.timeout is not None, "node.timeout must be set"
    timeout_s = float(node.timeout) / 1000.0
    assert timeout_s == 30.0


# ---------------------------------------------------------------------------
# Regression: max_pipeline_duration is in ms on BOTH sides — unchanged
# ---------------------------------------------------------------------------


def test_max_pipeline_duration_stored_as_milliseconds():
    """Graph-level max_pipeline_duration is stored in ms and compared in ms.

    This is intentionally ms on both sides (engine.py:277) and must NOT be
    divided — only the per-node timeout needs the /1000 conversion.
    """
    graph = parse_dot('digraph { max_pipeline_duration="5m" }')
    assert graph.max_pipeline_duration == 300_000  # 5 * 60 * 1_000 ms
