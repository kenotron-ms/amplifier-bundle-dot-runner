"""Tests for pipeline handler — DOT file path resolution.

Spec coverage: resolve_dot_path and _expand_path_variables.
"""

import os

from amplifier_module_loop_pipeline.context import PipelineContext
from amplifier_module_loop_pipeline.handlers.pipeline import (
    resolve_dot_path,
)


class TestResolveDotPath:
    """Tests for resolve_dot_path() and _expand_path_variables()."""

    def test_absolute_path_unchanged(self) -> None:
        """Absolute paths are returned as-is."""
        ctx = PipelineContext()
        result = resolve_dot_path("/abs/child.dot", source_dir="/parent", context=ctx)
        assert result == "/abs/child.dot"

    def test_relative_to_source_dir(self) -> None:
        """Relative paths resolve against source_dir."""
        ctx = PipelineContext()
        result = resolve_dot_path(
            "child.dot", source_dir="/parent/pipelines", context=ctx
        )
        assert result == "/parent/pipelines/child.dot"

    def test_relative_subdirectory(self) -> None:
        """Subdirectory paths resolve correctly."""
        ctx = PipelineContext()
        result = resolve_dot_path("sub/child.dot", source_dir="/parent", context=ctx)
        assert result == "/parent/sub/child.dot"

    def test_variable_expansion(self) -> None:
        """$language token is expanded from context."""
        ctx = PipelineContext()
        ctx.set("language", "python")
        result = resolve_dot_path(
            "$language/tasks.dot", source_dir="/pipelines", context=ctx
        )
        assert result == "/pipelines/python/tasks.dot"

    def test_variable_expansion_then_absolute(self) -> None:
        """If expansion produces an absolute path, use it."""
        ctx = PipelineContext()
        ctx.set("base", "/absolute/root")
        result = resolve_dot_path("$base/child.dot", source_dir="/parent", context=ctx)
        assert result == "/absolute/root/child.dot"

    def test_empty_source_dir_uses_cwd(self) -> None:
        """Empty source_dir falls back to os.getcwd()."""
        ctx = PipelineContext()
        result = resolve_dot_path("child.dot", source_dir="", context=ctx)
        assert result == os.path.join(os.getcwd(), "child.dot")

    def test_no_variable_in_path(self) -> None:
        """Paths without $ are not modified by context."""
        ctx = PipelineContext()
        ctx.set("language", "python")
        result = resolve_dot_path("plain/child.dot", source_dir="/parent", context=ctx)
        assert result == "/parent/plain/child.dot"

    def test_unknown_variable_left_unchanged(self) -> None:
        """Unknown $tokens survive expansion unchanged."""
        ctx = PipelineContext()
        ctx.set("language", "python")
        result = resolve_dot_path(
            "$unknown/$language/child.dot", source_dir="/parent", context=ctx
        )
        assert result == "/parent/$unknown/python/child.dot"
