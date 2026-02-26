"""Pipeline handler — DOT file path resolution.

Resolves dot_file paths by expanding $variable tokens from context,
then resolving absolute or relative paths against a source directory.
"""

from __future__ import annotations

import os
import re

from ..context import PipelineContext


def _expand_path_variables(path: str, context: PipelineContext) -> str:
    """Replace $variable tokens using context.get().

    Unknown $tokens are left unchanged. Context values are coerced to str.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = context.get(name)
        if value is None:
            return match.group(0)  # leave unknown token unchanged
        return str(value)

    return re.sub(r"\$(\w+)", _replace, path)


def resolve_dot_path(dot_file: str, source_dir: str, context: PipelineContext) -> str:
    """Resolve a dot_file path.

    1. Expand $variable tokens from context values.
    2. If path is absolute (starts with /), return as-is.
    3. Otherwise resolve relative to source_dir.
    4. If source_dir is empty, resolve relative to cwd.
    """
    expanded = _expand_path_variables(dot_file, context)

    if os.path.isabs(expanded):
        return expanded

    if source_dir:
        return os.path.join(source_dir, expanded)

    return os.path.join(os.getcwd(), expanded)
