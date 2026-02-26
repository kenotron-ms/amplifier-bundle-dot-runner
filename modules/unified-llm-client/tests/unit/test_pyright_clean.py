"""Regression test: pyright must report zero errors on unified_llm/."""

import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_pyright_zero_errors():
    """Run pyright on the unified_llm package and assert zero errors."""
    result = subprocess.run(
        ["pyright", "unified_llm/"],
        capture_output=True,
        text=True,
        cwd="/home/bkrabach/dev/attractor-next/unified-llm-client",
    )
    # pyright exits 0 on success, 1 on errors
    assert result.returncode == 0, (
        f"pyright reported errors:\n{result.stdout}\n{result.stderr}"
    )
