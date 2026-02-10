"""Tests for the PipelineContext store.

Covers spec Section 5.1 (Context): thread-safe key-value store with
namespaces, clone, snapshot, and update operations.
"""

import threading

from amplifier_module_loop_pipeline.context import PipelineContext


# --- Basic get/set ---


def test_context_set_get():
    """Basic set and get."""
    ctx = PipelineContext()
    ctx.set("outcome", "success")
    assert ctx.get("outcome") == "success"


def test_context_get_missing_returns_default():
    """get() returns default for missing keys."""
    ctx = PipelineContext()
    assert ctx.get("missing") is None
    assert ctx.get("missing", "fallback") == "fallback"


def test_context_get_string():
    """get_string() returns string values with default."""
    ctx = PipelineContext()
    ctx.set("count", 42)
    assert ctx.get_string("count") == "42"
    assert ctx.get_string("missing") == ""
    assert ctx.get_string("missing", "n/a") == "n/a"


def test_context_overwrite():
    """Setting the same key overwrites the previous value."""
    ctx = PipelineContext()
    ctx.set("key", "v1")
    ctx.set("key", "v2")
    assert ctx.get("key") == "v2"


# --- update (bulk) ---


def test_context_update():
    """update() merges multiple key-value pairs."""
    ctx = PipelineContext()
    ctx.set("existing", "keep")
    ctx.update({"new_key": "new_val", "existing": "replaced"})
    assert ctx.get("new_key") == "new_val"
    assert ctx.get("existing") == "replaced"


# --- snapshot ---


def test_context_snapshot():
    """snapshot() returns a copy of all values."""
    ctx = PipelineContext()
    ctx.set("key", "value")
    ctx.set("count", 42)
    snap = ctx.snapshot()
    assert snap == {"key": "value", "count": 42}


def test_context_snapshot_is_isolated():
    """Modifying the snapshot should not affect the context."""
    ctx = PipelineContext()
    ctx.set("key", "value")
    snap = ctx.snapshot()
    snap["key"] = "modified"
    assert ctx.get("key") == "value"  # original unchanged


# --- clone ---


def test_context_clone_is_isolated():
    """clone() creates an independent copy for parallel branches."""
    ctx = PipelineContext()
    ctx.set("key", "original")
    clone = ctx.clone()
    clone.set("key", "modified")
    assert ctx.get("key") == "original"  # parent unchanged
    assert clone.get("key") == "modified"


def test_context_clone_copies_all_values():
    """clone() copies all existing values."""
    ctx = PipelineContext()
    ctx.set("a", 1)
    ctx.set("b", 2)
    clone = ctx.clone()
    assert clone.get("a") == 1
    assert clone.get("b") == 2


def test_context_clone_new_keys_dont_leak():
    """New keys in clone should not appear in parent."""
    ctx = PipelineContext()
    clone = ctx.clone()
    clone.set("new_key", "value")
    assert ctx.get("new_key") is None


# --- Namespace conventions ---


def test_context_namespace_keys():
    """Context supports namespaced keys (dot-separated)."""
    ctx = PipelineContext()
    ctx.set("context.last_stage", "implement")
    ctx.set("graph.goal", "Build feature")
    ctx.set("internal.retry_count.step1", 3)
    ctx.set("work.item_id", "task-42")

    assert ctx.get("context.last_stage") == "implement"
    assert ctx.get("graph.goal") == "Build feature"
    assert ctx.get("internal.retry_count.step1") == 3
    assert ctx.get("work.item_id") == "task-42"


# --- Thread safety ---


def test_context_thread_safety():
    """Concurrent writes from multiple threads should not corrupt data."""
    ctx = PipelineContext()
    errors: list[str] = []

    def writer(prefix: str, count: int) -> None:
        try:
            for i in range(count):
                ctx.set(f"{prefix}.{i}", i)
        except Exception as e:
            errors.append(str(e))

    threads = [
        threading.Thread(target=writer, args=(f"t{t}", 100))
        for t in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    # Verify all values were written
    for t in range(10):
        for i in range(100):
            assert ctx.get(f"t{t}.{i}") == i


# --- append_log ---


def test_context_append_log():
    """append_log adds entries to an append-only log."""
    ctx = PipelineContext()
    ctx.append_log("Started pipeline")
    ctx.append_log("Node 'plan' completed")
    logs = ctx.get_logs()
    assert len(logs) == 2
    assert logs[0] == "Started pipeline"
    assert logs[1] == "Node 'plan' completed"


def test_context_clone_copies_logs():
    """clone() copies the log history."""
    ctx = PipelineContext()
    ctx.append_log("entry1")
    clone = ctx.clone()
    clone.append_log("entry2")
    assert len(ctx.get_logs()) == 1  # parent unchanged
    assert len(clone.get_logs()) == 2
