"""Artifact store for pipeline execution outputs.

Named, typed storage for large outputs produced during pipeline execution.
Small artifacts (<= 100KB) are held in memory; large artifacts (> 100KB)
are written to disk at ``{base_dir}/artifacts/{name}.json``.

Spec coverage: ART-001–004, Section 5.5.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# 100 KB = 102400 bytes.  Artifacts *above* this are file-backed.
FILE_BACKING_THRESHOLD = 102_400


@dataclass
class Artifact:
    """Metadata and data for a single stored artifact.

    Spec Section 5.5: ArtifactInfo + stored data.
    """

    name: str
    artifact_type: str
    data: Any
    size: int
    timestamp: str
    is_file_backed: bool = False


class ArtifactStore:
    """Named, typed artifact storage with file-backing for large items.

    Artifacts whose serialised size exceeds ``FILE_BACKING_THRESHOLD``
    are written to ``{base_dir}/artifacts/{name}.json`` instead of being
    held in memory.

    Spec Section 5.5: ArtifactStore.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir
        self._artifacts: dict[str, Artifact] = {}
        self._lock = asyncio.Lock()  # L-12: thread safety for parallel handlers

    # -- public API ----------------------------------------------------------

    def store(
        self,
        name: str | None = None,
        data: Any = None,
        artifact_type: str = "text",
        *,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Store an artifact, file-backing if over the size threshold.

        Args:
            name: Artifact identifier (positional, backward-compat).
            data: The data to store.
            artifact_type: Type label (default ``"text"``).
            artifact_id: Spec-preferred identifier (L-13). Alias for *name*.

        Returns the ``Artifact`` metadata object.
        """
        name = artifact_id or name
        if name is None:
            raise TypeError("store() requires 'name' or 'artifact_id'")
        size = _byte_size(data)
        is_file_backed = size > FILE_BACKING_THRESHOLD
        timestamp = datetime.now(timezone.utc).isoformat()

        if is_file_backed:
            path = self._artifact_path(name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _write_json(path, data)
            stored_data = path  # keep the *path* in memory
        else:
            stored_data = data

        artifact = Artifact(
            name=name,
            artifact_type=artifact_type,
            data=stored_data,
            size=size,
            timestamp=timestamp,
            is_file_backed=is_file_backed,
        )
        self._artifacts[name] = artifact
        return artifact

    def get(
        self, name: str | None = None, *, artifact_id: str | None = None
    ) -> Artifact | None:
        """Retrieve an artifact by name, reading from disk if file-backed.

        Args:
            name: Artifact identifier (positional, backward-compat).
            artifact_id: Spec-preferred identifier (L-13). Alias for *name*.

        Returns ``None`` if the artifact does not exist.
        """
        name = artifact_id or name
        if name is None:
            raise TypeError("get() requires 'name' or 'artifact_id'")
        artifact = self._artifacts.get(name)
        if artifact is None:
            return None

        if artifact.is_file_backed:
            # Return a copy with the actual data loaded from disk
            loaded_data = _read_json(artifact.data)  # .data is the file path
            return Artifact(
                name=artifact.name,
                artifact_type=artifact.artifact_type,
                data=loaded_data,
                size=artifact.size,
                timestamp=artifact.timestamp,
                is_file_backed=artifact.is_file_backed,
            )

        return artifact

    def list(self) -> list[str]:
        """Return the names of all stored artifacts."""
        return list(self._artifacts.keys())

    def has(self, name: str | None = None, *, artifact_id: str | None = None) -> bool:
        """Check whether an artifact with *name* exists.

        Args:
            name: Artifact identifier (positional, backward-compat).
            artifact_id: Spec-preferred identifier (L-13). Alias for *name*.

        L-10: Spec-required ``has(name)`` method.
        """
        name = artifact_id or name
        if name is None:
            raise TypeError("has() requires 'name' or 'artifact_id'")
        return name in self._artifacts

    def remove(
        self, name: str | None = None, *, artifact_id: str | None = None
    ) -> None:
        """Remove an artifact by name.

        Args:
            name: Artifact identifier (positional, backward-compat).
            artifact_id: Spec-preferred identifier (L-13). Alias for *name*.

        If the artifact is file-backed, the backing file is also deleted.
        Removing a non-existent artifact is a no-op.

        L-10: Spec-required ``remove(name)`` method.
        """
        name = artifact_id or name
        if name is None:
            raise TypeError("remove() requires 'name' or 'artifact_id'")
        artifact = self._artifacts.pop(name, None)
        if artifact is not None and artifact.is_file_backed:
            path = artifact.data  # .data holds the file path for file-backed
            if isinstance(path, str) and os.path.exists(path):
                os.remove(path)

    def clear(self) -> None:
        """Remove all artifacts.

        File-backed artifacts have their backing files deleted.

        L-10: Spec-required ``clear()`` method.
        """
        for artifact in self._artifacts.values():
            if artifact.is_file_backed:
                path = artifact.data
                if isinstance(path, str) and os.path.exists(path):
                    os.remove(path)
        self._artifacts.clear()

    # -- L-12: async wrappers with lock for parallel handler safety ----------

    async def async_store(
        self,
        name: str,
        data: Any,
        artifact_type: str = "text",
    ) -> Artifact:
        """Thread-safe async version of :meth:`store`."""
        async with self._lock:
            return self.store(name, data, artifact_type)

    async def async_get(self, name: str) -> Artifact | None:
        """Thread-safe async version of :meth:`get`."""
        async with self._lock:
            return self.get(name)

    async def async_remove(self, name: str) -> None:
        """Thread-safe async version of :meth:`remove`."""
        async with self._lock:
            self.remove(name)

    async def async_clear(self) -> None:
        """Thread-safe async version of :meth:`clear`."""
        async with self._lock:
            self.clear()

    # -- internal helpers ----------------------------------------------------

    def _artifact_path(self, name: str) -> str:
        """Return the on-disk path for a file-backed artifact."""
        return os.path.join(self._base_dir, "artifacts", f"{name}.json")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _byte_size(data: Any) -> int:
    """Estimate the byte size of *data*.

    - ``str``: length in UTF-8 bytes.
    - ``bytes`` / ``bytearray``: raw length.
    - Anything else: length of its JSON serialisation.
    """
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    return len(json.dumps(data, default=str).encode("utf-8"))


def _write_json(path: str, data: Any) -> None:
    """Serialise *data* to a JSON file.

    ``bytes`` and ``bytearray`` are stored as a JSON object with a
    ``__bytes__`` key so they can round-trip through ``_read_json``.
    """
    if isinstance(data, (bytes, bytearray)):
        payload: Any = {"__bytes__": list(data)}
    else:
        payload = data
    with open(path, "w") as f:
        json.dump(payload, f, default=str)


def _read_json(path: str) -> Any:
    """Read a JSON file written by ``_write_json``."""
    with open(path) as f:
        payload = json.load(f)
    # Reconstruct bytes if needed
    if isinstance(payload, dict) and "__bytes__" in payload:
        return bytes(payload["__bytes__"])
    return payload
