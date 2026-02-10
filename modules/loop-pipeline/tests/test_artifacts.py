"""Tests for the artifact store.

Spec coverage: ART-001–004, Section 5.5.
"""

from amplifier_module_loop_pipeline.artifacts import Artifact, ArtifactStore


class TestArtifactDataclass:
    """Tests for the Artifact dataclass."""

    def test_artifact_has_required_fields(self):
        """Artifact stores name, type, data, size, and timestamp."""
        a = Artifact(
            name="test",
            artifact_type="text",
            data="hello",
            size=5,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert a.name == "test"
        assert a.artifact_type == "text"
        assert a.data == "hello"
        assert a.size == 5
        assert a.timestamp == "2025-01-01T00:00:00Z"

    def test_artifact_is_file_backed_default_false(self):
        """Artifact defaults to not file-backed."""
        a = Artifact(
            name="test",
            artifact_type="text",
            data="hello",
            size=5,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert a.is_file_backed is False

    def test_artifact_is_file_backed_can_be_set(self):
        """Artifact can be marked as file-backed."""
        a = Artifact(
            name="test",
            artifact_type="text",
            data="/path/to/file",
            size=200_000,
            timestamp="2025-01-01T00:00:00Z",
            is_file_backed=True,
        )
        assert a.is_file_backed is True


class TestArtifactStoreBasics:
    """Tests for basic ArtifactStore operations (in-memory, small artifacts)."""

    def test_store_and_get_small_text(self, tmp_path):
        """Store a small text artifact and retrieve it."""
        store = ArtifactStore(base_dir=str(tmp_path))
        artifact = store.store("greeting", "hello world", artifact_type="text")
        assert artifact.name == "greeting"
        assert artifact.artifact_type == "text"
        assert artifact.size == len("hello world")
        assert artifact.is_file_backed is False

        retrieved = store.get("greeting")
        assert retrieved is not None
        assert retrieved.data == "hello world"

    def test_store_and_get_json_artifact(self, tmp_path):
        """Store a JSON dict artifact and retrieve it."""
        store = ArtifactStore(base_dir=str(tmp_path))
        data = {"key": "value", "count": 42}
        artifact = store.store("config", data, artifact_type="json")
        assert artifact.artifact_type == "json"

        retrieved = store.get("config")
        assert retrieved is not None
        assert retrieved.data == {"key": "value", "count": 42}

    def test_get_missing_returns_none(self, tmp_path):
        """Getting a non-existent artifact returns None."""
        store = ArtifactStore(base_dir=str(tmp_path))
        assert store.get("nonexistent") is None

    def test_list_returns_artifact_names(self, tmp_path):
        """List returns names of all stored artifacts."""
        store = ArtifactStore(base_dir=str(tmp_path))
        store.store("alpha", "aaa", artifact_type="text")
        store.store("beta", "bbb", artifact_type="text")
        store.store("gamma", "ccc", artifact_type="text")
        names = store.list()
        assert sorted(names) == ["alpha", "beta", "gamma"]

    def test_list_empty_store(self, tmp_path):
        """List on empty store returns empty list."""
        store = ArtifactStore(base_dir=str(tmp_path))
        assert store.list() == []

    def test_store_overwrites_existing(self, tmp_path):
        """Storing with same name replaces the artifact."""
        store = ArtifactStore(base_dir=str(tmp_path))
        store.store("item", "old data", artifact_type="text")
        store.store("item", "new data", artifact_type="text")
        retrieved = store.get("item")
        assert retrieved is not None
        assert retrieved.data == "new data"

    def test_default_type_is_text(self, tmp_path):
        """Default artifact_type is 'text'."""
        store = ArtifactStore(base_dir=str(tmp_path))
        artifact = store.store("simple", "data")
        assert artifact.artifact_type == "text"

    def test_artifact_has_timestamp(self, tmp_path):
        """Stored artifact has a timestamp."""
        store = ArtifactStore(base_dir=str(tmp_path))
        artifact = store.store("timed", "data")
        assert artifact.timestamp  # non-empty string
        assert len(artifact.timestamp) > 0


class TestArtifactStoreFileBacking:
    """Tests for file-backed artifacts (>100KB threshold)."""

    def _make_large_data(self, size_bytes: int = 150_000) -> str:
        """Create a string larger than the 100KB threshold."""
        return "x" * size_bytes

    def test_large_artifact_is_file_backed(self, tmp_path):
        """Artifacts >100KB are written to disk."""
        store = ArtifactStore(base_dir=str(tmp_path))
        large_data = self._make_large_data()
        artifact = store.store("big", large_data, artifact_type="text")
        assert artifact.is_file_backed is True
        assert artifact.size == len(large_data)

    def test_large_artifact_written_to_disk(self, tmp_path):
        """File-backed artifact creates a file in {base_dir}/artifacts/."""
        store = ArtifactStore(base_dir=str(tmp_path))
        large_data = self._make_large_data()
        store.store("big", large_data, artifact_type="text")

        artifact_path = tmp_path / "artifacts" / "big.json"
        assert artifact_path.exists()

    def test_large_artifact_retrievable(self, tmp_path):
        """File-backed artifacts can be retrieved with correct data."""
        store = ArtifactStore(base_dir=str(tmp_path))
        large_data = self._make_large_data(120_000)
        store.store("big", large_data, artifact_type="text")

        retrieved = store.get("big")
        assert retrieved is not None
        assert retrieved.data == large_data

    def test_small_artifact_not_written_to_disk(self, tmp_path):
        """Artifacts under 100KB are NOT written to disk."""
        store = ArtifactStore(base_dir=str(tmp_path))
        store.store("small", "tiny data", artifact_type="text")

        artifacts_dir = tmp_path / "artifacts"
        if artifacts_dir.exists():
            assert not (artifacts_dir / "small.json").exists()

    def test_exactly_100kb_stays_in_memory(self, tmp_path):
        """Artifact of exactly 100KB (102400 bytes) is not file-backed."""
        store = ArtifactStore(base_dir=str(tmp_path))
        # Exactly 100KB = 102400 bytes
        data = "x" * 102_400
        artifact = store.store("boundary", data, artifact_type="text")
        assert artifact.is_file_backed is False

    def test_over_100kb_is_file_backed(self, tmp_path):
        """Artifact of 100KB + 1 byte IS file-backed."""
        store = ArtifactStore(base_dir=str(tmp_path))
        data = "x" * 102_401
        artifact = store.store("over", data, artifact_type="text")
        assert artifact.is_file_backed is True

    def test_large_json_artifact_roundtrip(self, tmp_path):
        """Large JSON data round-trips through file backing."""
        store = ArtifactStore(base_dir=str(tmp_path))
        # Create a large JSON-serializable dict
        large_data = {"items": ["item_" + str(i) for i in range(20_000)]}
        store.store("big_json", large_data, artifact_type="json")

        retrieved = store.get("big_json")
        assert retrieved is not None
        assert retrieved.data == large_data

    def test_large_binary_bytes_artifact(self, tmp_path):
        """Large binary (bytes) artifacts are file-backed."""
        store = ArtifactStore(base_dir=str(tmp_path))
        data = b"\x00" * 150_000
        artifact = store.store("binary", data, artifact_type="binary")
        assert artifact.is_file_backed is True

        retrieved = store.get("binary")
        assert retrieved is not None
        assert retrieved.data == data
