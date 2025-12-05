"""Unit tests for data source adapters."""

import json
from unittest.mock import MagicMock, patch


from loft.autonomous.data_sources import (
    CaseData,
    CompositeAdapter,
    CourtListenerAdapter,
    LocalFileAdapter,
    create_courtlistener_adapter,
    create_local_adapter,
)


class TestCaseData:
    """Tests for CaseData dataclass."""

    def test_default_values(self):
        """Test default values for CaseData."""
        case = CaseData(id="test_1", domain="contracts", facts="Test facts")

        assert case.id == "test_1"
        assert case.domain == "contracts"
        assert case.facts == "Test facts"
        assert case.text == ""
        assert case.metadata == {}

    def test_with_metadata(self):
        """Test CaseData with metadata."""
        metadata = {"case_name": "Doe v. Smith", "court": "Supreme Court"}
        case = CaseData(
            id="test_2",
            domain="torts",
            facts="Negligence claim",
            text="Full case text",
            metadata=metadata,
        )

        assert case.metadata == metadata
        assert case.text == "Full case text"

    def test_to_dict(self):
        """Test converting CaseData to dictionary."""
        case = CaseData(
            id="test_3",
            domain="property",
            facts="Property dispute",
            metadata={"court": "District Court"},
        )

        result = case.to_dict()

        assert result["id"] == "test_3"
        assert result["domain"] == "property"
        assert result["facts"] == "Property dispute"
        assert result["court"] == "District Court"

    def test_to_dict_metadata_merged(self):
        """Test that metadata is merged into dict."""
        case = CaseData(
            id="case_1",
            domain="contracts",
            facts="Test",
            metadata={"extra_field": "value", "another": 123},
        )

        result = case.to_dict()

        assert "extra_field" in result
        assert result["extra_field"] == "value"
        assert result["another"] == 123


class TestLocalFileAdapter:
    """Tests for LocalFileAdapter."""

    def test_load_json_array(self, tmp_path):
        """Test loading cases from JSON array."""
        cases = [
            {"id": "1", "domain": "contracts", "facts": "Case 1"},
            {"id": "2", "domain": "torts", "facts": "Case 2"},
        ]

        file_path = tmp_path / "cases.json"
        with open(file_path, "w") as f:
            json.dump(cases, f)

        adapter = LocalFileAdapter([str(file_path)])
        loaded = list(adapter.get_cases())

        assert len(loaded) == 2
        assert loaded[0].id == "1"
        assert loaded[1].id == "2"

    def test_load_json_with_cases_key(self, tmp_path):
        """Test loading cases from JSON with 'cases' key."""
        data = {
            "metadata": {"source": "test"},
            "cases": [
                {"id": "1", "domain": "contracts", "facts": "Case 1"},
            ],
        }

        file_path = tmp_path / "cases.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        adapter = LocalFileAdapter([str(file_path)])
        loaded = list(adapter.get_cases())

        assert len(loaded) == 1
        assert loaded[0].id == "1"

    def test_load_jsonl(self, tmp_path):
        """Test loading cases from JSONL file."""
        file_path = tmp_path / "cases.jsonl"
        with open(file_path, "w") as f:
            f.write('{"id": "1", "domain": "contracts", "facts": "Case 1"}\n')
            f.write('{"id": "2", "domain": "torts", "facts": "Case 2"}\n')

        adapter = LocalFileAdapter([str(file_path)])
        loaded = list(adapter.get_cases())

        assert len(loaded) == 2

    def test_load_from_directory(self, tmp_path):
        """Test loading cases from directory."""
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        with open(cases_dir / "batch1.json", "w") as f:
            json.dump([{"id": "1", "domain": "contracts", "facts": "C1"}], f)

        with open(cases_dir / "batch2.json", "w") as f:
            json.dump([{"id": "2", "domain": "torts", "facts": "C2"}], f)

        adapter = LocalFileAdapter([str(cases_dir)])
        loaded = list(adapter.get_cases())

        assert len(loaded) == 2

    def test_get_case_count(self, tmp_path):
        """Test getting case count."""
        cases = [{"id": str(i), "facts": f"Case {i}"} for i in range(5)]

        file_path = tmp_path / "cases.json"
        with open(file_path, "w") as f:
            json.dump(cases, f)

        adapter = LocalFileAdapter([str(file_path)])

        assert adapter.get_case_count() == 5

    def test_limit_cases(self, tmp_path):
        """Test limiting number of cases returned."""
        cases = [{"id": str(i), "facts": f"Case {i}"} for i in range(10)]

        file_path = tmp_path / "cases.json"
        with open(file_path, "w") as f:
            json.dump(cases, f)

        adapter = LocalFileAdapter([str(file_path)])
        loaded = list(adapter.get_cases(limit=3))

        assert len(loaded) == 3

    def test_source_name(self, tmp_path):
        """Test source name property."""
        adapter = LocalFileAdapter([str(tmp_path / "a"), str(tmp_path / "b")])

        assert "2 paths" in adapter.source_name

    def test_missing_file_warning(self, tmp_path, caplog):
        """Test warning for missing file."""
        adapter = LocalFileAdapter([str(tmp_path / "nonexistent.json")])
        list(adapter.get_cases())

        assert "does not exist" in caplog.text

    def test_parse_case_generates_id(self, tmp_path):
        """Test that missing ID is auto-generated."""
        cases = [{"domain": "contracts", "facts": "No ID case"}]

        file_path = tmp_path / "cases.json"
        with open(file_path, "w") as f:
            json.dump(cases, f)

        adapter = LocalFileAdapter([str(file_path)])
        loaded = list(adapter.get_cases())

        assert len(loaded) == 1
        assert loaded[0].id.startswith("case_")


class TestCourtListenerAdapter:
    """Tests for CourtListenerAdapter."""

    def test_default_domain_mappings(self):
        """Test default domain mappings are set."""
        adapter = CourtListenerAdapter()

        assert "statute of frauds" in adapter._domain_mappings
        assert adapter._domain_mappings["statute of frauds"] == "contracts"
        assert "adverse possession" in adapter._domain_mappings

    def test_custom_domain_mappings(self):
        """Test custom domain mappings override defaults."""
        custom = {"my_query": "custom_domain"}
        adapter = CourtListenerAdapter(domain_mappings=custom)

        assert adapter._domain_mappings["my_query"] == "custom_domain"
        # Default mappings should still be present
        assert "statute of frauds" in adapter._domain_mappings

    def test_custom_search_queries(self):
        """Test custom search queries."""
        queries = ["custom query 1", "custom query 2"]
        adapter = CourtListenerAdapter(search_queries=queries)

        assert adapter._search_queries == queries

    def test_source_name(self):
        """Test source name property."""
        adapter = CourtListenerAdapter()

        assert adapter.source_name == "CourtListener API"

    @patch("loft.autonomous.data_sources.CourtListenerAdapter._get_client")
    def test_fetch_cases_caching(self, mock_get_client):
        """Test that cases are cached after first fetch."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.documents = []
        mock_client.search.return_value = mock_result
        mock_get_client.return_value = mock_client

        adapter = CourtListenerAdapter(search_queries=["test"])

        # First call triggers fetch
        list(adapter.get_cases())
        assert adapter._fetched is True

        # Second call uses cache
        list(adapter.get_cases())
        assert mock_client.search.call_count == 1

    def test_get_case_count_triggers_fetch(self):
        """Test that get_case_count triggers fetch."""
        with patch.object(CourtListenerAdapter, "_fetch_cases"):
            adapter = CourtListenerAdapter()
            adapter._cached_cases = []
            adapter._fetched = True

            count = adapter.get_case_count()

            assert count == 0


class TestCompositeAdapter:
    """Tests for CompositeAdapter."""

    def test_combines_multiple_adapters(self, tmp_path):
        """Test combining multiple adapters."""
        # Create two adapters with different data
        file1 = tmp_path / "cases1.json"
        with open(file1, "w") as f:
            json.dump([{"id": "1", "facts": "C1"}], f)

        file2 = tmp_path / "cases2.json"
        with open(file2, "w") as f:
            json.dump([{"id": "2", "facts": "C2"}], f)

        adapter1 = LocalFileAdapter([str(file1)])
        adapter2 = LocalFileAdapter([str(file2)])

        composite = CompositeAdapter([adapter1, adapter2])
        loaded = list(composite.get_cases())

        assert len(loaded) == 2

    def test_respects_limit(self, tmp_path):
        """Test that limit is respected across adapters."""
        file1 = tmp_path / "cases1.json"
        with open(file1, "w") as f:
            json.dump([{"id": str(i), "facts": f"C{i}"} for i in range(5)], f)

        file2 = tmp_path / "cases2.json"
        with open(file2, "w") as f:
            json.dump([{"id": str(i + 5), "facts": f"C{i}"} for i in range(5)], f)

        adapter1 = LocalFileAdapter([str(file1)])
        adapter2 = LocalFileAdapter([str(file2)])

        composite = CompositeAdapter([adapter1, adapter2])
        loaded = list(composite.get_cases(limit=3))

        assert len(loaded) == 3

    def test_total_case_count(self, tmp_path):
        """Test total case count across adapters."""
        file1 = tmp_path / "cases1.json"
        with open(file1, "w") as f:
            json.dump([{"id": "1", "facts": "C1"}, {"id": "2", "facts": "C2"}], f)

        file2 = tmp_path / "cases2.json"
        with open(file2, "w") as f:
            json.dump([{"id": "3", "facts": "C3"}], f)

        adapter1 = LocalFileAdapter([str(file1)])
        adapter2 = LocalFileAdapter([str(file2)])

        composite = CompositeAdapter([adapter1, adapter2])

        assert composite.get_case_count() == 3

    def test_source_name_combined(self, tmp_path):
        """Test combined source name."""
        file1 = tmp_path / "cases1.json"
        file1.touch()
        file2 = tmp_path / "cases2.json"
        file2.touch()

        adapter1 = LocalFileAdapter([str(file1)])
        adapter2 = LocalFileAdapter([str(file2)])

        composite = CompositeAdapter([adapter1, adapter2])

        assert " + " in composite.source_name


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_local_adapter(self, tmp_path):
        """Test create_local_adapter factory."""
        file_path = tmp_path / "cases.json"
        with open(file_path, "w") as f:
            json.dump([{"id": "1", "facts": "Test"}], f)

        adapter = create_local_adapter([str(file_path)])

        assert isinstance(adapter, LocalFileAdapter)
        assert adapter.get_case_count() == 1

    def test_create_courtlistener_adapter(self):
        """Test create_courtlistener_adapter factory."""
        adapter = create_courtlistener_adapter(
            search_queries=["test query"],
            max_cases_per_query=25,
        )

        assert isinstance(adapter, CourtListenerAdapter)
        assert adapter._search_queries == ["test query"]
        assert adapter._max_per_query == 25

    def test_create_courtlistener_adapter_defaults(self):
        """Test create_courtlistener_adapter with defaults."""
        adapter = create_courtlistener_adapter()

        assert isinstance(adapter, CourtListenerAdapter)
        assert len(adapter._search_queries) > 0
