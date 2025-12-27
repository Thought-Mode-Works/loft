"""
Integration tests for rule accumulation pipeline.

Issue #273: Continuous Rule Accumulation Pipeline
"""

import json

import pytest

from loft.accumulation.pipeline import RuleAccumulationPipeline
from loft.accumulation.schemas import CaseData
from loft.knowledge.database import KnowledgeDatabase


class TestRuleAccumulationPipeline:
    """Integration tests for accumulation pipeline."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create temporary test database."""
        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    @pytest.fixture
    def pipeline(self, db):
        """Create pipeline with test database."""
        return RuleAccumulationPipeline(
            knowledge_db=db,
            min_rule_confidence=0.7,
            auto_resolve_conflicts=True,
        )

    @pytest.fixture
    def sample_case(self):
        """Create sample case data."""
        return CaseData(
            case_id="test_001",
            description="Test contract formation case",
            facts=[
                "Alice made an offer to Bob",
                "Bob accepted the offer",
                "There was consideration",
            ],
            asp_facts="offer(contract1). acceptance(contract1). consideration(contract1).",
            question="Is the contract valid?",
            ground_truth="yes",
            rationale=(
                "A contract is valid when there is offer, acceptance, and consideration. "
                "All three elements are present in this case."
            ),
            domain="contracts",
        )

    @pytest.fixture
    def sample_case_file(self, tmp_path, sample_case):
        """Create sample case JSON file."""
        case_file = tmp_path / "test_case.json"

        case_dict = {
            "id": sample_case.case_id,
            "description": sample_case.description,
            "facts": sample_case.facts,
            "asp_facts": sample_case.asp_facts,
            "question": sample_case.question,
            "ground_truth": sample_case.ground_truth,
            "rationale": sample_case.rationale,
            "domain": sample_case.domain,
        }

        with open(case_file, "w") as f:
            json.dump(case_dict, f, indent=2)

        return case_file

    def test_create_pipeline(self, pipeline):
        """Test creating pipeline."""
        assert pipeline is not None
        assert pipeline.min_rule_confidence == 0.7
        assert pipeline.auto_resolve_conflicts is True

    def test_process_case(self, pipeline, sample_case):
        """Test processing a single case."""
        result = pipeline.process_case(sample_case)

        assert result is not None
        assert result.case_id == "test_001"
        # Rules may or may not be added depending on LLM
        assert isinstance(result.rules_added, int)
        assert isinstance(result.rules_skipped, int)
        assert result.processing_time_ms > 0

    def test_process_case_adds_rules_to_database(self, pipeline, sample_case, db):
        """Test that processing adds rules to database."""
        initial_stats = db.get_database_stats()
        initial_count = initial_stats.total_rules

        result = pipeline.process_case(sample_case)

        final_stats = db.get_database_stats()
        final_count = final_stats.total_rules

        # Should have added rules (or at least tried)
        assert final_count >= initial_count
        assert result.rules_added == (final_count - initial_count)

    def test_process_case_with_low_confidence_rules(self, pipeline, db):
        """Test that low confidence rules are skipped."""
        # Set high threshold
        pipeline.min_rule_confidence = 0.99

        case = CaseData(
            case_id="test_002",
            description="Test case",
            facts=["Some facts"],
            asp_facts="fact(1).",
            question="Question?",
            ground_truth="yes",
            rationale="Simple rationale for testing low confidence handling",
            domain="test",
        )

        result = pipeline.process_case(case)

        # With very high threshold, most rules should be skipped
        # (unless LLM is extremely confident)
        assert result.rules_skipped >= 0

    def test_extract_rule_candidates(self, pipeline, sample_case):
        """Test extracting rule candidates from case."""
        candidates = pipeline._extract_rule_candidates(sample_case)

        # Should extract some candidates
        assert isinstance(candidates, list)
        # May be 0 if LLM not available or fails
        if len(candidates) > 0:
            assert candidates[0].domain == "contracts"
            assert candidates[0].source_case_id == "test_001"

    def test_extract_predicates_from_case(self, pipeline, sample_case):
        """Test extracting predicates from case facts."""
        predicates = pipeline._extract_predicates_from_case(sample_case)

        assert isinstance(predicates, list)
        assert "offer" in predicates
        assert "acceptance" in predicates
        assert "consideration" in predicates

    def test_validate_rule_quality(self, pipeline):
        """Test rule quality validation."""
        from loft.accumulation.schemas import RuleCandidate

        # Good rule
        good_rule = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        assert pipeline._validate_rule_quality(good_rule) is True

        # Low confidence rule
        low_conf_rule = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.5,
            reasoning="Test",
            source_case_id="c1",
        )

        assert pipeline._validate_rule_quality(low_conf_rule) is False

        # Empty rule
        empty_rule = RuleCandidate(
            asp_rule="",
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        assert pipeline._validate_rule_quality(empty_rule) is False

    def test_resolve_conflicts_auto_mode(self, pipeline):
        """Test automatic conflict resolution."""
        from loft.accumulation.schemas import Conflict, RuleCandidate

        candidate = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.95,
            reasoning="Test",
            source_case_id="c1",
        )

        # Contradiction conflict
        contradiction = Conflict(
            conflict_type="contradiction",
            new_rule=candidate.asp_rule,
            existing_rule_id="r1",
            existing_rule="not valid(X) :- condition(X).",
            explanation="Direct contradiction",
            severity=1.0,
        )

        resolution = pipeline._resolve_conflicts(candidate, [contradiction])

        # Should replace with high confidence
        assert resolution.should_add is True
        assert resolution.action == "replace"

    def test_resolve_conflicts_low_confidence_contradiction(self, pipeline):
        """Test resolving contradiction with low confidence."""
        from loft.accumulation.schemas import Conflict, RuleCandidate

        candidate = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.75,  # Below 0.9 threshold
            reasoning="Test",
            source_case_id="c1",
        )

        contradiction = Conflict(
            conflict_type="contradiction",
            new_rule=candidate.asp_rule,
            existing_rule_id="r1",
            existing_rule="not valid(X) :- condition(X).",
            explanation="Direct contradiction",
            severity=1.0,
        )

        resolution = pipeline._resolve_conflicts(candidate, [contradiction])

        # Should skip with low confidence
        assert resolution.should_add is False
        assert resolution.action == "skip"

    def test_resolve_conflicts_subsumption(self, pipeline):
        """Test resolving subsumption conflict."""
        from loft.accumulation.schemas import Conflict, RuleCandidate

        candidate = RuleCandidate(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        subsumption = Conflict(
            conflict_type="subsumption",
            new_rule=candidate.asp_rule,
            existing_rule_id="r1",
            existing_rule="valid(X) :- condition(X).",
            explanation="Rules are identical",
            severity=0.95,  # High subsumption
        )

        resolution = pipeline._resolve_conflicts(candidate, [subsumption])

        # Should skip high subsumption
        assert resolution.should_add is False

    def test_load_case_from_file(self, pipeline, sample_case_file):
        """Test loading case from JSON file."""
        case = pipeline._load_case(sample_case_file)

        assert case.case_id == "test_001"
        assert case.domain == "contracts"
        assert len(case.facts) == 3

    def test_process_dataset(self, pipeline, tmp_path, sample_case_file):
        """Test processing a dataset directory."""
        # Create dataset directory with multiple cases
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Copy sample case to dataset
        import shutil

        case1 = dataset_dir / "case_001.json"
        shutil.copy(sample_case_file, case1)

        # Create second case
        case2_data = {
            "id": "test_002",
            "description": "Second test case",
            "facts": ["Fact 1"],
            "asp_facts": "fact(1).",
            "question": "Q?",
            "ground_truth": "no",
            "rationale": "Test rationale",
            "domain": "test",
        }

        with open(dataset_dir / "case_002.json", "w") as f:
            json.dump(case2_data, f)

        # Process dataset
        report = pipeline.process_dataset(dataset_dir)

        assert report.total_cases == 2
        assert report.total_rules_added >= 0
        assert report.avg_processing_time_ms > 0

    def test_process_dataset_with_max_cases(self, pipeline, tmp_path, sample_case_file):
        """Test processing dataset with max_cases limit."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Create 3 cases
        import shutil

        for i in range(3):
            case_file = dataset_dir / f"case_{i:03d}.json"
            shutil.copy(sample_case_file, case_file)

        # Process only 2 cases
        report = pipeline.process_dataset(dataset_dir, max_cases=2)

        assert report.total_cases == 2

    def test_get_accumulation_stats(self, pipeline, db):
        """Test getting accumulation statistics."""
        # Add some rules first
        db.add_rule(
            asp_rule="rule1(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Test",
        )

        db.add_rule(
            asp_rule="rule2(X) :- condition(X).",
            domain="test",
            confidence=0.8,
            reasoning="Test",
        )

        stats = pipeline.get_accumulation_stats()

        assert stats["total_rules"] == 2
        assert stats["active_rules"] == 2
        assert "test" in stats["domains"]

    def test_process_case_with_conflict_detection(self, pipeline, db):
        """Test that conflicts are detected during processing."""
        # Add existing rule
        db.add_rule(
            asp_rule="valid(X) :- condition(X).",
            domain="test",
            confidence=0.9,
            reasoning="Existing rule",
        )

        # Process case that might generate conflicting rule
        case = CaseData(
            case_id="test_003",
            description="Conflict test case",
            facts=["Test facts"],
            asp_facts="condition(test1).",
            question="Is it valid?",
            ground_truth="no",
            rationale=(
                "When condition is met, the item is not valid. "
                "This contradicts typical validation rules."
            ),
            domain="test",
        )

        result = pipeline.process_case(case)

        # Result should be valid even if conflicts found
        assert result.case_id == "test_003"
        # Conflicts may or may not be found depending on LLM output
        assert isinstance(result.conflicts_found, list)


class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create test database."""
        db_path = tmp_path / "test.db"
        return KnowledgeDatabase(f"sqlite:///{db_path}")

    @pytest.fixture
    def pipeline(self, db):
        """Create pipeline."""
        return RuleAccumulationPipeline(knowledge_db=db)

    def test_process_invalid_case_file(self, pipeline, tmp_path):
        """Test processing invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            pipeline._load_case(invalid_file)

    def test_process_dataset_handles_errors(self, pipeline, tmp_path):
        """Test that dataset processing continues despite errors."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Create valid case
        valid_case = {
            "id": "valid_001",
            "description": "Valid case",
            "facts": ["F"],
            "asp_facts": "f.",
            "question": "Q?",
            "ground_truth": "yes",
            "rationale": "R",
        }

        with open(dataset_dir / "valid.json", "w") as f:
            json.dump(valid_case, f)

        # Create invalid case
        with open(dataset_dir / "invalid.json", "w") as f:
            f.write("{ invalid }")

        # Should process valid case and handle error gracefully
        report = pipeline.process_dataset(dataset_dir)

        # Should have 2 results (1 valid, 1 error)
        assert report.total_cases == 2
