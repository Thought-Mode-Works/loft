"""
Unit tests for accumulation schemas.

Issue #273: Continuous Rule Accumulation Pipeline
"""

from datetime import datetime


from loft.accumulation.schemas import (
    AccumulationResult,
    BatchAccumulationReport,
    CaseData,
    Conflict,
    ConflictResolution,
    RuleCandidate,
)


class TestConflict:
    """Test Conflict schema."""

    def test_create_conflict(self):
        """Test creating a conflict."""
        conflict = Conflict(
            conflict_type="contradiction",
            new_rule="valid(X) :- condition(X).",
            existing_rule_id="rule123",
            existing_rule="not valid(X) :- condition(X).",
            explanation="Rules contradict each other",
            severity=1.0,
        )

        assert conflict.conflict_type == "contradiction"
        assert conflict.severity == 1.0
        assert "rule123" in conflict.existing_rule_id

    def test_conflict_string_representation(self):
        """Test conflict string formatting."""
        conflict = Conflict(
            conflict_type="subsumption",
            new_rule="p(X) :- q(X).",
            existing_rule_id="abc123def456",
            existing_rule="p(X) :- q(X), r(X).",
            explanation="New rule subsumes existing",
            severity=0.8,
        )

        str_repr = str(conflict)
        assert "subsumption" in str_repr
        assert "abc123de" in str_repr  # First 8 chars of ID


class TestConflictResolution:
    """Test ConflictResolution schema."""

    def test_create_resolution_add(self):
        """Test creating add resolution."""
        resolution = ConflictResolution(
            should_add=True,
            action="add",
            reason="No significant conflicts",
        )

        assert resolution.should_add is True
        assert resolution.action == "add"
        assert len(resolution.rules_to_archive) == 0

    def test_create_resolution_replace(self):
        """Test creating replace resolution."""
        resolution = ConflictResolution(
            should_add=True,
            action="replace",
            reason="New rule has higher confidence",
            rules_to_archive=["rule1", "rule2"],
        )

        assert resolution.should_add is True
        assert len(resolution.rules_to_archive) == 2

    def test_resolution_string_representation(self):
        """Test resolution string formatting."""
        resolution = ConflictResolution(
            should_add=False,
            action="skip",
            reason="Contradiction detected",
        )

        str_repr = str(resolution)
        assert "skip" in str_repr
        assert "Contradiction" in str_repr


class TestRuleCandidate:
    """Test RuleCandidate schema."""

    def test_create_rule_candidate(self):
        """Test creating a rule candidate."""
        candidate = RuleCandidate(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X).",
            domain="contracts",
            confidence=0.85,
            reasoning="Contract requires offer and acceptance",
            source_case_id="case001",
            principle="Basic contract formation",
        )

        assert candidate.asp_rule.startswith("valid_contract")
        assert candidate.domain == "contracts"
        assert candidate.confidence == 0.85
        assert candidate.source_case_id == "case001"

    def test_candidate_string_representation(self):
        """Test candidate string formatting."""
        candidate = RuleCandidate(
            asp_rule="p(X) :- q(X)." * 20,  # Long rule
            domain="test",
            confidence=0.9,
            reasoning="Test",
            source_case_id="c1",
        )

        str_repr = str(candidate)
        assert "test" in str_repr
        assert len(str_repr) < 100  # Should be truncated


class TestAccumulationResult:
    """Test AccumulationResult schema."""

    def test_create_result(self):
        """Test creating accumulation result."""
        result = AccumulationResult(
            case_id="case001",
            rules_added=3,
            rules_skipped=1,
        )

        assert result.case_id == "case001"
        assert result.rules_added == 3
        assert result.rules_skipped == 1
        assert isinstance(result.timestamp, datetime)

    def test_success_rate_calculation(self):
        """Test success rate property."""
        result = AccumulationResult(
            case_id="case001",
            rules_added=7,
            rules_skipped=3,
        )

        assert result.success_rate == 0.7

    def test_success_rate_with_zero_total(self):
        """Test success rate when no rules processed."""
        result = AccumulationResult(
            case_id="case001",
            rules_added=0,
            rules_skipped=0,
        )

        assert result.success_rate == 0.0

    def test_result_with_conflicts(self):
        """Test result with conflicts."""
        conflict = Conflict(
            conflict_type="contradiction",
            new_rule="p(X).",
            existing_rule_id="r1",
            existing_rule="not p(X).",
            explanation="Direct contradiction",
            severity=1.0,
        )

        result = AccumulationResult(
            case_id="case001",
            rules_added=2,
            rules_skipped=1,
            conflicts_found=[conflict],
        )

        assert len(result.conflicts_found) == 1
        assert result.conflicts_found[0].conflict_type == "contradiction"

    def test_result_string_representation(self):
        """Test result string formatting."""
        result = AccumulationResult(
            case_id="case_xyz",
            rules_added=5,
            rules_skipped=2,
        )

        str_repr = str(result)
        assert "case_xyz" in str_repr
        assert "+5" in str_repr
        assert "-2" in str_repr


class TestBatchAccumulationReport:
    """Test BatchAccumulationReport schema."""

    def test_create_empty_report(self):
        """Test creating empty report."""
        report = BatchAccumulationReport(results=[])

        assert report.total_cases == 0
        assert report.total_rules_added == 0
        assert report.total_rules_skipped == 0

    def test_create_report_with_results(self):
        """Test creating report from results."""
        results = [
            AccumulationResult(case_id="c1", rules_added=3, rules_skipped=1),
            AccumulationResult(case_id="c2", rules_added=2, rules_skipped=0),
            AccumulationResult(case_id="c3", rules_added=1, rules_skipped=2),
        ]

        report = BatchAccumulationReport(results=results)

        assert report.total_cases == 3
        assert report.total_rules_added == 6
        assert report.total_rules_skipped == 3

    def test_overall_success_rate(self):
        """Test overall success rate calculation."""
        results = [
            AccumulationResult(case_id="c1", rules_added=8, rules_skipped=2),
            AccumulationResult(case_id="c2", rules_added=6, rules_skipped=4),
        ]

        report = BatchAccumulationReport(results=results)

        assert report.overall_success_rate == 0.7  # 14/(14+6)

    def test_rules_per_case(self):
        """Test rules per case calculation."""
        results = [
            AccumulationResult(case_id="c1", rules_added=4, rules_skipped=0),
            AccumulationResult(case_id="c2", rules_added=2, rules_skipped=0),
        ]

        report = BatchAccumulationReport(results=results)

        assert report.rules_per_case == 3.0  # 6/2

    def test_report_with_conflicts(self):
        """Test report aggregates conflicts."""
        conflict1 = Conflict(
            conflict_type="contradiction",
            new_rule="p.",
            existing_rule_id="r1",
            existing_rule="not p.",
            explanation="Test",
            severity=1.0,
        )

        conflict2 = Conflict(
            conflict_type="subsumption",
            new_rule="q.",
            existing_rule_id="r2",
            existing_rule="q.",
            explanation="Test",
            severity=0.8,
        )

        results = [
            AccumulationResult(
                case_id="c1",
                rules_added=1,
                rules_skipped=1,
                conflicts_found=[conflict1],
            ),
            AccumulationResult(
                case_id="c2",
                rules_added=1,
                rules_skipped=1,
                conflicts_found=[conflict2, conflict1],
            ),
        ]

        report = BatchAccumulationReport(results=results)

        assert report.total_conflicts == 3

    def test_report_to_string(self):
        """Test report string formatting."""
        results = [
            AccumulationResult(case_id="c1", rules_added=3, rules_skipped=1),
        ]

        report = BatchAccumulationReport(results=results)
        report_str = report.to_string()

        assert "Batch Accumulation Report" in report_str
        assert "Cases Processed: 1" in report_str
        assert "Rules Added: 3" in report_str

    def test_report_string_representation(self):
        """Test report short string."""
        results = [
            AccumulationResult(case_id="c1", rules_added=10, rules_skipped=5),
        ]

        report = BatchAccumulationReport(results=results)
        str_repr = str(report)

        assert "1 cases" in str_repr
        assert "+10 rules" in str_repr


class TestCaseData:
    """Test CaseData schema."""

    def test_create_case_data(self):
        """Test creating case data."""
        case = CaseData(
            case_id="case001",
            description="Test case description",
            facts=["Fact 1", "Fact 2"],
            asp_facts="fact(1). fact(2).",
            question="Is this valid?",
            ground_truth="yes",
            rationale="Because of legal principle X",
            domain="contracts",
        )

        assert case.case_id == "case001"
        assert len(case.facts) == 2
        assert case.domain == "contracts"

    def test_create_case_from_dict(self):
        """Test creating case from dictionary."""
        case_dict = {
            "id": "case002",
            "description": "Another test case",
            "facts": ["Fact A", "Fact B", "Fact C"],
            "asp_facts": "factA. factB. factC.",
            "question": "Question?",
            "ground_truth": "no",
            "rationale": "Rationale here",
            "legal_citations": ["Citation 1", "Citation 2"],
            "difficulty": "hard",
            "domain": "torts",
        }

        case = CaseData.from_dict(case_dict)

        assert case.case_id == "case002"
        assert len(case.facts) == 3
        assert len(case.legal_citations) == 2
        assert case.difficulty == "hard"
        assert case.domain == "torts"

    def test_case_from_dict_with_defaults(self):
        """Test creating case from dict with missing optional fields."""
        case_dict = {
            "id": "case003",
            "description": "Minimal case",
            "facts": ["Fact"],
            "asp_facts": "fact.",
            "question": "Q?",
            "ground_truth": "unknown",
            "rationale": "R",
        }

        case = CaseData.from_dict(case_dict)

        assert case.case_id == "case003"
        assert case.legal_citations == []
        assert case.difficulty == "medium"
        assert case.domain is None

    def test_case_string_representation(self):
        """Test case string formatting."""
        case = CaseData(
            case_id="c123",
            description="A very long description that should be truncated " * 5,
            facts=["F"],
            asp_facts="f.",
            question="Q?",
            ground_truth="yes",
            rationale="R",
        )

        str_repr = str(case)
        assert "c123" in str_repr
        assert len(str_repr) < 100  # Should be truncated
