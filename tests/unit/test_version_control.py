"""
Unit tests for version control system.

Tests CoreState, VersionControl, diff computation, and storage.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from loft.version_control import (
    CoreState,
    Rule,
    Commit,
    StratificationLevel,
    VersionControl,
    VersionControlError,
    compute_diff,
    create_state_id,
    create_commit_id,
    VersionStorage,
)


class TestRule:
    """Tests for Rule class."""

    def test_rule_creation(self) -> None:
        """Test creating a rule."""
        rule = Rule(
            rule_id="rule1",
            content="fact(a).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
        )
        assert rule.rule_id == "rule1"
        assert rule.content == "fact(a)."
        assert rule.level == StratificationLevel.TACTICAL

    def test_rule_serialization(self) -> None:
        """Test rule to/from dict."""
        rule = Rule(
            rule_id="rule1",
            content="fact(a).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
        )
        rule_dict = rule.to_dict()
        restored_rule = Rule.from_dict(rule_dict)

        assert restored_rule.rule_id == rule.rule_id
        assert restored_rule.content == rule.content
        assert restored_rule.level == rule.level


class TestCoreState:
    """Tests for CoreState class."""

    def test_core_state_creation(self) -> None:
        """Test creating a core state."""
        rules = [
            Rule(
                "r1",
                "fact(a).",
                StratificationLevel.TACTICAL,
                0.9,
                "llm",
                datetime.utcnow().isoformat(),
            )
        ]
        state = CoreState(
            state_id="state1",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={"param1": "value1"},
            metrics={"accuracy": 0.85},
        )
        assert state.state_id == "state1"
        assert len(state.rules) == 1

    def test_core_state_serialization(self) -> None:
        """Test core state to/from JSON."""
        rules = [
            Rule(
                "r1",
                "fact(a).",
                StratificationLevel.OPERATIONAL,
                0.8,
                "llm",
                datetime.utcnow().isoformat(),
            )
        ]
        state = CoreState(
            state_id="state1",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={"test": "config"},
            metrics={"test_metric": 0.75},
        )

        json_str = state.to_json()
        restored_state = CoreState.from_json(json_str)

        assert restored_state.state_id == state.state_id
        assert len(restored_state.rules) == len(state.rules)
        assert restored_state.configuration == state.configuration

    def test_get_rules_by_level(self) -> None:
        """Test filtering rules by level."""
        rules = [
            Rule(
                "r1",
                "fact(a).",
                StratificationLevel.CONSTITUTIONAL,
                1.0,
                "human",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r2",
                "fact(b).",
                StratificationLevel.TACTICAL,
                0.8,
                "llm",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r3",
                "fact(c).",
                StratificationLevel.TACTICAL,
                0.85,
                "llm",
                datetime.utcnow().isoformat(),
            ),
        ]
        state = CoreState(
            state_id="state1",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

        tactical_rules = state.get_rules_by_level(StratificationLevel.TACTICAL)
        assert len(tactical_rules) == 2

        const_rules = state.get_rules_by_level(StratificationLevel.CONSTITUTIONAL)
        assert len(const_rules) == 1

    def test_count_rules_by_level(self) -> None:
        """Test counting rules by level."""
        rules = [
            Rule(
                "r1",
                "fact(a).",
                StratificationLevel.CONSTITUTIONAL,
                1.0,
                "human",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r2",
                "fact(b).",
                StratificationLevel.TACTICAL,
                0.8,
                "llm",
                datetime.utcnow().isoformat(),
            ),
        ]
        state = CoreState(
            state_id="state1",
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

        counts = state.count_rules_by_level()
        assert counts["constitutional"] == 1
        assert counts["tactical"] == 1
        assert counts["strategic"] == 0
        assert counts["operational"] == 0


class TestDiff:
    """Tests for diff computation."""

    def test_diff_no_changes(self) -> None:
        """Test diff with identical states."""
        rules = [
            Rule(
                "r1",
                "fact(a).",
                StratificationLevel.TACTICAL,
                0.9,
                "llm",
                datetime.utcnow().isoformat(),
            )
        ]
        state1 = CoreState("s1", datetime.utcnow().isoformat(), rules, {}, {})
        state2 = CoreState("s2", datetime.utcnow().isoformat(), rules, {}, {})

        diff = compute_diff(state1, state2)
        assert not diff.has_changes()

    def test_diff_rule_added(self) -> None:
        """Test diff with added rule."""
        state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
        state2 = CoreState(
            "s2",
            datetime.utcnow().isoformat(),
            [
                Rule(
                    "r1",
                    "fact(a).",
                    StratificationLevel.TACTICAL,
                    0.9,
                    "llm",
                    datetime.utcnow().isoformat(),
                )
            ],
            {},
            {},
        )

        diff = compute_diff(state1, state2)
        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["added"] == 1

    def test_diff_rule_removed(self) -> None:
        """Test diff with removed rule."""
        state1 = CoreState(
            "s1",
            datetime.utcnow().isoformat(),
            [
                Rule(
                    "r1",
                    "fact(a).",
                    StratificationLevel.TACTICAL,
                    0.9,
                    "llm",
                    datetime.utcnow().isoformat(),
                )
            ],
            {},
            {},
        )
        state2 = CoreState("s2", datetime.utcnow().isoformat(), [], {}, {})

        diff = compute_diff(state1, state2)
        assert diff.has_changes()
        counts = diff.count_by_type()
        assert counts["removed"] == 1

    def test_diff_config_changed(self) -> None:
        """Test diff with configuration changes."""
        state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {"param": "old"}, {})
        state2 = CoreState("s2", datetime.utcnow().isoformat(), [], {"param": "new"}, {})

        diff = compute_diff(state1, state2)
        assert diff.has_changes()
        assert len(diff.config_changes) == 1

    def test_diff_summary(self) -> None:
        """Test diff summary generation."""
        state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
        state2 = CoreState(
            "s2",
            datetime.utcnow().isoformat(),
            [
                Rule(
                    "r1",
                    "fact(a).",
                    StratificationLevel.TACTICAL,
                    0.9,
                    "llm",
                    datetime.utcnow().isoformat(),
                )
            ],
            {},
            {},
        )

        diff = compute_diff(state1, state2)
        summary = diff.summary()
        assert "1 rules added" in summary


class TestVersionStorage:
    """Tests for VersionStorage."""

    def test_storage_initialization(self) -> None:
        """Test storage initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = VersionStorage(Path(tmpdir))
            assert storage.commits_dir.exists()
            assert storage.heads_dir.exists()
            assert storage.tags_dir.exists()

    def test_save_and_load_commit(self) -> None:
        """Test saving and loading commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = VersionStorage(Path(tmpdir))

            state = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            commit = Commit(
                commit_id="c1",
                parent_id=None,
                state=state,
                message="Test commit",
                author="test",
                timestamp=datetime.utcnow().isoformat(),
            )

            storage.save_commit(commit)
            loaded_commit = storage.load_commit("c1")

            assert loaded_commit is not None
            assert loaded_commit.commit_id == commit.commit_id
            assert loaded_commit.message == commit.message

    def test_branch_operations(self) -> None:
        """Test branch save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = VersionStorage(Path(tmpdir))

            storage.save_branch("main", "commit123")
            commit_id = storage.load_branch("main")

            assert commit_id == "commit123"

    def test_tag_operations(self) -> None:
        """Test tag save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = VersionStorage(Path(tmpdir))

            storage.save_tag("v1.0.0", "commit456")
            commit_id = storage.load_tag("v1.0.0")

            assert commit_id == "commit456"

    def test_head_operations(self) -> None:
        """Test HEAD get/set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = VersionStorage(Path(tmpdir))

            storage.set_head("ref: refs/heads/main")
            head = storage.get_head()

            assert head == "ref: refs/heads/main"

    def test_get_current_branch(self) -> None:
        """Test getting current branch name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = VersionStorage(Path(tmpdir))

            storage.set_head("ref: refs/heads/develop")
            branch = storage.get_current_branch()

            assert branch == "develop"


class TestVersionControl:
    """Tests for VersionControl class."""

    def test_version_control_initialization(self) -> None:
        """Test VC initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))
            assert vc.storage is not None

    def test_commit(self) -> None:
        """Test committing a state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            commit_id = vc.commit(state, "Initial commit")

            assert commit_id is not None
            assert len(commit_id) > 0

    def test_log(self) -> None:
        """Test viewing commit history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            state2 = CoreState("s2", datetime.utcnow().isoformat(), [], {}, {})

            vc.commit(state1, "Commit 1")
            vc.commit(state2, "Commit 2")

            log = vc.log(max_count=10)
            assert len(log) == 2
            assert log[0].message == "Commit 2"  # Most recent first

    def test_diff(self) -> None:
        """Test diff between commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            state2 = CoreState(
                "s2",
                datetime.utcnow().isoformat(),
                [
                    Rule(
                        "r1",
                        "fact(a).",
                        StratificationLevel.TACTICAL,
                        0.9,
                        "llm",
                        datetime.utcnow().isoformat(),
                    )
                ],
                {},
                {},
            )

            commit1 = vc.commit(state1, "Commit 1")
            commit2 = vc.commit(state2, "Commit 2")

            diff = vc.diff(commit1, commit2)
            assert diff.has_changes()

    def test_rollback(self) -> None:
        """Test rollback to previous state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {"test": "value1"}, {})
            state2 = CoreState("s2", datetime.utcnow().isoformat(), [], {"test": "value2"}, {})

            commit1 = vc.commit(state1, "Commit 1")
            vc.commit(state2, "Commit 2")

            rolled_back_state = vc.rollback(commit1)
            assert rolled_back_state.configuration["test"] == "value1"

    def test_branch(self) -> None:
        """Test creating a branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            vc.commit(state, "Commit 1")

            vc.branch("feature")
            branches = vc.list_branches()

            assert "feature" in branches

    def test_checkout(self) -> None:
        """Test checking out a branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            vc.commit(state, "Commit 1")

            vc.branch("feature")
            vc.checkout("feature")

            assert vc.get_current_branch() == "feature"

    def test_merge_no_conflicts(self) -> None:
        """Test merging without conflicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            # Create initial commit
            state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            vc.commit(state1, "Initial commit")

            # Create feature branch
            vc.branch("feature")
            vc.checkout("feature")

            # Add rule on feature branch
            state2 = CoreState(
                "s2",
                datetime.utcnow().isoformat(),
                [
                    Rule(
                        "r1",
                        "fact(a).",
                        StratificationLevel.TACTICAL,
                        0.9,
                        "llm",
                        datetime.utcnow().isoformat(),
                    )
                ],
                {},
                {},
            )
            vc.commit(state2, "Add rule")

            # Go back to main and merge
            vc.checkout("main")
            merge_commit = vc.merge("feature")

            assert merge_commit is not None

    def test_tag(self) -> None:
        """Test tagging a commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            state = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            vc.commit(state, "Commit 1")

            vc.tag("v1.0.0")
            tags = vc.list_tags()

            assert "v1.0.0" in tags

    def test_rollback_constitutional_protection(self) -> None:
        """Test that rollback prevents removing constitutional rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))

            # Commit without constitutional rule
            state1 = CoreState("s1", datetime.utcnow().isoformat(), [], {}, {})
            commit1 = vc.commit(state1, "Initial commit without constitutional rules")

            # Commit with constitutional rule
            state2 = CoreState(
                "s2",
                datetime.utcnow().isoformat(),
                [
                    Rule(
                        "r1",
                        "const_rule.",
                        StratificationLevel.CONSTITUTIONAL,
                        1.0,
                        "human",
                        datetime.utcnow().isoformat(),
                    )
                ],
                {},
                {},
            )
            vc.commit(state2, "Add constitutional rule")

            # Try to rollback to state without constitutional rule (should fail)
            with pytest.raises(VersionControlError, match="constitutional"):
                vc.rollback(commit1)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_state_id(self) -> None:
        """Test state ID generation."""
        id1 = create_state_id()
        id2 = create_state_id()

        assert len(id1) == 16
        assert id1 != id2  # Should be unique

    def test_create_commit_id(self) -> None:
        """Test commit ID generation."""
        id1 = create_commit_id()
        id2 = create_commit_id()

        assert len(id1) == 16
        assert id1 != id2  # Should be unique
