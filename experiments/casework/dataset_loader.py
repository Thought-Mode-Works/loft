"""
Dataset loader for legal scenario test cases.

Loads and manages collections of legal test scenarios in JSON format.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class LegalScenario:
    """A legal test scenario."""

    scenario_id: str
    description: str
    facts: List[str]
    question: str
    ground_truth: str
    rationale: str
    asp_facts: Optional[str] = None
    legal_citations: Optional[List[str]] = None
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalScenario":
        """Create scenario from dictionary."""
        return cls(
            scenario_id=data["id"],
            description=data["description"],
            facts=data.get("facts", []),
            question=data["question"],
            ground_truth=data["ground_truth"],
            rationale=data["rationale"],
            asp_facts=data.get("asp_facts"),
            legal_citations=data.get("legal_citations", []),
            difficulty=data.get("difficulty", "medium"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.scenario_id,
            "description": self.description,
            "facts": self.facts,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "rationale": self.rationale,
            "asp_facts": self.asp_facts,
            "legal_citations": self.legal_citations,
            "difficulty": self.difficulty,
        }


class DatasetLoader:
    """Loads and manages legal scenario datasets."""

    def __init__(self, dataset_dir: Path):
        """Initialize dataset loader."""
        self.dataset_dir = Path(dataset_dir)

    def load_all(self) -> List[LegalScenario]:
        """Load all scenarios from dataset directory."""
        scenarios = []

        if not self.dataset_dir.exists():
            return scenarios

        for json_file in sorted(self.dataset_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                scenario = LegalScenario.from_dict(data)
                scenarios.append(scenario)

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        return scenarios

    def load_by_difficulty(self, difficulty: str) -> List[LegalScenario]:
        """Load scenarios of specific difficulty."""
        all_scenarios = self.load_all()
        return [s for s in all_scenarios if s.difficulty == difficulty]

    def save_scenario(self, scenario: LegalScenario) -> None:
        """Save a scenario to the dataset."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.dataset_dir / f"{scenario.scenario_id}.json"

        with open(output_path, "w") as f:
            json.dump(scenario.to_dict(), f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        scenarios = self.load_all()

        return {
            "total_scenarios": len(scenarios),
            "by_difficulty": {
                "easy": len([s for s in scenarios if s.difficulty == "easy"]),
                "medium": len([s for s in scenarios if s.difficulty == "medium"]),
                "hard": len([s for s in scenarios if s.difficulty == "hard"]),
            },
            "by_ground_truth": {
                "enforceable": len([s for s in scenarios if s.ground_truth == "enforceable"]),
                "unenforceable": len([s for s in scenarios if s.ground_truth == "unenforceable"]),
            },
        }
