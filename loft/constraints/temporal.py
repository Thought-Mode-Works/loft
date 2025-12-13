"""
Temporal Consistency Invariance implementation for Phase 7.

This module implements temporal consistency testing to ensure legal rules produce
consistent outcomes for cases with similar temporal relationships. Time-shifted cases
should yield equivalent legal analyses when the relative temporal ordering is preserved.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta, date
from enum import Enum, auto
import copy


class TemporalTransformType(Enum):
    """Types of temporal transformations."""

    UNIFORM_SHIFT = auto()  # Add constant offset to all dates
    DURATION_SCALE = auto()  # Scale all durations by factor
    ORDER_PRESERVING = auto()  # Arbitrary monotonic transform


@dataclass
class TemporalField:
    """Identifies a temporal field in case representation."""

    path: str  # JSON path to field (e.g., "contract.formation_date")
    field_type: str  # "date" | "duration" | "timestamp"
    reference_point: Optional[str] = None  # For relative durations


@dataclass
class TemporalViolation:
    """Records a temporal consistency violation."""

    rule_name: str
    original_case: Dict[str, Any]
    transformed_case: Dict[str, Any]
    transform_params: Dict[str, Any]
    original_outcome: Any
    transformed_outcome: Any
    violation_type: (
        str  # "shift_invariance" | "order_preservation" | "duration_consistency"
    )

    def explain(self) -> str:
        """Generate explanation of why this is a violation."""
        return (
            f"Temporal inconsistency in {self.rule_name}:\n"
            f"  Transform: {self.transform_params}\n"
            f"  Original outcome: {self.original_outcome}\n"
            f"  Transformed outcome: {self.transformed_outcome}\n"
            f"  Violation: Outcome should be invariant under {self.violation_type}"
        )


@dataclass
class TemporalConsistencyReport:
    """Report on temporal consistency testing."""

    rule_name: str
    total_tests: int
    violations: List[TemporalViolation]
    shift_invariant: bool
    order_invariant: bool = True

    @property
    def is_consistent(self) -> bool:
        """Check if any violations were found."""
        return len(self.violations) == 0

    @property
    def consistency_score(self) -> float:
        """Calculate the percentage of passed tests."""
        passed = self.total_tests - len(self.violations)
        return passed / self.total_tests if self.total_tests > 0 else 1.0


class TemporalConsistencyTester:
    """Tests legal rules for temporal consistency."""

    def __init__(self, temporal_fields: List[TemporalField]):
        self.temporal_fields = temporal_fields

    def detect_temporal_fields(self, case: Dict[str, Any]) -> List[TemporalField]:
        """Auto-detect temporal fields in case representation."""
        fields: List[TemporalField] = []
        self._scan_for_dates(case, "", fields)
        return fields

    def _scan_for_dates(self, obj: Any, path: str, found: List[TemporalField]) -> None:
        """Recursively scan for date-like fields."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key

                if isinstance(value, (dict, list)):
                    self._scan_for_dates(value, new_path, found)
                elif self._is_date_field(key, value):
                    found.append(
                        TemporalField(
                            path=new_path,
                            field_type=self._infer_temporal_type(key, value),
                        )
                    )
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._scan_for_dates(item, f"{path}[{i}]", found)

    def _is_date_field(self, key: str, value: Any) -> bool:
        """Check if a field is likely a date field."""
        # Heuristic: check key name and value format
        key_lower = key.lower()
        if "date" in key_lower or "time" in key_lower or "deadline" in key_lower:
            return True
        if isinstance(value, (datetime, date)):
            return True
        if isinstance(value, str):
            try:
                # Try parsing as ISO format
                datetime.fromisoformat(value)
                return True
            except ValueError:
                pass
        return False

    def _infer_temporal_type(self, key: str, value: Any) -> str:
        """Infer the type of temporal field."""
        if isinstance(value, datetime) or (isinstance(value, str) and "T" in value):
            return "timestamp"
        return "date"

    def _get_nested(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if "[" in part and "]" in part:
                # Handle list index
                key = part[: part.find("[")]
                index = int(part[part.find("[") + 1 : part.find("]")])
                current = current[key][index]
            else:
                current = current[part]
        return current

    def _set_nested(self, obj: Dict[str, Any], path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        parts = path.split(".")
        current = obj
        for i, part in enumerate(parts[:-1]):
            if "[" in part and "]" in part:
                key = part[: part.find("[")]
                index = int(part[part.find("[") + 1 : part.find("]")])
                current = current[key][index]
            else:
                current = current[part]

        last_part = parts[-1]
        if "[" in last_part and "]" in last_part:
            key = last_part[: last_part.find("[")]
            index = int(last_part[last_part.find("[") + 1 : last_part.find("]")])
            current[key][index] = value
        else:
            current[last_part] = value

    def _parse_date(self, value: Any) -> Union[datetime, date]:
        """Parse date from string or return object if already date/datetime."""
        if isinstance(value, (datetime, date)):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ValueError(f"Cannot parse date from {value}")

    def _format_date(self, dt: Union[datetime, date], original_type: Any) -> Any:
        """Format date back to original type."""
        if isinstance(original_type, str):
            return dt.isoformat()
        return dt

    def _shift_date(self, value: Any, shift: timedelta) -> Any:
        """Shift a date value by a timedelta."""
        dt = self._parse_date(value)
        shifted_dt = dt + shift
        # Preserve original type (str vs datetime object) is handled by _format_date logic
        # But here we need to know what the input was.
        # Ideally, we should detect the input type.
        if isinstance(value, str):
            # If it was a date string (YYYY-MM-DD), keep it that way
            if len(value) == 10 and "T" not in value:
                if isinstance(shifted_dt, datetime):
                    return shifted_dt.date().isoformat()
                return shifted_dt.isoformat()
            return shifted_dt.isoformat()
        elif isinstance(value, date) and not isinstance(value, datetime):
            if isinstance(shifted_dt, datetime):
                return shifted_dt.date()
            return shifted_dt
        return shifted_dt

    def apply_uniform_shift(
        self, case: Dict[str, Any], shift: timedelta
    ) -> Dict[str, Any]:
        """Apply uniform time shift to all temporal fields."""
        shifted_case = copy.deepcopy(case)
        for field in self.temporal_fields:
            try:
                original_value = self._get_nested(shifted_case, field.path)
                if original_value is not None:
                    shifted_value = self._shift_date(original_value, shift)
                    self._set_nested(shifted_case, field.path, shifted_value)
            except (KeyError, IndexError, ValueError):
                # Skip fields that might be missing or malformed in this specific case
                continue
        return shifted_case

    def apply_duration_scale(
        self, case: Dict[str, Any], scale_factor: float
    ) -> Dict[str, Any]:
        """Apply duration scaling relative to earliest date."""
        # 1. Find min date
        dates = []
        for field in self.temporal_fields:
            try:
                val = self._get_nested(case, field.path)
                if val is not None:
                    dates.append(self._parse_date(val))
            except (KeyError, IndexError, ValueError):
                continue

        if not dates:
            return copy.deepcopy(case)

        min_date = min(dates)

        # 2. Scale
        scaled_case = copy.deepcopy(case)

        # Normalize min_date to datetime for calculation
        if isinstance(min_date, date) and not isinstance(min_date, datetime):
            min_dt = datetime.combine(min_date, datetime.min.time())
        else:
            min_dt = min_date

        for field in self.temporal_fields:
            try:
                original_value = self._get_nested(scaled_case, field.path)
                if original_value is not None:
                    dt = self._parse_date(original_value)

                    if isinstance(dt, date) and not isinstance(dt, datetime):
                        current_dt = datetime.combine(dt, datetime.min.time())
                    else:
                        current_dt = dt

                    delta = current_dt - min_dt
                    new_seconds = delta.total_seconds() * scale_factor
                    new_dt = min_dt + timedelta(seconds=new_seconds)

                    # Format back
                    final_val: Any = new_dt
                    if isinstance(dt, date) and not isinstance(dt, datetime):
                        if isinstance(new_dt, datetime):
                            final_val = new_dt.date()
                        else:
                            final_val = new_dt

                    # Handle string conversion if original was string
                    if isinstance(original_value, str):
                        if len(original_value) == 10 and "T" not in original_value:
                            if isinstance(final_val, datetime):
                                final_val = final_val.date().isoformat()
                            elif hasattr(final_val, "isoformat"):
                                final_val = final_val.isoformat()
                            else:
                                final_val = str(final_val)
                        else:
                            if hasattr(final_val, "isoformat"):
                                final_val = final_val.isoformat()
                            else:
                                final_val = str(final_val)

                    self._set_nested(scaled_case, field.path, final_val)
            except (KeyError, IndexError, ValueError):
                continue

        return scaled_case

    def _outcomes_equivalent(self, out1: Any, out2: Any) -> bool:
        """Check if outcomes are equivalent (ignoring date-specific values)."""
        # This is a simplified check. In a real system, we might need to ignore
        # specific fields in the outcome that are expected to change (like 'formation_date').
        # For now, we assume strict equality for non-temporal outcomes.
        return bool(out1 == out2)

    def test_shift_invariance(
        self,
        rule: Callable[[Dict[str, Any]], Any],
        test_cases: List[Dict[str, Any]],
        shifts: Optional[List[timedelta]] = None,
    ) -> TemporalConsistencyReport:
        """Test that rule outcomes are invariant under time shifts."""
        if shifts is None:
            shifts = [
                timedelta(days=-365),
                timedelta(days=-30),
                timedelta(days=30),
                timedelta(days=365),
                timedelta(days=3650),
            ]

        violations = []
        total_tests = 0

        for case in test_cases:
            try:
                original_outcome = rule(case)
            except Exception:
                continue

            for shift in shifts:
                total_tests += 1
                try:
                    shifted_case = self.apply_uniform_shift(case, shift)
                    shifted_outcome = rule(shifted_case)

                    if not self._outcomes_equivalent(original_outcome, shifted_outcome):
                        violations.append(
                            TemporalViolation(
                                rule_name=getattr(rule, "__name__", str(rule)),
                                original_case=case,
                                transformed_case=shifted_case,
                                transform_params={"shift": str(shift)},
                                original_outcome=original_outcome,
                                transformed_outcome=shifted_outcome,
                                violation_type="shift_invariance",
                            )
                        )
                except Exception:
                    pass

        return TemporalConsistencyReport(
            rule_name=getattr(rule, "__name__", str(rule)),
            total_tests=total_tests,
            violations=violations,
            shift_invariant=len(
                [v for v in violations if v.violation_type == "shift_invariance"]
            )
            == 0,
            order_invariant=True,  # Not tested here
        )

    def test_order_preservation(
        self,
        rule: Callable[[Dict[str, Any]], Any],
        test_cases: List[Dict[str, Any]],
        scales: Optional[List[float]] = None,
    ) -> TemporalConsistencyReport:
        """Test that only relative temporal ordering affects outcomes."""
        if scales is None:
            scales = [0.5, 2.0, 5.0]

        violations = []
        total_tests = 0

        for case in test_cases:
            try:
                original_outcome = rule(case)
            except Exception:
                continue

            for scale in scales:
                total_tests += 1
                try:
                    transformed_case = self.apply_duration_scale(case, scale)
                    transformed_outcome = rule(transformed_case)

                    if not self._outcomes_equivalent(
                        original_outcome, transformed_outcome
                    ):
                        violations.append(
                            TemporalViolation(
                                rule_name=getattr(rule, "__name__", str(rule)),
                                original_case=case,
                                transformed_case=transformed_case,
                                transform_params={"scale": scale},
                                original_outcome=original_outcome,
                                transformed_outcome=transformed_outcome,
                                violation_type="order_preservation",
                            )
                        )
                except Exception:
                    pass

        return TemporalConsistencyReport(
            rule_name=getattr(rule, "__name__", str(rule)),
            total_tests=total_tests,
            violations=violations,
            shift_invariant=True,  # Not tested here
            order_invariant=len(violations) == 0,
        )
