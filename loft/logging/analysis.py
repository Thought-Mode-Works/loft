"""
Log analysis utilities for LOFT system.

Tools for searching, filtering, and analyzing logs to debug issues
and extract insights.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LogEntry:
    """
    Parsed log entry.
    """

    timestamp: datetime
    level: str
    component: str
    function: str
    line: int
    message: str
    extras: Dict[str, Any]

    def matches(self, **filters: Any) -> bool:
        """
        Check if this log entry matches the given filters.

        Args:
            **filters: Key-value pairs to match

        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            if key == "level" and self.level != value:
                return False
            elif key == "component" and self.component != value:
                return False
            elif key == "function" and self.function != value:
                return False
            elif key == "message_contains" and value not in self.message:
                return False
            elif key == "after" and self.timestamp < value:
                return False
            elif key == "before" and self.timestamp > value:
                return False
        return True


class LogAnalyzer:
    """
    Utility for analyzing LOFT logs.
    """

    def __init__(self, log_dir: Path = Path("logs")):
        """
        Initialize log analyzer.

        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = log_dir

    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """
        Parse a log line into a LogEntry.

        Args:
            line: Raw log line

        Returns:
            LogEntry if parsing succeeded, None otherwise
        """
        # Pattern for loguru format
        pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| (\w+) \| ([^:]+):([^:]+):(\d+) \| (.+)$"

        match = re.match(pattern, line)
        if not match:
            return None

        timestamp_str, level, component, module, function, line_no, message = (
            match.groups()
        )

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None

        return LogEntry(
            timestamp=timestamp,
            level=level.strip(),
            component=component,
            function=function,
            line=int(line_no),
            message=message,
            extras={},
        )

    def search_logs(
        self,
        log_file: str = "loft.log",
        **filters: Any,
    ) -> List[LogEntry]:
        """
        Search logs with filters.

        Args:
            log_file: Log file name
            **filters: Filters to apply (level, component, function, etc.)

        Returns:
            List of matching log entries

        Example:
            >>> analyzer = LogAnalyzer()
            >>> entries = analyzer.search_logs(
            ...     log_file="llm_interactions.log",
            ...     component="llm",
            ...     after=datetime(2024, 1, 1)
            ... )
        """
        log_path = self.log_dir / log_file
        if not log_path.exists():
            return []

        entries = []
        with open(log_path, "r") as f:
            for line in f:
                entry = self.parse_log_line(line.strip())
                if entry and entry.matches(**filters):
                    entries.append(entry)

        return entries

    def get_llm_interactions(
        self,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> List[LogEntry]:
        """
        Get all LLM interactions within a time range.

        Args:
            after: Start time (inclusive)
            before: End time (inclusive)

        Returns:
            List of LLM interaction log entries
        """
        filters: Dict[str, Any] = {"component": "llm"}
        if after:
            filters["after"] = after
        if before:
            filters["before"] = before

        return self.search_logs(log_file="llm_interactions.log", **filters)

    def get_validation_failures(
        self,
        after: Optional[datetime] = None,
    ) -> List[LogEntry]:
        """
        Get all validation failures.

        Args:
            after: Start time (inclusive)

        Returns:
            List of validation failure log entries
        """
        filters: Dict[str, Any] = {
            "component": "validation",
            "message_contains": "FAIL",
        }
        if after:
            filters["after"] = after

        return self.search_logs(log_file="validation.log", **filters)

    def get_errors(
        self,
        after: Optional[datetime] = None,
    ) -> List[LogEntry]:
        """
        Get all errors.

        Args:
            after: Start time (inclusive)

        Returns:
            List of error log entries
        """
        filters: Dict[str, Any] = {}
        if after:
            filters["after"] = after

        return self.search_logs(log_file="errors.log", **filters)

    def aggregate_by_component(
        self,
        entries: List[LogEntry],
    ) -> Dict[str, int]:
        """
        Aggregate log entries by component.

        Args:
            entries: List of log entries

        Returns:
            Dictionary mapping component to count
        """
        counts: Dict[str, int] = defaultdict(int)
        for entry in entries:
            counts[entry.component] += 1
        return dict(counts)

    def aggregate_by_level(
        self,
        entries: List[LogEntry],
    ) -> Dict[str, int]:
        """
        Aggregate log entries by level.

        Args:
            entries: List of log entries

        Returns:
            Dictionary mapping level to count
        """
        counts: Dict[str, int] = defaultdict(int)
        for entry in entries:
            counts[entry.level] += 1
        return dict(counts)

    def get_metrics_over_time(
        self,
        entries: List[LogEntry],
        interval: timedelta = timedelta(hours=1),
    ) -> Dict[datetime, int]:
        """
        Get log entry counts over time.

        Args:
            entries: List of log entries
            interval: Time interval for aggregation

        Returns:
            Dictionary mapping timestamp to count
        """
        if not entries:
            return {}

        # Find time range
        start_time = min(entry.timestamp for entry in entries)
        end_time = max(entry.timestamp for entry in entries)

        # Create buckets
        buckets: Dict[datetime, int] = {}
        current_time = start_time
        while current_time <= end_time:
            buckets[current_time] = 0
            current_time += interval

        # Fill buckets
        for entry in entries:
            bucket_time = start_time
            while bucket_time <= entry.timestamp:
                bucket_time += interval
            bucket_time -= interval
            buckets[bucket_time] += 1

        return buckets

    def generate_report(
        self,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> str:
        """
        Generate a summary report of system activity.

        Args:
            after: Start time (inclusive)
            before: End time (inclusive)

        Returns:
            Formatted report string
        """
        filters: Dict[str, Any] = {}
        if after:
            filters["after"] = after
        if before:
            filters["before"] = before

        entries = self.search_logs(**filters)

        if not entries:
            return "No log entries found."

        # Aggregate statistics
        component_counts = self.aggregate_by_component(entries)
        level_counts = self.aggregate_by_level(entries)

        # LLM interactions
        llm_interactions = self.get_llm_interactions(after=after, before=before)

        # Validation failures
        validation_failures = self.get_validation_failures(after=after)

        # Errors
        errors = self.get_errors(after=after)

        # Build report
        report = []
        report.append("=" * 60)
        report.append("LOFT System Log Report")
        report.append("=" * 60)
        report.append("")

        if after:
            report.append(
                f"Time Range: {after.isoformat()} to {before.isoformat() if before else 'now'}"
            )
        report.append(f"Total Log Entries: {len(entries)}")
        report.append("")

        report.append("Log Levels:")
        for level, count in sorted(level_counts.items()):
            report.append(f"  {level:8s}: {count}")
        report.append("")

        report.append("Components:")
        for component, count in sorted(
            component_counts.items(), key=lambda x: x[1], reverse=True
        ):
            report.append(f"  {component:20s}: {count}")
        report.append("")

        report.append(f"LLM Interactions: {len(llm_interactions)}")
        report.append(f"Validation Failures: {len(validation_failures)}")
        report.append(f"Errors: {len(errors)}")
        report.append("")

        if errors:
            report.append("Recent Errors:")
            for error in errors[-5:]:  # Last 5 errors
                report.append(f"  [{error.timestamp}] {error.message[:80]}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def trace_request(self, request_id: float) -> List[LogEntry]:
        """
        Trace a specific request through the system.

        Args:
            request_id: Request ID to trace

        Returns:
            List of log entries related to this request
        """
        # Search all log files for this request_id
        entries = []
        for log_file in self.log_dir.glob("*.log"):
            with open(log_file, "r") as f:
                for line in f:
                    if str(request_id) in line:
                        entry = self.parse_log_line(line.strip())
                        if entry:
                            entries.append(entry)

        return sorted(entries, key=lambda e: e.timestamp)


def search_logs_cli(
    log_dir: str = "logs",
    component: Optional[str] = None,
    level: Optional[str] = None,
    after: Optional[str] = None,
    message_contains: Optional[str] = None,
) -> None:
    """
    Command-line interface for searching logs.

    Args:
        log_dir: Directory containing log files
        component: Filter by component
        level: Filter by log level
        after: Filter by time (ISO format)
        message_contains: Filter by message content
    """
    analyzer = LogAnalyzer(Path(log_dir))

    filters: Dict[str, Any] = {}
    if component:
        filters["component"] = component
    if level:
        filters["level"] = level
    if after:
        filters["after"] = datetime.fromisoformat(after)
    if message_contains:
        filters["message_contains"] = message_contains

    entries = analyzer.search_logs(**filters)

    print(f"Found {len(entries)} matching entries:\n")
    for entry in entries:
        print(
            f"[{entry.timestamp}] {entry.level:8s} {entry.component:15s} | {entry.message}"
        )
