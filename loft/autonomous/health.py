"""
Health Endpoint Server for Autonomous Test Harness.

This module provides an HTTP health endpoint for Docker HEALTHCHECK
and monitoring integration.

Features:
- Lightweight HTTP server
- JSON status responses
- Docker HEALTHCHECK compatible
- Non-blocking background operation
"""

import json
import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from loft.autonomous.config import HealthConfig
from loft.autonomous.schemas import RunState, RunStatus

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status representation.

    Attributes:
        healthy: Whether the service is healthy
        status: Current run status
        run_id: Current run ID
        uptime_seconds: Server uptime
        progress: Run progress if available
        last_updated: Last state update time
    """

    def __init__(
        self,
        healthy: bool = True,
        status: str = "idle",
        run_id: Optional[str] = None,
        uptime_seconds: float = 0.0,
        progress: Optional[Dict[str, Any]] = None,
        last_updated: Optional[datetime] = None,
    ):
        """Initialize health status.

        Args:
            healthy: Whether service is healthy
            status: Current status string
            run_id: Current run ID
            uptime_seconds: Server uptime
            progress: Progress dictionary
            last_updated: Last update time
        """
        self.healthy = healthy
        self.status = status
        self.run_id = run_id
        self.uptime_seconds = uptime_seconds
        self.progress = progress or {}
        self.last_updated = last_updated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "healthy": self.healthy,
            "status": self.status,
            "run_id": self.run_id,
            "uptime_seconds": self.uptime_seconds,
            "progress": self.progress,
            "last_updated": (self.last_updated.isoformat() if self.last_updated else None),
            "timestamp": datetime.now().isoformat(),
        }


class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health endpoint."""

    server: "HealthServer"

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health" or self.path == "/":
            self._handle_health()
        elif self.path == "/status":
            self._handle_status()
        elif self.path == "/ready":
            self._handle_ready()
        else:
            self._send_response(404, {"error": "Not found"})

    def _handle_health(self) -> None:
        """Handle health check request."""
        health_status = self.server.get_health_status()
        status_code = 200 if health_status.healthy else 503
        self._send_response(status_code, health_status.to_dict())

    def _handle_status(self) -> None:
        """Handle detailed status request."""
        health_status = self.server.get_health_status()
        self._send_response(200, health_status.to_dict())

    def _handle_ready(self) -> None:
        """Handle readiness check."""
        health_status = self.server.get_health_status()
        ready = health_status.status in ["running", "idle", "completed"]
        status_code = 200 if ready else 503
        self._send_response(status_code, {"ready": ready, "status": health_status.status})

    def _send_response(self, status_code: int, data: Dict[str, Any]) -> None:
        """Send JSON response.

        Args:
            status_code: HTTP status code
            data: Response data
        """
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))


class HealthServer:
    """HTTP server for health checks.

    Runs in a background thread and provides health status
    for Docker HEALTHCHECK and monitoring systems.

    Attributes:
        config: Health endpoint configuration
    """

    def __init__(self, config: HealthConfig):
        """Initialize the health server.

        Args:
            config: Health configuration
        """
        self._config = config
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[datetime] = None
        self._current_state: Optional[RunState] = None
        self._running = False

    @property
    def config(self) -> HealthConfig:
        """Get configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    def start(self) -> None:
        """Start the health server in background thread."""
        if not self._config.enabled:
            logger.info("Health endpoint disabled")
            return

        if self._running:
            logger.warning("Health server already running")
            return

        try:
            self._server = HTTPServer(
                (self._config.host, self._config.port),
                HealthRequestHandler,
            )
            self._server.timeout = 1.0  # Short timeout for checking _running flag
            self._server.get_health_status = self.get_health_status

            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._start_time = datetime.now()
            self._running = True
            self._thread.start()

            logger.info(f"Health server started on {self._config.host}:{self._config.port}")

        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop the health server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.server_close()
            self._server = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        logger.info("Health server stopped")

    def update_state(self, state: RunState) -> None:
        """Update current run state.

        Args:
            state: New run state
        """
        self._current_state = state

    def get_health_status(self) -> HealthStatus:
        """Get current health status.

        Returns:
            HealthStatus object
        """
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        if self._current_state is None:
            return HealthStatus(
                healthy=True,
                status="idle",
                uptime_seconds=uptime,
            )

        healthy = self._current_state.status not in [RunStatus.FAILED]
        progress_dict = (
            self._current_state.progress.to_dict() if self._current_state.progress else {}
        )

        return HealthStatus(
            healthy=healthy,
            status=self._current_state.status.value,
            run_id=self._current_state.run_id,
            uptime_seconds=uptime,
            progress=progress_dict,
            last_updated=self._current_state.last_updated,
        )

    def _serve(self) -> None:
        """Serve requests in background."""
        while self._running and self._server:
            try:
                self._server.handle_request()
            except Exception as e:
                if self._running:
                    logger.error(f"Health server error: {e}")


def create_health_server(config: HealthConfig) -> HealthServer:
    """Factory function to create a health server.

    Args:
        config: Health configuration

    Returns:
        Configured HealthServer
    """
    return HealthServer(config)
