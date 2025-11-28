"""
Interactive CLI Playground for LOFT

This module provides an interactive CLI for exploring LOFT's capabilities:
- NL <-> ASP translation
- Gap identification
- Rule generation
- Validation pipeline
- Rule incorporation
"""

from .session import PlaygroundSession
from .cli import main

__all__ = ["PlaygroundSession", "main"]
