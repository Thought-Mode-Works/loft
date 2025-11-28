"""
Automated Casework Exploration for LOFT

This module implements batch exploration of legal scenarios to track
the system's learning curve over multiple cases.
"""

from .explorer import CaseworkExplorer
from .dataset_loader import DatasetLoader
from .metrics import LearningMetrics
from .reporting import ReportGenerator

__all__ = ["CaseworkExplorer", "DatasetLoader", "LearningMetrics", "ReportGenerator"]
