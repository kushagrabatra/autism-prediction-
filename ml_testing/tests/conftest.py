"""Pytest configuration and shared fixtures for the test suite."""

import sys
from pathlib import Path

# Ensure the project root (ml_testing/) is on sys.path so that
# ``import app`` and ``import main`` work without package installation.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
