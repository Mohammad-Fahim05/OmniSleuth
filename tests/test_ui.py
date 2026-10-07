"""Unit tests for Phase 15 - Streamlit UI module."""

import importlib
import pytest


def test_ui_module_imports():
    """Verify that the UI app module can be imported and defined without syntax errors."""
    mod = importlib.import_module("omnisleuth.ui.app")
    assert mod is not None
