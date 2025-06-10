"""
Pytest configuration for integration tests.
"""

import pytest

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test that makes real API calls"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests in integration directory."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration) 