"""Pytest configuration file for bench tests."""

import pytest
import os


@pytest.fixture(autouse=True)
def clean_environment():
    """Fixture to clean environment variables between tests."""
    # Store original environment
    initial_env = os.environ.copy()

    # Run the test
    yield

    # Clean up environment variables specifically for our tests
    for key in list(os.environ.keys()):
        if key.startswith("BENCH_") and key not in initial_env:
            os.environ.pop(key)
        elif key.startswith("BENCH_") and key in initial_env:
            os.environ[key] = initial_env[key]
