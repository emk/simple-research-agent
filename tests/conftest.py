"""Test configuration for pytest."""

from __future__ import annotations

import pytest

from simple_research.client import Client


@pytest.fixture(scope="session")
def dotenv() -> None:
    """Load the .env file for testing."""
    from dotenv import load_dotenv

    load_dotenv()


@pytest.fixture(scope="session")
def client(dotenv: None) -> Client:
    """Create a client for testing."""
    from simple_research.client import Client

    return Client()
