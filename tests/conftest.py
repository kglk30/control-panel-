"""pytest configuration and shared fixtures."""

import pytest_asyncio

from control_panel.persistence.database import close_database


@pytest_asyncio.fixture(autouse=True)
async def close_test_database():
    yield
    await close_database()
