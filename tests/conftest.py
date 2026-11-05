"""Shared pytest fixtures for Lab 05 unit tests."""

import pytest

from latedeliveries.pipeline import load_orders, split_orders


@pytest.fixture(scope="session")
def orders():
    """Load the full orders dataset once per test session."""
    return load_orders()


@pytest.fixture(scope="session")
def split(orders):
    """Split the orders once per session into X_train, X_test, y_train, y_test."""
    return split_orders(orders)


@pytest.fixture
def sample_orders_df(orders):
    """Provide a small slice of orders for quick component testing."""
    return orders.head(50)
