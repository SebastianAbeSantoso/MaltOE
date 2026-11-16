"""Shared test setup.

Tests that download the public evaluation datasets are opt-in, so the default run (and the
post-task hook) is fast and works offline:

    MALTOE_TEST_NETWORK=1 uv run pytest
"""

import os

import pytest

NETWORK = os.environ.get("MALTOE_TEST_NETWORK") == "1"
needs_network = pytest.mark.skipif(
    not NETWORK, reason="set MALTOE_TEST_NETWORK=1 to download datasets"
)


@pytest.fixture(scope="session")
def dataset_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("harness-cache")
