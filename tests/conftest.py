"""Shared pytest fixtures.

The API budget guards are disabled suite-wide so tests that mock
``httpx.post`` are not throttled by the real rate-limit store. Tests that
exercise the guards (``test_rate_limit.py``) set their own env values and
store path, which take precedence.
"""

import pytest


@pytest.fixture(autouse=True)
def _disable_api_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYADLINK_MIN_INTERVAL_SECONDS",
        "DYADLINK_RATE_PER_MINUTE",
        "DYADLINK_SESSION_CALLS_PER_DAY",
        "DYADLINK_GLOBAL_CALLS_PER_DAY",
        "DYADLINK_DAILY_TOKEN_BUDGET",
    ):
        monkeypatch.setenv(name, "0")
