import pytest

from modules import rate_limit


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "rate.json"
    rate_limit._reset_store_for_tests(path)
    monkeypatch.setenv("DYADLINK_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("DYADLINK_RATE_PER_MINUTE", "0")
    monkeypatch.setenv("DYADLINK_SESSION_CALLS_PER_DAY", "0")
    monkeypatch.setenv("DYADLINK_GLOBAL_CALLS_PER_DAY", "0")
    monkeypatch.setenv("DYADLINK_DAILY_TOKEN_BUDGET", "0")
    yield path
    rate_limit._reset_store_for_tests(path)


def test_calls_allowed_then_recorded(store) -> None:
    rate_limit.set_current_token("visitor-1")
    rate_limit.check_allowed()
    rate_limit.record_call(total_tokens=120)
    snapshot = rate_limit.usage_snapshot()
    assert snapshot["session_calls"] == 1
    assert snapshot["session_tokens"] == 120
    assert snapshot["global_calls"] == 1


def test_session_daily_cap(store, monkeypatch) -> None:
    monkeypatch.setenv("DYADLINK_SESSION_CALLS_PER_DAY", "2")
    rate_limit.set_current_token("visitor-2")
    rate_limit.record_call()
    rate_limit.record_call()
    with pytest.raises(rate_limit.BudgetExceededError) as exc:
        rate_limit.check_allowed()
    assert exc.value.reason == "session_daily"


def test_global_daily_call_cap(store, monkeypatch) -> None:
    monkeypatch.setenv("DYADLINK_GLOBAL_CALLS_PER_DAY", "1")
    rate_limit.set_current_token("visitor-3")
    rate_limit.record_call()
    rate_limit.set_current_token("visitor-4")
    with pytest.raises(rate_limit.BudgetExceededError) as exc:
        rate_limit.check_allowed()
    assert exc.value.reason == "global_calls"


def test_min_interval_guard(store, monkeypatch) -> None:
    monkeypatch.setenv("DYADLINK_MIN_INTERVAL_SECONDS", "3600")
    rate_limit.set_current_token("visitor-5")
    rate_limit.record_call()
    with pytest.raises(rate_limit.BudgetExceededError) as exc:
        rate_limit.check_allowed()
    assert exc.value.reason == "interval"


def test_token_isolation(store, monkeypatch) -> None:
    monkeypatch.setenv("DYADLINK_SESSION_CALLS_PER_DAY", "1")
    rate_limit.set_current_token("visitor-6")
    rate_limit.record_call()
    with pytest.raises(rate_limit.BudgetExceededError):
        rate_limit.check_allowed()
    rate_limit.set_current_token("visitor-7")
    rate_limit.check_allowed()  # other visitor unaffected


def test_usage_snapshot_empty(store) -> None:
    rate_limit.set_current_token("nobody")
    snapshot = rate_limit.usage_snapshot()
    assert snapshot["session_calls"] == 0
    assert snapshot["global_calls"] == 0
