"""Cost protection for third-party API calls.

Two independent guards, both keyed by the visitor's URL token:

* a per-session sliding-window rate limit plus a daily call cap, and
* a server-wide daily token budget so concentrated traffic (e.g. a room
  full of judges) cannot exhaust the DeepSeek key.

State lives in a small JSON file in the temp directory — same pattern as
``session_store`` — so limits survive process restarts. Limits of ``0``
disable the corresponding guard. All thresholds are env-configurable; see
``.env.example``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
_STORE_PATH = Path(
    os.getenv(
        "DYADLINK_RATE_STORE",
        str(Path(tempfile.gettempdir()) / "dyadlink_rate_limit.json"),
    )
)
_WINDOW_SECONDS = 60.0
_RETENTION_SECONDS = 26 * 3600.0
_MAX_ENTRIES = 2000
_ANONYMOUS = "anonymous"

_current_token = _ANONYMOUS


class BudgetExceededError(Exception):
    """Raised when a rate or budget guard refuses an API call."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def set_current_token(token: str) -> None:
    """Remember which visitor token this script run belongs to."""
    global _current_token
    _current_token = token or _ANONYMOUS


def current_token() -> str:
    return _current_token


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _config() -> dict[str, float]:
    return {
        "per_minute": _env_int("DYADLINK_RATE_PER_MINUTE", 12),
        "min_interval": _env_float("DYADLINK_MIN_INTERVAL_SECONDS", 1.5),
        "session_daily": _env_int("DYADLINK_SESSION_CALLS_PER_DAY", 300),
        "daily_token_budget": _env_int("DYADLINK_DAILY_TOKEN_BUDGET", 500000),
        "global_daily_calls": _env_int("DYADLINK_GLOBAL_CALLS_PER_DAY", 5000),
    }


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load() -> dict:
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(store: dict) -> None:
    try:
        _STORE_PATH.write_text(
            json.dumps(store, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def _prune(store: dict, now: float) -> None:
    cutoff = now - _RETENTION_SECONDS
    today = _today()
    tokens = [k for k in store if k != "__global__"]
    for key in tokens:
        record = store.get(key)
        if not isinstance(record, dict):
            store.pop(key, None)
            continue
        calls = record.get("calls")
        if isinstance(calls, list):
            record["calls"] = [
                ts for ts in calls if isinstance(ts, (int, float)) and ts >= cutoff
            ]
        days = record.get("days")
        if isinstance(days, dict):
            record["days"] = {
                day: value
                for day, value in days.items()
                if day >= today and isinstance(value, dict)
            }
    # Bound store growth: drop the oldest records beyond the cap.
    if len(tokens) > _MAX_ENTRIES:
        for key in tokens[: len(tokens) - _MAX_ENTRIES]:
            store.pop(key, None)


def _token_record(store: dict, token: str, now: float) -> dict:
    record = store.get(token)
    if not isinstance(record, dict):
        record = {"calls": [], "days": {}}
        store[token] = record
    calls = record.get("calls")
    if not isinstance(calls, list):
        calls = []
        record["calls"] = calls
    record["calls"] = [ts for ts in calls if isinstance(ts, (int, float)) and ts >= now - _WINDOW_SECONDS]
    days = record.get("days")
    if not isinstance(days, dict):
        days = {}
        record["days"] = days
    today = days.get(_today())
    if not isinstance(today, dict):
        today = {"calls": 0, "tokens": 0}
        days[_today()] = today
    return record


def _global_record(store: dict) -> dict:
    record = store.get("__global__")
    if not isinstance(record, dict):
        record = {"days": {}}
        store["__global__"] = record
    days = record.get("days")
    if not isinstance(days, dict):
        days = {}
        record["days"] = days
    today = days.get(_today())
    if not isinstance(today, dict):
        today = {"calls": 0, "tokens": 0}
        days[_today()] = today
    return record


def check_allowed(token: str | None = None) -> None:
    """Raise :class:`BudgetExceededError` when a guard refuses the call."""
    token = token or current_token()
    cfg = _config()
    now = time.time()

    with _LOCK:
        store = _load()
        _prune(store, now)
        record = _token_record(store, token, now)
        calls: list[float] = record["calls"]
        today = record["days"][_today()]
        glob = _global_record(store)
        global_today = glob["days"][_today()]

        if cfg["min_interval"] > 0 and calls:
            if now - calls[-1] < cfg["min_interval"]:
                raise BudgetExceededError("interval")
        if cfg["per_minute"] > 0 and len(calls) >= cfg["per_minute"]:
            raise BudgetExceededError("rate")
        if cfg["session_daily"] > 0 and today.get("calls", 0) >= cfg["session_daily"]:
            raise BudgetExceededError("session_daily")
        if (
            cfg["global_daily_calls"] > 0
            and global_today.get("calls", 0) >= cfg["global_daily_calls"]
        ):
            raise BudgetExceededError("global_calls")
        if cfg["daily_token_budget"] > 0 and (
            global_today.get("tokens", 0) >= cfg["daily_token_budget"]
        ):
            raise BudgetExceededError("global_budget")
        _save(store)


def record_call(token: str | None = None, total_tokens: int = 0) -> None:
    """Account for one finished API call (call it after success only)."""
    token = token or current_token()
    now = time.time()
    tokens = total_tokens if type(total_tokens) is int and total_tokens > 0 else 0

    with _LOCK:
        store = _load()
        _prune(store, now)
        record = _token_record(store, token, now)
        record["calls"].append(now)
        today = record["days"][_today()]
        today["calls"] = today.get("calls", 0) + 1
        today["tokens"] = today.get("tokens", 0) + tokens
        glob = _global_record(store)
        global_today = glob["days"][_today()]
        global_today["calls"] = global_today.get("calls", 0) + 1
        global_today["tokens"] = global_today.get("tokens", 0) + tokens
        _save(store)


def usage_snapshot(token: str | None = None) -> dict[str, int]:
    """Return today's usage for display in settings/debug UI."""
    token = token or current_token()
    with _LOCK:
        store = _load()
        record = store.get(token)
        today = {}
        if isinstance(record, dict):
            days = record.get("days")
            if isinstance(days, dict) and isinstance(days.get(_today()), dict):
                today = days[_today()]
        glob = store.get("__global__")
        global_today = {}
        if isinstance(glob, dict):
            days = glob.get("days")
            if isinstance(days, dict) and isinstance(days.get(_today()), dict):
                global_today = days[_today()]
    return {
        "session_calls": int(today.get("calls", 0)),
        "session_tokens": int(today.get("tokens", 0)),
        "global_calls": int(global_today.get("calls", 0)),
        "global_tokens": int(global_today.get("tokens", 0)),
        "daily_token_budget": int(_config()["daily_token_budget"]),
    }


def _reset_store_for_tests(path: str | Path) -> None:
    """Point the store at ``path`` and clear it (tests only)."""
    global _STORE_PATH
    _STORE_PATH = Path(path)
    try:
        _STORE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
