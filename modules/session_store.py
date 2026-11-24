"""Token-keyed server-side persistence for consent and conversation history.

A short token is placed in the page URL (?c=...) so it survives refresh
without relying on browser storage (localStorage can be blocked by
tracking-prevention settings). Consent flags and merged chat messages are
kept in a JSON file on the app host; nothing leaves the server.
"""

import json
import tempfile
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_STORE_PATH = Path(tempfile.gettempdir()) / "dyadlink_store.json"
_MAX_TOKENS = 500
_MAX_MESSAGES = 400


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


def _record(token: str) -> dict:
    with _LOCK:
        store = _load()
        if token not in store:
            if len(store) >= _MAX_TOKENS:
                oldest = next(iter(store))
                store.pop(oldest, None)
            store[token] = {}
            _save(store)
        return store[token]


def get_record(token: str) -> dict:
    if not token:
        return {}
    with _LOCK:
        record = _load().get(token)
    return record if isinstance(record, dict) else {}


def set_consent(token: str) -> None:
    with _LOCK:
        store = _load()
        record = store.setdefault(token, {})
        record["consent"] = "1"
        _save(store)


def has_consent(token: str) -> bool:
    return get_record(token).get("consent") == "1"


def get_messages(token: str) -> list[dict]:
    messages = get_record(token).get("messages")
    return messages if isinstance(messages, list) else []


def save_messages(token: str, messages: list[dict]) -> None:
    with _LOCK:
        store = _load()
        record = store.setdefault(token, {})
        record["messages"] = messages[-_MAX_MESSAGES:]
        _save(store)


_MAX_CAPTURES = 16


def get_captures(token: str) -> list[dict]:
    captures = get_record(token).get("captures")
    return captures if isinstance(captures, list) else []


def save_captures(token: str, captures: list[dict]) -> None:
    """Persist sanitized capture dicts; keeps the most recent entries."""
    if not token:
        return
    clean = [c for c in captures if isinstance(c, dict)]
    with _LOCK:
        store = _load()
        record = store.setdefault(token, {})
        record["captures"] = clean[-_MAX_CAPTURES:]
        _save(store)


_MAX_SAMPLES = 200
_MAX_FEATURES_PER_SAMPLE = 24


def get_feature_samples(token: str) -> list[dict]:
    samples = get_record(token).get("feature_samples")
    return samples if isinstance(samples, list) else []


def append_feature_sample(token: str, sample: dict) -> None:
    """Append one numeric feature sample; keeps the most recent entries."""
    if not token or not isinstance(sample, dict) or not sample:
        return
    clean = {
        key: float(value)
        for key, value in list(sample.items())[:_MAX_FEATURES_PER_SAMPLE]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not clean:
        return
    clean["ts"] = time.time()
    with _LOCK:
        store = _load()
        record = store.setdefault(token, {})
        samples = record.get("feature_samples")
        if not isinstance(samples, list):
            samples = []
        samples.append(clean)
        record["feature_samples"] = samples[-_MAX_SAMPLES:]
        _save(store)


def get_settings(token: str) -> dict:
    settings = get_record(token).get("settings")
    return settings if isinstance(settings, dict) else {}


def save_settings(token: str, settings: dict) -> None:
    """Persist small user preferences (e.g. accessibility options)."""
    if not token or not isinstance(settings, dict):
        return
    clean = {
        str(key)[:40]: value
        for key, value in list(settings.items())[:20]
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    with _LOCK:
        store = _load()
        record = store.setdefault(token, {})
        record["settings"] = clean
        _save(store)


def clear_token(token: str) -> None:
    if not token:
        return
    with _LOCK:
        store = _load()
        if token in store:
            del store[token]
            _save(store)
