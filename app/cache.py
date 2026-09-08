import hashlib
import json
import time
from pathlib import Path

from app.config import get_settings

CACHE_DIR = Path(".cache")


def _path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.strip().lower().encode()).hexdigest()[:32]
    return CACHE_DIR / namespace / f"{digest}.json"


def get(namespace: str, key: str):
    try:
        payload = json.loads(_path(namespace, key).read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    age_hours = (time.time() - payload.get("ts", 0)) / 3600
    if age_hours > get_settings().cache_ttl_hours:
        return None
    return payload.get("value")


def put(namespace: str, key: str, value) -> None:
    path = _path(namespace, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ts": time.time(), "value": value}))
    except (OSError, TypeError, ValueError):
        pass
