import json
import os

_LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
_LOG_FILE = os.path.join(_LOGS_DIR, "sessions.jsonl")


def log_session(entry: dict) -> None:
    """Append a session entry to logs/sessions.jsonl (newline-delimited JSON)."""
    os.makedirs(_LOGS_DIR, exist_ok=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
