# utils.py
import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
HISTORY_JSON = os.path.join(OUTPUT_DIR, "filtered_pairs_history.json")

def ensure_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def ts_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def append_history(payload: Dict[str, Any]):
    ensure_dirs()
    record = {"ts": ts_now_iso(), **payload}
    if not os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump([record], f, ensure_ascii=False, indent=2)
        return
    try:
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    arr.append(record)
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
