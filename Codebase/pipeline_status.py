"""
pipeline_status.py — Shared recording/analysis state file.

Written by the pipeline, read by the Flask web app.
File location: detections/pipeline_status.json
"""

import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()


def _path(det_dir: Path) -> Path:
    return Path(det_dir) / "pipeline_status.json"


def _update(det_dir: Path, key: str, value: bool) -> None:
    """Read-modify-write pipeline_status.json under a process-local lock."""
    path = _path(det_dir)
    with _lock:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {"recording": False, "analyzing": False}
        data[key] = value
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)


def set_recording(det_dir: Path, value: bool) -> None:
    _update(det_dir, "recording", value)


def set_analyzing(det_dir: Path, value: bool) -> None:
    _update(det_dir, "analyzing", value)


def read(det_dir: Path) -> dict:
    try:
        with open(_path(det_dir)) as f:
            return json.load(f)
    except Exception:
        return {"recording": False, "analyzing": False}
