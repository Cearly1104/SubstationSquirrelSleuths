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

_LED_AVAILABLE = False
try:
    import Jetson.GPIO as GPIO
    PIN_R = 11
    PIN_G = 13
    PIN_B = 15
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([PIN_R, PIN_G, PIN_B], GPIO.OUT, initial=GPIO.LOW)
    _LED_AVAILABLE = True
except Exception:
    pass

def _set_led(r: bool, g: bool, b: bool) -> None:
    if not _LED_AVAILABLE:
        return
    try:
        import Jetson.GPIO as GPIO
        GPIO.output(PIN_R, GPIO.HIGH if r else GPIO.LOW)
        GPIO.output(PIN_G, GPIO.HIGH if g else GPIO.LOW)
        GPIO.output(PIN_B, GPIO.HIGH if b else GPIO.LOW)
    except Exception:
        pass

def _update_led(data: dict) -> None:
    error     = data.get("error", False)
    recording = data.get("recording", False)
    analyzing = data.get("analyzing", False)
    if error:
        _set_led(r=True,  g=False, b=False)  # red    - error
    elif analyzing:
        _set_led(r=True,  g=False, b=True)   # purple - analyzing
    elif recording:
        _set_led(r=True,  g=True,  b=False)  # yellow - recording
    else:
        _set_led(r=False, g=True,  b=False)  # green  - idle


def _path(det_dir: Path) -> Path:
    return Path(det_dir) / "pipeline_status.json"


def _update(det_dir: Path, updates: dict) -> None:
    """Read-modify-write pipeline_status.json under a process-local lock."""
    path = _path(det_dir)
    with _lock:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {"recording": False, "analyzing": False, "error": False, "error_msg": ""}
        data.update(updates)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        _update_led(data)


def set_recording(det_dir: Path, value: bool) -> None:
    _update(det_dir, {"recording": value})

def set_analyzing(det_dir: Path, value: bool) -> None:
    _update(det_dir, {"analyzing": value})

def set_error(det_dir: Path, message: str) -> None:
    _update(det_dir, {"error": True, "error_msg": message})

def clear_error(det_dir: Path) -> None:
    _update(det_dir, {"error": False, "error_msg": ""})


def read(det_dir: Path) -> dict:
    try:
        with open(_path(det_dir)) as f:
            return json.load(f)
    except Exception:
         return {"recording": False, "analyzing": False, "error": False, "error_msg": ""}
