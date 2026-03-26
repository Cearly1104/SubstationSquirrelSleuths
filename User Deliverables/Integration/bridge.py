"""
bridge.py — Integration bridge between the Visual Analysis subsystem and
the User Deliverables (Flask) subsystem.

The analysis subsystem produces:
  - Detection events (species, timestamp, confidence)
  - Captured images (annotated frames)
  - Recorded video clips (tracking sequences, heatmaps)
  - System telemetry (uptime, GPU temp, storage, camera health)

This bridge watches the analysis output directory and mirrors relevant
files into the Flask data directory, organized by date.  It also pushes
events and system stats via the Flask API so the dashboard updates in
near-real-time.

Usage:
    python bridge.py                         # uses defaults / env vars
    SSS_ANALYSIS_DIR=/path/to/output python bridge.py

Environment variables:
    SSS_ANALYSIS_DIR  — where the analysis subsystem writes output
    SSS_DATA_DIR      — the Flask app's data directory (same var the app uses)
    SSS_API_URL       — Flask API base URL   (default: http://127.0.0.1:5000)
    SSS_API_KEY       — shared API key        (default: empty / no auth)
    SSS_POLL_INTERVAL — seconds between polls (default: 5)
"""

import json
import os
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANALYSIS_DIR = Path(os.environ.get("SSS_ANALYSIS_DIR", "/home/sss/analysis_output"))
DATA_DIR = Path(os.environ.get("SSS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
API_URL = os.environ.get("SSS_API_URL", "http://127.0.0.1:5000")
API_KEY = os.environ.get("SSS_API_KEY", "")
POLL_INTERVAL = int(os.environ.get("SSS_POLL_INTERVAL", "5"))

LOG = logging.getLogger("sss-bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

# Track files we have already processed so we don't duplicate work.
_seen_files: set[str] = set()
_seen_events: set[str] = set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _ensure_day_dirs(day: str):
    """Create the day folder structure under DATA_DIR."""
    for sub in ("images", "videos"):
        (DATA_DIR / day / sub).mkdir(parents=True, exist_ok=True)


def _post_event(day: str, ts: str, event_type: str):
    """Push a detection event to the Flask API."""
    try:
        resp = requests.post(
            f"{API_URL}/api/events",
            headers=_api_headers(),
            json={"day": day, "ts": ts, "event": event_type},
            timeout=5,
        )
        if resp.ok:
            LOG.info("Event pushed: %s %s", ts, event_type)
        else:
            LOG.warning("Event push failed (%s): %s", resp.status_code, resp.text)
    except requests.RequestException as exc:
        LOG.warning("Event push error: %s", exc)


def _post_system_stats(stats: dict):
    """Push system telemetry to the Flask API."""
    try:
        resp = requests.post(
            f"{API_URL}/api/system-stats",
            headers=_api_headers(),
            json=stats,
            timeout=5,
        )
        if not resp.ok:
            LOG.warning("Stats push failed (%s): %s", resp.status_code, resp.text)
    except requests.RequestException as exc:
        LOG.warning("Stats push error: %s", exc)


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------

def sync_media():
    """Copy new images/videos from the analysis output into DATA_DIR."""
    if not ANALYSIS_DIR.is_dir():
        return

    day = _today()
    _ensure_day_dirs(day)

    # The analysis subsystem is expected to write files into sub-folders:
    #   ANALYSIS_DIR/images/   — annotated frames (jpg/png)
    #   ANALYSIS_DIR/videos/   — recorded clips   (mp4)
    for media_type in ("images", "videos"):
        src_dir = ANALYSIS_DIR / media_type
        if not src_dir.is_dir():
            continue
        dst_dir = DATA_DIR / day / media_type
        for f in src_dir.iterdir():
            if not f.is_file():
                continue
            key = f"{media_type}/{f.name}"
            if key in _seen_files:
                continue
            dst = dst_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                LOG.info("Copied %s -> %s", f, dst)
            _seen_files.add(key)


def sync_events():
    """Read new events from the analysis subsystem's event log."""
    events_file = ANALYSIS_DIR / "events.json"
    if not events_file.is_file():
        return

    try:
        with open(events_file) as fh:
            events = json.load(fh)
    except Exception:
        return

    day = _today()
    for entry in events:
        ts = entry.get("ts", "")
        etype = entry.get("event", "unknown")
        key = f"{ts}|{etype}"
        if key in _seen_events:
            continue
        _seen_events.add(key)
        _post_event(day, ts, etype)


def sync_system_stats():
    """Read and forward system telemetry."""
    stats_file = ANALYSIS_DIR / "system_stats.json"
    if not stats_file.is_file():
        return

    try:
        with open(stats_file) as fh:
            stats = json.load(fh)
        _post_system_stats(stats)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    LOG.info("SSS Bridge starting")
    LOG.info("  Analysis dir : %s", ANALYSIS_DIR)
    LOG.info("  Data dir     : %s", DATA_DIR)
    LOG.info("  API URL      : %s", API_URL)
    LOG.info("  Poll interval: %ds", POLL_INTERVAL)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            sync_media()
            sync_events()
            sync_system_stats()
        except Exception:
            LOG.exception("Sync cycle error")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
