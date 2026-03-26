"""
Substation Squirrel Sleuths — Flask Application
Serves the user deliverables dashboard over a local WiFi access point
hosted on the Jetson Orin Nano.
"""

import os
import json
import secrets
import time
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_from_directory, abort, Response
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Shared data directory where the analysis subsystem writes outputs.
# On the Jetson this will be an absolute path; default works for local dev.
DATA_DIR = Path(os.environ.get("SSS_DATA_DIR", BASE_DIR / "data"))

# Static demo assets (shipped with repo — training images, demo videos, etc.)
DEMO_ASSETS_DIR = BASE_DIR / "assets"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.environ.get("SSS_SECRET_KEY", secrets.token_hex(32))

# ---------------------------------------------------------------------------
# User accounts — loaded from a JSON file so they can be managed on-device
# without touching Python code.  Falls back to a default admin account.
# ---------------------------------------------------------------------------

ACCOUNTS_FILE = Path(os.environ.get("SSS_ACCOUNTS_FILE", BASE_DIR / "accounts.json"))


def _load_accounts():
    if ACCOUNTS_FILE.is_file():
        with open(ACCOUNTS_FILE) as f:
            return json.load(f)
    return [{"username": "admin", "password": "sss"}]


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Data helpers — read from the shared data directory
# ---------------------------------------------------------------------------

def _read_json(path, default=None):
    """Safely read a JSON file, returning *default* on any failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def get_days():
    """Return a sorted list of day folders available in DATA_DIR.

    Each day folder is expected to contain:
      events.json   — list of {ts, event}
      images/       — captured stills
      videos/       — recorded clips
    """
    if not DATA_DIR.is_dir():
        return []
    days = []
    for entry in sorted(DATA_DIR.iterdir(), reverse=True):
        if entry.is_dir() and _looks_like_date(entry.name):
            events = _read_json(entry / "events.json", [])
            images = _list_media(entry / "images")
            videos = _list_media(entry / "videos")
            days.append({
                "day": entry.name,
                "events": events,
                "images": images,
                "videos": videos,
            })
    return days


def _looks_like_date(name):
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _list_media(folder):
    if not folder.is_dir():
        return []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
    return sorted(
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in allowed
    )


def get_system_stats():
    """Read system stats written by the analysis subsystem."""
    stats_file = DATA_DIR / "system_stats.json"
    defaults = {
        "uptime": "N/A",
        "storage_free": "N/A",
        "gpu_temp": "N/A",
        "status": "OFFLINE",
    }
    return _read_json(stats_file, defaults)


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        accounts = _load_accounts()
        match = next(
            (a for a in accounts
             if a["username"].lower() == username and a["password"] == password),
            None,
        )
        if match:
            session["user"] = match["username"]
            next_page = request.args.get("next", url_for("dashboard"))
            return redirect(next_page)
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    days = get_days()
    today = datetime.now().strftime("%Y-%m-%d")
    today_data = next((d for d in days if d["day"] == today), None)
    stats = get_system_stats()
    return render_template(
        "dashboard.html",
        user=session["user"],
        today=today,
        today_data=today_data,
        stats=stats,
        days=days,
    )


@app.route("/archive")
@login_required
def archive():
    days = get_days()
    return render_template("archive.html", user=session["user"], days=days)


@app.route("/day/<day_id>")
@login_required
def day_detail(day_id):
    days = get_days()
    day_data = next((d for d in days if d["day"] == day_id), None)
    if not day_data:
        abort(404)
    return render_template("day.html", user=session["user"], day=day_data)


# ---------------------------------------------------------------------------
# Routes — Serve media from the data directory
# ---------------------------------------------------------------------------

@app.route("/data/<day_id>/images/<filename>")
@login_required
def serve_image(day_id, filename):
    folder = DATA_DIR / day_id / "images"
    if not folder.is_dir():
        abort(404)
    return send_from_directory(str(folder), filename)


@app.route("/data/<day_id>/videos/<filename>")
@login_required
def serve_video(day_id, filename):
    folder = DATA_DIR / day_id / "videos"
    if not folder.is_dir():
        abort(404)
    return send_from_directory(str(folder), filename)


@app.route("/demo/<path:filename>")
@login_required
def serve_demo_asset(filename):
    return send_from_directory(str(DEMO_ASSETS_DIR), filename)


# ---------------------------------------------------------------------------
# Routes — MJPEG live feed (from PoE cameras via RTSP)
# ---------------------------------------------------------------------------

# Camera config loaded from cameras.json:
#   [{"id": "cam0", "url": "rtsp://192.168.8.10:554/stream"}, ...]
# Supports RTSP, HTTP MJPEG, or any URL that OpenCV can open.

CAMERAS_FILE = Path(os.environ.get("SSS_CAMERAS_FILE", BASE_DIR / "cameras.json"))


def _load_cameras():
    if CAMERAS_FILE.is_file():
        with open(CAMERAS_FILE) as f:
            return {cam["id"]: cam["url"] for cam in json.load(f)}
    return {}


def _mjpeg_stream(stream_url):
    """Generator that reads frames from a camera and yields MJPEG."""
    import cv2
    cap = cv2.VideoCapture(stream_url)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            _, buf = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/feed/<cam_id>")
@login_required
def live_feed(cam_id):
    """MJPEG endpoint — proxies a PoE camera's RTSP stream as browser-viewable MJPEG."""
    cameras = _load_cameras()
    if cam_id not in cameras:
        abort(404)
    return Response(
        _mjpeg_stream(cameras[cam_id]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Routes — API (used by the analysis subsystem to push data)
# ---------------------------------------------------------------------------

@app.route("/api/events", methods=["POST"])
def api_post_event():
    """Receive a detection event from the analysis subsystem.

    Expected JSON body: {day: "YYYY-MM-DD", ts: "...", event: "detection"|...}
    An API key is required via the X-API-Key header.
    """
    api_key = os.environ.get("SSS_API_KEY", "")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True)
    if not payload or "day" not in payload:
        return jsonify({"error": "bad request"}), 400

    day = payload["day"]
    day_dir = DATA_DIR / day
    day_dir.mkdir(parents=True, exist_ok=True)

    events_file = day_dir / "events.json"
    events = _read_json(events_file, [])
    events.append({"ts": payload.get("ts", ""), "event": payload.get("event", "unknown")})
    with open(events_file, "w") as f:
        json.dump(events, f, indent=2)

    return jsonify({"ok": True})


@app.route("/api/system-stats", methods=["POST"])
def api_post_stats():
    """Receive system stats from the analysis subsystem."""
    api_key = os.environ.get("SSS_API_KEY", "")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "bad request"}), 400

    stats_file = DATA_DIR / "system_stats.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(payload, f, indent=2)

    return jsonify({"ok": True})


@app.route("/api/days")
@login_required
def api_get_days():
    return jsonify(get_days())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("SSS_DEBUG", "0") == "1")
