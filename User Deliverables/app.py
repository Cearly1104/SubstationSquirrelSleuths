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

import settings_io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Shared data directory where the analysis subsystem writes outputs.
# On the Jetson this will be an absolute path; default works for local dev.
DATA_DIR = Path(os.environ.get("SSS_DATA_DIR", BASE_DIR / "data"))

# Static demo assets (shipped with repo — training images, demo videos, etc.)
DEMO_ASSETS_DIR = BASE_DIR / "assets"

# Path to the Codebase pipeline's settings.json.  Default points at the
# sibling Codebase/ directory in the dev repo layout; on the Jetson this
# should be set explicitly via the SSS_SETTINGS_FILE env var.
SETTINGS_FILE = Path(
    os.environ.get("SSS_SETTINGS_FILE", BASE_DIR.parent / "Codebase" / "settings.json")
)

# Frozen "known good" defaults shipped alongside the pipeline.  Used by the
# Settings page's "Load Defaults" button so admins can recover from a bad edit.
SETTINGS_DEFAULTS_FILE = Path(
    os.environ.get(
        "SSS_SETTINGS_DEFAULTS",
        BASE_DIR.parent / "Codebase" / "settings.default.json",
    )
)

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

# Login attempt log — one line per attempt, appended in CSV format.
LOGIN_LOG_FILE = Path(os.environ.get("SSS_LOGIN_LOG", BASE_DIR / "login_attempts.log"))


def _load_accounts():
    if ACCOUNTS_FILE.is_file():
        with open(ACCOUNTS_FILE) as f:
            return json.load(f)
    return [{"username": "admin", "password": "sss", "role": "admin"}]


def _log_login_attempt(username, result):
    """Append a login attempt record to the login log.

    Each line is: ISO-timestamp,result,username,ip,user-agent
    Safe against filesystem errors so a full disk can't break auth.
    """
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "-")
        ua = (request.headers.get("User-Agent") or "-").replace(",", " ").replace("\n", " ")
        safe_user = (username or "-").replace(",", " ").replace("\n", " ")
        line = f"{ts},{result},{safe_user},{ip},{ua}\n"
        LOGIN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOGIN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


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


def admin_required(f):
    """Like login_required, but also requires the session to be an admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        if session.get("role") != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def _inject_role():
    """Make `is_admin` available to every template without passing it explicitly."""
    return {"is_admin": session.get("role") == "admin"}


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

    Each day folder contains event sub-folders named by timestamp
    (e.g. ``13:12:11``).  Inside each event folder:
      Images/   — captured stills
      Videos/   — recorded clips
    """
    if not DATA_DIR.is_dir():
        return []
    days = []
    for entry in sorted(DATA_DIR.iterdir(), reverse=True):
        if entry.is_dir() and _looks_like_date(entry.name):
            events = _collect_events(entry)
            all_images = []
            all_videos = []
            for ev in events:
                all_images.extend(
                    {"file": f, "event": ev["ts"]} for f in ev["images"]
                )
                all_videos.extend(
                    {"file": f, "event": ev["ts"]} for f in ev["videos"]
                )
            days.append({
                "day": entry.name,
                "events": events,
                "images": all_images,
                "videos": all_videos,
            })
    return days


def _collect_events(day_dir):
    """Scan *day_dir* for timestamp sub-folders and return event dicts."""
    events = []
    for sub in sorted(day_dir.iterdir()):
        if sub.is_dir() and not _looks_like_date(sub.name):
            images = _list_media(sub / "Images") + _list_media(sub / "images")
            videos = _list_media(sub / "Videos") + _list_media(sub / "videos")
            events.append({
                "ts": sub.name,
                "images": images,
                "videos": videos,
            })
    return events


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
    """Read live system stats directly from the Jetson hardware."""
    stats = {
        "uptime": "N/A",
        "storage_free": "N/A",
        "gpu_temp": "N/A",
        "status": "OFFLINE",
    }
    # Uptime
    try:
        with open("/proc/uptime") as f:
            secs = int(float(f.read().split()[0]))
            h, m = divmod(secs // 60, 60)
            stats["uptime"] = f"{h}h {m}m"
    except Exception:
        pass
    # GPU temperature (thermal_zone1 = gpu-thermal on Orin Nano)
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone1/temp") as f:
            millideg = int(f.read().strip())
            stats["gpu_temp"] = f"{millideg / 1000:.0f} °C"
    except Exception:
        pass
    # Storage free
    try:
        st = os.statvfs("/")
        free_gib = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        total_gib = (st.f_blocks * st.f_frsize) / (1024 ** 3)
        pct_free = (st.f_bavail / st.f_blocks) * 100
        stats["storage_free"] = f"{pct_free:.0f}% — {free_gib:.1f} GiB"
    except Exception:
        pass
    # Status
    stats["status"] = "ONLINE"
    return stats


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
            _log_login_attempt(match["username"], "success")
            session["user"] = match["username"]
            session["role"] = match.get("role", "user")
            next_page = request.args.get("next", url_for("dashboard"))
            return redirect(next_page)
        _log_login_attempt(username, "failure")
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("role", None)
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


@app.route("/settings", methods=["GET", "POST"])
@admin_required
def settings_page():
    """Admin-only editor for the Codebase pipeline's settings.json.

    GET  — render the current settings.json contents in a textarea.
    POST — parse, validate, and atomically save the submitted JSON.

    Note: changes do not affect the running pipeline until it is restarted.
    Mode-switching / live reload will be wired up in a later phase.
    """
    error = None
    success = None
    info = None
    text = ""

    if request.method == "POST":
        action = request.form.get("action", "save")

        if action == "load_defaults":
            # Populate the editor with the frozen defaults but DO NOT save —
            # the admin still has to click Save Settings to commit them.
            try:
                text = settings_io.load_text(SETTINGS_DEFAULTS_FILE)
                info = (
                    "Defaults loaded into the editor. Review them and click "
                    "Save Settings to apply, or Discard Changes to cancel."
                )
            except settings_io.SettingsError as e:
                error = f"Could not load defaults: {e}"
                # Fall back to whatever is currently on disk so the editor
                # isn't left blank.
                try:
                    text = settings_io.load_text(SETTINGS_FILE)
                except settings_io.SettingsError:
                    pass
        else:
            text = request.form.get("settings_text", "")
            try:
                parsed = settings_io.parse(text)
                settings_io.validate(parsed)
                settings_io.save_atomic(SETTINGS_FILE, parsed)
                # Re-read from disk so the editor reflects the canonical formatting.
                text = settings_io.load_text(SETTINGS_FILE)
                success = "Settings saved. Restart the pipeline for changes to take effect."
            except settings_io.SettingsError as e:
                error = str(e)
    else:
        try:
            text = settings_io.load_text(SETTINGS_FILE)
        except settings_io.SettingsError as e:
            error = str(e)

    return render_template(
        "settings.html",
        user=session["user"],
        settings_text=text,
        settings_path=str(SETTINGS_FILE),
        defaults_available=SETTINGS_DEFAULTS_FILE.is_file(),
        error=error,
        success=success,
        info=info,
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

@app.route("/data/<day_id>/<event_id>/images/<filename>")
@login_required
def serve_image(day_id, event_id, filename):
    for name in ("Images", "images"):
        folder = DATA_DIR / day_id / event_id / name
        if folder.is_dir():
            return send_from_directory(str(folder), filename)
    abort(404)


@app.route("/data/<day_id>/<event_id>/videos/<filename>")
@login_required
def serve_video(day_id, event_id, filename):
    for name in ("Videos", "videos"):
        folder = DATA_DIR / day_id / event_id / name
        if folder.is_dir():
            return send_from_directory(str(folder), filename)
    abort(404)


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
            return {cam["id"]: cam for cam in json.load(f)}
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
        _mjpeg_stream(cameras[cam_id]["url"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/feed/<cam_id>/hq")
@login_required
def live_feed_hq(cam_id):
    """MJPEG endpoint — high-quality stream (falls back to default URL)."""
    cameras = _load_cameras()
    if cam_id not in cameras:
        abort(404)
    cam = cameras[cam_id]
    stream_url = cam.get("url_hq", cam["url"])
    return Response(
        _mjpeg_stream(stream_url),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live/<cam_id>")
@login_required
def live_view(cam_id):
    """Full-page high-quality live view for a single camera."""
    cameras = _load_cameras()
    if cam_id not in cameras:
        abort(404)
    return render_template("live.html", cam_id=cam_id)


# ---------------------------------------------------------------------------
# Routes — API (used by the analysis subsystem to push data)
# ---------------------------------------------------------------------------

@app.route("/api/events", methods=["POST"])
def api_post_event():
    """Receive a detection event from the analysis subsystem.

    Expected JSON body: {day: "YYYY-MM-DD", ts: "HH:MM:SS"}
    Creates the event folder structure:  data/<day>/<ts>/Images/  and  Videos/
    An API key is required via the X-API-Key header.
    """
    api_key = os.environ.get("SSS_API_KEY", "")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True)
    if not payload or "day" not in payload or "ts" not in payload:
        return jsonify({"error": "bad request"}), 400

    day = payload["day"]
    ts = payload["ts"]
    event_dir = DATA_DIR / day / ts
    (event_dir / "Images").mkdir(parents=True, exist_ok=True)
    (event_dir / "Videos").mkdir(parents=True, exist_ok=True)

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


@app.route("/api/system-stats", methods=["GET"])
@login_required
def api_get_stats():
    return jsonify(get_system_stats())


@app.route("/api/days")
@login_required
def api_get_days():
    return jsonify(get_days())


# ---------------------------------------------------------------------------
# Captive portal detection — triggers automatic browser pop-up when a client
# connects to JetsonAP.  Each OS probes a different URL to check for internet;
# we intercept those probes and redirect to the dashboard.
#
# Requires dnsmasq on the Jetson to be configured with:
#   address=/#/10.42.0.1
# so that ALL DNS queries resolve to this device.
# ---------------------------------------------------------------------------

PORTAL_REDIRECT = "http://10.42.0.1:5000/"


@app.route("/generate_204")          # Android / Chrome
@app.route("/gen_204")               # Android (older)
def captive_android():
    return redirect(PORTAL_REDIRECT, 302)


@app.route("/hotspot-detect.html")   # macOS / iOS
@app.route("/library/test/success.html")
def captive_apple():
    # Apple expects either a redirect or a page that does NOT contain "Success"
    return redirect(PORTAL_REDIRECT, 302)


@app.route("/ncsi.txt")              # Windows — expects plain "Microsoft NCSI"
def captive_windows_ncsi():
    return redirect(PORTAL_REDIRECT, 302)


@app.route("/connecttest.txt")       # Windows 10+
@app.route("/redirect")
def captive_windows():
    return redirect(PORTAL_REDIRECT, 302)


@app.route("/connectivity-check")    # Ubuntu / GNOME
@app.route("/check_network_status.txt")  # Firefox
def captive_linux():
    return redirect(PORTAL_REDIRECT, 302)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("SSS_DEBUG", "0") == "1")
