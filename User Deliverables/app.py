"""
Substation Squirrel Sleuths — Flask Application
Serves the user deliverables dashboard over a local WiFi access point
hosted on the Jetson Orin Nano.
"""

import os
import json
import secrets
import subprocess
import threading
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

# Shared data directory — points at the pipeline's detections/ output folder.
# Structure: detections/<date>/<timestamp>/<cam_name>/<files>
DATA_DIR = Path(os.environ.get("SSS_DATA_DIR", BASE_DIR.parent / "Codebase" / "detections"))

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

    Pipeline output structure:
      detections/<date>/<timestamp>/<cam_name>/<files>

    Each episode is one camera subfolder inside a timestamp folder.
    Only the three analysis outputs are exposed to the website:
      *_tracked.mp4, *_heatmap_final.png, *_birdseye_final.png
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
                    {"file": f, "event_ts": ev["ts"], "event_cam": ev["cam"]}
                    for f in ev["images"]
                )
                all_videos.extend(
                    {"file": f, "event_ts": ev["ts"], "event_cam": ev["cam"]}
                    for f in ev["videos"]
                )
            days.append({
                "day": entry.name,
                "events": events,
                "images": all_images,
                "videos": all_videos,
            })
    return days


def _collect_events(day_dir):
    """Scan *day_dir* for timestamp/cam sub-folders and return event dicts.

    Structure: <day_dir>/<timestamp>/<cam_name>/<analysis files>
    """
    events = []
    for ts_dir in sorted(day_dir.iterdir()):
        if not ts_dir.is_dir() or _looks_like_date(ts_dir.name):
            continue
        for cam_dir in sorted(ts_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            images, videos = _list_analysis_outputs(cam_dir)
            if images or videos:
                events.append({
                    "ts": ts_dir.name,
                    "cam": cam_dir.name,
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


def _list_analysis_outputs(cam_dir):
    """Return only the three analysis outputs from a camera episode folder.

    Recognized outputs:
      *_tracked.mp4         -> video
      *_heatmap_final.png   -> image
      *_birdseye_final.png  -> image
    """
    images = []
    videos = []
    for f in sorted(cam_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if name.endswith("_tracked.mp4"):
            videos.append(name)
        elif name.endswith("_heatmap_final.png"):
            images.append(name)
        elif name.endswith("_birdseye_final.png"):
            images.append(name)
    return images, videos


def get_enabled_cameras():
    """Return the list of enabled camera entries from settings.json.

    Reads fresh on every call so dashboard reflects changes without restart.
    Each entry is a dict with at least: id (int), name (str), enabled (bool).
    """
    try:
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
        return [cam for cam in settings.get("cameras", []) if cam.get("enabled")]
    except Exception:
        return []


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
    enabled_cameras = get_enabled_cameras()
    return render_template(
        "dashboard.html",
        user=session["user"],
        today=today,
        today_data=today_data,
        stats=stats,
        days=days,
        enabled_cameras=enabled_cameras,
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

        if action == "set_mode":
            new_mode = (request.form.get("mode") or "").strip().upper()
            try:
                parsed = settings_io.parse(settings_io.load_text(SETTINGS_FILE))
                if new_mode not in parsed.get("modes", {}):
                    error = f"Unknown mode '{new_mode}'. Valid modes: {', '.join(parsed.get('modes', {}).keys())}"
                else:
                    parsed["default_mode"] = new_mode
                    settings_io.save_atomic(SETTINGS_FILE, parsed)
                    success = f"Mode set to {new_mode}. Restart the pipeline for the change to take effect."
                text = settings_io.load_text(SETTINGS_FILE)
            except settings_io.SettingsError as e:
                error = str(e)
                try:
                    text = settings_io.load_text(SETTINGS_FILE)
                except settings_io.SettingsError:
                    pass

        elif action == "load_defaults":
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

    # Derive current mode and available modes for the mode-selector UI
    current_mode = None
    available_modes = []
    try:
        parsed_ui = settings_io.parse(text) if text else {}
        current_mode = parsed_ui.get("default_mode", "")
        available_modes = list(parsed_ui.get("modes", {}).keys())
    except Exception:
        pass

    return render_template(
        "settings.html",
        user=session["user"],
        settings_text=text,
        settings_path=str(SETTINGS_FILE),
        defaults_available=SETTINGS_DEFAULTS_FILE.is_file(),
        error=error,
        success=success,
        info=info,
        current_mode=current_mode,
        available_modes=available_modes,
    )


@app.route("/log")
@login_required
def log_page():
    """Full-page system log viewer."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_dates = []
    if DATA_DIR.is_dir():
        for entry in sorted(DATA_DIR.iterdir(), reverse=True):
            if entry.is_dir() and _looks_like_date(entry.name):
                if (entry / f"{entry.name}_log.txt").is_file():
                    log_dates.append(entry.name)
    lifetime_exists = (DATA_DIR / "lifetime_log.txt").is_file()
    return render_template(
        "log.html",
        user=session["user"],
        today=today,
        log_dates=log_dates,
        lifetime_exists=lifetime_exists,
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

@app.route("/data/<day_id>/<ts>/<cam>/images/<filename>")
@login_required
def serve_image(day_id, ts, cam, filename):
    folder = DATA_DIR / day_id / ts / cam
    if folder.is_dir():
        return send_from_directory(str(folder), filename)
    abort(404)


@app.route("/data/<day_id>/<ts>/<cam>/videos/<filename>")
@login_required
def serve_video(day_id, ts, cam, filename):
    folder = DATA_DIR / day_id / ts / cam
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

# ---------------------------------------------------------------------------
# Shared frame store — one decode thread per camera URL, all MJPEG clients
# read from the cached latest frame.  Avoids spawning a new cv2.VideoCapture
# per browser connection which is very heavy on the Jetson.
# ---------------------------------------------------------------------------

class _FrameStore:
    def __init__(self):
        self.frame: bytes | None = None
        self.lock = threading.Lock()


_frame_stores: dict[str, _FrameStore] = {}
_frame_stores_lock = threading.Lock()


_PREVIEW_FPS     = 5          # decode rate for the dashboard preview
_PREVIEW_WIDTH   = 854        # downscale width (854×480 for 16:9 source)
_PREVIEW_HEIGHT  = 480
_PREVIEW_QUALITY = 60         # JPEG quality — fine for monitoring


def _capture_loop(url: str, store: _FrameStore) -> None:
    """Background thread: drain the RTSP buffer cheaply and decode at low fps.

    grab() pulls a compressed frame off the network without decoding — very
    cheap.  retrieve() is only called every (1 / _PREVIEW_FPS) seconds so
    the expensive H.264 decode + JPEG encode only happens 5× per second
    regardless of the source stream's fps.
    """
    import cv2
    interval = 1.0 / _PREVIEW_FPS
    cap = cv2.VideoCapture(url)
    last_decode = 0.0

    while True:
        ok = cap.grab()           # drain buffer, no decode
        if not ok:
            cap.release()
            time.sleep(2.0)
            cap = cv2.VideoCapture(url)
            last_decode = 0.0
            continue

        now = time.monotonic()
        if now - last_decode >= interval:
            ok, frame = cap.retrieve()    # decode only this one frame
            if ok:
                frame = cv2.resize(frame, (_PREVIEW_WIDTH, _PREVIEW_HEIGHT))
                _, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_QUALITY]
                )
                with store.lock:
                    store.frame = buf.tobytes()
                last_decode = now


def _get_frame_store(url: str) -> _FrameStore:
    """Return (and lazily start) the shared frame store for *url*."""
    with _frame_stores_lock:
        if url not in _frame_stores:
            store = _FrameStore()
            _frame_stores[url] = store
            t = threading.Thread(target=_capture_loop, args=(url, store), daemon=True)
            t.start()
        return _frame_stores[url]


def _load_cameras():
    """Build a camera dict keyed by camera name.

    settings.json is the authoritative source (all cameras defined there are
    included).  cameras.json can overlay additional fields (e.g. url_hq) for
    any subset of cameras.  Auth credentials are injected into the URL from
    environment variables if specified in settings.json.
    """
    cameras = {}

    # Base: read every camera from settings.json
    try:
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
        for cam in settings.get("cameras", []):
            name = cam.get("name")
            if not name:
                continue
            url = cam.get("url", "")
            # Inject RTSP auth credentials from env vars if provided
            auth = cam.get("auth") or {}
            un_env = auth.get("username_env")
            pw_env = auth.get("password_env")
            if un_env and pw_env:
                un = os.environ.get(un_env, "")
                pw = os.environ.get(pw_env, "")
                if un and pw and "://" in url:
                    scheme, rest = url.split("://", 1)
                    url = f"{scheme}://{un}:{pw}@{rest}"
            cameras[name] = {"id": name, "url": url}
    except Exception:
        pass

    # Overlay: cameras.json can add/override fields (e.g. url_hq)
    if CAMERAS_FILE.is_file():
        try:
            with open(CAMERAS_FILE) as f:
                for entry in json.load(f):
                    cam_id = entry.get("id")
                    if cam_id:
                        cameras[cam_id] = {**cameras.get(cam_id, {}), **entry}
        except Exception:
            pass

    return cameras


def _mjpeg_stream(stream_url):
    """Generator that serves MJPEG from the shared per-URL frame store.

    Only one cv2.VideoCapture runs per unique URL regardless of how many
    clients are connected, which keeps CPU use manageable on the Jetson.
    Frames are delivered to the browser at ~15 fps.
    """
    store = _get_frame_store(stream_url)
    while True:
        with store.lock:
            frame = store.frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(1 / 15)


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

    Expected JSON body: {day: "YYYY-MM-DD", ts: "...", cam: "cam_name"}
    Creates the event folder: detections/<day>/<ts>/<cam>/
    An API key is required via the X-API-Key header.
    """
    api_key = os.environ.get("SSS_API_KEY", "")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True)
    if not payload or "day" not in payload or "ts" not in payload or "cam" not in payload:
        return jsonify({"error": "bad request"}), 400

    event_dir = DATA_DIR / payload["day"] / payload["ts"] / payload["cam"]
    event_dir.mkdir(parents=True, exist_ok=True)

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


@app.route("/api/log")
@login_required
def api_get_log():
    """Return log file contents as a JSON array of lines.

    Query params:
      date=YYYY-MM-DD  — return that day's daily log (default: today)
      lifetime=1       — return the lifetime log instead
      tail=N           — only return the last N lines
    """
    tail = request.args.get("tail", type=int)

    if request.args.get("lifetime"):
        log_path = DATA_DIR / "lifetime_log.txt"
    else:
        date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        if not _looks_like_date(date_str):
            return jsonify({"error": "invalid date"}), 400
        log_path = DATA_DIR / date_str / f"{date_str}_log.txt"

    if not log_path.is_file():
        return jsonify({"lines": [], "exists": False})

    try:
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
        if tail:
            lines = lines[-tail:]
        return jsonify({"lines": [l.rstrip("\n") for l in lines], "exists": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/restart-pipeline", methods=["POST"])
@login_required
@admin_required
def api_restart_pipeline():
    try:
        subprocess.run(
            ["sudo", "/usr/bin/systemctl", "restart", "sss-pipeline"],
            check=True, timeout=15
        )
        return jsonify({"ok": True})
    except subprocess.CalledProcessError as e:
        return jsonify({"ok": False, "error": f"systemctl exited {e.returncode}"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/pipeline-status", methods=["GET"])
@login_required
def api_pipeline_status():
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "sss-pipeline"],
            capture_output=True, text=True, timeout=5
        )
        return jsonify({"status": result.stdout.strip()})
    except Exception as e:
        return jsonify({"status": "unknown", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# File browser — admin-only view of the filesystem rooted at FILES_ROOT
# ---------------------------------------------------------------------------

# Root directory exposed by the file browser.  Defaults to the SSS project
# root (one level above User Deliverables/).  Override via SSS_FILES_ROOT.
FILES_ROOT = Path(
    os.environ.get("SSS_FILES_ROOT", BASE_DIR.parent)
).resolve()

# File extensions treated as viewable images (shown as inline preview)
_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
# File extensions streamed as video in the browser
_VIDEO_EXTS  = {".mp4", ".webm", ".mov"}
# Plain-text extensions opened in the log viewer or as raw text
_TEXT_EXTS   = {".txt", ".log", ".json", ".csv", ".md", ".py"}


def _resolve_browse_path(subpath: str) -> Path | None:
    """Resolve *subpath* relative to FILES_ROOT and verify it stays inside.

    Returns the resolved Path, or None if the path escapes the root.
    """
    try:
        target = (FILES_ROOT / subpath).resolve()
        target.relative_to(FILES_ROOT)   # raises ValueError if outside
        return target
    except (ValueError, Exception):
        return None


def _file_size_str(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _file_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:  return "image"
    if ext in _VIDEO_EXTS:  return "video"
    if ext in _TEXT_EXTS:   return "text"
    return "file"


@app.route("/browse", defaults={"subpath": ""})
@app.route("/browse/<path:subpath>")
@admin_required
def file_browser(subpath):
    """Admin file browser rooted at FILES_ROOT."""
    target = _resolve_browse_path(subpath)
    if target is None or not target.exists():
        abort(404)

    # If it's a file, serve it for download / inline viewing
    if target.is_file():
        kind = _file_kind(target)
        mime_map = {
            "image": None,          # send_from_directory will sniff it
            "video": "video/mp4",
            "text":  "text/plain",
            "file":  "application/octet-stream",
        }
        as_attachment = kind == "file"
        return send_from_directory(
            str(target.parent),
            target.name,
            mimetype=mime_map[kind],
            as_attachment=as_attachment,
        )

    # It's a directory — build listing
    entries_dirs  = []
    entries_files = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        try:
            stat = child.stat()
        except OSError:
            continue
        rel = child.relative_to(FILES_ROOT)
        info = {
            "name":     child.name,
            "path":     str(rel),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        if child.is_dir():
            entries_dirs.append(info)
        else:
            info["size"] = _file_size_str(stat.st_size)
            info["kind"] = _file_kind(child)
            info["ext"]  = child.suffix.lower().lstrip(".")
            entries_files.append(info)

    # Build breadcrumb from subpath parts
    crumbs = []
    parts  = Path(subpath).parts if subpath else []
    for i, part in enumerate(parts):
        crumbs.append({
            "name": part,
            "path": str(Path(*parts[: i + 1])),
        })

    parent_path = str(Path(subpath).parent) if subpath else None
    if parent_path == ".":
        parent_path = ""

    return render_template(
        "files.html",
        user=session["user"],
        current_path=subpath,
        crumbs=crumbs,
        dirs=entries_dirs,
        files=entries_files,
        parent_path=parent_path,
        root_name=FILES_ROOT.name,
    )


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
