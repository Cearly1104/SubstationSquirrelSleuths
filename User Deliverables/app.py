import io
import os
import json
import zipfile
from pathlib import Path
from datetime import datetime, date
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_from_directory, abort, Response
)

# ----------- CONFIG -----------
APP_SECRET = "CHANGE_ME"
DATA_ROOT = Path("/home/sss/data")     # <-- change
USERNAME = "viewer"
PASSWORD = "change-me"

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".webm"}  # browser-friendly
ALLOWED_LOG_EXT   = {".jsonl", ".json", ".csv", ".log", ".txt"}
# ------------------------------

app = Flask(__name__)
app.secret_key = APP_SECRET


# ---------- Helpers ----------
def require_login():
    return session.get("logged_in") is True

def safe_join(base: Path, *parts) -> Path:
    p = base.joinpath(*parts).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError("Unsafe path traversal")
    return p

def list_days():
    """Return sorted day folders like YYYY-MM-DD, newest first."""
    if not DATA_ROOT.exists():
        return []
    days = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
    # Keep only YYYY-MM-DD-ish folders
    valid = []
    for d in days:
        try:
            datetime.strptime(d, "%Y-%m-%d")
            valid.append(d)
        except ValueError:
            pass
    return sorted(valid, reverse=True)

def day_paths(day: str):
    base = safe_join(DATA_ROOT, day)
    return {
        "base": base,
        "images": safe_join(base, "images"),
        "videos": safe_join(base, "videos"),
        "logs":   safe_join(base, "logs"),
    }

def list_files(path: Path, allowed_ext: set[str]):
    if not path.exists():
        return []
    files = []
    for f in sorted(path.iterdir()):
        if f.is_file() and f.suffix.lower() in allowed_ext:
            files.append(f.name)
    return files

def count_events(day: str) -> int:
    """Event count = number of JSONL lines in logs/detections.jsonl."""
    p = safe_join(DATA_ROOT, day, "logs", "detections.jsonl")
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def read_logs(day: str, limit: int = 200):
    p = safe_join(DATA_ROOT, day, "logs", "detections.jsonl")
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"raw": line})
    return rows[-limit:]

def latest_day():
    days = list_days()
    return days[0] if days else None

def zip_stream_for_day(day: str, include=("images","videos","logs")):
    """
    Stream a zip in-memory (fine for moderate datasets).
    If your days are huge (GBs), switch to a temp file on disk.
    """
    paths = day_paths(day)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if "images" in include and paths["images"].exists():
            for fn in list_files(paths["images"], ALLOWED_IMAGE_EXT):
                full = safe_join(paths["images"], fn)
                z.write(full, arcname=f"{day}/images/{fn}")

        if "videos" in include and paths["videos"].exists():
            for fn in list_files(paths["videos"], ALLOWED_VIDEO_EXT):
                full = safe_join(paths["videos"], fn)
                z.write(full, arcname=f"{day}/videos/{fn}")

        if "logs" in include and paths["logs"].exists():
            for fn in list_files(paths["logs"], ALLOWED_LOG_EXT):
                full = safe_join(paths["logs"], fn)
                z.write(full, arcname=f"{day}/logs/{fn}")

    mem.seek(0)
    return mem
# ----------------------------


# ---------- Auth ----------
@app.get("/login")
def login_get():
    return render_template("login.html")

@app.post("/login")
def login_post():
    user = request.form.get("username", "")
    pw = request.form.get("password", "")
    if user == USERNAME and pw == PASSWORD:
        session["logged_in"] = True
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Invalid credentials")

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_get"))
# -------------------------


# ---------- Pages ----------
@app.get("/")
def root():
    return redirect(url_for("dashboard") if require_login() else url_for("login_get"))

@app.get("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("login_get"))

    day = latest_day()
    if not day:
        return render_template("dashboard.html", day=None)

    paths = day_paths(day)
    images = list_files(paths["images"], ALLOWED_IMAGE_EXT)
    videos = list_files(paths["videos"], ALLOWED_VIDEO_EXT)
    logs = read_logs(day, limit=120)

    summary = {
        "day": day,
        "event_count": count_events(day),
        "image_count": len(images),
        "video_count": len(videos),
        "log_count": max(count_events(day), len(logs)),  # rough
    }

    # Show only a few “quick view” items
    images_preview = images[-12:]  # latest 12
    videos_preview = videos[-6:]   # latest 6

    return render_template(
        "dashboard.html",
        day=day,
        summary=summary,
        images_preview=images_preview,
        videos_preview=videos_preview,
        logs=logs
    )

@app.get("/archive")
def archive():
    if not require_login():
        return redirect(url_for("login_get"))

    days = list_days()
    items = []
    for d in days:
        paths = day_paths(d)
        imgs = list_files(paths["images"], ALLOWED_IMAGE_EXT)
        vids = list_files(paths["videos"], ALLOWED_VIDEO_EXT)
        evts = count_events(d)
        items.append({
            "day": d,
            "event_count": evts,
            "image_count": len(imgs),
            "video_count": len(vids),
        })

    return render_template("archive.html", items=items)

@app.get("/day/<day>")
def day_view(day):
    if not require_login():
        return redirect(url_for("login_get"))

    # validate day exists
    if day not in list_days():
        abort(404)

    paths = day_paths(day)
    images = list_files(paths["images"], ALLOWED_IMAGE_EXT)
    videos = list_files(paths["videos"], ALLOWED_VIDEO_EXT)
    logs = read_logs(day, limit=500)

    return render_template(
        "day.html",
        day=day,
        event_count=count_events(day),
        images=images,
        videos=videos,
        logs=logs
    )
# ---------------------------


# ---------- Static serving ----------
@app.get("/data/<day>/<kind>/<path:filename>")
def serve_data(day, kind, filename):
    if not require_login():
        abort(403)
    if day not in list_days():
        abort(404)

    paths = day_paths(day)
    if kind == "images":
        base = paths["images"]
        allowed = ALLOWED_IMAGE_EXT
    elif kind == "videos":
        base = paths["videos"]
        allowed = ALLOWED_VIDEO_EXT
    elif kind == "logs":
        base = paths["logs"]
        allowed = ALLOWED_LOG_EXT
    else:
        abort(404)

    fp = safe_join(base, filename)
    if not fp.exists() or fp.suffix.lower() not in allowed:
        abort(404)

    return send_from_directory(base, filename, as_attachment=False)
# -----------------------------------


# ---------- Download endpoints ----------
@app.get("/download/day/<day>.zip")
def download_day_zip(day):
    if not require_login():
        abort(403)
    if day not in list_days():
        abort(404)

    mem = zip_stream_for_day(day, include=("images","videos","logs"))
    return Response(
        mem.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{day}_all.zip"'}
    )

@app.get("/download/day/<day>/<kind>.zip")
def download_day_kind_zip(day, kind):
    if not require_login():
        abort(403)
    if day not in list_days():
        abort(404)
    if kind not in ("images","videos","logs"):
        abort(404)

    mem = zip_stream_for_day(day, include=(kind,))
    return Response(
        mem.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{day}_{kind}.zip"'}
    )
# ----------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
