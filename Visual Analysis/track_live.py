"""Live-camera version of track.py — runs YOLO tracking on an RTSP stream."""

from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# --- Settings ---------------------------------------------------------------
MODEL_PATH = r"C:\Users\cpear\Documents\VisualAnalysis\runs\detect\train11\best.pt"
RTSP_URL = "rtsp://admin:123456@10.0.0.11:554/profile1"  # matches Codebase/main.py cam0
CONF = 0.25
SAVE_OUTPUT = False   # set True to also record the annotated stream
OUT_PATH = r"C:\Users\cpear\Documents\VisualAnalysis\squirrel_live.mp4"

# Ask FFmpeg for TCP transport (more reliable than UDP on most home networks).
# Must be set BEFORE cv2.VideoCapture is constructed.
import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# --- Model ------------------------------------------------------------------
model = YOLO(MODEL_PATH)
model.to("cuda")

squirrel_id = None
for cid, name in model.names.items():
    if "squirrel" in name.lower():
        squirrel_id = cid
        break
print("Squirrel class id:", squirrel_id)

# --- Capture ----------------------------------------------------------------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
# Keep the internal buffer tiny so we always display the freshest frame.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Could not open stream: {RTSP_URL}")

# Probe dimensions from the first frame (RTSP metadata is often unreliable).
ok, first = cap.read()
if not ok:
    raise RuntimeError("Stream opened but first read failed")
height, width = first.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width, height))

cv2.namedWindow("YOLO Squirrel Tracking (live)", cv2.WINDOW_NORMAL)

track_history = defaultdict(list)
consecutive_failures = 0

frame = first
try:
    while True:
        result = model.track(
            frame,
            persist=True,
            classes=[squirrel_id] if squirrel_id is not None else None,
            conf=CONF,
            tracker="botsort.yaml",
            verbose=False,
        )[0]

        frame = result.plot()

        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, _w, _h = box
                trail = track_history[track_id]
                trail.append((float(x), float(y)))
                # Cap trail length so long sessions don't grow unbounded.
                if len(trail) > 120:
                    del trail[:-120]

                if len(trail) >= 2:
                    pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (230, 0, 230), 10)

        cv2.imshow("YOLO Squirrel Tracking (live)", frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Grab the next frame. RTSP streams hiccup — tolerate a few misses.
        ok, frame = cap.read()
        if not ok:
            consecutive_failures += 1
            print(f"[live] read failed ({consecutive_failures})")
            if consecutive_failures >= 30:
                print("[live] too many failures, giving up")
                break
            continue
        consecutive_failures = 0
finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
