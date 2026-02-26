from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\cpear\Documents\VisualAnalysis\runs\detect\train11\best.pt")
model.to("cuda")

# --- find squirrel class id ---
squirrel_id = None
for cid, name in model.names.items():
    if "squirrel" in name.lower():
        squirrel_id = cid
        break

print("Squirrel class id:", squirrel_id)

video_path = r"C:\Users\cpear\Downloads\Trail Cam Tuesdays  4 (Eastern Gray Squirrel) [9CVmV3aG29s].mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

cv2.namedWindow("YOLO Squirrel Tracking", cv2.WINDOW_NORMAL)

# Get video properties for writer
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback if metadata is weird

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video path
out_path = r"C:\Users\cpear\Documents\VisualAnalysis\squirrel_annotated.mp4"

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID" for .avi
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

track_history = defaultdict(list)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run tracking on this frame (no internal saving)
    result = model.track(
        frame,
        persist=True,
        classes=[squirrel_id] if squirrel_id is not None else None,
        conf=0.25,
        tracker="botsort.yaml",
    )[0]

    # Let Ultralytics draw boxes/IDs first
    frame = result.plot()

    if result.boxes is not None and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            if len(track) >= 2:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=(230, 0, 230),
                    thickness=10,
                )

    # Show live
    cv2.imshow("YOLO Squirrel Tracking", frame)

    # Write annotated frame to output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved annotated video to:", out_path)
