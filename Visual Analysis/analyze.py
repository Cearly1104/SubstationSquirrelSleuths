import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO, solutions

# ==========================
# Paths and configuration
# ==========================

# Root folder that contains videos and image SUB-folders
input_root = r"C:\Users\cpear\Documents\VisualAnalysis\input"

# Where to put outputs
output_root = r"C:\Users\cpear\Documents\VisualAnalysis\output"

os.makedirs(output_root, exist_ok=True)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Default FPS for image sequences (since they have no inherent FPS)
IMAGE_SEQ_FPS = 30


# ==========================
# Model setup (global)
# ==========================

model_path = r"C:\Users\cpear\Documents\VisualAnalysis\runs\detect\train11\best.pt"
model = YOLO(model_path)
model.to("cpu")

# Find squirrel class id
squirrel_id = None
for cid, name in model.names.items():
    if "squirrel" in name.lower():
        squirrel_id = cid
        break

print("Squirrel class id:", squirrel_id)


def create_heatmap_solution():
    """Create a fresh Heatmap solution instance."""
    return solutions.Heatmap(
        show=False,
        model=model_path,
        colormap=cv2.COLORMAP_INFERNO,
        device="cpu",
        classes=[squirrel_id] if squirrel_id is not None else None,
    )


# ==========================
# Frame-processing core
# ==========================

def process_stream(frame_iter, base_name, fps, frame_size):
    """
    Process an iterable of frames (video or image sequence).
    Writes two videos: tracked overlay and heatmap overlay.
    """
    width, height = frame_size

    # Output video paths
    out_track_path = os.path.join(output_root, f"{base_name}_tracked.mp4")
    out_heat_path = os.path.join(output_root, f"{base_name}_heatmap.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_track = cv2.VideoWriter(out_track_path, fourcc, fps, (width, height))
    out_heat = cv2.VideoWriter(out_heat_path, fourcc, fps, (width, height))

    # Windows (optional; comment out if you don't want GUI)
    cv2.namedWindow("YOLO Squirrel Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("YOLO Heatmap", cv2.WINDOW_NORMAL)

    # Tracking state
    track_history = defaultdict(list)

    # New heatmap solution for this stream
    heatmap = create_heatmap_solution()

    for frame in frame_iter:
        if frame is None:
            break

        # Ensure frame is expected size
        frame = cv2.resize(frame, (width, height))
        im0 = frame.copy()

        # --------------------------
        # 1) YOLO tracking
        # --------------------------
        result = model.track(
            frame,
            persist=True,
            classes=[squirrel_id] if squirrel_id is not None else None,
            conf=0.25,
            tracker="botsort.yaml",
        )[0]

        # Let Ultralytics draw boxes/IDs first
        track_frame = result.plot()

        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                x = float(x)
                y = float(y)

                track = track_history[track_id]
                track.append((x, y))

                if len(track) >= 2:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        track_frame,
                        [points],
                        isClosed=False,
                        color=(230, 0, 230),
                        thickness=10,
                    )

        # --------------------------
        # 2) Heatmap via solutions.Heatmap
        # --------------------------
        results_hm = heatmap(im0)
        heat_frame = results_hm.plot_im

        # Same fallback logic you had: if no current track_ids, but we have
        # a cumulative heatmap, blend it onto the frame.
        if not heatmap.track_ids:
            if hasattr(heatmap, "heatmap") and heatmap.heatmap is not None:
                heatmap_norm = cv2.normalize(
                    heatmap.heatmap, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

                heatmap_color = cv2.applyColorMap(
                    heatmap_norm, heatmap.colormap
                )

                heat_frame = cv2.addWeighted(im0, 0.5, heatmap_color, 0.5, 0)

        # --------------------------
        # Show (optional) and write
        # --------------------------
        cv2.imshow("YOLO Squirrel Tracking", track_frame)
        # Resize heat for display only
        display_heat = cv2.resize(heat_frame, (1280, 720))
        cv2.imshow("YOLO Heatmap", display_heat)

        out_track.write(track_frame)
        out_heat.write(heat_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out_track.release()
    out_heat.release()
    cv2.destroyAllWindows()

    print(f"Saved tracked video to: {out_track_path}")
    print(f"Saved heatmap video to: {out_heat_path}")


# ==========================
# Video and image sequence helpers
# ==========================

def process_video_file(video_path):
    """Process a single video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video: {video_path}")

    def frame_gen():
        while True:
            success, frame = cap.read()
            if not success:
                break
            yield frame

    process_stream(frame_gen(), base_name, fps, (width, height))
    cap.release()


def process_image_folder(folder_path):
    """Process a folder of images as a stitched frame sequence."""
    # Collect image files
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    if not files:
        return

    files.sort()  # ensure consistent order

    # Read first image to get size
    first = cv2.imread(files[0])
    if first is None:
        print(f"Could not read first image in folder: {folder_path}")
        return

    height, width = first.shape[:2]
    base_name = os.path.basename(folder_path.rstrip(r"\/"))
    print(f"Processing image sequence in folder: {folder_path}")

    def frame_gen():
        for p in files:
            frame = cv2.imread(p)
            if frame is None:
                continue
            yield frame

    process_stream(frame_gen(), base_name, IMAGE_SEQ_FPS, (width, height))


# ==========================
# Walk the input_root
# ==========================

def main():
    # wait until signal is given to start processing 
    if(1):
        # First, process all video files directly under input_root
        for entry in os.scandir(input_root):
            if entry.is_file() and entry.name.lower().endswith(VIDEO_EXTS):
                process_video_file(entry.path)

        # Then, treat each subfolder under input_root as a potential image sequence
        for entry in os.scandir(input_root):
            if entry.is_dir():
                process_image_folder(entry.path)


if __name__ == "__main__":
    main()
