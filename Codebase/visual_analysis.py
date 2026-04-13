import os
import gc
import subprocess
import torch
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import pipeline_status


@dataclass
class AnalysisResult:
    cam_name: str
    episode_timestamp: datetime
    stage: str        # "TRACK", "HEATMAP", "HOMOGRAPHY"
    completed_at: datetime
    output_file: str  # filename only


# ========================== Homography Defaults ==========================

HEAT_RADIUS = 16
HEAT_BLUR = 31
HEAT_INCREMENT = 0.08
HEAT_MAX = 6.0
HEAT_SPEED_REF = 4.0
HEAT_SPEED_POWER = 1.8
HEAT_MIN_MOTION_WEIGHT = 0.08
HEAT_VIS_GAMMA = 0.85
MAX_TRAIL = 80
COLORMAP = cv2.COLORMAP_INFERNO


# ========================== Helpers ==========================

def _next_filename(base, ext):
    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _get_class_id(model_names, needle):
    if needle is None:
        return None
    needle = needle.lower()
    items = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    for cid, name in items:
        if needle in str(name).lower():
            return int(cid)
    return None


def _release_model(model):
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _transcode_h264(path: str) -> str:
    """Re-encode an mp4v file to H.264 in-place for browser compatibility.

    Tries h264_nvenc first (Jetson hardware encoder), falls back to libx264.
    Returns the original path unchanged if transcoding fails.
    """
    tmp = path + ".tmp.mp4"
    for encoder in ("h264_nvenc", "libx264"):
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-c:v", encoder, "-preset", "fast", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            os.replace(tmp, path)
            return path
    # Clean up temp file if both encoders failed
    try:
        os.remove(tmp)
    except OSError:
        pass
    print(f"[analysis] Warning: H.264 transcode failed for {path}, file left as mp4v")
    return path


def _order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ========================== Analysis Worker ==========================

def analysis_worker(config, analysis_queue, stop, status_dir=None, logging_queue=None):
    """Watches for completed episode videos and runs sequential analysis on each."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[analysis] Worker running on {device}")

    while not stop.is_set():
        try:
            item = analysis_queue.get(timeout=0.5)
        except Empty:
            continue

        video_path = item["clip_path"]
        cam_name = item["cam_name"]
        timestamp = item["timestamp"]
        group_dir = item["group_dir"]

        print(f"[analysis] Received episode: {video_path} (cam: {cam_name})")

        episode_ts_str = timestamp.strftime("%Y-%m-%d_%I.%M.%S%p")
        date_str = timestamp.strftime("%Y-%m-%d")
        output_dir = Path("detections") / date_str / group_dir / episode_ts_str
        output_dir.mkdir(parents=True, exist_ok=True)

        if status_dir:
            pipeline_status.set_analyzing(status_dir, True)
        try:
            run_analysis(video_path, output_dir, cam_name, timestamp, config, device, stop, logging_queue)
        except Exception as e:
            import traceback
            print(f"[analysis] Error processing {video_path}: {e}")
            traceback.print_exc()
        finally:
            if status_dir:
                pipeline_status.set_analyzing(status_dir, False)


# ========================== Stage 1: Tracking ==========================
def stage_track(video_path, output_dir, base, config, device, stop, class_filter="squirrel"):
    from ultralytics import YOLO

    print("[analysis] Stage 1/3: Tracking")

    model = YOLO(str(config["model_path"])).to(device)
    class_id = _get_class_id(model.names, class_filter)
    conf = config["detection_confidence"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(Path(output_dir) / f"{base}_tracked.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_history = defaultdict(list)

    while cap.isOpened() and not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            classes=[class_id] if class_id is not None else None,
            conf=conf,
            tracker="botsort.yaml",
            verbose=False,
            half=True,
        )[0]

        frame = result.plot()

        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            ids = result.boxes.id.int().cpu().tolist()
            for box, tid in zip(boxes, ids):
                x, y = float(box[0]), float(box[1])
                track_history[tid].append((x, y))
                if len(track_history[tid]) >= 2:
                    pts = np.array(track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (230, 0, 230), 10)

        writer.write(frame)

    cap.release()
    writer.release()
    _release_model(model)
    _transcode_h264(out_path)

    print(f"[analysis]   Saved: {out_path}")
    return out_path


# ========================== Stage 2: Heatmap ==========================
def stage_heatmap(video_path, output_dir, base, config, device, stop, class_filter="squirrel"):
    from ultralytics import YOLO, solutions

    print("[analysis] Stage 2/3: Heatmap")

    # Resolve class filter for solutions.Heatmap
    temp_model = YOLO(str(config["model_path"])).to(device)
    class_id = _get_class_id(temp_model.names, class_filter)
    del temp_model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_path = str(Path(output_dir) / f"{base}_heatmap.mp4")
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    heatmap = solutions.Heatmap(
        show=False,
        model=str(config["model_path"]),
        colormap=cv2.COLORMAP_INFERNO,
        device=device,
        conf=config["detection_confidence"],
        half=True,
        classes=[class_id] if class_id is not None else None,
    )

    last_frame = None

    while cap.isOpened() and not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame

        results = heatmap(frame)
        out_frame = results.plot_im

        if not heatmap.track_ids and hasattr(heatmap, "heatmap") and heatmap.heatmap is not None:
            hm_norm = cv2.normalize(heatmap.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_norm, heatmap.colormap)
            out_frame = cv2.addWeighted(frame, 0.5, hm_color, 0.5, 0)

        writer.write(out_frame)

    cap.release()
    writer.release()
    _transcode_h264(vid_path)

    img_path = None
    if last_frame is not None and hasattr(heatmap, "heatmap") and heatmap.heatmap is not None:
        hm_norm = cv2.normalize(heatmap.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, heatmap.colormap)
        final = cv2.addWeighted(last_frame, 0.5, hm_color, 0.5, 0)
        img_path = str(Path(output_dir) / f"{base}_heatmap_final.png")
        cv2.imwrite(img_path, final)

    del heatmap
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    print(f"[analysis]   Saved: {vid_path}")
    if img_path:
        print(f"[analysis]   Saved: {img_path}")
    return vid_path, img_path


# ========================== Stage 3: Homography ==========================
def stage_homography(video_path, output_dir, base, config, device, stop, class_filter="squirrel"):
    from ultralytics import YOLO

    print("[analysis] Stage 3/3: Homography bird's-eye heatmap")

    source_points = config["source_points"]
    PLANE_WIDTH_M = config["plane_width_m"]
    PLANE_HEIGHT_M = config["plane_height_m"]
    PIXELS_PER_METER = config["pixels_per_meter"]
    conf = config["detection_confidence"]

    if source_points is None:
        print("[analysis]   Skipped — no source_points provided.")
        return None

    src_pts = _order_points(np.asarray(source_points, dtype=np.float32))

    bev_w = int(PLANE_WIDTH_M * PIXELS_PER_METER)
    bev_h = int(PLANE_HEIGHT_M * PIXELS_PER_METER)
    dst_pts = np.array(
        [[0, 0], [bev_w - 1, 0], [bev_w - 1, bev_h - 1], [0, bev_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    model = YOLO(str(config["model_path"])).to(device)
    class_id = _get_class_id(model.names, class_filter)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = str(Path(output_dir) / f"{base}_birdseye.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (bev_w, bev_h))

    heatmap_acc = np.zeros((bev_h, bev_w), dtype=np.float32)
    bev_tracks = defaultdict(list)
    last_bev_points = {}
    camera_ref = np.array([bev_w // 2, bev_h - 1], dtype=np.float32)
    bev_overlay = None

    def make_bev_canvas():
        canvas = np.full((bev_h, bev_w, 3), 255, dtype=np.uint8)
        step = PIXELS_PER_METER
        for x in range(0, bev_w, step):
            cv2.line(canvas, (x, 0), (x, bev_h), (210, 210, 210), 1)
        for y in range(0, bev_h, step):
            cv2.line(canvas, (0, y), (bev_w, y), (210, 210, 210), 1)
        cv2.rectangle(canvas, (0, 0), (bev_w - 1, bev_h - 1), (80, 80, 80), 2)
        return canvas

    while cap.isOpened() and not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame, persist=True, conf=conf,
            tracker="bytetrack.yaml",
            classes=[class_id] if class_id is not None else None,
            verbose=False,
            half=True,
        )[0]

        bev_base = make_bev_canvas()

        if result.boxes is not None and result.boxes.is_track:
            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                foot = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(foot, H)[0, 0]
                bx, by = int(mapped[0]), int(mapped[1])

                if 0 <= bx < bev_w and 0 <= by < bev_h:
                    prev = last_bev_points.get(tid)
                    speed_px = 0.0 if prev is None else float(np.hypot(bx - prev[0], by - prev[1]))

                    motion_w = max(HEAT_MIN_MOTION_WEIGHT,
                                   1.0 / (1.0 + (speed_px / HEAT_SPEED_REF) ** HEAT_SPEED_POWER))
                    stamp = np.zeros_like(heatmap_acc)
                    cv2.circle(stamp, (bx, by), HEAT_RADIUS, HEAT_INCREMENT * motion_w, -1)
                    heatmap_acc += stamp
                    last_bev_points[tid] = (bx, by)

                    bev_tracks[tid].append((bx, by))
                    bev_tracks[tid] = bev_tracks[tid][-MAX_TRAIL:]

                    if len(bev_tracks[tid]) > 1:
                        pts = np.array(bev_tracks[tid], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(bev_base, [pts], False, (255, 255, 0), 2)

        blur_k = HEAT_BLUR if HEAT_BLUR % 2 == 1 else HEAT_BLUR + 1
        heat_blur = cv2.GaussianBlur(heatmap_acc, (blur_k, blur_k), 0)
        heat_scaled = np.clip(heat_blur / max(HEAT_MAX, 1e-4), 0.0, 1.0)
        heat_shaped = np.power(heat_scaled, HEAT_VIS_GAMMA)
        heat_norm = (heat_shaped * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, COLORMAP)

        bev_overlay = cv2.addWeighted(bev_base, 0.35, heat_color, 0.65, 0)
        cv2.circle(bev_overlay, tuple(camera_ref.astype(int)), 7, (0, 255, 0), -1)

        writer.write(bev_overlay)

    cap.release()
    writer.release()
    _release_model(model)
    _transcode_h264(out_path)

    img_path = None
    if bev_overlay is not None:
        img_path = str(Path(output_dir) / f"{base}_birdseye_final.png")
        cv2.imwrite(img_path, bev_overlay)

    print(f"[analysis]   Saved: {out_path}")
    if img_path:
        print(f"[analysis]   Saved: {img_path}")
    return out_path, img_path


# ========================== Pipeline Entry Point ==========================
def run_analysis(video_path, output_dir, cam_name, timestamp, config, device, stop, logging_queue=None):
    video_path = str(video_path)
    base = f"{cam_name}_{timestamp.strftime('%Y-%m-%d_%I.%M.%S%p')}"

    print(f"[analysis] Starting sequential analysis on: {video_path}")

    track_out = stage_track(video_path, output_dir, base, config, device, stop)
    if logging_queue and track_out:
        logging_queue.put(AnalysisResult(
            cam_name=cam_name, episode_timestamp=timestamp, stage="TRACK",
            completed_at=datetime.now(), output_file=Path(track_out).name
        ))

    heatmap_out = stage_heatmap(video_path, output_dir, base, config, device, stop)
    if logging_queue and heatmap_out:
        _, img_path = heatmap_out
        if img_path:
            logging_queue.put(AnalysisResult(
                cam_name=cam_name, episode_timestamp=timestamp, stage="HEATMAP",
                completed_at=datetime.now(), output_file=Path(img_path).name
            ))

    homography_out = stage_homography(video_path, output_dir, base, config, device, stop)
    if logging_queue and homography_out:
        _, img_path = homography_out
        if img_path:
            logging_queue.put(AnalysisResult(
                cam_name=cam_name, episode_timestamp=timestamp, stage="HOMOGRAPHY",
                completed_at=datetime.now(), output_file=Path(img_path).name
            ))

    print("[analysis] All stages complete.")
    return {
        "track": track_out,
        "heatmap": heatmap_out,
        "homography": homography_out,
    }
