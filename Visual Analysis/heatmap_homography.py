import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = r"C:\Users\cpear\Documents\VisualAnalysis\squirrelmapping.mp4"
MODEL_PATH = r"C:\Users\cpear\Documents\VisualAnalysis\runs\detect\train10\weights\best.pt"
DEVICE = "cuda"
CONF = 0.20
CLASS_NAME_CONTAINS = "squirrel"  # Set to None to keep all classes
TRACKER_CFG = "bytetrack.yaml"

# Bird's-eye plane configuration
PLANE_WIDTH_M = 8.0
PLANE_HEIGHT_M = 6.0
PIXELS_PER_METER = 100

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
USE_WARPED_GROUND_BACKGROUND = False
USE_LIVE_WARP_BACKGROUND = False


def next_available_filename(base, ext):
    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def order_points(pts):
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def select_ground_points(frame):
    """Interactive selection of 4 points on the image ground plane."""
    window = "Select 4 Ground Points"
    points = []
    preview = frame.copy()

    def redraw():
        nonlocal preview
        preview = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(preview, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(
                preview,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], True, (0, 200, 255), 2)

        cv2.putText(
            preview,
            "Click 4 ground corners (clockwise). c=confirm, r=reset, q=quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            redraw()
        elif key == ord("c") and len(points) == 4:
            break
        elif key == ord("q"):
            cv2.destroyWindow(window)
            raise RuntimeError("Calibration cancelled by user.")

    cv2.destroyWindow(window)
    return order_points(np.array(points, dtype=np.float32))


def get_class_id(model_names, needle):
    if needle is None:
        return None

    needle = needle.lower()
    if isinstance(model_names, dict):
        items = model_names.items()
    else:
        items = enumerate(model_names)

    for class_id, class_name in items:
        if needle in str(class_name).lower():
            return int(class_id)
    return None


def make_bev_canvas(bev_w, bev_h):
    canvas = np.full((bev_h, bev_w, 3), 255, dtype=np.uint8)
    # Draw metric grid every meter
    step = PIXELS_PER_METER
    for x in range(0, bev_w, step):
        cv2.line(canvas, (x, 0), (x, bev_h), (210, 210, 210), 1)
    for y in range(0, bev_h, step):
        cv2.line(canvas, (0, y), (bev_w, y), (210, 210, 210), 1)

    cv2.rectangle(canvas, (0, 0), (bev_w - 1, bev_h - 1), (80, 80, 80), 2)
    return canvas


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Video has no readable frames.")

    src_pts = select_ground_points(first_frame)

    bev_w = int(PLANE_WIDTH_M * PIXELS_PER_METER)
    bev_h = int(PLANE_HEIGHT_M * PIXELS_PER_METER)
    dst_pts = np.array(
        [[0, 0], [bev_w - 1, 0], [bev_w - 1, bev_h - 1], [0, bev_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Rewind after calibration frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    class_id = get_class_id(model.names, CLASS_NAME_CONTAINS)
    if CLASS_NAME_CONTAINS and class_id is None:
        print(f"Warning: class containing '{CLASS_NAME_CONTAINS}' not found. Using all classes.")

    heatmap_acc = np.zeros((bev_h, bev_w), dtype=np.float32)
    cam_tracks = defaultdict(list)
    bev_tracks = defaultdict(list)
    last_bev_points = {}

    camera_reference = np.array([bev_w // 2, bev_h - 1], dtype=np.float32)

    # Prepare writer once display sizes are known
    cam_display_h = bev_h
    cam_display_w = int(frame_w * (cam_display_h / frame_h))
    out_w = cam_display_w + bev_w
    out_h = bev_h

    output_path = next_available_filename("birdseye_heatmap", ".mp4")
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    cv2.namedWindow("Bird's-Eye Demo", cv2.WINDOW_NORMAL)

    static_bev_background = None
    if USE_WARPED_GROUND_BACKGROUND:
        static_bev_background = cv2.warpPerspective(first_frame, H, (bev_w, bev_h))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            conf=CONF,
            tracker=TRACKER_CFG,
            classes=[class_id] if class_id is not None else None,
            verbose=False,
        )[0]

        annotated = result.plot()

        bev_base = make_bev_canvas(bev_w, bev_h)
        if USE_WARPED_GROUND_BACKGROUND:
            if USE_LIVE_WARP_BACKGROUND:
                warped = cv2.warpPerspective(frame, H, (bev_w, bev_h))
            else:
                warped = static_bev_background
            bev_base = cv2.addWeighted(warped, 0.80, bev_base, 0.20, 0)

        if result.boxes is not None and result.boxes.is_track:
            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                foot = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(foot, H)[0, 0]
                bx, by = int(mapped[0]), int(mapped[1])

                # Draw foot point on camera view
                fx, fy = int(foot[0, 0, 0]), int(foot[0, 0, 1])
                cv2.circle(annotated, (fx, fy), 4, (0, 255, 255), -1)

                if 0 <= bx < bev_w and 0 <= by < bev_h:
                    prev_pt = last_bev_points.get(track_id)
                    if prev_pt is None:
                        speed_px = 0.0
                    else:
                        speed_px = float(np.hypot(bx - prev_pt[0], by - prev_pt[1]))

                    # Penalize fast movement so short run-through paths don't saturate.
                    motion_weight = 1.0 / (1.0 + (speed_px / HEAT_SPEED_REF) ** HEAT_SPEED_POWER)
                    motion_weight = max(HEAT_MIN_MOTION_WEIGHT, motion_weight)
                    increment = HEAT_INCREMENT * motion_weight
                    stamp = np.zeros_like(heatmap_acc, dtype=np.float32)
                    cv2.circle(stamp, (bx, by), HEAT_RADIUS, increment, -1)
                    heatmap_acc += stamp
                    last_bev_points[track_id] = (bx, by)

                    cam_tracks[track_id].append((fx, fy))
                    bev_tracks[track_id].append((bx, by))
                    cam_tracks[track_id] = cam_tracks[track_id][-MAX_TRAIL:]
                    bev_tracks[track_id] = bev_tracks[track_id][-MAX_TRAIL:]

                    if len(cam_tracks[track_id]) > 1:
                        cam_pts = np.array(cam_tracks[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated, [cam_pts], False, (255, 255, 0), 2)
                    if len(bev_tracks[track_id]) > 1:
                        bev_pts = np.array(bev_tracks[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(bev_base, [bev_pts], False, (255, 255, 0), 2)

                    # Distance estimate in meters from camera-side edge center on BEV
                    dist_m = np.linalg.norm(mapped - camera_reference) / PIXELS_PER_METER
                    cv2.putText(
                        annotated,
                        f"ID {track_id} ~{dist_m:.2f} m",
                        (fx + 6, max(20, fy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # Create BEV heatmap overlay
        blur_ksize = HEAT_BLUR if HEAT_BLUR % 2 == 1 else HEAT_BLUR + 1
        heat_blur = cv2.GaussianBlur(heatmap_acc, (blur_ksize, blur_ksize), 0)
        heat_scaled = np.clip(heat_blur / max(HEAT_MAX, 1e-4), 0.0, 1.0)
        heat_shaped = np.power(heat_scaled, HEAT_VIS_GAMMA)
        heat_norm = (heat_shaped * 255.0).astype(np.uint8)

        heat_color = cv2.applyColorMap(heat_norm, COLORMAP)
        bev_overlay = cv2.addWeighted(bev_base, 0.35, heat_color, 0.65, 0)

        # Camera reference marker used for distance estimate
        cv2.circle(bev_overlay, tuple(camera_reference.astype(int)), 7, (0, 255, 0), -1)
        cv2.putText(
            bev_overlay,
            "camera reference",
            (int(camera_reference[0]) + 10, int(camera_reference[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            bev_overlay,
            "Bird's-eye heatmap (ground plane)",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        if USE_WARPED_GROUND_BACKGROUND:
            cv2.putText(
                bev_overlay,
                "background: warped ground estimate",
                (14, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cam_resized = cv2.resize(annotated, (cam_display_w, cam_display_h))
        combined = np.hstack([cam_resized, bev_overlay])

        writer.write(combined)
        cv2.imshow("Bird's-Eye Demo", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved output video: {output_path}")


if __name__ == "__main__":
    main()
