import os
import cv2
import numpy as np  # Needed for normalization
from ultralytics import solutions

def next_available_filename(base, ext):
    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1

cap = cv2.VideoCapture(r"C:\Users\cpear\Downloads\SquirrelTest.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS,
))
video_output_path = next_available_filename("heatmap_output", ".avi")
video_writer = cv2.VideoWriter(
    video_output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 30,
    (w, h),
)

# Initialize heatmap
heatmap = solutions.Heatmap(
    show=False,
    model="C:\\Users\\cpear\\Documents\\VisualAnalysis\\runs\\detect\\train10\\weights\\best.pt",
    colormap=cv2.COLORMAP_INFERNO,
    device="cuda",
    conf=0.2,
    # region=region_points,
    # classes=[0, 2],
)

cv2.namedWindow("YOLO Heatmap", cv2.WINDOW_NORMAL)
last_frame = None

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    last_frame = im0

    results = heatmap(im0)  # run Heatmap solution
    out_frame = results.plot_im  # processed frame

    # ------- Custom blending logic -------
    if not heatmap.track_ids:  # Check if track list is empty
        if hasattr(heatmap, 'heatmap') and heatmap.heatmap is not None:
            # Normalize the cumulative heatmap data (0-255)
            # Same blending logic as the library (alpha=0.5)
            heatmap_norm = cv2.normalize(
                heatmap.heatmap,
                  None, 0, 255,
                  cv2.NORM_MINMAX
                  ).astype(np.uint8)
            
            # Apply color map
            heatmap_color = cv2.applyColorMap(heatmap_norm, heatmap.colormap)
            
            # Blend with the original frame
            out_frame = cv2.addWeighted(im0, 0.5, heatmap_color, 0.5, 0)
    # ---------------------------------------

    # Write full-res frame to video
    video_writer.write(out_frame)

    # Resize for display
    display_frame = cv2.resize(out_frame, (1280, 720))
    cv2.imshow("YOLO Heatmap", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

if last_frame is not None and hasattr(heatmap, "heatmap") and heatmap.heatmap is not None:
    heatmap_norm = cv2.normalize(heatmap.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, heatmap.colormap)
    final_frame = cv2.addWeighted(last_frame, 0.5, heatmap_color, 0.5, 0)
    image_output_path = next_available_filename("heatmap_final", ".png")
    cv2.imwrite(image_output_path, final_frame)
