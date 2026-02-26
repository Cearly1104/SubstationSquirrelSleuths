from pathlib import Path
from ultralytics import YOLO


def main():
    weights_path = r"C:\\Users\\cpear\\Documents\\VisualAnalysis\\runs\\detect\\train10\\weights\\best.pt"

    video_path_str = input("Path to input video: ").strip()
    video_path = Path(video_path_str)

    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        return

    # Load model
    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    # Run prediction, let Ultralytics handle annotation + saving
    print("Running inference...\n")
    results = model.predict(
        source=str(video_path),
        save=True,                      # saves annotated video
        project="runs/squirrel", # output root
        name="annotated",               # subfolder name
        exist_ok=True,
        conf=0.25,                      # confidence threshold (adjust if needed)
        device="cuda",                  # or "cpu"
    )

    # Track the single highest confidence score
    max_conf = 0.0

    for r in results:
        if r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
            continue

        frame_max = float(r.boxes.conf.max().item())
        if frame_max > max_conf:
            max_conf = frame_max

    # YOLO stores the save dir on each result
    out_dir = Path(results[0].save_dir)

    print("\n=== Done ===")
    print(f"Annotated output saved in: {out_dir}")
    if max_conf > 0:
        print(f"Highest detection confidence in this video: {max_conf:.3f}")
    else:
        print("No detections found (confidence above threshold).")


if __name__ == "__main__":
    main()