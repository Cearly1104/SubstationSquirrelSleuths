from ultralytics import YOLO
import os

# Load model weights
weight_folder_path = r"C:\Users\bnoah\OneDrive\Documents\Class\F25\EE490\Detection\2_25_26_yolo11_weights"

# Weights Denoted with XYZ, where X=previous detection dataset, Y=previous analysis dataset, Z=campus acquired dataset
models = {
    "001": "2_25_26_yolo11l_001.pt",
    "010": "2_25_26_yolo11l_010.pt",
    "011": "2_25_26_yolo11l_011.pt",
    "100": "2_25_26_yolo11l_100.pt",
    "101": "2_25_26_yolo11l_101.pt",
    "110": "2_25_26_yolo11l_110.pt",
    "111": "2_25_26_yolo11l_111.pt",
}

# Video path
folder_path = r"C:\Users\bnoah\OneDrive\Documents\Class\F25\EE490\Detection\spring_testings_videos"
file_name = "Squirrel_Test.mp4"
video_path = os.path.join(folder_path, file_name)

# Frame time and detection percentage data dictionary (stores data relative to keys)
benchmark_data = {}

# Loop through all models
for key, weight_file in models.items():

    print(f"\nRunning Inference with {key}:")

    weight_path = os.path.join(weight_folder_path, weight_file)
    model = YOLO(weight_path)

    results = model.predict(
        source=video_path,
        save=True,
        save_txt=True,
        conf=0.15,
        project="annoated_video_outputs_LowC",
        name=f"Spring_{key}",   # unique folder per model
        exist_ok=True
    )

    benchmark_data[key] = []

    for frame_idx, r in enumerate(results):

        frame_data = {
            "frame": frame_idx,
            "preprocess_ms": r.speed['preprocess'],
            "inference_ms": r.speed['inference'],
            "postprocess_ms": r.speed['postprocess'],
            "total_ms": (
                r.speed['preprocess']
                + r.speed['inference']
                + r.speed['postprocess']
            )
        }

        benchmark_data[key].append(frame_data)

            # Squirrel Detection Percentage
        total_frames = len(results)
        squirrel_frames = 0

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                squirrel_frames += 1

        detection_percentage = (squirrel_frames / total_frames) * 100

        # Structure benchmark_data[key] to store both timing and accuracy
        benchmark_data[key] = {
            "frame_data": benchmark_data[key],
            "squirrel_detection_percent": detection_percentage,
            "total_frames": total_frames,
            "frames_with_squirrel": squirrel_frames
        }
