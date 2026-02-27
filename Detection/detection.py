from ultralytics import YOLO
import cv2
import os
import threading
from datetime import datetime

latest_frame = None
reader_running = True

def rtsp_reader(cap, frame_lock):
    global latest_frame, reader_running
    while reader_running:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame

def run_squirrel_detector(triggers, write_complete, SUB_FEED, DETECT_DIR, WEIGHTS_DIR):
    global reader_running, latest_frame

    # 1. Load model weights
    weight_folder_path = WEIGHTS_DIR
    models = {
        "s_ds2": "12_4_25_yolo11s_ds2.pt",
        "s_ds3": "12_4_25_yolo11s_ds3.pt",
        "l_ds2": "12_4_25_yolo11l_ds2.pt",
        "l_ds3": "12_4_25_yolo11l_ds3.pt",
        "best": "best.pt"
    }

    desired_key = "best"
    weight_path = os.path.join(weight_folder_path, models[desired_key])
    model = YOLO(weight_path)

    # 2. Stream source: RTSP or MP4
    # For RTSP, use something like:
    # stream_source = "rtsp://user:pass@192.168.1.50:554/your/stream/path"
    stream_source = SUB_FEED

    cap = cv2.VideoCapture(stream_source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/stream: {stream_source}")

    # For RTSP, this can help reduce latency (not always honored, but good to try)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Get FPS and frame size for VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback for RTSP or weird files

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 3. Shared state for latest-frame reader
    latest_frame = None
    frame_lock = threading.Lock()
    reader_running = True  # flag to stop reader thread cleanly

    cap_read = cap.read()

    # Start the reader thread
    reader_thread = threading.Thread(target=rtsp_reader, args=(cap, frame_lock), daemon=True)
    reader_thread.start()

    # 4. Episode recording state
    episodes_root=os.path.join(DETECT_DIR, "episodes")
    # os.makedirs(episodes_root, exist_ok=True)

    episode_index = 0
    in_episode = False
    episode_writer = None
    episode_file = None
    last_detection_time = None  # datetime of last detection
    no_detection_delay = 2.0    # seconds after last detection to end episode
    frame_idx = 0               # processed frame index (not RTSP source index)

    cv2.namedWindow("YOLO Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Stream", 960, 540)

    print("Starting stream. Press 'q' in the video window or Ctrl+C in the terminal to exit.")

    try:
        while True:
            # --- Get the latest frame from the reader thread ---
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                # No frame available yet, skip this iteration
                continue

            # Current timestamp as datetime + string for logging
            now_dt = datetime.now()
            timestamp_str = now_dt.strftime("%Y-%m-%d_%H-%M-%S-%f")

            # Run inference
            results = model(frame, conf=0.10, verbose=False)
            r = results[0]
            boxes = r.boxes

            # Decide if squirrel is present in this frame
            squirrel_present = False
            if boxes is not None and len(boxes) > 0:
                # If you later add more classes, filter by class_id here.
                squirrel_present = True

            # --- EPISODE STATE LOGIC ---

            if squirrel_present:
                # Update time of last detection
                last_detection_time = now_dt

                # Start new episode if not already in one
                if not in_episode:
                    episode_name = f"episode_{episode_index:03d}"
                    episode_dir = os.path.join(episodes_root, episode_name)
                    os.makedirs(episode_dir, exist_ok=True)

                    # Open MP4 writer for this episode
                    clip_path = os.path.join(episode_dir, "clip.mp4")
                    episode_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

                    # Open TXT file for this episode
                    detections_path = os.path.join(episode_dir, "detections.txt")
                    episode_file = open(detections_path, "w", encoding="utf-8")

                    in_episode = True
                    print(f"Started {episode_name}")
                    # Trigger queue
                    triggers.put(timestamp_str)
                    timestamp_start = timestamp_str

            else:
                # No squirrel this frame
                if in_episode and last_detection_time is not None:
                    # Check how long since the last detection
                    delta_seconds = (now_dt - last_detection_time).total_seconds()
                    if delta_seconds >= no_detection_delay:
                        # End episode
                        print(f"Ending episode_{episode_index:03d}")
                        in_episode = False
                        episode_index += 1

                        # Close writer and file
                        if episode_writer is not None:
                            episode_writer.release()
                            episode_writer = None
                            write_complete.put(timestamp_start)
                        if episode_file is not None:
                            episode_file.close()
                            episode_file = None

            # --- WRITING DATA WHEN IN EPISODE ---

            # Draw annotations for visualization
            annotated = r.plot()

            if in_episode and episode_writer is not None:
                # Write annotated frame to current episode clip
                episode_writer.write(annotated)

                # If there are detections, log them
                if squirrel_present and episode_file is not None:
                    # Use normalized xywh for easier analysis
                    xywhn = boxes.xywhn.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()

                    for i in range(len(xywhn)):
                        xc, yc, w, h = xywhn[i]
                        class_id = int(cls[i])
                        confidence = float(conf[i])

                        # One line per detection:
                        # frame_idx, timestamp_str, class_id, x_center, y_center, width, height, conf
                        episode_file.write(
                            f"{frame_idx},{timestamp_str},{class_id},"
                            f"{xc:.6f},{yc:.6f},{w:.6f},{h:.6f},{confidence:.4f}\n"
                        )

            # Show in a window (optional)
            cv2.imshow(f"YOLO11 Stream - {desired_key}", annotated)
            frame_idx += 1

            # Press 'q' in the window to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Pressed q — exiting.")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected — exiting.")

    finally:
        # Stop reader thread and clean up
        reader_running = False
        reader_thread.join(timeout=2.0)

        cap.release()
        cv2.destroyAllWindows()

        # Clean up if we were in an episode when exiting
        if episode_writer is not None:
            episode_writer.release()
        if episode_file is not None and not episode_file.closed:
            episode_file.close()

