from ultralytics import YOLO
import cv2
import os
import threading
import time
import queue
from datetime import datetime
from dataclasses import dataclass

# Dictionary of each camera's frame state
# Contains a frame, timestamp, and write lock
states = {}

# Detection event dataclass, holds an event's type, associated camera id, and timestamp
@dataclass
class Event:
    camera_id: int
    event_type: str
    timestamp: datetime

def initialize_frames(cameras):
    for cam in cameras:
        states[cam.id] = {
            "frame": None,
            "timestamp": None,
            "lock": threading.Lock()
        }

def helper(camera, stop):

    state = states[camera.id]
    
    # Use OpenCV to open the camera feed for frame capture
    cap = cv2.VideoCapture(camera.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/stream: {camera.url}")
    
    # Set low frame capture buffersize for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Grab frame+current timestamp in a loop 
    while not stop.is_set():
        ret, frame = cap.read()
        timestamp = datetime.now()
        if not ret:
            time.sleep(0.01)
            continue
        
        # Update the camera's latest frame info if unlocked
        with state["lock"]:
            state["frame"] = frame
            state["timestamp"] = timestamp


def batch_frames(cameras):

    batch = []

    for cam in cameras:
        state = states[cam.id]

        with state["lock"]:
            frame = state["frame"]
            timestamp = state["timestamp"]

        if frame is None:
            continue

        batch.append((cam.id, frame, timestamp))

    return batch

# TODO add start and end smoothing
def squirrel_detector(cameras, detection_queues, analysis_queue, detection_dir, model_path, confidence, fps, stop):

    cam_lookup = {cam.id: cam for cam in cameras}

    initialize_frames(cameras)

    ## Subdirectory Creation
    # Level 1 detection subdirectories
    detection_episodes_dir = detection_dir / "episodes"

    # Create all directories at once
    for path in [detection_episodes_dir]:
        path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    episode_state = {
        cam.id: {
            "in_episode": False,
            "last_detection_time": None,
            "episode_index": 0
        }
        for cam in cameras
    }
    no_detection_delay = 2.0

    while not stop.is_set():
        loop_start = time.time()

        batch = batch_frames(cameras)

        if len(batch) < 1:
            time.sleep(0.01)
            continue

        # Pull only frames to be passed in as the model's input
        # Done by unpacking the batch tuple into just a list of frames
        frames = [frame for _, frame, _ in batch]

        results = model(frames, confidence, verbose=False)

        for i, (cam_id, frame, timestamp) in enumerate(batch):

            result = results[i]
            boxes = result.boxes

            squirrel_present = boxes is not None and len(boxes) > 0

            state = episode_state[cam_id]


            if squirrel_present:
                state["last_detection_time"] = timestamp

                if not state["in_episode"]:
                    event = Event(cam_id, "START", timestamp)
                    detection_queues[cam_id].put(event)

                    state["in_episode"] = True
                    print(f"Start episode {state['episode_index']} (cam {cam_id})")


                    cam_name = cam_lookup[cam_id].name
                    episode_name = f"ep_{state['episode_index']:03d}"

                    episode_dir = detection_episodes_dir / cam_name / episode_name
                    episode_dir.mkdir(parents=True, exist_ok=True)

            else:
                if state["in_episode"]:
                    last_time = state["last_detection_time"]
                    now = datetime.now()
                    if last_time and ((timestamp - last_time).total_seconds() >= no_detection_delay
                    or (now - last_time).total_seconds() >= no_detection_delay):
                        # TODO use 'now' if thats what triggered end of episode
                        event = Event(cam_id, "END", timestamp)
                        detection_queues[cam_id].put(event)

                        print(f"End episode {state['episode_index']} (cam {cam_id})")

                        state["in_episode"] = False
                        state["episode_index"] += 1

            annotated = result.plot()
            cv2.imshow(cam_name, annotated)

        cv2.waitKey(1)

        elapsed = time.time() - loop_start
        sleep_time = max(0, (1 / fps) - elapsed)
        time.sleep(sleep_time)





def run_squirrel_detector(triggers, write_complete, SUB_FEED, detection_dir, WEIGHTS_DIR):
    global reader_running, latest_frame

    ## Subdirectory Creation
    # Level 1 detection subdirectories
    detection_episodes_dir = detection_dir / "episodes"
    detection_inputs_dir = detection_dir / "input_frames"

    # Create all directories at once
    for path in [detection_episodes_dir, detection_inputs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # 1. Load model weights
    weight_folder_path = WEIGHTS_DIR
    models = {
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
                    episode_dir = os.path.join(detection_episodes_dir, episode_name)
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

