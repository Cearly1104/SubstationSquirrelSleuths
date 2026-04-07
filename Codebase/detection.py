from ultralytics import YOLO
import cv2
import os
import threading
import time
import queue
from datetime import datetime
from dataclasses import dataclass

# Dictionary of each camera's frame state
# Contains a frame, timestamp, and write lock to eliminate race conditions
states = {}

# Detection event dataclass, holds an event's type, associated camera id, and timestamp
# to easily pass necessary detection info to the capture system
@dataclass
class Event:
    camera_id: int
    event_type: str
    timestamp: datetime

# Detection helper function to grab a camera frame and its associated timestamp to be passed to the YOLO model
# One helper per camera
def detection_helper(camera, stop):

    state = states[camera.id]
    
    # Use OpenCV to open the camera feed for frame capture
    cap = cv2.VideoCapture(camera.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/stream: {camera.url}")
    
    # Set low frame capture buffersize for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Grab frame+current timestamp in a continuous loop 
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
            state["last_update"] = time.monotonic()

# Function to take current frame states at an instance of time and group them in camera order to be processed together in a batch
def batch_frames(cameras, frozen_threshold):

    batch = []
    now = time.monotonic()
    for cam in cameras:
        state = states[cam.id]

        with state["lock"]:
            frame = state["frame"]
            timestamp = state["timestamp"]
            last_update = state["last_update"]

        # Ensure the state has a frame and that it has been updated recently to avoid reprocessing old frames
        if frame is not None and now - last_update <= frozen_threshold:
            batch.append((cam.id, frame, timestamp))

    # Returns a list of fresh cam_id, frame, timestamp tuples
    return batch

# TODO add start and end smoothing
# Main detection function, takes fresh frames from helper->batch functions and processes them with a YOLO model
# Model outputs squirrel detections and prompts the capture system to record squirrel detection episodes
# Detection logs are taken and annotated processed frames are stored for analysis system usage, prompted by episode write completion
def squirrel_detector(cameras, detection_queues, analysis_queue, detection_dir, model_path, frozen_threshold, confidence, fps, stop):

    cam_lookup = {cam.id: cam for cam in cameras}

    # Initialize camera frame states
    for cam in cameras:
        states[cam.id] = {
            "frame": None,
            "timestamp": None,
            "last_update": 0,
            "lock": threading.Lock()
        }

    model = YOLO(model_path)

    # Initialize and keep track of each camera's episode status
    episode_state = {
        cam.id: {
            "in_episode": False,
            "last_detection_time": None,
            "episode_dir": None,
            "episode_writer": None,
            "detection_streak": 0,
            "no_detection_streak": 0
        }
        for cam in cameras
    }

    # Episode control variables
    episode_cutoff = 2.0    # Seconds to 
    # start_threshold = 3
    # end_threshold = 5

    # Main loop, parses frames, associates results with proper camera and prompts capture system for episode start and end
    # Frames with detections are annotated with bounding boxes are saved and path directories are passed to analysis system for tracking
    while not stop.is_set():
        loop_start = time.time()

        batch = batch_frames(cameras, frozen_threshold)

        if not batch:
            time.sleep(0.01)
            continue

        # Pull only frames to be passed in as the model's input
        # Done by unpacking the batch tuple into just a list of frames
        frames = [frame for _, frame, _ in batch]
        results = model(frames, conf=confidence, verbose=False)

        # loop over each result, comparing and updating episode status as appropriate
        for i, (cam_id, frame, timestamp) in enumerate(batch):

            result = results[i]
            boxes = result.boxes
            squirrel_present = boxes is not None and len(boxes) > 0

            state = episode_state[cam_id]
            cam_name = cam_lookup[cam_id].name

            # Start/End episode based on squirrel presence
            if squirrel_present:
                state["last_detection_time"] = timestamp

                # Start an episode if squirrel detected and not already in one
                if not state["in_episode"]:
                    event = Event(cam_id, "START", timestamp)
                    detection_queues[cam_id].put(event)

                    state["in_episode"] = True
                    
                    # Set up episode naming and storage
                    episode_name = timestamp.strftime("ep_%Y-%m-%d_%I.%M.%S%p") # Format: ep_YYYY-MM-DD_HH.MM.SS AM/PM
                    episode_dir = detection_dir / episode_name / cam_name
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    state["episode_dir"] = episode_dir

                    # Open an OpenCV VideoWriter to write processed frames into an mp4 for analysis input
                    h, w = result.orig_img.shape[:2]
                    state["episode_writer"] = cv2.VideoWriter(
                        str(episode_dir / "_raw.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h)
                    )

                    #TODO remove
                    print(f"Start episode {state['episode_dir']} (cam {cam_id})")


            else:
                # End an episode if currently in one, but squirrel not detected for the episode cutoff time
                if state["in_episode"]:
                    last_time = state["last_detection_time"]
                    if last_time and (timestamp - last_time).total_seconds() >= episode_cutoff:
                        event = Event(cam_id, "END", timestamp)
                        detection_queues[cam_id].put(event)

                        # TODO remove
                        print(f"End episode {state['episode_dir']} (cam {cam_id})")

                        # Release VideoWriter and pass completed episode to analysis
                        if state["episode_writer"] is not None:
                            state["episode_writer"].release()
                            state["episode_writer"] = None

                        clip_path = state["episode_dir"] / "_raw.mp4"
                        if clip_path.exists() and clip_path.stat().st_size > 0:
                            analysis_queue.put(str(clip_path))
                        else:
                            #TODO convert to error
                            print(f"Warning: clip missing or empty for episode {state['episode_dir']} (cam {cam_id}), skipping analysis")

                        # Update camera's episode state after episode completion
                        state["in_episode"] = False
                        state["episode_dir"] = None

            # Write processed frame to output folder
            if state["in_episode"] and state["episode_writer"] is not None:
                state["episode_writer"].write(frame)

        #TODO remove these window lines for headless deployment
            cv2.imshow(cam_name, result.plot())
        # Press 'q' in the window to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Pressed q — exiting.")
            break

        # Processing FPS controlled by looping at intervals of 1/fps
        # If loop was faster than 1/fps, wait for (1/fps - (Loop end time - start time)) seconds
        elapsed = time.time() - loop_start
        if elapsed < 1/fps:
            time.sleep(1/fps - elapsed)


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

