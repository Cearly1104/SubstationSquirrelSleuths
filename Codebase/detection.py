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
def squirrel_detector(cameras, config, detection_dir, detection_queues, analysis_queue, stop):

    cam_lookup = {cam.id: cam for cam in cameras}

    # Initialize camera frame states
    for cam in cameras:
        states[cam.id] = {
            "frame": None,
            "timestamp": None,
            "last_update": 0,
            "lock": threading.Lock(),
        }

    model = YOLO(config["model_path"])

    # Initialize and keep track of each camera's episode status
    episode_state = {
        cam.id: {
            "in_episode": False,
            "last_detection_time": None,
            "episode_dir": None,
            "episode_writer": None,
            "annotated_writer": None,
            "detection_streak": 0,
            "no_detection_streak": 0
        }
        for cam in cameras
    }

    # Episode control variables
    episode_cutoff = 2.0    # Seconds to 
    consecutive_frame_threshold = 1 # Required consecutive detection frames before an episode can be started
    # start_threshold = 3
    # end_threshold = 5

    # Main loop, parses frames, associates results with proper camera and prompts capture system for episode start and end
    # Frames with detections are annotated with bounding boxes are saved and path directories are passed to analysis system for tracking
    try: 
        while not stop.is_set():
            loop_start = time.time()

            batch = batch_frames(cameras, config["frozen_camera_threshold"])

            if not batch:
                time.sleep(0.01)
                continue

            # Pull only frames to be passed in as the model's input
            # Done by unpacking the batch tuple into just a list of frames
            frames = [frame for _, frame, _ in batch]
            results = model(frames, conf=config["detection_confidence"], verbose=False)

            # Loop over each result, comparing and updating episode status as appropriate
            for i, (cam_id, frame, timestamp) in enumerate(batch):

                result = results[i]
                boxes = result.boxes
                squirrel_present = boxes is not None and len(boxes) > 0

                state = episode_state[cam_id]
                cam_name = cam_lookup[cam_id].name

                # Start/End episode based on squirrel presence
                if squirrel_present:
                    state["last_detection_time"] = timestamp
                    state["consecutive_frames"] += 1
                    # Start an episode if squirrel detected and not already in one
                    if not state["in_episode"] and state["consecutive_frames"] > consecutive_frame_threshold:
                        event = Event(cam_id, "START", timestamp)
                        detection_queues[cam_id].put(event)

                        state["in_episode"] = True
                        
                        # Set up episode naming and storage
                        episode_name = timestamp.strftime("ep_%Y-%m-%d_%I.%M.%S%p") # Format: ep_YYYY-MM-DD_HH.MM.SS AM/PM
                        episode_dir = detection_dir / episode_name / cam_name
                        episode_dir.mkdir(parents=True, exist_ok=True)
                        state["episode_dir"] = episode_dir

                        # Open an OpenCV VideoWriter to write raw processed frames into an mp4 for analysis input
                        h, w = result.orig_img.shape[:2]
                        state["episode_writer"] = cv2.VideoWriter(
                            str(episode_dir / "_raw.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            config["detection_fps"],
                            (w, h)
                        )
                        # If enabled, do the same but for an annotated video output
                        if config["save_annotated"]:
                            state["annotated_writer"] = cv2.VideoWriter(
                            str(episode_dir / "_untracked.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            config["detection_fps"],
                            (w, h)
                        )

                        #TODO remove
                        print(f"Start episode {state['episode_dir']} (cam {cam_id})")


                else:
                    state["consecutive_frames"] = 0
                    # End an episode if currently in one, but squirrel not detected for the episode cutoff time
                    if state["in_episode"]:
                        last_time = state["last_detection_time"]
                        if last_time and (timestamp - last_time).total_seconds() >= episode_cutoff:
                            event = Event(cam_id, "END", timestamp)
                            detection_queues[cam_id].put(event)

                            # TODO remove
                            print(f"End episode {state['episode_dir']} (cam {cam_id})")

                            # Release VideoWriter(s) and pass completed episode to analysis
                            if state["episode_writer"] is not None:
                                state["episode_writer"].release()
                                state["episode_writer"] = None

                            if state["annotated_writer"] is not None:
                                state["annotated_writer"].release()
                                state["annotated_writer"] = None

                            clip_path = state["episode_dir"] / "_raw.mp4"
                            if clip_path.exists() and clip_path.stat().st_size > 0:
                                analysis_queue.put({
                                    "clip_path": str(clip_path),
                                    "cam_name": cam_name,
                                    "timestamp": timestamp
                                })
                            else:
                                #TODO convert to error
                                print(f"Warning: clip missing or empty for episode {state['episode_dir']} (cam {cam_id}), skipping analysis")

                            # Update camera's episode state after episode completion
                            state["in_episode"] = False
                            state["episode_dir"] = None

                # Write processed frame(s) to output folder
                if state["in_episode"]:
                    if state["episode_writer"] is not None:
                        state["episode_writer"].write(frame)
                    if state["annotated_writer"] is not None:
                        state["annotated_writer"].write(result.plot())

            # #TODO remove these window lines for headless deployment
            #     cv2.imshow(cam_name, result.plot())
            # # Press 'q' in the window to exit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Pressed q — exiting.")
            #     break

            # Processing FPS controlled by looping at intervals of 1/fps
            # If loop was faster than 1/fps, wait for (1/fps - (Loop end time - start time)) seconds
            elapsed = time.time() - loop_start
            if elapsed < 1/config["detection_fps"]:
                time.sleep(1/config["detection_fps"] - elapsed)
        
    # Release writers on shutdown
    finally:
        #TODO remove
        cv2.destroyAllWindows()

        for cam_id, state in episode_state.items():
            if state["episode_writer"] is not None:
                state["episode_writer"].release()
                state["episode_writer"] = None
            if state["annotated_writer"] is not None:
                state["annotated_writer"].release()
                state["annotated_writer"] = None
