from ultralytics import YOLO
import cv2
import threading
import time
import torch
import queue
from datetime import datetime
from dataclasses import dataclass
import pipeline_status
from visual_analysis import AnalysisResult

# Dictionary of each camera's frame state
# Contains a frame, timestamp, and write lock to eliminate race conditions
states = {}

# Detection event dataclass, holds an event's type, associated camera id, and timestamp
# to easily pass necessary detection info to the capture system and user deliverables text logger
@dataclass
class Event:
    camera_id: int
    event_type: str
    timestamp: datetime
    confidence: float

@dataclass
class EpisodeSummary:
    camera_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    squirrels: int
    avg_confidence: float


# Detection helper function to grab a camera frame and its associated timestamp to be passed to the YOLO model
# One helper per camera
def detection_helper(camera, stop):

    state = states[camera.id]

    # Retry connecting to the stream — cameras and demo RTSP streams may not
    # be available the instant the pipeline starts.  Retry for up to 30 s
    # before giving up, so a slow-starting source doesn't silently kill the
    # thread.
    cap = None
    for _ in range(30):
        if stop.is_set():
            return
        cap = cv2.VideoCapture(camera.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            break
        cap.release()
        cap = None
        time.sleep(1)

    if cap is None:
        raise RuntimeError(f"Could not open video/stream after 30 attempts: {camera.url}")

    # Set low frame capture buffersize for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Grab frame+current timestamp in a continuous loop
    # Track consecutive empty reads — if the stream stalls after connecting
    # (e.g. mediamtx accepted the connection before the publisher was ready),
    # reopen the capture rather than looping on empty reads indefinitely.
    empty_count = 0
    while not stop.is_set():
        ret, frame = cap.read()
        timestamp = datetime.now()
        if not ret:
            empty_count += 1
            time.sleep(0.01)
            # ~3 s of consecutive empty reads → reconnect
            if empty_count >= 300:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(camera.url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                empty_count = 0
            continue
        empty_count = 0

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


# Main detection function, takes fresh frames from helper->batch functions and processes them with a YOLO model
# Model outputs squirrel detections and prompts the capture system to record squirrel detection episodes
# Detection logs are taken and annotated processed frames are stored for analysis system usage, prompted by episode write completion
def squirrel_detector(cameras, config, detection_dir, detection_queues, logging_queue, stop, status_dir=None):

    cam_lookup = {cam.id: cam for cam in cameras}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config["model_path"]).to(device)
    print(f"Detection model running on {device}")

    # Initialize and keep track of each camera's episode status
    episode_state = {
        cam.id: {
            "in_episode": False,
            "last_detection_time": None,
            "episode_dir": None,
            "episode_writer": None,
            "annotated_writer": None,
            "detection_streak": 0,
            "no_detection_streak": 0,
            "consecutive_frames": 0,

            # Logging info
            "start_time": None,
            "conf_sum": 0.0,
            "conf_count": 0,
            "peak_squirrels": 0
        }
        for cam in cameras
    }

    # Episode control variables
    episode_cutoff = config["episode_cutoff"]       # Consecutive seconds of no squirrel detection before an episode can be ended
    consecutive_frame_threshold = config["consecutive_frame_threshold"] # Required consecutive detection frames before an episode can be started

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
                conf = float(boxes.conf.max()) if squirrel_present else 0.0
                
                state = episode_state[cam_id]
                cam_name = cam_lookup[cam_id].name

                # Start/End episode based on squirrel presence
                if squirrel_present:
                    state["last_detection_time"] = timestamp
                    state["consecutive_frames"] += 1
                    state["peak_squirrels"] = max(state["peak_squirrels"], len(boxes))
                    state["conf_sum"] += conf
                    state["conf_count"] += 1

                    # Start an episode if squirrel detected and not already in one
                    if not state["in_episode"] and state["consecutive_frames"] > consecutive_frame_threshold:
                        event = Event(cam_id, "START", timestamp, conf)
                        detection_queues[cam_id].put(event)
                        logging_queue.put(event)

                        state["in_episode"] = True
                        state["start_time"] = timestamp
                        if status_dir:
                            pipeline_status.set_recording(status_dir, True)

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
                            event = Event(cam_id, "END", timestamp, conf)
                            
                            avg_conf = (state["conf_sum"] / state["conf_count"] if state["conf_count"] > 0 else 0.0)
                            summary = EpisodeSummary(
                                camera_id=cam_id,
                                start_timestamp=state["start_time"],
                                end_timestamp=timestamp,
                                squirrels=state["peak_squirrels"],
                                avg_confidence=avg_conf
                            )

                            detection_queues[cam_id].put(event)
                            logging_queue.put(event)
                            logging_queue.put(summary)

                            # TODO remove
                            print(f"End episode {state['episode_dir']} (cam {cam_id})")

                            # Release VideoWriter(s) and pass completed episode to analysis
                            if state["episode_writer"] is not None:
                                state["episode_writer"].release()
                                state["episode_writer"] = None

                            if state["annotated_writer"] is not None:
                                state["annotated_writer"].release()
                                state["annotated_writer"] = None


                            # Update camera's episode state after episode completion
                            state["in_episode"] = False
                            if status_dir:
                                still_recording = any(s["in_episode"] for s in episode_state.values())
                                pipeline_status.set_recording(status_dir, still_recording)
                            state["episode_dir"] = None
                            state["consecutive_frames"] = 0
                            state["last_detection_time"] = None
                            state["start_time"] = None
                            state["conf_sum"] = 0.0
                            state["conf_count"] = 0
                            state["peak_squirrels"] = 0
                            

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

# Episode-less detection for RECORDING mode, only serves to see detection window(s) when run_detection_window setting is enabled
def recording_mode_detector(cameras, config, timestamp, output_dir, stop):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config["model_path"]).to(device)
    print(f"Recording mode detection window running on {device}")

    writers = {}
    if config["save_annotated"]:
        for cam in cameras:
            cam_dir = output_dir / timestamp.strftime("%Y-%m-%d_%I.%M.%S%p") / "final"
            cam_dir.mkdir(parents=True, exist_ok=True)
            writers[cam.id] = None

    try:
        while not stop.is_set():
            loop_start = time.time()

            batch = batch_frames(cameras, config["frozen_camera_threshold"])
            if not batch:
                time.sleep(0.01)
                continue

            frames = [frame for _, frame, _ in batch]
            results = model(frames, conf=config["detection_confidence"], verbose=False)

            for i, (cam_id, _, _) in enumerate(batch):
                cam_name = next(cam.name for cam in cameras if cam.id == cam_id)
                annotated = results[i].plot()

                if config["run_detection_window"]:
                    cv2.imshow(f"Detection - {cam_name}", annotated)

                if config["save_annotated"]:
                    if writers[cam_id] is None:
                        h, w = annotated.shape[:2]
                        final_dir = output_dir / timestamp.strftime("%Y-%m-%d_%I.%M.%S%p") / "final"
                        writers[cam_id] = cv2.VideoWriter(
                            str(final_dir / f"{cam_name}_{timestamp.strftime('%Y-%m-%d_%I.%M.%S%p')}_annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            config["detection_fps"],
                            (w, h)
                        )
                    writers[cam_id].write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - loop_start
            if elapsed < 1 / config["detection_fps"]:
                time.sleep(1 / config["detection_fps"] - elapsed)

    finally:
        if config["run_detection_window"]:
            cv2.destroyAllWindows()
        for writer in writers.values():
            if writer is not None:
                writer.release()
                
def episode_logger(cameras, config, logging_queue, det_mode_dir, stop_event):

    cam_lookup = {cam.id: cam for cam in cameras}
    
    LifetimeLog = det_mode_dir / "lifetime_log.txt"
    date = None
    DailyLog = None

    # Write lifetime header lines if file doesn't exist yet
    if not LifetimeLog.exists():
        print((" Lifetime Squirrel Detection Log ").center(170, "=") + "\n")
        with open(LifetimeLog, "w", encoding="utf-8") as f:
            f.write((" Lifetime Squirrel Detection Log ").center(170, "=") + "\n")

    while not stop_event.is_set():

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # Handle date rollovers, write daily header lines if daily log doesnt exist yet
        if date != date_str:
            date = date_str

            daily_dir = det_mode_dir / date_str
            daily_dir.mkdir(parents=True, exist_ok=True)

            DailyLog = daily_dir / f"{date_str}_log.txt"

            date_header = (f" {date_str} ").center(170, "=")

            # Write to daily log (create if doesn't exist)
            if not DailyLog.exists():

                print(date_header + "\n")
                with open(DailyLog, "w", encoding="utf-8") as f:
                    f.write(date_header + "\n\n")

                print(date_header + "\n")
                # Append date header to lifetime log too
                with open(LifetimeLog, "a", encoding="utf-8") as f:
                    f.write("\n" + date_header + "\n\n")

        # Check logging queue for new episode events
        try:
            entry = logging_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if entry is None:
            break

        # If entry is only an Event, write the Event fields into the logs
        elif isinstance(entry, Event):

            ts = entry.timestamp.strftime("%I:%M:%S.%f")[:-3] + entry.timestamp.strftime("%p")
            e_type = entry.event_type
            cam_name = cam_lookup[entry.camera_id].name
            conf = f"{entry.confidence:.2f}" if entry.confidence is not None else ""

            if e_type == "START":
                line = (f"[{ts}] - EP_{e_type:<7} - Camera: {cam_name:<20} - Confidence: {conf}")
            elif e_type == "END":
                line = (f"[{ts}] - EP_{e_type:<7} - Camera: {cam_name:<20}")
            else:
                continue

            print(line)

            with open(LifetimeLog, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if DailyLog:
                with open(DailyLog, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

        # If entry is an AnalysisResult, log the completed analysis stage
        elif isinstance(entry, AnalysisResult):
            ts  = entry.completed_at.strftime("%I:%M:%S.%f")[:-3] + entry.completed_at.strftime("%p")
            ep  = entry.episode_timestamp.strftime("%I:%M:%S.%f")[:-3] + entry.episode_timestamp.strftime("%p")
            line = (
                f"[{ts}] - AN_{entry.stage:<9} - Camera: {entry.cam_name:<20} - "
                f"Episode: {ep} - Output: {entry.output_file}"
            )
            print(line)
            with open(LifetimeLog, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if DailyLog:
                with open(DailyLog, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

        # If entry is an Episode Summary, write the Summary fields into the logs
        elif isinstance(entry, EpisodeSummary):
            start = entry.start_timestamp.strftime("%I:%M:%S.%f")[:-3] + entry.start_timestamp.strftime("%p")
            end = entry.end_timestamp.strftime("%I:%M:%S.%f")[:-3] + entry.end_timestamp.strftime("%p")

            duration = entry.end_timestamp - entry.start_timestamp
            total_seconds = duration.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            milliseconds = int((total_seconds % 1) * 1000)
            duration_str = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

            line = (
                f"[{end}] - EP_SUMMARY - "
                f"Camera: {cam_lookup[entry.camera_id].name:<20} - "
                f"Start: {start} - End: {end} - "
                f"Duration: {duration_str} - "
                f"Squirrels: {entry.squirrels:02d} - "
                f"Avg Confidence: {entry.avg_confidence:.2f}"
            )

            print(line)

            with open(LifetimeLog, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if DailyLog:
                with open(DailyLog, "a", encoding="utf-8") as f:
                    f.write(line + "\n")