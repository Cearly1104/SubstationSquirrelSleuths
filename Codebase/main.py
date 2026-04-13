import os
import sys
import threading
import queue
import time
import enum
import json
import platform
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import detection
import visual_capture
import visual_analysis

# Enum for different system modes
class Mode(enum.Enum):
    DETECTION = enum.auto()
    RECORDING = enum.auto()
    DEBUG = enum.auto()

# Camera dataclass, holds a camera's id number, user-defined name, and RTSP url
@dataclass
class Camera:
    id: int
    name: str
    url: str

current_mode = None

# Load json contents from settings.json file at root
def load_settings(path="settings.json"):
    with open(path, "r") as f:
        settings = json.load(f)
    return settings

# Ensure json contents are valid
def validate_settings(settings):
    required_params = ["default_mode", "modes", "cameras",]

    for param in required_params:
        if param not in settings:
            raise ValueError(f"Missing required parameter: {param} in settings.json")
        
    if settings["default_mode"] not in settings["modes"]:
        raise ValueError("Default mode doesn't exist in settings.json's modes list")
    
    # Validate each camera field's parameters
    for cam_cfg in settings["cameras"]:
        cam_id = cam_cfg.get("id", "unknown")

        # Validate enabled field is a boolean if present
        enabled = cam_cfg.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError(f"Camera id:{cam_id} has a non-boolean 'enabled' field in settings.json")

        # Validate URL exists
        url = cam_cfg.get("url")
        if not url:
            raise ValueError(f"Camera id:{cam_id} is missing a URL in settings.json")
    
    
# Bulid camera list from json contents
def build_cameras(settings):
    cameras = []

    for cam_cfg in settings["cameras"]:
        if cam_cfg.get("enabled", True):

            # Save name as either name field, or cam{id} if name field doesnt exist
            name = cam_cfg.get("name", f"cam{cam_cfg['id']}")
            url = cam_cfg["url"]
            auth = cam_cfg.get("auth", {})
            un_env = auth.get("username_env")
            pw_env = auth.get("password_env")

            # if username and password env var fields are set, insure the env vars exist and insert them into the rtsp feed url
            if un_env and pw_env:
                username = os.environ.get(un_env)
                password = os.environ.get(pw_env)

                if not username or not password:
                    raise EnvironmentError(f"Camera '{cam_cfg['name']}' requires env vars '{un_env}' and '{pw_env}', but one or both are not set.")

                # Add "username:password@" into the rtsp url if the camera has a set password
                url = url.replace("rtsp://", f"rtsp://{username}:{password}@")

            cameras.append(Camera(id=cam_cfg["id"], name=name, url=url))

    if not cameras:
        raise ValueError("No cameras enabled in settings.json")

    return cameras
    
    
def run_detection_mode(settings, stop_event):

    threads = []
    
    # Verify weights file exists in the right path before doing any work, if not close with error
    WEIGHTS_DIR = Path("Weights")
    if not os.path.exists(WEIGHTS_DIR) or not any(WEIGHTS_DIR.iterdir()):
        print("Model weights not found, stopping process...")
        sys.exit(1)
        

    ########################## Settings ###########################
    cfg = settings["modes"]["DETECTION"]

    CAMERAS = build_cameras(settings)

    # Detection system settings
    detection_cfg = {
        "model_path": WEIGHTS_DIR / cfg["model_name"],
        "detection_fps": cfg["detection_fps"],
        "detection_confidence": cfg["detection_confidence"],
        "frozen_camera_threshold": cfg["frozen_camera_threshold"],
        "save_annotated": cfg["save_annotated"],
        "episode_cutoff": cfg["episode_cutoff"],
        "consecutive_frame_threshold": cfg["episode_cutoff"],
        "episode_grouping_cutoff": cfg["episode_grouping_cutoff"]
    }

    # Capture system settings
    capture_cfg = {
        "rolling_buffer_length": cfg["rolling_buffer_length"],
        "segment_time": cfg["segment_time"]
    }

    # Analysis system settings
    analysis_cfg = {
        "model_path": WEIGHTS_DIR / cfg["model_name"],
        "source_points": cfg["source_points"],
        "plane_width_m": cfg["plane_width_m"],
        "plane_height_m": cfg["plane_height_m"],
        "pixels_per_meter": cfg["pixels_per_meter"],
        "detection_confidence": cfg["detection_confidence"]
    }

    # User Deliverables system settings
    deliverables_cfg = {
        "detection_logging": cfg["detection_logging"]
    }

    ####################### Directory setup #######################
    det_mode_dir = Path("detections")
    temp_dir = det_mode_dir / "temp"
    daily_dir = det_mode_dir / datetime.now().strftime("%Y-%m-%d")
    detection_dir = temp_dir / "detection"
    capture_dir = temp_dir / "capture"

    for path in [det_mode_dir, temp_dir, daily_dir, detection_dir, capture_dir]:
        path.mkdir(parents=True, exist_ok=True)

    ####################### Process Setup ########################
    ## Multithreading Queues 
    # One detection queue per camera, signals capture system to start recording
    # Analysis queue - episode video paths are submitted here after finalization
    detection_queues = {cam.id:queue.Queue() for cam in CAMERAS}
    analysis_queue = queue.Queue()
    logging_queue = queue.Queue()

    # Initialize camera frame states here to avoid race condition
    for cam in CAMERAS:
            detection.states[cam.id] = {
                "frame": None,
                "timestamp": None,
                "last_update": 0,
                "lock": threading.Lock()
            }

    # Start the main detection thread
    detection_thread = threading.Thread(
        target=detection.squirrel_detector,
        args=(CAMERAS, detection_cfg, detection_dir, detection_queues, logging_queue, stop_event, det_mode_dir)
    )
    detection_thread.start()
    threads.append(detection_thread)

    # Start the logging thread if enabled
    if deliverables_cfg["detection_logging"]:
        logging_thread = threading.Thread(
            target=detection.episode_logger,
            args=(CAMERAS, logging_queue, det_mode_dir, stop_event)
        )
        logging_thread.start()
        threads.append(logging_thread)
    
    # Start a detection helper and recorder thread per camera
    for cam in CAMERAS:
        t0 = threading.Thread(target=detection.detection_helper, args=(cam, stop_event))
        t1 = threading.Thread(target=visual_capture.episode_recorder,
            args=(cam, capture_cfg, capture_dir, detection_queues[cam.id], analysis_queue, stop_event))
        
        t0.start()
        t1.start()

        threads.append(t0)
        threads.append(t1)

    # Start the analysis worker thread (processes one video at a time, sequentially)
    analysis_thread = threading.Thread(
        target=visual_analysis.analysis_worker,
        args=(analysis_cfg, analysis_queue, stop_event, det_mode_dir),
        kwargs={"logging_queue": logging_queue if deliverables_cfg["detection_logging"] else None},
        daemon=True
    )
    analysis_thread.start()
    threads.append(analysis_thread)

    return threads

def run_recording_mode(settings, stop_event):

    threads = []

    ########################## Settings ###########################
    cfg = settings["modes"]["RECORDING"]

    detection_cfg = {
        "run_detection_window": cfg["run_detection_window"],
        "model_path": Path("Weights") / cfg["model_name"],
        "detection_fps": cfg["detection_fps"],
        "detection_confidence": cfg["detection_confidence"],
        "frozen_camera_threshold": cfg["frozen_camera_threshold"],
        "save_annotated": cfg["save_annotated"]
    }

    capture_cfg = {
        "save_mp4": cfg["save_mp4"],
        "delete_segments": cfg["delete_segments"],
        "segment_time": cfg["segment_time"],
    }

    if detection_cfg["run_detection_window"] is True:
        # Verify weights file exists in the right path before doing any work, if not close with error
        WEIGHTS_DIR = Path("Weights")
        if not os.path.exists(WEIGHTS_DIR) or not any(WEIGHTS_DIR.iterdir()):
            print("Recording mode detection enabled, but model weights not found, stopping process...")
            sys.exit(1)
        # Verify system isn't headless when detection window is enabled, if it is, close with error
        if platform.system() == "Linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("Recording mode detection window enabled, but no display found, stopping process...")
            sys.exit(1)

    CAMERAS = build_cameras(settings)


    ####################### Directory setup #######################
    rec_mode_dir = Path("recordings")
    rec_mode_dir.mkdir(parents=True, exist_ok=True)

    # Status file always lives in detections/ so the web app can find it regardless of mode
    status_dir = Path("detections")
    status_dir.mkdir(parents=True, exist_ok=True)

    # Initialize camera frame states here to avoid race condition
    for cam in CAMERAS:
            detection.states[cam.id] = {
                "frame": None,
                "timestamp": None,
                "last_update": 0,
                "lock": threading.Lock()
            }

    timestamp = datetime.now()
    for cam in CAMERAS:
        t = threading.Thread(target=visual_capture.recording_mode_recorder, args=(cam, capture_cfg, timestamp, rec_mode_dir, stop_event, status_dir))
        t.start()
        threads.append(t)

    if detection_cfg["run_detection_window"]:
        t1 = threading.Thread(target=detection.recording_mode_detector, args=(CAMERAS, detection_cfg, timestamp, rec_mode_dir, stop_event))
        t1.start()
        threads.append(t1)
        for cam in CAMERAS:
            t2 = threading.Thread(target=detection.detection_helper, args=(cam, stop_event))
            t2.start()
            threads.append(t2)
        
    return threads



if __name__ == "__main__":

    # Load user settings from settings.json
    settings = load_settings()
    validate_settings(settings)

    stop_event = threading.Event()
    threads = []

    # Set system's mode
    current_mode = Mode[settings["default_mode"]]
    
    # Run system in selected mode
    if current_mode == Mode.DETECTION:
        threads = run_detection_mode(settings, stop_event)
    elif current_mode == Mode.RECORDING:
        threads = run_recording_mode(settings, stop_event)
    # elif current_mode == Mode.DEBUG:
    #     threads = run_debug_mode(stop_event)
    else:
        print("Mode invalid, shutting down...")
        stop_event.set()
        exit(1)
        
    print("System is running")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
    
        for t in threads:
            t.join()