import os
import sys
import threading
import queue
import time
import enum
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

def run_detection_mode(stop_event):

    threads = []
    
    # Verify weights file exists in the right path before doing any work, if not close with error
    WEIGHTS_DIR = Path("Weights")
    if not os.path.exists(WEIGHTS_DIR) or not any(WEIGHTS_DIR.iterdir()):
        print("Model weights not found, stopping process...")
        sys.exit(1)
        

    ########################## Settings ###########################

    MODEL_NAME = "best.pt"
    MODEL_PATH = WEIGHTS_DIR / MODEL_NAME
    DETECTION_FPS = 10
    DETECTION_CONFIDENCE = 0.10
    FROZEN_THRESHOLD = 1.0

    CAMERAS = [               # Camera class objects
#        Camera(id=0, name="cam0", url="rtsp://admin:123456@10.0.0.11:554/profile1"),   # main (highest quality) feed
#        Camera(id=1, name="cam1", url="rtsp://admin:123456@10.0.0.11:554/profile2"),   # sub (low quality) feed
#        Camera(id=2, name="cam2", url="rtsp://admin:123456@10.0.0.11:554/profile3"),   # third (lowest quality) feed
        Camera(id=3, name="cam3", url="rtsp://localhost:8554/desktop"),                 # desktop stream feed
        Camera(id=4, name="cam4", url="rtsp://localhost:8554/desktop")
    ]

    BUFFER_LENGTH = 3       # Pre-trigger recording time in seconds

    # Set these once calibrated for each camera's ground plane.
    # Order: top-left, top-right, bottom-right, bottom-left.
    SOURCE_POINTS = None  # e.g. [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    ####################### Directory setup #######################
    # Level 1 directory
    det_mode_dir = Path("detections")

    # Level 2 directories
    temp_dir = det_mode_dir / "temp"
    daily_dir = det_mode_dir / datetime.now().strftime("%Y-%m-%d")

    # Level 3 directories
    detection_dir = temp_dir / "detection"
    capture_dir = temp_dir / "capture"
    analysis_dir = temp_dir / "analysis"

    # Create all directories at once
    for path in [det_mode_dir, temp_dir, daily_dir, detection_dir, capture_dir, analysis_dir]:
        path.mkdir(parents=True, exist_ok=True)

    ####################### Process Setup ########################
    ## Multithreading Queues 
    # One detection queue per camera, signals capture system to start recording
    # Analysis queue - episode videos are submitted here after finalization
    detection_queues = {cam.id:queue.Queue() for cam in CAMERAS}
    analysis_queue = queue.Queue()

    # Start the main detection thread
    detection_thread = threading.Thread(
        target=detection.squirrel_detector,
        args=(CAMERAS, detection_queues, analysis_queue, detection_dir, MODEL_PATH, FROZEN_THRESHOLD, DETECTION_CONFIDENCE, DETECTION_FPS, stop_event)
    )
    detection_thread.start()
    threads.append(detection_thread)

    # Start a detection helper and recorder thread per camera
    for cam in CAMERAS:
        t0 = threading.Thread(target=detection.detection_helper, args=(cam, stop_event))
        t1 = threading.Thread(target=visual_capture.episode_recorder,
            args=(cam, detection_queues[cam.id], capture_dir, daily_dir, BUFFER_LENGTH, stop_event))
        
        t0.start()
        t1.start()

        threads.append(t0)
        threads.append(t1)

    # # Start the analysis worker thread (processes one video at a time, sequentially)
    # analysis_thread = threading.Thread(
    #     target=visual_analysis.analysis_worker,
    #     args=(analysis_queue, daily_dir, MODEL_PATH, SOURCE_POINTS, stop_event),
    #     daemon=True
    # )
    # analysis_thread.start()
    # threads.append(analysis_thread)

    return threads

def run_recording_mode(stop_event):

    threads = []

    CAMERAS = [               # Camera class objects
#        Camera(id=0, name="cam0", url="rtsp://admin:123456@10.0.0.11:554/profile1"),   # main (highest quality) feed
#        Camera(id=1, name="cam1", url="rtsp://admin:123456@10.0.0.11:554/profile2"),   # sub (low quality) feed
#        Camera(id=2, name="cam2", url="rtsp://admin:123456@10.0.0.11:554/profile3"),   # third (lowest quality) feed
        Camera(id=3, name="cam3", url="rtsp://localhost:8554/desktop"),                 # desktop stream feed
        Camera(id=4, name="cam4", url="rtsp://localhost:8554/desktop")
    ]

    SAVE_MP4 = True
    DELETE_SEGMENTS = True

    ####################### Directory setup #######################
    # Level 1 directory
    rec_mode_dir = Path("recordings")
    rec_mode_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    for cam in CAMERAS:
        t = threading.Thread(target=visual_capture.recording_mode_recorder, args=(cam, timestamp, SAVE_MP4, DELETE_SEGMENTS, rec_mode_dir, stop_event))
        t.start()
        threads.append(t)

    return threads


if __name__ == "__main__":

    stop_event = threading.Event()
    threads = []

    DEFAULT_MODE = Mode.RECORDING
    current_mode = DEFAULT_MODE

    
    if current_mode == Mode.DETECTION:
        threads = run_detection_mode(stop_event)
    elif current_mode == Mode.RECORDING:
        threads = run_recording_mode(stop_event)
    # elif current_mode == Mode.DEBUG:
    #     threads = run_debug_mode(stop_event)
    else:
        print("Mode invalid, shutting down...")
        stop_event.set()
        exit(1)
        

    #######################  Testing ########################
    print("System is running")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
    
        for t in threads:
            t.join()