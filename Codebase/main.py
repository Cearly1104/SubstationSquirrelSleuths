import os
import sys
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import detection
import visual_capture
import visual_analysis

# Camera dataclass, holds a camera's id number, user-defined name, and RTSP url
@dataclass
class Camera:
    id: int
    name: str
    url: str

if __name__ == "__main__":

    # Verify weights file exists in the right path before doing any work, if not close with error
    WEIGHTS_DIR = Path("weights")
    if not os.path.exists(WEIGHTS_DIR) or not any(WEIGHTS_DIR.iterdir()):
        print("Model weights not found, stopping process...")
        sys.exit(1)
        

    ########################## Settings ###########################

    MODEL_NAME = "best.pt"
    MODEL_PATH = WEIGHTS_DIR / MODEL_NAME
    DETECTION_FPS = 10

    CAMERAS = [               # Camera class objects
        Camera(id=0, name="cam0", url="rtsp://admin:123456@10.0.0.11:554/profile1"),   # main (highest quality) feed
#        Camera(id=1, name="cam1", url="rtsp://admin:123456@10.0.0.11:554/profile2"),   # sub (low quality) feed
#        Camera(id=2, name="cam2", url="rtsp://admin:123456@10.0.0.11:554/profile3"),   # third (lowest quality) feed
#        Camera(id=3, name="cam3", url="rtsp://localhost:8554/desktop")                 # desktop stream feed
    ]

    BUFFER_LENGTH = 15       # Pre-trigger recording time in seconds

    # Set these once calibrated for each camera's ground plane.
    # Order: top-left, top-right, bottom-right, bottom-left.
    SOURCE_POINTS = None  # e.g. [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    ####################### Directory setup #######################
    # Level 1 directories
    temp_dir = Path("temp")
    results_dir = Path("results")

    # Level 2 directories under temp
    detection_dir = temp_dir / "detection"
    capture_dir = temp_dir / "capture"
    analysis_dir = temp_dir / "analysis"

    # Level 3 directories get handled individually within individual .py subfiles

    # Create all directories at once
    for path in [temp_dir, results_dir, detection_dir, capture_dir, analysis_dir]:
        path.mkdir(parents=True, exist_ok=True)

    ####################### Process Setup ########################
    ## Multithreading Queues 
    # One frame queue per camera, det. helpers pass a frame from their assigned camera to the main det. thread per frame-cycle
    # One detection queue per camera, signals capture system to start recording
    # Analysis queue - episode videos are submitted here after finalization
    detection_queues = {cam.id:queue.Queue() for cam in CAMERAS}
    analysis_queue = queue.Queue()

    stop_event = threading.Event
    threads = []

    # Start the main detection thread
    detection_thread = threading.Thread(
        target=detection.squirrel_detector,
        args=(CAMERAS, detection_queues, analysis_queue, detection_dir, MODEL_PATH, DETECTION_FPS, stop_event)
    )
    detection_thread.start()
    threads.append(detection_thread)

    # Start a detection helper and recorder thread per camera
    for cam in CAMERAS:
        t0 = threading.Thread(target=detection.helper, args=(cam, stop_event), daemon=True)
        t1 = threading.Thread(target=visual_capture.episode_recorder,
            args=(cam, detection_queues[cam.id], capture_dir, results_dir, BUFFER_LENGTH, stop_event))
        
        t0.start()
        t1.start()
        
        threads.append(t0)
        threads.append(t1)

    # Start the analysis worker thread (processes one video at a time, sequentially)
    analysis_thread = threading.Thread(
        target=visual_analysis.analysis_worker,
        args=(analysis_queue, results_dir, MODEL_PATH, SOURCE_POINTS),
        daemon=True
    )
    analysis_thread.start()
    threads.append(analysis_thread)

    #######################  Testing ########################
    print("Rolling buffer started for all cameras.")

    # Manual detection trigger testing, triggers all cameras simultaneously
    while True:
        input("Press ENTER to START episodes for all cameras...")
        for cam_id, q in detection_queues.items():
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            q.put((cam_id, "START", start_time))
        print(f"Sent START signal to all cameras at {start_time}")

        input("Press ENTER again to END episodes for all cameras...")
        for cam_id, q in detection_queues.items():
            end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            q.put((cam_id, "END", end_time))
        print(f"Sent END signal to all cameras at {end_time}")

        # Wait briefly for finalize_episode to write the mp4, then queue analysis
        time.sleep(3)
        for cam in CAMERAS:
            episode_video = results_dir / f"{cam.name}_{start_time}.mp4"
            if episode_video.exists():
                print(f"[main] Queuing analysis for: {episode_video}")
                analysis_queue.put(str(episode_video))
            else:
                print(f"[main] Episode video not found yet: {episode_video}")

        print("Episode completed. Press Ctrl+C to exit or start another episode.")
    
