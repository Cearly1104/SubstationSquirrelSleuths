import os
import sys
import threading
import queue
from datetime import datetime
from pathlib import Path

import visual_capture

if __name__ == "__main__":

    # Verify weights files exist in the right path before doing any work, if not close with error
    WEIGHTS_DIR = "weights"
    if not os.path.exists(WEIGHTS_DIR):
        print("Model weights not found, stopping process...")
        sys.exit(1)

    ########################## Settings ###########################
    FEEDS = {               # RTSP feed urls
        "cam0": "rtsp://admin:123456@10.0.0.11:554/profile1",   # main (highest quality) feed
        #"cam1": "rtsp://admin:123456@10.0.0.11:554/profile2",   # sub (low quality) feed
#        "cam2": "rtsp://admin:123456@10.0.0.11:554/profile3",  # third (lowest quality) feed
#        "cam3": "rtsp://localhost:8554/desktop"                # desktop stream feed
    }
    BUFFER_LENGTH = 15       # Pre-trigger recording time in seconds

    ####################### Directory setup #######################
    # Level 1 directories
    temp_dir = Path("temp")
    results_dir = Path("results")

    # Level 2 directories under temp
    detection_dir = temp_dir / "detection"
    capture_dir = temp_dir / "capture"
    analysis_dir = temp_dir / "analysis"

    # Level 3 directories
    detection_episodes_dir = detection_dir / "episodes"
    detection_output_dir = detection_dir / "output"
    # No level 3 capture directories
    #analysis_..._dir = analysis_dir / "..." goes here

    # Create all directories at once
    for path in [temp_dir, results_dir, detection_dir, capture_dir, 
                 analysis_dir, detection_episodes_dir, detection_output_dir]:
        
        path.mkdir(parents=True, exist_ok=True)

    ####################### Process Setup ########################
    # One detection queue per camera
    detection_queues = {cam_name: queue.Queue() for cam_name in FEEDS}

    # Start one recorder thread per camera
    threads = []
    for cam_id, (cam_name, feed_url) in enumerate(FEEDS.items()):
        t = threading.Thread(
            target=visual_capture.episode_recorder,
            args=(feed_url, cam_id, capture_dir, results_dir, BUFFER_LENGTH, detection_queues[cam_name])
        )
        t.start()
        threads.append(t)

    #######################  Testing ########################
    print("Rolling buffer started for all cameras.")

    # Manual detection trigger testing, triggers all cameras simultaneously
    while True:
        input("Press ENTER to START episodes for all cameras...")
        for cam_name, q in detection_queues.items():
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            q.put((cam_name, "START", start_time))
        print(f"Sent START signal to all cameras at {start_time}")

        input("Press ENTER again to END episodes for all cameras...")
        for cam_name, q in detection_queues.items():
            end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            q.put((cam_name, "END", end_time))
        print(f"Sent END signal to all cameras at {end_time}")

        print("Episode completed. Press Ctrl+C to exit or start another episode.")
    