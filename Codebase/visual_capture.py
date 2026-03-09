import os
import time
import subprocess
import threading
import shutil
import signal
import queue
from pathlib import Path

SEGMENT_TIME = 1

def episode_recorder(feed_url, cam_id, capture_dir, results_dir, buffer_length, detection_queue):

    camera_name = f"cam{cam_id}"
    
    # Subdirectory Creation
    # Level 1 capture subdirectory
    camera_dir = capture_dir / camera_name

    # Level 2 capture subdirectories  
    buffer_dir = camera_dir / "buffer"
    episode_dir = camera_dir / "episode"

    # Create all directories at once
    for path in [camera_dir, buffer_dir, episode_dir]:
        path.mkdir(parents=True, exist_ok=True)


    # Clear buffer if needed
    for seg in buffer_dir.glob("seg*.ts"):
        try:
            seg.unlink()
        except Exception as e:
            print(f"[{camera_name}] Failed to delete segment {seg}: {e}")

    max_segments = int(buffer_length / SEGMENT_TIME)

    # Start ffmpeg segment recorder (copy mode)
    buffer = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-use_wallclock_as_timestamps", "1",
        "-fflags", "+genpts",
        "-i", feed_url,
        "-an",
        "-c:v", "copy",
        "-f", "segment",
        "-segment_time", str(SEGMENT_TIME),
        "-reset_timestamps", "1",
        str(buffer_dir / "seg%d.ts")
    ]

    proc = subprocess.Popen(buffer)

    episode_active = False
    start_time = None
    end_time = None

    try:
        while True:

            # Get message from detection queue
            try:
                msg = detection_queue.get_nowait()
            except queue.Empty:
                msg = None

            # If a message was found, check if it was start or end signal
            if msg and msg[0] == camera_name:
                # Start -> save start timestamp, activate episode logic (stop deleting oldest segments until episode ends,
                #  giving buffer_length seconds of pre-episode footage)
                if msg[1] == "START":
                    episode_active = True
                    start_time = msg[2]
                    print(f"[{camera_name}] Episode START at {start_time}")

                # End -> save end timestamp, finalize the episode by copying video segments into a new directory 
                # and concatenating them into a single .mp4 of the entire episode
                elif msg[1] == "END":
                    end_time = msg[2]
                    print(f"[{camera_name}] Episode END at {end_time}")

                    finalize_episode(camera_name, buffer_dir, episode_dir, results_dir, start_time, end_time)
                    episode_active = False
                    start_time = None
                    end_time = None

            # Rolling video storage when NOT in episode, deletes oldest segments until segments.length <= max_segments
            if not episode_active:
                segments = sorted(buffer_dir.glob("seg*.ts"), key=lambda p: p.stat().st_mtime)
                # Only delete if there are more than max_segments, if so delete oldest excess segments all at once
                excess = len(segments) - max_segments
                if excess > 0:
                    for seg in segments[:excess]:
                        try:
                            seg.unlink()
                        except Exception as e:
                            print(f"Failed to delete {seg}: {e}")

            time.sleep(0.1)

    finally:
        print(f"[{camera_name}] Shutting down.")
        proc.terminate()
        proc.wait()



def finalize_episode(camera_name, buffer_dir, episode_dir, results_dir, start_time, end_time):

    segments = sorted(buffer_dir.glob("seg*.ts"), key=lambda p: p.stat().st_mtime)

    # Give FFmpeg a second to finish writing last segment (avoid concatenating an unfinished file)
    time.sleep(1)

    episode_temp_dir = episode_dir / f"{start_time}"
    episode_temp_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        shutil.copy(seg, episode_temp_dir / seg.name)

    # Build concat list
    concat_file = episode_temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in sorted(episode_temp_dir.glob("seg*.ts")):
            f.write(f"file '{seg.name}'\n")

    output_file = results_dir / f"{camera_name}_{start_time}.mp4"
    
    concat_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_file)
    ]

    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    print(f"[{camera_name}] Episode saved → {output_file}")

    # Cleanup temp directory
    shutil.rmtree(episode_temp_dir)