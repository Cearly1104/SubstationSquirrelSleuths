import time
import subprocess
import shutil
import queue

SEGMENT_TIME = 1

def episode_recorder(cam, detection_queue, capture_dir, results_dir, buf_len, stop):

    ## Subdirectory Creation
    # Level 1 capture subdirectories
    camera_dir = capture_dir / cam.name

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
            print(f"[{cam.name}] Failed to delete segment {seg}: {e}")

    max_segments = int(buf_len / SEGMENT_TIME)

    # Start ffmpeg segment recorder (copy mode)
    buffer = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-use_wallclock_as_timestamps", "1",
        "-fflags", "+genpts",
        "-i", cam.url,
        "-an",
        "-c:v", "copy",
        "-f", "segment",
        "-segment_time", str(SEGMENT_TIME),
        "-reset_timestamps", "1",
        str(buffer_dir / "seg%d.ts")
    ]

    proc = subprocess.Popen(buffer, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    episode_active = False
    start_time = None
    end_time = None

    try:
        while not stop.is_set():

            # Store a list of events in case of parallel detection triggers
            events = []

            # Add each event from detection queue to events list, don't block
            while True:
                try:
                    events.append(detection_queue.get_nowait())
                except queue.Empty:
                    break

            # Check if each event found is a START or END signal
            for event in events:
                # Start -> save start timestamp, activate episode logic (stop deleting oldest segments until episode ends,
                #  giving buffer_length seconds of pre-episode footage)
                if event.event_type == "START" and not episode_active:
                    episode_active = True
                    start_time = event.timestamp
                    print(f"[{cam.name}] Episode START at {start_time}")

                # End -> save end timestamp, finalize the episode by copying video segments into a new directory 
                # and concatenating them into a single .mp4 of the entire episode
                elif event.event_type == "END" and episode_active:
                    end_time = event.timestamp
                    print(f"[{cam.name}] Episode END at {end_time}")

                    if start_time is not None:
                        finalize_episode(cam.name, buffer_dir, episode_dir, results_dir, start_time)
                    episode_active = False
                    start_time = None
                    end_time = None

            # Rolling video storage when NOT in episode, deletes oldest segments until segments.length <= max_segments
            if not episode_active:
                segments = sorted(buffer_dir.glob("seg*.ts"), key=lambda p: p.stat().st_mtime)
                finished_segments = segments[:-1]    # Avoid deleting newest segment since it may be actively under writing
                # Only delete if there are more than max_segments, if so delete oldest excess segments all at once
                excess = len(finished_segments) - max_segments
                if excess > 0:
                    for seg in finished_segments[:excess]:
                        try:
                            seg.unlink()
                        except Exception as e:
                            print(f"Failed to delete {seg}: {e}")

            time.sleep(0.1)

    finally:
        print(f"[{cam.name}] Shutting down.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[{cam.name}] ffmpeg didn't exit, killing...")
            proc.kill()



def finalize_episode(cam_name, buffer_dir, episode_dir, results_dir, start_time):

    # Store all segment files currently in the buffer folder, sorted by their modification time
    segments = sorted(buffer_dir.glob("seg*.ts"), key=lambda p: p.stat().st_mtime)

    # Give FFmpeg a second to finish writing last segment (avoid concatenating an unfinished file)
    time.sleep(1)

    episode_temp_dir = episode_dir / start_time.strftime("%I.%M.%S%p")
    episode_temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy segments into a new per-episode temp folder
    for seg in segments:
        shutil.copy(seg, episode_temp_dir / seg.name)

    # Build concat list out of present segments, used by FFmpeg to concatenate segments into one video
    concat_file = episode_temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in sorted(episode_temp_dir.glob("seg*.ts")):
            f.write(f"file '{seg.name}'\n")

    # Output episode named with camera id and detection timestamp
    output_file = results_dir / f"{cam_name}_{start_time.strftime('%Y-%m-%d_%I.%M.%S%p')}_raw.mp4" 
    
    # Concate segments into .mp4
    concat_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_file)
    ]

    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    print(f"[{cam_name}] Episode saved → {output_file}")

    # Cleanup temp directory
    shutil.rmtree(episode_temp_dir)