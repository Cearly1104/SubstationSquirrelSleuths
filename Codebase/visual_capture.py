import time
import subprocess
import shutil
import queue
import pipeline_status

from pathlib import Path

def episode_recorder(cam, config, capture_dir, detection_queue, stop):

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

    max_segments = int(config["rolling_buffer_length"] / config["segment_time"])

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
        "-segment_time", str(config["segment_time"]),
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
                    group_dir = event.group_dir
                    print(f"[{cam.name}] Episode START at {start_time}")

                # End -> save end timestamp, finalize the episode by copying video segments into a new directory 
                # and concatenating them into a single .mp4 of the entire episode
                elif event.event_type == "END" and episode_active:
                    end_time = event.timestamp
                    print(f"[{cam.name}] Episode END at {end_time}")

                    if start_time is not None:
                        finalize_episode(cam.name, buffer_dir, episode_dir, start_time, group_dir)
                    episode_active = False
                    start_time = None
                    end_time = None

            # Rolling video storage when NOT in episode, deletes oldest segments until segments.length <= max_segments
            if not episode_active:
                segments = sorted(buffer_dir.glob("seg*.ts"), key=lambda p: int(p.stem[3:]))
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



def finalize_episode(cam_name, buffer_dir, episode_dir, start_time, group_dir):

    episode_ts_str = start_time.strftime('%Y-%m-%d_%I.%M.%S%p')
    date_str = start_time.strftime("%Y-%m-%d")
    results_path = Path("detections") / date_str / group_dir / episode_ts_str / cam_name
    results_path.mkdir(parents=True, exist_ok=True)
    

    # Give FFmpeg a second to finish writing last segment (avoid concatenating an unfinished file)
    time.sleep(1)

    # Store all segment files currently in the buffer folder, sorted by their segment number
    segments = sorted(buffer_dir.glob("seg*.ts"), key=lambda p: int(p.stem[3:]))  # strips "seg" prefix, parses remainder as int

    if not segments:
        print(f"Warning: no segments found for {cam_name}, skipping episode")
        return

    episode_temp_dir = episode_dir / start_time.strftime("%I.%M.%S%p")
    episode_temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy segments into a new per-episode temp folder
    for seg in segments:
        shutil.copy(seg, episode_temp_dir / seg.name)

    # Build concat list out of present segments, used by FFmpeg to concatenate segments into one video
    concat_file = episode_temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in sorted(episode_temp_dir.glob("seg*.ts"), key=lambda p: int(p.stem[3:])):
            f.write(f"file '{seg.name}'\n")

    # Output episode named with camera id and detection timestamp
    output_file = results_path / f"{cam_name}_{episode_ts_str}_raw.mp4"
    
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

    print(f"[{cam_name}] Episode saved -> {output_file}")

def recording_mode_recorder(cam, config, timestamp, results_dir, stop, status_dir=None):

    ## Subdirectory creation
    output_dir = results_dir / timestamp.strftime("%Y-%m-%d_%I.%M.%S%p")
    segment_dir = output_dir / "segments" / cam.name

    for path in [output_dir, segment_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Start ffmpeg segment recorder (copy mode)
    recorder = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-use_wallclock_as_timestamps", "1",
        "-fflags", "+genpts",
        "-i", cam.url,
        "-an",
        "-c:v", "copy",
        "-f", "segment",
        "-segment_time", str(config["segment_time"]),
        "-segment_format", "mpegts",
        "-reset_timestamps", "1",
        str(segment_dir / "seg%d.ts")
    ]

    print(f"{cam.name} recorder running")
    rec = subprocess.Popen(recorder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    if status_dir:
        pipeline_status.set_recording(status_dir, True)

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        print(f"[{cam.name}] Shutting down.")
        rec.terminate()
        if status_dir:
            pipeline_status.set_recording(status_dir, False)

        # give filesystem time to finish final segment
        time.sleep(0.5)

        # Concatenate segments if user chose to
        if config["save_mp4"]:
            result = recording_mode_finalizer(cam, timestamp, output_dir, segment_dir)
            if result:
                concat, concat_file = result

                print(f"{cam.name} concatenator running")
                result = subprocess.run(concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

                if result.returncode == 0:
                    # delete concat.txt
                    try:
                        concat_file.unlink()
                    except Exception as e:
                        print(f"{cam.name} failed to delete concat.txt: {e}")

                    # Delete segments folder after concatenation if user chose to
                    if config["delete_segments"]:
                        shutil.rmtree(segment_dir)

                        # try to clean up /segments/
                        try:
                            if not any(segment_dir.parent.iterdir()):
                                segment_dir.parent.rmdir()
                        except Exception:
                            pass

                else:
                    print(f"{cam.name} mp4 failed, keeping segments despite settings")
            else:
                print(f"Warning: concatenation failed")

        try:
            rec.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[{cam.name}] ffmpeg didn't exit, killing...")
            rec.kill()


def recording_mode_finalizer(cam, timestamp, output_dir, segment_dir):

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Store all segment files currently in the segment folder, sorted by their segment number
    segments = sorted(segment_dir.glob("seg*.ts"), key=lambda p: int(p.stem[3:]))  # strips "seg" prefix, parses remainder as int

    if not segments:
        print(f"Warning: no segments found for {cam.name}, skipping episode")
        return None
    
    # Build concat list out of present segments, used by FFmpeg to concatenate segments into one video
    concat_file = segment_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in segments:
            f.write(f"file '{seg.name}'\n")

    # Output episode named with camera id and detection timestamp
    output_file = final_dir / f"{cam.name}_{timestamp.strftime('%Y-%m-%d_%I.%M.%S%p')}.mp4"
    
    concatenator = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_file)
    ]

    return concatenator, concat_file
    


