import subprocess
from pathlib import Path

CAMERA_FEED = "rtsp://admin:123456@10.0.0.11:554/"
OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_next_filename():
    """Return the next available training_recording_X.ts filename."""
    index = 0
    while True:
        filename = OUTPUT_DIR / f"training_recording_{index}.ts"
        if not filename.exists():
            return filename
        index += 1

def main():
    output_file = get_next_filename()

    print(output_file)

    print("Press ENTER to start recording...")
    input()

    ffmpeg_cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", CAMERA_FEED,
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(output_file)
    ]

    proc = subprocess.Popen(ffmpeg_cmd)
    print(f"Recording started → {output_file}")
    print("Press ENTER again to stop recording or Ctrl+C")

    try:
        input()  # wait for second Enter
        print("Stopping recording...")
    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Stopping recording...")

    # Stop FFmpeg cleanly
    proc.send_signal(subprocess.signal.SIGINT)
    proc.wait()
    print(f"Recording saved: {output_file}")

if __name__ == "__main__":
    main()