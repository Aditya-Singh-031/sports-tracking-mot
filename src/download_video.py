# src/download_video.py
import argparse
import subprocess
from pathlib import Path

def download_video(url: str, output_dir: Path, filename: str = "source_video.mp4"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "-o", str(out_path),
        url,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Saved to:", out_path.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--sport", type=str, default="cricket", choices=["cricket", "football"])
    parser.add_argument("--filename", type=str, default="source_video.mp4")
    args = parser.parse_args()

    base = Path("data") / args.sport
    download_video(args.url, base, args.filename)