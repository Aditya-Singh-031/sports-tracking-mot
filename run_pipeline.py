# run_pipeline.py  (root level)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import run_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports MOT Pipeline")
    parser.add_argument("--video",   required=True)
    parser.add_argument("--sport",   default="cricket", choices=["cricket", "football"])
    parser.add_argument("--model",   default="yolo11x.pt")
    parser.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--conf",    type=float, default=0.40)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--skip",    type=int, default=1)
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        sport=args.sport,
        model=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        output_dir=args.output,
        frame_skip=args.skip,
    )