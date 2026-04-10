"""
run_full.py  —  Full pipeline: both trackers, all analytics.
Usage:  python run_full.py --sport cricket
"""
import argparse
import json
from pathlib import Path
from src.pipeline  import run_pipeline
from src.analytics import run_analytics
from src.utils     import setup_logger

logger = setup_logger("RunFull")

VIDEO_MAP = {
    "cricket":  "data/cricket/source_video.mp4",
    "football": "data/football/source_video.mp4",
}


def main(sport: str, model: str = "yolo11x.pt", conf: float = 0.40):
    video_path = VIDEO_MAP[sport]
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video missing: {video_path}")

    output_dir   = f"outputs/{sport}"
    all_metrics  = {}

    for tracker in ["botsort.yaml", "bytetrack.yaml"]:
        name = tracker.replace(".yaml", "")
        logger.info(f"\n{'='*60}\n  Running {name.upper()} on {sport.upper()}\n{'='*60}")

        data = run_pipeline(
            video_path=video_path, sport=sport,
            model=model, tracker_config=tracker,
            conf=conf, output_dir=output_dir,
        )

        metrics = run_analytics(
            data_path=f"{output_dir}/tracking_data_{name}.json",
            video_path=video_path,
            output_dir=output_dir,
        )
        all_metrics[name] = metrics

    # ── Comparison summary ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY — {sport.upper()}")
    print(f"{'='*60}")
    keys = ["total_unique_ids", "id_switches_approx",
            "avg_players_per_frame", "avg_processing_fps"]
    print(f"  {'Metric':<35} {'BoT-SORT':>12} {'ByteTrack':>12}")
    print(f"  {'-'*59}")
    for k in keys:
        bv = all_metrics.get("botsort",    {}).get(k, "-")
        yv = all_metrics.get("bytetrack",  {}).get(k, "-")
        print(f"  {k:<35} {str(bv):>12} {str(yv):>12}")
    print(f"{'='*60}\n")

    # Save comparison JSON
    comp_path = f"outputs/{sport}/comparison_summary.json"
    with open(comp_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Comparison saved: {comp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="cricket", choices=["cricket", "football"])
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--conf",  type=float, default=0.40)
    args = parser.parse_args()
    main(args.sport, args.model, args.conf)