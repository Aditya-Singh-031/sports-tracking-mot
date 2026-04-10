import json
import cv2
from pathlib import Path
from src.heatmap import generate_heatmap
from src.metrics import compute_metrics, plot_count_over_time, print_metrics
from src.utils import setup_logger

logger = setup_logger("Analytics")


def run_analytics(data_path: str, video_path: str, output_dir: str) -> dict:
    with open(data_path) as f:
        data = json.load(f)

    output_dir = Path(output_dir)
    tracker = data.get("tracker", "botsort")
    sport   = data.get("sport", "cricket")

    # --- Background frame from video ---
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    # 1. Heatmap
    positions = [tuple(p) for p in data.get("all_positions", [])]
    generate_heatmap(
        positions=positions,
        video_w=data["video_w"],
        video_h=data["video_h"],
        output_path=str(output_dir / f"heatmap_{tracker}.png"),
        sport=sport,
        background_frame=first_frame if ret else None,
    )

    # 2. Count-over-time chart
    plot_count_over_time(data, str(output_dir / f"count_over_time_{tracker}.png"))

    # 3. Metrics
    metrics = compute_metrics(data)
    metrics_path = str(output_dir / f"metrics_{tracker}.json")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print_metrics(metrics)
    logger.info(f"Metrics saved: {metrics_path}")

    return metrics