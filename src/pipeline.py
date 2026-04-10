import cv2
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

from src.tracker import Tracker
from src.annotator import Annotator
from src.speed import SpeedEstimator
from src.utils import setup_logger, inspect_video, create_video_writer, save_frame

logger = setup_logger("Pipeline")

# Frames at which to auto-save screenshots (as fraction of total)
SCREENSHOT_AT = [0.10, 0.25, 0.50, 0.75, 0.90]


def run_pipeline(
    video_path: str,
    sport: str = "cricket",
    model: str = "yolo11x.pt",
    tracker_config: str = "botsort.yaml",
    conf: float = 0.40,
    output_dir: str = None,
    frame_skip: int = 1,
) -> dict:

    video_path  = Path(video_path)
    output_dir  = Path(output_dir) if output_dir else Path("outputs") / sport
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screenshots").mkdir(exist_ok=True)

    info      = inspect_video(str(video_path))
    fps       = info["fps"]
    W, H      = info["width"], info["height"]
    total     = info["frames"]

    tracker_name = "botsort" if "botsort" in tracker_config else "bytetrack"
    out_video    = str(output_dir / f"annotated_{tracker_name}.mp4")

    tracker   = Tracker(model_name=model, tracker_config=tracker_config, conf=conf)
    annotator = Annotator(trail_length=45)
    speed_est = SpeedEstimator(fps=fps, sport=sport)
    writer    = create_video_writer(out_video, fps, W, H)

    screenshot_frames = {int(total * t) for t in SCREENSHOT_AT}

    # ---- Stats accumulators ----
    count_over_time = []
    seen_ids        = set()
    id_switches     = 0
    prev_max_id     = 0
    frame_times     = []

    cap = cv2.VideoCapture(str(video_path))
    frame_num = 0

    logger.info(f"Pipeline start | {sport} | {tracker_name} | {model} | conf={conf}")
    pbar = tqdm(total=total, desc=f"[{tracker_name}] Tracking", unit="frame", ncols=90)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        if frame_num % frame_skip == 0:
            results  = tracker.track(frame, persist=True)
            annotated = annotator.draw(frame, results, frame_num)

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids   = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                # ── Filter out tiny bboxes (crowd noise, distant background) ──
                MIN_AREA = 2000  # px² — roughly 45×45px minimum person size
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                valid = areas > MIN_AREA
                boxes, ids, confs = boxes[valid], ids[valid], confs[valid]

                count = len(ids)
                count_over_time.append((frame_num, count))

                for tid, box in zip(ids, boxes):
                    seen_ids.add(int(tid))
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    speed_est.update(int(tid), (cx, cy), frame_num)
                    if int(tid) > prev_max_id:
                        if prev_max_id > 0:
                            id_switches += 1
                        prev_max_id = int(tid)
            else:
                count_over_time.append((frame_num, 0))
        else:
            annotated = frame   # pass-through for skipped frames

        writer.write(annotated)

        # Save screenshots
        if frame_num in screenshot_frames:
            shot = str(output_dir / "screenshots" / f"frame_{frame_num:05d}_{tracker_name}.jpg")
            save_frame(annotated, shot)

        frame_times.append(time.perf_counter() - t0)
        frame_num += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    avg_fps = 1.0 / (sum(frame_times) / max(len(frame_times), 1))

    # ---- Speed summary ----
    speed_summary  = speed_est.get_summary()
    speed_per_id   = speed_est.get_per_id_stats()

    # ---- Save tracking data JSON ----
    data = {
        "sport":              sport,
        "tracker":            tracker_name,
        "model":              model,
        "total_frames":       frame_num,
        "total_unique_ids":   len(seen_ids),
        "id_switches_approx": id_switches,
        "avg_fps":            round(avg_fps, 2),
        "count_over_time":    count_over_time,
        "all_positions":      annotator.get_all_positions(),
        "speed_summary":      speed_summary,
        "speed_per_id":       speed_per_id,
        "video_w":            W,
        "video_h":            H,
    }

    data_path = str(output_dir / f"tracking_data_{tracker_name}.json")
    with open(data_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Output video : {out_video}")
    logger.info(f"Unique IDs   : {len(seen_ids)}")
    logger.info(f"Speed        : {avg_fps:.1f} FPS avg")
    if speed_summary:
        logger.info(f"Fastest player: ID {speed_summary.get('fastest_player_id')} @ {speed_summary.get('global_max_kmh')} km/h")

    return data


# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports MOT Pipeline")
    parser.add_argument("--video",    required=True,  help="Path to input video")
    parser.add_argument("--sport",    default="cricket", choices=["cricket", "football"])
    parser.add_argument("--model",    default="yolo11x.pt")
    parser.add_argument("--tracker",  default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--conf",     type=float, default=0.40)
    parser.add_argument("--output",   default=None,  help="Output directory")
    parser.add_argument("--skip",     type=int,   default=1, help="Process every N frames")
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