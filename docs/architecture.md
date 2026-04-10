# System Architecture — Sports Multi-Object Tracking Pipeline

## Overview

This document describes the internal design of each module in the pipeline, the data flow between components, and the reasoning behind every architectural decision.

---

## 1. High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         INPUT STAGE                                      │
│                                                                          │
│  YouTube URL → yt-dlp → raw.webm (AV1)                                  │
│                      → FFmpeg → source_video.mp4 (H.264, 1080p)         │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        DETECTION STAGE                                   │
│                                                                          │
│  OpenCV VideoCapture                                                     │
│       │ frame (BGR, 1920×1080)                                           │
│       ▼                                                                  │
│  YOLOv11x.predict()                                                      │
│       │ conf=0.35, classes=[0], imgsz=1280, device=cuda                  │
│       ▼                                                                  │
│  DetectionResult: [x1,y1,x2,y2, conf, class_id]  (N × 6)               │
│       │ filter: area > 2000 px²                                          │
│       ▼                                                                  │
│  Filtered detections → Tracker                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         TRACKING STAGE                                   │
│                                                                          │
│  BoT-SORT (primary)                                                      │
│  ├── CMC module: cv2.calcOpticalFlowPyrLK on sparse feature points       │
│  │       → inter-frame homography H (2×3 affine)                        │
│  │       → correct all track positions by H before matching             │
│  ├── Kalman filter: predict next bbox for each active track              │
│  ├── Stage 1 matching: IoU ≥ 0.5, Hungarian algorithm                   │
│  ├── Stage 2 matching: IoU ≥ 0.3 on unmatched tracks + detections       │
│  └── Track lifecycle:                                                    │
│       NEW → ACTIVE (≥3 confirmed frames) → LOST → DELETED (>60 frames)  │
│                                                                          │
│  ByteTrack (comparison)                                                  │
│  ├── High-conf detections (≥0.5): Stage 1 IoU matching                  │
│  ├── Low-conf detections (0.1–0.5): Stage 2 matching on lost tracks     │
│  └── No CMC — pure motion-based                                         │
│                                                                          │
│  Output per frame: List[Track(id, bbox, conf)]                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        ▼                     ▼
         ┌──────────────────────┐   ┌──────────────────────────┐
         │   ANNOTATION LAYER   │   │     ANALYTICS LAYER       │
         │                      │   │                           │
         │ annotator.py         │   │ heatmap.py                │
         │  • colored bbox      │   │  • accumulate center XY   │
         │  • ID label          │   │  • gaussian blur          │
         │  • confidence %      │   │  • log-scale colormap     │
         │  • trajectory trail  │   │                           │
         │    (deque, 45 frames)│   │ speed_estimator.py        │
         │  • frame counter     │   │  • Δ(cx,cy) per frame     │
         │  • player count HUD  │   │  • smooth with EMA        │
         └──────────┬───────────┘   │  • px/frame → km/h        │
                    │               │    (calibration: 8 px/m)  │
                    ▼               │                           │
         OpenCV VideoWriter         │ birdseye.py               │
         (annotated_output.mp4)     │  • 4-point homography     │
                                    │  • map bbox center to     │
                                    │    top-down court plane   │
                                    │                           │
                                    │ metrics.py                │
                                    │  • unique ID count        │
                                    │  • ID switch detection    │
                                    │  • avg/max players/frame  │
                                    │  • frames with 0 detect.  │
                                    └──────────────────────────┘
                                               │
                                               ▼
                                    outputs/cricket/*.json
                                    outputs/cricket/*.png
                                    outputs/cricket/*.mp4
```

---

## 2. Module Responsibilities

### `src/detector.py` — Detection Wrapper

Wraps `ultralytics.YOLO` to provide a clean interface for the pipeline. Handles:
- Model loading and CUDA device placement
- Single-pass inference with configurable conf/imgsz
- Filtering to person class only
- Minimum bounding-box area gate (default: 2000 px²)

**Key design choice:** Area filtering happens here (not in the tracker) so the tracker never sees noise detections. This reduces unnecessary track creation and ID fragmentation.

### `src/tracker.py` — Tracker Integration

Wraps Ultralytics' built-in BoT-SORT and ByteTrack. Exposes a unified `.update(detections, frame)` API so the pipeline can switch trackers with a single config flag.

**Key design choice:** Unified interface means the entire pipeline — annotation, analytics, output — is tracker-agnostic. Switching from BoT-SORT to ByteTrack changes exactly one line in `config.yaml`.

### `src/annotator.py` — Visual Overlay

Draws bounding boxes, ID labels, confidence scores, and trajectory trails on each frame. Each unique ID receives a deterministic color (seeded RNG on ID value) that remains constant for the entire video.

**Trajectory trail:** A `collections.deque(maxlen=45)` per track ID stores the last 45 centroid positions. These are drawn as a tapered polyline — thick near current position, thin at oldest point — giving an intuitive motion direction indicator.

### `src/heatmap.py` — Position Density Map

Accumulates all bounding box centroids into a 2D histogram (same resolution as input video). A large Gaussian blur (sigma=25) smooths the histogram into a density map. The result is rendered with `cv2.COLORMAP_JET` (blue=low density, red=high density) and overlaid at 60% opacity on a reference frame.

### `src/birdseye.py` — Top-View Projection

Applies a perspective homography to map the playing field from camera view to a standardized top-down coordinate system. The four source points are manually calibrated to the cricket pitch corners visible in the footage. This gives spatially accurate player positions independent of camera angle.

**Limitation:** The homography is static. When the camera pans far enough that the calibration points leave frame, the projection degrades. A dynamic multi-homography approach (per camera angle segment) would fully solve this.

### `src/speed_estimator.py` — Speed Estimation

Computes per-frame displacement of each track's centroid in pixels, applies an exponential moving average (α=0.3) to smooth noise, then converts to km/h using a calibration factor of ~8 px/m derived from the known cricket pitch length (20.12m) and its pixel span in broadcast footage.

**Accuracy note:** Speed values are approximate (±20%) due to: (a) static calibration on a dynamic camera, (b) 2D projection losing depth, (c) EMA lag on rapid accelerations.

### `src/metrics.py` — Evaluation Metrics

Computes:
- `total_unique_ids`: distinct track IDs seen across the video
- `id_switches_approx`: frames where a track's ID changes by comparing consecutive frames' ID sets
- `avg_players_per_frame`, `max_players_per_frame`
- `frames_with_no_detection`: proxy for replay/scoreboard segments
- `avg_processing_fps`: wall-clock throughput

**Note:** True MOTA/MOTP require ground-truth annotations, which are not available for broadcast footage. The metrics computed here are heuristic approximations useful for tracker comparison but are not formal MOTA scores.

---

## 3. Camera Challenges in Broadcast Sports

Broadcast cricket is one of the hardest multi-object tracking scenarios because:

| Challenge | Frequency | Tracker Impact |
|---|---|---|
| Hard camera cuts | Every ~3–8s | All active tracks lost simultaneously |
| Pan/zoom | Continuous | Without CMC: entire field appears to shift, causing mass mismatches |
| Slow-motion replay | Every wicket/boundary | 4× frame rate reduction, players at unnatural positions |
| Identical uniforms | Always (team kit) | ReID features indistinguishable within a team |
| Partial occlusion | Frequent | Players cluster at fielding positions |
| Player out of frame | Frequent | Boundary fielders exit and re-enter frame |

BoT-SORT's CMC solves the pan/zoom problem. Hard camera cuts remain the largest source of ID fragmentation and are a known open problem in broadcast sports tracking — the correct solution is shot boundary detection followed by tracker reset.

---

## 4. Configuration Reference (`configs/config.yaml`)

```yaml
detection:
  model: yolo11x.pt
  conf_threshold: 0.35
  imgsz: 1280
  device: cuda
  classes: [0]              # person only
  min_area_px2: 2000

tracking:
  primary: botsort
  comparison: bytetrack
  track_buffer: 60          # frames to keep lost tracks alive
  match_thresh: 0.5         # Stage 1 IoU threshold
  second_match_thresh: 0.3  # Stage 2 IoU threshold
  min_box_area: 2000

annotation:
  trail_length: 45          # frames of trajectory history
  box_thickness: 2
  font_scale: 0.7
  show_confidence: true

analytics:
  heatmap_blur_sigma: 25
  speed_ema_alpha: 0.3
  speed_calibration_px_per_m: 8.0

output:
  fps: 25
  codec: mp4v
  quality_crf: 18
```

---

## 5. Output Files Reference

| File | Description | Used by |
|---|---|---|
| `annotated_botsort.mp4` | Full annotated output video | Submission, Streamlit app |
| `heatmap_botsort.png` | Player position density heatmap | Streamlit app, screenshots |
| `count_over_time_botsort.png` | Player count per frame chart | Streamlit app |
| `metrics_botsort.json` | Quantitative tracking metrics | Streamlit app overview |
| `tracking_data_botsort.json` | Per-frame tracks + speed data | Streamlit speed tab |
| `comparison_summary.json` | BoT-SORT vs ByteTrack side-by-side | Streamlit comparison tab |
| `screenshots/*.jpg` | Sample frames at regular intervals | README, submission |
