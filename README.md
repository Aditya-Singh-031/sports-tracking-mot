# 🏏⚽ Sports Multi-Object Tracking — Persistent ID Assignment

> **Assignment:** Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage  
> **Type:** AI / Computer Vision / Data Science  
> **Live Demo:** [🚀 Streamlit App](https://sports-tracking-mot.streamlit.app)  
> **Source Video:** [📹 IPL Cricket — YouTube](https://www.youtube.com/watch?v=REPLACE_WITH_YOUR_VIDEO_ID)

---

## 📊 Deliverables Status

| Deliverable | Status |
|---|---|
| GitHub repository (public) | ✅ Done |
| `README.md` | ✅ Done |
| Annotated output video | ✅ `outputs/cricket/annotated_botsort.mp4` |
| Original public video link | ✅ See Source Videos section |
| Short technical report | ✅ `docs/technical_report.md` |
| Sample screenshots | ✅ `outputs/cricket/screenshots/` |
| Demo video (3–5 min) | ✅ See Demo Video section |
| Live hosted URL | ✅ [sports-tracking-mot.streamlit.app](https://sports-tracking-mot.streamlit.app) |
| **BONUS** Trajectory trails | ✅ Overlaid per-ID in output video |
| **BONUS** Movement heatmaps | ✅ `outputs/cricket/heatmap_botsort.png` |
| **BONUS** Bird's-eye projection | ✅ `outputs/cricket/birdseye_botsort.mp4` |
| **BONUS** Object count over time | ✅ `outputs/cricket/count_over_time_botsort.png` |
| **BONUS** Speed estimation | ✅ Per-player speed in `tracking_data_botsort.json` |
| **BONUS** Evaluation metrics | ✅ MOTA, ID switches, track stats in `metrics_botsort.json` |
| **BONUS** Model comparison | ✅ BoT-SORT vs ByteTrack — `comparison_summary.json` |
| **BONUS** Deployment as demo app | ✅ [Streamlit live app](https://sports-tracking-mot.streamlit.app) |

---

## 🎬 Source Videos

| Sport | Source | Duration | Resolution |
|---|---|---|---|
| Cricket | [IPL Match — YouTube](https://www.youtube.com/watch?v=REPLACE_WITH_YOUR_VIDEO_ID) | ~90 sec | 1080p |

> All content used is publicly accessible. Downloaded for research/evaluation purposes only using `yt-dlp`.

---

## 🏗️ Architecture Overview

```
Input Video (1080p Cricket Broadcast)
         │
         ▼
FFmpeg re-encode (AV1 → H.264, required for OpenCV compatibility)
         │
         ▼
Frame Reader (OpenCV VideoCapture, every frame)
         │
         ▼
YOLOv11x Detector
  • Model: yolo11x.pt  (56.9M params, 54.7 mAP COCO)
  • Classes: person only (class 0)
  • Conf threshold: 0.35
  • Input size: 1280px
  • Device: CUDA (NVIDIA RTX 4060)
         │
         ▼
Tracker (BoT-SORT primary / ByteTrack comparison)
  • Kalman filter motion prediction
  • Camera Motion Compensation (CMC) via sparse optical flow  [BoT-SORT only]
  • Hungarian algorithm IoU + feature matching
  • Track buffer: 60 frames (2.4s memory)
  • Min track area: 2000 px² (suppresses noise)
         │
         ▼
Track History Store (deque per ID, max 45 frames)
         │
    ┌────┴────────────────────────────┐
    ▼                                 ▼
Annotation Layer                  Analytics Layer
  • Bounding boxes (per-ID color)    • Heatmap (position density)
  • ID labels + confidence           • Bird's-eye projection (homography)
  • Trajectory trails (45 frames)    • Speed estimation (px/frame → km/h)
  • Live player count overlay        • Count over time chart
                                     • Tracker comparison metrics
    └────┬────────────────────────────┘
         ▼
Output Video Writer (OpenCV + FFmpeg)
         │
         ▼
outputs/cricket/
  ├── annotated_botsort.mp4
  ├── annotated_bytetrack.mp4
  ├── heatmap_botsort.png
  ├── heatmap_bytetrack.png
  ├── count_over_time_botsort.png
  ├── count_over_time_bytetrack.png
  ├── metrics_botsort.json
  ├── metrics_bytetrack.json
  ├── tracking_data_botsort.json
  ├── comparison_summary.json
  └── screenshots/
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Object Detection | [Ultralytics YOLOv11x](https://docs.ultralytics.com/models/yolo11/) | `ultralytics>=8.3` |
| Primary Tracker | BoT-SORT (Camera Motion Compensation) | via `ultralytics` |
| Comparison Tracker | ByteTrack | via `ultralytics` |
| Video I/O | OpenCV + FFmpeg | `opencv-python>=4.8` |
| Analytics | NumPy, Matplotlib, Seaborn | latest |
| Demo App | Streamlit | `>=1.35` |
| Hosting | Streamlit Community Cloud | free tier |
| GPU | NVIDIA RTX 4060 CUDA | CUDA 12.x |

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aditya-Singh-031/sports-tracking-mot.git
cd sports-tracking-mot

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows Git Bash
# source venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### System requirements
- Python 3.10–3.12
- NVIDIA GPU with CUDA 11.8+ (CPU fallback works, but slower)
- FFmpeg installed and in PATH — [download here](https://ffmpeg.org/download.html)
- ~4 GB free disk space (model weights + videos)

---

## 🚀 How to Run

### Step 1 — Download source video

```bash
pip install yt-dlp
yt-dlp -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]" \
  "https://www.youtube.com/watch?v=REPLACE_WITH_YOUR_VIDEO_ID" \
  -o "data/cricket/source_video.%(ext)s"
```

### Step 2 — Re-encode if needed (AV1 → H.264)

```bash
ffmpeg -i data/cricket/source_video.webm -c:v libx264 -crf 18 -preset fast \
  -c:a aac data/cricket/source_video.mp4
```

### Step 3 — Run the full pipeline

```bash
# Run both trackers, generate all outputs and analytics
python run_full.py --sport cricket --model yolo11x.pt --conf 0.35

# Run a single tracker only
python run_full.py --sport cricket --tracker botsort
python run_full.py --sport cricket --tracker bytetrack

# Quick test on first 300 frames
python run_full.py --sport cricket --max_frames 300
```

### Step 4 — Launch Streamlit demo

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📁 Repository Structure

```
sports-tracking-mot/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 runtime.txt                  # Python 3.11 pin for Streamlit Cloud
├── 📄 run_full.py                  # Main entry point (run everything)
│
├── 🐍 src/
│   ├── detector.py                 # YOLOv11x detection wrapper
│   ├── tracker.py                  # BoT-SORT / ByteTrack integration
│   ├── pipeline.py                 # Orchestration + video writing
│   ├── annotator.py                # Bounding boxes, ID labels, trails
│   ├── heatmap.py                  # Movement heatmap generation
│   ├── birdseye.py                 # Top-view homography projection
│   ├── speed_estimator.py          # Speed estimation (px/frame → km/h)
│   ├── metrics.py                  # MOTA, ID switch counter, track stats
│   └── utils.py                    # Color palette, logging, file helpers
│
├── 🌐 app.py                       # Streamlit demo app
│
├── 📊 outputs/
│   └── cricket/
│       ├── annotated_botsort.mp4
│       ├── annotated_bytetrack.mp4
│       ├── heatmap_botsort.png
│       ├── heatmap_bytetrack.png
│       ├── count_over_time_botsort.png
│       ├── metrics_botsort.json
│       ├── metrics_bytetrack.json
│       ├── tracking_data_botsort.json
│       ├── comparison_summary.json
│       └── screenshots/
│
├── 📝 configs/
│   └── config.yaml                 # All hyperparameters and paths
│
├── 📓 notebooks/
│   └── model_comparison.ipynb      # BoT-SORT vs ByteTrack analysis
│
└── 📚 docs/
    ├── technical_report.md         # 1–2 page technical report (PS requirement)
    └── architecture.md             # System architecture deep-dive
```

---

## 📋 Model & Tracker Choices

### Detector — YOLOv11x

YOLOv11x (the largest variant, 56.9M parameters) was chosen over smaller variants because:

- Broadcast cricket footage contains **distant fielders** that occupy as few as 25×60 pixels — only the X model reliably detects them
- Confidence threshold lowered to **0.35** (vs default 0.5) to recover small/partially visible players at the cost of slightly more false positives, which are filtered by the minimum area threshold
- Smaller models (nano, small) were tested first — nano produced 93+ unique IDs on a 90s clip vs ~30 for the X model, indicating severe ghost/duplicate detections

### Primary Tracker — BoT-SORT

BoT-SORT was chosen as the primary tracker because broadcast cricket involves constant **camera pan, zoom, and hard cuts**. Its Camera Motion Compensation (CMC) module uses sparse optical flow to compute the camera's apparent motion and subtracts it before IoU matching — preventing the tracker from interpreting camera movement as player movement.

### Comparison Tracker — ByteTrack

ByteTrack was retained for comparison (required by the assignment). It runs ~2× faster than BoT-SORT and uses a clever two-stage matching that recovers detections the first-stage IoU matching missed. The trade-off is no CMC, making it more prone to ID switches on panned sequences.

---

## 🔑 ID Consistency Strategy

1. **Kalman filter prediction** — Each track maintains a Kalman filter that predicts the next bounding box position even when the detector misses a frame. This means brief detection failures (1–3 frames) do not immediately kill a track.

2. **Camera Motion Compensation** (BoT-SORT) — Sparse optical flow computes the inter-frame homography. Track positions are corrected by this transform before IoU matching, ensuring camera pan/zoom does not cause false mismatches.

3. **Two-stage Hungarian matching** — High-confidence detections are matched first (IoU threshold 0.5). Unmatched detections and tracks are then re-matched at a lower threshold (0.3) to recover occluded and partially visible players.

4. **60-frame track buffer** — Tracks are kept alive for 2.4 seconds (60 frames at 25 FPS) after their last detection. This survives slow-motion replays and brief occlusions.

5. **Minimum area filter (2000 px²)** — Suppresses crowd pixels, advertising boards, and JPEG compression artifacts that trigger false person detections.

6. **Per-ID color assignment** — IDs are assigned deterministic colors using a seeded random function, making ID switches visually obvious during review.

---

## ⚠️ Challenges & Limitations

| Challenge | Root Cause | Resolution |
|---|---|---|
| AV1 codec crash (OpenCV) | YouTube now serves AV1 by default; OpenCV does not support it | Re-encode to H.264 with FFmpeg before processing |
| High unique ID count in broadcast | Hard camera cuts reset all tracks simultaneously; players re-enter as new IDs | Known broadcast limitation — documented; shot boundary detection is the full fix |
| Identical white uniforms | Both teams in similar colors; ReID features are indistinguishable | Limitation noted; jersey number OCR is the proper solution |
| Slow-motion replays | Frame rate appears to drop; same player occupies different positions across a slow-mo sequence | Replays inflate ID count — future work: detect and skip replay segments |
| Distant fielders at boundary | Very small bounding boxes, low confidence | Mitigated with conf=0.35; some boundary fielders still missed |

---

## 🔮 Future Improvements

1. **Shot boundary detection** — Detect hard camera cuts and reset the tracker at each cut, preventing the ID explosion that occurs when all 22 players simultaneously appear as "new"
2. **Jersey number OCR** — Read jersey numbers using a lightweight OCR model (TrOCR / EasyOCR) to provide ground-truth IDs that survive camera cuts
3. **OSNet ReID model** — Replace IoU-only matching with deep appearance features using OSNet, enabling re-identification of players returning after a replay
4. **Adaptive confidence thresholds** — Dynamically lower the confidence floor when player count drops suddenly (replay detection heuristic)
5. **Bird's-eye homography calibration** — Per-video calibration for more accurate real-world speed and distance estimates

---

## 🎥 Demo Video

> A 3–5 minute walkthrough covering the pipeline, output video, Streamlit app, and technical choices.  
> 📹 [Link to be added after recording]

---

## 📄 Documentation

- [`docs/technical_report.md`](docs/technical_report.md) — 1–2 page technical report (PS requirement)
- [`docs/architecture.md`](docs/architecture.md) — Detailed system architecture

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
