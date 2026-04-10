# 🏏⚽ Sports Multi-Object Tracking with Persistent IDs

> **Assignment:** Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage  
> **Type:** AI / Computer Vision / Data Science  
> **Live Demo:** [🔗 Streamlit App](https://share.streamlit.io/user/aditya-singh-031) ← update after deploy

---

## 📊 Progress Tracker

- [x] Step 1 — Plan + Repository Setup
- [ ] Step 2 — Environment Setup (Python + CUDA)
- [ ] Step 3 — Video Acquisition (Cricket + Football)
- [ ] Step 4 — Detection Pipeline (YOLO11)
- [ ] Step 5 — Persistent ID Tracking (BoT-SORT)
- [ ] Step 6 — Annotated Output Video
- [ ] Step 7 — Bonus Analytics
- [ ] Step 8 — Tracker Comparison (BoT-SORT vs ByteTrack)
- [ ] Step 9 — Streamlit Demo App
- [ ] Step 10 — Live Hosting
- [ ] Step 11 — Technical Report
- [ ] Step 12 — Demo Video
- [ ] Step 13 — Final Cleanup + Submission

---

## 🎯 Problem Statement Mapping

| PS Requirement | Implementation | Status |
|---|---|---|
| Publicly available sports/event video | IPL cricket + Premier League football from YouTube | ✅ Planned |
| Detect all relevant subjects | YOLO11-X person detector | ✅ Planned |
| Unique and persistent IDs | BoT-SORT tracker (motion + appearance) | ✅ Planned |
| Stable IDs under rapid motion | BoT-SORT camera motion compensation | ✅ Planned |
| Stable IDs under occlusion | Track buffering + re-identification | ✅ Planned |
| Stable IDs under similar appearance | Deep appearance features via ReID | ✅ Planned |
| Annotated output video | Bounding boxes + ID labels + trajectory trails | ✅ Planned |
| Clean modular code | Structured `src/` package | ✅ Planned |
| README.md | This file | ✅ Planned |
| Short technical report | `docs/technical_report.md` + PDF | ✅ Planned |
| Screenshots | `assets/demo_screenshots/` | ✅ Planned |
| Demo video (3–5 min) | Recorded walkthrough | ✅ Planned |
| Live hosted URL | Streamlit Community Cloud | ✅ Planned |
| **BONUS:** Trajectory visualization | Per-ID trail overlays | ✅ Planned |
| **BONUS:** Movement heatmaps | Position density maps | ✅ Planned |
| **BONUS:** Bird's-eye projection | Homography top-down view | ✅ Planned |
| **BONUS:** Object count over time | Time-series chart | ✅ Planned |
| **BONUS:** Speed estimation | Pixel displacement → real-world speed | ✅ Planned |
| **BONUS:** Evaluation metrics | MOTA, ID switches, track stats | ✅ Planned |
| **BONUS:** Model comparison | BoT-SORT vs ByteTrack | ✅ Planned |
| **BONUS:** Deployment as app | Streamlit demo with preloaded results | ✅ Planned |

---

## 🏗️ Architecture Overview
Input Video (Cricket / Football)
↓
Frame Extraction (OpenCV)
↓
YOLO11-X Detection (persons only, conf ≥ 0.4)
↓
BoT-SORT Tracker (persistent IDs across frames)
↓
Track History Store (deque per ID)
↓
Annotation Layer:
- Bounding boxes
- ID labels
- Trajectory trails
- Live player count
↓
Analytics Layer:
- Heatmap (position density)
- Bird's-eye view (homography)
- Speed estimation
- Count over time
- Tracker comparison
↓
Outputs:
- annotated_output.mp4
- heatmap.png / birdseye.mp4
- count_over_time.png
- technical_report.pdf
- Streamlit demo app



---

## 🎬 Source Videos

| Sport | Source | Duration | Resolution |
|---|---|---|---|
| Cricket | TBD — IPL/International match clip (YouTube) | ~90 sec | 1080p |
| Football | TBD — Premier League clip (YouTube) | ~90 sec | 1080p |

> Links will be updated once videos are selected in Step 3.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [Ultralytics YOLO11-X](https://docs.ultralytics.com/models/yolo11/) |
| Primary Tracker | BoT-SORT |
| Comparison Tracker | ByteTrack |
| Video Processing | OpenCV, FFmpeg |
| Analytics | NumPy, Matplotlib, Seaborn |
| Metrics | py-motmetrics |
| Demo App | Streamlit |
| Hosting | Streamlit Community Cloud |
| GPU | NVIDIA RTX 4060 + CUDA |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Aditya-Singh-031/sports-tracking-mot.git
cd sports-tracking-mot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Run full pipeline on cricket video
python src/pipeline.py --sport cricket --video data/cricket/source_video.mp4

# Run with ByteTrack instead of BoT-SORT
python src/pipeline.py --sport cricket --video data/cricket/source_video.mp4 --tracker bytetrack

# Launch Streamlit demo app
streamlit run app.py
```

---

## 📁 Repository Structure
sports-tracking-mot/
├── src/ # Core pipeline modules
├── configs/ # Tracker and project configs
├── data/ # Input videos (gitignored, download separately)
├── models/ # YOLO weights (gitignored, auto-downloaded)
├── outputs/ # Generated outputs per sport
├── notebooks/ # Model comparison notebook
├── docs/ # Technical report + architecture notes
└── assets/ # Screenshots for README


---

## 📋 Model & Tracker Choices

> To be filled in Step 5 after implementation.

---

## 🔑 ID Consistency Strategy

> To be filled in Step 5 after implementation.

---

## ⚠️ Challenges & Limitations

> To be filled after running the pipeline.

---

## 🔮 Future Improvements

> To be filled in final step.

---

## 📄 License

MIT