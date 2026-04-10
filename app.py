import streamlit as st
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Player Tracker",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.main-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 0.95rem; opacity: 0.8; margin: 0.4rem 0 0; }
.section-header {
    font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
    border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1rem;
}
.info-box { background:#e8f4fd; border-left:4px solid #2196f3; padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; font-size:0.9rem; }
.warn-box { background:#fff8e1; border-left:4px solid #ffc107; padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; font-size:0.9rem; }
div[data-testid="stMetric"] { background:#f8f9fa; border-radius:8px; padding:0.75rem 1rem; border:1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_color(tid):
    np.random.seed(int(tid) * 7 + 31)
    r, g, b = np.random.randint(80, 255, 3)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_outputs(sport, tracker):
    base = Path("outputs") / sport
    shots_dir = base / "screenshots"
    shots = sorted(shots_dir.glob(f"*_{tracker}.jpg")) if shots_dir.exists() else []
    return {
        "metrics":     load_json(base / f"metrics_{tracker}.json"),
        "data":        load_json(base / f"tracking_data_{tracker}.json"),
        "comparison":  load_json(base / "comparison_summary.json"),
        "heatmap":     base / f"heatmap_{tracker}.png",
        "count_chart": base / f"count_over_time_{tracker}.png",
        "video":       base / f"annotated_{tracker}.mp4",
        "screenshots": shots,
    }


def frames_from_video(video_path, n=6):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        fnum = int(total * (i + 0.5) / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def plot_speed_chart(data):
    speed_per_id = data.get("speed_per_id", {})
    if not speed_per_id:
        return None
    ids  = list(speed_per_id.keys())[:30]   # cap at 30 for readability
    maxs = [speed_per_id[i]["max_kmh"] for i in ids]
    avgs = [speed_per_id[i]["avg_kmh"] for i in ids]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    x = np.arange(len(ids))
    w = 0.38
    ax.bar(x - w/2, maxs, w, label="Max speed",  color="#1565C0", alpha=0.85)
    ax.bar(x + w/2, avgs, w, label="Avg speed",  color="#42A5F5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ID {i}" for i in ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Per-Player Speed Estimates (Top 30 IDs)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    sport   = st.selectbox("Sport", ["cricket", "football"], index=0)
    tracker = st.selectbox(
        "Tracker", ["botsort", "bytetrack"], index=0,
        format_func=lambda x: "BoT-SORT (Recommended)" if x == "botsort" else "ByteTrack (Fast)"
    )
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
**Model:** YOLOv11x  
**Classes:** Person only  
**Conf threshold:** 0.35  
**Input:** 1920×1080 @ 25 FPS  
**Sport:** IPL Cricket Broadcast  
    """)
    st.divider()
    st.markdown("### 🔗 Resources")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Aditya-Singh-031/sports-tracking-mot)")
    st.markdown("[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11x-blue)](https://docs.ultralytics.com)")
    st.markdown("[![BoT-SORT](https://img.shields.io/badge/Tracker-BoT--SORT-green)](https://github.com/NirAharon/BoT-SORT)")


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
out      = load_outputs(sport, tracker)
metrics  = out["metrics"]
data     = out["data"]
comp     = out["comparison"]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏏 Sports Player Tracker — Multi-Object Detection &amp; Persistent ID</h1>
  <p>YOLOv11x &nbsp;·&nbsp; BoT-SORT + ByteTrack &nbsp;·&nbsp; Cricket Broadcast &nbsp;·&nbsp; 1920×1080 @ 25 FPS &nbsp;·&nbsp; RTX 4060 CUDA</p>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "🎬 Output Video", "🌡️ Heatmap & Charts",
                "⚡ Speed Analysis", "📈 Model Comparison", "📄 Technical Report"])

tab_overview, tab_video, tab_heatmap, tab_speed, tab_compare, tab_report = tabs


# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab_overview:
    if not metrics:
        st.warning("No results found for this tracker/sport combination.")
        st.code("python run_full.py --sport cricket --model yolo11x.pt --conf 0.35")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🎯 Unique IDs",        metrics.get("total_unique_ids", "—"))
        c2.metric("📹 Total Frames",       metrics.get("total_frames", "—"))
        c3.metric("👥 Avg Players/Frame",  metrics.get("avg_players_per_frame", "—"))
        c4.metric("🔝 Max in Frame",       metrics.get("max_players_per_frame", "—"))
        c5.metric("⚡ Avg FPS",            metrics.get("avg_processing_fps", "—"))

        st.markdown("")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p class="section-header">📋 All Metrics</p>', unsafe_allow_html=True)
            for k, v in metrics.items():
                if k not in ("tracker", "sport"):
                    st.markdown(f"- **{k.replace('_', ' ').title()}:** `{v}`")

        with col_b:
            st.markdown('<p class="section-header">⚡ Speed Summary</p>', unsafe_allow_html=True)
            if data:
                spd = data.get("speed_summary", {})
                if spd:
                    st.metric("Max Speed Detected", f"{spd.get('global_max_kmh', '—')} km/h")
                    st.metric("Global Average",      f"{spd.get('global_avg_kmh', '—')} km/h")
                    st.metric("Fastest Player ID",   f"ID {spd.get('fastest_player_id', '—')}")
                    st.caption(spd.get("note", ""))
                else:
                    st.info("Speed data not available.")

            st.markdown("")
            st.markdown('<p class="section-header">📸 Sample Screenshots</p>', unsafe_allow_html=True)
            shots = out["screenshots"]
            if shots:
                cols = st.columns(3)
                for col, shot in zip(cols, shots[:3]):
                    col.image(str(shot), use_column_width=True)
            else:
                st.info("No screenshots found. Run pipeline first.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — OUTPUT VIDEO
# ══════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown('<p class="section-header">🎬 Annotated Output Video</p>', unsafe_allow_html=True)
    video_path = out["video"]
    if Path(video_path).exists():
        st.markdown('<div class="info-box">📌 Each color = unique player ID. Boxes show detection. Trails show movement history (last 45 frames).</div>', unsafe_allow_html=True)
        with open(str(video_path), "rb") as vf:
            st.video(vf.read())

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.success("✅ Colored box = unique player ID")
        c2.info("📍 Label = ID number")
        c3.warning("〰️ Trail = movement history")

        st.markdown("")
        st.markdown('<p class="section-header">🖼️ Frame Samples</p>', unsafe_allow_html=True)
        with st.spinner("Extracting frames..."):
            frames = frames_from_video(str(video_path), n=6)
        if frames:
            cols = st.columns(3)
            for i, frame in enumerate(frames):
                cols[i % 3].image(frame, use_column_width=True, caption=f"Frame sample {i+1}")
    else:
        st.warning(f"Video not found: `{video_path}`")
        st.code("python run_full.py --sport cricket --model yolo11x.pt --conf 0.35")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — HEATMAP & CHARTS
# ══════════════════════════════════════════════════════════════════
with tab_heatmap:
    cl, cr = st.columns(2)
    with cl:
        st.markdown('<p class="section-header">🌡️ Movement Heatmap</p>', unsafe_allow_html=True)
        if Path(out["heatmap"]).exists():
            st.image(str(out["heatmap"]), use_column_width=True)
            st.caption("Cumulative player position density. Brighter = more time spent in that zone.")
        else:
            st.info("Heatmap not found.")

    with cr:
        st.markdown('<p class="section-header">📈 Player Count Over Time</p>', unsafe_allow_html=True)
        if Path(out["count_chart"]).exists():
            st.image(str(out["count_chart"]), use_column_width=True)
            st.caption("Players tracked per frame. Zero-count frames = scoreboards/replays/transitions.")
        else:
            st.info("Count chart not found.")

    if data:
        cot = data.get("count_over_time", [])
        if cot:
            st.markdown("")
            st.markdown('<p class="section-header">📊 Interactive Count Timeline</p>', unsafe_allow_html=True)
            fps = data.get("fps", 25.0)
            df_cot = pd.DataFrame({
                "Time (s)":        [f / fps for f, _ in cot],
                "Players Tracked": [c for _, c in cot],
            }).set_index("Time (s)")
            st.area_chart(df_cot, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — SPEED ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab_speed:
    st.markdown('<p class="section-header">⚡ Speed Estimation</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Speed is estimated from pixel displacement using a calibration of ~8 px/m for cricket broadcast footage. Values are approximate.</div>', unsafe_allow_html=True)

    if data:
        spd = data.get("speed_summary", {})
        speed_per_id = data.get("speed_per_id", {})

        if spd:
            c1, c2, c3 = st.columns(3)
            c1.metric("🏃 Max Speed",      f"{spd.get('global_max_kmh', 0)} km/h")
            c2.metric("📊 Global Average", f"{spd.get('global_avg_kmh', 0)} km/h")
            c3.metric("🥇 Fastest Player", f"ID {spd.get('fastest_player_id', '?')}")

        if speed_per_id:
            st.markdown("")
            fig = plot_speed_chart(data)
            if fig:
                st.pyplot(fig, use_container_width=True)

            st.markdown("")
            st.markdown('<p class="section-header">📋 Per-Player Speed Table</p>', unsafe_allow_html=True)
            df_spd = pd.DataFrame([
                {"Player ID": int(tid),
                 "Max Speed (km/h)": v["max_kmh"],
                 "Avg Speed (km/h)": v["avg_kmh"]}
                for tid, v in speed_per_id.items()
            ]).sort_values("Max Speed (km/h)", ascending=False).reset_index(drop=True)
            st.dataframe(df_spd, use_container_width=True, height=280)
    else:
        st.warning("No tracking data found.")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<p class="section-header">📈 BoT-SORT vs ByteTrack — Direct Comparison</p>', unsafe_allow_html=True)

    if comp:
        bs = comp.get("botsort", {})
        bt = comp.get("bytetrack", {})

        rows = [
            ("Total Unique IDs",           bs.get("total_unique_ids"),           bt.get("total_unique_ids")),
            ("ID Switches (approx.)",      bs.get("id_switches_approx"),         bt.get("id_switches_approx")),
            ("Avg Players / Frame",        bs.get("avg_players_per_frame"),      bt.get("avg_players_per_frame")),
            ("Max Players in Frame",       bs.get("max_players_per_frame"),      bt.get("max_players_per_frame")),
            ("Frames with 0 Detections",   bs.get("frames_with_no_detection"),   bt.get("frames_with_no_detection")),
            ("Avg Processing FPS",         bs.get("avg_processing_fps"),         bt.get("avg_processing_fps")),
        ]
        df_comp = pd.DataFrame(rows, columns=["Metric", "BoT-SORT", "ByteTrack"])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ BoT-SORT — Recommended")
            st.markdown("""
- Camera Motion Compensation via sparse optical flow
- Fewer ID switches on broadcast footage with pan/zoom
- Kalman filter survives brief occlusions
- **Trade-off:** ~16 FPS (slower)
""")
        with c2:
            st.markdown("#### ⚡ ByteTrack — Fast")
            st.markdown("""
- ~2× faster than BoT-SORT (30 FPS)
- Two-stage association recovers low-confidence detections
- Better suited for real-time deployment
- **Trade-off:** No camera motion compensation
""")

        st.markdown("")
        st.markdown('<p class="section-header">🌡️ Side-by-Side Heatmaps</p>', unsafe_allow_html=True)
        h1 = Path(f"outputs/{sport}/heatmap_botsort.png")
        h2 = Path(f"outputs/{sport}/heatmap_bytetrack.png")
        c1, c2 = st.columns(2)
        if h1.exists(): c1.image(str(h1), caption="BoT-SORT heatmap", use_column_width=True)
        if h2.exists(): c2.image(str(h2), caption="ByteTrack heatmap", use_column_width=True)

        st.markdown("")
        st.markdown('<p class="section-header">📈 Side-by-Side Count Charts</p>', unsafe_allow_html=True)
        cc1 = Path(f"outputs/{sport}/count_over_time_botsort.png")
        cc2 = Path(f"outputs/{sport}/count_over_time_bytetrack.png")
        c1, c2 = st.columns(2)
        if cc1.exists(): c1.image(str(cc1), caption="BoT-SORT count over time", use_column_width=True)
        if cc2.exists(): c2.image(str(cc2), caption="ByteTrack count over time", use_column_width=True)
    else:
        st.info("Run both trackers first: `python run_full.py --sport cricket`")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — TECHNICAL REPORT
# ══════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown('<p class="section-header">📄 Technical Report</p>', unsafe_allow_html=True)

    st.markdown("""
### 🔍 Model / Detector — YOLOv11x
The largest YOLOv11 variant (56.9M parameters, 54.7 mAP COCO) was selected for maximum accuracy on
1080p broadcast footage where distant fielders occupy as few as 25×60 pixels. Smaller variants (nano, small)
were tested first but produced unacceptable ghost detection rates.

### 🎯 Tracking Algorithm — BoT-SORT + ByteTrack
**BoT-SORT** combines Kalman filter motion prediction, Hungarian algorithm assignment, and sparse optical
flow camera motion compensation (CMC). CMC is critical for broadcast cricket which involves constant pan and zoom.

**ByteTrack** uses every detection (including low-confidence) in a two-stage matching process. Faster than
BoT-SORT but lacks CMC, leading to more ID switches on panned sequences.

### ✅ Why This Combination
- YOLOv11x over smaller variants: nano model produced 93 IDs in test; X model is substantially cleaner
- BoT-SORT as primary: CMC correctly separates camera displacement from player motion before IoU matching
- ByteTrack retained as comparison for the assignment's model comparison requirement

### 🔒 ID Consistency Strategy
1. **Kalman filter prediction** — maintains track position during missed frames
2. **Camera motion compensation** (BoT-SORT only) — optical flow corrects for pan/zoom
3. **Two-stage Hungarian matching** — recovers occluded and low-confidence detections
4. **60-frame track buffer** — 2.4 seconds of memory survives camera cuts
5. **Min area filter 2000 px²** — suppresses crowd noise and compression artifacts

### ⚠️ Challenges & Failure Cases
| Challenge | Impact | Resolution |
|-----------|--------|------------|
| AV1 codec (OpenCV incompatible) | Pipeline blocked | ✅ FFmpeg H.264 re-encode |
| Hard camera cuts in broadcast | Major ID explosion | ⚠️ Inherent to broadcast — documented |
| Identical white uniforms | ReID ineffective | ⚠️ Known limitation |
| Slow-motion replays | Duplicate IDs | ⚠️ Shot boundary detection needed |
| Distant fielders (small bbox) | Low confidence | ✅ conf=0.35 + min area filter |

### 🚀 Possible Improvements
1. **Jersey number OCR** for cross-cut re-identification
2. **Shot boundary detection** to reset tracker at camera cuts
3. **Bird's-eye homography** for stable coordinate tracking
4. **OSNet ReID model** for appearance-based re-ID
5. **Track buffer 125 frames** to survive full replay sequences
""")

    report_path = Path("technical_report.md")
    if report_path.exists():
        st.download_button(
            "📥 Download Full Technical Report (Markdown)",
            data=report_path.read_text(encoding="utf-8"),
            file_name="technical_report.md",
            mime="text/markdown",
        )
    else:
        st.info("technical_report.md not found in project root.")
