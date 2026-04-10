import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger("Metrics")


def compute_metrics(data: dict) -> dict:
    cot = data.get("count_over_time", [])
    counts = [c for _, c in cot]
    return {
        "tracker":                    data.get("tracker", "?"),
        "sport":                      data.get("sport", "?"),
        "total_frames":               data.get("total_frames", 0),
        "total_unique_ids":           data.get("total_unique_ids", 0),
        "id_switches_approx":         data.get("id_switches_approx", 0),
        "avg_players_per_frame":      round(float(np.mean(counts)), 2) if counts else 0,
        "max_players_per_frame":      int(max(counts)) if counts else 0,
        "frames_with_no_detection":   counts.count(0),
        "avg_processing_fps":         round(data.get("avg_fps", 0), 2),
    }


def plot_count_over_time(data: dict, output_path: str):
    cot = data.get("count_over_time", [])
    if not cot:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fps = 25.0
    times  = [f / fps for f, _ in cot]
    counts = [c for _, c in cot]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    ax.fill_between(times, counts, alpha=0.25, color='#1E88E5')
    ax.plot(times, counts, color='#1565C0', lw=1.5, label='Frame count')

    # Rolling average
    window = 25
    if len(counts) > window:
        roll = np.convolve(counts, np.ones(window) / window, mode='valid')
        rt = times[window // 2: window // 2 + len(roll)]
        ax.plot(rt, roll, color='#E53935', lw=2, linestyle='--', label=f'{window}-frame avg')
        ax.legend(fontsize=11)

    tracker = data.get("tracker", "")
    sport   = data.get("sport", "")
    ax.set_title(f'{sport.capitalize()} — Player Count Over Time ({tracker.upper()})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Players in Frame')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"Count-over-time chart: {output_path}")


def print_metrics(metrics: dict):
    print(f"\n{'='*55}")
    print(f"  METRICS — {metrics['tracker'].upper()} | {metrics['sport'].upper()}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k not in ('tracker', 'sport'):
            print(f"  {k:<35} {v}")
    print(f"{'='*55}\n")