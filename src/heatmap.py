import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger("Heatmap")


def generate_heatmap(
    positions: list,
    video_w: int,
    video_h: int,
    output_path: str,
    sport: str = "cricket",
    background_frame: np.ndarray = None,
):
    if not positions:
        logger.warning("No positions — skipping heatmap")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

    if background_frame is not None:
        bg = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
        ax.imshow(bg, extent=[0, video_w, video_h, 0], aspect='auto', alpha=0.35)

    heatmap, xedges, yedges = np.histogram2d(
        xs, ys, bins=(100, 56), range=[[0, video_w], [0, video_h]]
    )
    heatmap = heatmap.T
    smooth = cv2.GaussianBlur(heatmap.astype(np.float32), (21, 21), 0)

    im = ax.imshow(
        smooth, extent=[0, video_w, video_h, 0],
        origin='upper', cmap='inferno', alpha=0.72, aspect='auto'
    )
    plt.colorbar(im, ax=ax, shrink=0.75, label='Player presence density')
    ax.set_title(f'{sport.capitalize()} — Player Movement Heatmap', fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_xlim(0, video_w)
    ax.set_ylim(video_h, 0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"Heatmap saved: {output_path}")