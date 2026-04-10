import cv2
import numpy as np
import yaml
import logging
from pathlib import Path


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def inspect_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    print(f"Video: {path}")
    print(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {frames}, Duration: {duration:.1f}s")
    return {"width": w, "height": h, "fps": fps, "frames": frames, "duration": duration}


def get_color(track_id: int) -> tuple:
    """Deterministic unique BGR color per track ID."""
    np.random.seed(int(track_id) * 7 + 31)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output_path}")
    return writer


def save_frame(frame: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, frame)