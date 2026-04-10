# src/utils.py (add this function)
import cv2

def inspect_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    print(f"Video: {path}")
    print(f"Resolution: {int(w)}x{int(h)}, FPS: {fps:.2f}, Frames: {int(frame_count)}, Duration: {duration:.1f}s")
    cap.release()