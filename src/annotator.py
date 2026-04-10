import cv2
import numpy as np
from collections import defaultdict, deque
from src.utils import get_color


class Annotator:
    def __init__(self, trail_length: int = 45, show_conf: bool = False):
        self.trail_length = trail_length
        self.show_conf = show_conf
        # track_id → deque of (cx, cy) positions
        self.track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))

    def update(self, track_id: int, cx: int, cy: int):
        self.track_history[track_id].append((cx, cy))

    def draw(self, frame: np.ndarray, results, frame_num: int = 0) -> np.ndarray:
        out = frame.copy()
        active_count = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes   = results.boxes.xyxy.cpu().numpy()
            ids     = results.boxes.id.cpu().numpy().astype(int)
            confs   = results.boxes.conf.cpu().numpy()
            active_count = len(ids)

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                color = get_color(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.update(tid, cx, cy)

                # Bounding box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                # Label with filled background
                label = f"ID {tid}" + (f"  {conf:.2f}" if self.show_conf else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(out, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # Trajectory trail (fades from thin/transparent → thick)
                trail = list(self.track_history[tid])
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(alpha * 4))
                    cv2.line(out, trail[i - 1], trail[i], color, thickness)

        self._draw_hud(out, active_count, frame_num)
        return out

    def _draw_hud(self, frame: np.ndarray, count: int, frame_num: int):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (290, 65), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Players tracked: {count}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 80), 2)
        cv2.putText(frame, f"Frame: {frame_num}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    def get_all_positions(self) -> list:
        """Flat list of all (x,y) positions for heatmap."""
        positions = []
        for trail in self.track_history.values():
            positions.extend(list(trail))
        return positions

    def get_active_ids(self) -> set:
        return set(self.track_history.keys())