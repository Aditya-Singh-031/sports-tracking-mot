import numpy as np
from collections import defaultdict
from src.utils import setup_logger

logger = setup_logger("Speed")


class SpeedEstimator:
    """
    Pixel-displacement speed estimator.
    Cricket pitch = 20.12m, typically ~160px in broadcast → ~8 px/m
    """
    CALIBRATION = {"cricket": 8.0, "football": 6.0}

    def __init__(self, fps: float, sport: str = "cricket"):
        self.fps = fps
        self.px_per_m = self.CALIBRATION.get(sport, 8.0)
        self._prev: dict[int, tuple] = {}   # tid → (pos, frame_num)
        self.speeds: dict[int, list] = defaultdict(list)

    def update(self, track_id: int, position: tuple, frame_num: int):
        if track_id in self._prev:
            prev_pos, prev_frame = self._prev[track_id]
            dt = (frame_num - prev_frame) / self.fps
            if dt > 0:
                dx, dy = position[0] - prev_pos[0], position[1] - prev_pos[1]
                dist_m = np.hypot(dx, dy) / self.px_per_m
                speed_kmh = (dist_m / dt) * 3.6
                if speed_kmh < 40:  # filter noise (>40 km/h is unrealistic for walking)
                    self.speeds[track_id].append(round(speed_kmh, 2))
        self._prev[track_id] = (position, frame_num)

    def get_per_id_stats(self) -> dict:
        stats = {}
        for tid, sl in self.speeds.items():
            if sl:
                stats[int(tid)] = {
                    "max_kmh": round(float(max(sl)), 2),
                    "avg_kmh": round(float(np.mean(sl)), 2),
                }
        return stats

    def get_summary(self) -> dict:
        all_s = [s for sl in self.speeds.values() for s in sl]
        if not all_s:
            return {}
        fastest_id = max(self.speeds, key=lambda t: max(self.speeds[t]) if self.speeds[t] else 0)
        return {
            "global_max_kmh": round(float(max(all_s)), 2),
            "global_avg_kmh": round(float(np.mean(all_s)), 2),
            "fastest_player_id": int(fastest_id),
            "note": "Approximate — based on pixel calibration",
        }