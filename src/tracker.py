import os
import torch
from ultralytics import YOLO
from src.utils import setup_logger

logger = setup_logger("Tracker")


class Tracker:
    def __init__(
        self,
        model_name: str = "yolo11x.pt",
        tracker_config: str = "botsort.yaml",
        conf: float = 0.45,
        iou: float = 0.50,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device

        # Resolve tracker config: prefer project configs/ folder, fallback to builtin
        project_cfg = os.path.join("configs", tracker_config)
        self.tracker_config = project_cfg if os.path.exists(project_cfg) else tracker_config
        
        logger.info(f"Loading {model_name} | tracker={self.tracker_config} | device={self.device}")
        self.model = YOLO(model_name)

    def track(self, frame, persist: bool = True):
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            tracker=self.tracker_config,
            persist=persist,
            verbose=False,
            device=self.device,
        )
        return results[0]

    def reset(self):
        self.model = YOLO(self.model_name)
        logger.info("Tracker state reset")