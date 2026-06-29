# backend/inference/detector.py
# ARIA Stage 4 - YOLOv8-nano wrapper

from ultralytics import YOLO
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)

_MODEL_PATH = "backend/models/yolov8n.pt"
_CONFIDENCE_THRESHOLD = 0.35


class ObjectDetector:
    def __init__(self, model_path: str = _MODEL_PATH, device: str = "cuda"):
        logger.info(f"[Detector] Loading YOLOv8n from {model_path} on {device}")
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        if device == "cuda":
            self.model.model.half()
        logger.info("[Detector] YOLOv8n ready")

    def detect(self, image: Image.Image) -> list:
        results = self.model(image, verbose=False, conf=_CONFIDENCE_THRESHOLD)
        detections = []
        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls)]
                conf  = float(box.conf)
                xyxy  = box.xyxy[0].tolist()
                cx    = (xyxy[0] + xyxy[2]) / 2.0
                cy    = (xyxy[1] + xyxy[3]) / 2.0
                detections.append({
                    "label":      label,
                    "confidence": round(conf, 3),
                    "bbox_xyxy":  [round(v, 1) for v in xyxy],
                    "center_xy":  [round(cx, 1), round(cy, 1)],
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def build_object_map(self, detections: list) -> dict:
        obj_map = {}
        for det in detections:
            label = det["label"]
            if label not in obj_map:
                obj_map[label] = det["center_xy"]
        return obj_map
