# backend/inference/object_map_state.py
# ARIA Stage 4 - Shared object map state
# Updated by detector every frame. Read by tracker to init SAM2 on a named object.

import threading

_lock = threading.Lock()
_state = {
    "object_map": {},      # {"label": [cx, cy], ...}  — latest frame
    "raw_detections": [],  # full detection list from latest frame
    "frame_index": 0,
}


def update(detections: list, obj_map: dict) -> None:
    """Called by bridge server after each YOLO pass."""
    with _lock:
        _state["object_map"] = obj_map
        _state["raw_detections"] = detections
        _state["frame_index"] += 1


def get_object_map() -> dict:
    """Returns a snapshot of the latest object map."""
    with _lock:
        return dict(_state["object_map"])


def get_center_for(label: str) -> list | None:
    """
    Returns [cx, cy] for the named label from the latest YOLO frame.
    Returns None if label not currently detected.
    Case-insensitive match attempted if exact match fails.
    """
    with _lock:
        obj_map = _state["object_map"]
        if label in obj_map:
            return list(obj_map[label])
        label_lower = label.lower()
        for k, v in obj_map.items():
            if k.lower() == label_lower:
                return list(v)
        return None


def get_frame_index() -> int:
    with _lock:
        return _state["frame_index"]
