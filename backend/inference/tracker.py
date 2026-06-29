# backend/inference/tracker.py
# ARIA Stage 4 - SAM2 streaming object tracker

import sys, os, logging
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_SAM2_REPO   = os.path.abspath("backend/sam2_repo")
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_SAM2_CKPT   = os.path.abspath("backend/models/sam2.1_hiera_tiny.pt")


def _get_mask_center(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(np.mean(xs)), float(np.mean(ys))]


class ObjectTracker:
    def __init__(self, device: str = "cuda"):
        self._original_cwd = os.getcwd()
        os.chdir(_SAM2_REPO)
        sys.path.insert(0, _SAM2_REPO)

        from sam2.build_sam import build_sam2_video_predictor
        logger.info("[Tracker] Loading SAM2 hiera_tiny...")
        self.device = device
        self.predictor = build_sam2_video_predictor(
            config_file=_SAM2_CONFIG,
            ckpt_path=_SAM2_CKPT,
            device=device,
        )
        # Do NOT call .half() — SAM2.1 manages its own internal precision.
        # fp32 inference uses ~224MB VRAM which fits in our budget.

        os.chdir(self._original_cwd)
        logger.info("[Tracker] SAM2 ready")

    def track_sequence(self, frames: list, label: str, seed_bbox: list) -> list:
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="aria_sam2_")
        try:
            for i, frame in enumerate(frames):
                frame.convert("RGB").save(
                    os.path.join(tmpdir, f"{i:05d}.jpg"), quality=95
                )

            os.chdir(_SAM2_REPO)
            inference_state = self.predictor.init_state(video_path=tmpdir)

            box_np = np.array(seed_bbox, dtype=np.float32)
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box_np,
            )
            os.chdir(self._original_cwd)

            results = []
            last_known_center = None

            os.chdir(_SAM2_REPO)
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state
            ):
                os.chdir(self._original_cwd)
                obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids)

                if 1 in obj_ids_list:
                    idx = obj_ids_list.index(1)
                    mask = (mask_logits[idx, 0].cpu().float().numpy() > 0.0)
                    center = _get_mask_center(mask)
                else:
                    center = None

                if center is not None:
                    last_known_center = center
                    source = "sam2"
                elif last_known_center is not None:
                    center = last_known_center
                    source = "fallback"
                else:
                    source = "no_mask"

                results.append({
                    "frame":       frame_idx,
                    "mask_center": [round(center[0], 1), round(center[1], 1)] if center else None,
                    "source":      source,
                })
                os.chdir(_SAM2_REPO)

            os.chdir(self._original_cwd)
            self.predictor.reset_state(inference_state)
            return results

        finally:
            os.chdir(self._original_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
