"""
ARIA Stage 3 — Qwen2.5-Omni-3B Inference Module
Wraps thinker-only vision+language inference.

Usage:
    from backend.inference.omni import OmniInference
    omni = OmniInference()
    omni.load()
    text, latency = omni.run("What do you see?", image=pil_image)

Notes:
    - Uses 4-bit NF4 quantization (bitsandbytes) to fit in 8GB VRAM
    - model.thinker.generate() only — native Talker audio disabled
      (broken in transformers 5.x on Windows, see Stage 3 handoff)
    - device_map="auto" splits across GPU+CPU as needed
    - RTX 5060 sm_120: requires torch 2.7.0+cu128
"""

import time
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

MODEL_PATH     = "C:/Users/abhir/aria/models/qwen2.5-omni-3b"
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "You are ARIA, a spatially-aware AI assistant running on a Meta Quest 3. "
    "You can see the user's environment through the camera. "
    "Be concise — 1 to 2 sentences max. No markdown, no bullet points."
)


class OmniInference:
    """
    Singleton-pattern wrapper for Qwen2.5-Omni-3B thinker inference.
    Load once at server startup, reuse for all requests.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path  = model_path
        self.processor   = None
        self.model       = None
        self._loaded     = False
        self.load_time_s = 0.0
        self.vram_gb     = 0.0

    def load(self) -> None:
        """Load model and processor. Call once at startup."""
        if self._loaded:
            return

        t0 = time.perf_counter()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.load_time_s = time.perf_counter() - t0
        if torch.cuda.is_available():
            self.vram_gb = torch.cuda.memory_allocated(0) / 1e9

        self._loaded = True

    def run(
        self,
        text: str,
        image: Image.Image = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> tuple[str, dict]:
        """
        Run inference. Returns (response_text, latency_dict).

        Args:
            text:           User query string.
            image:          Optional PIL Image from Quest camera frame.
            max_new_tokens: Max tokens to generate.

        Returns:
            response_text:  Model response string.
            latency:        Dict with preprocess_s, inference_s, total_s.
        """
        if not self._loaded:
            raise RuntimeError("OmniInference.load() must be called before run()")

        t0 = time.perf_counter()

        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"{SYSTEM_PROMPT}\n\n{text}"},
            ]
            images = [image]
        else:
            content = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{text}"}]
            images  = None

        messages  = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if images:
            inputs = self.processor(
                text=[chat_text], images=images, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[chat_text], return_tensors="pt"
            )

        inputs = {
            k: v.to("cuda") if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        t_preprocess = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.thinker.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t_inference = time.perf_counter() - t1

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        response  = self.processor.decode(
            generated, skip_special_tokens=True
        ).strip()

        latency = {
            "preprocess_s": round(t_preprocess, 3),
            "inference_s":  round(t_inference,  3),
            "total_s":      round(time.perf_counter() - t0, 3),
        }
        return response, latency

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> dict:
        return {
            "loaded":      self._loaded,
            "model_path":  self.model_path,
            "load_time_s": round(self.load_time_s, 1),
            "vram_gb":     round(self.vram_gb, 2),
        }