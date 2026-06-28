"""
ARIA Stage 3 - Qwen2.5-Omni-3B Inference Module
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

MODEL_PATH = "C:/Users/abhir/aria/models/qwen2.5-omni-3b"
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "You are ARIA, a spatially-aware AI assistant running on a Meta Quest 3. "
    "You can see the user's environment through the camera. "
    "Be concise - 1 to 2 sentences max. No markdown, no bullet points."
)


class OmniInference:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self._loaded = False
        self.load_time_s = 0.0
        self.vram_gb = 0.0

    def load(self):
        if self._loaded:
            return
        t0 = time.perf_counter()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
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

    def run(self, text, image=None, max_new_tokens=MAX_NEW_TOKENS):
        if not self._loaded:
            raise RuntimeError("call load() first")
        t0 = time.perf_counter()
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{text}"},
            ]
            images = [image]
        else:
            content = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{text}"}]
            images = None
        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if images:
            inputs = self.processor(text=[chat_text], images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=[chat_text], return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        t_pre = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.thinker.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        t_inf = time.perf_counter() - t1
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        response = self.processor.decode(generated, skip_special_tokens=True).strip()
        return response, {
            "preprocess_s": round(t_pre, 3),
            "inference_s": round(t_inf, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        }

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def status(self):
        return {
            "loaded": self._loaded,
            "model_path": self.model_path,
            "load_time_s": round(self.load_time_s, 1),
            "vram_gb": round(self.vram_gb, 2),
        }
