"""
ARIA Bridge Server — Stage 3
Replaces Ollama + Whisper + Piper with:
  - Qwen2.5-Omni-3B (4-bit, thinker-only) for vision+language
  - Piper TTS for speech output
  - Whisper (CPU) for speech-to-text from Quest mic audio

Model loads ONCE at startup and stays hot for all requests.

Message protocol (from Unity ARIABridgeClient):
  - JSON  {"type": "text",      "content": "..."}
  - JSON  {"type": "handshake"}
  - JSON  {"type": "frame_text", "content": "...", "frame_b64": "<base64 JPEG>"}
  - Binary: raw JPEG frame (legacy compat)

Response protocol (to Unity):
  - JSON  {"type": "response", "text": "...", "annotations": [...], "latency": {...}}
  - Binary: WAV audio bytes
"""

import asyncio
import base64
import io
import json
import logging
import time
import wave
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import websockets
from PIL import Image
from piper.voice import PiperVoice
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

# ── Config ────────────────────────────────────────────────────────────────────
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
MODEL_PATH     = "C:/Users/abhir/aria/models/qwen2.5-omni-3b"
VOICE_MODEL    = "backend/tts/voices/en_US-lessac-medium.onnx"
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "You are ARIA, a spatially-aware AI assistant running on a Meta Quest 3. "
    "You can see the user's environment through the camera. "
    "Be concise — 1 to 2 sentences max. No markdown, no bullet points."
)

ANNOTATION_TEMPLATE = {
    "type": "label",
    "text": "",
    "position": {"x": 0.0, "y": 1.5, "z": 2.0},
    "color": "#00FF88",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("aria.bridge")

# ── Global model handles (loaded once at startup) ─────────────────────────────
processor: Qwen2_5OmniProcessor = None
model:     Qwen2_5OmniForConditionalGeneration = None
tts_voice: PiperVoice = None
_model_load_time: float = 0.0


def load_models():
    global processor, model, tts_voice, _model_load_time

    logger.info("Loading Qwen2.5-Omni-3B (4-bit)...")
    t0 = time.perf_counter()

    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    _model_load_time = time.perf_counter() - t0
    vram_gb = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"Model loaded in {_model_load_time:.1f}s | VRAM: {vram_gb:.2f}GB")

    logger.info("Loading Piper TTS...")
    tts_voice = PiperVoice.load(VOICE_MODEL)
    logger.info("Piper TTS loaded.")


def run_inference(text: str, image: Image.Image = None) -> tuple[str, dict]:
    """
    Run thinker-only inference. Returns (response_text, latency_dict).
    Image is optional — text-only if None.
    """
    t0 = time.perf_counter()

    if image is not None:
        content = [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"{SYSTEM_PROMPT}\n\n{text}"},
        ]
        images = [image]
    else:
        content = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{text}"}]
        images = None

    messages = [{"role": "user", "content": content}]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if images:
        inputs = processor(text=[chat_text], images=images, return_tensors="pt")
    else:
        inputs = processor(text=[chat_text], return_tensors="pt")

    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    t_preprocess = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.thinker.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    t_inference = time.perf_counter() - t1

    input_len    = inputs["input_ids"].shape[1]
    generated    = output_ids[0][input_len:]
    response     = processor.decode(generated, skip_special_tokens=True).strip()

    latency = {
        "preprocess_s":  round(t_preprocess, 3),
        "inference_s":   round(t_inference,  3),
        "total_s":       round(time.perf_counter() - t0, 3),
    }
    return response, latency


def run_tts(text: str) -> tuple[bytes, float]:
    """Synthesize text to WAV bytes using Piper. Returns (wav_bytes, duration_s)."""
    t0 = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        tts_voice.synthesize_wav(text, wf)
    wav_bytes    = buf.getvalue()
    tts_time     = time.perf_counter() - t0
    return wav_bytes, tts_time


async def handle_client(websocket):
    client_addr = websocket.remote_address
    logger.info("Client connected: %s", client_addr)

    try:
        async for raw_message in websocket:

            t_request_start = time.perf_counter()

            # ── Binary frame (legacy — JPEG bytes direct) ─────────────────
            if isinstance(raw_message, bytes):
                frame_size = len(raw_message)
                logger.info("Binary frame: %d bytes from %s", frame_size, client_addr)
                try:
                    image = Image.open(io.BytesIO(raw_message)).convert("RGB")
                    response_text, inf_latency = run_inference(
                        "What do you see? Describe briefly.", image
                    )
                except Exception as e:
                    logger.error("Frame inference error: %s", e)
                    response_text = "I couldn't process that frame."
                    inf_latency   = {}

                wav_bytes, tts_time = run_tts(response_text)
                total_s = time.perf_counter() - t_request_start

                annotation = {**ANNOTATION_TEMPLATE, "text": response_text}
                meta = json.dumps({
                    "type":        "response",
                    "text":        response_text,
                    "annotations": [annotation],
                    "latency": {
                        **inf_latency,
                        "tts_s":   round(tts_time, 3),
                        "total_s": round(total_s,  3),
                    },
                })
                await websocket.send(meta)
                await websocket.send(wav_bytes)
                logger.info("Binary frame handled in %.2fs", total_s)
                continue

            # ── JSON messages ─────────────────────────────────────────────
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error", "message": "Expected JSON"
                }))
                continue

            msg_type = msg.get("type", "unknown")
            logger.info("Message type='%s' from %s", msg_type, client_addr)

            # Handshake
            if msg_type == "handshake":
                await websocket.send(json.dumps({
                    "type":        "handshake_ack",
                    "server":      "ARIA Bridge Stage 3",
                    "model":       "Qwen2.5-Omni-3B (4-bit)",
                    "tts":         "Piper en_US-lessac-medium",
                    "vram_gb":     round(torch.cuda.memory_allocated(0)/1e9, 2),
                    "model_load_s": round(_model_load_time, 1),
                }))

            # Text only
            elif msg_type == "text":
                content = msg.get("content", "").strip()
                if not content:
                    await websocket.send(json.dumps({
                        "type": "error", "message": "Empty content"
                    }))
                    continue

                response_text, inf_latency = run_inference(content, image=None)
                wav_bytes, tts_time        = run_tts(response_text)
                total_s = time.perf_counter() - t_request_start

                annotation = {**ANNOTATION_TEMPLATE, "text": response_text}
                await websocket.send(json.dumps({
                    "type":        "response",
                    "text":        response_text,
                    "annotations": [annotation],
                    "latency": {
                        **inf_latency,
                        "tts_s":   round(tts_time, 3),
                        "total_s": round(total_s,  3),
                    },
                }))
                await websocket.send(wav_bytes)
                logger.info("Text request handled in %.2fs", total_s)

            # Frame + text (Stage 3 primary path)
            elif msg_type == "frame_text":
                content  = msg.get("content",   "What do you see?").strip()
                frame_b64 = msg.get("frame_b64", "")

                image = None
                if frame_b64:
                    try:
                        img_bytes = base64.b64decode(frame_b64)
                        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    except Exception as e:
                        logger.warning("Could not decode frame_b64: %s", e)

                response_text, inf_latency = run_inference(content, image=image)
                wav_bytes, tts_time        = run_tts(response_text)
                total_s = time.perf_counter() - t_request_start

                annotation = {**ANNOTATION_TEMPLATE, "text": response_text}
                await websocket.send(json.dumps({
                    "type":        "response",
                    "text":        response_text,
                    "annotations": [annotation],
                    "latency": {
                        **inf_latency,
                        "tts_s":   round(tts_time, 3),
                        "total_s": round(total_s,  3),
                    },
                }))
                await websocket.send(wav_bytes)
                logger.info(
                    "frame_text handled in %.2fs (vision=%s)",
                    total_s, image is not None
                )

            else:
                await websocket.send(json.dumps({
                    "type": "error", "message": f"Unknown type: {msg_type}"
                }))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Client disconnected cleanly: %s", client_addr)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning("Connection dropped: %s — %s", client_addr, e)
    except Exception as e:
        logger.exception("Unhandled error for %s: %s", client_addr, e)


async def main():
    logger.info("=" * 55)
    logger.info("ARIA Bridge Server — Stage 3")
    logger.info("=" * 55)
    load_models()
    logger.info("Starting WebSocket on ws://%s:%d", WEBSOCKET_HOST, WEBSOCKET_PORT)
    async with websockets.serve(handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
        logger.info("Server ready. Waiting for connections...")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())