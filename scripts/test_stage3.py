"""
ARIA Stage 3 — End-to-End Test
Connects to the live bridge server and sends:
  1. Handshake
  2. Text-only request
  3. Frame + text request (vision)
Measures and prints hot latency for each phase.
Does NOT reload the model — server stays hot between requests.
"""

import asyncio
import base64
import json
import time
import wave
import io
import numpy as np
import sounddevice as sd
import websockets
from PIL import Image

SERVER_URI  = "ws://192.168.1.99:8765"
IMAGE_PATH  = "scripts/test_image.jpg"

def load_image_as_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def play_wav_bytes(wav_bytes: bytes):
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            frames      = wf.readframes(wf.getnframes())
            audio       = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"    [audio playback error: {e}]")

async def run_test():
    print("=" * 60)
    print("ARIA Stage 3 — End-to-End Test")
    print(f"Server: {SERVER_URI}")
    print("=" * 60)

    frame_b64 = load_image_as_b64(IMAGE_PATH)

    async with websockets.connect(SERVER_URI) as ws:

        # ── Phase 1: Handshake ────────────────────────────────────────
        print("\n[Phase 1] Handshake...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "handshake"}))
        raw = await ws.recv()
        handshake_time = time.perf_counter() - t0
        ack = json.loads(raw)
        print(f"    Server:        {ack.get('server')}")
        print(f"    Model:         {ack.get('model')}")
        print(f"    VRAM:          {ack.get('vram_gb')}GB")
        print(f"    Model load:    {ack.get('model_load_s')}s (cold start)")
        print(f"    Handshake RTT: {handshake_time*1000:.0f}ms")

        # ── Phase 2: Text-only (hot) ──────────────────────────────────
        print("\n[Phase 2] Text-only request (hot)...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "type":    "text",
            "content": "Hello ARIA, what can you do?",
        }))

        # Receive JSON metadata
        raw_meta = await ws.recv()
        meta     = json.loads(raw_meta)
        # Receive WAV audio
        wav_bytes = await ws.recv()
        text_total = time.perf_counter() - t0

        print(f"    Response:      {meta.get('text')}")
        lat = meta.get("latency", {})
        print(f"    Inference:     {lat.get('inference_s')}s")
        print(f"    TTS:           {lat.get('tts_s')}s")
        print(f"    HOT LATENCY:   {text_total:.2f}s (wall clock)")
        print("    Playing audio...")
        play_wav_bytes(wav_bytes)

        # ── Phase 3: Vision (frame + text, hot) ──────────────────────
        print("\n[Phase 3] Vision request — frame + text (hot)...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "type":      "frame_text",
            "content":   "What do you see in front of me? Be brief.",
            "frame_b64": frame_b64,
        }))

        raw_meta  = await ws.recv()
        meta      = json.loads(raw_meta)
        wav_bytes = await ws.recv()
        vision_total = time.perf_counter() - t0

        print(f"    Response:      {meta.get('text')}")
        lat = meta.get("latency", {})
        print(f"    Inference:     {lat.get('inference_s')}s")
        print(f"    TTS:           {lat.get('tts_s')}s")
        print(f"    HOT LATENCY:   {vision_total:.2f}s (wall clock)")
        print("    Playing audio...")
        play_wav_bytes(wav_bytes)

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 3 LATENCY SUMMARY")
        print("=" * 60)
        print(f"  Cold start (model load):     {ack.get('model_load_s')}s")
        print(f"  Phase 1 — Handshake RTT:     {handshake_time*1000:.0f}ms")
        print(f"  Phase 2 — Text hot latency:  {text_total:.2f}s")
        print(f"  Phase 3 — Vision hot latency:{vision_total:.2f}s")
        print("=" * 60)
        print("Stage 3 COMPLETE")

asyncio.run(run_test())