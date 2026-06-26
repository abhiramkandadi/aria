# System architecture overview

## Data flow

```
Quest 3 (capture)
    │
    ├── passthrough camera frames (JPEG, ~5fps)
    └── microphone audio (PCM chunks)
         │
         │ WebSocket (local WiFi — no internet)
         ▼
Windows Laptop (inference)
    │
    ├── Whisper STT → transcribed text
    ├── Qwen 3.5 9B (Ollama) → response text + annotation JSON
    └── Piper TTS → audio response
         │
         │ WebSocket response payload
         ▼
Quest 3 (render)
    ├── spatial audio playback
    ├── floating text labels (world-locked)
    └── 3D arrows anchored to real objects
```

## Annotation JSON schema

The bridge server returns this payload to Unity after each inference cycle:

```json
{
  "session_id": "uuid",
  "response_text": "The pipe on your left is the water shutoff valve.",
  "audio_url": "ws://local/tts/chunk",
  "annotations": [
    {
      "type": "label",
      "text": "Water shutoff valve",
      "position": { "x": -0.4, "y": 0.1, "z": 1.2 },
      "color": "teal",
      "duration_sec": 8
    },
    {
      "type": "arrow",
      "direction": { "x": -1, "y": 0, "z": 0 },
      "origin": { "x": 0, "y": 0, "z": 0 },
      "color": "teal",
      "duration_sec": 8
    }
  ]
}
```

## Network topology

```
Quest 3 (192.168.x.x) ──WiFi──► Laptop (192.168.x.x:8765)
                                  WebSocket server
                                  No external routes
```

## Key constraints

| Constraint | Value | Reason |
|---|---|---|
| Max VRAM | 8GB | RTX 5060 laptop GPU |
| Model size | ≤7B Q4 | Fits with headroom for KV cache |
| Frame rate to backend | ~3-5 fps | WebSocket + inference latency |
| Response cycle | 2-5 sec | Acceptable for guided-task UX |
| Network | Local WiFi only | Offline-first requirement |
