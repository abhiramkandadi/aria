# ADR-001: All inference runs locally — no cloud API calls

**Date:** 2025-06-26
**Status:** accepted

## Context
The system needs to work in environments with no internet connectivity (field operations, disaster response, remote sites). Cloud APIs add latency, require network access, and create a data privacy surface for sensitive use cases.

## Decision
All model inference — vision, language, STT, TTS — runs on the local Windows laptop. The Quest 3 and laptop communicate over local WiFi only. No data leaves the local network at any point.

## Consequences
- Works fully offline — core advantage for target use cases
- Constrained to models that fit in 8GB VRAM (7-9B parameter range)
- Response latency ~2-5s vs ~0.5s cloud — acceptable for guided-task interactions
- Mitigated by choosing best-in-class small models (Qwen 3.5 9B)

## Alternatives considered
- **Cloud API with offline fallback**: rejected — still fails when offline, adds complexity
- **On-device inference on Quest 3**: rejected — Snapdragon XR2 Gen 2 cannot run 7B+ models at interactive speeds
