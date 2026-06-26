<div align="center">

# ARIA
### Augmented Reality Intelligent Assistant

*A spatially-aware AI that sees what you see and guides you through the real world — no cloud required.*

[![Status](https://img.shields.io/badge/status-proof%20of%20concept-teal)]()
[![Platform](https://img.shields.io/badge/platform-Meta%20Quest%203-blue)]()
[![Inference](https://img.shields.io/badge/inference-fully%20local-green)]()
[![License](https://img.shields.io/badge/license-MIT-gray)]()

</div>

---

## What is ARIA?

ARIA is a mixed reality AI assistant that runs **entirely offline**. Put on a Quest 3, activate a session, and an AI that can see your real environment responds with voice, floating text labels, and 3D spatial arrows anchored to objects in your actual space — like a video call with an expert who can see what you see and point at things in your world.

No cloud. No internet dependency. Works in a bunker, a disaster zone, or a remote field site.

### The interaction model

```
You speak → ARIA sees your environment + hears your question
          → understands spatially what you're looking at
          → responds with voice + floating MR annotations in your real space
```

---

## Why this matters

Existing AI assistants are screen-bound. Existing MR applications are pre-scripted and static. ARIA is the first system to combine:

- **Fully local multimodal inference** — vision + language, no API key required
- **Genuine spatial awareness** — annotations anchored to real objects, not floating HUD elements
- **Offline-first architecture** — designed for environments with no connectivity
- **Consumer hardware** — Meta Quest 3 ($499) + a gaming laptop, not a $50k enterprise rig

### Phase 1 — Quest 3 proof of concept (current)
Validates the full technical loop on accessible hardware.

### Phase 2 — Ray-Ban Display migration (next)
Same backend, same model, same concept. Swap the Unity MR frontend for the Meta Wearables Device Access Toolkit web app. Glasses form factor. Always-on. Invisible to bystanders.

---

## Architecture

```
┌─────────────────────────────────┐
│         Meta Quest 3            │
│  passthrough cam  +  mic        │
│  MR render: voice, labels,      │
│  arrows anchored in real space  │
└──────────┬──────────────────────┘
           │ frames + audio (WiFi, local only)
           ▼
┌─────────────────────────────────┐
│         Windows Laptop          │
│                                 │
│  Whisper STT → Qwen 3.5 (Ollama)│
│      ↓                          │
│  annotation JSON + Piper TTS    │
│                                 │
│  Unity WebSocket bridge         │
└─────────────────────────────────┘
           no internet required
```

Full architecture docs → [`docs/architecture/`](docs/architecture/)

---

## Tech stack

| Layer | Technology |
|---|---|
| MR rendering | Unity 6 + Meta XR SDK + MRTK3 |
| Speech-to-text | Whisper (local, `openai-whisper`) |
| Vision + language | Qwen 3.5 9B via Ollama |
| Text-to-speech | Piper TTS (local) |
| Bridge | Python WebSocket server (FastAPI + websockets) |
| Target hardware | Meta Quest 3, Windows laptop (RTX 5060, i9, 32GB) |

---

## Repo structure

```
aria/
├── unity-client/          # Unity project (Quest 3 MR frontend)
├── backend/
│   ├── bridge/            # WebSocket server (Unity ↔ Python)
│   ├── inference/         # Ollama client, vision preprocessing
│   ├── stt/               # Whisper speech-to-text pipeline
│   └── tts/               # Piper text-to-speech pipeline
├── docs/
│   ├── architecture/      # System diagrams, data flow specs
│   ├── decisions/         # Architecture Decision Records (ADRs)
│   └── media/             # Demo screenshots, videos
├── scripts/               # Setup, launch, and test scripts
└── .github/               # Issue templates, PR templates
```

---

## Getting started

> Full setup guide: [`docs/SETUP.md`](docs/SETUP.md)

**Prerequisites**
- Windows 10/11 laptop with NVIDIA GPU (8GB+ VRAM)
- Meta Quest 3 headset
- Unity 6 (see setup guide for exact version)
- Python 3.11+
- Ollama

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/aria.git
cd aria

# 2. Start the backend
cd backend
pip install -r requirements.txt
python bridge/server.py

# 3. Pull the model
ollama pull qwen2.5:7b-instruct

# 4. Open Unity project and deploy to Quest
# See docs/SETUP.md for full Unity setup
```

---

## Roadmap

| Stage | Status | Description |
|---|---|---|
| 1 | 🔄 In progress | Unity + Quest 3 passthrough + WebSocket bridge |
| 2 | ⬜ Planned | Local inference pipeline (Whisper + Qwen + Piper) |
| 3 | ⬜ Planned | Spatial annotation system (floating labels + arrows) |
| 4 | ⬜ Planned | Full session loop — voice in, MR annotations out |
| 5 | ⬜ Planned | Ray-Ban Display migration |

---

## Demo

*Demo video and screenshots will be added after Stage 4 completion.*

---

## Contact

Built by Abhiram — undergraduate researcher at Stevens Institute of Technology (SAIRG), robotics engineer at ENVER Studio.

---

<div align="center">
<sub>Proof of concept · Not affiliated with Meta Platforms · Built on open-source models</sub>
</div>
