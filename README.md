# Vision Assistant

> **First open-source Vision + Learning + Acting agent for home use**

Real-time face recognition system that learns and improves over time. Designed for smart home applications - doorbell cameras, security, family recognition.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Features

- **Real-time Face Recognition** - 82%+ accuracy at 1-2 meter distance
- **Auto-Learning** - Automatically improves as it sees you in different conditions
- **Multi-Camera Support** - Unified tracking across cameras
- **Scene Understanding** - GPT-4o Vision for context awareness
- **Visit Tracking** - Logs who visited, when, and for how long
- **Flexible Tagging** - Groups (family/friends/public) + Roles (daughter, delivery_guy, etc.)
- **Voice Greetings** - Speaks personalized greetings
- **Agent-Ready** - Deep Agents framework for intelligent decisions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VISION ASSISTANT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        DEVICE LAYER (24/7)                          │    │
│  │    Camera (door)  │  Camera (living)  │  Microphone  │  Speaker    │    │
│  └──────────────────────────────┬─────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     PERCEPTION LAYER (GPU)                          │    │
│  │   Motion Detection → Person Detection → Face Detection → Embedding │    │
│  │        (OpenCV)         (YOLOv8n)        (InsightFace)    (512-D)  │    │
│  └──────────────────────────────┬─────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    STATE STORE + EVENT BUS                          │    │
│  │   House State  │  Room States  │  Person Presence  │  Events Queue │    │
│  │                                                                     │    │
│  │   Priority: CRITICAL(0) → HIGH(10) → NORMAL(50) → LOW(100)         │    │
│  └──────────────────────────────┬─────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      MAIN AGENT (Deep Agents)                       │    │
│  │                                                                     │    │
│  │   Tools: analyze_scene │ speak │ notify_owner │ get_house_state    │    │
│  │   APIs:  GPT-4o (outside/fast) │ Free APIs (inside/relaxed)        │    │
│  └──────────────────────────────┬─────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         OUTPUT LAYER                                │    │
│  │        Voice (TTS)  │  Display (UI)  │  Telegram  │  Event Logs    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **See:** [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design with code examples.

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (recommended)
- Webcam or IP camera

### Installation

```bash
# Clone the repository
git clone https://github.com/KlementMultiverse/vision-assistant.git
cd vision-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Live pipeline with recognition + tagging
python -m src.v2.live_pipeline

# Calibrate threshold for your face
python calibrate_face.py

# Test recognition accuracy
python recognize_only.py
```

---

## Database Schema

```
┌─────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   devices   │       │    persons      │       │   embeddings    │
├─────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)     │       │ id (PK)         │◄──────│ person_id (FK)  │
│ type        │       │ name            │       │ embedding       │
│ location    │       │ group_type      │       │ confidence      │
│ zone        │       │ role            │       │ captured_at     │
│ status      │       │ visit_count     │       └─────────────────┘
└──────┬──────┘       └────────┬────────┘
       │                       │
       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  camera_states  │     │  observations   │     │ vision_results  │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ camera_id (FK)  │     │ person_id (FK)  │     │ observation_id  │
│ motion_detected │     │ camera_id (FK)  │     │ description     │
│ persons_count   │     │ start_time      │     │ safety_level    │
└─────────────────┘     │ vision_context  │     └─────────────────┘
                        └─────────────────┘
```

---

## Configuration

### Recognition Threshold

| Threshold | Behavior |
|-----------|----------|
| 0.20 | Very lenient - may false positive |
| **0.25** | **Recommended** - balanced |
| 0.35 | Strict - may miss in bad lighting |

### Tagging System

| Group | Purpose | Greeting |
|-------|---------|----------|
| `family` | Household members | "Welcome home, [Name]!" |
| `friends` | Known visitors | "Hello, [Name]!" |
| `public` | Everyone else | "Hello!" |

---

## Performance

Tested on RTX 4060 (8GB VRAM):

| Metric | Value |
|--------|-------|
| FPS | 25-30 |
| Recognition accuracy | 82%+ |
| Face detection | 89% of frames |
| VRAM usage | ~1.5GB |

---

## Hardware Requirements

### Minimum
- CPU: Any modern quad-core
- RAM: 8GB
- GPU: NVIDIA GTX 1060 or equivalent
- Camera: 720p webcam

### Recommended
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 16GB
- GPU: NVIDIA RTX 3060+ (8GB VRAM)
- Camera: 1080p webcam or IP camera

---

## Privacy

- **All processing is local** - no cloud APIs for face recognition
- **Database is local** - your face data never leaves your machine
- **No telemetry** - zero data collection
- **.gitignore** - prevents accidental commits of personal data

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| Foundation | In Progress | Devices, State, Events, Database |
| Vision Intelligence | Planned | GPT-4o, Trigger Controller, Agent |
| Multi-Camera | Planned | Cross-camera tracking, zones |
| Conversation | Planned | STT, TTS, dialogue |
| Telegram Bot | Planned | Notifications, HITL |

> **See:** [TRACKER.md](TRACKER.md) for detailed progress tracking.

---

## Project Structure

```
vision-assistant/
├── src/v2/
│   ├── core/               # Devices, State, Events
│   ├── perception/         # Detection modules
│   │   ├── motion/         # Motion detection
│   │   ├── person/         # Person detection (YOLO)
│   │   └── face/           # Face detection (InsightFace)
│   ├── storage/            # Database layer
│   ├── agent/              # LLM agent (Deep Agents)
│   ├── vision/             # GPT-4o Vision client
│   └── live_pipeline.py    # Main pipeline
├── ARCHITECTURE.md         # Complete system design
├── TRACKER.md              # Sprint tracking
└── README.md
```

---

## Contributing

Contributions welcome! Please read the contributing guidelines first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face detection and recognition
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Person detection
- [Frigate NVR](https://github.com/blakeblackshear/frigate) - Architecture inspiration
- [Deep Agents](https://github.com/langchain-ai/deepagents) - LLM agent framework

---

<p align="center">
  <b>Built for the open-source community</b>
</p>
