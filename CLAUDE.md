# Vision Assistant - Project Context for Claude

> **Last Updated:** 2026-02-07
> **Status:** Phase 1 Implementation Ready

---

## Project Overview

**Vision Assistant** is a complete Home AI system for smart security and family recognition.

### What's Built (v2 - Complete)
- Face recognition with auto-learning (82%+ accuracy)
- Person detection (YOLOv8n) + tracking
- Visit logging + tagging system (family/friends/public + roles)
- Voice greetings (TTS)
- VisionDB with persons, embeddings, visits tables

### What's Next (v3 - In Progress)
- Multi-camera support with unified state
- GPT-4o Vision for scene understanding
- Main Agent (Deep Agents) with tools
- Telegram bot for notifications + HITL
- House state management

---

## Architecture

**See:** [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design.

### Key Components

```
Device Layer → Perception → State Store → Event Bus → Agent → Output
   (24/7)       (GPU)        (Memory)     (Priority)   (LLM)   (Voice/TG)
```

### Implementation Phases

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Foundation (devices, state, events, DB) | **IN PROGRESS** |
| Phase 2 | Vision Intelligence (GPT-4o, agent) | TODO |
| Phase 3 | Multi-Camera | TODO |
| Phase 4 | Conversation (STT/TTS) | TODO |
| Phase 5 | Telegram Bot | TODO |

---

## Document Registry

| File | Purpose | Status |
|------|---------|--------|
| `ARCHITECTURE.md` | Complete system architecture v3.0 | Current |
| `RESEARCH.md` | Best practices research | Current |
| `ISSUES.md` | GitHub issues documentation | Current |
| `README.md` | Project overview + quick start | Current |
| `requirements.txt` | Python dependencies | Current |

---

## Code Structure

```
src/v2/
├── core/                    # NEW - Core abstractions
│   ├── devices.py           # Device, CameraState models
│   ├── state.py             # HouseState, PersonPresence
│   └── events.py            # EventBus, Event types
│
├── perception/              # EXISTING - Detection modules
│   ├── motion/              # Motion detection
│   ├── person/              # Person detection (YOLO)
│   └── face/                # Face detection (InsightFace)
│
├── storage/                 # EXISTING - Database layer
│   └── schema.py            # VisionDB (extending)
│
├── agent/                   # NEW - LLM agent
│   ├── main.py              # MainAgent (Deep Agents)
│   ├── tools.py             # analyze_scene, speak, notify
│   └── trigger.py           # TriggerController
│
├── vision/                  # NEW - Vision API
│   ├── client.py            # GPT-4o client
│   └── schemas.py           # VisionResult, PersonDescription
│
└── live_pipeline.py         # EXISTING - Main pipeline (refactoring)
```

---

## GitHub Issues

### Phase 1: Foundation
- [#1](https://github.com/KlementMultiverse/vision-assistant/issues/1) Device Registry + Camera State
- [#2](https://github.com/KlementMultiverse/vision-assistant/issues/2) State Store with Pub/Sub
- [#3](https://github.com/KlementMultiverse/vision-assistant/issues/3) Event Bus
- [#4](https://github.com/KlementMultiverse/vision-assistant/issues/4) Database Migration
- [#5](https://github.com/KlementMultiverse/vision-assistant/issues/5) Pipeline Refactor

### Phase 2: Vision Intelligence
- [#6](https://github.com/KlementMultiverse/vision-assistant/issues/6) Vision Client (GPT-4o)
- [#7](https://github.com/KlementMultiverse/vision-assistant/issues/7) Trigger Controller
- [#8](https://github.com/KlementMultiverse/vision-assistant/issues/8) Main Agent (Deep Agents)

---

## Key Design Decisions

### 1. Event-Driven Architecture
- **Why:** Decouple perception from actions
- **How:** Priority queue (CRITICAL=0, HIGH=10, NORMAL=50, LOW=100)
- **Pattern:** Frigate-style events (NEW, UPDATE, END lifecycle)

### 2. Devices as First-Class Entities
- **Why:** Multi-camera, multi-device support
- **How:** Device registry with status, capabilities, heartbeat
- **Pattern:** Home Assistant hub-network-device

### 3. State Hierarchy
- **Why:** Know who's where at all times
- **How:** HouseState → RoomState → CameraState → PersonPresence
- **Pattern:** Centralized state store with pub/sub

### 4. GPT-4o for Scene Understanding
- **Why:** Understand context beyond face recognition
- **How:** Structured output (Pydantic), dynamic prompts with context
- **Pattern:** Feed known info TO vision (not just FROM)

### 5. Inside vs Outside
- **Why:** Different response requirements
- **Outside:** Fast (GPT-4o, paid, instant response)
- **Inside:** Relaxed (Gemini/Qwen, free, can batch)

---

## Credentials

**Location:** `~/.claude/.env`

Contains:
- GITHUB_TOKEN
- OPENAI_API_KEY (for GPT-4o)
- TELEGRAM_BOT_TOKEN (future)

---

## Running the System

### Current (v2)
```bash
# Activate venv
source venv/bin/activate

# Run live pipeline
python -m src.v2.live_pipeline

# Calibrate face threshold
python calibrate_face.py
```

### Future (v3)
```bash
# Docker Compose (production)
docker compose up -d

# Development
python -m src.v2.main
```

---

## Best Practices for This Project

1. **Type hints everywhere** - All dataclasses, all functions
2. **Immutable where possible** - `@dataclass(frozen=True)` for models
3. **Events, not callbacks** - Publish events, subscribe handlers
4. **Structured output** - Pydantic models for API responses
5. **Existing code first** - Extend VisionDB, don't replace
6. **Test with self** - Delete DB, appear as unknown, verify learning

---

## Open Source References

- [Frigate NVR](https://github.com/blakeblackshear/frigate) - Detection pipeline, event lifecycle
- [Double Take](https://github.com/jakowenko/double-take) - Face recognition on events
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection + embeddings
- [Deep Agents](https://github.com/langchain-ai/deepagents) - LLM agent framework

---

*This file helps Claude understand the project context across sessions.*
