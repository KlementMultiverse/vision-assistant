# Vision Assistant - Project Tracker

> **Last Updated:** 2026-02-07
> **Next Session:** Start Phase 1 implementation

---

## Current Sprint: Phase 1 Foundation

### Sprint Goal
Build the core infrastructure: devices, state, events, database.

### Progress

| Issue | Task | Status | Assignee |
|-------|------|--------|----------|
| [#1](https://github.com/KlementMultiverse/vision-assistant/issues/1) | Device Registry + Camera State | TODO | - |
| [#2](https://github.com/KlementMultiverse/vision-assistant/issues/2) | State Store with Pub/Sub | TODO | - |
| [#3](https://github.com/KlementMultiverse/vision-assistant/issues/3) | Event Bus | TODO | - |
| [#4](https://github.com/KlementMultiverse/vision-assistant/issues/4) | Database Migration | TODO | - |
| [#5](https://github.com/KlementMultiverse/vision-assistant/issues/5) | Pipeline Refactor | TODO | - |

### Blockers
- None currently

---

## Completed Work

### 2026-02-07: Architecture & Planning
- [x] Complete system architecture (ARCHITECTURE.md v3.0)
- [x] Best practices research (RESEARCH.md)
- [x] GitHub issues created (#1-#8)
- [x] Project CLAUDE.md created
- [x] README updated with architecture links
- [x] Pushed to GitHub

### Previous: v2 Implementation
- [x] Face recognition with auto-learning
- [x] Person detection (YOLOv8n) + tracking
- [x] Visit logging + tagging system
- [x] Voice greetings (TTS)
- [x] VisionDB with persons, embeddings, visits

---

## Upcoming Phases

### Phase 2: Vision Intelligence
| Issue | Task | Dependencies |
|-------|------|--------------|
| [#6](https://github.com/KlementMultiverse/vision-assistant/issues/6) | Vision Client (GPT-4o) | Phase 1 |
| [#7](https://github.com/KlementMultiverse/vision-assistant/issues/7) | Trigger Controller | #3 Event Bus |
| [#8](https://github.com/KlementMultiverse/vision-assistant/issues/8) | Main Agent (Deep Agents) | #6, #7 |

### Phase 3: Multi-Camera
- Multi-camera manager
- Cross-camera re-identification (OSNet)
- Room/zone state management

### Phase 4: Conversation
- STT integration (Whisper)
- TTS with speaker routing
- Conversation state machine

### Phase 5: Telegram Bot
- Bot setup
- Notification routing
- HITL decision interface

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-07 | Event-driven architecture | Decouple perception from actions, enable multi-camera |
| 2026-02-07 | Devices as first-class entities | Support multi-camera, mic, speaker management |
| 2026-02-07 | GPT-4o for outside, free APIs inside | Cost optimization + speed where needed |
| 2026-02-07 | Extend VisionDB, don't replace | Preserve existing data and functionality |

---

## Session Notes

### Next Session Priorities
1. Start implementing Issue #1 (Device Registry)
2. Create `src/v2/core/` directory structure
3. Write unit tests as we go

### Questions to Resolve
- TimescaleDB vs plain PostgreSQL for events table
- Redis vs in-memory for event bus (POC vs production)

---

## Links

- **GitHub:** https://github.com/KlementMultiverse/vision-assistant
- **Issues:** https://github.com/KlementMultiverse/vision-assistant/issues
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Research:** [RESEARCH.md](RESEARCH.md)
