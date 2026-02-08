# Vision Assistant - GitHub Issues

> **Purpose:** Issues to create on GitHub for Phase 1 implementation.
> **Created:** 2026-02-07

To create these issues, run `gh auth login` first, then:
```bash
# Create all issues
gh issue create --title "TITLE" --body "BODY" --label "LABELS"
```

---

## Phase 1: Foundation

### Issue 1: Device Registry & Camera State

**Title:** `[Core] Implement Device Registry and Camera State Management`

**Labels:** `enhancement`, `core`, `phase-1`

**Body:**
```markdown
## Summary
Implement the device registry and camera state management as the foundation for multi-camera support.

## Requirements
- [ ] Create `Device` dataclass with type hints
- [ ] Create `DeviceType` enum (camera, microphone, speaker)
- [ ] Create `DeviceStatus` enum (online, offline, error)
- [ ] Create `CameraState` dataclass
- [ ] Implement `DeviceRegistry` class with CRUD operations
- [ ] Add device table to database
- [ ] Add camera_states table to database

## Files to Create
- `src/v2/core/devices.py` - Device models
- `src/v2/core/registry.py` - DeviceRegistry class

## Reference
See ARCHITECTURE.md Section 2 for full specification.

## Acceptance Criteria
- [ ] Can register/unregister devices
- [ ] Can update device status
- [ ] Camera state updates correctly
- [ ] Unit tests pass
```

---

### Issue 2: State Store with Pub/Sub

**Title:** `[Core] Implement Central State Store with Pub/Sub`

**Labels:** `enhancement`, `core`, `phase-1`

**Body:**
```markdown
## Summary
Implement the central state store that manages house state, room states, and person presence with pub/sub notifications.

## Requirements
- [ ] Create `HouseState` dataclass
- [ ] Create `HomeMode` enum (home, away, night, vacation)
- [ ] Create `AlertLevel` enum (normal, elevated, high, critical)
- [ ] Create `RoomState` dataclass
- [ ] Create `PersonPresence` dataclass with `VisitState` enum
- [ ] Implement `StateStore` class with pub/sub
- [ ] Add house_state table (singleton)

## Files to Create
- `src/v2/core/state.py` - State models and StateStore

## Reference
See ARCHITECTURE.md Section 4 for full specification.

## Acceptance Criteria
- [ ] Can update house state
- [ ] Can track person presence
- [ ] Subscribers receive state change notifications
- [ ] Unit tests pass
```

---

### Issue 3: Event Bus Implementation

**Title:** `[Core] Implement Priority-Based Event Bus`

**Labels:** `enhancement`, `core`, `phase-1`

**Body:**
```markdown
## Summary
Implement the event bus with priority queue for event-driven architecture.

## Requirements
- [ ] Create `EventType` enum (all event types)
- [ ] Create `EventPriority` enum (CRITICAL=0, HIGH=10, NORMAL=50, LOW=100)
- [ ] Create `Event` dataclass
- [ ] Implement `EventBus` class with priority queue
- [ ] Support async event handlers
- [ ] Add events table to database

## Files to Create
- `src/v2/core/events.py` - Event models and EventBus

## Reference
See ARCHITECTURE.md Section 5 for full specification.

## Acceptance Criteria
- [ ] Events processed by priority
- [ ] Can subscribe/unsubscribe handlers
- [ ] Async handlers work correctly
- [ ] Unit tests pass
```

---

### Issue 4: Database Schema Migration

**Title:** `[Database] Extend VisionDB with New Tables`

**Labels:** `enhancement`, `database`, `phase-1`

**Body:**
```markdown
## Summary
Extend the existing VisionDB with new tables for devices, state, events, etc.

## New Tables
- [ ] `devices` - All cameras, mics, speakers
- [ ] `camera_states` - Current state per camera
- [ ] `observations` - Enhanced visit sessions
- [ ] `vision_results` - GPT Vision output
- [ ] `events` - All system events
- [ ] `house_state` - Global state (singleton)
- [ ] `conversations` - Agent dialogues
- [ ] `conversation_messages` - Individual messages
- [ ] `zones` - Detection zones per camera

## Migration Strategy
1. Create migration script
2. Keep existing tables (persons, embeddings, visits)
3. Add new tables
4. Add foreign key relationships

## Files to Modify
- `src/v2/storage/schema.py` - Add new tables

## Reference
See ARCHITECTURE.md Section 7 for full SQL schema.

## Acceptance Criteria
- [ ] All tables created
- [ ] Foreign keys work
- [ ] Indexes created
- [ ] Migration is reversible
```

---

### Issue 5: Refactor Pipeline to Event-Driven

**Title:** `[Pipeline] Refactor v2 Pipeline to Emit Events`

**Labels:** `enhancement`, `pipeline`, `phase-1`

**Body:**
```markdown
## Summary
Refactor the existing v2 pipeline to emit events instead of handling everything inline.

## Requirements
- [ ] Integrate EventBus into LivePipeline
- [ ] Emit MOTION_DETECTED on motion
- [ ] Emit PERSON_DETECTED on person detection
- [ ] Emit FACE_RECOGNIZED on face match
- [ ] Emit UNKNOWN_PERSON for unknowns
- [ ] Keep existing UI working

## Files to Modify
- `src/v2/live_pipeline.py`

## Reference
See ARCHITECTURE.md Section 3 for pipeline architecture.

## Acceptance Criteria
- [ ] Events emitted correctly
- [ ] Existing functionality preserved
- [ ] Can subscribe to events
- [ ] Unit tests pass
```

---

## Phase 2: Vision Intelligence

### Issue 6: Vision API Client

**Title:** `[Vision] GPT-4o Vision Client with Structured Output`

**Labels:** `enhancement`, `vision`, `phase-2`

**Body:**
```markdown
## Summary
Implement GPT-4o Vision client that returns structured Pydantic output.

## Requirements
- [ ] Create `VisionResult` Pydantic model
- [ ] Create `PersonDescription` Pydantic model
- [ ] Implement `VisionClient` class
- [ ] Dynamic prompt building with context
- [ ] Retry with exponential backoff
- [ ] Store results in vision_results table

## Files to Create
- `src/v2/vision/client.py` - VisionClient
- `src/v2/vision/schemas.py` - Pydantic models
- `src/v2/vision/prompts.py` - Prompt templates

## Reference
See ARCHITECTURE.md Section 6.2 and RESEARCH.md Section 3.

## Acceptance Criteria
- [ ] Returns structured VisionResult
- [ ] Handles rate limits
- [ ] Retries on failure
- [ ] Logs API calls
```

---

### Issue 7: Trigger Controller

**Title:** `[Agent] Implement Trigger Controller`

**Labels:** `enhancement`, `agent`, `phase-2`

**Body:**
```markdown
## Summary
Implement the trigger controller that decides when to wake the LLM agent.

## Requirements
- [ ] Subscribe to relevant events
- [ ] Handle UNKNOWN_PERSON (CRITICAL priority)
- [ ] Handle PERSON_ARRIVED (HIGH priority)
- [ ] Implement event batching
- [ ] Different behavior for inside/outside zones
- [ ] Heartbeat checks (1min outside, 5min inside)

## Files to Create
- `src/v2/agent/trigger.py` - TriggerController

## Reference
See ARCHITECTURE.md Section 5.3.

## Acceptance Criteria
- [ ] Agent wakes on correct events
- [ ] Priority respected
- [ ] Batching works
- [ ] Heartbeats trigger correctly
```

---

### Issue 8: Main Agent with Tools

**Title:** `[Agent] Implement Main Agent with Deep Agents`

**Labels:** `enhancement`, `agent`, `phase-2`

**Body:**
```markdown
## Summary
Implement the main LLM agent using Deep Agents framework with custom tools.

## Tools to Implement
- [ ] `analyze_scene(camera_id, context)` - GPT-4o Vision
- [ ] `speak(message, speaker_id)` - TTS output
- [ ] `notify_owner(message, priority)` - Telegram notification
- [ ] `get_house_state()` - Current state
- [ ] `update_person(person_id, updates)` - Update person info

## Files to Create
- `src/v2/agent/main.py` - MainAgent class
- `src/v2/agent/tools.py` - Tool definitions

## Reference
See ARCHITECTURE.md Section 6.

## Acceptance Criteria
- [ ] Agent responds to events
- [ ] All tools work
- [ ] Greets family members
- [ ] Handles unknowns appropriately
```

---

## Quick Create Commands

Once authenticated with `gh auth login`:

```bash
# Phase 1
gh issue create --title "[Core] Implement Device Registry and Camera State Management" --label "enhancement,core,phase-1"
gh issue create --title "[Core] Implement Central State Store with Pub/Sub" --label "enhancement,core,phase-1"
gh issue create --title "[Core] Implement Priority-Based Event Bus" --label "enhancement,core,phase-1"
gh issue create --title "[Database] Extend VisionDB with New Tables" --label "enhancement,database,phase-1"
gh issue create --title "[Pipeline] Refactor v2 Pipeline to Emit Events" --label "enhancement,pipeline,phase-1"

# Phase 2
gh issue create --title "[Vision] GPT-4o Vision Client with Structured Output" --label "enhancement,vision,phase-2"
gh issue create --title "[Agent] Implement Trigger Controller" --label "enhancement,agent,phase-2"
gh issue create --title "[Agent] Implement Main Agent with Deep Agents" --label "enhancement,agent,phase-2"
```
