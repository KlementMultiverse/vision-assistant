# Vision Assistant: Complete System Architecture

> **Version:** 3.0
> **Date:** 2026-02-07
> **Status:** Design Complete - Ready for Implementation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Device Layer](#2-device-layer)
3. [Perception Pipeline](#3-perception-pipeline)
4. [State Management](#4-state-management)
5. [Event-Driven Architecture](#5-event-driven-architecture)
6. [Agent Architecture](#6-agent-architecture)
7. [Database Schema](#7-database-schema)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Implementation Phases](#9-implementation-phases)

---

## 1. System Overview

### 1.1 Vision

A complete **Home AI System** that:
- Sees and understands who's home (face recognition + scene understanding)
- Talks to people at the door (voice greetings, Telegram notifications)
- Manages the entire house (multi-camera, multi-room awareness)
- Supports personal assistants for each family member
- Scales from POC (laptop) to production (cloud/on-premise)

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VISION ASSISTANT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        DEVICE LAYER (24/7)                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │ Camera  │  │ Camera  │  │ Camera  │  │   Mic   │  │ Speaker │     │  │
│  │  │ (door)  │  │ (living)│  │(kitchen)│  │         │  │         │     │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │  │
│  └───────┼────────────┼────────────┼────────────┼────────────┼──────────┘  │
│          │            │            │            │            │             │
│          ▼            ▼            ▼            ▼            ▼             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     PERCEPTION LAYER (Local GPU)                      │  │
│  │                                                                       │  │
│  │   Motion Detection → Person Detection → Face Detection → Embedding   │  │
│  │        (OpenCV)         (YOLOv8n)        (InsightFace)    (512-D)    │  │
│  │                                                                       │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        STATE STORE + EVENT BUS                        │  │
│  │                                                                       │  │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │  │
│  │   │ House State │    │ Room States │    │Person States│              │  │
│  │   └─────────────┘    └─────────────┘    └─────────────┘              │  │
│  │                                                                       │  │
│  │   Events: motion | person_detected | face_recognized | unknown_alert │  │
│  │   Priority: CRITICAL(0) | HIGH(10) | NORMAL(50) | LOW(100)           │  │
│  │                                                                       │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     TRIGGER CONTROLLER (Decision)                     │  │
│  │                                                                       │  │
│  │   When to wake LLM:                                                  │  │
│  │   • Unknown person at door → CRITICAL → Wake GPT-4o immediately      │  │
│  │   • Known person arrives → HIGH → Wake with delay (batch if needed)  │  │
│  │   • Motion only → LOW → Update state, no LLM                         │  │
│  │   • Heartbeat (inside: 5min, outside: 1min)                          │  │
│  │                                                                       │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      MAIN AGENT (Deep Agents)                         │  │
│  │                                                                       │  │
│  │   Tools:                                                             │  │
│  │   • analyze_scene(frame, context) → GPT-4o Vision                    │  │
│  │   • speak(message, speaker_id) → TTS                                 │  │
│  │   • listen(mic_id, duration) → STT                                   │  │
│  │   • notify_owner(message, priority) → Telegram                       │  │
│  │   • get_house_state() → Current state                                │  │
│  │   • update_person(id, updates) → Tag/update person                   │  │
│  │                                                                       │  │
│  │   APIs:                                                              │  │
│  │   • Outside/Door: GPT-4o (fast, paid) - instant response             │  │
│  │   • Inside: Gemini/Qwen/NVIDIA (free) - can be slower                │  │
│  │                                                                       │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         OUTPUT LAYER                                  │  │
│  │                                                                       │  │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │  │
│  │   │  Voice   │    │ Display  │    │ Telegram │    │  Logs    │      │  │
│  │   │ (TTS)    │    │  (UI)    │    │  (Bot)   │    │ (Events) │      │  │
│  │   └──────────┘    └──────────┘    └──────────┘    └──────────┘      │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Device Layer

### 2.1 Device Model

Devices are first-class entities with state, capabilities, and lifecycle.

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime

class DeviceType(Enum):
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SPEAKER = "speaker"

class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    INITIALIZING = "initializing"

@dataclass
class Device:
    id: str                          # "cam_front_door", "mic_living", "spk_door"
    type: DeviceType
    name: str                        # Human-friendly name
    location: str                    # "front_door", "living_room", "kitchen"
    zone: str                        # "outside", "inside"
    status: DeviceStatus
    connection_uri: str              # "rtsp://...", "/dev/video0", "hw:0,0"
    capabilities: List[str]          # ["video", "audio", "night_vision", "ptz"]
    last_heartbeat: datetime
    error_message: Optional[str] = None
    config: dict = None              # Device-specific config
```

### 2.2 Camera State

Each camera maintains its own state.

```python
@dataclass
class CameraState:
    camera_id: str
    status: DeviceStatus

    # Detection state
    motion_detected: bool = False
    persons_in_frame: int = 0
    faces_detected: int = 0

    # Current tracking
    active_tracks: List[str] = None     # Track IDs currently in frame
    known_persons: List[int] = None     # Person IDs recognized
    unknown_count: int = 0

    # Frame buffer
    last_frame_time: datetime = None
    frame_quality: float = 0.0          # 0-1, for selecting best frame

    # Stats
    fps: float = 0.0
    processing_latency_ms: float = 0.0

    # Vision API state
    last_vision_call: datetime = None
    pending_vision_request: bool = False
```

### 2.3 Device Registry

```python
class DeviceRegistry:
    """Central registry for all devices."""

    def __init__(self, db: Database):
        self.db = db
        self._devices: Dict[str, Device] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    async def register(self, device: Device) -> None:
        """Register a new device."""
        self._devices[device.id] = device
        await self.db.upsert_device(device)
        await self._notify("device_registered", device)

    async def update_status(self, device_id: str, status: DeviceStatus) -> None:
        """Update device status."""
        if device := self._devices.get(device_id):
            device.status = status
            device.last_heartbeat = datetime.utcnow()
            await self.db.update_device_status(device_id, status)
            await self._notify("status_changed", device)

    def get_cameras(self, zone: Optional[str] = None) -> List[Device]:
        """Get all cameras, optionally filtered by zone."""
        return [
            d for d in self._devices.values()
            if d.type == DeviceType.CAMERA
            and (zone is None or d.zone == zone)
        ]

    def get_by_location(self, location: str) -> List[Device]:
        """Get all devices at a location."""
        return [d for d in self._devices.values() if d.location == location]
```

---

## 3. Perception Pipeline

### 3.1 Pipeline Architecture

Based on Frigate and Double Take patterns: **Process frames efficiently, act on EVENTS.**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DETECTION PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │  CAMERA  │───►│    MOTION    │───►│    PERSON    │───►│  OBJECT TRACKER    │    │
│  │  30 fps  │    │  DETECTOR    │    │  DETECTOR    │    │  (Kalman Filter)   │    │
│  └──────────┘    │   <1ms CPU   │    │  ~10ms GPU   │    │                    │    │
│                  │              │    │              │    │  Assigns track_id  │    │
│                  │  Decision:   │    │  Decision:   │    │  Manages lifecycle │    │
│                  │  - No motion?│    │  - No person?│    │                    │    │
│                  │    → Skip    │    │    → Skip    │    │  Emits EVENTS:     │    │
│                  │              │    │              │    │  - NEW             │    │
│                  │  - Motion?   │    │  - Person?   │    │  - UPDATE          │    │
│                  │    → Next    │    │    → Track   │    │  - END             │    │
│                  └──────────────┘    └──────────────┘    └─────────┬──────────┘    │
│                                                                     │               │
│                                                      ┌──────────────┴──────────────┐│
│                                                      │       EVENT ROUTER          ││
│                                                      │                             ││
│                                                      │  On "NEW" event only:       ││
│                                                      │  ┌─────────────────────────┐││
│                                                      │  │    FACE RECOGNIZER      │││
│                                                      │  │    ~20ms GPU            │││
│                                                      │  │                         │││
│                                                      │  │  - Detect face          │││
│                                                      │  │  - Extract embedding    │││
│                                                      │  │  - Search database      │││
│                                                      │  │  - Retry up to 10x      │││
│                                                      │  │    for best quality     │││
│                                                      │  └─────────────────────────┘││
│                                                      └──────────────┬──────────────┘│
│                                                                     │               │
│                                          ┌──────────────────────────┴──────────────┐│
│                                          │           STATE MACHINE                 ││
│                                          │                                         ││
│                                          │  DETECTING → ACTIVE → STATIONARY → LOST ││
│                                          │                                         ││
│                                          │  - Debounced confirmations              ││
│                                          │  - Timeout-based departures             ││
│                                          │  - Per-object state tracking            ││
│                                          └─────────────────────────────────────────┘│
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Data Models

```python
@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class ObjectState(Enum):
    """State of a tracked object."""
    DETECTING = auto()   # Just appeared, confirming
    ACTIVE = auto()      # Confirmed, actively tracked
    STATIONARY = auto()  # Not moving for N frames
    LOST = auto()        # Not detected for M frames
    ENDED = auto()       # Track terminated


@dataclass
class TrackedObject:
    """A tracked person with full lifecycle."""
    track_id: str
    object_id: Optional[str]
    state: ObjectState = ObjectState.DETECTING
    box: Optional[BoundingBox] = None
    face_name: Optional[str] = None
    face_confidence: float = 0.0
    frames_seen: int = 0
    frames_lost: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
```

---

## 4. State Management

### 4.1 State Hierarchy

```
HouseState (singleton)
    │
    ├── RoomState (per room)
    │       │
    │       └── CameraState (per camera)
    │
    └── PersonPresence (per tracked person)
```

### 4.2 House State

```python
class HomeMode(Enum):
    HOME = "home"           # Someone is home
    AWAY = "away"           # Everyone left
    NIGHT = "night"         # Night mode (sleeping)
    VACATION = "vacation"   # Extended away
    GUEST = "guest"         # Guest mode (limited access)

class AlertLevel(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"   # Unknown person detected
    HIGH = "high"           # Multiple unknowns or suspicious
    CRITICAL = "critical"   # Emergency

@dataclass
class HouseState:
    mode: HomeMode
    alert_level: AlertLevel
    family_home: List[str]          # Names of family members home
    guests_present: int
    unknowns_present: int
    room_occupancy: Dict[str, List[int]]  # room -> person_ids
    active_observations: int
    last_activity: datetime
    last_entry: datetime
    last_exit: datetime

    def anyone_home(self) -> bool:
        return len(self.family_home) > 0 or self.guests_present > 0
```

### 4.3 Person Presence

```python
class VisitState(Enum):
    APPROACHING = "approaching"
    AT_DOOR = "at_door"
    ENTERING = "entering"
    INSIDE = "inside"
    LEAVING = "leaving"
    LEFT = "left"

@dataclass
class PersonPresence:
    person_id: int
    name: str
    group: str                      # "family", "friends", "public"
    role: Optional[str]             # "daughter", "delivery_guy"
    zone: str                       # "outside", "inside"
    current_room: Optional[str]
    current_camera: Optional[str]
    visit_state: VisitState
    entry_time: datetime
    last_seen: datetime
    greeted: bool
    conversation_active: bool
    track_ids: Dict[str, str]       # camera_id -> track_id
    last_embedding: np.ndarray
    last_confidence: float
```

### 4.4 State Store

```python
class StateStore:
    """Central state store with pub/sub for changes."""

    def __init__(self):
        self.house_state = HouseState(...)
        self.room_states: Dict[str, RoomState] = {}
        self.camera_states: Dict[str, CameraState] = {}
        self.person_presence: Dict[int, PersonPresence] = {}
        self._subscribers: Dict[str, List[Callable]] = {}

    async def update_house_state(self, **updates) -> None:
        """Update house state and notify subscribers."""
        for key, value in updates.items():
            setattr(self.house_state, key, value)
        await self._publish("house_state", self.house_state)

    def get_persons_in_zone(self, zone: str) -> List[PersonPresence]:
        """Get all persons in a zone (inside/outside)."""
        return [p for p in self.person_presence.values() if p.zone == zone]
```

---

## 5. Event-Driven Architecture

### 5.1 Event Types

```python
class EventType(Enum):
    # Device events
    DEVICE_ONLINE = "device_online"
    DEVICE_OFFLINE = "device_offline"
    DEVICE_ERROR = "device_error"

    # Perception events
    MOTION_DETECTED = "motion_detected"
    PERSON_DETECTED = "person_detected"
    FACE_RECOGNIZED = "face_recognized"

    # High-level events
    PERSON_ARRIVED = "person_arrived"
    PERSON_LEFT = "person_left"
    UNKNOWN_PERSON = "unknown_person"
    FAMILY_HOME = "family_home"
    HOUSE_EMPTY = "house_empty"

    # Agent events
    GREETING_SENT = "greeting_sent"
    NOTIFICATION_SENT = "notification_sent"

    # System events
    HEARTBEAT = "heartbeat"
    VISION_RESULT = "vision_result"

class EventPriority(Enum):
    CRITICAL = 0    # Unknown at door, emergency
    HIGH = 10       # Person arrived, requires attention
    NORMAL = 50     # Regular events
    LOW = 100       # Logs, metrics, heartbeats

@dataclass
class Event:
    id: str
    type: EventType
    priority: EventPriority
    source: str
    timestamp: datetime
    data: dict
    target: Optional[str] = None
    processed: bool = False
```

### 5.2 Event Bus

```python
class EventBus:
    """Priority-based async event bus."""

    def __init__(self, max_queue_size: int = 1000):
        self._queue: List[PrioritizedEvent] = []
        self._handlers: Dict[EventType, List[Callable]] = {}

    async def publish(self, event: Event) -> None:
        """Publish event to the bus."""
        heapq.heappush(self._queue, PrioritizedEvent(
            priority=event.priority.value,
            timestamp=event.timestamp.timestamp(),
            event=event
        ))

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
```

### 5.3 Trigger Controller

```python
class TriggerController:
    """Decides when to invoke the LLM agent."""

    async def _on_unknown(self, event: Event) -> None:
        """Handle unknown person - CRITICAL priority."""
        zone = self.state.camera_states[event.data["camera_id"]].zone

        if zone == "outside":
            # Outside = immediate response
            await self.agent.handle_unknown_at_door(event)
        else:
            # Inside = batch with other events
            self._pending_events.append(event)
            await self._maybe_trigger()

    async def _on_arrival(self, event: Event) -> None:
        """Handle known person arrival."""
        presence = self.state.person_presence.get(event.data["person_id"])

        if presence and presence.group == "family":
            if not presence.greeted:
                await self.agent.greet_family(presence)
```

---

## 6. Agent Architecture

### 6.1 Main Agent (Deep Agents)

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

class AnalyzeSceneInput(BaseModel):
    camera_id: str = Field(description="Camera to analyze")
    context: dict = Field(description="Known context (persons, state)")

class SpeakInput(BaseModel):
    message: str = Field(description="Message to speak")
    speaker_id: str = Field(default="default")

class NotifyInput(BaseModel):
    message: str = Field(description="Notification message")
    priority: str = Field(default="normal")
    include_image: bool = Field(default=False)

@tool(args_schema=AnalyzeSceneInput)
async def analyze_scene(camera_id: str, context: dict) -> dict:
    """Analyze scene using GPT-4o Vision."""
    frame = await camera_manager.get_best_frame(camera_id)
    prompt = build_vision_prompt(context)
    return await vision_api.analyze(frame, prompt)

@tool(args_schema=SpeakInput)
async def speak(message: str, speaker_id: str = "default") -> str:
    """Speak a message through TTS."""
    await tts_manager.speak(message, speaker_id)
    return f"Spoke: {message}"

@tool(args_schema=NotifyInput)
async def notify_owner(message: str, priority: str = "normal",
                       include_image: bool = False) -> str:
    """Send notification via Telegram."""
    await telegram_bot.send_notification(message, priority, include_image)
    return f"Notified owner: {message}"

@tool
async def get_house_state() -> dict:
    """Get current state of the house."""
    return state_store.house_state.to_dict()

@tool
async def update_person(person_id: int, updates: dict) -> str:
    """Update person information."""
    await db.update_person(person_id, **updates)
    return f"Updated person {person_id}"
```

### 6.2 Vision Output Schema

```python
class PersonDescription(BaseModel):
    position: str           # "left", "center", "right"
    appearance: str         # "man in blue shirt"
    activity: str           # "standing", "walking"
    carrying: List[str]     # ["package", "bag"]
    age_range: str          # "child", "adult", "elderly"

class VisionResult(BaseModel):
    description: str
    person_count: int
    people: List[PersonDescription]
    objects: List[str]
    activities: List[str]
    safety_level: str       # "safe", "caution", "alert"
    suggested_role: Optional[str]
    recommended_action: Optional[str]
```

---

## 7. Database Schema

### 7.1 Entity Relationship Diagram

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

### 7.2 SQL Schema

```sql
-- Devices (cameras, mics, speakers)
CREATE TABLE devices (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('camera', 'microphone', 'speaker')),
    name TEXT NOT NULL,
    location TEXT NOT NULL,
    zone TEXT NOT NULL CHECK (zone IN ('inside', 'outside')),
    status TEXT DEFAULT 'offline',
    connection_uri TEXT,
    capabilities JSONB DEFAULT '[]',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Persons (known and unknown)
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT,
    group_type TEXT DEFAULT 'public',
    role TEXT,
    status TEXT DEFAULT 'active',
    visit_count INTEGER DEFAULT 0,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Face embeddings (512-D vectors)
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding BYTEA NOT NULL,
    confidence REAL,
    lighting TEXT,
    camera_id TEXT,
    captured_at TIMESTAMPTZ DEFAULT NOW()
);

-- Camera states (current state per camera)
CREATE TABLE camera_states (
    camera_id TEXT PRIMARY KEY REFERENCES devices(id),
    status TEXT DEFAULT 'offline',
    motion_detected BOOLEAN DEFAULT FALSE,
    persons_in_frame INTEGER DEFAULT 0,
    faces_detected INTEGER DEFAULT 0,
    active_tracks JSONB DEFAULT '[]',
    known_persons JSONB DEFAULT '[]',
    unknown_count INTEGER DEFAULT 0,
    last_frame_time TIMESTAMPTZ,
    fps REAL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Observations (visit sessions with context)
CREATE TABLE observations (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(id),
    camera_id TEXT REFERENCES devices(id),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    state TEXT DEFAULT 'active',
    frame_count INTEGER DEFAULT 0,
    avg_confidence REAL,
    best_frame_path TEXT,
    vision_context JSONB,
    metadata JSONB DEFAULT '{}'
);

-- Vision API results
CREATE TABLE vision_results (
    id SERIAL PRIMARY KEY,
    observation_id INTEGER REFERENCES observations(id),
    camera_id TEXT REFERENCES devices(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    description TEXT,
    people_detected JSONB,
    objects JSONB,
    activities JSONB,
    safety_level TEXT,
    suggested_role TEXT,
    raw_response JSONB,
    model TEXT,
    latency_ms INTEGER
);

-- Events (all system events)
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL,
    priority INTEGER NOT NULL,
    source TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data JSONB DEFAULT '{}',
    processed BOOLEAN DEFAULT FALSE
);

-- House state (singleton)
CREATE TABLE house_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    mode TEXT DEFAULT 'away',
    alert_level TEXT DEFAULT 'normal',
    family_home JSONB DEFAULT '[]',
    guests_count INTEGER DEFAULT 0,
    unknowns_count INTEGER DEFAULT 0,
    room_occupancy JSONB DEFAULT '{}',
    active_observations INTEGER DEFAULT 0,
    last_activity TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations (agent dialogues)
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(id),
    observation_id INTEGER REFERENCES observations(id),
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    status TEXT DEFAULT 'active',
    summary TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Conversation messages
CREATE TABLE conversation_messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('agent', 'user', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    audio_path TEXT
);

-- Detection zones per camera
CREATE TABLE zones (
    id SERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES devices(id),
    name TEXT NOT NULL,
    zone_type TEXT DEFAULT 'detection',
    coordinates JSONB NOT NULL,
    triggers JSONB DEFAULT '["person"]',
    enabled BOOLEAN DEFAULT TRUE,
    UNIQUE(camera_id, name)
);
```

---

## 8. Deployment Architecture

### 8.1 Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: vision_assistant
      POSTGRES_USER: vision
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  perception:
    build: ./services/perception
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://vision:${DB_PASSWORD}@postgres:5432/vision_assistant
    depends_on:
      - redis
      - postgres

  agent:
    build: ./services/agent
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://vision:${DB_PASSWORD}@postgres:5432/vision_assistant
    depends_on:
      - redis
      - postgres
      - perception

  api:
    build: ./services/api
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres

  telegram:
    build: ./services/telegram
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - api

volumes:
  postgres_data:
  redis_data:
```

### 8.2 Service Responsibilities

| Service | Responsibility | GPU | State |
|---------|---------------|-----|-------|
| **perception** | Camera input, detection, embeddings | Yes | Stateless |
| **agent** | LLM reasoning, decisions, actions | No | Stateless |
| **api** | REST API, WebSocket for UI | No | Stateless |
| **telegram** | Telegram bot interface | No | Stateless |
| **postgres** | Persistent storage | No | Stateful |
| **redis** | Events, cache, pub/sub | No | Stateful |

---

## 9. Implementation Phases

### Phase 1: Foundation (Current Sprint)

**Goal:** Modular architecture with existing v2 functionality.

| Task | Priority | Status |
|------|----------|--------|
| Device registry + camera state | High | TODO |
| State store with pub/sub | High | TODO |
| Event bus implementation | High | TODO |
| Database schema migration | High | TODO |
| Refactor v2 pipeline to use events | Medium | TODO |

### Phase 2: Vision Intelligence

**Goal:** Add GPT-4o scene understanding.

| Task | Priority | Status |
|------|----------|--------|
| Vision API client (structured output) | High | TODO |
| Trigger controller | High | TODO |
| Main agent with tools | High | TODO |
| Vision prompts + schemas | Medium | TODO |

### Phase 3: Multi-Camera

**Goal:** Support multiple cameras with unified tracking.

| Task | Priority | Status |
|------|----------|--------|
| Multi-camera manager | High | TODO |
| Cross-camera re-identification | High | TODO |
| Room/zone state management | Medium | TODO |

### Phase 4: Conversation

**Goal:** Two-way voice interaction.

| Task | Priority | Status |
|------|----------|--------|
| STT integration (Whisper) | High | TODO |
| TTS with speaker routing | High | TODO |
| Conversation state machine | Medium | TODO |

### Phase 5: Telegram Bot

**Goal:** Remote control and notifications.

| Task | Priority | Status |
|------|----------|--------|
| Telegram bot setup | High | TODO |
| Notification routing | High | TODO |
| HITL decision interface | Medium | TODO |

---

## Appendix A: Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Web Framework | FastAPI | 0.100+ |
| LLM Framework | Deep Agents / LangGraph | Latest |
| Database | PostgreSQL + TimescaleDB | 15 + 2.x |
| Cache/Events | Redis | 7.x |
| Face Detection | InsightFace | 0.7+ |
| Person Detection | YOLOv8n | Latest |
| Re-ID | OSNet_x1_0 | Latest |
| TTS | pyttsx3 / Edge TTS | Latest |
| STT | Whisper | Latest |
| Vision API | OpenAI GPT-4o | Latest |
| Containerization | Docker + Compose | 24.x |
| GPU | NVIDIA CUDA | 12.x |

---

## Appendix B: Open Source References

### Person Detection & Tracking
- [Frigate NVR](https://github.com/blakeblackshear/frigate) - Full NVR with detection
- [DeepSORT](https://github.com/nwojke/deep_sort) - Multi-object tracking
- [torchreid](https://github.com/KaiyangZhou/deep-person-reid) - Person re-identification

### Face Recognition
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection + embeddings
- [CompreFace](https://github.com/exadel-inc/CompreFace) - Face recognition service

### LLM & Agents
- [Deep Agents](https://github.com/langchain-ai/deepagents) - Production agent framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent graphs

### Smart Home
- [Home Assistant](https://github.com/home-assistant/core) - Home automation
- [Frigate Integration](https://github.com/blakeblackshear/frigate-hass-integration)

---

*Document generated by Vision Assistant Architecture System*
*Last Updated: 2026-02-07*
