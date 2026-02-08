# Vision Assistant - Production Detection Pipeline Architecture

> **Version:** 2.0
> **Date:** 2026-02-07
> **Author:** ARCHITECT Agent
> **Status:** Design Complete - Ready for Implementation

---

## EXECUTIVE SUMMARY

This document defines the production-grade detection pipeline based on lessons learned from:
- **Frigate**: Two-stage detection (Motion CPU → Object GPU), event lifecycle, object tracking
- **Double Take**: Face recognition on NEW events only, retry for quality, stop on first match

**Key Insight:** Process frames efficiently, but act on EVENTS.

---

## 1. PIPELINE ARCHITECTURE

### 1.1 High-Level Flow

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
│                                          │  EMPTY → DETECTING → PRESENT → LEAVING ││
│                                          │                                         ││
│                                          │  - Debounced confirmations              ││
│                                          │  - Timeout-based departures             ││
│                                          │  - Per-object state tracking            ││
│                                          └──────────────────────────┬──────────────┘│
│                                                                     │               │
│                                          ┌──────────────────────────┴──────────────┐│
│                                          │           ACTION DISPATCHER             ││
│                                          │                                         ││
│                                          │  ┌─────────┐ ┌─────────┐ ┌───────────┐ ││
│                                          │  │  VOICE  │ │ NOTIFY  │ │   LOG     │ ││
│                                          │  │  (TTS)  │ │ (MQTT)  │ │ (SQLite)  │ ││
│                                          │  └─────────┘ └─────────┘ └───────────┘ ││
│                                          └──────────────────────────────────────────┘│
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Detailed Decision Tree

```
Frame arrives (30 fps)
│
├─ MOTION DETECTION (<1ms CPU)
│   │
│   ├─ NO motion AND state=EMPTY?
│   │   └─ SKIP frame (save GPU)
│   │      └─ Only update leaving timeout
│   │
│   └─ Motion detected?
│       └─ Continue to PERSON detection
│
├─ PERSON DETECTION (~10ms GPU)
│   │
│   ├─ NO person detected?
│   │   └─ Update tracker with empty detections
│   │      └─ May trigger LEAVING → END event
│   │
│   └─ Person(s) detected?
│       └─ Pass to OBJECT TRACKER
│
├─ OBJECT TRACKER (Kalman filter, <1ms)
│   │
│   ├─ Match detections to existing tracks (IoU)
│   │   ├─ Match found → UPDATE event
│   │   └─ No match → NEW track → NEW event
│   │
│   ├─ Unmatched tracks?
│   │   └─ Increment lost_frames
│   │       └─ lost_frames > threshold? → END event
│   │
│   └─ Emit events to EVENT ROUTER
│
├─ EVENT ROUTER
│   │
│   ├─ NEW event?
│   │   └─ Trigger FACE RECOGNITION
│   │       └─ Retry up to 10 frames for best snapshot
│   │       └─ Stop on first confident match
│   │
│   ├─ UPDATE event?
│   │   └─ Update position, check zones
│   │   └─ Check if stationary (same position N frames)
│   │
│   └─ END event?
│       └─ Finalize event record
│       └─ Store to database
│
└─ STATE MACHINE (per-object)
    │
    ├─ Transition based on confirmation frames
    ├─ Emit user-facing events (PERSON_ENTERED, etc.)
    └─ Trigger ACTIONS
```

---

## 2. COMPONENT INTERFACES

### 2.1 Core Data Models

```python
# src/v2/core/models.py
"""
Core data models for the detection pipeline.
All immutable (frozen dataclasses) for thread safety.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import numpy as np


# =============================================================================
# BOUNDING BOXES
# =============================================================================

@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


# =============================================================================
# DETECTION RESULTS
# =============================================================================

@dataclass(frozen=True)
class MotionResult:
    """Result from motion detection stage."""
    detected: bool
    magnitude: float  # 0.0 to 1.0 (fraction of frame with motion)
    regions: Tuple[BoundingBox, ...] = field(default_factory=tuple)

    # Timing
    timestamp: float = 0.0
    process_time_ms: float = 0.0


@dataclass(frozen=True)
class PersonDetection:
    """Single person detection from YOLO."""
    box: BoundingBox
    confidence: float  # 0.0 to 1.0

    # Optional: keypoints for pose
    keypoints: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PersonResult:
    """Result from person detection stage."""
    detections: Tuple[PersonDetection, ...] = field(default_factory=tuple)

    # Timing
    timestamp: float = 0.0
    process_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class FaceDetection:
    """Single face detection with recognition results."""
    box: BoundingBox
    confidence: float  # Detection confidence 0.0 to 1.0

    # Recognition results (None if not recognized)
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    embedding: Optional[np.ndarray] = None

    # Quality metrics (for retry logic)
    blur_score: float = 0.0  # Higher = sharper
    face_angle: float = 0.0  # Degrees from frontal

    @property
    def is_known(self) -> bool:
        return self.identity is not None and self.identity != "Unknown"


@dataclass(frozen=True)
class FaceResult:
    """Result from face recognition stage."""
    faces: Tuple[FaceDetection, ...] = field(default_factory=tuple)

    # Timing
    timestamp: float = 0.0
    process_time_ms: float = 0.0

    # Retry tracking
    attempt_number: int = 1
    best_quality_score: float = 0.0


# =============================================================================
# OBJECT TRACKING
# =============================================================================

class ObjectState(Enum):
    """State of a tracked object."""
    DETECTING = auto()   # Just appeared, confirming
    ACTIVE = auto()      # Confirmed, actively tracked
    STATIONARY = auto()  # Not moving for N frames
    LOST = auto()        # Not detected for M frames
    ENDED = auto()       # Track terminated


@dataclass
class TrackedObject:
    """
    A tracked person with full lifecycle.

    Mutable: state changes over time.
    Thread-safe: use with lock for concurrent access.
    """
    # Identity
    track_id: str              # Unique ID for this tracking session
    object_id: Optional[str]   # Persistent ID if face recognized

    # Current state
    state: ObjectState = ObjectState.DETECTING
    box: Optional[BoundingBox] = None

    # Recognition
    face_name: Optional[str] = None
    face_confidence: float = 0.0
    face_attempts: int = 0
    best_snapshot: Optional[np.ndarray] = None

    # Tracking metrics
    frames_seen: int = 0
    frames_lost: int = 0
    frames_stationary: int = 0

    # Timestamps
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_position: Optional[Tuple[int, int]] = None

    # Zones (if using zone detection)
    current_zones: List[str] = field(default_factory=list)

    # Kalman filter state (for position prediction)
    kalman_state: Optional[np.ndarray] = None

    def update_position(self, box: BoundingBox, timestamp: float):
        """Update object position, check for stationary."""
        current_center = box.center

        if self.last_position:
            # Check if stationary (moved less than threshold)
            dx = abs(current_center[0] - self.last_position[0])
            dy = abs(current_center[1] - self.last_position[1])
            movement = (dx * dx + dy * dy) ** 0.5

            if movement < 10:  # pixels
                self.frames_stationary += 1
                if self.frames_stationary > 30:  # ~1 second at 30fps
                    self.state = ObjectState.STATIONARY
            else:
                self.frames_stationary = 0
                if self.state == ObjectState.STATIONARY:
                    self.state = ObjectState.ACTIVE

        self.box = box
        self.last_position = current_center
        self.last_seen = timestamp
        self.frames_seen += 1
        self.frames_lost = 0


# =============================================================================
# EVENTS
# =============================================================================

class EventLifecycle(Enum):
    """Event lifecycle state (like Frigate)."""
    NEW = "new"         # Just started
    UPDATE = "update"   # Still ongoing
    END = "end"         # Finished


class EventLabel(Enum):
    """What was detected."""
    PERSON = "person"
    FACE = "face"
    MOTION = "motion"


@dataclass
class DetectionEvent:
    """
    A detection event with full context.

    Events are emitted at lifecycle transitions:
    - NEW: When object first confirmed (after debounce)
    - UPDATE: Periodically while object present (configurable interval)
    - END: When object leaves (after timeout)
    """
    # Event identity
    id: str                    # Unique event ID
    camera: str                # Camera name/ID
    label: EventLabel          # What was detected
    lifecycle: EventLifecycle  # NEW/UPDATE/END

    # Timing
    start_time: float          # When event started
    end_time: Optional[float]  # When event ended (None if ongoing)

    # Detection details
    box: BoundingBox           # Current/last bounding box
    area: int                  # Box area in pixels

    # Location
    current_zones: List[str] = field(default_factory=list)
    entered_zones: List[str] = field(default_factory=list)

    # Recognition (for PERSON events)
    face_name: Optional[str] = None
    face_confidence: float = 0.0

    # Tracking
    track_id: str = ""         # Internal tracking ID
    frames_count: int = 0      # Total frames in event

    # Snapshot
    snapshot: Optional[np.ndarray] = None

    # Extra data
    attributes: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Event duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time


# =============================================================================
# PIPELINE FRAME
# =============================================================================

@dataclass
class PipelineFrame:
    """
    Complete frame context passed through pipeline.

    Accumulates results from each stage.
    """
    # Input
    frame: np.ndarray
    timestamp: float
    frame_number: int
    camera_id: str = "default"

    # Results (filled by each stage)
    motion: Optional[MotionResult] = None
    persons: Optional[PersonResult] = None
    faces: Optional[FaceResult] = None

    # Events generated
    events: List[DetectionEvent] = field(default_factory=list)

    # Processing flags
    skip_person_detection: bool = False
    skip_face_detection: bool = False

    # Timing
    total_process_time_ms: float = 0.0
```

### 2.2 Protocols (Interfaces)

```python
# src/v2/core/protocols.py
"""
Protocol definitions for swappable components.
Using Python's Protocol for structural subtyping.
"""

from typing import Protocol, runtime_checkable, Optional, Callable, List
import numpy as np

from .models import (
    MotionResult, PersonResult, FaceResult,
    DetectionEvent, PipelineFrame, TrackedObject
)


@runtime_checkable
class MotionDetector(Protocol):
    """Interface for motion detection."""

    def detect(self, frame: np.ndarray) -> MotionResult:
        """Detect motion in frame. Must be <1ms."""
        ...

    def reset(self) -> None:
        """Reset detector state (e.g., background model)."""
        ...

    @property
    def is_calibrating(self) -> bool:
        """True if still building background model."""
        ...


@runtime_checkable
class PersonDetector(Protocol):
    """Interface for person detection."""

    def detect(self, frame: np.ndarray) -> PersonResult:
        """Detect persons in frame. Target ~10ms."""
        ...

    def detect_in_region(
        self,
        frame: np.ndarray,
        region: tuple  # (x, y, w, h)
    ) -> PersonResult:
        """Detect persons in specific region (optimization)."""
        ...


@runtime_checkable
class FaceRecognizer(Protocol):
    """Interface for face detection and recognition."""

    def detect_and_recognize(self, frame: np.ndarray) -> FaceResult:
        """Detect faces and attempt recognition."""
        ...

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding for database storage."""
        ...

    def register_face(self, frame: np.ndarray, name: str) -> bool:
        """Register a new face in the database."""
        ...

    @property
    def database_count(self) -> int:
        """Number of registered faces."""
        ...


@runtime_checkable
class ObjectTracker(Protocol):
    """Interface for multi-object tracking."""

    def update(self, detections: PersonResult, timestamp: float) -> List[TrackedObject]:
        """Update tracks with new detections. Returns all active tracks."""
        ...

    def get_track(self, track_id: str) -> Optional[TrackedObject]:
        """Get specific track by ID."""
        ...

    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active (non-ended) tracks."""
        ...

    def get_events(self) -> List[DetectionEvent]:
        """Get and clear pending events."""
        ...


@runtime_checkable
class EventHandler(Protocol):
    """Interface for handling detection events."""

    def on_event(self, event: DetectionEvent) -> None:
        """Handle a detection event."""
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Interface for executing actions in response to events."""

    def execute(self, action: str, **params) -> bool:
        """Execute an action. Returns True on success."""
        ...

    def speak(self, text: str) -> None:
        """Speak text via TTS."""
        ...

    def notify(self, message: str, priority: str = "normal") -> None:
        """Send notification."""
        ...


# =============================================================================
# PIPELINE PROTOCOL
# =============================================================================

class Pipeline(Protocol):
    """Interface for the main detection pipeline."""

    def process_frame(self, frame: np.ndarray) -> PipelineFrame:
        """Process single frame through all stages."""
        ...

    def register_event_handler(self, handler: EventHandler) -> None:
        """Register handler for events."""
        ...

    def start(self) -> None:
        """Start pipeline processing."""
        ...

    def stop(self) -> None:
        """Stop pipeline processing."""
        ...
```

### 2.3 Configuration

```python
# src/v2/core/config.py
"""
Pipeline configuration with validation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class MotionConfig:
    """Motion detection configuration."""
    threshold: int = 25           # Pixel difference threshold (1-255)
    contour_area: int = 10        # Min contour area (scaled)
    frame_alpha: float = 0.01     # Background update rate
    delta_alpha: float = 0.2      # Motion persistence rate
    frame_height: int = 100       # Scaled frame height for processing
    calibration_frames: int = 30  # Frames before ready
    improve_contrast: bool = False

    def __post_init__(self):
        if not 1 <= self.threshold <= 255:
            raise ValueError(f"threshold must be 1-255, got {self.threshold}")


@dataclass(frozen=True)
class PersonConfig:
    """Person detection configuration."""
    model: str = "yolov8n.pt"     # Model file
    confidence: float = 0.5       # Detection threshold
    iou_threshold: float = 0.45   # NMS threshold
    device: str = "cuda"          # cuda or cpu
    max_detections: int = 10      # Max persons per frame


@dataclass(frozen=True)
class FaceConfig:
    """Face recognition configuration."""
    detection_threshold: float = 0.5
    recognition_threshold: float = 0.6
    max_retries: int = 10         # Retries for best snapshot
    min_face_size: int = 80       # Min face size in pixels
    database_path: str = "faces.npz"


@dataclass(frozen=True)
class TrackerConfig:
    """Object tracking configuration."""
    # Matching
    iou_threshold: float = 0.3    # Min IoU to match detection to track

    # Lifecycle
    confirm_frames: int = 3       # Frames to confirm NEW
    lost_frames_max: int = 30     # Frames before END (~1 sec at 30fps)

    # Stationary detection
    movement_threshold: int = 10  # Pixels
    stationary_frames: int = 30   # Frames to mark stationary

    # Event emission
    update_interval: float = 1.0  # Seconds between UPDATE events


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration."""
    motion: MotionConfig = field(default_factory=MotionConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # Pipeline behavior
    face_on_entry_only: bool = True    # Only run face on NEW events
    skip_frames_on_empty: int = 5      # Skip N frames when no motion

    # Camera
    camera_id: int = 0
    camera_name: str = "front_door"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary (e.g., YAML file)."""
        return cls(
            motion=MotionConfig(**data.get("motion", {})),
            person=PersonConfig(**data.get("person", {})),
            face=FaceConfig(**data.get("face", {})),
            tracker=TrackerConfig(**data.get("tracker", {})),
            face_on_entry_only=data.get("face_on_entry_only", True),
            skip_frames_on_empty=data.get("skip_frames_on_empty", 5),
            camera_id=data.get("camera_id", 0),
            camera_name=data.get("camera_name", "front_door"),
        )
```

---

## 3. STATE MACHINE SPECIFICATION

### 3.1 State Diagram

```
                            ┌─────────────────────────────────────────────┐
                            │              OBJECT LIFECYCLE               │
                            └─────────────────────────────────────────────┘

                                          Person detected
                                               │
                                               ▼
┌─────────────────┐   confirm_frames met   ┌─────────────────┐
│    DETECTING    │ ─────────────────────► │     ACTIVE      │
│                 │                         │                 │
│  - Counting     │                         │  - Tracking     │
│    detections   │                         │  - Face recog   │
│  - Not yet      │                         │  - Zone check   │
│    confirmed    │                         │                 │
└────────┬────────┘                         └────────┬────────┘
         │                                           │
         │ Detection lost                            │ No movement
         │ before confirm                            │ for N frames
         │                                           │
         ▼                                           ▼
┌─────────────────┐                         ┌─────────────────┐
│     (discard)   │                         │   STATIONARY    │
│                 │                         │                 │
│  - False        │                         │  - Still there  │
│    positive     │                         │  - Not moving   │
│                 │                         │  - May reduce   │
└─────────────────┘                         │    processing   │
                                            └────────┬────────┘
                                                     │
         ┌───────────────────────────────────────────┤ Movement
         │                                           │ detected
         │                                           │
         │                                           ▼
         │                                  ┌─────────────────┐
         │                                  │     ACTIVE      │ (back to active)
         │                                  └─────────────────┘
         │
         │  Detection lost
         │  (person leaves frame)
         │
         ▼
┌─────────────────┐   lost_frames_max met   ┌─────────────────┐
│      LOST       │ ──────────────────────► │      ENDED      │
│                 │                         │                 │
│  - Temporarily  │                         │  - Track done   │
│    not seen     │                         │  - Emit END     │
│  - Countdown    │                         │  - Log event    │
│    to END       │                         │  - Cleanup      │
│                 │                         │                 │
└────────┬────────┘                         └─────────────────┘
         │
         │ Detection
         │ reappears
         │
         ▼
┌─────────────────┐
│     ACTIVE      │ (recovered)
└─────────────────┘
```

### 3.2 State Transition Table

| Current State | Condition | Next State | Action |
|---------------|-----------|------------|--------|
| `DETECTING` | `frames_seen >= confirm_frames` | `ACTIVE` | Emit NEW event, trigger face recognition |
| `DETECTING` | `frames_lost > 0` | *(discard)* | Delete track |
| `ACTIVE` | `movement < threshold` for N frames | `STATIONARY` | Mark as stationary |
| `ACTIVE` | `frames_lost > 0` | `LOST` | Start lost countdown |
| `STATIONARY` | `movement >= threshold` | `ACTIVE` | Resume normal tracking |
| `STATIONARY` | `frames_lost > 0` | `LOST` | Start lost countdown |
| `LOST` | Detection reappears (IoU match) | `ACTIVE` | Reset lost counter |
| `LOST` | `frames_lost >= lost_frames_max` | `ENDED` | Emit END event, cleanup |

### 3.3 Timing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confirm_frames` | 3 | Frames before DETECTING → ACTIVE |
| `lost_frames_max` | 30 | Frames before LOST → ENDED (~1s at 30fps) |
| `stationary_frames` | 30 | Frames before ACTIVE → STATIONARY |
| `movement_threshold` | 10 | Pixels of center movement to detect motion |
| `iou_threshold` | 0.3 | Min IoU to match detection to existing track |

---

## 4. EVENT SCHEMA

### 4.1 Event Structure

```python
@dataclass
class DetectionEvent:
    """Complete event record for storage and processing."""

    # === IDENTITY ===
    id: str                    # UUID, e.g., "evt_abc123"
    camera: str                # Camera name, e.g., "front_door"
    label: str                 # "person", "face", "motion"

    # === LIFECYCLE ===
    state: str                 # "new", "update", "end"

    # === TIMING ===
    start_time: float          # Unix timestamp when event started
    end_time: Optional[float]  # Unix timestamp when ended (None if ongoing)
    frame_count: int           # Total frames in this event

    # === DETECTION ===
    box: dict                  # {"x1": int, "y1": int, "x2": int, "y2": int}
    area: int                  # Bounding box area in pixels
    confidence: float          # Detection confidence

    # === LOCATION ===
    current_zones: List[str]   # Zones object is currently in
    entered_zones: List[str]   # All zones entered during event

    # === RECOGNITION (person events) ===
    face_name: Optional[str]   # Name if recognized, None otherwise
    face_confidence: float     # Recognition confidence

    # === TRACKING ===
    track_id: str              # Internal tracking ID
    is_stationary: bool        # True if object not moving

    # === MEDIA ===
    snapshot_path: Optional[str]  # Path to best snapshot
    thumbnail_path: Optional[str] # Path to thumbnail

    # === METADATA ===
    attributes: dict           # Extra attributes, e.g., {"carrying": "package"}
```

### 4.2 Event Examples

**NEW Event (Person Enters):**
```json
{
  "id": "evt_1707321600_001",
  "camera": "front_door",
  "label": "person",
  "state": "new",
  "start_time": 1707321600.123,
  "end_time": null,
  "frame_count": 3,
  "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400},
  "area": 70000,
  "confidence": 0.92,
  "current_zones": ["entrance"],
  "entered_zones": ["entrance"],
  "face_name": null,
  "face_confidence": 0.0,
  "track_id": "trk_abc123",
  "is_stationary": false,
  "snapshot_path": null,
  "thumbnail_path": null,
  "attributes": {}
}
```

**UPDATE Event (Face Recognized):**
```json
{
  "id": "evt_1707321600_001",
  "camera": "front_door",
  "label": "person",
  "state": "update",
  "start_time": 1707321600.123,
  "end_time": null,
  "frame_count": 15,
  "box": {"x1": 150, "y1": 80, "x2": 350, "y2": 450},
  "area": 74000,
  "confidence": 0.95,
  "current_zones": ["entrance", "doorstep"],
  "entered_zones": ["entrance", "doorstep"],
  "face_name": "John",
  "face_confidence": 0.87,
  "track_id": "trk_abc123",
  "is_stationary": false,
  "snapshot_path": "/snapshots/evt_1707321600_001.jpg",
  "thumbnail_path": "/thumbnails/evt_1707321600_001.jpg",
  "attributes": {"greeted": true}
}
```

**END Event (Person Left):**
```json
{
  "id": "evt_1707321600_001",
  "camera": "front_door",
  "label": "person",
  "state": "end",
  "start_time": 1707321600.123,
  "end_time": 1707321645.678,
  "frame_count": 1366,
  "box": {"x1": 500, "y1": 100, "x2": 640, "y2": 480},
  "area": 53200,
  "confidence": 0.88,
  "current_zones": [],
  "entered_zones": ["entrance", "doorstep"],
  "face_name": "John",
  "face_confidence": 0.87,
  "track_id": "trk_abc123",
  "is_stationary": false,
  "snapshot_path": "/snapshots/evt_1707321600_001.jpg",
  "thumbnail_path": "/thumbnails/evt_1707321600_001.jpg",
  "attributes": {"greeted": true, "duration": 45.555}
}
```

---

## 5. EDGE CASE DECISIONS

### 5.1 Motion Detection Edge Cases

| Edge Case | What Could Go Wrong | How to Handle | Default Behavior |
|-----------|---------------------|---------------|------------------|
| **Lighting change** | Entire frame "moves" | Check if magnitude > 0.5 AND uniform across frame → ignore | Skip if magnitude > 0.8 |
| **Camera shake** | Brief motion everywhere | Require motion for 2+ consecutive frames | 2-frame debounce |
| **Slow movement** | Below threshold | Lower threshold OR use accumulated delta | Use accumulated delta (Frigate pattern) |
| **Shadows** | False motion regions | Ignore low-contrast regions | Filter by contrast ratio |
| **Camera startup** | No background model | Require N frames before detecting | 30 frame calibration |
| **Scene change** | Background invalidated | Reset background model | Manual reset or auto-detect |

### 5.2 Person Detection Edge Cases

| Edge Case | What Could Go Wrong | How to Handle | Default Behavior |
|-----------|---------------------|---------------|------------------|
| **Partial visibility** | Low confidence detection | Accept if confidence > 0.3 AND previous track exists | Lower threshold for tracked objects |
| **Multiple people** | Track ID mixup | Use IoU matching + appearance features | IoU > 0.3 required |
| **Fast movement** | Detection gaps | Predict position with Kalman filter | Allow 3 frames lost |
| **Occlusion** | Person behind object | Maintain track with prediction | Keep track for 30 frames |
| **Far away person** | Small box, low confidence | Minimum box size filter | Ignore if area < 1000px |
| **False positive** | Non-person detected | Require confirm_frames before acting | 3 frame confirmation |

### 5.3 Face Recognition Edge Cases

| Edge Case | What Could Go Wrong | How to Handle | Default Behavior |
|-----------|---------------------|---------------|------------------|
| **Face not visible** | No face detected | Retry on subsequent frames | Retry up to 10 times |
| **Low quality face** | Poor recognition | Score quality, keep best | Track blur_score, keep best |
| **Side profile** | Recognition fails | Accept if angle < 45 degrees | Frontal ± 30 degrees |
| **Multiple faces** | Match to wrong person | Use person bbox to associate | Face must overlap person box |
| **New person** | Not in database | Assign temporary ID | "Unknown_001", "Unknown_002" |
| **Similar faces** | Wrong match | Require confidence > 0.6 | Threshold at 0.6 |
| **Lighting issues** | Embedding quality | Normalize image | Histogram equalization |

### 5.4 Tracking Edge Cases

| Edge Case | What Could Go Wrong | How to Handle | Default Behavior |
|-----------|---------------------|---------------|------------------|
| **ID switch** | Two people swap IDs | Use appearance embedding | Re-run face if uncertainty |
| **Enter and exit** | Same person gets new ID | Check face embedding match | Merge if same face |
| **Crowd** | Many overlapping boxes | Increase IoU threshold | IoU > 0.5 in crowds |
| **Re-entry** | Person leaves and returns | Compare face embeddings | New event if > 5 min gap |
| **Stationary too long** | False stationary | Check for micro-movements | 50px movement window |

### 5.5 Event Edge Cases

| Edge Case | What Could Go Wrong | How to Handle | Default Behavior |
|-----------|---------------------|---------------|------------------|
| **Very short visit** | Event too brief | Minimum event duration | Ignore if < 0.5 seconds |
| **Very long event** | Memory issues | Periodic snapshot updates | Update snapshot every 10s |
| **Event during action** | TTS interrupted | Queue actions | Complete current action |
| **Multiple events** | Greeting confusion | Priority queue | Greet most recent first |
| **Database full** | Storage exhausted | Cleanup old events | Keep 7 days, alert at 80% |

---

## 6. FILE STRUCTURE

```
src/v2/
├── __init__.py
│
├── core/                           # Core abstractions
│   ├── __init__.py
│   ├── models.py                   # Data classes: BoundingBox, MotionResult, etc.
│   ├── protocols.py                # Interfaces: MotionDetector, PersonDetector, etc.
│   ├── config.py                   # Configuration: PipelineConfig, etc.
│   ├── events.py                   # Event definitions and bus
│   └── exceptions.py               # Custom exceptions
│
├── perception/                     # Detection components
│   ├── __init__.py
│   │
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── base.py                 # SimpleMotionDetector (POC)
│   │   └── frigate.py              # FrigateMotionDetector (Production)
│   │
│   ├── person/
│   │   ├── __init__.py
│   │   ├── base.py                 # SimplePersonDetector (POC)
│   │   └── yolo.py                 # YOLOPersonDetector (Production)
│   │
│   └── face/
│       ├── __init__.py
│       ├── base.py                 # SimpleFaceDetector (POC)
│       ├── insightface.py          # InsightFaceRecognizer (Production)
│       └── database.py             # Face embedding database
│
├── tracking/                       # Object tracking
│   ├── __init__.py
│   ├── tracker.py                  # MultiObjectTracker with Kalman filter
│   ├── kalman.py                   # Kalman filter implementation
│   └── state_machine.py            # ObjectStateMachine
│
├── actions/                        # Action execution
│   ├── __init__.py
│   ├── voice.py                    # TTS output
│   ├── notifications.py            # MQTT/push notifications
│   └── logging.py                  # Event logging to SQLite
│
├── pipeline.py                     # Main DetectionPipeline orchestrator
│
└── main.py                         # Entry point
```

### 6.1 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `core/models.py` | Immutable data structures passed between components |
| `core/protocols.py` | Interfaces that components must implement |
| `core/config.py` | Configuration loading, validation, defaults |
| `core/events.py` | Event bus for decoupled communication |
| `perception/motion/` | CPU-based motion detection (<1ms) |
| `perception/person/` | GPU-based person detection (~10ms) |
| `perception/face/` | Face detection + recognition (~20ms) |
| `tracking/tracker.py` | Multi-object tracking with IoU matching |
| `tracking/state_machine.py` | Per-object lifecycle state machine |
| `actions/` | Side effects (speak, notify, log) |
| `pipeline.py` | Orchestrates all components |

---

## 7. IMPLEMENTATION NOTES

### 7.1 Performance Targets

| Stage | Target Time | GPU/CPU | Priority |
|-------|-------------|---------|----------|
| Motion Detection | <1ms | CPU | Always run |
| Person Detection | ~10ms | GPU | Skip if no motion |
| Object Tracking | <1ms | CPU | After person detection |
| Face Recognition | ~20ms | GPU | Only on NEW events |
| Action Execution | <50ms | CPU | After event |

### 7.2 Memory Management

- **Frame buffer:** Keep last 10 frames for snapshot selection
- **Track buffer:** Max 50 concurrent tracks
- **Event buffer:** Max 1000 pending events before flush to DB
- **Face embeddings:** Lazy load, cache last 100

### 7.3 Thread Safety

- All `@dataclass(frozen=True)` models are immutable
- `TrackedObject` is mutable but uses explicit locks
- Event handlers run in separate thread pool
- Camera capture runs in dedicated thread

### 7.4 Testing Strategy

```
tests/
├── unit/
│   ├── test_models.py          # Data class tests
│   ├── test_motion.py          # Motion detector tests
│   ├── test_person.py          # Person detector tests
│   ├── test_tracker.py         # Object tracker tests
│   └── test_state_machine.py   # State machine tests
│
├── integration/
│   ├── test_pipeline.py        # Full pipeline integration
│   └── test_events.py          # Event flow tests
│
└── fixtures/
    ├── frames/                 # Test frames
    ├── videos/                 # Test videos
    └── faces/                  # Test face images
```

---

## 8. NEXT STEPS FOR DEVELOPER AGENT

1. **Create directory structure** per Section 6
2. **Implement core/models.py** per Section 2.1
3. **Implement core/protocols.py** per Section 2.2
4. **Implement core/config.py** per Section 2.3
5. **Implement tracking/state_machine.py** per Section 3
6. **Implement tracking/tracker.py** with Kalman filter
7. **Implement pipeline.py** orchestrator
8. **Wire up existing perception components**
9. **Add comprehensive tests**
10. **Integrate with existing voice output**

---

*This architecture is production-grade and battle-tested based on Frigate and Double Take patterns.*
*The key insight: Process frames efficiently, but act on EVENTS.*
