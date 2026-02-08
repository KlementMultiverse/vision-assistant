#!/usr/bin/env python3
"""
Core data models for the detection pipeline.

All immutable (frozen dataclasses) for thread safety except TrackedObject
which needs to maintain mutable state.
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
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

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside the box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def overlaps(self, other: BoundingBox) -> bool:
        """Check if this box overlaps with another."""
        return self.iou(other) > 0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x1, self.y1, self.width, self.height)

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> BoundingBox:
        """Create from (x, y, width, height) format."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)


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

    @property
    def region_count(self) -> int:
        return len(self.regions)


@dataclass(frozen=True)
class PersonDetection:
    """Single person detection from YOLO."""
    box: BoundingBox
    confidence: float  # 0.0 to 1.0

    # Optional: keypoints for pose
    keypoints: Optional[Any] = None  # np.ndarray causes issues with frozen

    @property
    def area(self) -> int:
        return self.box.area


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

    @property
    def detected(self) -> bool:
        return self.count > 0


@dataclass(frozen=True)
class FaceDetection:
    """Single face detection with recognition results."""
    box: BoundingBox
    confidence: float  # Detection confidence 0.0 to 1.0

    # Recognition results (None if not recognized)
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    embedding: Optional[Any] = None  # np.ndarray

    # Quality metrics (for retry logic)
    blur_score: float = 0.0  # Higher = sharper
    face_angle: float = 0.0  # Degrees from frontal

    @property
    def is_known(self) -> bool:
        return self.identity is not None and self.identity != "Unknown"

    @property
    def area(self) -> int:
        return self.box.area


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

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def detected(self) -> bool:
        return self.count > 0

    @property
    def has_known_face(self) -> bool:
        return any(f.is_known for f in self.faces)


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
    object_id: Optional[str] = None   # Persistent ID if face recognized

    # Current state
    state: ObjectState = ObjectState.DETECTING
    box: Optional[BoundingBox] = None

    # Recognition
    face_name: Optional[str] = None
    face_confidence: float = 0.0
    face_attempts: int = 0
    best_snapshot: Optional[np.ndarray] = None
    best_snapshot_quality: float = 0.0

    # Tracking metrics
    frames_seen: int = 0
    frames_lost: int = 0
    frames_stationary: int = 0

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_position: Optional[Tuple[int, int]] = None

    # Zones (if using zone detection)
    current_zones: List[str] = field(default_factory=list)
    entered_zones: List[str] = field(default_factory=list)

    # Kalman filter state (for position prediction)
    kalman_state: Optional[np.ndarray] = None

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Configuration (set by tracker)
    _movement_threshold: int = 10
    _stationary_frames_threshold: int = 30

    def update_position(self, box: BoundingBox, timestamp: float) -> None:
        """Update object position, check for stationary."""
        with self._lock:
            current_center = box.center

            if self.last_position is not None:
                # Check if stationary (moved less than threshold)
                dx = abs(current_center[0] - self.last_position[0])
                dy = abs(current_center[1] - self.last_position[1])
                movement = (dx * dx + dy * dy) ** 0.5

                if movement < self._movement_threshold:
                    self.frames_stationary += 1
                    if (self.frames_stationary > self._stationary_frames_threshold
                            and self.state == ObjectState.ACTIVE):
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

    def mark_lost(self) -> None:
        """Mark object as temporarily not detected."""
        with self._lock:
            self.frames_lost += 1
            if self.state in (ObjectState.ACTIVE, ObjectState.STATIONARY):
                self.state = ObjectState.LOST

    def mark_ended(self) -> None:
        """Mark object tracking as ended."""
        with self._lock:
            self.state = ObjectState.ENDED

    def recover(self, box: BoundingBox, timestamp: float) -> None:
        """Recover lost track."""
        with self._lock:
            if self.state == ObjectState.LOST:
                self.state = ObjectState.ACTIVE
            # Inline position update to avoid deadlock (don't call update_position which acquires lock)
            current_center = box.center
            if self.last_position is not None:
                dx = abs(current_center[0] - self.last_position[0])
                dy = abs(current_center[1] - self.last_position[1])
                movement = (dx * dx + dy * dy) ** 0.5
                if movement < self._movement_threshold:
                    self.frames_stationary += 1
                else:
                    self.frames_stationary = 0
            self.box = box
            self.last_position = current_center
            self.last_seen = timestamp
            self.frames_seen += 1
            self.frames_lost = 0

    def confirm(self) -> None:
        """Confirm detecting object as active."""
        with self._lock:
            if self.state == ObjectState.DETECTING:
                self.state = ObjectState.ACTIVE

    def set_identity(self, name: str, confidence: float, snapshot: Optional[np.ndarray] = None) -> None:
        """Set face identity."""
        with self._lock:
            self.face_name = name
            self.face_confidence = confidence
            if snapshot is not None:
                self.best_snapshot = snapshot

    @property
    def duration(self) -> float:
        """Time this object has been tracked in seconds."""
        return self.last_seen - self.first_seen

    @property
    def is_confirmed(self) -> bool:
        """Whether object is confirmed (past DETECTING state)."""
        return self.state not in (ObjectState.DETECTING, ObjectState.ENDED)

    @property
    def is_active(self) -> bool:
        """Whether object is actively being tracked."""
        return self.state in (ObjectState.ACTIVE, ObjectState.STATIONARY)

    @property
    def is_ended(self) -> bool:
        """Whether tracking has ended."""
        return self.state == ObjectState.ENDED


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
    end_time: Optional[float] = None  # When event ended (None if ongoing)

    # Detection details
    box: Optional[BoundingBox] = None  # Current/last bounding box
    area: int = 0              # Box area in pixels

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

    @property
    def is_ongoing(self) -> bool:
        """Whether event is still ongoing."""
        return self.lifecycle != EventLifecycle.END

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "camera": self.camera,
            "label": self.label.value,
            "lifecycle": self.lifecycle.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "box": self.box.to_tuple() if self.box else None,
            "area": self.area,
            "current_zones": self.current_zones,
            "entered_zones": self.entered_zones,
            "face_name": self.face_name,
            "face_confidence": self.face_confidence,
            "track_id": self.track_id,
            "frames_count": self.frames_count,
            "duration": self.duration,
            "attributes": self.attributes,
        }


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

    @property
    def has_motion(self) -> bool:
        return self.motion is not None and self.motion.detected

    @property
    def has_persons(self) -> bool:
        return self.persons is not None and self.persons.detected

    @property
    def has_faces(self) -> bool:
        return self.faces is not None and self.faces.detected

    @property
    def person_count(self) -> int:
        return self.persons.count if self.persons else 0

    @property
    def face_count(self) -> int:
        return self.faces.count if self.faces else 0


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing core models...")

    # Test BoundingBox
    box1 = BoundingBox(100, 100, 200, 200)
    box2 = BoundingBox(150, 150, 250, 250)
    print(f"Box1: {box1}, area={box1.area}, center={box1.center}")
    print(f"Box2: {box2}, area={box2.area}, center={box2.center}")
    print(f"IoU: {box1.iou(box2):.3f}")

    # Test MotionResult
    motion = MotionResult(detected=True, magnitude=0.15, regions=(box1,))
    print(f"Motion: {motion}")

    # Test PersonDetection
    person = PersonDetection(box=box1, confidence=0.95)
    print(f"Person: {person}")

    # Test TrackedObject
    tracked = TrackedObject(track_id="trk_001")
    tracked.update_position(box1, time.time())
    print(f"Tracked: {tracked.track_id}, state={tracked.state.name}")

    # Test DetectionEvent
    event = DetectionEvent(
        id="evt_001",
        camera="front_door",
        label=EventLabel.PERSON,
        lifecycle=EventLifecycle.NEW,
        start_time=time.time(),
        box=box1,
        area=box1.area,
        track_id="trk_001"
    )
    print(f"Event: {event.id}, lifecycle={event.lifecycle.value}")

    print("\nAll models work correctly!")
