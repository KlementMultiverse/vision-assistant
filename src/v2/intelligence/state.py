#!/usr/bin/env python3
"""
Presence State Machine
======================
Tracks visitor presence with proper state transitions.

States:
  EMPTY     → No one detected
  DETECTING → Motion/person seen, confirming...
  PRESENT   → Confirmed person present
  LEAVING   → Person was here, checking if left

Transitions require multiple frames to confirm (debouncing).
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Callable


class PresenceState(Enum):
    """Visitor presence states."""
    EMPTY = auto()      # No one here
    DETECTING = auto()  # Might be someone, confirming
    PRESENT = auto()    # Confirmed person present
    LEAVING = auto()    # Was here, checking if left


class EventType(Enum):
    """Events emitted by state machine."""
    PERSON_ENTERED = auto()   # New person arrived
    PERSON_IDENTIFIED = auto() # Face recognized
    PERSON_LEFT = auto()       # Person departed
    MOTION_STARTED = auto()    # Motion began
    MOTION_STOPPED = auto()    # Motion ended


@dataclass
class Event:
    """An event from the state machine."""
    type: EventType
    timestamp: float
    data: dict = field(default_factory=dict)


@dataclass
class DetectionFrame:
    """Single frame detection results."""
    timestamp: float
    has_motion: bool
    person_count: int
    face_names: List[str] = field(default_factory=list)


class PresenceTracker:
    """
    Tracks presence with debounced state transitions.

    Key behaviors:
    - Requires N consecutive frames to confirm detection
    - Requires M seconds of no detection to confirm departure
    - Emits events on state changes
    - Tracks known vs unknown faces
    """

    def __init__(
        self,
        confirm_frames: int = 3,      # Frames to confirm presence
        leave_timeout: float = 5.0,   # Seconds before "left"
        motion_timeout: float = 2.0,  # Seconds of no motion = stopped
    ):
        self.confirm_frames = confirm_frames
        self.leave_timeout = leave_timeout
        self.motion_timeout = motion_timeout

        # State
        self._state = PresenceState.EMPTY
        self._detection_count = 0
        self._last_person_time: Optional[float] = None
        self._last_motion_time: Optional[float] = None
        self._motion_active = False
        self._current_faces: List[str] = []
        self._greeted = False

        # Event callbacks
        self._callbacks: List[Callable[[Event], None]] = []

        # History for debugging
        self._events: List[Event] = []

    @property
    def state(self) -> PresenceState:
        return self._state

    @property
    def is_present(self) -> bool:
        return self._state == PresenceState.PRESENT

    @property
    def current_faces(self) -> List[str]:
        return self._current_faces.copy()

    @property
    def should_greet(self) -> bool:
        """Should we greet? Only once per presence."""
        return self._state == PresenceState.PRESENT and not self._greeted

    def mark_greeted(self):
        """Mark that we've greeted the current visitor."""
        self._greeted = True

    def on_event(self, callback: Callable[[Event], None]):
        """Register event callback."""
        self._callbacks.append(callback)

    def _emit(self, event_type: EventType, **data):
        """Emit an event."""
        event = Event(
            type=event_type,
            timestamp=time.time(),
            data=data
        )
        self._events.append(event)
        if len(self._events) > 100:
            self._events = self._events[-50:]

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"Event callback error: {e}")

    def update(self, detection: DetectionFrame) -> Optional[Event]:
        """
        Update state with new detection.

        Args:
            detection: Current frame detection results

        Returns:
            Event if state changed, None otherwise
        """
        now = detection.timestamp
        event = None

        # Track motion
        if detection.has_motion:
            if not self._motion_active:
                self._motion_active = True
                self._emit(EventType.MOTION_STARTED)
            self._last_motion_time = now
        elif self._motion_active:
            if self._last_motion_time and (now - self._last_motion_time) > self.motion_timeout:
                self._motion_active = False
                self._emit(EventType.MOTION_STOPPED)

        # State machine
        if self._state == PresenceState.EMPTY:
            if detection.person_count > 0:
                self._state = PresenceState.DETECTING
                self._detection_count = 1

        elif self._state == PresenceState.DETECTING:
            if detection.person_count > 0:
                self._detection_count += 1
                if self._detection_count >= self.confirm_frames:
                    # Confirmed!
                    self._state = PresenceState.PRESENT
                    self._last_person_time = now
                    self._current_faces = detection.face_names.copy()
                    self._greeted = False
                    self._emit(EventType.PERSON_ENTERED, faces=self._current_faces)
                    event = Event(EventType.PERSON_ENTERED, now, {"faces": self._current_faces})
            else:
                # Lost detection, reset
                self._state = PresenceState.EMPTY
                self._detection_count = 0

        elif self._state == PresenceState.PRESENT:
            if detection.person_count > 0:
                self._last_person_time = now
                # Update faces if new ones identified
                new_faces = [f for f in detection.face_names if f not in self._current_faces and f != "Unknown"]
                if new_faces:
                    self._current_faces.extend(new_faces)
                    self._emit(EventType.PERSON_IDENTIFIED, faces=new_faces)
            else:
                # No person, start leaving countdown
                self._state = PresenceState.LEAVING

        elif self._state == PresenceState.LEAVING:
            if detection.person_count > 0:
                # They're back
                self._state = PresenceState.PRESENT
                self._last_person_time = now
            elif self._last_person_time and (now - self._last_person_time) > self.leave_timeout:
                # Confirmed left
                self._state = PresenceState.EMPTY
                self._emit(EventType.PERSON_LEFT, faces=self._current_faces)
                event = Event(EventType.PERSON_LEFT, now, {"faces": self._current_faces})
                self._current_faces = []
                self._detection_count = 0

        return event

    def get_status(self) -> dict:
        """Get current status for display."""
        return {
            "state": self._state.name,
            "motion": self._motion_active,
            "faces": self._current_faces,
            "greeted": self._greeted,
            "detection_count": self._detection_count,
        }


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing PresenceTracker...")

    tracker = PresenceTracker(confirm_frames=3, leave_timeout=2.0)

    # Register callback
    def on_event(e):
        print(f"  EVENT: {e.type.name} - {e.data}")
    tracker.on_event(on_event)

    # Simulate frames
    frames = [
        # No one
        DetectionFrame(0.0, False, 0),
        DetectionFrame(0.1, False, 0),
        # Motion, person appears
        DetectionFrame(0.2, True, 1),
        DetectionFrame(0.3, True, 1),
        DetectionFrame(0.4, True, 1),  # Should confirm PRESENT
        DetectionFrame(0.5, True, 1, ["John"]),  # Face identified
        # Person still there
        DetectionFrame(1.0, False, 1),
        DetectionFrame(1.5, False, 1),
        # Person leaves
        DetectionFrame(2.0, False, 0),
        DetectionFrame(2.5, False, 0),
        DetectionFrame(3.0, False, 0),
        DetectionFrame(4.5, False, 0),  # Should confirm LEFT
    ]

    for f in frames:
        tracker.update(f)
        status = tracker.get_status()
        print(f"t={f.timestamp:.1f} person={f.person_count} → {status['state']}")

    print("\n✅ PresenceTracker works!")
