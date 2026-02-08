#!/usr/bin/env python3
"""
Object State Machine for tracked objects.

Implements the state transitions defined in the architecture:
  DETECTING → ACTIVE → STATIONARY → LOST → ENDED

Each transition has specific conditions and actions.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple
from enum import Enum, auto

from ..core.models import ObjectState, TrackedObject, BoundingBox, DetectionEvent, EventLifecycle, EventLabel
from ..core.config import TrackerConfig

logger = logging.getLogger(__name__)


# =============================================================================
# STATE TRANSITIONS
# =============================================================================

class TransitionType(Enum):
    """Types of state transitions."""
    CONFIRM = auto()       # DETECTING → ACTIVE
    DISCARD = auto()       # DETECTING → (deleted)
    STATIONARY = auto()    # ACTIVE → STATIONARY
    MOVE = auto()          # STATIONARY → ACTIVE
    LOSE = auto()          # ACTIVE/STATIONARY → LOST
    RECOVER = auto()       # LOST → ACTIVE
    END = auto()           # LOST → ENDED


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: ObjectState
    to_state: ObjectState
    transition_type: TransitionType
    timestamp: float
    track_id: str
    reason: str = ""


# =============================================================================
# STATE MACHINE
# =============================================================================

class ObjectStateMachine:
    """
    Manages state transitions for a single tracked object.

    State Diagram:
        DETECTING → ACTIVE ⟷ STATIONARY
             ↓         ↓          ↓
           (discard)  LOST ←──────┘
                       ↓
                     ENDED

    Transition Rules:
    - DETECTING → ACTIVE: frames_seen >= confirm_frames
    - DETECTING → (discard): frames_lost > 0
    - ACTIVE → STATIONARY: no movement for stationary_frames
    - STATIONARY → ACTIVE: movement detected
    - ACTIVE/STATIONARY → LOST: frames_lost > 0
    - LOST → ACTIVE: detection recovered
    - LOST → ENDED: frames_lost >= lost_frames_max
    """

    def __init__(
        self,
        track: TrackedObject,
        config: TrackerConfig,
        on_transition: Optional[Callable[[StateTransition], None]] = None
    ):
        """
        Initialize state machine for a track.

        Args:
            track: The tracked object to manage
            config: Tracker configuration
            on_transition: Callback for state transitions
        """
        self.track = track
        self.config = config
        self._on_transition = on_transition
        self._transitions: List[StateTransition] = []

        # Configure the track
        track._movement_threshold = config.movement_threshold
        track._stationary_frames_threshold = config.stationary_frames

    def _emit_transition(
        self,
        from_state: ObjectState,
        to_state: ObjectState,
        transition_type: TransitionType,
        reason: str = ""
    ) -> StateTransition:
        """Record and emit a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            transition_type=transition_type,
            timestamp=time.time(),
            track_id=self.track.track_id,
            reason=reason
        )
        self._transitions.append(transition)

        logger.debug(
            f"Track {self.track.track_id}: {from_state.name} → {to_state.name} ({reason})"
        )

        if self._on_transition:
            self._on_transition(transition)

        return transition

    def update_detected(self, box: BoundingBox, timestamp: float) -> Optional[StateTransition]:
        """
        Object was detected in this frame.

        Args:
            box: Bounding box of detection
            timestamp: Frame timestamp

        Returns:
            StateTransition if state changed
        """
        old_state = self.track.state

        if old_state == ObjectState.DETECTING:
            # Update position
            self.track.update_position(box, timestamp)

            # Check if confirmed
            if self.track.frames_seen >= self.config.confirm_frames:
                self.track.confirm()
                return self._emit_transition(
                    old_state, ObjectState.ACTIVE,
                    TransitionType.CONFIRM,
                    f"Confirmed after {self.track.frames_seen} frames"
                )

        elif old_state == ObjectState.ACTIVE:
            # Normal tracking update
            self.track.update_position(box, timestamp)

            # Check if became stationary (handled inside update_position)
            if self.track.state == ObjectState.STATIONARY:
                return self._emit_transition(
                    old_state, ObjectState.STATIONARY,
                    TransitionType.STATIONARY,
                    f"Stationary for {self.track.frames_stationary} frames"
                )

        elif old_state == ObjectState.STATIONARY:
            # Update and check for movement
            prev_stationary = self.track.frames_stationary
            self.track.update_position(box, timestamp)

            # Check if started moving (handled inside update_position)
            if self.track.state == ObjectState.ACTIVE:
                return self._emit_transition(
                    old_state, ObjectState.ACTIVE,
                    TransitionType.MOVE,
                    "Movement detected"
                )

        elif old_state == ObjectState.LOST:
            # Recovered!
            self.track.recover(box, timestamp)
            return self._emit_transition(
                old_state, ObjectState.ACTIVE,
                TransitionType.RECOVER,
                f"Recovered after {self.track.frames_lost} frames"
            )

        elif old_state == ObjectState.ENDED:
            # Should not happen - ended tracks shouldn't be updated
            logger.warning(f"Update called on ended track {self.track.track_id}")

        return None

    def update_not_detected(self, timestamp: float) -> Optional[StateTransition]:
        """
        Object was NOT detected in this frame.

        Args:
            timestamp: Frame timestamp

        Returns:
            StateTransition if state changed
        """
        old_state = self.track.state

        if old_state == ObjectState.DETECTING:
            # Not confirmed yet and lost - discard
            self.track.mark_ended()
            return self._emit_transition(
                old_state, ObjectState.ENDED,
                TransitionType.DISCARD,
                "Lost before confirmation"
            )

        elif old_state in (ObjectState.ACTIVE, ObjectState.STATIONARY):
            # Start lost countdown
            self.track.mark_lost()
            return self._emit_transition(
                old_state, ObjectState.LOST,
                TransitionType.LOSE,
                "Detection lost"
            )

        elif old_state == ObjectState.LOST:
            # Continue countdown
            self.track.frames_lost += 1

            # Check if should end
            if self.track.frames_lost >= self.config.lost_frames_max:
                self.track.mark_ended()
                return self._emit_transition(
                    old_state, ObjectState.ENDED,
                    TransitionType.END,
                    f"Lost for {self.track.frames_lost} frames"
                )

        return None

    def force_end(self, reason: str = "Forced end") -> StateTransition:
        """
        Force the track to end.

        Args:
            reason: Reason for ending

        Returns:
            StateTransition
        """
        old_state = self.track.state
        self.track.mark_ended()
        return self._emit_transition(
            old_state, ObjectState.ENDED,
            TransitionType.END,
            reason
        )

    @property
    def is_confirmed(self) -> bool:
        """Whether object has been confirmed."""
        return self.track.state not in (ObjectState.DETECTING, ObjectState.ENDED)

    @property
    def is_active(self) -> bool:
        """Whether object is actively being tracked."""
        return self.track.state in (ObjectState.ACTIVE, ObjectState.STATIONARY)

    @property
    def is_ended(self) -> bool:
        """Whether tracking has ended."""
        return self.track.state == ObjectState.ENDED

    @property
    def transitions(self) -> List[StateTransition]:
        """Get all transitions for this object."""
        return list(self._transitions)

    def get_summary(self) -> dict:
        """Get summary of tracking for this object."""
        return {
            "track_id": self.track.track_id,
            "state": self.track.state.name,
            "frames_seen": self.track.frames_seen,
            "frames_lost": self.track.frames_lost,
            "duration": self.track.duration,
            "face_name": self.track.face_name,
            "transitions_count": len(self._transitions),
        }


# =============================================================================
# HELPER: Create detection event from transition
# =============================================================================

def create_event_from_transition(
    track: TrackedObject,
    transition: StateTransition,
    camera: str = "default"
) -> Optional[DetectionEvent]:
    """
    Create a DetectionEvent from a state transition.

    Args:
        track: The tracked object
        transition: The state transition
        camera: Camera name

    Returns:
        DetectionEvent or None if no event should be emitted
    """
    import uuid

    # Determine lifecycle based on transition
    if transition.transition_type == TransitionType.CONFIRM:
        lifecycle = EventLifecycle.NEW
    elif transition.transition_type == TransitionType.END:
        lifecycle = EventLifecycle.END
    elif transition.transition_type in (TransitionType.RECOVER, TransitionType.MOVE):
        lifecycle = EventLifecycle.UPDATE
    else:
        # DISCARD, LOSE, STATIONARY don't emit user-facing events
        return None

    return DetectionEvent(
        id=f"evt_{uuid.uuid4().hex[:12]}",
        camera=camera,
        label=EventLabel.PERSON,
        lifecycle=lifecycle,
        start_time=track.first_seen,
        end_time=track.last_seen if lifecycle == EventLifecycle.END else None,
        box=track.box,
        area=track.box.area if track.box else 0,
        current_zones=track.current_zones,
        entered_zones=track.entered_zones,
        face_name=track.face_name,
        face_confidence=track.face_confidence,
        track_id=track.track_id,
        frames_count=track.frames_seen,
        snapshot=track.best_snapshot,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing ObjectStateMachine...")

    # Create config
    config = TrackerConfig(
        confirm_frames=3,
        lost_frames_max=5,
        movement_threshold=10,
        stationary_frames=3
    )

    # Create track
    track = TrackedObject(track_id="test_001")

    # Track transitions
    transitions = []

    def on_transition(t):
        transitions.append(t)
        print(f"  Transition: {t.from_state.name} → {t.to_state.name}")

    # Create state machine
    sm = ObjectStateMachine(track, config, on_transition)

    # Simulate detections
    box = BoundingBox(100, 100, 200, 200)

    print("\n1. Detecting phase (3 frames to confirm)...")
    for i in range(3):
        sm.update_detected(box, time.time())
        print(f"   Frame {i+1}: state={track.state.name}")

    print("\n2. Active tracking...")
    for i in range(5):
        sm.update_detected(box, time.time())
        print(f"   Frame {i+1}: state={track.state.name}")

    print("\n3. Stationary (same position)...")
    for i in range(5):
        sm.update_detected(box, time.time())  # Same box
        print(f"   Frame {i+1}: state={track.state.name}")

    print("\n4. Lost detection...")
    for i in range(3):
        sm.update_not_detected(time.time())
        print(f"   Frame {i+1}: state={track.state.name}, lost={track.frames_lost}")

    print("\n5. Recover...")
    sm.update_detected(box, time.time())
    print(f"   state={track.state.name}")

    print("\n6. Lose until end...")
    for i in range(10):
        sm.update_not_detected(time.time())
        print(f"   Frame {i+1}: state={track.state.name}")
        if track.state == ObjectState.ENDED:
            break

    print(f"\nTotal transitions: {len(transitions)}")
    print(f"Summary: {sm.get_summary()}")

    # Test event creation
    print("\n7. Test event creation...")
    track2 = TrackedObject(track_id="test_002")
    sm2 = ObjectStateMachine(track2, config)

    for i in range(3):
        t = sm2.update_detected(box, time.time())
        if t:
            event = create_event_from_transition(track2, t)
            if event:
                print(f"   Event: {event.lifecycle.value}")

    print("\nStateMachine OK!")
