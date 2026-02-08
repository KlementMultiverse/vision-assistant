#!/usr/bin/env python3
"""
Multi-Object Tracker with IoU-based matching.

Tracks multiple objects across frames using:
- IoU (Intersection over Union) for detection-to-track matching
- State machine for lifecycle management
- Event emission on state changes
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np

from ..core.models import (
    TrackedObject, ObjectState, BoundingBox,
    PersonResult, PersonDetection, DetectionEvent, EventLifecycle, EventLabel
)
from ..core.config import TrackerConfig
from ..core.events import EventType, Event, EventBus
from .state_machine import ObjectStateMachine, StateTransition, TransitionType, create_event_from_transition

logger = logging.getLogger(__name__)


# =============================================================================
# HUNGARIAN ALGORITHM FOR OPTIMAL ASSIGNMENT
# =============================================================================

def compute_iou_matrix(
    detections: List[PersonDetection],
    tracks: List[TrackedObject]
) -> np.ndarray:
    """
    Compute IoU matrix between detections and tracks.

    Args:
        detections: List of person detections
        tracks: List of tracked objects

    Returns:
        IoU matrix of shape (len(detections), len(tracks))
    """
    n_det = len(detections)
    n_trk = len(tracks)

    if n_det == 0 or n_trk == 0:
        return np.zeros((n_det, n_trk))

    iou_matrix = np.zeros((n_det, n_trk))

    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            if trk.box is not None:
                iou_matrix[i, j] = det.box.iou(trk.box)

    return iou_matrix


def greedy_matching(
    iou_matrix: np.ndarray,
    threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy matching algorithm for detection-track assignment.

    Args:
        iou_matrix: IoU matrix (detections x tracks)
        threshold: Minimum IoU for valid match

    Returns:
        Tuple of:
        - matches: List of (detection_idx, track_idx) pairs
        - unmatched_detections: List of detection indices
        - unmatched_tracks: List of track indices
    """
    n_det, n_trk = iou_matrix.shape

    if n_det == 0:
        return [], [], list(range(n_trk))
    if n_trk == 0:
        return [], list(range(n_det)), []

    matches = []
    used_dets = set()
    used_trks = set()

    # Sort by IoU descending
    indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)

    for det_idx, trk_idx in zip(indices[0], indices[1]):
        if det_idx in used_dets or trk_idx in used_trks:
            continue

        iou = iou_matrix[det_idx, trk_idx]
        if iou < threshold:
            break  # No more valid matches

        matches.append((int(det_idx), int(trk_idx)))
        used_dets.add(det_idx)
        used_trks.add(trk_idx)

    unmatched_dets = [i for i in range(n_det) if i not in used_dets]
    unmatched_trks = [j for j in range(n_trk) if j not in used_trks]

    return matches, unmatched_dets, unmatched_trks


# =============================================================================
# MULTI-OBJECT TRACKER
# =============================================================================

class MultiObjectTracker:
    """
    Multi-object tracker with state machine lifecycle.

    Features:
    - IoU-based detection-track matching
    - State machine for each tracked object
    - Event emission on lifecycle changes
    - Track cleanup and management
    """

    def __init__(
        self,
        config: TrackerConfig,
        camera_name: str = "default",
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize tracker.

        Args:
            config: Tracker configuration
            camera_name: Camera name for events
            event_bus: Optional event bus for emitting events
        """
        self.config = config
        self.camera_name = camera_name
        self.event_bus = event_bus

        # Active tracks
        self._tracks: Dict[str, ObjectStateMachine] = {}

        # Pending events (consumed by get_events)
        self._pending_events: List[DetectionEvent] = []

        # Statistics
        self._total_tracks = 0
        self._frame_count = 0

        # Last update times for UPDATE event throttling
        self._last_update_event: Dict[str, float] = {}

    def _generate_track_id(self) -> str:
        """Generate unique track ID."""
        self._total_tracks += 1
        return f"trk_{uuid.uuid4().hex[:8]}_{self._total_tracks}"

    def _on_transition(self, transition: StateTransition) -> None:
        """Handle state machine transition."""
        track = self._tracks.get(transition.track_id)
        if not track:
            return

        # Create detection event from transition
        event = create_event_from_transition(
            track.track,
            transition,
            self.camera_name
        )

        if event:
            self._pending_events.append(event)

            # Emit to event bus if available
            if self.event_bus:
                event_type = {
                    EventLifecycle.NEW: EventType.OBJECT_NEW,
                    EventLifecycle.UPDATE: EventType.OBJECT_UPDATE,
                    EventLifecycle.END: EventType.OBJECT_END,
                }.get(event.lifecycle, EventType.OBJECT_UPDATE)

                self.event_bus.emit(Event(
                    type=event_type,
                    timestamp=time.time(),
                    data={
                        "track_id": transition.track_id,
                        "lifecycle": event.lifecycle.value,
                        "face_name": event.face_name,
                    },
                    source="tracker",
                    detection_event=event
                ))

    def update(self, detections: PersonResult, timestamp: float) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: Person detections for this frame
            timestamp: Frame timestamp

        Returns:
            List of all active TrackedObject instances
        """
        self._frame_count += 1

        # Get active tracks (not ended)
        active_tracks = [sm for sm in self._tracks.values() if not sm.is_ended]

        # Handle empty detections
        if not detections.detected:
            for sm in active_tracks:
                sm.update_not_detected(timestamp)
            self._cleanup_ended_tracks()
            return self.get_active_tracks()

        # Get detection list
        det_list = list(detections.detections)
        track_list = [sm.track for sm in active_tracks]

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(det_list, track_list)

        # Match detections to tracks
        matches, unmatched_dets, unmatched_trks = greedy_matching(
            iou_matrix, self.config.iou_threshold
        )

        # Update matched tracks
        for det_idx, trk_idx in matches:
            det = det_list[det_idx]
            sm = active_tracks[trk_idx]
            sm.update_detected(det.box, timestamp)

        # Mark unmatched tracks as not detected
        for trk_idx in unmatched_trks:
            sm = active_tracks[trk_idx]
            sm.update_not_detected(timestamp)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = det_list[det_idx]

            # Check if detection meets minimum area requirement (1000 px default from PersonConfig)
            min_area = 1000  # Minimum detection area in pixels
            if det.area < min_area:
                logger.debug(f"Skipping detection with small area: {det.area} < {min_area}")
                continue

            # Check track limit
            if len(self._tracks) >= self.config.max_tracks:
                logger.warning(f"Max tracks ({self.config.max_tracks}) reached")
                continue

            # Create new track
            track_id = self._generate_track_id()
            track = TrackedObject(
                track_id=track_id,
                box=det.box,
                first_seen=timestamp,
                last_seen=timestamp,
                last_position=det.box.center
            )
            track.frames_seen = 1

            sm = ObjectStateMachine(
                track, self.config, self._on_transition
            )
            self._tracks[track_id] = sm

            logger.debug(f"Created new track {track_id}")

        # Emit periodic UPDATE events for active tracks
        self._emit_periodic_updates(timestamp)

        # Cleanup ended tracks
        self._cleanup_ended_tracks()

        return self.get_active_tracks()

    def _emit_periodic_updates(self, timestamp: float) -> None:
        """Emit UPDATE events at configured interval."""
        for track_id, sm in self._tracks.items():
            if not sm.is_active:
                continue

            last_update = self._last_update_event.get(track_id, 0)
            if timestamp - last_update >= self.config.update_interval:
                self._last_update_event[track_id] = timestamp

                event = DetectionEvent(
                    id=f"evt_{uuid.uuid4().hex[:12]}",
                    camera=self.camera_name,
                    label=EventLabel.PERSON,
                    lifecycle=EventLifecycle.UPDATE,
                    start_time=sm.track.first_seen,
                    box=sm.track.box,
                    area=sm.track.box.area if sm.track.box else 0,
                    current_zones=sm.track.current_zones,
                    entered_zones=sm.track.entered_zones,
                    face_name=sm.track.face_name,
                    face_confidence=sm.track.face_confidence,
                    track_id=track_id,
                    frames_count=sm.track.frames_seen,
                )
                self._pending_events.append(event)

    def _cleanup_ended_tracks(self) -> None:
        """Remove ended tracks."""
        ended = [tid for tid, sm in self._tracks.items() if sm.is_ended]
        for tid in ended:
            del self._tracks[tid]
            if tid in self._last_update_event:
                del self._last_update_event[tid]

    def get_track(self, track_id: str) -> Optional[TrackedObject]:
        """Get specific track by ID."""
        sm = self._tracks.get(track_id)
        return sm.track if sm else None

    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active (non-ended) tracks."""
        return [
            sm.track for sm in self._tracks.values()
            if not sm.is_ended
        ]

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get all confirmed (past DETECTING) tracks."""
        return [
            sm.track for sm in self._tracks.values()
            if sm.is_confirmed and not sm.is_ended
        ]

    def get_events(self) -> List[DetectionEvent]:
        """Get and clear pending events."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def get_new_events(self) -> List[DetectionEvent]:
        """Get NEW lifecycle events only."""
        return [e for e in self.get_events() if e.lifecycle == EventLifecycle.NEW]

    def set_face_identity(
        self,
        track_id: str,
        name: str,
        confidence: float,
        snapshot: Optional[np.ndarray] = None
    ) -> bool:
        """
        Set face identity for a track.

        Args:
            track_id: Track to update
            name: Face name
            confidence: Recognition confidence
            snapshot: Optional face snapshot

        Returns:
            True if track was updated
        """
        sm = self._tracks.get(track_id)
        if sm and not sm.is_ended:
            sm.track.set_identity(name, confidence, snapshot)

            # Emit person identified event
            if self.event_bus and name != "Unknown":
                self.event_bus.emit_simple(
                    EventType.PERSON_IDENTIFIED,
                    source="tracker",
                    track_id=track_id,
                    name=name,
                    confidence=confidence
                )
            return True
        return False

    def end_all_tracks(self) -> List[DetectionEvent]:
        """
        End all active tracks (e.g., on shutdown).

        Returns:
            List of END events
        """
        events = []
        for sm in self._tracks.values():
            if not sm.is_ended:
                transition = sm.force_end("Tracker shutdown")
                event = create_event_from_transition(
                    sm.track, transition, self.camera_name
                )
                if event:
                    events.append(event)

        self._tracks.clear()
        self._last_update_event.clear()

        return events

    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        active = sum(1 for sm in self._tracks.values() if sm.is_active)
        detecting = sum(1 for sm in self._tracks.values()
                       if sm.track.state == ObjectState.DETECTING)
        lost = sum(1 for sm in self._tracks.values()
                  if sm.track.state == ObjectState.LOST)

        return {
            "total_tracks_created": self._total_tracks,
            "current_tracks": len(self._tracks),
            "active_tracks": active,
            "detecting_tracks": detecting,
            "lost_tracks": lost,
            "frame_count": self._frame_count,
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing MultiObjectTracker...")

    # Create config
    config = TrackerConfig(
        confirm_frames=2,
        lost_frames_max=3,
        iou_threshold=0.3,
        update_interval=0.5
    )

    # Create tracker
    tracker = MultiObjectTracker(config, camera_name="test_cam")

    # Simulate detections
    print("\n1. First detection...")
    det1 = PersonDetection(
        box=BoundingBox(100, 100, 200, 200),
        confidence=0.9
    )
    result1 = PersonResult(detections=(det1,), timestamp=time.time())
    tracks = tracker.update(result1, time.time())
    print(f"   Active tracks: {len(tracks)}")
    for t in tracks:
        print(f"   - {t.track_id}: {t.state.name}")

    # Get events
    events = tracker.get_events()
    print(f"   Events: {len(events)}")

    print("\n2. Second detection (confirm)...")
    time.sleep(0.1)
    tracks = tracker.update(result1, time.time())
    events = tracker.get_events()
    print(f"   Active tracks: {len(tracks)}")
    print(f"   Events: {len(events)}")
    for e in events:
        print(f"   - {e.lifecycle.value}: {e.track_id}")

    print("\n3. Add second person...")
    det2 = PersonDetection(
        box=BoundingBox(400, 100, 500, 300),
        confidence=0.85
    )
    result2 = PersonResult(detections=(det1, det2), timestamp=time.time())
    tracks = tracker.update(result2, time.time())
    print(f"   Active tracks: {len(tracks)}")

    print("\n4. Person 1 moves...")
    det1_moved = PersonDetection(
        box=BoundingBox(150, 150, 250, 250),  # Moved slightly
        confidence=0.9
    )
    result3 = PersonResult(detections=(det1_moved, det2), timestamp=time.time())
    tracks = tracker.update(result3, time.time())
    print(f"   Active tracks: {len(tracks)}")

    print("\n5. Person 2 leaves...")
    result4 = PersonResult(detections=(det1_moved,), timestamp=time.time())
    for i in range(5):
        tracks = tracker.update(result4, time.time())
        events = tracker.get_events()
        print(f"   Frame {i+1}: tracks={len(tracks)}, events={len(events)}")
        for e in events:
            print(f"     - {e.lifecycle.value}")

    print("\n6. Statistics...")
    stats = tracker.get_statistics()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n7. Set face identity...")
    if tracks:
        track_id = tracks[0].track_id
        tracker.set_face_identity(track_id, "John", 0.85)
        track = tracker.get_track(track_id)
        print(f"   {track.track_id}: face_name={track.face_name}")

    print("\n8. End all tracks...")
    end_events = tracker.end_all_tracks()
    print(f"   End events: {len(end_events)}")

    print("\nMultiObjectTracker OK!")
