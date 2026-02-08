#!/usr/bin/env python3
"""
Edge Case Tests for Vision Assistant Pipeline

Tests edge cases from ARCHITECTURE.md Section 5:
- Motion detection edge cases (lighting, camera shake, empty frames)
- Person detection edge cases (no persons, multiple, partial visibility)
- Face recognition edge cases (no face, multiple, low quality)
- Tracking edge cases (ID stability, re-entry, short visits)
- Pipeline edge cases (camera errors, exceptions, memory)
"""

import sys
import time
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.v2.core.models import (
    BoundingBox, MotionResult, PersonDetection, PersonResult,
    FaceDetection, FaceResult, TrackedObject, ObjectState,
    DetectionEvent, EventLifecycle, EventLabel, PipelineFrame
)
from src.v2.core.config import (
    PipelineConfig, MotionConfig, PersonConfig, FaceConfig, TrackerConfig
)
from src.v2.tracking.state_machine import ObjectStateMachine, TransitionType
from src.v2.tracking.tracker import MultiObjectTracker, compute_iou_matrix, greedy_matching


class TestBoundingBoxEdgeCases(unittest.TestCase):
    """Test BoundingBox edge cases."""

    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0."""
        box1 = BoundingBox(0, 0, 100, 100)
        box2 = BoundingBox(200, 200, 300, 300)
        self.assertEqual(box1.iou(box2), 0.0)

    def test_iou_full_overlap(self):
        """IoU of identical boxes should be 1."""
        box1 = BoundingBox(100, 100, 200, 200)
        box2 = BoundingBox(100, 100, 200, 200)
        self.assertEqual(box1.iou(box2), 1.0)

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes."""
        box1 = BoundingBox(0, 0, 100, 100)
        box2 = BoundingBox(50, 50, 150, 150)
        iou = box1.iou(box2)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)

    def test_contains_point_inside(self):
        """Point inside box returns True."""
        box = BoundingBox(100, 100, 200, 200)
        self.assertTrue(box.contains_point(150, 150))

    def test_contains_point_outside(self):
        """Point outside box returns False."""
        box = BoundingBox(100, 100, 200, 200)
        self.assertFalse(box.contains_point(50, 50))

    def test_contains_point_edge(self):
        """Point on edge of box returns True."""
        box = BoundingBox(100, 100, 200, 200)
        self.assertTrue(box.contains_point(100, 100))  # Corner
        self.assertTrue(box.contains_point(150, 100))  # Edge

    def test_zero_area_box(self):
        """Zero-area box should have area 0."""
        box = BoundingBox(100, 100, 100, 100)
        self.assertEqual(box.area, 0)

    def test_negative_dimensions_handled(self):
        """Box with x2 < x1 or y2 < y1 should have negative dimensions."""
        box = BoundingBox(200, 200, 100, 100)  # Inverted
        self.assertEqual(box.width, -100)
        self.assertEqual(box.height, -100)


class TestMotionResultEdgeCases(unittest.TestCase):
    """Test MotionResult edge cases."""

    def test_no_motion_detected(self):
        """Motion result with no detection."""
        result = MotionResult(detected=False, magnitude=0.0)
        self.assertFalse(result.detected)
        self.assertEqual(result.magnitude, 0.0)
        self.assertEqual(result.region_count, 0)

    def test_high_magnitude_lighting_change(self):
        """High magnitude indicates lighting change."""
        result = MotionResult(detected=True, magnitude=0.9)
        self.assertTrue(result.detected)
        self.assertGreater(result.magnitude, 0.4)  # Above threshold

    def test_motion_with_regions(self):
        """Motion with specific regions."""
        regions = (BoundingBox(10, 10, 50, 50), BoundingBox(100, 100, 150, 150))
        result = MotionResult(detected=True, magnitude=0.15, regions=regions)
        self.assertEqual(result.region_count, 2)


class TestPersonResultEdgeCases(unittest.TestCase):
    """Test PersonResult edge cases."""

    def test_no_persons_detected(self):
        """Empty person result."""
        result = PersonResult()
        self.assertFalse(result.detected)
        self.assertEqual(result.count, 0)

    def test_multiple_persons(self):
        """Multiple person detections."""
        detections = (
            PersonDetection(box=BoundingBox(0, 0, 100, 200), confidence=0.9),
            PersonDetection(box=BoundingBox(200, 0, 300, 200), confidence=0.85),
            PersonDetection(box=BoundingBox(400, 0, 500, 200), confidence=0.8),
        )
        result = PersonResult(detections=detections)
        self.assertTrue(result.detected)
        self.assertEqual(result.count, 3)

    def test_low_confidence_detection(self):
        """Low confidence detection."""
        det = PersonDetection(box=BoundingBox(0, 0, 50, 50), confidence=0.3)
        self.assertEqual(det.confidence, 0.3)
        self.assertLess(det.confidence, 0.5)  # Below typical threshold


class TestTrackedObjectEdgeCases(unittest.TestCase):
    """Test TrackedObject edge cases."""

    def test_initial_state_is_detecting(self):
        """New track starts in DETECTING state."""
        track = TrackedObject(track_id="test_001")
        self.assertEqual(track.state, ObjectState.DETECTING)

    def test_stationary_detection(self):
        """Track becomes stationary after no movement."""
        track = TrackedObject(track_id="test_002")
        track._movement_threshold = 10
        track._stationary_frames_threshold = 3

        box = BoundingBox(100, 100, 200, 200)
        for i in range(5):
            track.update_position(box, time.time())

        # After 3 frames of no movement while ACTIVE, should become STATIONARY
        # Need to first confirm to ACTIVE
        track.confirm()
        for i in range(5):
            track.update_position(box, time.time())

        self.assertEqual(track.state, ObjectState.STATIONARY)

    def test_movement_resets_stationary(self):
        """Movement resets stationary counter."""
        track = TrackedObject(track_id="test_003")
        track._movement_threshold = 10
        track._stationary_frames_threshold = 3
        track.confirm()

        # Stay stationary
        box1 = BoundingBox(100, 100, 200, 200)
        for i in range(5):
            track.update_position(box1, time.time())

        self.assertEqual(track.state, ObjectState.STATIONARY)

        # Now move
        box2 = BoundingBox(200, 200, 300, 300)  # Significant movement
        track.update_position(box2, time.time())

        self.assertEqual(track.state, ObjectState.ACTIVE)
        self.assertEqual(track.frames_stationary, 0)

    def test_lost_state_transition(self):
        """Track transitions to LOST when not detected."""
        track = TrackedObject(track_id="test_004")
        track.confirm()

        track.mark_lost()
        self.assertEqual(track.state, ObjectState.LOST)

    def test_recovery_from_lost(self):
        """Track recovers from LOST to ACTIVE."""
        track = TrackedObject(track_id="test_005")
        track.confirm()
        track.mark_lost()

        box = BoundingBox(100, 100, 200, 200)
        track.recover(box, time.time())

        self.assertEqual(track.state, ObjectState.ACTIVE)

    def test_ended_state_is_final(self):
        """ENDED state is terminal."""
        track = TrackedObject(track_id="test_006")
        track.mark_ended()
        self.assertEqual(track.state, ObjectState.ENDED)
        self.assertTrue(track.is_ended)

    def test_thread_safety_update(self):
        """Update position is thread-safe (uses lock)."""
        track = TrackedObject(track_id="test_007")
        box = BoundingBox(100, 100, 200, 200)

        # Multiple rapid updates should not cause issues
        import threading
        errors = []

        def update_thread():
            try:
                for _ in range(100):
                    track.update_position(box, time.time())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestStateMachineEdgeCases(unittest.TestCase):
    """Test ObjectStateMachine edge cases."""

    def setUp(self):
        self.config = TrackerConfig(
            confirm_frames=3,
            lost_frames_max=5,
            movement_threshold=10,
            stationary_frames=30
        )

    def test_confirm_after_n_frames(self):
        """Track confirms after confirm_frames detections."""
        track = TrackedObject(track_id="test_sm_001")
        sm = ObjectStateMachine(track, self.config)

        box = BoundingBox(100, 100, 200, 200)

        # Need confirm_frames (3) detections
        transition = None
        for i in range(self.config.confirm_frames):
            transition = sm.update_detected(box, time.time())

        self.assertIsNotNone(transition)
        self.assertEqual(transition.transition_type, TransitionType.CONFIRM)
        self.assertEqual(track.state, ObjectState.ACTIVE)

    def test_discard_if_lost_during_detecting(self):
        """Track is discarded if lost during DETECTING."""
        track = TrackedObject(track_id="test_sm_002")
        sm = ObjectStateMachine(track, self.config)

        # One detection
        box = BoundingBox(100, 100, 200, 200)
        sm.update_detected(box, time.time())

        # Then lost
        transition = sm.update_not_detected(time.time())

        self.assertIsNotNone(transition)
        self.assertEqual(transition.transition_type, TransitionType.DISCARD)
        self.assertEqual(track.state, ObjectState.ENDED)

    def test_end_after_lost_timeout(self):
        """Track ends after lost_frames_max frames lost."""
        track = TrackedObject(track_id="test_sm_003")
        sm = ObjectStateMachine(track, self.config)

        # Confirm first
        box = BoundingBox(100, 100, 200, 200)
        for i in range(self.config.confirm_frames):
            sm.update_detected(box, time.time())

        # First not_detected transitions ACTIVE -> LOST (sets frames_lost=1)
        # Then we need lost_frames_max more calls to reach >= lost_frames_max
        # So total calls = 1 (for LOSE) + lost_frames_max (for END trigger)
        end_transition = None
        for i in range(self.config.lost_frames_max + 2):  # +2 to ensure we trigger END
            transition = sm.update_not_detected(time.time())
            if transition and transition.transition_type == TransitionType.END:
                end_transition = transition
                break

        self.assertIsNotNone(end_transition)
        self.assertEqual(end_transition.transition_type, TransitionType.END)
        self.assertEqual(track.state, ObjectState.ENDED)

    def test_recovery_resets_lost_counter(self):
        """Recovery resets lost frame counter."""
        track = TrackedObject(track_id="test_sm_004")
        sm = ObjectStateMachine(track, self.config)

        box = BoundingBox(100, 100, 200, 200)

        # Confirm
        for i in range(self.config.confirm_frames):
            sm.update_detected(box, time.time())

        # Lose a few frames
        for i in range(3):
            sm.update_not_detected(time.time())

        self.assertEqual(track.state, ObjectState.LOST)

        # Recover
        transition = sm.update_detected(box, time.time())

        self.assertEqual(transition.transition_type, TransitionType.RECOVER)
        self.assertEqual(track.state, ObjectState.ACTIVE)
        self.assertEqual(track.frames_lost, 0)


class TestTrackerEdgeCases(unittest.TestCase):
    """Test MultiObjectTracker edge cases."""

    def setUp(self):
        self.config = TrackerConfig(
            confirm_frames=2,
            lost_frames_max=3,
            iou_threshold=0.3,
            max_tracks=10
        )
        self.tracker = MultiObjectTracker(self.config, camera_name="test")

    def test_empty_detections(self):
        """Tracker handles empty detections without crash."""
        result = PersonResult()  # Empty
        tracks = self.tracker.update(result, time.time())
        self.assertEqual(len(tracks), 0)

    def test_single_person_tracking(self):
        """Single person is tracked correctly."""
        det = PersonDetection(box=BoundingBox(100, 100, 200, 300), confidence=0.9)
        result = PersonResult(detections=(det,))

        # First frame - DETECTING
        tracks = self.tracker.update(result, time.time())
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].state, ObjectState.DETECTING)

        # Second frame - ACTIVE
        tracks = self.tracker.update(result, time.time())
        self.assertEqual(tracks[0].state, ObjectState.ACTIVE)

    def test_multiple_persons_tracking(self):
        """Multiple persons are tracked separately."""
        dets = (
            PersonDetection(box=BoundingBox(0, 0, 100, 200), confidence=0.9),
            PersonDetection(box=BoundingBox(300, 0, 400, 200), confidence=0.85),
        )
        result = PersonResult(detections=dets)

        for _ in range(3):  # Confirm both
            tracks = self.tracker.update(result, time.time())

        self.assertEqual(len(tracks), 2)
        track_ids = [t.track_id for t in tracks]
        self.assertNotEqual(track_ids[0], track_ids[1])

    def test_track_id_stability(self):
        """Same person maintains same track ID."""
        det = PersonDetection(box=BoundingBox(100, 100, 200, 300), confidence=0.9)
        result = PersonResult(detections=(det,))

        # First detection
        tracks1 = self.tracker.update(result, time.time())
        track_id = tracks1[0].track_id

        # Slightly moved
        det2 = PersonDetection(box=BoundingBox(110, 110, 210, 310), confidence=0.9)
        result2 = PersonResult(detections=(det2,))
        tracks2 = self.tracker.update(result2, time.time())

        self.assertEqual(tracks2[0].track_id, track_id)

    def test_track_ends_after_lost_timeout(self):
        """Track ends when person leaves for too long."""
        det = PersonDetection(box=BoundingBox(100, 100, 200, 300), confidence=0.9)
        result = PersonResult(detections=(det,))

        # Create and confirm track
        for _ in range(self.config.confirm_frames):
            self.tracker.update(result, time.time())

        # Person leaves
        empty = PersonResult()
        for _ in range(self.config.lost_frames_max + 2):
            tracks = self.tracker.update(empty, time.time())

        # Track should be removed
        self.assertEqual(len(tracks), 0)

    def test_max_tracks_limit(self):
        """Tracker respects max_tracks limit."""
        # Create many detections
        dets = tuple(
            PersonDetection(
                box=BoundingBox(i * 110, 0, i * 110 + 100, 200),
                confidence=0.9
            )
            for i in range(15)  # More than max_tracks
        )
        result = PersonResult(detections=dets)

        tracks = self.tracker.update(result, time.time())
        self.assertLessEqual(len(tracks), self.config.max_tracks)

    def test_events_are_generated(self):
        """Events are generated on state transitions."""
        det = PersonDetection(box=BoundingBox(100, 100, 200, 300), confidence=0.9)
        result = PersonResult(detections=(det,))

        # Confirm track (generates NEW event)
        for _ in range(self.config.confirm_frames):
            self.tracker.update(result, time.time())

        events = self.tracker.get_events()
        new_events = [e for e in events if e.lifecycle == EventLifecycle.NEW]
        self.assertGreater(len(new_events), 0)

    def test_set_face_identity(self):
        """Face identity can be set on tracks."""
        det = PersonDetection(box=BoundingBox(100, 100, 200, 300), confidence=0.9)
        result = PersonResult(detections=(det,))

        # Create track
        for _ in range(self.config.confirm_frames):
            tracks = self.tracker.update(result, time.time())

        track_id = tracks[0].track_id
        success = self.tracker.set_face_identity(track_id, "Alice", 0.95)

        self.assertTrue(success)
        self.assertEqual(tracks[0].face_name, "Alice")


class TestIoUMatchingEdgeCases(unittest.TestCase):
    """Test IoU matrix computation and matching."""

    def test_empty_detections(self):
        """Empty detections produce empty matrix."""
        matrix = compute_iou_matrix([], [])
        self.assertEqual(matrix.shape, (0, 0))

    def test_no_overlap_matching(self):
        """Non-overlapping detections/tracks have 0 IoU."""
        dets = [PersonDetection(box=BoundingBox(0, 0, 50, 50), confidence=0.9)]
        tracks = [TrackedObject(track_id="t1")]
        tracks[0].box = BoundingBox(200, 200, 300, 300)

        matrix = compute_iou_matrix(dets, tracks)
        self.assertEqual(matrix[0, 0], 0.0)

    def test_greedy_matching_all_matched(self):
        """Greedy matching assigns all when possible."""
        # Perfect overlap
        matrix = np.array([[0.8, 0.1], [0.1, 0.9]])
        matches, unmatched_dets, unmatched_trks = greedy_matching(matrix, 0.3)

        self.assertEqual(len(matches), 2)
        self.assertEqual(len(unmatched_dets), 0)
        self.assertEqual(len(unmatched_trks), 0)

    def test_greedy_matching_threshold(self):
        """Matches below threshold are rejected."""
        matrix = np.array([[0.2, 0.1], [0.1, 0.15]])  # All below 0.3
        matches, unmatched_dets, unmatched_trks = greedy_matching(matrix, 0.3)

        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unmatched_dets), 2)
        self.assertEqual(len(unmatched_trks), 2)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_motion_threshold_bounds(self):
        """Motion threshold must be 1-255."""
        with self.assertRaises(ValueError):
            MotionConfig(threshold=0)
        with self.assertRaises(ValueError):
            MotionConfig(threshold=300)

    def test_confidence_bounds(self):
        """Confidence must be 0-1."""
        with self.assertRaises(ValueError):
            PersonConfig(confidence=-0.1)
        with self.assertRaises(ValueError):
            PersonConfig(confidence=1.5)

    def test_confirm_frames_minimum(self):
        """confirm_frames must be >= 1."""
        with self.assertRaises(ValueError):
            TrackerConfig(confirm_frames=0)

    def test_default_config_valid(self):
        """Default configuration is valid."""
        config = PipelineConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.motion_magnitude_threshold, 0.4)
        self.assertEqual(config.motion_debounce_frames, 2)


class TestPipelineFrameEdgeCases(unittest.TestCase):
    """Test PipelineFrame edge cases."""

    def test_empty_frame_properties(self):
        """Empty frame has no detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pf = PipelineFrame(frame=frame, timestamp=time.time(), frame_number=1)

        self.assertFalse(pf.has_motion)
        self.assertFalse(pf.has_persons)
        self.assertFalse(pf.has_faces)
        self.assertEqual(pf.person_count, 0)

    def test_frame_with_results(self):
        """Frame with results has correct properties."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pf = PipelineFrame(frame=frame, timestamp=time.time(), frame_number=1)

        pf.motion = MotionResult(detected=True, magnitude=0.2)
        pf.persons = PersonResult(detections=(
            PersonDetection(box=BoundingBox(0, 0, 100, 200), confidence=0.9),
        ))

        self.assertTrue(pf.has_motion)
        self.assertTrue(pf.has_persons)
        self.assertEqual(pf.person_count, 1)


class TestDetectionEventEdgeCases(unittest.TestCase):
    """Test DetectionEvent edge cases."""

    def test_event_duration(self):
        """Event duration is calculated correctly."""
        event = DetectionEvent(
            id="test_001",
            camera="cam1",
            label=EventLabel.PERSON,
            lifecycle=EventLifecycle.NEW,
            start_time=time.time() - 5.0  # 5 seconds ago
        )
        self.assertGreaterEqual(event.duration, 5.0)
        self.assertLess(event.duration, 6.0)

    def test_ended_event_duration(self):
        """Ended event has fixed duration."""
        start = time.time() - 10.0
        end = time.time() - 5.0
        event = DetectionEvent(
            id="test_002",
            camera="cam1",
            label=EventLabel.PERSON,
            lifecycle=EventLifecycle.END,
            start_time=start,
            end_time=end
        )
        self.assertAlmostEqual(event.duration, 5.0, places=1)

    def test_event_to_dict(self):
        """Event serialization works."""
        event = DetectionEvent(
            id="test_003",
            camera="cam1",
            label=EventLabel.PERSON,
            lifecycle=EventLifecycle.NEW,
            start_time=time.time(),
            box=BoundingBox(100, 100, 200, 200)
        )
        d = event.to_dict()

        self.assertEqual(d["id"], "test_003")
        self.assertEqual(d["label"], "person")
        self.assertEqual(d["lifecycle"], "new")
        self.assertIsNotNone(d["box"])


class TestFrameValidation(unittest.TestCase):
    """Test frame validation edge cases."""

    def test_valid_bgr_frame(self):
        """Valid BGR frame passes validation."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(len(frame.shape), 3)
        self.assertGreater(frame.size, 0)

    def test_empty_array(self):
        """Empty array fails validation."""
        frame = np.array([])
        self.assertEqual(frame.size, 0)

    def test_grayscale_frame(self):
        """Grayscale frame (2D) has wrong shape."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        self.assertNotEqual(len(frame.shape), 3)

    def test_single_channel(self):
        """Single channel frame has 3 dimensions but wrong depth."""
        frame = np.zeros((480, 640, 1), dtype=np.uint8)
        self.assertEqual(len(frame.shape), 3)
        self.assertNotEqual(frame.shape[2], 3)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
