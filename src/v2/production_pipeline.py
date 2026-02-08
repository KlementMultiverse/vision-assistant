#!/usr/bin/env python3
"""
Production Detection Pipeline
==============================
Full implementation based on architecture document.

Pipeline stages:
1. Frame capture (30 fps)
2. Motion detection (<1ms CPU) - skip if no motion and EMPTY
3. Person detection (~10ms GPU) - skip if no motion
4. Object tracking (<1ms) - IoU matching, state machine
5. Face recognition (~20ms GPU) - only on NEW events
6. Action execution - voice greetings, notifications

Run:
  python -m src.v2.production_pipeline --headless
  python -m src.v2.production_pipeline --camera 0
"""

import cv2
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime

import numpy as np

# Core imports
from src.v2.core.models import (
    BoundingBox, MotionResult, PersonDetection, PersonResult,
    FaceDetection, FaceResult, TrackedObject, ObjectState,
    DetectionEvent, EventLifecycle, EventLabel, PipelineFrame
)
from src.v2.core.config import PipelineConfig, TrackerConfig
from src.v2.core.events import EventType, Event, EventBus, get_event_bus
from src.v2.core.protocols import (
    MotionDetectorAdapter, PersonDetectorAdapter, FaceRecognizerAdapter
)

# Tracking imports
from src.v2.tracking.tracker import MultiObjectTracker

# Perception imports (existing POC modules)
from src.v2.perception.motion.base import SimpleMotionDetector
from src.v2.perception.person.base import SimplePersonDetector
from src.v2.perception.face.base import SimpleFaceDetector

# Voice output
from src.v2.understanding.voice import SimpleVoice, get_response

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE STATISTICS
# =============================================================================

@dataclass
class PipelineStats:
    """Pipeline performance statistics."""
    frames_processed: int = 0
    frames_skipped: int = 0
    total_persons_detected: int = 0
    total_faces_recognized: int = 0
    total_events: int = 0

    # Timing averages (ms)
    avg_motion_ms: float = 0.0
    avg_person_ms: float = 0.0
    avg_face_ms: float = 0.0
    avg_total_ms: float = 0.0

    # Tracking
    active_tracks: int = 0
    total_tracks: int = 0

    def update_timing(self, motion_ms: float, person_ms: float, face_ms: float, total_ms: float) -> None:
        """Update timing averages using exponential moving average."""
        alpha = 0.1
        self.avg_motion_ms = alpha * motion_ms + (1 - alpha) * self.avg_motion_ms
        self.avg_person_ms = alpha * person_ms + (1 - alpha) * self.avg_person_ms
        self.avg_face_ms = alpha * face_ms + (1 - alpha) * self.avg_face_ms
        self.avg_total_ms = alpha * total_ms + (1 - alpha) * self.avg_total_ms


# =============================================================================
# PRODUCTION PIPELINE
# =============================================================================

class ProductionPipeline:
    """
    Production-grade detection pipeline.

    Implements the full architecture:
    - Two-stage detection (motion → person)
    - IoU-based multi-object tracking
    - State machine lifecycle (DETECTING → ACTIVE → LOST → ENDED)
    - Face recognition on NEW events only
    - Voice greetings and notifications
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        camera_id: Optional[int] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            camera_id: Camera ID (overrides config if provided)
            event_bus: Event bus (creates new one if not provided)
        """
        self.config = config or PipelineConfig()
        self._camera_id = camera_id if camera_id is not None else self.config.camera_id
        self.event_bus = event_bus or get_event_bus()

        # Initialize stats
        self.stats = PipelineStats()

        # State
        self._running = False
        self._frame_number = 0
        self._skip_countdown = 0
        self._last_motion_time = 0.0
        self._pending_face_checks: Dict[str, int] = {}  # track_id -> attempts
        self._motion_debounce_count = 0  # Consecutive frames with motion (camera shake filter)
        self._camera_error_count = 0  # Track camera read errors for retry logic
        self._max_camera_errors = 10  # Max consecutive camera errors before giving up

        # Greeted tracks (to avoid duplicate greetings)
        self._greeted_tracks: set = set()

        # Frame buffer for snapshot selection
        self._frame_buffer: List[np.ndarray] = []
        self._max_buffer_size = 10

        logger.info("Initializing Production Pipeline...")
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Loading components...")

        # Motion detector
        logger.info("  [1/5] Motion detector...")
        raw_motion = SimpleMotionDetector(
            threshold=self.config.motion.threshold,
            min_area=self.config.motion.min_area
        )
        self.motion_detector = MotionDetectorAdapter(raw_motion)

        # Person detector
        logger.info("  [2/5] Person detector (YOLOv8n)...")
        raw_person = SimplePersonDetector(
            confidence=self.config.person.confidence,
            device=self.config.person.device
        )
        self.person_detector = PersonDetectorAdapter(raw_person)

        # Face detector
        logger.info("  [3/5] Face detector (InsightFace)...")
        raw_face = SimpleFaceDetector(
            det_thresh=self.config.face.detection_threshold,
            use_gpu=self.config.face.use_gpu
        )
        self.face_detector = FaceRecognizerAdapter(raw_face)
        self._raw_face = raw_face  # Keep reference for registration

        # Voice output
        logger.info("  [4/5] Voice output...")
        if self.config.enable_voice:
            self.voice = SimpleVoice()
        else:
            self.voice = None

        # Object tracker
        logger.info("  [5/5] Object tracker...")
        self.tracker = MultiObjectTracker(
            config=self.config.tracker,
            camera_name=self.config.camera_name,
            event_bus=self.event_bus
        )

        # Camera
        logger.info(f"Opening camera {self._camera_id}...")
        self.cap = cv2.VideoCapture(self._camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self._camera_id}")

        # Register event handlers
        self.event_bus.subscribe(EventType.OBJECT_NEW, self._on_new_object)
        self.event_bus.subscribe(EventType.PERSON_IDENTIFIED, self._on_person_identified)

        logger.info("Pipeline ready!")

    def _on_new_object(self, event: Event) -> None:
        """Handle new object event - trigger face recognition."""
        track_id = event.data.get("track_id")
        if track_id and self.config.face_on_entry_only:
            self._pending_face_checks[track_id] = 0
            logger.debug(f"Queued face check for {track_id}")

    def _on_person_identified(self, event: Event) -> None:
        """Handle person identified event - greet them."""
        name = event.data.get("name")
        track_id = event.data.get("track_id")

        if name and name != "Unknown" and track_id not in self._greeted_tracks:
            self._greeted_tracks.add(track_id)
            if self.voice:
                self.voice.speak_async(f"Welcome back, {name}!")
            logger.info(f"Greeted {name}")

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate frame before processing.

        Args:
            frame: Frame to validate

        Returns:
            True if frame is valid
        """
        if frame is None:
            logger.warning("Received None frame")
            return False
        if frame.size == 0:
            logger.warning("Received empty frame (size=0)")
            return False
        if len(frame.shape) != 3:
            logger.warning(f"Invalid frame shape: {frame.shape}")
            return False
        return True

    def process_frame(self, frame: np.ndarray) -> PipelineFrame:
        """
        Process single frame through all pipeline stages.

        Args:
            frame: BGR image from camera

        Returns:
            PipelineFrame with all results
        """
        start_time = time.time()
        self._frame_number += 1
        timestamp = start_time

        # Edge case: Validate frame before processing
        if not self._validate_frame(frame):
            logger.warning(f"Invalid frame at frame_number={self._frame_number}")
            # Return empty result without crashing
            return PipelineFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=timestamp,
                frame_number=self._frame_number,
                camera_id=self.config.camera_name,
                skip_person_detection=True
            )

        # Create pipeline frame
        pf = PipelineFrame(
            frame=frame,
            timestamp=timestamp,
            frame_number=self._frame_number,
            camera_id=self.config.camera_name
        )

        # Update frame buffer
        self._frame_buffer.append(frame.copy())
        if len(self._frame_buffer) > self._max_buffer_size:
            self._frame_buffer.pop(0)

        # Stage 1: Motion Detection (<1ms)
        try:
            motion_start = time.time()
            motion_result = self.motion_detector.detect(frame)
            motion_ms = (time.time() - motion_start) * 1000
            pf.motion = motion_result
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            pf.motion = MotionResult(detected=False, magnitude=0.0)
            motion_ms = 0.0

        # Edge case: Skip if lighting change (magnitude too high)
        if pf.motion.magnitude > self.config.motion_magnitude_threshold:
            logger.debug(f"Skipping frame: motion magnitude {pf.motion.magnitude:.2f} > threshold (lighting change?)")
            pf.skip_person_detection = True
            self._motion_debounce_count = 0  # Reset debounce on lighting change
            pf.total_process_time_ms = (time.time() - start_time) * 1000
            return pf

        # Edge case: Camera shake filter - require consecutive motion frames
        if pf.motion.detected:
            self._motion_debounce_count += 1
        else:
            self._motion_debounce_count = 0

        # Only consider motion "real" after debounce threshold met
        effective_motion = (self._motion_debounce_count >= self.config.motion_debounce_frames)

        # Edge case: Skip if no motion and no active tracks
        active_tracks = self.tracker.get_active_tracks()
        if not effective_motion and not active_tracks:
            self._skip_countdown += 1
            if self._skip_countdown < self.config.skip_frames_on_empty:
                pf.skip_person_detection = True
                self.stats.frames_skipped += 1
                pf.total_process_time_ms = (time.time() - start_time) * 1000
                return pf
        self._skip_countdown = 0

        if effective_motion:
            self._last_motion_time = timestamp

        # Stage 2: Person Detection (~10ms)
        try:
            person_start = time.time()
            person_result = self.person_detector.detect(frame)
            person_ms = (time.time() - person_start) * 1000
            pf.persons = person_result
            self.stats.total_persons_detected += person_result.count
        except Exception as e:
            logger.error(f"Person detection error: {e}")
            person_result = PersonResult(detections=(), timestamp=timestamp)
            pf.persons = person_result
            person_ms = 0.0

        # Stage 3: Object Tracking (<1ms)
        try:
            self.tracker.update(person_result, timestamp)
        except Exception as e:
            logger.error(f"Tracker update error: {e}")

        # Get events
        events = self.tracker.get_events()
        pf.events = events
        self.stats.total_events += len(events)

        # Stage 4: Face Recognition (on NEW events only)
        face_ms = 0.0
        face_result = None

        if self._pending_face_checks:
            face_start = time.time()
            face_result = self._process_face_recognition(frame, timestamp)
            face_ms = (time.time() - face_start) * 1000
            pf.faces = face_result

        # Stage 5: Handle greetings for unknown persons
        self._handle_unknown_greetings(events)

        # Update stats
        pf.total_process_time_ms = (time.time() - start_time) * 1000
        self.stats.frames_processed += 1
        self.stats.update_timing(motion_ms, person_ms, face_ms, pf.total_process_time_ms)

        tracker_stats = self.tracker.get_statistics()
        self.stats.active_tracks = tracker_stats["active_tracks"]
        self.stats.total_tracks = tracker_stats["total_tracks_created"]

        return pf

    def _process_face_recognition(self, frame: np.ndarray, timestamp: float) -> Optional[FaceResult]:
        """
        Process face recognition for pending tracks.

        Implements retry logic from architecture:
        - Retry up to max_retries for best quality
        - Stop on first confident match
        """
        if not self._pending_face_checks:
            return None

        face_result = self.face_detector.detect_and_recognize(frame)

        if not face_result.detected:
            # Increment attempts for all pending
            to_remove = []
            for track_id, attempts in self._pending_face_checks.items():
                attempts += 1
                self._pending_face_checks[track_id] = attempts
                if attempts >= self.config.face.max_retries:
                    to_remove.append(track_id)
                    logger.debug(f"Face check timeout for {track_id}")

            for tid in to_remove:
                del self._pending_face_checks[tid]

            return face_result

        # Match faces to tracks
        confirmed_tracks = self.tracker.get_confirmed_tracks()

        for face in face_result.faces:
            # Find track that contains this face (face should be inside person box)
            matching_track = None
            best_containment = 0.0

            for track in confirmed_tracks:
                if track.box is None:
                    continue

                # Check if face center is inside person box (better than IoU for face-in-person matching)
                face_center = face.box.center
                if track.box.contains_point(face_center[0], face_center[1]):
                    # Calculate how well contained the face is
                    # Prefer tracks where face is more centered
                    person_center = track.box.center
                    dx = abs(face_center[0] - person_center[0]) / max(track.box.width, 1)
                    dy = abs(face_center[1] - person_center[1]) / max(track.box.height, 1)
                    containment = 1.0 - (dx + dy) / 2  # Higher is better
                    if containment > best_containment:
                        best_containment = containment
                        matching_track = track

            if matching_track and matching_track.track_id in self._pending_face_checks:
                if face.is_known:
                    # Found known face - update track and remove from pending
                    self.tracker.set_face_identity(
                        matching_track.track_id,
                        face.identity,
                        face.identity_confidence,
                        frame[face.box.y1:face.box.y2, face.box.x1:face.box.x2]
                    )
                    del self._pending_face_checks[matching_track.track_id]
                    self.stats.total_faces_recognized += 1
                    logger.info(f"Recognized {face.identity} for track {matching_track.track_id}")
                else:
                    # Unknown face - increment attempts
                    attempts = self._pending_face_checks[matching_track.track_id] + 1
                    self._pending_face_checks[matching_track.track_id] = attempts

                    if attempts >= self.config.face.max_retries:
                        # Give up - mark as Unknown
                        self.tracker.set_face_identity(
                            matching_track.track_id,
                            "Unknown",
                            0.0
                        )
                        del self._pending_face_checks[matching_track.track_id]

        return face_result

    def _handle_unknown_greetings(self, events: List[DetectionEvent]) -> None:
        """Greet unknown visitors after face detection timeout."""
        for event in events:
            if event.lifecycle != EventLifecycle.NEW:
                continue

            track_id = event.track_id
            if track_id in self._greeted_tracks:
                continue

            # Edge case: Filter very short visits (< min_event_duration)
            if event.duration < self.config.min_event_duration:
                logger.debug(f"Skipping greeting for short event: {event.duration:.2f}s < {self.config.min_event_duration}s")
                continue

            # Check if face check is complete (not in pending)
            if track_id not in self._pending_face_checks:
                track = self.tracker.get_track(track_id)
                if track and not track.face_name:
                    # Unknown visitor
                    self._greeted_tracks.add(track_id)
                    if self.voice:
                        self.voice.speak_async(get_response("greeting"))
                    logger.info(f"Greeted unknown visitor (track {track_id})")

    def run(self, show_video: bool = True, max_frames: int = 0) -> None:
        """
        Run main processing loop.

        Args:
            show_video: Whether to show video window
            max_frames: Max frames to process (0 = unlimited)
        """
        self._running = True

        if show_video:
            logger.info("Controls: 'r' register, 's' save, 'q' quit")
        else:
            logger.info("Headless mode - console output")

        fps_time = time.time()
        fps_count = 0
        fps = 0

        try:
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    self._camera_error_count += 1
                    logger.warning(f"Could not read frame (error {self._camera_error_count}/{self._max_camera_errors})")

                    if self._camera_error_count >= self._max_camera_errors:
                        logger.error("Max camera errors reached, attempting reconnection...")
                        # Try to reconnect
                        self.cap.release()
                        time.sleep(1.0)  # Wait before retry
                        self.cap = cv2.VideoCapture(self._camera_id)
                        if self.cap.isOpened():
                            logger.info("Camera reconnected successfully")
                            self._camera_error_count = 0
                        else:
                            logger.error("Camera reconnection failed, exiting")
                            break
                    continue  # Skip this frame and try again

                # Reset error count on successful read
                self._camera_error_count = 0

                # Process frame
                pf = self.process_frame(frame)

                # FPS calculation
                fps_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_count
                    fps_count = 0
                    fps_time = time.time()

                    # Print status in headless mode
                    if not show_video:
                        self._print_status(fps)

                # Check max frames
                if max_frames > 0 and self._frame_number >= max_frames:
                    logger.info(f"Reached {max_frames} frames")
                    break

                # Video display
                if show_video:
                    annotated = self._draw_annotations(frame, pf, fps)
                    cv2.imshow("Production Pipeline", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self._register_face(frame)
                    elif key == ord('s'):
                        self._save_faces()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def _print_status(self, fps: int) -> None:
        """Print status to console."""
        ts = datetime.now().strftime("%H:%M:%S")
        tracks = self.tracker.get_active_tracks()
        names = [t.face_name or "?" for t in tracks]
        names_str = ", ".join(names) if names else "-"

        print(f"[{ts}] {fps}fps | "
              f"Motion:{self.stats.avg_motion_ms:.1f}ms | "
              f"Person:{self.stats.avg_person_ms:.1f}ms | "
              f"Tracks:{len(tracks)} | "
              f"Faces:[{names_str}]")

    def _draw_annotations(self, frame: np.ndarray, pf: PipelineFrame, fps: int) -> np.ndarray:
        """Draw annotations on frame."""
        annotated = frame.copy()

        # Draw person boxes
        if pf.persons:
            for det in pf.persons.detections:
                color = (0, 255, 0)  # Green
                cv2.rectangle(
                    annotated,
                    (det.box.x1, det.box.y1),
                    (det.box.x2, det.box.y2),
                    color, 2
                )

        # Draw tracked objects with names
        for track in self.tracker.get_active_tracks():
            if track.box is None:
                continue

            # Color based on state
            color = {
                ObjectState.DETECTING: (0, 255, 255),  # Yellow
                ObjectState.ACTIVE: (0, 255, 0),       # Green
                ObjectState.STATIONARY: (255, 165, 0), # Orange
                ObjectState.LOST: (0, 0, 255),         # Red
            }.get(track.state, (255, 255, 255))

            cv2.rectangle(
                annotated,
                (track.box.x1, track.box.y1),
                (track.box.x2, track.box.y2),
                color, 2
            )

            # Label
            label = track.face_name or track.track_id[:8]
            label += f" ({track.state.name[:3]})"
            cv2.putText(
                annotated, label,
                (track.box.x1, track.box.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2
            )

        # Status overlay
        lines = [
            f"FPS: {fps}",
            f"Tracks: {len(self.tracker.get_active_tracks())}",
            f"Motion: {pf.motion.detected if pf.motion else False}",
            f"Process: {pf.total_process_time_ms:.1f}ms",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                annotated, line,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )

        return annotated

    def _register_face(self, frame: np.ndarray) -> None:
        """Register a face from current frame."""
        name = input("Enter name: ").strip()
        if name and self._raw_face.register_face(frame, name):
            print(f"Registered: {name}")
            if self.voice:
                self.voice.speak_async(f"Registered {name}")
        else:
            print("Could not register (need exactly 1 face)")

    def _save_faces(self) -> None:
        """Save face database."""
        path = self.config.face.database_path
        self._raw_face.database.save(path)
        print(f"Saved {self._raw_face.database.count} faces to {path}")

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down pipeline...")

        self._running = False

        # End all tracks
        end_events = self.tracker.end_all_tracks()
        logger.info(f"Ended {len(end_events)} tracks")

        # Stop voice
        if self.voice:
            self.voice.stop()

        # Release camera
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        logger.info("Pipeline shutdown complete")

    def get_status(self) -> dict:
        """Get current pipeline status."""
        tracks = self.tracker.get_active_tracks()
        return {
            "running": self._running,
            "frame_number": self._frame_number,
            "active_tracks": len(tracks),
            "track_names": [t.face_name for t in tracks if t.face_name],
            "stats": {
                "frames_processed": self.stats.frames_processed,
                "frames_skipped": self.stats.frames_skipped,
                "avg_motion_ms": self.stats.avg_motion_ms,
                "avg_person_ms": self.stats.avg_person_ms,
                "avg_face_ms": self.stats.avg_face_ms,
                "avg_total_ms": self.stats.avg_total_ms,
            }
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Production Detection Pipeline")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--headless", action="store_true", help="Run without video window")
    parser.add_argument("--frames", "-n", type=int, default=0, help="Max frames (0=unlimited)")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice output")
    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        camera_id=args.camera,
        enable_voice=not args.no_voice
    )

    try:
        pipeline = ProductionPipeline(config=config, camera_id=args.camera)
        pipeline.run(show_video=not args.headless, max_frames=args.frames)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
