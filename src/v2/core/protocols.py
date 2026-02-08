#!/usr/bin/env python3
"""
Protocol definitions for swappable components.

Using Python's Protocol for structural subtyping.
This allows any class that implements these methods to be used,
without requiring explicit inheritance.
"""

from typing import Protocol, runtime_checkable, Optional, Callable, List, Tuple
import numpy as np

from .models import (
    MotionResult, PersonResult, FaceResult,
    DetectionEvent, PipelineFrame, TrackedObject, BoundingBox
)


# =============================================================================
# DETECTOR PROTOCOLS
# =============================================================================

@runtime_checkable
class MotionDetector(Protocol):
    """Interface for motion detection."""

    def detect(self, frame: np.ndarray) -> MotionResult:
        """
        Detect motion in frame. Must be <1ms.

        Args:
            frame: BGR image from camera

        Returns:
            MotionResult with detection info
        """
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
        """
        Detect persons in frame. Target ~10ms.

        Args:
            frame: BGR image from camera

        Returns:
            PersonResult with detection info
        """
        ...

    def detect_in_region(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int]  # (x, y, w, h)
    ) -> PersonResult:
        """
        Detect persons in specific region (optimization).

        Args:
            frame: BGR image from camera
            region: (x, y, width, height) to search

        Returns:
            PersonResult with detection info (coordinates in full frame)
        """
        ...


@runtime_checkable
class FaceRecognizer(Protocol):
    """Interface for face detection and recognition."""

    def detect_and_recognize(self, frame: np.ndarray) -> FaceResult:
        """
        Detect faces and attempt recognition.

        Args:
            frame: BGR image from camera

        Returns:
            FaceResult with faces and identities
        """
        ...

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding for database storage.

        Args:
            face_image: Cropped face image

        Returns:
            512-D embedding vector or None if no face
        """
        ...

    def register_face(self, frame: np.ndarray, name: str) -> bool:
        """
        Register a new face in the database.

        Args:
            frame: Image containing exactly one face
            name: Name to associate with face

        Returns:
            True if registration successful
        """
        ...

    @property
    def database_count(self) -> int:
        """Number of registered faces."""
        ...


# =============================================================================
# TRACKER PROTOCOL
# =============================================================================

@runtime_checkable
class ObjectTracker(Protocol):
    """Interface for multi-object tracking."""

    def update(self, detections: PersonResult, timestamp: float) -> List[TrackedObject]:
        """
        Update tracks with new detections. Returns all active tracks.

        Args:
            detections: Person detections for this frame
            timestamp: Frame timestamp

        Returns:
            List of all active TrackedObject instances
        """
        ...

    def get_track(self, track_id: str) -> Optional[TrackedObject]:
        """
        Get specific track by ID.

        Args:
            track_id: The track ID to look up

        Returns:
            TrackedObject or None if not found
        """
        ...

    def get_active_tracks(self) -> List[TrackedObject]:
        """
        Get all active (non-ended) tracks.

        Returns:
            List of active TrackedObject instances
        """
        ...

    def get_events(self) -> List[DetectionEvent]:
        """
        Get and clear pending events.

        Returns:
            List of events since last call
        """
        ...


# =============================================================================
# EVENT HANDLING PROTOCOLS
# =============================================================================

@runtime_checkable
class EventHandler(Protocol):
    """Interface for handling detection events."""

    def on_event(self, event: DetectionEvent) -> None:
        """
        Handle a detection event.

        Args:
            event: The detection event to handle
        """
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Interface for executing actions in response to events."""

    def execute(self, action: str, **params) -> bool:
        """
        Execute an action. Returns True on success.

        Args:
            action: Action name to execute
            **params: Action parameters

        Returns:
            True if action executed successfully
        """
        ...

    def speak(self, text: str) -> None:
        """
        Speak text via TTS.

        Args:
            text: Text to speak
        """
        ...

    def notify(self, message: str, priority: str = "normal") -> None:
        """
        Send notification.

        Args:
            message: Notification message
            priority: Priority level (low, normal, high, critical)
        """
        ...


# =============================================================================
# PIPELINE PROTOCOL
# =============================================================================

@runtime_checkable
class Pipeline(Protocol):
    """Interface for the main detection pipeline."""

    def process_frame(self, frame: np.ndarray) -> PipelineFrame:
        """
        Process single frame through all stages.

        Args:
            frame: BGR image from camera

        Returns:
            PipelineFrame with all results
        """
        ...

    def register_event_handler(self, handler: EventHandler) -> None:
        """
        Register handler for events.

        Args:
            handler: Event handler to register
        """
        ...

    def start(self) -> None:
        """Start pipeline processing."""
        ...

    def stop(self) -> None:
        """Stop pipeline processing."""
        ...


# =============================================================================
# ADAPTER HELPERS
# =============================================================================

class MotionDetectorAdapter:
    """Adapter to wrap existing SimpleMotionDetector to match protocol."""

    def __init__(self, detector):
        """
        Wrap an existing motion detector.

        Args:
            detector: SimpleMotionDetector instance
        """
        self._detector = detector
        self._frame_count = 0
        self._calibration_frames = 30

    def detect(self, frame: np.ndarray) -> MotionResult:
        """Detect motion and convert to MotionResult."""
        import time
        start = time.time()

        result = self._detector.detect(frame)
        self._frame_count += 1

        process_time_ms = (time.time() - start) * 1000

        # Convert boxes from (x, y, w, h) to BoundingBox
        regions = tuple(
            BoundingBox.from_xywh(x, y, w, h)
            for x, y, w, h in result.boxes
        )

        return MotionResult(
            detected=result.detected,
            magnitude=result.magnitude,
            regions=regions,
            timestamp=time.time(),
            process_time_ms=process_time_ms
        )

    def reset(self) -> None:
        """Reset the detector."""
        self._detector.prev_frame = None
        self._frame_count = 0

    @property
    def is_calibrating(self) -> bool:
        """True if still calibrating."""
        return self._frame_count < self._calibration_frames


class PersonDetectorAdapter:
    """Adapter to wrap existing SimplePersonDetector to match protocol."""

    def __init__(self, detector):
        """
        Wrap an existing person detector.

        Args:
            detector: SimplePersonDetector instance
        """
        self._detector = detector

    def detect(self, frame: np.ndarray) -> PersonResult:
        """Detect persons and convert to PersonResult."""
        from .models import PersonDetection
        import time

        result = self._detector.detect(frame)

        detections = tuple(
            PersonDetection(
                box=BoundingBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2),
                confidence=b.confidence
            )
            for b in result.boxes
        )

        return PersonResult(
            detections=detections,
            timestamp=time.time(),
            process_time_ms=result.inference_ms
        )

    def detect_in_region(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> PersonResult:
        """Detect in region - for now just detect full frame."""
        # TODO: Implement region-based detection for optimization
        return self.detect(frame)


class FaceRecognizerAdapter:
    """Adapter to wrap existing SimpleFaceDetector to match protocol."""

    def __init__(self, detector):
        """
        Wrap an existing face detector.

        Args:
            detector: SimpleFaceDetector instance
        """
        self._detector = detector

    def detect_and_recognize(self, frame: np.ndarray) -> FaceResult:
        """Detect and recognize faces."""
        from .models import FaceDetection
        import time

        result = self._detector.detect(frame)

        faces = tuple(
            FaceDetection(
                box=BoundingBox(x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2),
                confidence=f.confidence,
                identity=f.name,
                identity_confidence=f.similarity,
                embedding=f.embedding
            )
            for f in result.faces
        )

        return FaceResult(
            faces=faces,
            timestamp=time.time(),
            process_time_ms=result.inference_ms
        )

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding from face image."""
        result = self._detector.detect(face_image)
        if result.faces:
            return result.faces[0].embedding
        return None

    def register_face(self, frame: np.ndarray, name: str) -> bool:
        """Register a face."""
        return self._detector.register_face(frame, name)

    @property
    def database_count(self) -> int:
        """Number of registered faces."""
        return self._detector.database.count


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing protocol definitions...")

    # Check that protocols are defined correctly
    print(f"MotionDetector is runtime_checkable: {runtime_checkable}")

    # Verify protocol methods
    import inspect
    for name, method in inspect.getmembers(MotionDetector, predicate=inspect.isfunction):
        if not name.startswith('_'):
            print(f"  MotionDetector.{name}")

    for name, method in inspect.getmembers(PersonDetector, predicate=inspect.isfunction):
        if not name.startswith('_'):
            print(f"  PersonDetector.{name}")

    for name, method in inspect.getmembers(FaceRecognizer, predicate=inspect.isfunction):
        if not name.startswith('_'):
            print(f"  FaceRecognizer.{name}")

    print("\nProtocol definitions OK!")
