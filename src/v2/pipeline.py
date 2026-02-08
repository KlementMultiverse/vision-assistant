#!/usr/bin/env python3
"""
Vision Pipeline - Production Grade
===================================
Smart detection pipeline with:
- Debounced state transitions
- Face detection only on entry
- Proper event handling
- Optimized for real-world use

Run:
  python -m src.v2.pipeline --headless
"""

import cv2
import time
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

# Import components
from src.v2.perception.motion.base import SimpleMotionDetector
from src.v2.perception.person.base import SimplePersonDetector
from src.v2.perception.face.base import SimpleFaceDetector
from src.v2.understanding.voice import SimpleVoice, get_response
from src.v2.intelligence.state import (
    PresenceTracker, PresenceState, EventType, Event, DetectionFrame
)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Detection thresholds
    motion_threshold: int = 25
    motion_min_area: int = 500
    person_confidence: float = 0.5
    face_threshold: float = 0.5
    face_match_threshold: float = 0.5

    # State machine
    confirm_frames: int = 3      # Frames to confirm person
    leave_timeout: float = 5.0   # Seconds before "left"

    # Optimization
    face_on_entry_only: bool = True   # Only detect face when person enters
    skip_frames_no_motion: int = 5     # Skip N frames when no motion

    # Display
    mirror_display: bool = True   # Mirror video for natural view

    # Storage
    face_db_path: str = "faces.db"  # Path to face database


class VisionPipeline:
    """
    Production-grade vision pipeline.

    Architecture:
      Camera → Motion Filter → Person Detection → State Machine → Actions
                                      ↓
                              Face (on entry only)
    """

    def __init__(self, camera_id: int = 0, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        print("=" * 60)
        print("  VISION PIPELINE - Production Grade")
        print("=" * 60)
        print()

        # Initialize components
        print("Loading components...")

        print("  [1/5] Motion detector...")
        self.motion = SimpleMotionDetector(
            threshold=self.config.motion_threshold,
            min_area=self.config.motion_min_area
        )

        print("  [2/5] Person detector (YOLOv8n)...")
        self.person = SimplePersonDetector(confidence=self.config.person_confidence)

        print("  [3/5] Face detector (InsightFace)...")
        self.face = SimpleFaceDetector(
            det_thresh=self.config.face_threshold,
            db_path=self.config.face_db_path,
            match_threshold=self.config.face_match_threshold
        )
        if self.face.database.count > 0:
            print(f"        Loaded {self.face.database.count} faces: {self.face.database.names}")

        print("  [4/5] Voice output...")
        self.voice = SimpleVoice()

        print("  [5/5] State machine...")
        self.tracker = PresenceTracker(
            confirm_frames=self.config.confirm_frames,
            leave_timeout=self.config.leave_timeout
        )

        # Register event handlers
        self.tracker.on_event(self._on_event)

        # Camera
        print()
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        # Stats
        self._frame_count = 0
        self._skip_count = 0
        self._last_face_check = 0
        self._pending_face_check = False

        print()
        print("Ready!")
        print()

    def _on_event(self, event: Event):
        """Handle state machine events."""
        if event.type == EventType.PERSON_ENTERED:
            faces = event.data.get("faces", [])
            if faces and faces[0] != "Unknown":
                # Known person
                name = faces[0]
                self.voice.speak_async(f"Welcome back, {name}!")
                print(f"  → Greeting known: {name}")
            else:
                # Unknown person - trigger face check
                self._pending_face_check = True

        elif event.type == EventType.PERSON_IDENTIFIED:
            faces = event.data.get("faces", [])
            if faces:
                name = faces[0]
                self.voice.speak_async(f"Hello, {name}!")
                print(f"  → Identified: {name}")

        elif event.type == EventType.PERSON_LEFT:
            faces = event.data.get("faces", [])
            name = faces[0] if faces else "visitor"
            print(f"  → {name} left")

    def process_frame(self, frame) -> dict:
        """
        Process single frame through pipeline.

        Returns detection results.
        """
        now = time.time()
        self._frame_count += 1

        results = {
            "motion": False,
            "persons": 0,
            "faces": [],
            "state": self.tracker.state.name,
            "skipped": False,
        }

        # Step 1: Motion detection (always runs, very fast)
        motion_result = self.motion.detect(frame)
        results["motion"] = motion_result.detected

        # Optimization: Skip processing if no motion and state is EMPTY
        if not motion_result.detected and self.tracker.state == PresenceState.EMPTY:
            self._skip_count += 1
            if self._skip_count < self.config.skip_frames_no_motion:
                results["skipped"] = True
                # Still update state machine with empty detection
                self.tracker.update(DetectionFrame(
                    timestamp=now,
                    has_motion=False,
                    person_count=0
                ))
                return results
        self._skip_count = 0

        # Step 2: Person detection
        person_result = self.person.detect(frame)
        results["persons"] = person_result.count

        # Step 3: Face detection (smart - only when needed)
        face_names = []
        should_check_face = (
            self._pending_face_check or
            (not self.config.face_on_entry_only and person_result.count > 0)
        )

        if should_check_face and person_result.count > 0:
            face_result = self.face.detect(frame)
            face_names = [f.name or "Unknown" for f in face_result.faces]
            results["faces"] = face_names
            self._pending_face_check = False
            self._last_face_check = now

        # Step 4: Update state machine
        detection = DetectionFrame(
            timestamp=now,
            has_motion=motion_result.detected,
            person_count=person_result.count,
            face_names=face_names
        )
        self.tracker.update(detection)
        results["state"] = self.tracker.state.name

        # Step 5: Handle greeting (only once per visit)
        if self.tracker.should_greet:
            if not face_names or all(f == "Unknown" for f in face_names):
                self.voice.speak_async(get_response("greeting"))
                print("  → Greeting unknown visitor")
            self.tracker.mark_greeted()

        return results

    def run(self, show_video: bool = True, max_frames: int = 0):
        """Run main loop."""
        if show_video:
            print("Controls: 'r' register, 's' save, 'q' quit")
        else:
            print("Headless mode - console output")
        print()

        fps_time = time.time()
        fps_count = 0
        fps = 0
        frame_num = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("ERROR: Could not read frame")
                break

            # Mirror flip for natural view
            if self.config.mirror_display:
                frame = cv2.flip(frame, 1)

            frame_num += 1

            # Process
            results = self.process_frame(frame)

            # FPS calculation
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

                # Print status every second
                if not show_video:
                    ts = datetime.now().strftime("%H:%M:%S")
                    status = self.tracker.get_status()
                    faces_str = ", ".join(status["faces"]) if status["faces"] else "-"
                    print(f"[{ts}] {fps}fps | State:{status['state']:10} | "
                          f"Motion:{status['motion']} | Faces:[{faces_str}]")

            # Max frames check
            if max_frames > 0 and frame_num >= max_frames:
                print(f"\nReached {max_frames} frames.")
                break

            # Video display
            if show_video:
                annotated = frame.copy()

                # Draw boxes
                if results["persons"] > 0:
                    person_result = self.person.detect(frame)
                    for box in person_result.boxes:
                        color = (0, 255, 0) if self.tracker.state == PresenceState.PRESENT else (0, 255, 255)
                        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)

                # Status overlay
                status = self.tracker.get_status()
                lines = [
                    f"FPS: {fps}",
                    f"State: {status['state']}",
                    f"Motion: {status['motion']}",
                    f"Faces: {status['faces']}",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(annotated, line, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Vision Pipeline", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    name = input("Name: ").strip()
                    face_id = self.face.register_face(frame, name) if name else None
                    if face_id:
                        print(f"Registered: {name} (ID: {face_id})")
                        self.voice.speak_async(f"Registered {name}")
                elif key == ord('s'):
                    # List all registered faces (SQLite auto-saves)
                    people = self.face.database.list_all()
                    print(f"\nRegistered faces ({len(people)}):")
                    for p in people:
                        print(f"  - {p['name']}: {p['embeddings']} embeddings, seen {p['total_seen']} times")
                    print()

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nShutting down...")
        self.voice.stop()
        self.face.database.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done!")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision Pipeline - Production")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", "-n", type=int, default=0)
    args = parser.parse_args()

    try:
        pipeline = VisionPipeline(camera_id=args.camera)
        pipeline.run(show_video=not args.headless, max_frames=args.frames)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        raise
