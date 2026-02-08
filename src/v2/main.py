#!/usr/bin/env python3
"""
Vision Assistant v2 - POC Integration
======================================
Combines all perception components into a working doorbell.

Pipeline:
  Camera → Motion → Person → Face → Voice

Run:
  python -m src.v2.main
"""

import cv2
import time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Import POC components
from src.v2.perception.motion.base import SimpleMotionDetector
from src.v2.perception.person.base import SimplePersonDetector
from src.v2.perception.face.base import SimpleFaceDetector
from src.v2.understanding.voice import SimpleVoice, get_response


@dataclass
class DetectionState:
    """Track what we've detected."""
    motion: bool = False
    person_count: int = 0
    known_faces: list = None
    unknown_faces: int = 0
    last_greeting: float = 0

    def __post_init__(self):
        if self.known_faces is None:
            self.known_faces = []

    def should_greet(self) -> bool:
        """Should we greet the visitor?"""
        # Greet if: person present, not greeted in last 30s
        if self.person_count == 0:
            return False
        if time.time() - self.last_greeting < 30:
            return False
        return True


class VisionAssistant:
    """
    POC Vision Assistant.

    Processes camera frames through perception pipeline.
    """

    def __init__(self, camera_id: int = 0):
        print("=" * 60)
        print("  VISION ASSISTANT v2 - POC")
        print("=" * 60)
        print()

        print("Loading components...")

        # Initialize components
        print("  [1/4] Motion detector...")
        self.motion = SimpleMotionDetector(threshold=25, min_area=500)

        print("  [2/4] Person detector (YOLOv8n)...")
        self.person = SimplePersonDetector(confidence=0.5)

        print("  [3/4] Face detector (InsightFace)...")
        self.face = SimpleFaceDetector(det_thresh=0.5)

        print("  [4/4] Voice output (pyttsx3)...")
        self.voice = SimpleVoice()

        # Camera
        print()
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        # State
        self.state = DetectionState()

        print()
        print("Ready!")
        print()

    def process_frame(self, frame) -> dict:
        """
        Process a single frame through the pipeline.

        Returns:
            dict with detection results
        """
        results = {
            "motion": False,
            "persons": 0,
            "faces": [],
            "action": None,
        }

        # Step 1: Motion detection (fast, filters quiet frames)
        motion_result = self.motion.detect(frame)
        results["motion"] = motion_result.detected

        if not motion_result.detected:
            self.state.motion = False
            return results

        self.state.motion = True

        # Step 2: Person detection (only if motion)
        person_result = self.person.detect(frame)
        results["persons"] = person_result.count
        self.state.person_count = person_result.count

        if person_result.count == 0:
            return results

        # Step 3: Face detection (only if person)
        face_result = self.face.detect(frame)
        results["faces"] = [
            {
                "name": f.name or "Unknown",
                "confidence": f.confidence,
                "similarity": f.similarity,
            }
            for f in face_result.faces
        ]

        # Update state
        self.state.known_faces = [f.name for f in face_result.faces if f.name]
        self.state.unknown_faces = sum(1 for f in face_result.faces if not f.name)

        # Step 4: Decide action
        if self.state.should_greet():
            if self.state.known_faces:
                # Known person
                name = self.state.known_faces[0]
                results["action"] = f"greet_known:{name}"
                self.voice.speak_async(f"Welcome back, {name}!")
            else:
                # Unknown person
                results["action"] = "greet_unknown"
                self.voice.speak_async(get_response("greeting"))

            self.state.last_greeting = time.time()

        return results

    def run(self, show_video: bool = True, max_frames: int = 0):
        """
        Run the main loop.

        Args:
            show_video: Whether to display video window
            max_frames: Stop after N frames (0 = unlimited)
        """
        if show_video:
            print("Controls:")
            print("  'r' - Register face")
            print("  's' - Save face database")
            print("  'q' - Quit")
        else:
            print("Running in headless mode (console output only)")
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

            frame_num += 1

            # Process
            results = self.process_frame(frame)

            # Calculate FPS
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

                # Print status every second in headless mode
                if not show_video:
                    ts = datetime.now().strftime("%H:%M:%S")
                    faces_str = ", ".join(f["name"] for f in results["faces"]) if results["faces"] else "-"
                    action_str = results["action"] or "-"
                    print(f"[{ts}] FPS:{fps} Motion:{results['motion']} Persons:{results['persons']} Faces:[{faces_str}] Action:{action_str}")

            # Check max frames
            if max_frames > 0 and frame_num >= max_frames:
                print(f"\nReached {max_frames} frames, stopping.")
                break

            # Draw annotations
            if show_video:
                annotated = frame.copy()

                # Draw person boxes (from person detector)
                person_result = self.person.detect(frame)
                for box in person_result.boxes:
                    cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)

                # Status overlay
                status = [
                    f"FPS: {fps}",
                    f"Motion: {results['motion']}",
                    f"Persons: {results['persons']}",
                    f"Faces: {len(results['faces'])}",
                ]
                if results["action"]:
                    status.append(f"Action: {results['action']}")

                for i, line in enumerate(status):
                    cv2.putText(
                        annotated, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                cv2.imshow("Vision Assistant v2", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    name = input("Enter name for face: ").strip()
                    if name and self.face.register_face(frame, name):
                        print(f"Registered: {name}")
                        self.voice.speak_async(f"Registered {name}")
                    else:
                        print("Could not register face")
                elif key == ord('s'):
                    self.face.database.save("faces.npz")
                    print(f"Saved {self.face.database.count} faces")

        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nShutting down...")
        self.voice.speak("Goodbye!")
        time.sleep(1)
        self.voice.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done!")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Vision Assistant v2 POC")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--headless", action="store_true", help="Run without video display")
    parser.add_argument("--frames", "-n", type=int, default=0, help="Stop after N frames (0=unlimited)")
    args = parser.parse_args()

    try:
        assistant = VisionAssistant(camera_id=args.camera)
        assistant.run(show_video=not args.headless, max_frames=args.frames)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
