#!/usr/bin/env python3
"""
Smart Vision Pipeline
======================
Production pipeline with:
- Auto-learning (adds new embeddings when conditions change)
- Visit tracking
- Unknown auto-tagging
- Voice greetings

Run: python -m src.v2.smart_pipeline
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

from src.v2.storage.schema import VisionDB, Person
from src.v2.perception.motion.base import SimpleMotionDetector
from src.v2.perception.person.base import SimplePersonDetector
from src.v2.understanding.voice import SimpleVoice


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Detection
    motion_threshold: int = 25
    motion_min_area: int = 500
    person_confidence: float = 0.5
    face_det_threshold: float = 0.4

    # Recognition
    match_threshold: float = 0.25
    min_detection_confidence: float = 0.5

    # Auto-learning
    auto_learn_enabled: bool = True
    auto_learn_min_difference: float = 0.3
    max_embeddings_per_person: int = 10

    # Behavior
    leave_timeout: float = 5.0
    greet_known: bool = True
    greet_unknown: bool = True

    # Display
    mirror_display: bool = True

    # Storage
    db_path: str = "vision.db"


class SmartPipeline:
    """
    Production vision pipeline with auto-learning.

    Features:
    - Recognizes known people
    - Auto-creates unknowns
    - Learns new embeddings over time
    - Tracks visits
    - Voice greetings
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        print("=" * 60)
        print("  SMART VISION PIPELINE")
        print("=" * 60)
        print()

        # Initialize components
        print("Loading components...")

        print("  [1/5] Database...")
        self.db = VisionDB(
            db_path=self.config.db_path,
            match_threshold=self.config.match_threshold,
            min_detection_confidence=self.config.min_detection_confidence
        )
        print(f"        {self.db.person_count} persons loaded")

        print("  [2/5] Motion detector...")
        self.motion = SimpleMotionDetector(
            threshold=self.config.motion_threshold,
            min_area=self.config.motion_min_area
        )

        print("  [3/5] Person detector...")
        self.person = SimplePersonDetector(confidence=self.config.person_confidence)

        print("  [4/5] Face detector...")
        from insightface.app import FaceAnalysis
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(
            ctx_id=0,
            det_thresh=self.config.face_det_threshold,
            det_size=(640, 640)
        )

        print("  [5/5] Voice...")
        self.voice = SimpleVoice()

        # State
        self._current_person: Optional[Person] = None
        self._current_visit_id: Optional[int] = None
        self._last_seen_time: float = 0
        self._greeted: bool = False
        self._frames_this_visit: int = 0
        self._embeddings_this_visit: List[tuple] = []
        self._auto_learned_this_visit: bool = False

        print()
        print("Ready!")
        print()

    def _on_person_enter(self, person: Optional[Person], embedding: np.ndarray, confidence: float):
        """Handle person entering."""
        self._greeted = False
        self._frames_this_visit = 0
        self._embeddings_this_visit = [(embedding, confidence)]
        self._auto_learned_this_visit = False

        if person:
            # Known person
            self._current_person = person
            self._current_visit_id = self.db.start_visit(person.id)

            if self.config.greet_known:
                name = person.display_name or person.name
                if person.group_type == "family":
                    self.voice.speak_async(f"Welcome home, {name}!")
                else:
                    self.voice.speak_async(f"Hello, {name}!")
                self._greeted = True

            print(f"  → {person.name} entered (visit #{self._current_visit_id})")
        else:
            # Unknown person - create new
            person_id, name = self.db.create_unknown(
                [(embedding, confidence)],
                vision_context="auto-detected"
            )
            self._current_person = self.db.get_person(person_id)
            self._current_visit_id = self.db.start_visit(person_id)

            if self.config.greet_unknown:
                self.voice.speak_async("Hello!")
                self._greeted = True

            print(f"  → New person: {name} (visit #{self._current_visit_id})")

    def _on_person_present(self, person: Optional[Person], embedding: np.ndarray, confidence: float, similarity: float):
        """Handle person still present - auto-learn if needed."""
        self._frames_this_visit += 1
        self._embeddings_this_visit.append((embedding, confidence))
        self._last_seen_time = time.time()

        # Auto-learn: add embedding if different enough (once per visit)
        if (self.config.auto_learn_enabled and
            person and
            not self._auto_learned_this_visit and
            confidence >= self.config.min_detection_confidence):

            added = self.db.auto_learn(
                person.id,
                embedding,
                confidence,
                min_difference=self.config.auto_learn_min_difference,
                max_embeddings=self.config.max_embeddings_per_person
            )
            if added:
                self._auto_learned_this_visit = True
                print(f"  + Auto-learned new embedding for {person.name}")

    def _on_person_leave(self):
        """Handle person leaving."""
        if self._current_visit_id and self._current_person:
            # End visit
            avg_conf = np.mean([e[1] for e in self._embeddings_this_visit]) if self._embeddings_this_visit else 0
            self.db.end_visit(
                self._current_visit_id,
                frame_count=self._frames_this_visit,
                avg_confidence=avg_conf
            )
            print(f"  ← {self._current_person.name} left (frames: {self._frames_this_visit})")

        self._current_person = None
        self._current_visit_id = None
        self._greeted = False

    def run(self, camera_id: int = 0, show_video: bool = True):
        """Run main loop."""
        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        if show_video:
            print("\nControls: 'q' quit, 'l' list persons")
        else:
            print("Headless mode")
        print()

        fps_time = time.time()
        fps_count = 0
        fps = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if self.config.mirror_display:
                    frame = cv2.flip(frame, 1)

                # Detect
                faces = self.face_app.get(frame)

                current_time = time.time()

                if faces:
                    face = faces[0]
                    embedding = face.normed_embedding
                    conf = float(face.det_score)
                    bbox = face.bbox.astype(int)

                    # Try to identify
                    person, sim, _ = self.db.identify(embedding)

                    if self._current_person is None:
                        # New arrival
                        self._on_person_enter(person, embedding, conf)
                    else:
                        # Still here
                        self._on_person_present(person, embedding, conf, sim)

                    self._last_seen_time = current_time

                    # Draw
                    if show_video:
                        if person:
                            color = (0, 255, 0)
                            label = f"{person.name} ({sim:.2f})"
                        else:
                            color = (0, 255, 255)
                            label = f"Unknown ({sim:.2f})" if sim else "Unknown"

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                else:
                    # No face
                    if self._current_person and (current_time - self._last_seen_time) > self.config.leave_timeout:
                        self._on_person_leave()

                # FPS
                fps_count += 1
                if current_time - fps_time >= 1.0:
                    fps = fps_count
                    fps_count = 0
                    fps_time = current_time

                # Display
                if show_video:
                    h, w = frame.shape[:2]

                    # Status
                    if self._current_person:
                        status = f"Present: {self._current_person.name}"
                        status_color = (0, 255, 0)
                    else:
                        status = "No one"
                        status_color = (128, 128, 128)

                    cv2.putText(frame, f"FPS: {fps} | {status}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    cv2.putText(frame, f"DB: {self.db.person_count} persons | Threshold: {self.config.match_threshold}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    if self._auto_learned_this_visit:
                        cv2.putText(frame, "Learned new embedding!", (10, h-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.imshow("Smart Pipeline", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('l'):
                        print("\nRegistered persons:")
                        for p in self.db.list_persons():
                            emb_count = self.db.get_embedding_count(p.id)
                            print(f"  - {p.name} ({p.group_type}): {emb_count} embeddings, {p.visit_count} visits")
                        print()

        finally:
            if self._current_person:
                self._on_person_leave()
            self.voice.stop()
            self.db.close()
            cap.release()
            cv2.destroyAllWindows()
            print("\nShutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Vision Pipeline")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--db", type=str, default="vision.db")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    config = PipelineConfig(
        db_path=args.db,
        match_threshold=args.threshold
    )

    try:
        pipeline = SmartPipeline(config)
        pipeline.run(camera_id=args.camera, show_video=not args.headless)
    except KeyboardInterrupt:
        print("\nInterrupted")
