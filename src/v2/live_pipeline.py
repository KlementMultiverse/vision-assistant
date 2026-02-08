#!/usr/bin/env python3
"""
Live Vision Pipeline
=====================
Fast real-time recognition with tagging UI.

Shows name instantly. Press 't' to tag current person.

Run: python -m src.v2.live_pipeline
"""

import cv2
import numpy as np
import time
from typing import Optional
from datetime import datetime

from src.v2.storage.schema import VisionDB, Person


class LivePipeline:
    """
    Fast live recognition with tagging.

    Features:
    - Instant name display
    - Press 't' to tag current person
    - Auto-learning on different conditions
    """

    def __init__(self, db_path: str = "vision.db", threshold: float = 0.25):
        print("=" * 60)
        print("  LIVE VISION PIPELINE")
        print("=" * 60)

        print("\nLoading...")

        self.db = VisionDB(db_path=db_path, match_threshold=threshold)
        print(f"  Database: {self.db.person_count} persons")

        from insightface.app import FaceAnalysis
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640))

        # State
        self.current_person: Optional[Person] = None
        self.current_embedding: Optional[np.ndarray] = None
        self.current_confidence: float = 0
        self.last_seen: float = 0
        self.auto_learned: bool = False

        # Tagging mode
        self.tagging: bool = False
        self.tag_step: int = 0  # 0=name, 1=group, 2=role
        self.tag_name: str = ""
        self.tag_group: str = ""
        self.tag_role: str = ""

        print("\nControls:")
        print("  't' - Tag current person")
        print("  'l' - List all persons")
        print("  'q' - Quit")
        print()

    def run(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        fps_time = time.time()
        fps_count = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            h, w = frame.shape[:2]
            now = time.time()

            # Detect faces
            faces = self.face_app.get(frame)

            if faces:
                face = faces[0]
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding
                conf = float(face.det_score)

                self.current_embedding = embedding
                self.current_confidence = conf
                self.last_seen = now

                # Identify
                person, sim, _ = self.db.identify(embedding)

                if person:
                    self.current_person = person

                    # Display name with role
                    if person.role:
                        label = f"{person.display_name or person.name} ({person.role})"
                    else:
                        label = person.display_name or person.name

                    color = (0, 255, 0) if person.group_type == "family" else \
                            (255, 200, 0) if person.group_type == "friends" else \
                            (0, 255, 255)

                    # Auto-learn
                    if not self.auto_learned and conf >= 0.5:
                        added = self.db.auto_learn(person.id, embedding, conf)
                        if added:
                            self.auto_learned = True
                            print(f"  + Auto-learned: {person.name}")

                else:
                    # Unknown - create on the fly
                    if self.current_person is None or self.current_person.name.startswith("unknown"):
                        person_id, name = self.db.create_unknown([(embedding, conf)])
                        self.current_person = self.db.get_person(person_id)
                        print(f"  New: {name}")

                    label = self.current_person.name if self.current_person else "Unknown"
                    color = (0, 255, 255)

                # Draw
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(display, label, (bbox[0], bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"{sim:.2f}" if sim else "", (bbox[0], bbox[3]+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            else:
                # No face
                if now - self.last_seen > 5:
                    self.current_person = None
                    self.auto_learned = False

            # FPS
            fps_count += 1
            if now - fps_time >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = now

            # Status bar
            cv2.putText(display, f"FPS: {fps} | Persons: {self.db.person_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if self.current_person and not self.tagging:
                cv2.putText(display, "Press 't' to tag", (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Tagging UI overlay
            if self.tagging:
                self._draw_tagging_ui(display)

            cv2.imshow("Live Pipeline", display)

            key = cv2.waitKey(1) & 0xFF

            if self.tagging:
                self._handle_tagging_input(key)
            else:
                if key == ord('q'):
                    break
                elif key == ord('t') and self.current_person:
                    self.tagging = True
                    self.tag_step = 0
                    self.tag_name = ""
                    self.tag_group = ""
                    self.tag_role = ""
                elif key == ord('l'):
                    self._list_persons()

        self.db.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

    def _draw_tagging_ui(self, frame):
        h, w = frame.shape[:2]

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, h//2-100), (w-50, h//2+120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        y = h//2 - 70

        cv2.putText(frame, "TAG PERSON", (w//2-80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 40

        if self.tag_step == 0:
            cv2.putText(frame, f"Name: {self.tag_name}_", (100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "(Enter to confirm, ESC to cancel)", (100, y+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        elif self.tag_step == 1:
            cv2.putText(frame, f"Name: {self.tag_name}", (100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 35
            cv2.putText(frame, "Group: [1] Family  [2] Friends  [3] Public", (100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif self.tag_step == 2:
            cv2.putText(frame, f"Name: {self.tag_name}", (100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(frame, f"Group: {self.tag_group}", (100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 35
            cv2.putText(frame, f"Role (optional): {self.tag_role}_", (100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "(e.g., daughter, son, delivery_guy, school_friend)", (100, y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.putText(frame, "(Enter to save, ESC to skip role)", (100, y+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _handle_tagging_input(self, key):
        if key == 27:  # ESC
            if self.tag_step == 2:
                # Save without role
                self._save_tag()
            else:
                self.tagging = False
            return

        if self.tag_step == 0:
            # Name input
            if key == 13:  # Enter
                if self.tag_name:
                    self.tag_step = 1
            elif key == 8:  # Backspace
                self.tag_name = self.tag_name[:-1]
            elif 32 <= key <= 126:
                self.tag_name += chr(key)

        elif self.tag_step == 1:
            # Group selection
            if key == ord('1'):
                self.tag_group = "family"
                self.tag_step = 2
            elif key == ord('2'):
                self.tag_group = "friends"
                self.tag_step = 2
            elif key == ord('3'):
                self.tag_group = "public"
                self.tag_step = 2

        elif self.tag_step == 2:
            # Role input
            if key == 13:  # Enter
                self._save_tag()
            elif key == 8:  # Backspace
                self.tag_role = self.tag_role[:-1]
            elif 32 <= key <= 126:
                self.tag_role += chr(key)

    def _save_tag(self):
        if self.current_person and self.tag_name and self.tag_group:
            role = self.tag_role if self.tag_role else None
            success = self.db.tag_person(
                self.current_person.id,
                self.tag_name,
                self.tag_group,
                role
            )
            if success:
                print(f"\n✅ Tagged: {self.tag_name} ({self.tag_group}" +
                      (f", {role})" if role else ")"))
                self.current_person = self.db.get_person(self.current_person.id)
            else:
                print(f"\n❌ Failed: Name '{self.tag_name}' already exists")

        self.tagging = False

    def _list_persons(self):
        print("\n" + "=" * 50)
        print("  REGISTERED PERSONS")
        print("=" * 50)
        for p in self.db.list_persons():
            emb = self.db.get_embedding_count(p.id)
            role = f" ({p.role})" if p.role else ""
            print(f"  {p.name}{role} [{p.group_type}] - {emb} embeddings, {p.visit_count} visits")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="vision.db")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    pipeline = LivePipeline(db_path=args.db, threshold=args.threshold)
    pipeline.run(camera_id=args.camera)
