#!/usr/bin/env python3
"""
Face Detection/Recognition - Production Version
================================================
InsightFace for face detection + embedding.
SQLite database for persistent face storage.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

from src.v2.storage.face_db import FaceDB


@dataclass
class FaceBox:
    """Single face detection with embedding."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    embedding: np.ndarray  # 512-D normalized embedding
    age: Optional[int] = None
    gender: Optional[str] = None  # "male" or "female"
    name: Optional[str] = None  # Identified name (if matched)
    similarity: float = 0.0  # Match similarity score

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class FaceResult:
    """What face detection returns."""
    detected: bool
    count: int
    faces: List[FaceBox]
    inference_ms: float = 0.0


class FaceDatabase:
    """Simple face database for identification."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize database.

        Args:
            threshold: Similarity threshold for matching (0.4-0.7)
        """
        self.embeddings: List[np.ndarray] = []
        self.names: List[str] = []
        self.threshold = threshold

    def register(self, name: str, embedding: np.ndarray):
        """Register a face with a name."""
        self.names.append(name)
        self.embeddings.append(embedding)

    def identify(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find closest matching person.

        Returns:
            (name, similarity) or (None, best_similarity)
        """
        if not self.embeddings:
            return None, 0.0

        # Compute similarities via dot product (embeddings are normalized)
        db_matrix = np.array(self.embeddings)
        similarities = np.dot(db_matrix, query_embedding)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.threshold:
            return self.names[best_idx], best_sim
        return None, best_sim

    def save(self, path: str):
        """Save database to disk."""
        np.savez(path,
                 embeddings=np.array(self.embeddings) if self.embeddings else np.array([]),
                 names=np.array(self.names))

    def load(self, path: str):
        """Load database from disk."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = list(data['embeddings'])
        self.names = list(data['names'])

    @property
    def count(self) -> int:
        return len(self.names)


class SimpleFaceDetector:
    """
    Production Face Detector using InsightFace.

    Detects faces and extracts embeddings for recognition.
    Uses SQLite database for persistent storage.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_thresh: float = 0.5,
        use_gpu: bool = True,
        db_path: str = "faces.db",
        match_threshold: float = 0.5
    ):
        """
        Initialize detector.

        Args:
            model_name: InsightFace model (buffalo_l = accurate, buffalo_s = fast)
            det_thresh: Detection confidence threshold
            use_gpu: Whether to use GPU
            db_path: Path to SQLite face database
            match_threshold: Similarity threshold for face matching
        """
        from insightface.app import FaceAnalysis

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers
        )

        ctx_id = 0 if use_gpu else -1
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(640, 640))

        # Use persistent SQLite database
        self.database = FaceDB(db_path=db_path, threshold=match_threshold)

    def detect(self, frame: np.ndarray) -> FaceResult:
        """
        Detect faces in frame.

        Args:
            frame: BGR image from camera

        Returns:
            FaceResult with detected faces
        """
        import time
        start = time.time()

        faces = self.app.get(frame)
        inference_ms = (time.time() - start) * 1000

        result_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)

            # Get embedding
            embedding = face.normed_embedding

            # Try to identify (FaceDB returns name, similarity, id)
            name, similarity, _ = self.database.identify(embedding)

            result_faces.append(FaceBox(
                x1=int(bbox[0]),
                y1=int(bbox[1]),
                x2=int(bbox[2]),
                y2=int(bbox[3]),
                confidence=float(face.det_score),
                embedding=embedding,
                age=int(face.age) if hasattr(face, 'age') and face.age is not None else None,
                gender="male" if getattr(face, 'gender', None) == 1 else "female" if getattr(face, 'gender', None) == 0 else None,
                name=name,
                similarity=similarity
            ))

        return FaceResult(
            detected=len(result_faces) > 0,
            count=len(result_faces),
            faces=result_faces,
            inference_ms=inference_ms
        )

    def register_face(self, frame: np.ndarray, name: str) -> Optional[int]:
        """
        Register a face from frame.

        Args:
            frame: Image containing the face
            name: Name to associate with face

        Returns:
            Face ID if registered, None otherwise
        """
        result = self.detect(frame)
        if result.detected and len(result.faces) == 1:
            face_id = self.database.register(name, result.faces[0].embedding)
            return face_id
        return None

    def detect_and_draw(self, frame: np.ndarray) -> Tuple[np.ndarray, FaceResult]:
        """
        Detect and draw boxes on frame.

        Args:
            frame: BGR image from camera

        Returns:
            Tuple of (annotated_frame, FaceResult)
        """
        result = self.detect(frame)

        annotated = frame.copy()
        for face in result.faces:
            # Color: green if known, yellow if unknown
            color = (0, 255, 0) if face.name else (0, 255, 255)

            cv2.rectangle(
                annotated,
                (face.x1, face.y1),
                (face.x2, face.y2),
                color, 2
            )

            # Label
            if face.name:
                label = f"{face.name} ({face.similarity:.2f})"
            else:
                label = f"Unknown ({face.confidence:.2f})"

            if face.age and face.gender:
                label += f" {face.gender[0].upper()}{face.age}"

            cv2.putText(
                annotated, label,
                (face.x1, face.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2
            )

        return annotated, result


# =============================================================================
# TEST - Run this file directly to test
# =============================================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  FACE DETECTION TEST (SQLite DB)")
    print("=" * 60)
    print("\nLoading InsightFace model...")

    detector = SimpleFaceDetector(db_path="faces.db")
    print(f"Database: {detector.database.count} faces loaded")
    if detector.database.names:
        print(f"Known: {detector.database.names}")
    print("\nControls:")
    print("  'r' - Register current face")
    print("  'l' - List all registered faces")
    print("  'q' - Quit")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        exit(1)

    fps_time = time.time()
    fps_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break

        # Mirror flip for natural view
        frame = cv2.flip(frame, 1)

        # Detect and draw
        annotated, result = detector.detect_and_draw(frame)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Show status
        status = f"Faces: {result.count} | {result.inference_ms:.1f}ms | {fps} FPS | DB: {detector.database.count}"
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Register face
            name = input("Enter name for face: ").strip()
            face_id = detector.register_face(frame, name) if name else None
            if face_id:
                print(f"Registered: {name} (ID: {face_id})")
            else:
                print("Could not register (need exactly 1 face)")
        elif key == ord('l'):
            # List faces
            people = detector.database.list_all()
            print(f"\nRegistered faces ({len(people)}):")
            for p in people:
                print(f"  - {p['name']}: {p['embeddings']} embeddings, seen {p['total_seen']} times")
            print()

    detector.database.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")
