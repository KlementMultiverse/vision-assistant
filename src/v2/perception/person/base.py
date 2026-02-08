#!/usr/bin/env python3
"""
Person Detection - POC Version
==============================
YOLOv8n for fast person detection.
No fancy tracking. Just detect persons.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PersonBox:
    """Single person detection."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, w, h) tuple."""
        return (self.x1, self.y1, self.width, self.height)


@dataclass
class PersonResult:
    """What person detection returns."""
    detected: bool
    count: int
    boxes: List[PersonBox]
    inference_ms: float = 0.0


class SimplePersonDetector:
    """
    POC Person Detector using YOLOv8n.

    Fast and accurate enough for real-time use.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize detector.

        Args:
            model_name: YOLO model to use (yolov8n.pt is fastest)
            confidence: Minimum confidence threshold (0-1)
            device: 'auto', 'cpu', 'cuda', or device index
        """
        from ultralytics import YOLO

        self.confidence = confidence
        self.model = YOLO(model_name)
        self.model.fuse()  # Optimize model

        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)

    def detect(self, frame: np.ndarray) -> PersonResult:
        """
        Detect persons in frame.

        Args:
            frame: BGR image from camera

        Returns:
            PersonResult with detection info
        """
        import time
        start = time.time()

        # Run inference - filter for person only (class 0)
        results = self.model.predict(
            source=frame,
            classes=[0],  # Person only
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False
        )

        inference_ms = (time.time() - start) * 1000

        # Extract boxes
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                boxes.append(PersonBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=conf
                ))

        return PersonResult(
            detected=len(boxes) > 0,
            count=len(boxes),
            boxes=boxes,
            inference_ms=inference_ms
        )

    def detect_and_draw(self, frame: np.ndarray) -> Tuple[np.ndarray, PersonResult]:
        """
        Detect and draw boxes on frame.

        Args:
            frame: BGR image from camera

        Returns:
            Tuple of (annotated_frame, PersonResult)
        """
        result = self.detect(frame)

        # Draw boxes
        annotated = frame.copy()
        for box in result.boxes:
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (0, 255, 0), 2
            )
            label = f"Person {box.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (box.x1, box.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

        return annotated, result


# =============================================================================
# TEST - Run this file directly to test
# =============================================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  PERSON DETECTION POC TEST")
    print("=" * 60)
    print("\nLoading model...")

    detector = SimplePersonDetector()
    print(f"Using device: {detector.device}")
    print("\nPress 'q' to quit\n")

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

        # Detect and draw
        annotated, result = detector.detect_and_draw(frame)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Show status
        status = f"Persons: {result.count} | {result.inference_ms:.1f}ms | {fps} FPS"
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Person Detection POC", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")
