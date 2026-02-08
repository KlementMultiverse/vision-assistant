#!/usr/bin/env python3
"""
Motion Detection - POC Version
==============================
Simple frame differencing that WORKS.
No fancy stuff. Just detect motion.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MotionResult:
    """What motion detection returns."""
    detected: bool
    boxes: List[Tuple[int, int, int, int]]  # [(x, y, w, h), ...]
    magnitude: float  # 0.0 - 1.0


class SimpleMotionDetector:
    """
    POC Motion Detector.

    Uses simple frame differencing.
    Good enough to prove the concept works.
    """

    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None

    def detect(self, frame: np.ndarray) -> MotionResult:
        """
        Detect motion in frame.

        Args:
            frame: BGR image from camera

        Returns:
            MotionResult with detection info
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # First frame - initialize
        if self.prev_frame is None:
            self.prev_frame = gray
            return MotionResult(detected=False, boxes=[], magnitude=0.0)

        # Compute difference
        delta = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area
        boxes = []
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
                total_area += area

        # Calculate magnitude
        frame_area = frame.shape[0] * frame.shape[1]
        magnitude = min(total_area / frame_area, 1.0)

        # Update previous frame
        self.prev_frame = gray

        return MotionResult(
            detected=len(boxes) > 0,
            boxes=boxes,
            magnitude=magnitude
        )


# =============================================================================
# TEST - Run this file directly to test
# =============================================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  MOTION DETECTION POC TEST")
    print("=" * 60)
    print("\nPress 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        exit(1)

    detector = SimpleMotionDetector()

    fps_time = time.time()
    fps_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break

        # Detect
        start = time.time()
        result = detector.detect(frame)
        detect_ms = (time.time() - start) * 1000

        # Draw boxes
        for (x, y, w, h) in result.boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Show status
        status = f"Motion: {result.detected} | Boxes: {len(result.boxes)} | Mag: {result.magnitude:.3f} | {detect_ms:.1f}ms | {fps} FPS"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Motion Detection POC", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")
