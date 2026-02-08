#!/usr/bin/env python3
"""
Motion Detector - Cheap Local Processing (Layer 1)
Runs on every frame (<1ms) to detect when VLM should be called.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MotionResult:
    """Result of motion detection."""
    detected: bool
    magnitude: float  # 0.0 to 1.0
    regions: List[Tuple[int, int, int, int]]  # Bounding boxes of motion
    frame_diff: Optional[np.ndarray] = None  # For visualization


class MotionDetector:
    """
    Efficient motion detection using frame differencing.

    This is Layer 1 of the hierarchical processing:
    - Runs on EVERY frame
    - Very fast (<1ms per frame)
    - Triggers VLM only when motion detected
    """

    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500,
        blur_size: int = 21,
        sensitivity: float = 0.01
    ):
        """
        Args:
            threshold: Pixel difference threshold (0-255)
            min_area: Minimum contour area to count as motion
            blur_size: Gaussian blur kernel size (must be odd)
            sensitivity: Fraction of frame that must change (0.0-1.0)
        """
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.sensitivity = sensitivity

        # State
        self.previous_frame: Optional[np.ndarray] = None
        self.motion_history: List[bool] = []
        self.history_size = 10

    def detect(self, frame: np.ndarray) -> MotionResult:
        """
        Detect motion in frame compared to previous frame.

        Args:
            frame: BGR image from camera

        Returns:
            MotionResult with detection info
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # First frame - no comparison possible
        if self.previous_frame is None:
            self.previous_frame = gray
            return MotionResult(
                detected=False,
                magnitude=0.0,
                regions=[],
                frame_diff=None
            )

        # Compute absolute difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)

        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        motion_regions = []
        total_motion_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
                total_motion_area += area

        # Calculate motion magnitude
        frame_area = frame.shape[0] * frame.shape[1]
        magnitude = min(total_motion_area / frame_area, 1.0)

        # Determine if significant motion
        detected = magnitude > self.sensitivity

        # Update history
        self.motion_history.append(detected)
        if len(self.motion_history) > self.history_size:
            self.motion_history = self.motion_history[-self.history_size:]

        # Update previous frame
        self.previous_frame = gray

        return MotionResult(
            detected=detected,
            magnitude=magnitude,
            regions=motion_regions,
            frame_diff=thresh
        )

    def is_scene_stable(self, frames: int = 5) -> bool:
        """
        Check if scene has been stable (no motion) for N frames.

        Useful for determining when to stop VLM processing.
        """
        if len(self.motion_history) < frames:
            return False
        return not any(self.motion_history[-frames:])

    def reset(self):
        """Reset detector state."""
        self.previous_frame = None
        self.motion_history = []


class SceneChangeDetector:
    """
    Detects if the overall scene has changed significantly.

    Different from motion detection:
    - Motion: someone walking through
    - Scene change: camera moved, lights turned on/off, new object placed

    This is Layer 2 - runs only when motion detected.
    """

    def __init__(self, change_threshold: float = 0.3):
        """
        Args:
            change_threshold: Histogram difference threshold (0.0-1.0)
        """
        self.change_threshold = change_threshold
        self.reference_histogram: Optional[np.ndarray] = None
        self.reference_frame: Optional[np.ndarray] = None

    def update_reference(self, frame: np.ndarray):
        """Set current frame as reference for comparison."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(self.reference_histogram, self.reference_histogram)
        self.reference_frame = gray.copy()

    def has_scene_changed(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if scene has changed significantly from reference.

        Returns:
            Tuple of (changed: bool, difference: float)
        """
        if self.reference_histogram is None:
            self.update_reference(frame)
            return False, 0.0

        # Calculate histogram of current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(current_hist, current_hist)

        # Compare histograms
        correlation = cv2.compareHist(
            self.reference_histogram,
            current_hist,
            cv2.HISTCMP_CORREL
        )

        # Correlation is 1.0 for identical, lower for different
        difference = 1.0 - correlation
        changed = difference > self.change_threshold

        return changed, difference


# Test
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    detector = MotionDetector()
    scene_detector = SceneChangeDetector()

    print("Motion Detection Test")
    print("Press 'q' to quit, 's' to set scene reference")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect motion
        result = detector.detect(frame)

        # Check scene change
        scene_changed, scene_diff = scene_detector.has_scene_changed(frame)

        # Draw motion regions
        for (x, y, w, h) in result.regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Status text
        status = f"Motion: {result.detected} ({result.magnitude:.3f})"
        scene_status = f"Scene change: {scene_changed} ({scene_diff:.3f})"

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, scene_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if result.detected:
            cv2.putText(frame, "MOTION!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Motion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            scene_detector.update_reference(frame)
            print("Scene reference updated")

    cap.release()
    cv2.destroyAllWindows()
