#!/usr/bin/env python3
"""
Face Calibration Tool
======================
Captures multiple embeddings of YOU to:
1. See how similar your own embeddings are
2. Find optimal confidence threshold
3. Test recognition accuracy

Run: python calibrate_face.py
"""

import cv2
import numpy as np
import time
import sys

sys.path.insert(0, '/home/intruder/vision-assistant')

from src.v2.perception.face.base import SimpleFaceDetector


class CalibrationTool:
    def __init__(self):
        self.detector = None
        self.cap = None
        self.embeddings = []
        self.captures = []
        self.phase = "capture"  # capture, analyze
        self.target_count = 10

    def run(self):
        print("=" * 60)
        print("  FACE CALIBRATION TOOL")
        print("=" * 60)
        print("\nThis tool captures multiple shots of YOUR face to:")
        print("  1. Measure embedding consistency")
        print("  2. Find optimal confidence threshold")
        print("  3. Test with different poses/lighting")
        print("\nLoading models...")

        self.detector = SimpleFaceDetector(db_path=":memory:")  # In-memory for testing

        print("\nOpening camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open camera")
            return

        print(f"\nWe'll capture {self.target_count} shots of your face.")
        print("Move slightly between captures for variety.")
        print("\nControls:")
        print("  SPACE - Capture current frame")
        print("  'a' - Analyze captured embeddings")
        print("  'r' - Reset and start over")
        print("  'q' - Quit")
        print()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Detect face
            result = self.detector.detect(frame)

            if self.phase == "capture":
                self._draw_capture_ui(display, result)
            else:
                self._draw_analyze_ui(display)

            cv2.imshow("Face Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and self.phase == "capture":
                self._capture(frame, result)
            elif key == ord('a') and len(self.embeddings) >= 2:
                self.phase = "analyze"
                self._analyze()
            elif key == ord('r'):
                self.embeddings = []
                self.captures = []
                self.phase = "capture"
                print("\nReset - start capturing again")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\nDone!")

    def _capture(self, frame, result):
        if not result.detected:
            print("No face detected - try again")
            return

        if result.count > 1:
            print("Multiple faces - only you should be in frame")
            return

        face = result.faces[0]
        self.embeddings.append(face.embedding)
        self.captures.append({
            'confidence': face.confidence,
            'timestamp': time.time(),
            'frame': frame.copy()
        })

        count = len(self.embeddings)
        print(f"Captured {count}/{self.target_count} (confidence: {face.confidence:.3f})")

        if count >= self.target_count:
            print(f"\n✅ Got {self.target_count} captures! Press 'a' to analyze.")

    def _analyze(self):
        """Analyze similarity between captured embeddings."""
        n = len(self.embeddings)
        print(f"\n{'='*60}")
        print(f"  ANALYSIS: {n} embeddings captured")
        print(f"{'='*60}")

        # Convert to matrix
        matrix = np.array(self.embeddings)

        # Compute pairwise similarities
        similarities = np.dot(matrix, matrix.T)

        # Get stats (excluding diagonal which is always 1.0)
        mask = ~np.eye(n, dtype=bool)
        pairwise = similarities[mask]

        # ALL data stats
        min_sim_all = float(np.min(pairwise))
        max_sim_all = float(np.max(pairwise))
        mean_sim_all = float(np.mean(pairwise))

        print(f"\nALL pairwise similarities:")
        print(f"  Min:  {min_sim_all:.4f}")
        print(f"  Max:  {max_sim_all:.4f}")
        print(f"  Mean: {mean_sim_all:.4f}")

        # QUALITY data only (>0.4 = good captures, not extreme angles)
        quality_threshold = 0.4
        quality_pairs = pairwise[pairwise >= quality_threshold]
        low_quality_count = len(pairwise) - len(quality_pairs)

        print(f"\nQUALITY pairs only (similarity >= {quality_threshold}):")
        if len(quality_pairs) > 0:
            min_sim = float(np.min(quality_pairs))
            max_sim = float(np.max(quality_pairs))
            mean_sim = float(np.mean(quality_pairs))
            print(f"  Count: {len(quality_pairs)}/{len(pairwise)} ({low_quality_count} outliers removed)")
            print(f"  Min:  {min_sim:.4f}")
            print(f"  Max:  {max_sim:.4f}")
            print(f"  Mean: {mean_sim:.4f}")

            # Suggested thresholds based on quality data
            print(f"\nSuggested thresholds (based on quality data):")
            print(f"  Family (must not miss):   {min_sim - 0.10:.3f}")
            print(f"  Friends:                  {min_sim - 0.05:.3f}")
            print(f"  Public/Strict:            {min_sim:.3f}")
        else:
            print(f"  No quality pairs found! All similarities < {quality_threshold}")
            print(f"  Try more consistent poses (frontal, good lighting)")
            min_sim = min_sim_all

        # Show histogram
        print(f"\nDistribution (all pairs):")
        hist, bins = np.histogram(pairwise, bins=10)
        max_hist = max(hist) if max(hist) > 0 else 1
        for i, count in enumerate(hist):
            bar = '█' * int(count / max_hist * 30)
            marker = " ← outliers" if bins[i+1] < quality_threshold else ""
            print(f"  {bins[i]:.3f}-{bins[i+1]:.3f}: {bar} ({count}){marker}")

        # Confidence stats
        confs = [c['confidence'] for c in self.captures]
        print(f"\nDetection confidence stats:")
        print(f"  Min:  {min(confs):.3f}")
        print(f"  Max:  {max(confs):.3f}")
        print(f"  Mean: {np.mean(confs):.3f}")

        # Recommendations
        print(f"\n{'='*60}")
        if low_quality_count > len(pairwise) * 0.3:
            print(f"⚠️  {low_quality_count} outlier pairs detected (extreme poses)")
            print("   For registration: use more consistent frontal poses")
            print("   For matching: outliers are filtered out automatically")
        print(f"\nThreshold recommendation:")
        if len(quality_pairs) > 0:
            print(f"  Use {min_sim - 0.05:.3f} for reliable recognition")
        else:
            print(f"  Run again with more consistent poses")
        print(f"{'='*60}\n")

    def _draw_capture_ui(self, frame, result):
        h, w = frame.shape[:2]
        count = len(self.embeddings)

        # Draw face box
        if result.detected:
            face = result.faces[0]
            cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {face.confidence:.2f}",
                       (face.x1, face.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status
        status = f"Captured: {count}/{self.target_count}"
        color = (0, 255, 0) if count >= self.target_count else (0, 255, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Instructions
        if result.detected:
            cv2.putText(frame, "SPACE to capture", (10, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No face - look at camera", (10, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if count >= 2:
            cv2.putText(frame, "'a' to analyze", (10, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _draw_analyze_ui(self, frame):
        h, w = frame.shape[:2]

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "ANALYSIS COMPLETE", (w//2-150, h//2-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Check terminal for results", (w//2-150, h//2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, "'r' to reset, 'q' to quit", (w//2-120, h//2+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


if __name__ == "__main__":
    tool = CalibrationTool()
    tool.run()
