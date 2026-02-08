#!/usr/bin/env python3
"""
Vision Assistant - Real-time Streaming UI
Continuous video with background VLM analysis
"""

import cv2
import time
import base64
import threading
import queue
import signal
import sys
from datetime import datetime
from dataclasses import dataclass
from vision import VisionModel
from voice import Voice


@dataclass
class AnalysisResult:
    person: bool = False
    action: str = "unknown"
    looking: bool = False
    description: str = ""
    timestamp: float = 0


class StreamingAgent:
    def __init__(self):
        self.vision = VisionModel("http://localhost:8080")
        self.voice = Voice()

        # State
        self.running = False
        self.latest_result = AnalysisResult()
        self.logs = []
        self.greeted = False
        self.last_person_time = 0

        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.analysis_thread = None

    def log(self, msg: str, icon: str = "ℹ️"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {icon} {msg}"
        self.logs.append(entry)
        if len(self.logs) > 10:
            self.logs = self.logs[-10:]
        print(entry)

    def analyze_loop(self):
        """Background thread for VLM analysis."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode()

            # Get description
            start = time.time()
            description = self.vision.describe(image_b64)
            elapsed = time.time() - start

            # Parse for person detection
            desc_lower = description.lower()
            person_words = ["person", "man", "woman", "individual", "someone", "human", "face", "people", "you"]
            has_person = any(w in desc_lower for w in person_words)

            result = AnalysisResult(
                person=has_person,
                description=description[:150],
                timestamp=time.time()
            )
            self.latest_result = result

            # React
            now = time.time()
            if has_person:
                self.last_person_time = now
                if not self.greeted:
                    self.log(f"Person detected!", "👤")
                    self.log(f"VLM says: {description[:80]}...", "🤖")
                    self.voice.speak_async("Hello! I can see you!")
                    self.greeted = True
            else:
                if self.greeted and (now - self.last_person_time > 5):
                    self.log("No one here, resetting...", "👻")
                    self.greeted = False

    def draw_overlay(self, frame):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        result = self.latest_result

        # Top bar - status
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 30), -1)

        # Person indicator
        if result.person:
            cv2.circle(frame, (40, 35), 20, (0, 255, 0), -1)
            cv2.putText(frame, "PERSON DETECTED", (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (40, 35), 20, (100, 100, 100), -1)
            cv2.putText(frame, "SCANNING...", (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        # FPS/Status
        cv2.putText(frame, "LIVE", (w - 80, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Bottom bar - VLM description
        cv2.rectangle(frame, (0, h - 100), (w, h), (20, 20, 30), -1)

        if result.description:
            # Wrap text
            text = result.description[:100]
            cv2.putText(frame, "VLM:", (10, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(frame, text[:60], (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if len(text) > 60:
                cv2.putText(frame, text[60:], (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Log panel on right side
        log_x = w - 350
        cv2.rectangle(frame, (log_x - 10, 80), (w, h - 110), (30, 30, 40), -1)
        cv2.putText(frame, "Agent Log", (log_x, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        for i, log_entry in enumerate(self.logs[-6:]):
            y = 130 + i * 25
            # Truncate long entries
            display_text = log_entry[:40] if len(log_entry) > 40 else log_entry
            cv2.putText(frame, display_text, (log_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def run(self):
        """Main loop - continuous video with background analysis."""
        print("\n" + "=" * 60)
        print("🤖 VISION ASSISTANT - Real-time Streaming")
        print("=" * 60)
        print("Press 'q' in window OR Ctrl+C in terminal to quit")
        print("Press 's' to speak status")
        print("=" * 60 + "\n")

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n🛑 Ctrl+C detected, shutting down...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.log("Vision Assistant started!", "✅")
        self.voice.speak_async("Vision assistant is now streaming.")

        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analyze_loop, daemon=True)
        self.analysis_thread.start()

        # Create window
        cv2.namedWindow("Vision Assistant", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vision Assistant", 1280, 720)

        frame_count = 0
        last_analysis_time = 0
        analysis_interval = 1.5  # Analyze every 1.5 seconds

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1
                now = time.time()

                # Send frame for analysis periodically
                if now - last_analysis_time > analysis_interval:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                        last_analysis_time = now

                # Draw overlay
                display_frame = self.draw_overlay(frame.copy())

                # Show frame
                cv2.imshow("Vision Assistant", display_frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    status = "I see someone" if self.latest_result.person else "No one here"
                    self.voice.speak_async(status)

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.voice.speak("Goodbye!")
            print("\n👋 Vision Assistant stopped.")


def main():
    import requests

    # Check VLM server
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        if r.status_code != 200:
            print("⚠️  VLM server not healthy!")
            return
    except:
        print("❌ VLM server not running!")
        print("   Start it with: ./start_server.sh")
        return

    print("✅ VLM Server: Running")

    agent = StreamingAgent()
    agent.run()


if __name__ == "__main__":
    main()
