#!/usr/bin/env python3
"""
JARVIS - Smart Home Vision Assistant
Natural speech, real personality, intelligent responses.
"""

import cv2
import time
import base64
import threading
import queue
import signal
import os
from datetime import datetime
from vision import VisionModel
from voice import Voice
from brain_v2 import SmartBrain


class Jarvis:
    """JARVIS - Your friendly home vision assistant."""

    def __init__(self):
        # Perception
        self.vision = VisionModel("http://localhost:8080")
        self.voice = Voice()

        # Brain
        self.brain = SmartBrain()

        # State
        self.running = False
        self.logs = []
        self.current_observation = ""
        self.last_speech = ""
        self.person_detected = False

        # Threading
        self.frame_queue = queue.Queue(maxsize=2)

    def log(self, msg: str, icon: str = "•"):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {icon} {msg}"
        self.logs.append(entry)
        if len(self.logs) > 12:
            self.logs = self.logs[-12:]
        print(entry)

    def perception_loop(self):
        """Background: VLM + Brain reasoning."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Encode
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode()

            # VLM perception
            start = time.time()
            observation = self.vision.describe(image_b64)
            elapsed = time.time() - start

            self.current_observation = observation
            short_obs = observation[:50] + "..." if len(observation) > 50 else observation
            self.log(f"See: {short_obs}", "👁️")

            # Detect person for UI
            obs_lower = observation.lower()
            self.person_detected = any(w in obs_lower for w in
                ["person", "man", "woman", "someone", "human", "people"])

            # Brain thinking (only if interesting)
            if self.brain.should_think(observation):
                self.log("Thinking...", "🧠")
                speech = self.brain.think(observation)

                if speech:
                    self.last_speech = speech
                    self.log(f"Say: {speech}", "🗣️")
                    self.voice.speak_async(speech)
                else:
                    self.log("Nothing new to say", "🤫")

    def draw_ui(self, frame):
        """Draw clean UI overlay."""
        h, w = frame.shape[:2]

        # Semi-transparent overlays
        overlay = frame.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 30), -1)

        # Bottom bar
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 30), -1)

        # Right panel for logs
        cv2.rectangle(overlay, (w - 350, 70), (w, h - 110), (25, 25, 35), -1)

        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status indicator
        if self.person_detected:
            cv2.circle(frame, (35, 30), 18, (0, 220, 0), -1)
            cv2.putText(frame, "PERSON", (65, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        else:
            cv2.circle(frame, (35, 30), 18, (80, 80, 80), -1)
            cv2.putText(frame, "WATCHING", (65, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

        # Title
        cv2.putText(frame, "JARVIS", (w - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Bottom: Current observation
        if self.current_observation:
            obs_text = self.current_observation[:90]
            cv2.putText(frame, "Seeing:", (15, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
            cv2.putText(frame, obs_text, (15, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Last speech
        if self.last_speech:
            cv2.putText(frame, "Said:", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            speech_text = self.last_speech[:80]
            cv2.putText(frame, f'"{speech_text}"', (60, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Log panel
        cv2.putText(frame, "Activity Log", (w - 340, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        for i, entry in enumerate(self.logs[-10:]):
            y = 120 + i * 20
            text = entry[:42] if len(entry) > 42 else entry
            cv2.putText(frame, text, (w - 340, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)

        return frame

    def run(self):
        """Main loop."""
        print("\n" + "=" * 50)
        print("   🤖 JARVIS - Smart Home Vision Assistant")
        print("=" * 50)
        print("   Press 'q' to quit")
        print("=" * 50 + "\n")

        # Ctrl+C handler
        def handler(sig, frame):
            print("\n👋 Shutting down...")
            self.running = False
        signal.signal(signal.SIGINT, handler)

        # Camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.log("JARVIS online", "✅")
        self.voice.speak("JARVIS is now watching. I'll let you know if anything interesting happens.")

        # Start perception thread
        threading.Thread(target=self.perception_loop, daemon=True).start()

        # Window
        cv2.namedWindow("JARVIS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JARVIS", 1280, 720)

        last_analysis = 0
        analysis_interval = 2.5  # Every 2.5 seconds

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                now = time.time()

                # Queue frame for analysis
                if now - last_analysis > analysis_interval:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                        last_analysis = now

                # Draw UI
                display = self.draw_ui(frame.copy())
                cv2.imshow("JARVIS", display)

                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.voice.speak("JARVIS signing off. Goodbye!")


def main():
    import requests

    # Check VLM
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        print("✅ VLM Server: OK")
    except:
        print("❌ VLM server not running!")
        return

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return
    print("✅ OpenAI API: OK")

    jarvis = Jarvis()
    jarvis.run()


if __name__ == "__main__":
    main()
