#!/usr/bin/env python3
"""
Vision Assistant - Smart Streaming with Brain
SmolVLM (local) + OpenAI Brain (API) + Memory
"""

import cv2
import time
import base64
import threading
import queue
import signal
import os
from datetime import datetime
from dataclasses import dataclass
from vision import VisionModel
from voice import Voice
from brain import Brain, Event


@dataclass
class AnalysisResult:
    person: bool = False
    description: str = ""
    timestamp: float = 0
    event: Event = None


class SmartAgent:
    def __init__(self):
        # Local perception
        self.vision = VisionModel("http://localhost:8080")
        self.voice = Voice()

        # Smart brain (API)
        self.brain = Brain()

        # State
        self.running = False
        self.latest_result = AnalysisResult()
        self.logs = []
        self.last_brain_call = 0
        self.brain_cooldown = 5  # Don't call brain more than once per 5 seconds

        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.analysis_thread = None

    def log(self, msg: str, icon: str = "ℹ️"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {icon} {msg}"
        self.logs.append(entry)
        if len(self.logs) > 15:
            self.logs = self.logs[-15:]
        print(entry)

    def analyze_loop(self):
        """Background thread: VLM perception + Brain reasoning."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode()

            # Step 1: Local VLM perception (always)
            start = time.time()
            description = self.vision.describe(image_b64)
            vlm_time = time.time() - start

            self.log(f"VLM ({vlm_time:.1f}s): {description[:60]}...", "👁️")

            # Detect if person/interesting
            desc_lower = description.lower()
            has_person = any(w in desc_lower for w in
                ["person", "man", "woman", "someone", "human", "face", "people"])

            result = AnalysisResult(
                person=has_person,
                description=description,
                timestamp=time.time()
            )

            # Step 2: Smart Brain analysis (only when interesting + cooldown)
            now = time.time()
            should_think = (
                self.brain.should_analyze(description) and
                (now - self.last_brain_call) > self.brain_cooldown
            )

            if should_think:
                self.log("Brain analyzing...", "🧠")
                self.last_brain_call = now

                event = self.brain.analyze(description)
                result.event = event

                self.log(f"Event: {event.event_type} | {event.urgency} | {event.action}", "🎯")

                # Speak if needed
                voice_msg = self.brain.get_voice_response(event)
                if voice_msg:
                    self.log(f"Speaking: {voice_msg}", "🔊")
                    self.voice.speak_async(voice_msg)

                # TODO: Store in Graphiti memory

            self.latest_result = result

    def draw_overlay(self, frame):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        result = self.latest_result

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 30), -1)

        # Status indicators
        if result.person:
            cv2.circle(frame, (40, 40), 25, (0, 255, 0), -1)
            status_text = "PERSON DETECTED"
            status_color = (0, 255, 0)
        else:
            cv2.circle(frame, (40, 40), 25, (100, 100, 100), -1)
            status_text = "MONITORING..."
            status_color = (150, 150, 150)

        cv2.putText(frame, status_text, (80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # Event badge (if brain analyzed)
        if result.event:
            event = result.event
            badge_colors = {
                "high": (0, 0, 255),
                "medium": (0, 165, 255),
                "low": (0, 255, 255),
                "none": (128, 128, 128)
            }
            color = badge_colors.get(event.urgency, (128, 128, 128))

            cv2.rectangle(frame, (w - 200, 10), (w - 10, 70), color, -1)
            cv2.putText(frame, event.event_type.upper(), (w - 190, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, event.action.upper(), (w - 190, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # LIVE indicator
        cv2.putText(frame, "LIVE", (w - 70, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Bottom panel - VLM + Brain output
        cv2.rectangle(frame, (0, h - 120), (w, h), (20, 20, 30), -1)

        # VLM observation
        cv2.putText(frame, "VLM:", (10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        if result.description:
            desc_line = result.description[:80]
            cv2.putText(frame, desc_line, (60, h - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Brain analysis
        if result.event:
            cv2.putText(frame, "BRAIN:", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            brain_line = result.event.description[:75]
            cv2.putText(frame, brain_line, (80, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Log panel
        log_x = w - 380
        cv2.rectangle(frame, (log_x - 10, 90), (w - 5, h - 130), (30, 30, 40), -1)
        cv2.putText(frame, "Agent Log", (log_x, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for i, log_entry in enumerate(self.logs[-8:]):
            y = 130 + i * 22
            display = log_entry[:45] if len(log_entry) > 45 else log_entry
            cv2.putText(frame, display, (log_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        return frame

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("🤖 SMART VISION ASSISTANT")
        print("   SmolVLM (local) + OpenAI Brain (API)")
        print("=" * 60)
        print("Press 'q' to quit | Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Handle Ctrl+C
        def signal_handler(sig, frame):
            print("\n🛑 Shutting down...")
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
        self.log("Smart Vision Assistant started!", "✅")
        self.voice.speak_async("Smart vision assistant is now active.")

        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analyze_loop, daemon=True)
        self.analysis_thread.start()

        # Create window
        cv2.namedWindow("Smart Vision Assistant", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Vision Assistant", 1280, 720)

        last_frame_time = 0
        analysis_interval = 2.0  # Analyze every 2 seconds

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                now = time.time()

                # Queue frame for analysis
                if now - last_frame_time > analysis_interval:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                        last_frame_time = now

                # Draw overlay
                display = self.draw_overlay(frame.copy())

                # Show
                cv2.imshow("Smart Vision Assistant", display)

                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.voice.speak_async("Status check. System operational.")

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.voice.speak("Goodbye!")
            print("\n👋 Smart Vision Assistant stopped.")


def main():
    import requests

    # Check VLM server
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        if r.status_code != 200:
            print("⚠️  VLM server not healthy!")
            return
        print("✅ VLM Server: Running")
    except:
        print("❌ VLM server not running! Start with: ./start_server.sh")
        return

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set!")
        print("   Export it: export OPENAI_API_KEY='sk-...'")
        return
    print("✅ OpenAI API: Configured")

    agent = SmartAgent()
    agent.run()


if __name__ == "__main__":
    main()
