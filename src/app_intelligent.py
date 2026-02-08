#!/usr/bin/env python3
"""
Intelligent Vision Agent
=======================

Hierarchical processing with continuous context awareness:

Layer 1: Motion Detection (every frame, <1ms)
    └── Only triggers VLM when something moves

Layer 2: Scene Change Detection (on motion, ~50ms)
    └── Determines if scene is meaningfully different

Layer 3: VLM Perception (on change, ~700ms)
    └── Full scene understanding

Layer 4: World Model (always running)
    └── Maintains context between observations

Layer 5: Brain (when world model changes)
    └── Decides what to say/do
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
from typing import Optional

from vision import VisionModel
from voice import Voice
from brain_v2 import SmartBrain
from motion_detector import MotionDetector, SceneChangeDetector
from world_model import WorldModel


@dataclass
class AgentState:
    """Current state of the agent."""
    mode: str = "watching"  # watching, detecting, analyzing, engaging
    last_vlm_time: float = 0
    last_speech_time: float = 0
    motion_count: int = 0
    vlm_count: int = 0
    brain_count: int = 0


class IntelligentAgent:
    """
    Vision agent with continuous context awareness.

    Key principle: Process efficiently, maintain context always.
    """

    def __init__(self):
        # Perception layers
        self.motion_detector = MotionDetector(
            threshold=15,       # Lower = more sensitive to pixel changes
            min_area=300,       # Smaller areas count as motion
            sensitivity=0.001   # 0.1% of frame = motion detected
        )
        self.scene_detector = SceneChangeDetector(change_threshold=0.05)  # More sensitive
        self.vision = VisionModel("http://localhost:8080")
        self.voice = Voice()
        self.brain = SmartBrain()

        # Context layer
        self.world = WorldModel(observation_window=10)

        # State
        self.state = AgentState()
        self.running = False
        self.logs = []

        # Threading
        self.frame_queue = queue.Queue(maxsize=2)

        # Timing controls
        self.min_vlm_interval = 1.5  # Don't call VLM more than once per 1.5 seconds
        self.min_brain_interval = 4.0  # Don't call brain more than once per 4 seconds

        # Force first analysis
        self.first_frame = True

    def log(self, msg: str, icon: str = "•"):
        """Add to activity log."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {icon} {msg}"
        self.logs.append(entry)
        if len(self.logs) > 15:
            self.logs = self.logs[-15:]
        print(entry)

    def process_frame(self, frame):
        """
        Hierarchical frame processing.

        Returns True if frame was fully processed (VLM called).
        """
        now = time.time()

        # Layer 1: Motion Detection (always, fast)
        motion_result = self.motion_detector.detect(frame)

        # Force first frame analysis to detect if person already present
        force_analyze = self.first_frame
        if self.first_frame:
            self.first_frame = False
            self.log("First frame - forcing analysis", "🔍")

        if motion_result.detected or force_analyze:
            self.state.motion_count += 1
            self.state.mode = "detecting"

            # Rate limit VLM calls
            if now - self.state.last_vlm_time < self.min_vlm_interval:
                return False

            # Layer 2: Scene Change Detection
            scene_changed, scene_diff = self.scene_detector.has_scene_changed(frame)

            if scene_changed or scene_diff > 0.1:
                self.state.mode = "analyzing"

                # Layer 3: VLM Perception
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_b64 = base64.b64encode(buffer).decode()

                start = time.time()
                observation = self.vision.describe(image_b64)
                vlm_time = time.time() - start

                self.state.vlm_count += 1
                self.state.last_vlm_time = now
                short_obs = observation[:50] + "..." if len(observation) > 50 else observation
                self.log(f"VLM ({vlm_time:.1f}s): {short_obs}", "👁️")

                # Update scene reference
                self.scene_detector.update_reference(frame)

                # Layer 4: World Model Update
                world_result = self.world.update(observation)

                if world_result["changed"]:
                    self.log(f"World: {world_result['change_type']}", "🌍")

                # Layer 5: Brain (only if world says to act)
                if world_result["should_act"]:
                    if now - self.state.last_speech_time >= self.min_brain_interval:
                        self.state.mode = "engaging"
                        self.log("Brain thinking...", "🧠")

                        # Give brain the full context
                        context = world_result["context"]
                        speech = self.brain.think(observation)  # TODO: Pass context

                        if speech:
                            self.state.brain_count += 1
                            self.state.last_speech_time = now
                            self.log(f"Say: {speech}", "🗣️")
                            self.voice.speak_async(speech)
                            self.world.mark_speech_sent()
                        else:
                            self.log("Nothing to say", "🤫")
                    else:
                        self.log("Brain: cooling down", "⏳")

                return True
            else:
                self.log(f"Scene stable (diff: {scene_diff:.2f})", "📷")
        else:
            self.state.mode = "watching"

        return False

    def draw_ui(self, frame, motion_result):
        """Draw intelligent UI overlay."""
        h, w = frame.shape[:2]

        # Create overlay
        overlay = frame.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 30), -1)

        # Bottom info bar
        cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 30), -1)

        # Right panel for logs
        cv2.rectangle(overlay, (w - 360, 80), (w, h - 100), (25, 25, 35), -1)

        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw motion regions
        for (x, y, mw, mh) in motion_result.regions:
            cv2.rectangle(frame, (x, y), (x + mw, y + mh), (0, 255, 255), 2)

        # Mode indicator with color
        mode_colors = {
            "watching": (100, 100, 100),
            "detecting": (0, 200, 255),
            "analyzing": (0, 255, 0),
            "engaging": (255, 100, 0),
        }
        color = mode_colors.get(self.state.mode, (100, 100, 100))

        cv2.circle(frame, (35, 35), 20, color, -1)
        cv2.putText(frame, self.state.mode.upper(), (65, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Stats
        stats = f"Motion: {self.state.motion_count} | VLM: {self.state.vlm_count} | Brain: {self.state.brain_count}"
        cv2.putText(frame, stats, (w - 350, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Title
        cv2.putText(frame, "INTELLIGENT AGENT", (w // 2 - 100, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # World model summary
        summary = self.world.get_summary()
        cv2.putText(frame, f"World: {summary}", (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Activity level
        activity = self.world.state.activity_level
        cv2.putText(frame, f"Activity: {activity}", (15, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # People count
        people = f"People tracked: {self.world.state.people_count}"
        cv2.putText(frame, people, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Log panel
        cv2.putText(frame, "Activity Log", (w - 350, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        for i, entry in enumerate(self.logs[-12:]):
            y = 125 + i * 18
            text = entry[:45] if len(entry) > 45 else entry
            cv2.putText(frame, text, (w - 350, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)

        return frame

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("   🧠 INTELLIGENT VISION AGENT")
        print("   Hierarchical Processing + Continuous Context")
        print("=" * 60)
        print("   Press 'q' to quit | Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Ctrl+C handler
        def handler(sig, frame):
            print("\n👋 Shutting down...")
            self.running = False
        signal.signal(signal.SIGINT, handler)

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.log("Agent online", "✅")
        self.voice.speak("Intelligent agent is now active. I'll let you know when I see something interesting.")

        # Initialize scene reference
        ret, frame = cap.read()
        if ret:
            self.scene_detector.update_reference(frame)
            self.log("Scene reference captured", "📷")

        # Create window
        cv2.namedWindow("Intelligent Agent", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Intelligent Agent", 1280, 720)

        frame_count = 0
        process_every_n = 3  # Process every 3rd frame for efficiency

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1

                # Motion detection on every frame
                motion_result = self.motion_detector.detect(frame)

                # Full processing every N frames when motion
                if frame_count % process_every_n == 0 and motion_result.detected:
                    self.process_frame(frame)
                elif not motion_result.detected:
                    self.state.mode = "watching"

                # Always draw UI
                display = self.draw_ui(frame.copy(), motion_result)
                cv2.imshow("Intelligent Agent", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset scene reference
                    self.scene_detector.update_reference(frame)
                    self.log("Scene reference reset", "🔄")

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.voice.speak("Intelligent agent signing off. Goodbye!")


def main():
    import requests

    # Check VLM server
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        print("✅ VLM Server: OK")
    except:
        print("❌ VLM server not running! Start with: ./start_server.sh")
        return

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        # Try loading from .env
        env_path = os.path.expanduser("~/.claude/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        os.environ["OPENAI_API_KEY"] = key
                        break

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("✅ OpenAI API: OK")

    agent = IntelligentAgent()
    agent.run()


if __name__ == "__main__":
    main()
