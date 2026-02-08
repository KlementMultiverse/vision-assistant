#!/usr/bin/env python3
"""
Vision Assistant Agent - PoC
Watches through webcam, detects people, and speaks.
"""

import time
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from camera import Camera
from vision import VisionModel
from voice import Voice


class VisionAgent:
    def __init__(self):
        self.camera = Camera(device=0)
        self.vision = VisionModel("http://localhost:8080")
        self.voice = Voice()

        # State
        self.last_person_seen = 0
        self.greeted = False
        self.idle_threshold = 5  # seconds without person before reset

    def run(self, interval: float = 2.0):
        """Main perception loop."""
        print("=" * 50)
        print("VISION ASSISTANT - PoC")
        print("=" * 50)
        print(f"Camera: /dev/video0")
        print(f"VLM: http://localhost:8080")
        print(f"Interval: {interval}s")
        print("=" * 50)
        print("Press Ctrl+C to stop\n")

        self.voice.speak("Vision assistant starting. I can see you now.")

        with self.camera:
            while True:
                try:
                    self.process_frame()
                    time.sleep(interval)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    self.voice.speak("Goodbye!")
                    break

    def process_frame(self):
        """Capture and process one frame."""
        image_b64 = self.camera.capture_base64()
        if not image_b64:
            print("[ERROR] Could not capture frame")
            return

        # Analyze the scene
        start = time.time()
        analysis = self.vision.analyze(image_b64)
        elapsed = time.time() - start

        # Log
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] ({elapsed:.2f}s) person={analysis['person']}, action={analysis['action']}, looking={analysis['looking']}")

        # React based on rules
        self.react(analysis)

    def react(self, analysis: dict):
        """Apply rules and react."""
        now = time.time()

        if analysis["person"]:
            self.last_person_seen = now

            # Greet if new person (or returning after idle)
            if not self.greeted:
                if analysis["looking"]:
                    self.voice.speak_async("Hello! I can see you looking at me.")
                else:
                    self.voice.speak_async("Hello! I see someone there.")
                self.greeted = True

            # React to specific actions
            action = analysis["action"]
            if "wav" in action:
                self.voice.speak_async("Hi there! Nice wave!")

        else:
            # No person - check if we should reset greeting
            if now - self.last_person_seen > self.idle_threshold:
                if self.greeted:
                    print("[STATE] No one here, resetting greeting state")
                    self.greeted = False


def main():
    # Check VLM server is running
    import requests
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        if r.status_code != 200:
            print("ERROR: VLM server not healthy")
            sys.exit(1)
    except:
        print("ERROR: VLM server not running at localhost:8080")
        print("Start it with: ./start_server.sh")
        sys.exit(1)

    agent = VisionAgent()
    agent.run(interval=2.0)


if __name__ == "__main__":
    main()
