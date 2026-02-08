#!/usr/bin/env python3
"""
Vision API Module - Reliable Scene Understanding
================================================

Uses GPT-4o for accurate scene analysis.
Outputs structured data for downstream processing.

Cost: ~$0.003 per call
"""

import os
import cv2
import base64
import json
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI

# Load API key
env_path = os.path.expanduser("~/.claude/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
                break


@dataclass
class Person:
    """Detected person in frame."""
    appearance: str
    position: str  # left, center, right
    action: str  # standing, sitting, walking, waiting
    facing: str  # toward_camera, away, left, right
    confidence: float = 1.0


@dataclass
class SceneAnalysis:
    """Structured scene understanding."""
    people_count: int
    people: list[Person]
    direction: str  # approaching, leaving, stationary, passing
    objects: list[str]
    setting: str  # indoor, outdoor, doorway
    activity: str  # waiting, passing, entering, exiting
    raw_description: str

    def has_person_waiting(self) -> bool:
        """Check if someone is waiting at door."""
        return (self.people_count > 0 and
                self.activity in ["waiting", "stationary"] and
                any(p.facing == "toward_camera" for p in self.people))

    def to_dict(self) -> dict:
        return {
            "people_count": self.people_count,
            "people": [{"appearance": p.appearance, "position": p.position,
                       "action": p.action, "facing": p.facing} for p in self.people],
            "direction": self.direction,
            "objects": self.objects,
            "setting": self.setting,
            "activity": self.activity
        }


class VisionAPI:
    """GPT-4o powered vision for reliable scene understanding."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.total_calls = 0
        self.total_cost = 0.0

    def analyze_frame(self, frame) -> SceneAnalysis:
        """
        Analyze a camera frame and return structured data.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            SceneAnalysis with structured scene data
        """
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buffer).decode()

        return self.analyze_base64(image_b64)

    def analyze_base64(self, image_b64: str) -> SceneAnalysis:
        """Analyze base64 encoded image."""

        prompt = """Analyze this security camera image. Be ACCURATE - only describe what you actually see.

Return a JSON object with this exact structure:
{
    "people_count": <number>,
    "people": [
        {
            "appearance": "<hair color, clothing, distinguishing features>",
            "position": "<left|center|right>",
            "action": "<standing|sitting|walking|waiting|unknown>",
            "facing": "<toward_camera|away|left|right>"
        }
    ],
    "direction": "<approaching|leaving|stationary|passing|none>",
    "objects": ["<visible objects>"],
    "setting": "<indoor|outdoor|doorway|entrance>",
    "activity": "<waiting|passing|entering|exiting|stationary|none>",
    "description": "<one sentence summary>"
}

Rules:
- Be precise, do not guess or hallucinate
- If no people visible, people_count = 0 and people = []
- direction refers to movement toward/away from camera
- activity describes the overall scene activity

Return ONLY valid JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        # Track costs
        self.total_calls += 1
        tokens = response.usage.total_tokens
        cost = tokens * 0.005 / 1000  # GPT-4o pricing
        self.total_cost += cost

        # Parse response
        try:
            data = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return SceneAnalysis(
                people_count=0,
                people=[],
                direction="none",
                objects=[],
                setting="unknown",
                activity="none",
                raw_description=response.choices[0].message.content
            )

        # Build structured response
        people = []
        for p in data.get("people", []):
            people.append(Person(
                appearance=p.get("appearance", "unknown"),
                position=p.get("position", "center"),
                action=p.get("action", "unknown"),
                facing=p.get("facing", "unknown")
            ))

        return SceneAnalysis(
            people_count=data.get("people_count", 0),
            people=people,
            direction=data.get("direction", "none"),
            objects=data.get("objects", []),
            setting=data.get("setting", "unknown"),
            activity=data.get("activity", "none"),
            raw_description=data.get("description", "")
        )

    def get_stats(self) -> dict:
        """Get API usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_cost": f"${self.total_cost:.4f}",
            "avg_cost_per_call": f"${self.total_cost/max(self.total_calls,1):.4f}"
        }


def capture_frame(camera_id: int = 0):
    """Capture a single frame from camera."""
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VISION API TEST")
    print("=" * 60)

    # Initialize
    vision = VisionAPI()

    # Capture and analyze
    print("\nCapturing frame...")
    frame = capture_frame()

    if frame is None:
        print("ERROR: Could not capture frame")
        exit(1)

    print("Analyzing with GPT-4o...")
    result = vision.analyze_frame(frame)

    print("\n--- STRUCTURED OUTPUT ---")
    print(json.dumps(result.to_dict(), indent=2))

    print(f"\n--- RAW DESCRIPTION ---")
    print(result.raw_description)

    print(f"\n--- DERIVED INFO ---")
    print(f"Person waiting at door: {result.has_person_waiting()}")

    print(f"\n--- API STATS ---")
    print(vision.get_stats())
