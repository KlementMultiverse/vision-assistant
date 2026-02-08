#!/usr/bin/env python3
"""
Brain Module - Intelligent reasoning using OpenAI API
Only called when SmolVLM detects something interesting.
"""

import os
import json
from openai import OpenAI
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class Event:
    """Detected event with analysis."""
    timestamp: str
    raw_observation: str  # From SmolVLM
    event_type: str       # person, package, vehicle, animal, unknown
    description: str      # Detailed analysis
    urgency: str          # high, medium, low, none
    action: str           # notify, alert, ignore, greet
    confidence: float     # 0-1


class Brain:
    """Intelligent reasoning layer using OpenAI."""

    SYSTEM_PROMPT = """You are an intelligent home security assistant analyzing camera footage.

Your job:
1. Analyze what the camera sees (from VLM description)
2. Classify the event type
3. Determine urgency and recommended action
4. Provide clear, concise description

Event types: person, package, vehicle, animal, weather, unknown
Urgency levels: high (threat/emergency), medium (needs attention), low (FYI), none (ignore)
Actions: alert (immediate), notify (can wait), greet (friendly), ignore (not important)

Respond in JSON format:
{
    "event_type": "person|package|vehicle|animal|weather|unknown",
    "description": "Clear description of what's happening",
    "urgency": "high|medium|low|none",
    "action": "alert|notify|greet|ignore",
    "confidence": 0.0-1.0,
    "details": {
        "is_familiar": false,
        "potential_threat": false,
        "package_visible": false,
        "vehicle_type": null,
        "additional_notes": ""
    }
}"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # Fast and cheap, upgrade to gpt-4o if needed
        self.history = []  # Recent events for context

    def analyze(self, vlm_observation: str, context: str = "") -> Event:
        """
        Analyze VLM observation and return structured event.

        Args:
            vlm_observation: Raw text from SmolVLM
            context: Additional context (time of day, recent events, etc.)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build prompt with context
        user_prompt = f"""Camera observation: {vlm_observation}

Time: {timestamp}
Context: {context if context else "Normal monitoring"}
Recent history: {self._get_recent_history()}

Analyze this and respond with JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)

            event = Event(
                timestamp=timestamp,
                raw_observation=vlm_observation,
                event_type=result.get("event_type", "unknown"),
                description=result.get("description", ""),
                urgency=result.get("urgency", "none"),
                action=result.get("action", "ignore"),
                confidence=result.get("confidence", 0.5)
            )

            # Store in history
            self.history.append(event)
            if len(self.history) > 20:
                self.history = self.history[-20:]

            return event

        except Exception as e:
            print(f"[BRAIN ERROR] {e}")
            return Event(
                timestamp=timestamp,
                raw_observation=vlm_observation,
                event_type="unknown",
                description=f"Analysis failed: {str(e)}",
                urgency="none",
                action="ignore",
                confidence=0.0
            )

    def should_analyze(self, vlm_observation: str) -> bool:
        """
        Decide if we should call the API for this observation.
        Saves money by filtering boring observations.
        """
        obs_lower = vlm_observation.lower()

        # Always analyze if person/package detected
        important_keywords = [
            "person", "people", "man", "woman", "someone", "human",
            "package", "box", "delivery", "parcel",
            "car", "vehicle", "truck", "van",
            "dog", "cat", "animal",
            "suspicious", "unknown", "unusual"
        ]

        return any(kw in obs_lower for kw in important_keywords)

    def _get_recent_history(self) -> str:
        """Get summary of recent events for context."""
        if not self.history:
            return "No recent events"

        recent = self.history[-3:]
        return "; ".join([
            f"{e.event_type}:{e.action}" for e in recent
        ])

    def get_voice_response(self, event: Event) -> Optional[str]:
        """Generate what to say based on event."""
        if event.action == "ignore":
            return None

        if event.action == "greet":
            if "package" in event.description.lower():
                return "I see a delivery! A package has arrived."
            return "Hello! I see someone at the door."

        if event.action == "notify":
            return f"Heads up: {event.description}"

        if event.action == "alert":
            return f"Alert! {event.description}"

        return None


# Quick test
if __name__ == "__main__":
    brain = Brain()

    # Test observations
    tests = [
        "A person is standing at the door holding a brown package. They are wearing a blue uniform.",
        "The scene shows an empty porch with no activity.",
        "A dog is walking across the lawn.",
        "Someone in dark clothing is looking through the window."
    ]

    for obs in tests:
        print(f"\n{'='*50}")
        print(f"VLM says: {obs}")

        if brain.should_analyze(obs):
            event = brain.analyze(obs)
            print(f"Event: {event.event_type} | Urgency: {event.urgency} | Action: {event.action}")
            print(f"Description: {event.description}")

            voice = brain.get_voice_response(event)
            if voice:
                print(f"Say: {voice}")
        else:
            print("Skipping analysis (nothing interesting)")
