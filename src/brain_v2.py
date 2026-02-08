#!/usr/bin/env python3
"""
Brain V2 - Intelligent Assistant with Persona
Not just classification - actual thinking and natural speech.
"""

import os
import json
from openai import OpenAI
from datetime import datetime
from typing import Optional


class SmartBrain:
    """An assistant that THINKS, not just classifies."""

    PERSONA = """You are JARVIS, a smart home vision assistant. You watch through a camera and help your owner stay aware of what's happening.

Your personality:
- Friendly but not overly chatty
- Observant and helpful
- You speak naturally, like a helpful friend
- You DON'T sound robotic or formal
- You're concise - a sentence or two max

Your job:
1. Watch the camera feed (you receive descriptions from a vision model)
2. Decide if something is worth mentioning to your owner
3. If yes, say something NATURAL and HELPFUL

When to speak:
- Someone arrives (delivery, visitor, family)
- Something unusual happens
- A package is delivered
- Something your owner should know about

When NOT to speak:
- Nothing interesting is happening
- You just said something similar (don't repeat)
- It's just normal activity (empty scene, etc.)

How to speak:
- Be natural: "Hey, looks like your Amazon package just arrived!"
- Not robotic: "Alert: Package detected. Delivery person present."
- Be brief: One or two sentences max
- Add personality: Be warm but not annoying"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.recent_speech = []  # Track what we said to avoid repetition
        self.last_scene = ""     # Track scene to detect changes

    def think(self, vlm_observation: str) -> Optional[str]:
        """
        Think about what VLM sees and decide what to say (if anything).

        Returns: What to say, or None if nothing worth saying.
        """
        # Skip if scene hasn't really changed
        if self._is_same_scene(vlm_observation):
            return None

        self.last_scene = vlm_observation

        now = datetime.now()
        time_context = now.strftime("%I:%M %p")  # e.g., "3:45 PM"

        prompt = f"""Current time: {time_context}

What the camera sees right now:
"{vlm_observation}"

What you said recently (don't repeat):
{self._get_recent_speech()}

Based on what you see, should you say something to your owner?
If yes, respond with ONLY what you would say (natural speech, 1-2 sentences).
If no (nothing interesting or you'd be repeating yourself), respond with just: SILENT"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.PERSONA},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )

            result = response.choices[0].message.content.strip()

            # Check if brain decided to stay silent
            if result.upper() == "SILENT" or "SILENT" in result.upper():
                return None

            # Don't repeat similar things
            if self._is_repetitive(result):
                return None

            # Track what we said
            self.recent_speech.append({
                "time": now.isoformat(),
                "said": result
            })
            if len(self.recent_speech) > 5:
                self.recent_speech = self.recent_speech[-5:]

            return result

        except Exception as e:
            print(f"[BRAIN ERROR] {e}")
            return None

    def should_think(self, vlm_observation: str) -> bool:
        """Quick check if we should even bother calling the API."""
        obs_lower = vlm_observation.lower()

        # Keywords that warrant thinking
        interesting = [
            "person", "people", "man", "woman", "someone", "human",
            "package", "box", "delivery", "parcel", "mail",
            "car", "vehicle", "truck", "van",
            "dog", "cat", "animal", "pet",
            "door", "entering", "leaving", "arrived",
            "unusual", "suspicious", "unknown"
        ]

        # Keywords that mean nothing interesting
        boring = [
            "empty", "no one", "nobody", "nothing",
            "wall", "floor", "ceiling",
            "the image shows", "the scene is"
        ]

        has_interesting = any(kw in obs_lower for kw in interesting)
        is_boring = any(kw in obs_lower for kw in boring)

        return has_interesting and not is_boring

    def _is_same_scene(self, new_observation: str) -> bool:
        """Check if scene is essentially the same as before."""
        if not self.last_scene:
            return False

        # Simple check - if 70%+ words match, it's same scene
        old_words = set(self.last_scene.lower().split())
        new_words = set(new_observation.lower().split())

        if not old_words or not new_words:
            return False

        overlap = len(old_words & new_words)
        similarity = overlap / max(len(old_words), len(new_words))

        return similarity > 0.7

    def _is_repetitive(self, new_speech: str) -> bool:
        """Check if we're about to say something we just said."""
        if not self.recent_speech:
            return False

        new_lower = new_speech.lower()
        for recent in self.recent_speech[-3:]:
            old_lower = recent["said"].lower()
            # Check for similar phrases
            if self._similarity(new_lower, old_lower) > 0.6:
                return True

        return False

    def _similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0
        overlap = len(words_a & words_b)
        return overlap / max(len(words_a), len(words_b))

    def _get_recent_speech(self) -> str:
        """Format recent speech for context."""
        if not self.recent_speech:
            return "Nothing yet"
        return "; ".join([s["said"][:50] for s in self.recent_speech[-3:]])


# Test
if __name__ == "__main__":
    brain = SmartBrain()

    tests = [
        "A person is standing at the front door holding a brown cardboard box. They are wearing a blue Amazon vest.",
        "The scene shows an empty porch with a welcome mat and some potted plants.",
        "A man in casual clothes is walking up the driveway towards the house.",
        "Someone in dark clothing is looking through the side window.",
        "A dog is sitting on the porch looking at the camera.",
        "The same person from before is still at the door, waiting."
    ]

    for obs in tests:
        print(f"\n{'='*60}")
        print(f"VLM: {obs[:70]}...")

        if brain.should_think(obs):
            response = brain.think(obs)
            if response:
                print(f"🗣️  JARVIS: \"{response}\"")
            else:
                print("🤫 (Silent - nothing new to say)")
        else:
            print("😴 (Skipped - not interesting)")
