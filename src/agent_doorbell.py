#!/usr/bin/env python3
"""
Vision Assistant - Deep Agents Implementation
=============================================

Smart doorbell agent using Deep Agents framework.

Architecture:
  - SmolVLM2 (local, free) for perception
  - GPT-4o-mini (API, cheap) for reasoning
  - pyttsx3 for TTS
  - HITL for sensitive actions

Usage:
  python agent_doorbell.py
"""

import os
import cv2
import base64
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

# Load API key
env_path = os.path.expanduser("~/.claude/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
                break

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Import existing modules
from vision import VisionModel
from voice import Voice


# =============================================================================
# WORLD STATE - Tracks context across observations
# =============================================================================
@dataclass
class WorldState:
    """Persistent state across observations."""
    person_present: bool = False
    person_since: Optional[float] = None
    last_greeting_time: float = 0
    person_type: str = "unknown"  # delivery, visitor, resident, passerby
    observations: list = field(default_factory=list)
    greeting_count: int = 0

    def person_duration(self) -> float:
        """How long has person been present (seconds)."""
        if self.person_present and self.person_since:
            return time.time() - self.person_since
        return 0

    def time_since_greeting(self) -> float:
        """Seconds since last greeting."""
        return time.time() - self.last_greeting_time

    def should_greet(self) -> bool:
        """Determine if we should greet."""
        # Don't greet if we greeted recently (within 30 seconds)
        if self.time_since_greeting() < 30:
            return False
        # Don't greet passersby
        if self.person_type == "passerby":
            return False
        # Greet if person present for at least 2 seconds
        return self.person_present and self.person_duration() >= 2


# Global state
world = WorldState()
vision = VisionModel("http://localhost:8080")
voice = Voice()


# =============================================================================
# TOOLS - Agent actions
# =============================================================================
@tool
def perceive() -> str:
    """
    Capture an image from the camera and describe what you see.
    Uses the local SmolVLM2 model for vision.
    Returns a text description of the current scene.
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error: Could not capture frame from camera"

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image_b64 = base64.b64encode(buffer).decode()

    # Get description from SmolVLM2
    description = vision.describe(
        image_b64,
        "Describe this doorbell camera image briefly. "
        "Focus on: people present, what they're doing, packages, vehicles. "
        "Be specific and concise."
    )

    # Update world state
    PERSON_WORDS = ["person", "man", "woman", "someone", "people", "individual", "he", "she", "they", "visitor", "delivery"]
    person_detected = any(word in description.lower() for word in PERSON_WORDS)

    if person_detected and not world.person_present:
        world.person_present = True
        world.person_since = time.time()
    elif not person_detected:
        world.person_present = False
        world.person_since = None
        world.person_type = "unknown"

    # Store observation
    world.observations.append({
        "time": datetime.now().isoformat(),
        "description": description,
        "person_detected": person_detected
    })
    if len(world.observations) > 10:
        world.observations = world.observations[-10:]

    return description


@tool
def speak(text: str) -> str:
    """
    Speak a message through the speakers.
    Use this to greet visitors, respond to delivery people, etc.

    Args:
        text: The message to speak
    """
    voice.speak_async(text)
    world.last_greeting_time = time.time()
    world.greeting_count += 1
    return f"Spoke: {text}"


@tool
def get_world_state() -> str:
    """
    Get the current state of the world.
    Returns information about person presence, duration, and history.
    """
    return f"""
Current State:
- Person present: {world.person_present}
- Person type: {world.person_type}
- Duration: {world.person_duration():.1f}s
- Should greet: {world.should_greet()}
- Time since last greeting: {world.time_since_greeting():.1f}s
- Total greetings: {world.greeting_count}

Recent observations:
{chr(10).join(f"  - {obs['time']}: {obs['description'][:60]}..." for obs in world.observations[-3:])}
"""


@tool
def classify_visitor(description: str) -> str:
    """
    Classify the type of visitor based on description.
    Updates world state with the classification.

    Args:
        description: The scene description from perceive()
    """
    desc_lower = description.lower()

    if any(word in desc_lower for word in ["delivery", "package", "box", "courier"]):
        world.person_type = "delivery"
    elif any(word in desc_lower for word in ["standing", "waiting", "door"]):
        world.person_type = "visitor"
    elif any(word in desc_lower for word in ["walking", "passing", "moving"]):
        world.person_type = "passerby"
    else:
        world.person_type = "unknown"

    return f"Classified as: {world.person_type}"


# =============================================================================
# AGENT SETUP
# =============================================================================
SYSTEM_PROMPT = """You are a smart doorbell assistant. Your job is to:

1. PERCEIVE: Use the perceive tool to see what's happening
2. CLASSIFY: Determine if it's a delivery person, visitor, resident, or passerby
3. DECIDE: Based on the situation, decide whether to speak
4. ACT: If appropriate, speak a greeting or response

Rules:
- For DELIVERY people: Say "Thank you! Please leave the package by the door."
- For VISITORS waiting: Say "Hello! How can I help you?"
- For PASSERSBY: Don't say anything
- Don't greet the same person multiple times (check world state)
- Be concise and friendly

Always check the world state first to understand the context before acting.
"""


def create_doorbell_agent():
    """Create the Deep Agent for doorbell."""
    # Use GPT-4o-mini for reasoning (cheap and fast)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_deep_agent(
        name="doorbell",
        model=model,
        tools=[perceive, speak, get_world_state, classify_visitor],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# =============================================================================
# MAIN LOOP
# =============================================================================
def run_continuous():
    """Run the agent in continuous mode."""
    print("=" * 60)
    print("  DEEP AGENTS DOORBELL")
    print("  Perception: SmolVLM2 (local)")
    print("  Reasoning: GPT-4o-mini")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    agent = create_doorbell_agent()

    while True:
        try:
            # Run one cycle
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Observing...")

            # Use the agent to handle the observation
            result = agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": "Check the camera and respond appropriately. Use perceive first, then get_world_state, then decide what to do."
                }]
            })

            # Extract final message
            final = result["messages"][-1].content if result.get("messages") else "No response"
            print(f"Agent: {final[:100]}...")

            # Wait before next cycle
            time.sleep(3)

        except KeyboardInterrupt:
            print("\nShutting down...")
            voice.speak("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


def run_single():
    """Run a single observation cycle."""
    print("Running single observation...")

    agent = create_doorbell_agent()

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Check the camera. Perceive what's there, check the world state, classify any visitor, and respond appropriately."
        }]
    })

    print("\n--- Agent Response ---")
    for msg in result.get("messages", []):
        if hasattr(msg, "content"):
            print(f"[{msg.type}] {msg.content}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_single()
    else:
        run_continuous()
