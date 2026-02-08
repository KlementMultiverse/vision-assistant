#!/usr/bin/env python3
"""
World Model - Continuous Context Awareness
Maintains the agent's understanding of what's happening over time.
"""

import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import re


@dataclass
class TrackedPerson:
    """A person being tracked by the world model."""
    id: str
    first_seen: float
    last_seen: float
    identified: bool = False
    name: Optional[str] = None
    role: Optional[str] = None  # "delivery", "visitor", "resident", etc.
    actions: List[str] = field(default_factory=list)
    engaged: bool = False  # Have we spoken to them?
    response_received: bool = False  # Did they respond?


@dataclass
class WorldState:
    """Current state of the world as understood by the agent."""
    activity_level: str = "idle"  # idle, motion, active, interaction
    people_count: int = 0
    has_package: bool = False
    door_activity: bool = False
    last_change: float = 0
    scene_stable_since: float = 0


class WorldModel:
    """
    Maintains continuous context of what's happening.

    This is the agent's "memory" of the current situation,
    allowing it to understand context without reprocessing every frame.
    """

    def __init__(self, observation_window: int = 10):
        # Current world state
        self.state = WorldState()

        # Tracked people (id -> TrackedPerson)
        self.people: Dict[str, TrackedPerson] = {}
        self.next_person_id = 1

        # Rolling observation window
        self.observations: List[Dict] = []
        self.observation_window = observation_window

        # Conversation tracking
        self.conversation_active = False
        self.conversation_with: Optional[str] = None
        self.waiting_for_response = False
        self.last_speech_time: float = 0
        self.response_timeout = 10.0  # seconds to wait for response

        # Change tracking
        self.last_significant_change: Optional[str] = None
        self.previous_people_count = 0

    def update(self, vlm_observation: str) -> Dict[str, Any]:
        """
        Integrate new VLM observation into world model.

        Returns:
            Dict with:
                - changed: bool - did something significant change?
                - change_type: str - what changed
                - should_act: bool - should brain take action?
                - context: str - full context for brain
        """
        now = time.time()
        obs_lower = vlm_observation.lower()

        # Store observation
        self.observations.append({
            "time": now,
            "text": vlm_observation,
        })
        if len(self.observations) > self.observation_window:
            self.observations = self.observations[-self.observation_window:]

        # Detect people
        person_detected = self._detect_person(obs_lower)

        # Detect objects/events
        package_detected = self._detect_package(obs_lower)
        door_activity = self._detect_door_activity(obs_lower)

        # Update state
        old_state = (self.state.people_count, self.state.has_package,
                     self.state.door_activity)

        if person_detected:
            self._update_people_tracking(vlm_observation, now)
        else:
            self._handle_no_person(now)

        self.state.has_package = package_detected
        self.state.door_activity = door_activity
        self.state.people_count = len([p for p in self.people.values()
                                       if now - p.last_seen < 5])

        # Determine activity level
        self._update_activity_level()

        # Detect changes
        new_state = (self.state.people_count, self.state.has_package,
                     self.state.door_activity)

        changed = old_state != new_state
        change_type = self._determine_change_type(old_state, new_state)

        if changed:
            self.state.last_change = now
            self.last_significant_change = change_type

        # Determine if we should act
        should_act = self._should_brain_act(change_type, now)

        return {
            "changed": changed,
            "change_type": change_type,
            "should_act": should_act,
            "context": self.get_context_for_brain(),
        }

    def _detect_person(self, obs_lower: str) -> bool:
        """Check if observation mentions a person."""
        person_words = ["person", "man", "woman", "someone", "human",
                        "people", "individual", "figure", "visitor"]
        return any(w in obs_lower for w in person_words)

    def _detect_package(self, obs_lower: str) -> bool:
        """Check if observation mentions a package."""
        package_words = ["package", "box", "parcel", "delivery", "cardboard"]
        return any(w in obs_lower for w in package_words)

    def _detect_door_activity(self, obs_lower: str) -> bool:
        """Check for door-related activity."""
        door_words = ["door", "entrance", "doorway", "doorbell", "knocking"]
        return any(w in obs_lower for w in door_words)

    def _update_people_tracking(self, observation: str, now: float):
        """Update tracking of people in scene."""
        # Simple approach: track total people count
        # More sophisticated: use face embeddings for identity

        current_count = self.state.people_count or 1

        # If we have tracked people, update last_seen
        if self.people:
            for person_id in self.people:
                self.people[person_id].last_seen = now
                self.people[person_id].actions.append(observation[:100])
                if len(self.people[person_id].actions) > 5:
                    self.people[person_id].actions = self.people[person_id].actions[-5:]
        else:
            # New person detected
            person_id = f"person_{self.next_person_id}"
            self.next_person_id += 1
            self.people[person_id] = TrackedPerson(
                id=person_id,
                first_seen=now,
                last_seen=now,
                identified=False,
                actions=[observation[:100]]
            )

    def _handle_no_person(self, now: float):
        """Handle case when no person is detected."""
        # Remove people who haven't been seen in a while
        stale_timeout = 10.0  # seconds
        stale_people = [
            pid for pid, p in self.people.items()
            if now - p.last_seen > stale_timeout
        ]
        for pid in stale_people:
            del self.people[pid]

        # Reset conversation if person left
        if not self.people and self.conversation_active:
            self.conversation_active = False
            self.conversation_with = None
            self.waiting_for_response = False

    def _update_activity_level(self):
        """Update overall activity level."""
        if self.conversation_active:
            self.state.activity_level = "interaction"
        elif self.state.people_count > 0:
            self.state.activity_level = "active"
        elif self.state.door_activity:
            self.state.activity_level = "motion"
        else:
            self.state.activity_level = "idle"

    def _determine_change_type(self, old_state, new_state) -> str:
        """Determine what type of change occurred."""
        old_people, old_package, old_door = old_state
        new_people, new_package, new_door = new_state

        if new_people > old_people:
            return "person_arrived"
        elif new_people < old_people and new_people == 0:
            return "person_left"
        elif new_package and not old_package:
            return "package_detected"
        elif new_door and not old_door:
            return "door_activity"
        elif new_people > 0 and old_people > 0:
            return "continued_presence"
        else:
            return "no_change"

    def _should_brain_act(self, change_type: str, now: float) -> bool:
        """Determine if brain should take action."""
        # Always act on significant events
        if change_type in ["person_arrived", "package_detected"]:
            return True

        # Act if we're waiting for response and timeout
        if self.waiting_for_response:
            if now - self.last_speech_time > self.response_timeout:
                self.waiting_for_response = False
                return True  # Follow up or give up

        # Act if person has been present but unengaged for a while
        for person in self.people.values():
            if not person.engaged and now - person.first_seen > 3.0:
                return True

        return False

    def get_context_for_brain(self) -> str:
        """Generate comprehensive context string for brain."""
        now = time.time()

        # Time context
        time_str = datetime.now().strftime("%I:%M %p")

        # Scene duration
        if self.state.last_change:
            duration = now - self.state.last_change
            if duration < 60:
                duration_str = f"{int(duration)} seconds"
            else:
                duration_str = f"{int(duration/60)} minutes"
        else:
            duration_str = "just started"

        # People context
        people_lines = []
        for person in self.people.values():
            time_present = now - person.first_seen
            status = "identified" if person.identified else "unknown"
            engaged = "engaged" if person.engaged else "not yet greeted"
            people_lines.append(
                f"  - {person.id}: {status}, {engaged}, present {int(time_present)}s"
            )
        people_str = "\n".join(people_lines) if people_lines else "  None"

        # Recent observations summary
        recent_obs = self.observations[-3:] if self.observations else []
        obs_lines = []
        for obs in recent_obs:
            text = obs["text"][:80] + "..." if len(obs["text"]) > 80 else obs["text"]
            obs_lines.append(f"  - {text}")
        obs_str = "\n".join(obs_lines) if obs_lines else "  No observations yet"

        # Conversation state
        if self.conversation_active:
            conv_str = f"Active conversation with {self.conversation_with}"
            if self.waiting_for_response:
                wait_time = now - self.last_speech_time
                conv_str += f", waiting for response ({int(wait_time)}s)"
        else:
            conv_str = "No active conversation"

        return f"""Current time: {time_str}
Activity level: {self.state.activity_level}
Scene stable for: {duration_str}

People present ({self.state.people_count}):
{people_str}

Notable objects:
  Package visible: {self.state.has_package}
  Door activity: {self.state.door_activity}

Recent observations:
{obs_str}

Conversation state: {conv_str}

Last significant change: {self.last_significant_change or 'None'}"""

    def mark_speech_sent(self, to_person: Optional[str] = None):
        """Mark that agent has spoken."""
        self.last_speech_time = time.time()
        self.waiting_for_response = True

        if to_person and to_person in self.people:
            self.people[to_person].engaged = True
            self.conversation_active = True
            self.conversation_with = to_person
        elif self.people:
            # Mark first person as engaged
            person_id = list(self.people.keys())[0]
            self.people[person_id].engaged = True
            self.conversation_active = True
            self.conversation_with = person_id

    def receive_response(self, response_text: str):
        """Handle receiving a response from the person."""
        self.waiting_for_response = False

        # Could parse response for identity clues
        # e.g., "I'm here to deliver a package" -> role = "delivery"
        if self.conversation_with and self.conversation_with in self.people:
            person = self.people[self.conversation_with]
            person.response_received = True

            # Simple role detection
            response_lower = response_text.lower()
            if any(w in response_lower for w in ["delivery", "package", "amazon", "ups"]):
                person.role = "delivery"
            elif any(w in response_lower for w in ["friend", "visit", "here to see"]):
                person.role = "visitor"

    def get_summary(self) -> str:
        """Get one-line summary of current state."""
        if self.state.people_count == 0:
            return "Scene is empty"
        elif self.state.people_count == 1:
            person = list(self.people.values())[0]
            role = person.role or "unknown person"
            return f"1 {role} present"
        else:
            return f"{self.state.people_count} people present"


# Test
if __name__ == "__main__":
    model = WorldModel()

    # Simulate observations
    observations = [
        "A person is approaching the front door.",
        "The person is standing at the door, looking at the camera.",
        "The person is wearing a blue uniform and holding a brown box.",
        "The person placed the box on the ground.",
        "The scene shows an empty porch with a box on the ground.",
    ]

    for obs in observations:
        print(f"\n{'='*60}")
        print(f"VLM: {obs}")
        result = model.update(obs)
        print(f"\nChanged: {result['changed']}")
        print(f"Change type: {result['change_type']}")
        print(f"Should act: {result['should_act']}")
        print(f"\n{result['context']}")

        import time
        time.sleep(1)
