#!/usr/bin/env python3
"""
Voice Module - POC Version
==========================
Simple TTS using pyttsx3.
Non-blocking audio playback.
"""

import threading
import queue
from typing import Optional
from dataclasses import dataclass


@dataclass
class VoiceConfig:
    """Voice configuration."""
    rate: int = 150  # Words per minute
    volume: float = 0.9  # 0.0 to 1.0


class SimpleVoice:
    """
    POC Voice output using pyttsx3.

    Uses a background thread for non-blocking speech.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        """
        Initialize voice.

        Args:
            config: Voice configuration
        """
        import pyttsx3

        self.config = config or VoiceConfig()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.config.rate)
        self.engine.setProperty('volume', self.config.volume)

        # Background thread for non-blocking speech
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background worker that processes speech queue."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:  # Shutdown signal
                    break
                self.engine.say(text)
                self.engine.runAndWait()
            except queue.Empty:
                continue

    def speak(self, text: str):
        """
        Speak text (blocking).

        Args:
            text: Text to speak
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def speak_async(self, text: str):
        """
        Speak text without blocking.

        Args:
            text: Text to speak
        """
        self._queue.put(text)

    def stop(self):
        """Stop all speech and shutdown."""
        self.engine.stop()
        self._running = False
        self._queue.put(None)  # Signal worker to stop
        self._thread.join(timeout=1.0)

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return not self._queue.empty()


# =============================================================================
# COMMON RESPONSES - Pre-defined for doorbell
# =============================================================================

RESPONSES = {
    "greeting": "Hello! How can I help you?",
    "delivery": "Thank you! Please leave the package by the door.",
    "wait": "Please wait, I'm notifying the owner.",
    "no_answer": "I'm sorry, no one is available right now. Please leave a message.",
    "goodbye": "Goodbye!",
}


def get_response(key: str) -> str:
    """Get a pre-defined response."""
    return RESPONSES.get(key, RESPONSES["greeting"])


# =============================================================================
# TEST - Run this file directly to test
# =============================================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  VOICE MODULE POC TEST")
    print("=" * 60)

    voice = SimpleVoice()

    print("\n1. Testing blocking speech...")
    voice.speak("Hello! This is a blocking test.")
    print("   Done.")

    print("\n2. Testing async speech...")
    voice.speak_async("This is an async test.")
    print("   Queued. Doing other work...")
    for i in range(3):
        print(f"   Working... {i+1}")
        time.sleep(1)

    print("\n3. Testing pre-defined responses...")
    for key in ["greeting", "delivery", "goodbye"]:
        print(f"   {key}: {get_response(key)}")
        voice.speak_async(get_response(key))
        time.sleep(2)

    print("\n4. Cleanup...")
    voice.stop()
    print("   Done.")

    print("\n✅ Voice Module WORKS!")
