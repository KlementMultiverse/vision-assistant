"""Voice module for TTS output."""

import pyttsx3
import threading


class Voice:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed
        self.engine.setProperty('volume', 0.9)
        self._lock = threading.Lock()

    def speak(self, text: str, block: bool = True):
        """Speak text through speakers."""
        with self._lock:
            print(f"[VOICE] {text}")
            self.engine.say(text)
            if block:
                self.engine.runAndWait()

    def speak_async(self, text: str):
        """Speak without blocking."""
        thread = threading.Thread(target=self.speak, args=(text, True))
        thread.start()
        return thread
