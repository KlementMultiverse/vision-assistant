"""Camera module for capturing frames."""

import cv2
import base64
from io import BytesIO


class Camera:
    def __init__(self, device: int = 0):
        self.device = device
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device}")
        return self

    def capture(self) -> tuple:
        """Capture a frame. Returns (success, frame)."""
        if self.cap is None:
            self.open()
        return self.cap.read()

    def capture_base64(self) -> str | None:
        """Capture frame and return as base64 JPEG."""
        ret, frame = self.capture()
        if not ret:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode()

    def release(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()
