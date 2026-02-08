#!/usr/bin/env python3
"""
Vision Assistant - Gradio Web UI
Live camera feed + Agent speaking panel
"""

import gradio as gr
import cv2
import time
import base64
import numpy as np
from datetime import datetime
from camera import Camera
from vision import VisionModel
from voice import Voice


# Global state
class AgentState:
    def __init__(self):
        self.running = False
        self.camera = None
        self.vision = VisionModel("http://localhost:8080")
        self.voice = Voice()
        self.logs = []
        self.greeted = False
        self.last_person_time = 0
        self.person_present = False

state = AgentState()


def add_log(message: str, msg_type: str = "info"):
    """Add a log entry with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "info": "ℹ️",
        "person": "👤",
        "speak": "🔊",
        "action": "🎯",
        "warning": "⚠️",
        "success": "✅"
    }
    icon = icons.get(msg_type, "•")
    entry = f"[{timestamp}] {icon} {message}"
    state.logs.append(entry)
    if len(state.logs) > 20:
        state.logs = state.logs[-20:]


def get_status_html():
    """Generate status panel HTML."""
    status = "🟢 RUNNING" if state.running else "⚪ STOPPED"
    person = "👤 Person detected" if state.person_present else "👻 No one here"

    return f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; color: white; font-family: 'Segoe UI', sans-serif;">
        <h2 style="margin: 0 0 20px 0; color: #00d4ff;">🤖 Vision Assistant</h2>
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px 25px; border-radius: 10px;">
                <div style="font-size: 14px; opacity: 0.7;">Status</div>
                <div style="font-size: 18px; font-weight: bold;">{status}</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px 25px; border-radius: 10px;">
                <div style="font-size: 14px; opacity: 0.7;">Detection</div>
                <div style="font-size: 18px; font-weight: bold;">{person}</div>
            </div>
        </div>
    </div>
    """


def get_logs_display():
    """Get formatted logs for display."""
    if not state.logs:
        return "Waiting to start..."
    return "\n".join(state.logs[-15:])


def create_placeholder():
    """Create a placeholder image when camera is off."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 50)
    cv2.putText(img, "Camera Inactive", (180, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
    cv2.putText(img, "Click START to begin", (190, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_and_get_frame():
    """Process frame from camera and return it."""
    if not state.running or not state.camera:
        return create_placeholder(), get_status_html(), get_logs_display()

    ret, frame = state.camera.capture()
    if not ret or frame is None:
        return create_placeholder(), get_status_html(), get_logs_display()

    # Convert to base64 for VLM
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode()

    # Analyze with VLM
    try:
        result = state.vision.analyze(image_b64)
        now = time.time()

        if result["person"]:
            state.person_present = True
            state.last_person_time = now

            if not state.greeted:
                add_log(f"Person detected! Action: {result['action']}", "person")

                if result["looking"]:
                    msg = "Hello! I can see you looking at me!"
                else:
                    msg = "Hello! I see someone there."

                add_log(f"Speaking: \"{msg}\"", "speak")
                state.voice.speak_async(msg)
                state.greeted = True
        else:
            state.person_present = False
            if state.greeted and (now - state.last_person_time > 5):
                add_log("No one here, resetting...", "info")
                state.greeted = False

    except Exception as e:
        add_log(f"Error: {str(e)[:50]}", "warning")

    # Add overlay to frame
    frame_display = frame.copy()
    overlay_color = (0, 255, 0) if state.person_present else (128, 128, 128)
    cv2.rectangle(frame_display, (10, 10), (200, 50), overlay_color, -1)
    cv2.rectangle(frame_display, (10, 10), (200, 50), (255, 255, 255), 2)

    status_text = "PERSON" if state.person_present else "SCANNING..."
    cv2.putText(frame_display, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
    return frame_rgb, get_status_html(), get_logs_display()


def start_agent():
    """Start the vision agent."""
    if state.running:
        return create_placeholder(), get_status_html(), get_logs_display()

    try:
        state.camera = Camera(0)
        state.camera.open()
        state.running = True
        state.logs = []
        add_log("Vision Assistant started!", "success")
        add_log("Watching for people...", "info")
        state.voice.speak_async("Vision assistant is now active.")
    except Exception as e:
        add_log(f"Failed to start: {e}", "warning")

    return process_and_get_frame()


def stop_agent():
    """Stop the vision agent."""
    state.running = False
    if state.camera:
        state.camera.release()
        state.camera = None

    add_log("Vision Assistant stopped.", "info")
    state.voice.speak_async("Goodbye!")
    state.greeted = False
    state.person_present = False

    return create_placeholder(), get_status_html(), get_logs_display()


# Build Gradio Interface
with gr.Blocks(title="Vision Assistant") as app:

    gr.Markdown("""
    # 🤖 Vision Assistant
    *AI-powered camera monitoring with voice interaction*
    ---
    """)

    with gr.Row():
        # Left: Camera Feed
        with gr.Column(scale=3):
            camera_feed = gr.Image(
                label="📷 Live Camera Feed",
                value=create_placeholder(),
                height=480
            )

        # Right: Agent Panel
        with gr.Column(scale=2):
            status_panel = gr.HTML(get_status_html())

            gr.Markdown("### 📝 Agent Log")
            log_display = gr.Textbox(
                value="Click START to begin...",
                label="",
                lines=12,
                max_lines=15,
                interactive=False
            )

    with gr.Row():
        start_btn = gr.Button("▶️ START", variant="primary", size="lg")
        stop_btn = gr.Button("⏹️ STOP", variant="stop", size="lg")
        refresh_btn = gr.Button("🔄 Refresh Frame", size="lg")

    # Event handlers
    start_btn.click(
        fn=start_agent,
        outputs=[camera_feed, status_panel, log_display]
    )

    stop_btn.click(
        fn=stop_agent,
        outputs=[camera_feed, status_panel, log_display]
    )

    refresh_btn.click(
        fn=process_and_get_frame,
        outputs=[camera_feed, status_panel, log_display]
    )


if __name__ == "__main__":
    import requests

    print("\n" + "=" * 50)
    print("🤖 VISION ASSISTANT - Web UI")
    print("=" * 50)

    # Check VLM server
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        if r.status_code == 200:
            print("✅ VLM Server: Running")
        else:
            print("⚠️  VLM Server: Not healthy!")
    except:
        print("⚠️  VLM Server: Not running!")
        print("   Start it first with: ./start_server.sh")

    print("\n🌐 Open http://localhost:7860 in your browser")
    print("   Click START, then click REFRESH to process frames")
    print("=" * 50 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
