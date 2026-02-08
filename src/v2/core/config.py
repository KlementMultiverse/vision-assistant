#!/usr/bin/env python3
"""
Pipeline configuration with validation.

All configs are frozen dataclasses for immutability.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


# =============================================================================
# COMPONENT CONFIGS
# =============================================================================

@dataclass(frozen=True)
class MotionConfig:
    """Motion detection configuration."""
    threshold: int = 25           # Pixel difference threshold (1-255)
    contour_area: int = 10        # Min contour area (scaled)
    frame_alpha: float = 0.01     # Background update rate
    delta_alpha: float = 0.2      # Motion persistence rate
    frame_height: int = 100       # Scaled frame height for processing
    calibration_frames: int = 30  # Frames before ready
    improve_contrast: bool = False
    min_area: int = 500           # Minimum motion area in pixels

    def __post_init__(self):
        if not 1 <= self.threshold <= 255:
            raise ValueError(f"threshold must be 1-255, got {self.threshold}")
        if self.frame_alpha < 0 or self.frame_alpha > 1:
            raise ValueError(f"frame_alpha must be 0-1, got {self.frame_alpha}")


@dataclass(frozen=True)
class PersonConfig:
    """Person detection configuration."""
    model: str = "yolov8n.pt"     # Model file
    confidence: float = 0.5       # Detection threshold
    iou_threshold: float = 0.45   # NMS threshold
    device: str = "auto"          # cuda, cpu, or auto
    max_detections: int = 10      # Max persons per frame
    min_box_area: int = 1000      # Min detection area in pixels

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError(f"iou_threshold must be 0-1, got {self.iou_threshold}")


@dataclass(frozen=True)
class FaceConfig:
    """Face recognition configuration."""
    detection_threshold: float = 0.5
    recognition_threshold: float = 0.6
    max_retries: int = 10         # Retries for best snapshot
    min_face_size: int = 80       # Min face size in pixels
    database_path: str = "faces.npz"
    model_name: str = "buffalo_l"  # InsightFace model
    use_gpu: bool = True

    def __post_init__(self):
        if not 0 <= self.detection_threshold <= 1:
            raise ValueError(f"detection_threshold must be 0-1, got {self.detection_threshold}")
        if not 0 <= self.recognition_threshold <= 1:
            raise ValueError(f"recognition_threshold must be 0-1, got {self.recognition_threshold}")


@dataclass(frozen=True)
class TrackerConfig:
    """Object tracking configuration."""
    # Matching
    iou_threshold: float = 0.3    # Min IoU to match detection to track

    # Lifecycle
    confirm_frames: int = 3       # Frames to confirm NEW
    lost_frames_max: int = 30     # Frames before END (~1 sec at 30fps)

    # Stationary detection
    movement_threshold: int = 10  # Pixels
    stationary_frames: int = 30   # Frames to mark stationary

    # Event emission
    update_interval: float = 1.0  # Seconds between UPDATE events

    # Track limits
    max_tracks: int = 50          # Max concurrent tracks

    def __post_init__(self):
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError(f"iou_threshold must be 0-1, got {self.iou_threshold}")
        if self.confirm_frames < 1:
            raise ValueError(f"confirm_frames must be >= 1, got {self.confirm_frames}")


# =============================================================================
# PIPELINE CONFIG
# =============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration."""
    motion: MotionConfig = field(default_factory=MotionConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # Pipeline behavior
    face_on_entry_only: bool = True    # Only run face on NEW events
    skip_frames_on_empty: int = 5      # Skip N frames when no motion
    enable_voice: bool = True          # Enable voice greetings
    enable_logging: bool = True        # Enable event logging

    # Camera
    camera_id: int = 0
    camera_name: str = "front_door"

    # Performance
    target_fps: int = 30
    process_every_n_frames: int = 1    # Process every Nth frame

    # Edge cases from architecture doc
    motion_magnitude_threshold: float = 0.4  # Skip if motion > this (lighting change) - Arch spec says 0.4
    min_event_duration: float = 0.5    # Minimum event duration in seconds
    reentry_timeout: float = 300.0     # Seconds before same face = new event (5 min)
    motion_debounce_frames: int = 2    # Consecutive frames with motion before triggering (camera shake filter)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary (e.g., YAML file)."""
        return cls(
            motion=MotionConfig(**data.get("motion", {})),
            person=PersonConfig(**data.get("person", {})),
            face=FaceConfig(**data.get("face", {})),
            tracker=TrackerConfig(**data.get("tracker", {})),
            face_on_entry_only=data.get("face_on_entry_only", True),
            skip_frames_on_empty=data.get("skip_frames_on_empty", 5),
            enable_voice=data.get("enable_voice", True),
            enable_logging=data.get("enable_logging", True),
            camera_id=data.get("camera_id", 0),
            camera_name=data.get("camera_name", "front_door"),
            target_fps=data.get("target_fps", 30),
            process_every_n_frames=data.get("process_every_n_frames", 1),
            motion_magnitude_threshold=data.get("motion_magnitude_threshold", 0.4),
            min_event_duration=data.get("min_event_duration", 0.5),
            reentry_timeout=data.get("reentry_timeout", 300.0),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "motion": {
                "threshold": self.motion.threshold,
                "contour_area": self.motion.contour_area,
                "frame_alpha": self.motion.frame_alpha,
                "delta_alpha": self.motion.delta_alpha,
                "frame_height": self.motion.frame_height,
                "calibration_frames": self.motion.calibration_frames,
                "improve_contrast": self.motion.improve_contrast,
                "min_area": self.motion.min_area,
            },
            "person": {
                "model": self.person.model,
                "confidence": self.person.confidence,
                "iou_threshold": self.person.iou_threshold,
                "device": self.person.device,
                "max_detections": self.person.max_detections,
                "min_box_area": self.person.min_box_area,
            },
            "face": {
                "detection_threshold": self.face.detection_threshold,
                "recognition_threshold": self.face.recognition_threshold,
                "max_retries": self.face.max_retries,
                "min_face_size": self.face.min_face_size,
                "database_path": self.face.database_path,
                "model_name": self.face.model_name,
                "use_gpu": self.face.use_gpu,
            },
            "tracker": {
                "iou_threshold": self.tracker.iou_threshold,
                "confirm_frames": self.tracker.confirm_frames,
                "lost_frames_max": self.tracker.lost_frames_max,
                "movement_threshold": self.tracker.movement_threshold,
                "stationary_frames": self.tracker.stationary_frames,
                "update_interval": self.tracker.update_interval,
                "max_tracks": self.tracker.max_tracks,
            },
            "face_on_entry_only": self.face_on_entry_only,
            "skip_frames_on_empty": self.skip_frames_on_empty,
            "enable_voice": self.enable_voice,
            "enable_logging": self.enable_logging,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "target_fps": self.target_fps,
            "process_every_n_frames": self.process_every_n_frames,
            "motion_magnitude_threshold": self.motion_magnitude_threshold,
            "min_event_duration": self.min_event_duration,
            "reentry_timeout": self.reentry_timeout,
            "motion_debounce_frames": self.motion_debounce_frames,
        }


# =============================================================================
# DEFAULT CONFIGS
# =============================================================================

# Fast config for testing
FAST_CONFIG = PipelineConfig(
    motion=MotionConfig(threshold=30, min_area=300),
    person=PersonConfig(confidence=0.4),
    face=FaceConfig(detection_threshold=0.4, max_retries=3),
    tracker=TrackerConfig(confirm_frames=2, lost_frames_max=15),
)

# Accurate config for production
ACCURATE_CONFIG = PipelineConfig(
    motion=MotionConfig(threshold=20, min_area=800),
    person=PersonConfig(confidence=0.6),
    face=FaceConfig(detection_threshold=0.6, recognition_threshold=0.7),
    tracker=TrackerConfig(confirm_frames=5, lost_frames_max=60),
)

# Low resource config (CPU only)
LOW_RESOURCE_CONFIG = PipelineConfig(
    person=PersonConfig(device="cpu"),
    face=FaceConfig(use_gpu=False),
    process_every_n_frames=2,
)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing configuration...")

    # Default config
    config = PipelineConfig()
    print(f"Default config created: {config.camera_name}")

    # Validate motion config
    try:
        bad_config = MotionConfig(threshold=300)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Validation works: {e}")

    # From dict
    data = {
        "camera_name": "back_door",
        "motion": {"threshold": 30},
        "tracker": {"confirm_frames": 5},
    }
    config2 = PipelineConfig.from_dict(data)
    print(f"From dict: {config2.camera_name}, threshold={config2.motion.threshold}")

    # To dict
    d = config.to_dict()
    print(f"To dict keys: {list(d.keys())}")

    # Preset configs
    print(f"FAST_CONFIG: confirm_frames={FAST_CONFIG.tracker.confirm_frames}")
    print(f"ACCURATE_CONFIG: confirm_frames={ACCURATE_CONFIG.tracker.confirm_frames}")

    print("\nConfiguration module OK!")
