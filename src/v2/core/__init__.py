#!/usr/bin/env python3
"""
Core module for Vision Assistant v2.

Contains data models, protocols, configuration, and events.
"""

from .models import (
    BoundingBox,
    MotionResult,
    PersonDetection,
    PersonResult,
    FaceDetection,
    FaceResult,
    ObjectState,
    TrackedObject,
    EventLifecycle,
    EventLabel,
    DetectionEvent,
    PipelineFrame,
)

from .protocols import (
    MotionDetector,
    PersonDetector,
    FaceRecognizer,
    ObjectTracker,
    EventHandler,
    ActionExecutor,
    Pipeline,
)

from .config import (
    MotionConfig,
    PersonConfig,
    FaceConfig,
    TrackerConfig,
    PipelineConfig,
)

from .events import (
    EventType,
    EventBus,
)

__all__ = [
    # Models
    "BoundingBox",
    "MotionResult",
    "PersonDetection",
    "PersonResult",
    "FaceDetection",
    "FaceResult",
    "ObjectState",
    "TrackedObject",
    "EventLifecycle",
    "EventLabel",
    "DetectionEvent",
    "PipelineFrame",
    # Protocols
    "MotionDetector",
    "PersonDetector",
    "FaceRecognizer",
    "ObjectTracker",
    "EventHandler",
    "ActionExecutor",
    "Pipeline",
    # Config
    "MotionConfig",
    "PersonConfig",
    "FaceConfig",
    "TrackerConfig",
    "PipelineConfig",
    # Events
    "EventType",
    "EventBus",
]
