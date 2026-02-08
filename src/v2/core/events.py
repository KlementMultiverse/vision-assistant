#!/usr/bin/env python3
"""
Event definitions and event bus for the detection pipeline.

Events are the primary way components communicate state changes.
The EventBus allows decoupled event handling.
"""

import logging
import threading
import queue
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .models import DetectionEvent, EventLifecycle, EventLabel

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """All event types in the pipeline."""

    # Object lifecycle events
    OBJECT_NEW = auto()           # New object detected and confirmed
    OBJECT_UPDATE = auto()        # Object position/state updated
    OBJECT_LOST = auto()          # Object temporarily lost
    OBJECT_RECOVERED = auto()     # Lost object recovered
    OBJECT_END = auto()           # Object tracking ended

    # Person-specific events
    PERSON_ENTERED = auto()       # Person confirmed entering
    PERSON_IDENTIFIED = auto()    # Person's face recognized
    PERSON_LEFT = auto()          # Person left the scene
    PERSON_STATIONARY = auto()    # Person stopped moving

    # Face events
    FACE_DETECTED = auto()        # Face detected (may be unknown)
    FACE_RECOGNIZED = auto()      # Known face recognized
    FACE_REGISTERED = auto()      # New face registered

    # Motion events
    MOTION_STARTED = auto()       # Motion began
    MOTION_STOPPED = auto()       # Motion ended

    # System events
    PIPELINE_STARTED = auto()     # Pipeline started
    PIPELINE_STOPPED = auto()     # Pipeline stopped
    CAMERA_ERROR = auto()         # Camera error occurred
    DETECTOR_ERROR = auto()       # Detector error occurred


# =============================================================================
# EVENT WRAPPER
# =============================================================================

@dataclass
class Event:
    """
    Wrapper for events with metadata.

    This is different from DetectionEvent - this is for the event bus.
    """
    type: EventType
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # Component that generated the event

    # Optional: Link to full DetectionEvent
    detection_event: Optional[DetectionEvent] = None


# =============================================================================
# EVENT BUS
# =============================================================================

# Type alias for event handlers
EventHandler = Callable[[Event], None]


class EventBus:
    """
    Central event bus for the pipeline.

    Features:
    - Subscribe to specific event types or all events
    - Async event handling with thread pool
    - Event history for debugging
    - Thread-safe
    """

    def __init__(
        self,
        max_workers: int = 4,
        history_size: int = 100,
        async_handlers: bool = True
    ):
        """
        Initialize event bus.

        Args:
            max_workers: Max threads for async handlers
            history_size: Number of events to keep in history
            async_handlers: Whether to run handlers asynchronously
        """
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._history: List[Event] = []
        self._history_size = history_size
        self._lock = threading.RLock()
        self._async = async_handlers

        if async_handlers:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self._executor = None

        self._running = True

    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler
    ) -> None:
        """
        Subscribe to events.

        Args:
            event_type: Event type to subscribe to, or None for all events
            handler: Callback function(event) -> None
        """
        with self._lock:
            if event_type is None:
                self._global_handlers.append(handler)
            else:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler
    ) -> bool:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        with self._lock:
            if event_type is None:
                if handler in self._global_handlers:
                    self._global_handlers.remove(handler)
                    return True
            else:
                if event_type in self._handlers:
                    if handler in self._handlers[event_type]:
                        self._handlers[event_type].remove(handler)
                        return True
            return False

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event: Event to emit
        """
        if not self._running:
            return

        with self._lock:
            # Add to history
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

            # Get handlers
            handlers = list(self._global_handlers)
            if event.type in self._handlers:
                handlers.extend(self._handlers[event.type])

        # Call handlers
        for handler in handlers:
            if self._async and self._executor:
                self._executor.submit(self._call_handler, handler, event)
            else:
                self._call_handler(handler, event)

    def emit_simple(
        self,
        event_type: EventType,
        source: str = "",
        **data
    ) -> Event:
        """
        Emit an event with simple data.

        Args:
            event_type: Type of event
            source: Source component
            **data: Event data

        Returns:
            The created event
        """
        import time
        event = Event(
            type=event_type,
            timestamp=time.time(),
            data=data,
            source=source
        )
        self.emit(event)
        return event

    def _call_handler(self, handler: EventHandler, event: Event) -> None:
        """Call handler with error handling."""
        try:
            handler(event)
        except Exception as e:
            logger.error(f"Event handler error for {event.type}: {e}")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (None for all)
            limit: Max events to return

        Returns:
            List of events (newest first)
        """
        with self._lock:
            if event_type is None:
                events = list(self._history)
            else:
                events = [e for e in self._history if e.type == event_type]
            return events[-limit:][::-1]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()

    def shutdown(self) -> None:
        """Shutdown the event bus."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)


# =============================================================================
# GLOBAL EVENT BUS INSTANCE
# =============================================================================

_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_bus
    if _global_bus:
        _global_bus.shutdown()
    _global_bus = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def subscribe(event_type: Optional[EventType], handler: EventHandler) -> None:
    """Subscribe to events on the global bus."""
    get_event_bus().subscribe(event_type, handler)


def emit(event: Event) -> None:
    """Emit an event on the global bus."""
    get_event_bus().emit(event)


def emit_simple(event_type: EventType, source: str = "", **data) -> Event:
    """Emit a simple event on the global bus."""
    return get_event_bus().emit_simple(event_type, source, **data)


# =============================================================================
# EVENT LOGGING HANDLER
# =============================================================================

class EventLogger:
    """Handler that logs all events."""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self._logger = logging.getLogger("events")

    def __call__(self, event: Event) -> None:
        """Log the event."""
        self._logger.log(
            self.log_level,
            f"[{event.type.name}] {event.source}: {event.data}"
        )


# =============================================================================
# EVENT FILTER
# =============================================================================

class EventFilter:
    """Filter events before passing to handler."""

    def __init__(
        self,
        handler: EventHandler,
        event_types: Optional[List[EventType]] = None,
        sources: Optional[List[str]] = None
    ):
        """
        Create filtered handler.

        Args:
            handler: Handler to call for matching events
            event_types: Event types to allow (None = all)
            sources: Sources to allow (None = all)
        """
        self.handler = handler
        self.event_types = set(event_types) if event_types else None
        self.sources = set(sources) if sources else None

    def __call__(self, event: Event) -> None:
        """Filter and handle event."""
        if self.event_types and event.type not in self.event_types:
            return
        if self.sources and event.source not in self.sources:
            return
        self.handler(event)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("Testing event system...")

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create bus
    bus = EventBus(async_handlers=False)  # Sync for testing

    # Test handler
    received_events = []

    def handler(event: Event):
        received_events.append(event)
        print(f"  Received: {event.type.name}")

    # Subscribe
    bus.subscribe(EventType.PERSON_ENTERED, handler)
    bus.subscribe(None, EventLogger())  # Log all

    # Emit events
    bus.emit_simple(EventType.MOTION_STARTED, source="motion")
    bus.emit_simple(EventType.PERSON_ENTERED, source="tracker", name="John")
    bus.emit_simple(EventType.PERSON_LEFT, source="tracker")

    # Check
    print(f"\nReceived {len(received_events)} PERSON_ENTERED events")

    # History
    history = bus.get_history(limit=10)
    print(f"History: {len(history)} events")

    # Test filter
    filtered_handler = EventFilter(
        handler,
        event_types=[EventType.PERSON_ENTERED, EventType.PERSON_LEFT]
    )
    bus.subscribe(None, filtered_handler)

    bus.emit_simple(EventType.MOTION_STARTED)  # Should be filtered
    bus.emit_simple(EventType.PERSON_LEFT)     # Should pass

    # Cleanup
    bus.shutdown()

    print("\nEvent system OK!")
