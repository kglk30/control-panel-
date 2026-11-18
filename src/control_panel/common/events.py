"""In-process synchronous event bus for Control Panel.

All Orchestrator state transitions emit typed events.
The CLI subscribes to display progress.
Handler exceptions are caught and logged — they must not crash the emitter.
"""

from __future__ import annotations

import traceback
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

__all__ = [
    "EventType",
    "Event",
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
]


class EventType(str, Enum):
    """All event types emitted by Control Panel components."""

    TASK_RECEIVED = "task_received"
    STATE_CHANGED = "state_changed"
    WORKER_STARTED = "worker_started"
    WORKER_STEP_COMPLETE = "worker_step_complete"
    WORKER_COMPLETED = "worker_completed"
    WORKER_FAILED = "worker_failed"
    INFERENCE_STARTED = "inference_started"
    INFERENCE_COMPLETED = "inference_completed"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    EXPERIENCE_STORED = "experience_stored"
    TEMPLATE_UPDATED = "template_updated"
    ERROR = "error"


@dataclass
class Event:
    """A single event emitted on the bus."""

    event_type: EventType
    task_id: Optional[str] = None
    worker_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Type alias for handler callables
Handler = Callable[[Event], None]


class EventBus:
    """Synchronous in-process publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Handler]] = {
            et: [] for et in EventType
        }
        self._global_subscribers: list[Handler] = []
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Subscribe to a specific event type."""
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe to ALL event types."""
        if handler not in self._global_subscribers:
            self._global_subscribers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Unsubscribe from a specific event type."""
        try:
            self._subscribers[event_type].remove(handler)
        except ValueError:
            pass

    def unsubscribe_all(self, handler: Handler) -> None:
        """Unsubscribe from global subscription."""
        try:
            self._global_subscribers.remove(handler)
        except ValueError:
            pass

    def emit(self, event: Event) -> None:
        """Emit an event to all matching subscribers synchronously and thread-safely."""
        all_handlers = (
            self._subscribers.get(event.event_type, []) + self._global_subscribers
        )
        with self._lock:
            for handler in all_handlers:
                try:
                    handler(event)
                except Exception:
                    # Handlers must not crash the emitter - log and continue
                    traceback.print_exc()

    def emit_simple(
        self,
        event_type: EventType,
        task_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        **data: Any,
    ) -> None:
        """Convenience method: construct and emit an Event."""
        self.emit(Event(
            event_type=event_type,
            task_id=task_id,
            worker_id=worker_id,
            data=dict(data),
        ))

    def clear(self) -> None:
        """Remove all subscribers. Used in testing."""
        for et in EventType:
            self._subscribers[et] = []
        self._global_subscribers = []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Return the global EventBus singleton, creating it if needed."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Reset the global singleton. Use in tests to ensure clean state."""
    global _bus
    _bus = EventBus()
