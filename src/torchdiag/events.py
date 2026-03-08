from __future__ import annotations

import time
from typing import Callable, Optional

from .models import EventType


def normalize_event_type(value: str | EventType) -> EventType:
    if isinstance(value, EventType):
        return value
    try:
        return EventType(value)
    except ValueError:
        return EventType.UNKNOWN


class EventContext:
    def __init__(
        self,
        event_type: EventType,
        on_event: Callable[[EventType, float, float, Optional[int]], None],
        step_getter: Callable[[], Optional[int]],
    ) -> None:
        self._event_type = event_type
        self._on_event = on_event
        self._step_getter = step_getter
        self._start: Optional[float] = None

    def __enter__(self) -> "EventContext":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time.time()
        start = self._start if self._start is not None else end
        step = self._step_getter()
        self._on_event(self._event_type, start, end, step)
