from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

from .collectors import MetricCollector, build_metadata
from .compare import compare_run_data, load_run
from .diagnosis import diagnose
from .events import EventContext, normalize_event_type
from .models import EventRecord, EventType, RunData, Snapshot, StepRecord
from .report import write_html as write_html_report


@dataclass
class _StepSession:
    step: int
    start_time: float
    batch_size: Optional[int]
    event_durations: Dict[EventType, float] = field(default_factory=dict)


class TorchDiag:
    def __init__(
        self,
        job_name: str,
        sample_interval_ms: int = 200,
        snapshot_interval_steps: int = 10,
    ) -> None:
        self._job_name = job_name
        self._sample_interval_ms = sample_interval_ms
        self._snapshot_interval_steps = snapshot_interval_steps

        self._collector = MetricCollector()
        self._samples = []
        self._events = []
        self._steps = []
        self._snapshots = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._active_step: Optional[_StepSession] = None
        self._run_data: Optional[RunData] = None
        self._metadata = None
        self._pending_step_events: Dict[int, Dict[EventType, float]] = {}

    def start(self) -> None:
        if self._sampler_thread is not None:
            return
        start_time = time.time()
        self._metadata = build_metadata(
            job_name=self._job_name,
            start_time=start_time,
            sample_interval_ms=self._sample_interval_ms,
            snapshot_interval_steps=self._snapshot_interval_steps,
        )
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._sampler_thread.start()

    def stop(self) -> None:
        if self._sampler_thread is None:
            return
        self._stop_event.set()
        self._sampler_thread.join(timeout=5)
        self._sampler_thread = None
        if self._metadata is None:
            raise RuntimeError("TorchDiag.start() was not called.")
        self._metadata.end_time = time.time()
        self._run_data = RunData(
            metadata=self._metadata,
            steps=list(self._steps),
            samples=list(self._samples),
            snapshots=list(self._snapshots),
            events=list(self._events),
        )
        self._run_data.findings = diagnose(self._run_data)

    @contextmanager
    def step_context(self, step: int, batch_size: Optional[int] = None) -> Iterator[None]:
        start_time = time.time()
        session = _StepSession(step=step, start_time=start_time, batch_size=batch_size)
        self._active_step = session
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            with self._lock:
                pending = self._pending_step_events.pop(step, {})
            event_durations = dict(session.event_durations)
            for event_type, duration in pending.items():
                event_durations[event_type] = event_durations.get(event_type, 0.0) + duration
            step_record = StepRecord(
                step=step,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                batch_size=batch_size,
                event_durations_ms=event_durations,
                dataloader_wait_ms=event_durations.get(EventType.DATALOADER_WAIT),
                forward_ms=event_durations.get(EventType.FORWARD),
                backward_ms=event_durations.get(EventType.BACKWARD),
                optimizer_ms=event_durations.get(EventType.OPTIMIZER),
                checkpoint_io_ms=event_durations.get(EventType.CHECKPOINT_IO),
            )
            with self._lock:
                self._steps.append(step_record)
                if (
                    self._snapshot_interval_steps > 0
                    and step % self._snapshot_interval_steps == 0
                    and self._samples
                ):
                    self._snapshots.append(
                        Snapshot(
                            timestamp=end_time,
                            step=step,
                            metrics=self._samples[-1],
                            samples_per_sec=(
                                batch_size / (duration_ms / 1000)
                                if batch_size and duration_ms > 0
                                else None
                            ),
                        )
                    )
            self._active_step = None

    def event(self, event_type: str | EventType, step: Optional[int] = None) -> EventContext:
        normalized = normalize_event_type(event_type)
        if step is None:
            return EventContext(normalized, self._record_event, self._current_step)
        return EventContext(normalized, self._record_event, lambda: step)

    def write_json(self, path: str) -> None:
        run_data = self._ensure_run_data()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(run_data.to_dict(), indent=2), encoding="utf-8")

    def write_html(self, path: str, compare_to: Optional[str] = None) -> None:
        run_data = self._ensure_run_data()
        comparison = None
        if compare_to:
            comparison_run = load_run(compare_to)
            comparison = compare_run_data(comparison_run, run_data)
        write_html_report(path, run_data, comparison)

    def _ensure_run_data(self) -> RunData:
        if self._run_data is None:
            raise RuntimeError("TorchDiag has not been stopped yet.")
        return self._run_data

    def _sampler_loop(self) -> None:
        interval_s = self._sample_interval_ms / 1000
        while not self._stop_event.is_set():
            timestamp = time.time()
            sample = self._collector.collect(timestamp)
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(interval_s)

    def _record_event(
        self, event_type: EventType, start_time: float, end_time: float, step: Optional[int]
    ) -> None:
        duration_ms = (end_time - start_time) * 1000
        record = EventRecord(
            event_type=event_type,
            step=step,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
        )
        with self._lock:
            self._events.append(record)
        if step is not None and self._active_step and step == self._active_step.step:
            self._active_step.event_durations[event_type] = (
                self._active_step.event_durations.get(event_type, 0.0) + duration_ms
            )
        elif step is not None:
            with self._lock:
                bucket = self._pending_step_events.setdefault(step, {})
                bucket[event_type] = bucket.get(event_type, 0.0) + duration_ms

    def _current_step(self) -> Optional[int]:
        return self._active_step.step if self._active_step else None
