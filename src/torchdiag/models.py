from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    DATALOADER_WAIT = "dataloader_wait"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    CHECKPOINT_IO = "checkpoint_io"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class RunMetadata:
    job_name: str
    start_time: float
    end_time: Optional[float]
    sample_interval_ms: int
    snapshot_interval_steps: int
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: Optional[str]
    hostname: str
    platform: str
    pid: int


@dataclass
class MetricSample:
    timestamp: float
    cpu_percent: float
    rss_mb: float
    gpu_utilization: Optional[float]
    gpu_mem_used_mb: Optional[float]
    gpu_mem_total_mb: Optional[float]
    torch_cuda_allocated_mb: Optional[float]
    torch_cuda_reserved_mb: Optional[float]


@dataclass
class EventRecord:
    event_type: EventType
    step: Optional[int]
    start_time: float
    end_time: float
    duration_ms: float


@dataclass
class StepRecord:
    step: int
    start_time: float
    end_time: float
    duration_ms: float
    batch_size: Optional[int]
    event_durations_ms: Dict[EventType, float] = field(default_factory=dict)
    dataloader_wait_ms: Optional[float] = None
    forward_ms: Optional[float] = None
    backward_ms: Optional[float] = None
    optimizer_ms: Optional[float] = None
    checkpoint_io_ms: Optional[float] = None


@dataclass
class Snapshot:
    timestamp: float
    step: int
    metrics: MetricSample
    samples_per_sec: Optional[float]


@dataclass
class DiagnosisFinding:
    issue_type: str
    severity: str
    confidence: float
    evidence: str
    recommendations: List[str]


@dataclass
class RunData:
    metadata: RunMetadata
    steps: List[StepRecord]
    samples: List[MetricSample]
    snapshots: List[Snapshot]
    events: List[EventRecord]
    findings: List[DiagnosisFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_dataclass(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RunData":
        return run_data_from_dict(data)


@dataclass
class ComparisonMetric:
    name: str
    run_a: Optional[float]
    run_b: Optional[float]
    delta: Optional[float]
    delta_pct: Optional[float]


@dataclass
class ComparisonResult:
    metrics: List[ComparisonMetric]
    diagnosis_diff: Dict[str, List[str]]


def _serialize_dataclass(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        result: Dict[str, Any] = {}
        for field_info in fields(obj):
            result[field_info.name] = _serialize_dataclass(getattr(obj, field_info.name))
        return result
    if isinstance(obj, list):
        return [_serialize_dataclass(item) for item in obj]
    if isinstance(obj, dict):
        return {
            _serialize_dataclass(key): _serialize_dataclass(value) for key, value in obj.items()
        }
    return obj


def _event_type_from_value(value: str) -> EventType:
    try:
        return EventType(value)
    except ValueError:
        return EventType.UNKNOWN


def run_data_from_dict(data: Dict[str, Any]) -> RunData:
    metadata = RunMetadata(**data["metadata"])

    samples = [MetricSample(**sample) for sample in data.get("samples", [])]
    events = [
        EventRecord(
            event_type=_event_type_from_value(event["event_type"]),
            step=event.get("step"),
            start_time=event["start_time"],
            end_time=event["end_time"],
            duration_ms=event["duration_ms"],
        )
        for event in data.get("events", [])
    ]

    steps = []
    for step in data.get("steps", []):
        event_durations_raw = step.get("event_durations_ms", {})
        event_durations = {
            _event_type_from_value(name): value for name, value in event_durations_raw.items()
        }
        steps.append(
            StepRecord(
                step=step["step"],
                start_time=step["start_time"],
                end_time=step["end_time"],
                duration_ms=step["duration_ms"],
                batch_size=step.get("batch_size"),
                event_durations_ms=event_durations,
                dataloader_wait_ms=step.get("dataloader_wait_ms"),
                forward_ms=step.get("forward_ms"),
                backward_ms=step.get("backward_ms"),
                optimizer_ms=step.get("optimizer_ms"),
                checkpoint_io_ms=step.get("checkpoint_io_ms"),
            )
        )

    snapshots = []
    for snapshot in data.get("snapshots", []):
        metrics = MetricSample(**snapshot["metrics"])
        snapshots.append(
            Snapshot(
                timestamp=snapshot["timestamp"],
                step=snapshot["step"],
                metrics=metrics,
                samples_per_sec=snapshot.get("samples_per_sec"),
            )
        )

    findings = [DiagnosisFinding(**finding) for finding in data.get("findings", [])]

    return RunData(
        metadata=metadata,
        steps=steps,
        samples=samples,
        snapshots=snapshots,
        events=events,
        findings=findings,
    )
