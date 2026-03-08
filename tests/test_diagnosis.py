from torchdiag.diagnosis import diagnose
from torchdiag.models import (
    EventType,
    MetricSample,
    RunData,
    RunMetadata,
    StepRecord,
)


def _make_metadata() -> RunMetadata:
    return RunMetadata(
        job_name="test",
        start_time=0.0,
        end_time=1.0,
        sample_interval_ms=100,
        snapshot_interval_steps=1,
        python_version="3.11",
        torch_version="2.x",
        cuda_available=True,
        cuda_device_name="FakeGPU",
        hostname="host",
        platform="linux",
        pid=123,
    )


def test_diagnosis_rules_trigger():
    steps = []
    durations = [100, 100, 100, 400, 100]
    for idx, duration in enumerate(durations):
        steps.append(
            StepRecord(
                step=idx,
                start_time=float(idx),
                end_time=float(idx + 1),
                duration_ms=duration,
                batch_size=32,
                event_durations_ms={EventType.DATALOADER_WAIT: 80.0},
                dataloader_wait_ms=80.0,
            )
        )

    samples = [
        MetricSample(
            timestamp=0.0,
            cpu_percent=50.0,
            rss_mb=500.0,
            gpu_utilization=10.0,
            gpu_mem_used_mb=950.0,
            gpu_mem_total_mb=1000.0,
            torch_cuda_allocated_mb=900.0,
            torch_cuda_reserved_mb=950.0,
        )
    ]

    run = RunData(
        metadata=_make_metadata(),
        steps=steps,
        samples=samples,
        snapshots=[],
        events=[],
    )

    findings = diagnose(run)
    issue_types = {finding.issue_type for finding in findings}

    assert "input_pipeline_starvation" in issue_types
    assert "gpu_underutilization" in issue_types
    assert "memory_pressure" in issue_types
    assert "unstable_step_times" in issue_types
