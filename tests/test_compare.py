import json

from torchdiag.compare import compare_runs
from torchdiag.models import MetricSample, RunData, RunMetadata, StepRecord


def _make_metadata(name: str) -> RunMetadata:
    return RunMetadata(
        job_name=name,
        start_time=0.0,
        end_time=1.0,
        sample_interval_ms=100,
        snapshot_interval_steps=1,
        python_version="3.11",
        torch_version="2.x",
        cuda_available=False,
        cuda_device_name=None,
        hostname="host",
        platform="linux",
        pid=123,
    )


def _make_run(name: str, step_ms: float) -> RunData:
    steps = [
        StepRecord(
            step=0,
            start_time=0.0,
            end_time=1.0,
            duration_ms=step_ms,
            batch_size=32,
        )
    ]
    samples = [
        MetricSample(
            timestamp=0.0,
            cpu_percent=10.0,
            rss_mb=200.0,
            gpu_utilization=None,
            gpu_mem_used_mb=None,
            gpu_mem_total_mb=None,
            torch_cuda_allocated_mb=None,
            torch_cuda_reserved_mb=None,
        )
    ]
    return RunData(
        metadata=_make_metadata(name),
        steps=steps,
        samples=samples,
        snapshots=[],
        events=[],
    )


def test_compare_runs(tmp_path):
    run_a = _make_run("a", 100.0)
    run_b = _make_run("b", 50.0)

    path_a = tmp_path / "run_a.json"
    path_b = tmp_path / "run_b.json"
    path_a.write_text(json.dumps(run_a.to_dict()), encoding="utf-8")
    path_b.write_text(json.dumps(run_b.to_dict()), encoding="utf-8")

    result = compare_runs(str(path_a), str(path_b))
    metrics = {metric.name: metric for metric in result.metrics}

    assert "avg_step_ms" in metrics
    assert metrics["avg_step_ms"].delta is not None
    assert metrics["avg_step_ms"].delta < 0
