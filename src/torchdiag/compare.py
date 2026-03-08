from __future__ import annotations

import json
from typing import Optional

from .diagnosis import compute_summary
from .models import ComparisonMetric, ComparisonResult, RunData


def load_run(path: str) -> RunData:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return RunData.from_dict(data)


def compare_runs(run_a_path: str, run_b_path: str) -> ComparisonResult:
    run_a = load_run(run_a_path)
    run_b = load_run(run_b_path)
    return compare_run_data(run_a, run_b)


def compare_run_data(run_a: RunData, run_b: RunData) -> ComparisonResult:
    summary_a = compute_summary(run_a)
    summary_b = compute_summary(run_b)

    metrics = [
        _compare_metric("avg_step_ms", summary_a["avg_step_ms"], summary_b["avg_step_ms"]),
        _compare_metric("throughput", summary_a["throughput"], summary_b["throughput"]),
        _compare_metric("avg_gpu_util", summary_a["avg_gpu_util"], summary_b["avg_gpu_util"]),
        _compare_metric(
            "peak_gpu_mem_mb", summary_a["peak_gpu_mem_mb"], summary_b["peak_gpu_mem_mb"]
        ),
        _compare_metric(
            "dataloader_wait_share",
            summary_a["dataloader_wait_share"],
            summary_b["dataloader_wait_share"],
        ),
    ]

    diagnosis_a = {finding.issue_type for finding in run_a.findings}
    diagnosis_b = {finding.issue_type for finding in run_b.findings}

    diagnosis_diff = {
        "only_in_a": sorted(list(diagnosis_a - diagnosis_b)),
        "only_in_b": sorted(list(diagnosis_b - diagnosis_a)),
        "shared": sorted(list(diagnosis_a & diagnosis_b)),
    }

    return ComparisonResult(metrics=metrics, diagnosis_diff=diagnosis_diff)


def _compare_metric(
    name: str, value_a: Optional[float], value_b: Optional[float]
) -> ComparisonMetric:
    delta = None
    delta_pct = None
    if value_a is not None and value_b is not None:
        delta = value_b - value_a
        if value_a != 0:
            delta_pct = delta / value_a
    return ComparisonMetric(
        name=name,
        run_a=value_a,
        run_b=value_b,
        delta=delta,
        delta_pct=delta_pct,
    )
