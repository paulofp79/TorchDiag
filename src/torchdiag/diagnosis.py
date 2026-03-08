from __future__ import annotations

import statistics
from typing import Dict, List, Optional

import psutil

from .models import DiagnosisFinding, RunData


def compute_summary(run: RunData) -> Dict[str, Optional[float]]:
    step_durations = [step.duration_ms for step in run.steps if step.duration_ms > 0]
    total_step_ms = sum(step_durations)

    batch_sizes = [step.batch_size for step in run.steps if step.batch_size]
    total_samples = sum(batch_sizes) if batch_sizes else None
    throughput = None
    if total_samples is not None and total_step_ms > 0:
        throughput = total_samples / (total_step_ms / 1000)

    dataloader_wait_ms = [
        step.dataloader_wait_ms
        for step in run.steps
        if step.dataloader_wait_ms is not None
    ]
    total_dataloader_wait_ms = sum(dataloader_wait_ms) if dataloader_wait_ms else 0.0
    dataloader_wait_share = (
        total_dataloader_wait_ms / total_step_ms if total_step_ms > 0 else None
    )

    gpu_utils = [
        sample.gpu_utilization
        for sample in run.samples
        if sample.gpu_utilization is not None
    ]
    avg_gpu_util = statistics.mean(gpu_utils) if gpu_utils else None

    gpu_mem_used = [
        sample.gpu_mem_used_mb
        for sample in run.samples
        if sample.gpu_mem_used_mb is not None
    ]
    peak_gpu_mem = max(gpu_mem_used) if gpu_mem_used else None

    rss_values = [sample.rss_mb for sample in run.samples if sample.rss_mb is not None]
    peak_rss = max(rss_values) if rss_values else None

    return {
        "avg_step_ms": statistics.mean(step_durations) if step_durations else None,
        "throughput": throughput,
        "dataloader_wait_share": dataloader_wait_share,
        "avg_gpu_util": avg_gpu_util,
        "peak_gpu_mem_mb": peak_gpu_mem,
        "peak_rss_mb": peak_rss,
    }


def diagnose(run: RunData) -> List[DiagnosisFinding]:
    findings: List[DiagnosisFinding] = []
    summary = compute_summary(run)

    if summary["dataloader_wait_share"] is not None and summary["dataloader_wait_share"] > 0.3:
        share = summary["dataloader_wait_share"]
        severity = "high" if share > 0.5 else "medium"
        findings.append(
            DiagnosisFinding(
                issue_type="input_pipeline_starvation",
                severity=severity,
                confidence=min(0.9, 0.5 + share),
                evidence=f"Dataloader wait accounts for {share:.0%} of step time.",
                recommendations=[
                    "Increase DataLoader workers or prefetch factor.",
                    "Move expensive transforms to GPU or offline preprocessing.",
                    "Ensure dataset storage has sufficient I/O bandwidth.",
                ],
            )
        )

    if summary["avg_gpu_util"] is not None and summary["avg_gpu_util"] < 40:
        util = summary["avg_gpu_util"]
        findings.append(
            DiagnosisFinding(
                issue_type="gpu_underutilization",
                severity="medium",
                confidence=0.6,
                evidence=f"Average GPU utilization is {util:.1f}%.",
                recommendations=[
                    "Increase batch size if memory allows.",
                    "Check dataloader and CPU bottlenecks.",
                    "Use mixed precision or fused ops where applicable.",
                ],
            )
        )

    gpu_pressure = _gpu_memory_pressure(run)
    cpu_pressure = _cpu_memory_pressure(run)
    if gpu_pressure or cpu_pressure:
        evidence = "Memory pressure detected."
        if gpu_pressure:
            evidence = f"GPU memory usage peaked at {gpu_pressure:.0%} of total."
        elif cpu_pressure:
            evidence = f"Process RSS peaked at {cpu_pressure:.0%} of system memory."
        findings.append(
            DiagnosisFinding(
                issue_type="memory_pressure",
                severity="high" if (gpu_pressure and gpu_pressure > 0.95) else "medium",
                confidence=0.7,
                evidence=evidence,
                recommendations=[
                    "Reduce batch size or enable gradient accumulation.",
                    "Use activation checkpointing for large models.",
                    "Monitor memory fragmentation and reserved vs allocated.",
                ],
            )
        )

    if _unstable_step_times(run):
        findings.append(
            DiagnosisFinding(
                issue_type="unstable_step_times",
                severity="medium",
                confidence=0.6,
                evidence="Step time variance is high across the run.",
                recommendations=[
                    "Check for dataloader variability or intermittent I/O.",
                    "Pin CPU threads and enable persistent workers.",
                    "Ensure no background checkpointing overlaps with steps.",
                ],
            )
        )

    return findings


def _gpu_memory_pressure(run: RunData) -> Optional[float]:
    ratios: List[float] = []
    for sample in run.samples:
        if sample.gpu_mem_used_mb is None or sample.gpu_mem_total_mb is None:
            continue
        if sample.gpu_mem_total_mb == 0:
            continue
        ratios.append(sample.gpu_mem_used_mb / sample.gpu_mem_total_mb)
    if not ratios:
        return None
    max_ratio = max(ratios)
    return max_ratio if max_ratio > 0.9 else None


def _cpu_memory_pressure(run: RunData) -> Optional[float]:
    if not run.samples:
        return None
    total_mem = psutil.virtual_memory().total / 1024**2
    peak_rss = max(sample.rss_mb for sample in run.samples if sample.rss_mb is not None)
    ratio = peak_rss / total_mem if total_mem > 0 else 0
    return ratio if ratio > 0.9 else None


def _unstable_step_times(run: RunData) -> bool:
    durations = [step.duration_ms for step in run.steps if step.duration_ms > 0]
    if len(durations) < 5:
        return False
    mean = statistics.mean(durations)
    if mean == 0:
        return False
    stdev = statistics.pstdev(durations)
    cv = stdev / mean
    median = statistics.median(durations)
    max_ratio = max(durations) / median if median else 0
    return cv > 0.2 or max_ratio > 1.5
