from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .compare import ComparisonResult
from .diagnosis import compute_summary
from .models import EventType, RunData


def render_html(run: RunData, comparison: Optional[ComparisonResult] = None) -> str:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    context = build_report_context(run, comparison)
    return template.render(**context)


def write_html(path: str, run: RunData, comparison: Optional[ComparisonResult] = None) -> None:
    html = render_html(run, comparison)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def build_report_context(
    run: RunData, comparison: Optional[ComparisonResult]
) -> Dict[str, object]:
    summary = compute_summary(run)
    summary_display = {
        "avg_step_ms": _format_value(summary.get("avg_step_ms"), suffix=" ms"),
        "throughput": _format_value(summary.get("throughput"), suffix=" samples/s"),
        "avg_gpu_util": _format_value(summary.get("avg_gpu_util"), suffix=" %"),
        "dataloader_wait_share": _format_value(
            summary.get("dataloader_wait_share"),
            formatter="{:.0%}",
            suffix="",
        ),
    }
    step_numbers = [step.step for step in run.steps]
    step_durations = [step.duration_ms for step in run.steps]

    start_time = run.metadata.start_time
    sample_times = [sample.timestamp - start_time for sample in run.samples]
    cpu_series = [sample.cpu_percent for sample in run.samples]
    gpu_series = [sample.gpu_utilization for sample in run.samples]
    rss_series = [sample.rss_mb for sample in run.samples]
    gpu_mem_series = [sample.gpu_mem_used_mb for sample in run.samples]

    step_fig = go.Figure()
    step_fig.add_trace(go.Scatter(x=step_numbers, y=step_durations, name="step_ms"))
    step_fig.update_layout(title="Step Duration (ms)", xaxis_title="Step", yaxis_title="ms")

    util_fig = go.Figure()
    util_fig.add_trace(go.Scatter(x=sample_times, y=cpu_series, name="CPU %"))
    if any(value is not None for value in gpu_series):
        util_fig.add_trace(go.Scatter(x=sample_times, y=gpu_series, name="GPU %"))
    util_fig.update_layout(
        title="CPU/GPU Utilization", xaxis_title="Time (s)", yaxis_title="Percent"
    )

    mem_fig = go.Figure()
    mem_fig.add_trace(go.Scatter(x=sample_times, y=rss_series, name="RSS MB"))
    if any(value is not None for value in gpu_mem_series):
        mem_fig.add_trace(go.Scatter(x=sample_times, y=gpu_mem_series, name="GPU Mem MB"))
    mem_fig.update_layout(
        title="Memory Usage", xaxis_title="Time (s)", yaxis_title="MB"
    )

    event_breakdown = _event_breakdown(run)
    event_fig = go.Figure(
        data=[
            go.Pie(
                labels=list(event_breakdown.keys()),
                values=list(event_breakdown.values()),
            )
        ]
    )
    event_fig.update_layout(title="Event Breakdown (ms)")

    slowest_steps = sorted(run.steps, key=lambda step: step.duration_ms, reverse=True)[:5]

    return {
        "metadata": run.metadata,
        "summary": summary_display,
        "findings": run.findings,
        "step_chart": pio.to_json(step_fig),
        "util_chart": pio.to_json(util_fig),
        "mem_chart": pio.to_json(mem_fig),
        "event_chart": pio.to_json(event_fig),
        "event_breakdown": event_breakdown,
        "slowest_steps": slowest_steps,
        "comparison": comparison,
    }


def _event_breakdown(run: RunData) -> Dict[str, float]:
    totals: Dict[EventType, float] = {event: 0.0 for event in EventType}
    for step in run.steps:
        accounted = 0.0
        for event_type, duration in step.event_durations_ms.items():
            totals[event_type] = totals.get(event_type, 0.0) + duration
            accounted += duration
        unknown = max(0.0, step.duration_ms - accounted)
        totals[EventType.UNKNOWN] = totals.get(EventType.UNKNOWN, 0.0) + unknown

    breakdown = {event.value: total for event, total in totals.items() if total > 0}
    return breakdown


def _format_value(
    value: Optional[float],
    formatter: str = "{:.2f}",
    suffix: str = "",
) -> str:
    if value is None:
        return "n/a"
    return f"{formatter.format(value)}{suffix}"
