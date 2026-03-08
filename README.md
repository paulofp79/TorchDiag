# torchdiag

PyTorch Performance Diagnostic Toolkit inspired by Oracle AWR/ASH-style diagnostics, but built for PyTorch training workloads.

`torchdiag` helps answer:
- Where training time is going
- Where GPU/CPU/memory pressure is happening
- Whether the bottleneck is compute, dataloader, memory, communication, or checkpoint I/O
- How two runs compare

## Why it exists
PyTorch’s flexible training loops make it easy to build models, but hard to consistently understand performance. `torchdiag` provides a lightweight, always-on diagnostic layer with a structured report, actionable findings, and run comparison—all without heavy profiler overhead.

## MVP scope
**Supported today**
- Linux + NVIDIA GPU + CUDA + single-process / single-GPU training
- CPU-only mode with graceful degradation
- Rule-based diagnosis
- JSON + HTML reports

**Not in v1**
- Distributed training
- Real-time dashboard server
- Prometheus/OpenTelemetry
- Heavy `torch.profiler` dependencies

## Installation
```bash
pip install -e .[dev]
```

## Quickstart
```python
from torchdiag import TorchDiag

diag = TorchDiag(
    job_name="mnist_baseline",
    sample_interval_ms=200,
    snapshot_interval_steps=10,
)

diag.start()

for step, batch in enumerate(train_loader):
    with diag.step_context(step=step, batch_size=len(batch[0])):
        with diag.event("forward"):
            outputs = model(batch[0])
            loss = criterion(outputs, batch[1])
        with diag.event("backward"):
            loss.backward()
        with diag.event("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

diag.stop()
diag.write_json("artifacts/run.json")
diag.write_html("artifacts/report.html")
```

To capture dataloader wait time, wrap the fetch:
```python
for step in range(steps):
    with diag.event("dataloader_wait", step=step):
        batch = next(loader)
    with diag.step_context(step=step, batch_size=len(batch[0])):
        ...
```

## Findings (rule-based)
Each report includes findings with severity, confidence, evidence, and concrete recommendations. Current rules:
- `input_pipeline_starvation`
- `gpu_underutilization`
- `memory_pressure`
- `unstable_step_times`

## Run comparison
Compare two runs with:
```python
from torchdiag import compare_runs

result = compare_runs("artifacts/run_a.json", "artifacts/run_b.json")
```

## Roadmap
- Distributed training support (DDP and multi-node)
- Communication-specific metrics and NCCL visibility
- Pluggable exporter interfaces (Prometheus, OTEL)
- Deeper integration with `torch.profiler` for opt-in tracing
- Baseline storage and regression alerts

## Examples
See `examples/basic_training_example.py` for a full runnable demo that produces:
- `artifacts/run.json`
- `artifacts/report.html`
