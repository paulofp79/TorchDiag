from __future__ import annotations

import platform
import socket
import sys
from typing import Optional

import psutil

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore

import torch

from .models import MetricSample, RunMetadata


class MetricCollector:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(None)
        self._nvml_handle = None
        self._nvml_available = False
        self._gpu_mem_total_mb: Optional[float] = None

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_available = True
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                self._gpu_mem_total_mb = mem_info.total / 1024**2
            except Exception:
                self._nvml_available = False
                self._nvml_handle = None

        self._torch_cuda_available = torch.cuda.is_available()

    def collect(self, timestamp: float) -> MetricSample:
        cpu_percent = self._process.cpu_percent(None)
        rss_mb = self._process.memory_info().rss / 1024**2

        gpu_util = None
        gpu_mem_used_mb = None
        gpu_mem_total_mb = self._gpu_mem_total_mb
        if self._nvml_available and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = float(util.gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_mem_used_mb = mem_info.used / 1024**2
                gpu_mem_total_mb = mem_info.total / 1024**2
            except Exception:
                gpu_util = None
                gpu_mem_used_mb = None

        torch_allocated = None
        torch_reserved = None
        if self._torch_cuda_available:
            torch_allocated = torch.cuda.memory_allocated() / 1024**2
            torch_reserved = torch.cuda.memory_reserved() / 1024**2

        return MetricSample(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            rss_mb=rss_mb,
            gpu_utilization=gpu_util,
            gpu_mem_used_mb=gpu_mem_used_mb,
            gpu_mem_total_mb=gpu_mem_total_mb,
            torch_cuda_allocated_mb=torch_allocated,
            torch_cuda_reserved_mb=torch_reserved,
        )


def build_metadata(
    job_name: str,
    start_time: float,
    sample_interval_ms: int,
    snapshot_interval_steps: int,
) -> RunMetadata:
    cuda_available = torch.cuda.is_available()
    device_name = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = None

    return RunMetadata(
        job_name=job_name,
        start_time=start_time,
        end_time=None,
        sample_interval_ms=sample_interval_ms,
        snapshot_interval_steps=snapshot_interval_steps,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_name=device_name,
        hostname=socket.gethostname(),
        platform=platform.platform(),
        pid=psutil.Process().pid,
    )
