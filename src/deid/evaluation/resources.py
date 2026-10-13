"""Resource measurement: disk footprint, peak RAM, initialisation time, latency percentiles."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil


def dir_size_bytes(path: str | Path, exts: Sequence[str] | None = None) -> int:
    """Size of all files below `path` (symlinks resolved, each target counted once)."""
    seen: set[str] = set()
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if exts and not f.endswith(tuple(exts)):
                continue
            real = os.path.realpath(os.path.join(root, f))
            if real in seen or not os.path.exists(real):
                continue
            seen.add(real)
            total += os.path.getsize(real)
    return total


class PeakMemory:
    """Samples this process's resident memory in a background thread (context manager)."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> "PeakMemory":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        mi = self.proc.memory_info()
        if hasattr(mi, "peak_wset"):          # Windows: OS-tracked peak working set
            self.peak = max(self.peak, mi.peak_wset)


def current_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def latency_summary(latencies_s: Sequence[float], chars: Sequence[int]) -> dict[str, Any]:
    if not latencies_s:
        return {"notes": 0}
    lat = np.asarray(latencies_s, dtype=float)
    total = float(lat.sum())
    return {
        "notes": int(len(lat)),
        "median_s": float(np.median(lat)),
        "p95_s": float(np.percentile(lat, 95)),
        "mean_s": float(lat.mean()),
        "max_s": float(lat.max()),
        "notes_per_s": float(len(lat) / total) if total else None,
        "chars_per_s": float(sum(chars) / total) if total else None,
        "mean_chars": float(np.mean(chars)) if chars else None,
    }


def machine_info() -> dict[str, Any]:
    import platform

    import torch
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
