"""Interpret one ``powermetrics`` sample document.

Only fields that Apple documents for the ``cpu_power``, ``gpu_power``, and
``tasks`` samplers are read. Power values are milliwatts. ``energy_impact`` is
an unlabelled relative score: it is stored as evidence and is never converted
into watt-hours.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Alternative spellings across powermetrics releases, all in milliwatts.
CPU_POWER_KEYS = ("cpu_power", "cpu_energy", "cpu_power_mw")
GPU_POWER_KEYS = ("gpu_power", "gpu_energy", "gpu_power_mw")
COMBINED_POWER_KEYS = ("combined_power", "combined_energy", "combined_power_mw")


@dataclass
class ProcessSample:
    pid: int
    name: str
    energy_impact: float | None = None
    cpu_seconds: float | None = None
    gpu_seconds: float | None = None


@dataclass
class PowerSample:
    at: float
    timestamp: datetime
    cpu_power_w: float | None = None
    gpu_power_w: float | None = None
    combined_power_w: float | None = None
    processes: list[ProcessSample] = field(default_factory=list)

    def power_w(self) -> float | None:
        """Prefer the reported combined value, then the CPU and GPU sum."""
        if self.combined_power_w is not None:
            return self.combined_power_w
        parts = [value for value in (self.cpu_power_w, self.gpu_power_w) if value is not None]
        return sum(parts) if parts else None

    def find(self, pid: int | None, name: str | None) -> ProcessSample | None:
        for process in self.processes:
            if pid is not None and process.pid == pid:
                return process
        if pid is None and name:
            for process in self.processes:
                if process.name == name:
                    return process
        return None


def _milliwatts(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value) / 1000.0)
    return None


def _seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value) / 1e9)
    return None


def parse_power_sample(document: dict[str, Any], at: float) -> PowerSample:
    """Return a sample; every field is optional so a partial document is usable."""
    processor = document.get("processor")
    processor = processor if isinstance(processor, dict) else {}
    sample = PowerSample(
        at=at,
        timestamp=_timestamp(document),
        cpu_power_w=_milliwatts(processor, CPU_POWER_KEYS),
        gpu_power_w=_milliwatts(processor, GPU_POWER_KEYS),
        combined_power_w=_milliwatts(processor, COMBINED_POWER_KEYS),
    )
    tasks = document.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            pid = task.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool):
                continue
            sample.processes.append(
                ProcessSample(
                    pid=pid,
                    name=str(task.get("name") or ""),
                    energy_impact=_number(task.get("energy_impact")),
                    cpu_seconds=_seconds(task.get("cputime_ns")),
                    gpu_seconds=_seconds(task.get("gputime_ns")),
                )
            )
    return sample


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _timestamp(document: dict[str, Any]) -> datetime:
    raw = document.get("timestamp")
    if isinstance(raw, str):
        for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S.%f %z"):
            try:
                return datetime.strptime(raw, fmt).astimezone(timezone.utc)
            except ValueError:
                continue
    return datetime.now(timezone.utc)
