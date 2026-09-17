"""Estimated compute energy from macOS ``powermetrics`` samples.

The provider launches one ``powermetrics`` process for the lifetime of a job,
integrates the sampled SoC power over time, and subtracts an operator
calibrated idle baseline. Every result states ``measurement_method`` and
``confidence`` so that nobody mistakes it for metered AC consumption.

Two conditions must hold before a number is reported:

* ``powermetrics`` needs root. The provider calls ``sudo -n`` so that a missing
  privilege becomes an immediate, recorded failure instead of a blocked prompt.
* The attributed process must appear in the per-process samples. Otherwise the
  window shows system power that cannot be tied to inference.

A future ``SmartPlugProvider`` or ``VictronMeterProvider`` implements the same
interface; the rest of the application does not change.
"""

import asyncio
import contextlib
import logging
import shutil
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

from contracts import ComputeEnergyMonitor, EnergyEstimate
from energy.mac_power.plist import extract_documents
from energy.mac_power.samples import PowerSample, ProcessSample, parse_power_sample

logger = logging.getLogger(__name__)

DEFAULT_SAMPLERS = "cpu_power,gpu_power,tasks"
CAVEAT = "Estimate from sampled SoC power; not metered AC consumption"


def attribute(
    samples: list[PowerSample],
    *,
    idle_baseline_w: float,
    max_gap_seconds: float,
) -> tuple[float, float, float | None, int, int]:
    """Integrate sampled power. Return energy, covered seconds, average, pairs, gaps."""
    energy_wh = 0.0
    covered = 0.0
    gapped = 0
    pairs = 0
    previous: PowerSample | None = None
    for sample in samples:
        power = sample.power_w()
        previous_power = previous.power_w() if previous is not None else None
        if previous is not None:
            gap = sample.at - previous.at
            if power is None or previous_power is None or gap <= 0 or gap > max_gap_seconds:
                gapped += 1
            else:
                energy_wh += (power + previous_power) / 2 * gap / 3600.0
                covered += gap
                pairs += 1
        previous = sample
    if covered <= 0:
        return 0.0, 0.0, None, 0, gapped
    average_power_w = energy_wh * 3600.0 / covered
    attributable = energy_wh - idle_baseline_w * covered / 3600.0
    return max(0.0, attributable), covered, average_power_w, pairs, gapped


class PowermetricsProvider(ComputeEnergyMonitor):
    """Measure estimated SoC energy for one running job at a time."""

    def __init__(
        self,
        *,
        llama_pid: int | None = None,
        process_name: str = "llama-server",
        idle_baseline_w: float | None = None,
        sample_interval_ms: int = 1000,
        max_duration_seconds: float = 3600,
        cleanup_timeout_seconds: float = 3,
        require_process: bool = True,
        executable: str = "/usr/bin/powermetrics",
        samplers: str = DEFAULT_SAMPLERS,
        process_factory: Callable[..., Any] = asyncio.create_subprocess_exec,
        platform: str = sys.platform,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("sample_interval_ms", sample_interval_ms),
            ("max_duration_seconds", max_duration_seconds),
            ("cleanup_timeout_seconds", cleanup_timeout_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if idle_baseline_w is not None and idle_baseline_w < 0:
            raise ValueError("idle_baseline_w must not be negative")
        self.llama_pid = llama_pid
        self.process_name = process_name
        self.idle_baseline_w = idle_baseline_w
        self.sample_interval_ms = sample_interval_ms
        self.max_duration_seconds = max_duration_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.require_process = require_process
        self.executable = executable
        self.samplers = samplers
        self.process_factory = process_factory
        self.platform = platform
        self.clock = clock
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._sudo = shutil.which("sudo") or "/usr/bin/sudo"

    async def start(self, job_id: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"Energy measurement already active for job {job_id}")
            record: dict[str, Any] = {
                "samples": [],
                "started_at": self.clock(),
                "reason": None,
                "stopped": False,
                "estimate": None,
                "observed": None,
                "handle": None,
                "readers": [],
                "watchdog": None,
            }
            self._jobs[job_id] = record
        if self.platform != "darwin":
            record["reason"] = f"powermetrics requires macOS; platform is {self.platform}"
            return
        argv = [
            self._sudo, "-n", self.executable,
            "--format", "plist",
            "--buffer-size", "0",
            "--sample-rate", str(self.sample_interval_ms),
            "--samplers", self.samplers,
            "--order", "cputime",
            "--show-process-energy",
            "-n", "-1",
        ]
        try:
            handle = await self.process_factory(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            record["reason"] = f"powermetrics could not start: {exc}"
            return
        record["handle"] = handle
        record["readers"] = [
            asyncio.create_task(self._read_stdout(record, handle)),
            asyncio.create_task(self._read_stderr(record, handle)),
        ]
        record["watchdog"] = asyncio.create_task(self._enforce_limit(record, handle))

    async def _read_stdout(self, record: dict[str, Any], handle: Any) -> None:
        buffer = b""
        try:
            while True:
                chunk = await handle.stdout.read(65536)
                if not chunk:
                    return
                buffer += chunk
                documents, buffer = extract_documents(buffer)
                for document in documents:
                    sample = parse_power_sample(document, self.clock())
                    record["samples"].append(sample)
                    observed = sample.find(self.llama_pid, self.process_name)
                    if observed is not None:
                        record["observed"] = observed
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record["reason"] = record["reason"] or f"Sample stream failed: {type(exc).__name__}"

    async def _read_stderr(self, record: dict[str, Any], handle: Any) -> None:
        try:
            while True:
                chunk = await handle.stderr.read(4096)
                if not chunk:
                    return
                text = chunk.decode("utf-8", "replace").strip()
                if text:
                    record["stderr"] = (record.get("stderr", "") + " " + text).strip()[:500]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("powermetrics stderr reader stopped", exc_info=True)

    async def _enforce_limit(self, record: dict[str, Any], handle: Any) -> None:
        try:
            await asyncio.sleep(self.max_duration_seconds)
        except asyncio.CancelledError:
            return
        record["reason"] = record["reason"] or (
            f"Measurement window reached the {self.max_duration_seconds:g} s limit"
        )
        await self._signal(handle, signal.SIGINT)

    async def stop(self, job_id: str) -> None:
        record = self._jobs.get(job_id)
        if record is None or record["stopped"]:
            return
        record["stopped"] = True
        watchdog = record.get("watchdog")
        if watchdog is not None:
            watchdog.cancel()
            with contextlib.suppress(BaseException):
                await watchdog
        handle = record.get("handle")
        if handle is not None and handle.returncode is None:
            await self._signal(handle, signal.SIGINT)
            if not await self._wait(handle):
                await self._signal(handle, signal.SIGTERM)
                if not await self._wait(handle):
                    handle.kill()
                    with contextlib.suppress(BaseException):
                        await handle.wait()
                    record["reason"] = record["reason"] or "powermetrics needed a hard kill"
        # Let the readers reach end of stream so that final output, including a
        # sudo or powermetrics error message, is recorded before cancellation.
        readers = record["readers"]
        if readers:
            _, pending = await asyncio.wait(readers, timeout=2)
            for reader in pending:
                reader.cancel()
        for reader in readers:
            with contextlib.suppress(BaseException):
                await reader
        if handle is not None and handle.returncode not in (0, None):
            detail = record.get("stderr") or f"exit status {handle.returncode}"
            record["reason"] = record["reason"] or (
                f"powermetrics exited early ({handle.returncode}): {detail}"
            )

    async def _wait(self, handle: Any) -> bool:
        try:
            await asyncio.wait_for(handle.wait(), self.cleanup_timeout_seconds)
            return True
        except TimeoutError:
            return False
        except ProcessLookupError:
            return True

    async def _signal(self, handle: Any, number: int) -> None:
        try:
            handle.send_signal(number)
        except (ProcessLookupError, PermissionError, ValueError):
            logger.debug("Cannot signal powermetrics with %s", number, exc_info=True)

    async def estimate(self, job_id: str) -> EnergyEstimate:
        record = self._jobs.get(job_id)
        if record is None:
            return EnergyEstimate(
                measurement_method="powermetrics",
                confidence="estimated",
                reason="No measurement was started for this job",
            )
        if record.get("estimate") is not None:
            return record["estimate"]
        estimate = self._build(record)
        record["estimate"] = estimate
        return estimate

    def _build(self, record: dict[str, Any]) -> EnergyEstimate:
        samples: list[PowerSample] = record["samples"]
        observed: ProcessSample | None = record.get("observed")
        base = EnergyEstimate(
            measurement_method="powermetrics",
            confidence="estimated",
            samples=[self._evidence(sample) for sample in samples],
        )
        elapsed = max(0.0, self.clock() - record["started_at"])
        if not samples:
            base.runtime_seconds = elapsed
            base.reason = self._failure_reason(record)
            return base
        if self.require_process and observed is None:
            base.runtime_seconds = elapsed
            base.reason = (
                f"No powermetrics task sample matched PID {self.llama_pid} or name "
                f"{self.process_name!r} in {len(samples)} samples, so system power is not "
                f"attributed to inference. {CAVEAT}"
            )
            return base
        if self.idle_baseline_w is None:
            base.runtime_seconds = elapsed
            base.reason = (
                "Idle baseline is not calibrated, so sampled system power cannot be attributed "
                f"to compute. Run 'energy-compute calibrate'. {CAVEAT}"
            )
            return base
        valued = [sample for sample in samples if sample.power_w() is not None]
        max_gap = max(2.0, 3 * self.sample_interval_ms / 1000.0)
        energy_wh, covered, average_w, pairs, gapped = attribute(
            valued, idle_baseline_w=self.idle_baseline_w, max_gap_seconds=max_gap
        )
        notes = [CAVEAT]
        if average_w is None:
            notes.append(f"No usable power value in {len(samples)} sample(s)")
        else:
            notes.append(
                f"{pairs} interval(s) of {len(samples)} sample(s) covered {covered:.1f} s "
                f"at an average {average_w:.1f} W"
            )
        notes.append(f"Subtracted a {self.idle_baseline_w:.1f} W idle baseline")
        if gapped:
            notes.append(f"{gapped} gap(s) or invalid sample(s) were excluded")
        if observed is not None:
            notes.append(
                f"Attributed process {observed.name} (PID {observed.pid}), "
                f"{observed.cpu_seconds or 0:.2f} CPU s, last energy impact "
                f"{observed.energy_impact if observed.energy_impact is not None else 'n/a'}"
            )
        if record.get("reason"):
            notes.append(record["reason"])
        base.estimated_wh = round(energy_wh, 6) if covered > 0 else None
        base.runtime_seconds = covered
        base.average_power_w = average_w
        base.reason = "; ".join(notes)
        return base

    def _failure_reason(self, record: dict[str, Any]) -> str:
        if record.get("reason"):
            return record["reason"]
        detail = record.get("stderr")
        if detail:
            return f"powermetrics produced no samples: {detail}"
        elapsed = self.clock() - record["started_at"]
        return (
            f"powermetrics produced no samples over {elapsed:.1f} s. It needs root access, for "
            f"example a sudoers entry for {self.executable}"
        )

    async def calibrate_idle(self, seconds: float = 5) -> EnergyEstimate:
        """Sample an idle machine to obtain the baseline for later subtraction.

        Run this only while the machine hosts no other work. The result reports
        the idle power of the whole SoC; the operator stores it as the baseline.
        """
        if seconds <= 0:
            raise ValueError("calibration seconds must be positive")
        provider = PowermetricsProvider(
            process_name=self.process_name,
            idle_baseline_w=0.0,
            sample_interval_ms=self.sample_interval_ms,
            max_duration_seconds=seconds,
            cleanup_timeout_seconds=self.cleanup_timeout_seconds,
            require_process=False,
            executable=self.executable,
            samplers=self.samplers,
            process_factory=self.process_factory,
            platform=self.platform,
            clock=self.clock,
        )
        await provider.start("calibration")
        try:
            await asyncio.sleep(seconds)
        finally:
            await provider.stop("calibration")
        return await provider.estimate("calibration")

    def _evidence(self, sample: PowerSample) -> dict[str, Any]:
        observed = sample.find(self.llama_pid, self.process_name)
        return {
            "timestamp": sample.timestamp.isoformat(),
            "cpu_power_w": sample.cpu_power_w,
            "gpu_power_w": sample.gpu_power_w,
            "combined_power_w": sample.combined_power_w,
            "llama_pid": observed.pid if observed else None,
            "llama_cpu_seconds": observed.cpu_seconds if observed else None,
            "llama_gpu_seconds": observed.gpu_seconds if observed else None,
            "llama_energy_impact": observed.energy_impact if observed else None,
        }
