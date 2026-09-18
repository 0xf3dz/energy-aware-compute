"""Command line entry point: ``offgrid-inference <command>``."""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from getpass import getuser
from pathlib import Path

from api.settings import Settings
from contracts import Availability, EnergyState, Job, JobResult
from scheduler.models import LOW_PRIORITY

BENCHMARK_PROMPT = (
    "Explain in one sentence why a sailing vessel limits its compute load when the "
    "battery state of charge is low."
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    if getattr(arguments, "demo", False):
        settings = settings.model_copy(update={"demo": True})
    try:
        return asyncio.run(arguments.handler(arguments, settings))
    except KeyboardInterrupt:
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offgrid-inference", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    migrate = commands.add_parser("migrate", help="apply database migrations")
    migrate.set_defaults(handler=_migrate)

    serve = commands.add_parser("serve", help="run the dashboard and the scheduler")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--demo", action="store_true", help="use mock providers only")
    serve.add_argument("--dashboard-only", action="store_true", help="no scheduler loop")
    serve.set_defaults(handler=_serve)

    tick = commands.add_parser("tick", help="run scheduler ticks without a web server")
    tick.add_argument("--count", type=int, default=1)
    tick.add_argument("--demo", action="store_true")
    tick.add_argument("--step-seconds", type=float, default=0, help="advance a fake clock")
    tick.set_defaults(handler=_tick)

    enqueue = commands.add_parser("enqueue-briefing", help="queue a weather briefing job")
    enqueue.add_argument("--deferrable", action="store_true")
    enqueue.add_argument("--priority", type=int, default=None)
    enqueue.add_argument("--estimated-energy-wh", type=float, default=5)
    enqueue.add_argument("--demo", action="store_true")
    enqueue.set_defaults(handler=_enqueue)

    benchmark = commands.add_parser(
        "benchmark", help="measure tokens, runtime, and estimated energy per request"
    )
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.add_argument("--max-tokens", type=int, default=256)
    benchmark.add_argument("--json", type=Path, default=None, help="write the report here")
    benchmark.set_defaults(handler=_benchmark)

    calibrate = commands.add_parser("calibrate", help="measure the idle power baseline")
    calibrate.add_argument("--seconds", type=float, default=5)
    calibrate.set_defaults(handler=_calibrate)

    smoke = commands.add_parser("smoke", help="offline end-to-end check with mocks")
    smoke.set_defaults(handler=_smoke)

    return parser


async def _migrate(arguments, settings: Settings) -> int:
    from db.migrate import migrate_url

    await migrate_url(settings.database_url)
    print(f"migrations applied to {settings.database_url}")
    return 0


async def _serve(arguments, settings: Settings) -> int:
    import uvicorn

    from api.app import create_app

    updates = {
        "host": arguments.host or settings.host,
        "port": arguments.port or settings.port,
        "serve_scheduler": not arguments.dashboard_only and settings.serve_scheduler,
    }
    settings = settings.model_copy(update=updates)
    print(f"dashboard on http://{settings.host}:{settings.port}/  (demo={settings.demo})")
    config = uvicorn.Config(create_app(settings), host=settings.host, port=settings.port,
                           log_level="info")
    await uvicorn.Server(config).serve()
    return 0


class FakeClock:
    """Deterministic clock for demonstrations and tests."""

    def __init__(self, start: datetime, step_seconds: float = 60) -> None:
        self.now = start
        self.step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, steps: int = 1) -> datetime:
        self.now = self.now + self.step * steps
        return self.now


async def _tick(arguments, settings: Settings) -> int:
    from api.runtime import build_runtime

    clock = (
        FakeClock(datetime.now(UTC), arguments.step_seconds)
        if arguments.step_seconds
        else None
    )
    runtime = await build_runtime(settings, clock=clock)
    try:
        for index in range(arguments.count):
            if clock is not None:
                clock.advance()
            decision = await runtime.scheduler.tick()
            if decision is None:
                print(f"tick {index + 1}: no pending job")
            else:
                print(f"tick {index + 1}: {decision.decision} {decision.job_id} :: {decision.reason}")
    finally:
        await runtime.close()
    return 0


async def _enqueue(arguments, settings: Settings) -> int:
    from api.runtime import build_runtime

    runtime = await build_runtime(settings)
    try:
        job = await runtime.enqueue_briefing(
            deferrable=arguments.deferrable,
            priority=arguments.priority if arguments.priority is not None
            else (LOW_PRIORITY if arguments.deferrable else 80),
            estimated_energy_wh=arguments.estimated_energy_wh,
        )
        print(json.dumps(job.model_dump(mode="json"), indent=1))
    finally:
        await runtime.close()
    return 0


async def _benchmark(arguments, settings: Settings) -> int:
    """Run controlled inference jobs and record energy next to token counts."""
    from api.runtime import build_runtime
    from contracts import GenerationRequest

    runtime = await build_runtime(settings)
    rows: list[dict] = []
    try:
        for index in range(arguments.runs):
            job = Job(
                workload="inference_benchmark",
                priority=100,
                deferrable=False,
                payload={"estimated_runtime_seconds": 60, "run": index + 1},
            )
            job = await runtime.queue.enqueue(job)
            claimed = await runtime.queue.claim(job.id)
            if claimed is None:
                print("another job is running; stop it before benchmarking", file=sys.stderr)
                return 1
            await runtime.monitor.start(claimed.id)
            error = None
            result = None
            try:
                response = await runtime.inference.generate(
                    GenerationRequest(
                        system="You are a concise technical assistant.",
                        prompt=BENCHMARK_PROMPT,
                        max_tokens=arguments.max_tokens,
                        temperature=0.2,
                    )
                )
                result = JobResult(data={"run": index + 1}, inference_metrics=[response.metrics])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            await runtime.monitor.stop(claimed.id)
            estimate = await runtime.monitor.estimate(claimed.id)
            await runtime.queue.finish(claimed, result, estimate, error=error)
            if error:
                print(f"run {index + 1} failed: {error}", file=sys.stderr)
                continue
            metrics = result.inference_metrics[0]
            rows.append(
                {
                    "run": index + 1,
                    "job_id": claimed.id,
                    "model": metrics.model,
                    "prompt_tokens": metrics.prompt_tokens,
                    "generated_tokens": metrics.generated_tokens,
                    "runtime_seconds": metrics.runtime_seconds,
                    "generation_tokens_per_second": metrics.generation_tokens_per_second,
                    "prompt_tokens_per_second": metrics.prompt_tokens_per_second,
                    "estimated_wh": estimate.estimated_wh,
                    "joules_per_generated_token": estimate.joules_per_generated_token(
                        metrics.generated_tokens
                    ),
                    "measurement_method": estimate.measurement_method,
                    "confidence": estimate.confidence,
                    "average_power_w": estimate.average_power_w,
                    "energy_reason": estimate.reason,
                }
            )
    finally:
        await runtime.close()
    report = {"generated_at": datetime.now(UTC).isoformat(), "runs": rows,
              "summary": _summarize(rows)}
    _print_report(report)
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=1))
        print(f"\nreport written to {arguments.json}")
    return 0


def _summarize(rows: list[dict]) -> dict:
    # Only measured runs enter the ratio, so the tokens match the energy.
    measured = [row for row in rows if row["estimated_wh"] is not None]
    generated = sum(row["generated_tokens"] or 0 for row in rows)
    runtime = sum(row["runtime_seconds"] or 0 for row in rows)
    energy = sum(row["estimated_wh"] for row in measured) if measured else None
    measured_tokens = sum(row["generated_tokens"] or 0 for row in measured)
    return {
        "requests": len(rows),
        "measured_requests": len(measured),
        "generated_tokens": generated,
        "runtime_seconds": round(runtime, 3),
        "generation_tokens_per_second": round(generated / runtime, 2) if runtime else None,
        "estimated_wh": round(energy, 6) if energy is not None else None,
        "estimated_wh_per_million_tokens": round(energy / measured_tokens * 1e6, 4)
        if energy is not None and measured_tokens
        else None,
        "joules_per_generated_token": round(energy * 3600 / measured_tokens, 3)
        if energy is not None and measured_tokens
        else None,
        "measurement_method": rows[0]["measurement_method"] if rows else None,
        "confidence": rows[0]["confidence"] if rows else None,
    }


def _current_user() -> str:
    """The account name for a sudoers rule. Never a host name."""
    return getuser() or "your-user"


def _print_report(report: dict) -> None:
    print("run  prompt_tok  gen_tok  runtime_s  gen_tok/s  est_Wh  J/tok")
    for row in report["runs"]:
        energy = row["estimated_wh"]
        energy_text = "n/a" if energy is None else f"{energy:.4f}"
        joules = row.get("joules_per_generated_token")
        joules_text = "n/a" if joules is None else f"{joules:.2f}"
        rate = row["generation_tokens_per_second"] or 0
        print(
            f"{row['run']:>3}  {row['prompt_tokens']!s:>10}  {row['generated_tokens']!s:>7}  "
            f"{row['runtime_seconds']:>9.2f}  {rate:>9.2f}  {energy_text:>7}  {joules_text:>6}"
        )
    print()
    print(json.dumps(report["summary"], indent=1))
    for row in report["runs"]:
        if row["estimated_wh"] is None:
            print(f"run {row['run']} has no energy estimate: {row['energy_reason']}")
    if report["runs"] and report["runs"][0]["estimated_wh"] is not None:
        print(f"energy reason: {report['runs'][0]['energy_reason']}")
    if report["summary"].get("joules_per_generated_token") is not None:
        print(
            "\nJ/tok boundary: SoC rails only (CPU, GPU, ANE), idle baseline subtracted, "
            "whole request divided by generated tokens.\nDRAM, storage, VRM losses, and "
            "the display are outside the measurement, so J/tok is a lower bound on whole "
            "machine energy."
        )


async def _calibrate(arguments, settings: Settings) -> int:
    from energy.mac_power import PowermetricsProvider

    provider = PowermetricsProvider(
        llama_pid=settings.llama_pid,
        sample_interval_ms=1000,
        max_duration_seconds=arguments.seconds,
    )
    estimate = await provider.calibrate_idle(seconds=arguments.seconds)
    print(json.dumps(estimate.model_dump(mode="json", exclude={"samples"}), indent=1))
    if estimate.average_power_w is None:
        print(
            "\nNo baseline was measured. powermetrics needs root access. Add a sudoers rule such as:\n"
            f"  {_current_user()}  ALL=(root) NOPASSWD: /usr/bin/powermetrics\n",
            file=sys.stderr,
        )
        return 1
    print(f"\nSet IDLE_BASELINE_W={estimate.average_power_w:.1f} in .env")
    return 0


async def _smoke(arguments, settings: Settings) -> int:
    """Offline demonstration: deferral on low energy, then a solar-surplus run."""
    from api.runtime import build_runtime

    settings = settings.model_copy(update={"demo": True, "serve_scheduler": False})
    clock = FakeClock(datetime.now(UTC).replace(microsecond=0))
    runtime = await build_runtime(settings, clock=clock)
    failures = 0
    try:
        job = await runtime.enqueue_briefing(
            deferrable=True, priority=LOW_PRIORITY, scheduled_at=clock()
        )
        print(f"queued deferrable job {job.id}")

        # 1. A constrained battery must defer the job.
        runtime.energy.state = EnergyState(
            timestamp=clock(),
            availability=Availability.FRESH,
            battery_soc=37,
            solar_power_w=120,
            ac_load_w=200,
        )
        decision = await runtime.scheduler.tick()
        print(f"[low energy]   {decision.decision}: {decision.reason}")
        failures += 0 if decision.decision == "DEFER" else 1

        # 2. A solar surplus must hold for the configured period before the run.
        runtime.energy.state = None
        for _ in range(8):
            clock.advance()
            decision = await runtime.scheduler.tick()
            print(f"[surplus wait] {decision.decision}: {decision.reason}")
            if decision.decision == "RUN":
                break
        else:
            failures += 1

        snapshot = await runtime.snapshot()
        recent = snapshot["jobs"]["recent"]
        print(f"completed jobs: {[(item['workload'], item['status']) for item in recent]}")
        print(f"today: {snapshot['today']}")
        print(f"decisions recorded: {len(snapshot['decisions'])}")
        if not recent or recent[0]["status"] != "COMPLETED":
            failures += 1
        if not snapshot["decisions"]:
            failures += 1
    finally:
        await runtime.close()
    print("smoke PASS" if failures == 0 else f"smoke FAIL ({failures} check(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
