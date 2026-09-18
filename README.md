# What is this?

A local compute scheduler for a solar powered sailing vessel. The system runs
local Qwen inference on a 32GB M2 Max Studio, measures the energy that the inference
uses, and moves deferrable work into the hours with a solar surplus.

The system schedules compute. It does not replace the Victron VRM dashboard.
Use VRM for electrical analysis. The dashboard of this project answers one
question: which compute does the machine run, and why?

# For what purpose?

This repository is a prototype for an offgrid local inference lab aboard my sailboat, with solar energy as its only power
source. The goal is to use local AI within the limits of the boat’s energy supply.

A boat has a limited energy budget. Navigation equipment and other vessel systems take priority over optional
compute. The scheduler uses battery charge and available solar power to decide whether a task can run or must wait.

The first workload is a daily marine weather briefing. The system downloads a forecast, calculates trends and
warnings, then asks local Qwen to produce a readable report. Future workloads could include document search,
maintenance-log analysis, and passage preparation.

The prototype will help answer these questions:

- How much estimated energy does each inference task use?
- Which workloads can the Mac Studio complete within the available energy budget?
- When can surplus solar power support optional compute?
- Which tasks must wait to protect the battery reserve?

Local inference keeps model execution aboard the boat and avoids dependence on a cloud AI service. Fresh weather
forecasts and VRM data still need internet access. But these can be downloaded in weekly batches, assuming internet access 
remains available at least once a week.

The aim is to establish a measured solution for a useful onboard AI system.
 
## Architecture

```text
        Victron VRM            Weather API
             |                      |
             v                      v
      energy/vrm             workloads/weather
             |                      |
             v                      |
      EnergyState                   |
             |                      |
             +----------+-----------+
                        v
                   scheduler  (policy: SOC, surplus, deadline, priority)
                        |
                        v
                    jobs/queue  (PostgreSQL)
                        |
                        v
                 workloads (weather_briefing, future services)
                        |
                        v
                 inference (llama.cpp, OpenAI compatible)
                        |
                        v
                      Qwen

                 powermetrics --> energy/mac_power --> EnergyEstimate
```

One rule holds everywhere: a workload depends on interfaces only. A workload
never calls VRM, `powermetrics`, llama.cpp, or another workload.

## Module map

| Path | Responsibility |
| --- | --- |
| `src/contracts.py` | Typed models and provider protocols. The shared interface. |
| `src/inference/` | `LlamaCppProvider` and `MockInferenceProvider`. |
| `src/energy/vrm/` | VRM adapter. Normalizes responses into `EnergyState`. |
| `src/energy/mac_power/` | `powermetrics` sampling and energy attribution. |
| `src/scheduler/` | Policy, engine, vocabulary. |
| `src/jobs/` | Worker and the in-memory queue double. |
| `src/db/` | PostgreSQL queue, cache, migrations, dashboard read model. |
| `src/workloads/` | Pluggable workloads. The weather briefing is the first. |
| `src/api/` | Settings, composition root, FastAPI dashboard, CLI. |

## Install

```bash
scripts/setup.sh
cp .env.example .env        # then fill in the values
docker compose up -d postgres
.venv/bin/python -m db.migrate
```

The database is optional for a demonstration. Set `DEMO=1` to run with mock
providers and an in-memory queue.

## Configure

Put the VRM access token, the installation identifier, and the forecast
position in `.env`. The file `.env.example` lists every setting.

A VRM token comes from the VRM portal under Preferences, Access tokens.
Deferrable work needs `LATITUDE` and `LONGITUDE` for the weather briefing.

## Run

```bash
.venv/bin/python -m api.cli serve --host 0.0.0.0 --port 8090   # dashboard + scheduler
.venv/bin/python -m api.cli tick --count 5                     # scheduler only
.venv/bin/python -m api.cli enqueue-briefing --deferrable --priority 20
.venv/bin/python -m api.cli benchmark --runs 5 --max-tokens 256
.venv/bin/python -m api.cli calibrate --seconds 10
.venv/bin/python -m api.cli smoke                              # offline check
scripts/start-inference.sh                                     # llama-server
```

The dashboard listens on port 8090. A LAN bind needs `DASHBOARD_TOKEN`, and the
token is then read from the query string or from a bearer header.

## Inference

`LlamaCppProvider` speaks the OpenAI chat API of `llama-server`. It reads the
per-request `timings` block for token counts and rates, and `GET /metrics` for
the server counters and gauges. Start the server with `--metrics`.

Telemetry that the system collects:

| Metric | Source |
| --- | --- |
| prompt tokens, generated tokens | `timings` of the request |
| prompt tok/s, generation tok/s | `timings` of the request |
| active requests | `llamacpp:requests_processing` |
| KV cache usage | `llamacpp:kv_cache_usage_ratio` |

The provider reports server counters as differences between two calls, so the
window is explicit in `interval_seconds`.

## Compute energy

`PowermetricsProvider` measures the estimated SoC power of the Mac, integrates
the samples over the runtime of a job, and subtracts a calibrated idle
baseline. The result carries three explicit fields:

```text
estimated_wh        the estimate, or null
measurement_method  "powermetrics"
confidence          "estimated"
```

The estimate is not metered AC consumption. `powermetrics` reports estimated
power. The reason string states the method, the sample count, the covered
seconds, the baseline, and the caveats.

The provider reports a number only when both conditions hold:

1. `powermetrics` runs. It needs root. The provider calls `sudo -n`, so a
   missing privilege becomes an immediate, recorded failure.
2. The `llama-server` process appears in the per-process samples.

Add a `sudoers` rule to allow the sampler without a password:

```text
f3dz ALL=(root) NOPASSWD: /usr/bin/powermetrics
```

Run `api.cli calibrate` on an idle machine and put the measured power in
`IDLE_BASELINE_W`. Without a baseline the estimate stays `null` and the reason
names the missing step.

A future `SmartPlugProvider` or `VictronMeterProvider` implements the same
`ComputeEnergyMonitor` interface. The rest of the system does not change.

### Energy per generated token

The estimate gives joules per generated token (J/tok) directly:

```text
J/tok = estimated_wh * 3600 / generated_tokens
```

This is the same relation as watts divided by tokens per second. The two forms
agree because the provider integrates the samples over time, so a changing load
needs no constant-power assumption.

The boundary of the number is the measured rails. `powermetrics` reports the
CPU, the GPU, and the ANE. Apple silicon has no DRAM rail, so DRAM, storage,
VRM losses, and the display stay outside the measurement. J/tok is therefore a
lower bound on whole machine energy. Compare a J/tok value only with a value
from the same boundary.

Three more properties of the number:

- The estimate is idle subtracted. J/tok is the marginal energy of the work,
  not the total draw of the machine.
- The numerator covers the whole request, prompt processing included. The
  denominator counts generated tokens only. A long prompt raises J/tok.
- The provider samples only while the job runs. A manual `powermetrics` window
  also covers the idle time between requests.

The benchmark prints the ratio per run and for the whole run set:

```bash
offgrid-inference benchmark --runs 3 --max-tokens 256
```

The dashboard shows it on the INFERENCE card as "SoC energy per generated
token". The value is `n/a` together with a reason until a measurement succeeds.

## VRM energy state

`VRMProvider` reads the diagnostics endpoint of one installation and normalizes
the documented system paths into `EnergyState`:

```text
battery_soc, solar_power_w, battery_power_w, ac_load_w, solar_forecast_wh
```

Rules of the adapter:

- Read power and battery measurements from the `system` service only.
  If its phase count is absent, use configuration from one unambiguous VE.Bus system.
  Require a power measurement for every configured phase.
  Do not use the age of unchanged phase configuration as the age of measured power.
- Mark a measurement `STALE` when it is older than the freshness limit, and
  `UNAVAILABLE` when no value exists.
- Keep the last good response in the cache, so an internet outage does not stop
  the scheduler.
- Respect `Retry-After` on a 429 response.
- Never send raw VRM JSON to a workload.

VRM supplies no solar forecast on the diagnostics endpoint, so
`solar_forecast_wh` stays null. The scheduler uses a forecast budget only when
a provider supplies the value.

The default scheduler interval (`POLL_SECONDS`) and VRM API interval
(`VRM_POLL_SECONDS`, minimum 5) are 5 seconds. The browser also refreshes every
5 seconds. A longer scheduler interval limits the API request frequency.
The ENERGY timestamp shows the measurement time, not the last browser refresh.

REST diagnostics can return unchanged measurements between GX uploads.
Faster requests cannot make those measurements newer.
[VRM real-time mode](https://www.victronenergy.com/media/pg/VRM_Portal_manual/en/real-time-data.html)
uses a separate connection with updates every two seconds.
This application does not implement that connection.

## Weather and the daily briefing

```text
Weather API -> provider -> Forecast -> deterministic parser
            -> compact table + trends + warnings -> inference -> briefing
```

The parser computes every number: extremes, totals, the pressure drop, the
direction shift, and the safety notes. The model interprets those numbers and
writes the text. The model must not invent a measurement, and it must name the
fields that are absent.

The job stores the raw API payload, the normalized forecast, the prompt, and
the generated text. Each briefing is auditable.

## Scheduling policy

The policy reads job metadata only. It never tests a workload name.

| Situation | Decision |
| --- | --- |
| Start time in the future | `WAIT`, the job stays `QUEUED` |
| Battery SOC unknown | `DEFER` |
| Battery SOC at or below the floor | `DEFER`, for every job |
| Priority 100 or more | `RUN` above the floor |
| Not deferrable | `RUN` above the floor |
| Deadline inside the runtime lead | `RUN` above the floor |
| SOC below the high threshold | `DEFER` |
| Surplus below the threshold | `DEFER` |
| Surplus above the threshold for less than the hold period | `DEFER` |
| Estimated energy above the forecast budget | `DEFER` |
| All conditions hold | `RUN` |

The safety floor applies to critical jobs as well. An empty battery stops the
boat, not the queue.

Hysteresis protects against a short spike and against a passing cloud. The hold
uses distinct, timely samples. A repeated cached sample cannot prove that a
surplus continued. A sample gap clears the hold. A stale but recent measurement
may still guard the floor for a critical or deadline job.

Every decision goes to `scheduler_decisions` with the energy state and a reason
in words:

```json
{
  "job": "document-ingestion-41",
  "decision": "DEFER",
  "battery_soc": 61,
  "solar_surplus_w": 83,
  "reason": "Solar surplus 83 W is below the 250 W threshold"
}
```

## Add a workload

1. Write a class with one method: `async def run(self, job: Job) -> JobResult`.
2. Give the constructor the providers that it needs.
3. Register the class in `build_runtime` in `src/api/runtime.py`.
4. Queue a job with `workload="<name>"` and the metadata: `priority`,
   `deadline`, `deferrable`, `estimated_energy_wh`.
5. Add a mock provider and a test.

No change is necessary in the scheduler, in the energy monitor, or in the
inference layer. A plugin can also register itself through `Worker.register`
or through a daily schedule with `Scheduler.register_daily`.

Example job metadata:

```python
Job(workload="document_ingestion", priority=20, deadline=None,
    deferrable=True, estimated_energy_wh=40)
```

## Offline behaviour

Local components keep working without internet:

```text
PostgreSQL, scheduler, llama.cpp, Qwen, energy monitor, cache, dashboard
```

The VRM adapter and the weather provider return `STALE` with cached data, or
`UNAVAILABLE` with a reason. A workload that has no forecast fails with an
explicit error. The system never fabricates a measurement.

## Dashboard

The dashboard shows four blocks:

| Block | Content |
| --- | --- |
| ENERGY | SOC, solar power, load, surplus, forecast, source status |
| INFERENCE | model, server status, tok/s, requests, KV usage, SoC J/tok |
| SCHEDULER | mode, monitor, thresholds, running and deferred jobs, last reason |
| TODAY | generated tokens, estimated Wh, jobs completed, jobs deferred |

The footer links to VRM for electrical detail.

## Tests and CI

```bash
pytest -q                        # unit and integration tests
python -m api.cli smoke          # offline scheduling demonstration
TEST_DATABASE_URL=... pytest -q  # adds the PostgreSQL tests
```

Use a separate database for `TEST_DATABASE_URL`. The PostgreSQL tests truncate
their tables:

```bash
createdb -h 127.0.0.1 -p 55432 energy_compute_test
TEST_DATABASE_URL=postgresql://127.0.0.1:55432/energy_compute_test pytest -q
```

Every external dependency has a double: `MockVRMProvider`,
`MockWeatherProvider`, `MockInferenceProvider`, `MockEnergyMonitor`, and an
in-memory queue. The test suite needs no boat, no VRM, no weather API, no Qwen,
no `llama.cpp`, and no Apple silicon. The CI workflow runs lint, migrations,
the tests, and the offline demonstration.

## Repository layout

```text
src/{contracts.py,inference,energy,jobs,db,scheduler,workloads,api}
tests/{unit,integration,mocks,fixtures}
scripts/{setup.sh,start-inference.sh,smoke-test.sh,power-probe.sh}
.github/workflows/ci.yml
docker-compose.yml
```
