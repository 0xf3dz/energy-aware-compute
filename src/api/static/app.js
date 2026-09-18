"use strict";

const REFRESH_MS = 5000;
const TOKEN = new URLSearchParams(location.search).get("token");
const AUTH = TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : "";

function fmt(value, unit = "", digits = 1) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return `${value.toFixed(digits)}${unit ? " " + unit : ""}`;
  return `${value}${unit ? " " + unit : ""}`;
}

function rows(target, pairs) {
  document.querySelector(`#${target} tbody`).innerHTML = pairs
    .map(([key, value, cls]) =>
      `<tr><td class="key">${key}</td><td class="${cls || ""}">${value}</td></tr>`)
    .join("");
}

function shortTime(value) {
  if (!value) return "n/a";
  return new Date(value).toISOString().replace("T", " ").slice(0, 19) + "Z";
}

function render(data) {
  const energy = data.energy.latest || {};
  let surplus =
    energy.solar_power_w !== null && energy.solar_power_w !== undefined &&
    energy.ac_load_w !== null && energy.ac_load_w !== undefined
      ? energy.solar_power_w - energy.ac_load_w
      : null;
  if (surplus !== null && energy.battery_power_w != null) {
    surplus = Math.min(surplus, energy.battery_power_w);
  }
  rows("energy", [
    ["State (UTC)", shortTime(energy.timestamp)],
    ["Source status", energy.availability || "UNAVAILABLE",
      (energy.availability || "unavailable").toLowerCase()],
    ["Battery SOC", fmt(energy.battery_soc, "%", 0)],
    ["Solar power", fmt(energy.solar_power_w, "W", 0)],
    ["AC load", fmt(energy.ac_load_w, "W", 0)],
    ["Battery power", fmt(energy.battery_power_w, "W", 0)],
    ["Estimated surplus", fmt(surplus, "W", 0)],
    ["Solar forecast", fmt(energy.solar_forecast_wh, "Wh", 0)],
    ["Samples cached", data.energy.samples.length],
  ]);
  document.querySelector("#energy-reason").textContent = energy.reason || "";

  const live = data.inference.live || {};
  const latest = data.inference.latest || {};
  rows("inference", [
    ["Model", data.runtime.model || "n/a"],
    ["Server", data.runtime.llama_url],
    ["Server status", live.availability === "FRESH" ? "ok" : "unavailable",
      live.availability === "FRESH" ? "fresh" : "unavailable"],
    ["Active requests", fmt(live.active_requests, "", 0)],
    ["KV cache usage", fmt(live.kv_cache_usage !== null && live.kv_cache_usage !== undefined
      ? live.kv_cache_usage * 100 : null, "%", 1)],
    ["Metric window", fmt(live.interval_seconds, "s", 1)],
    ["Prompt tok/s (window)", fmt(live.prompt_tokens_per_second, "", 1)],
    ["Generation tok/s (window)", fmt(live.generation_tokens_per_second, "", 1)],
    ["Last request prompt tokens", latest.prompt_tokens ?? "n/a"],
    ["Last request generated tokens", latest.generated_tokens ?? "n/a"],
    ["Last request tok/s", fmt(latest.generation_tokens_per_second, "", 1)],
    ["Last request runtime", fmt(latest.runtime_seconds, "s", 2)],
    ["SoC energy per generated token", fmt(latest.joules_per_generated_token, "J/tok", 2)],
  ]);
  document.querySelector("#inference-reason").textContent =
    latest.joules_per_generated_token == null
      ? "No SoC energy measurement. J/tok needs a successful powermetrics sample and a token count."
      : "J/tok: measured SoC rails only (CPU, GPU, ANE), idle baseline subtracted, " +
        "whole request divided by generated tokens. DRAM and storage are outside the measurement.";

  const policy = data.runtime.policy || {};
  rows("scheduler", [
    ["Mode", data.runtime.energy_source],
    ["Energy monitor", data.runtime.monitor],
    ["Scheduler interval", fmt(data.runtime.poll_seconds, "s", 0)],
    ["VRM API interval", fmt(data.runtime.vrm_poll_seconds, "s", 0)],
    ["Display interval", fmt(REFRESH_MS / 1000, "s", 0)],
    ["Battery floor", fmt(policy.battery_floor, "%", 0)],
    ["High SOC", fmt(policy.high_soc, "%", 0)],
    ["Surplus threshold", fmt(policy.surplus_w, "W", 0)],
    ["Surplus held", `${fmt(data.runtime.surplus_held_seconds, "s", 0)} of ${fmt(policy.surplus_hold_seconds, "s", 0)}`],
    ["Schedules", (data.runtime.schedules || []).map(s => `${s.workload} at ${s.at_utc}Z`).join(", ") || "none"],
  ]);

  const decision = (data.decisions || [])[0];
  document.querySelector("#decision").textContent = decision
    ? `${decision.decision} ${decision.workload || decision.job_id}\n${decision.reason}\nbattery ${fmt(decision.energy_state.battery_soc, "%", 0)}, solar ${fmt(decision.energy_state.solar_power_w, "W", 0)}, load ${fmt(decision.energy_state.ac_load_w, "W", 0)}`
    : "No decision recorded yet.";

  document.querySelector("#deferred tbody").innerHTML = (data.jobs.deferred || [])
    .map(job => `<tr><td>${job.workload}</td><td>${job.priority}</td><td>${fmt(job.estimated_energy_wh, "Wh", 1)}</td></tr>`)
    .join("") || "<tr><td colspan=\"3\">No deferred jobs</td></tr>";

  const today = data.today || {};
  rows("today", [
    ["Generated tokens", today.generated_tokens ?? 0],
    ["Inference requests", today.inference_requests ?? 0],
    ["Estimated inference energy", today.estimated_inference_wh == null
      ? "not measured" : fmt(today.estimated_inference_wh, "Wh", 3)],
    ["Jobs completed", today.jobs_completed ?? 0],
    ["Jobs deferred to solar", today.jobs_deferred_to_solar ?? 0],
    ["Jobs waiting for energy", today.jobs_waiting_for_energy ?? 0],
  ]);

  const jobs = [].concat(data.jobs.running || [], data.jobs.recent || []);
  document.querySelector("#jobs tbody").innerHTML = jobs.map(job => `<tr>
      <td>${job.workload}</td>
      <td class="${String(job.status).toLowerCase()}">${job.status}</td>
      <td>${job.priority}</td>
      <td>${job.deferrable ? "yes" : "no"}</td>
      <td>${shortTime(job.completed_at)}</td>
      <td>${fmt(job.actual_estimated_energy_wh, "", 3)}</td>
      <td class="error">${job.error || ""}</td>
    </tr>`).join("") || "<tr><td colspan=\"7\">No jobs yet</td></tr>";

  document.querySelector("#decisions tbody").innerHTML = (data.decisions || []).map(item => `<tr>
      <td>${shortTime(item.timestamp)}</td>
      <td>${item.workload || item.job_id.slice(0, 8)}</td>
      <td>${item.decision}</td>
      <td>${fmt(item.energy_state.battery_soc, "%", 0)}</td>
      <td>${fmt(item.energy_state.solar_power_w !== null && item.energy_state.ac_load_w !== null
        ? item.energy_state.solar_power_w - item.energy_state.ac_load_w : null, "", 0)}</td>
      <td>${item.reason}</td>
    </tr>`).join("") || "<tr><td colspan=\"6\">No decisions yet</td></tr>";

  const briefing = data.latest_briefing;
  document.querySelector("#briefing").textContent = briefing
    ? `${briefing.text}\n\n[${shortTime(briefing.created_at)} · ${briefing.job_id}]`
    : "No briefing has been generated yet.";

  document.querySelector("#runtime").textContent =
    `demo=${data.runtime.demo} poll=${data.runtime.poll_seconds}s ` +
    `source=${data.runtime.energy_source} monitor=${data.runtime.monitor}`;
  document.querySelector("#vrm").href = data.runtime.vrm_url || "#";

  const status = document.querySelector("#status");
  status.textContent = `updated ${new Date().toISOString().slice(11, 19)}Z`;
  status.className = "status ok";
}

async function refresh() {
  try {
    const response = await fetch(`/api/snapshot${AUTH}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    const status = document.querySelector("#status");
    status.textContent = `snapshot failed: ${error.message}`;
    status.className = "status bad";
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
