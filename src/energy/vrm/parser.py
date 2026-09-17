"""Normalize documented VRM diagnostic records, never formatted device totals.

Sources: https://vrm-api-docs.victronenergy.com/
https://github.com/victronenergy/venus/wiki/dbus#system
"""

import math
from datetime import datetime, timezone
from typing import Any

from contracts import Availability, EnergyState


def parse_diagnostics(raw: dict[str, Any], now: datetime, freshness_seconds: float) -> EnergyState:
    records = raw.get("records", [])
    if isinstance(records, dict):
        records = [records] if "dbusPath" in records else list(records.values())
    if not isinstance(records, list) or raw.get("success") is not True:
        raise ValueError("Invalid VRM diagnostics response")
    # A system service already selects the main battery and aggregates its chargers.
    # Do not add physical battery/charger readings to these aggregate measurements.
    paths: dict[str, tuple[float, datetime]] = {}
    instances = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("dbusServiceType") not in ("system", "com.victronenergy.system"):
            continue
        path = record.get("dbusPath")
        if not isinstance(path, str):
            continue
        value, stamp = record.get("rawValue"), record.get("timestamp")
        if isinstance(value, bool) or isinstance(stamp, bool):
            continue
        try:
            value = float(value)
            stamp = datetime.fromtimestamp(float(stamp), timezone.utc)
        except (ValueError, TypeError, OverflowError, OSError):
            continue
        if not math.isfinite(value) or stamp > now:
            continue
        instances.add(record.get("instance"))
        if path not in paths or stamp > paths[path][1]:
            paths[path] = value, stamp
        elif stamp == paths[path][1] and value != paths[path][0]:
            raise ValueError("Conflicting VRM system measurements")
    if len(instances) > 1:
        raise ValueError("Ambiguous VRM system instances")

    used: list[datetime] = []

    def measurement(path: str, *, minimum: float | None = None,
                    maximum: float | None = None) -> float | None:
        item = paths.get(path)
        if item is None:
            return None
        value, stamp = item
        if minimum is not None and value < minimum:
            return None
        if maximum is not None and value > maximum:
            return None
        used.append(stamp)
        return value

    soc = measurement("/Dc/Battery/Soc", minimum=0, maximum=100)
    battery = measurement("/Dc/Battery/Power")
    # This is measured DC-coupled PV, a conservative lower bound on all solar.
    # Never infer AC-coupled PV or a total from missing device readings.
    solar = measurement("/Dc/Pv/Power", minimum=0)
    phases = paths.get("/Ac/Consumption/NumberOfPhases")
    load = None
    if phases is not None and phases[0] in (1, 2, 3):
        phase_paths = [f"/Ac/Consumption/L{n}/Power" for n in range(1, int(phases[0]) + 1)]
        if all(path in paths and paths[path][0] >= 0 for path in phase_paths):
            load = sum(paths[path][0] for path in phase_paths)
            used.extend(paths[path][1] for path in phase_paths)
            used.append(phases[1])
    if not used:
        return EnergyState(timestamp=now, reason="No usable timestamped system measurements")
    stamp = min(used)  # Conservative age of the oldest contributing measurement.
    stale = (now - stamp).total_seconds() > freshness_seconds
    reasons = ["Solar power covers measured DC-coupled PV only; AC-coupled PV is excluded"]
    if stale:
        reasons.append("VRM measurement age exceeds freshness limit")
    return EnergyState(timestamp=stamp, availability=Availability.STALE if stale else Availability.FRESH,
                       battery_soc=soc, solar_power_w=solar, battery_power_w=battery,
                       ac_load_w=load, reason="; ".join(reasons))
