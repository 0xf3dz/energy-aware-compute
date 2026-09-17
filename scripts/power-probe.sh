#!/usr/bin/env bash
# Capture two raw powermetrics samples to verify field names and the baseline.
#
# powermetrics needs root. Run this with an interactive sudo, or add a sudoers
# rule such as:
#     f3dz ALL=(root) NOPASSWD: /usr/bin/powermetrics
set -euo pipefail

OUT="${1:-.data/powermetrics-probe.plist}"

mkdir -p "$(dirname "$OUT")"
sudo /usr/bin/powermetrics \
  --format plist --buffer-size 0 \
  --sample-rate "${SAMPLE_RATE_MS:-1000}" \
  --samplers cpu_power,gpu_power,tasks \
  --order cputime --show-process-energy \
  -n "${SAMPLES:-3}" >"$OUT"

echo "Wrote $OUT"
python3 - "$OUT" <<'PY'
import plistlib
import sys

raw = open(sys.argv[1], "rb").read()
for index, chunk in enumerate(raw.split(b"\0")):
    chunk = chunk.strip()
    if not chunk:
        continue
    document = plistlib.loads(chunk)
    processor = document.get("processor", {})
    powers = {key: value for key, value in processor.items() if "power" in key or "energy" in key}
    tasks = [task for task in document.get("tasks", []) if task.get("name") == "llama-server"]
    print(f"sample {index}: power keys {powers}")
    for task in tasks:
        print(f"  llama-server pid {task['pid']} impact {task.get('energy_impact')}")
    break
PY
