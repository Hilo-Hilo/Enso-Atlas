#!/usr/bin/env bash
# Lightweight dev API latency benchmark for Enso Atlas.

set -euo pipefail

BASE_URL="${1:-${ENSO_API_BASE_URL:-http://localhost:8000}}"
SAMPLES="${SAMPLES:-10}"
OUT_DIR="${OUT_DIR:-benchmarks}"

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]] || [ "$SAMPLES" -lt 1 ]; then
  echo "SAMPLES must be a positive integer (got: $SAMPLES)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
RAW_CSV="$OUT_DIR/dev-latency-${STAMP}.csv"

ENDPOINTS=(
  "/api/health"
  "/health"
  "/api/slides"
  "/api/tags"
  "/api/projects"
)

printf "endpoint,sample,http_code,total_ms\n" > "$RAW_CSV"

echo "Benchmarking $BASE_URL (${SAMPLES} samples/endpoint)"

for endpoint in "${ENDPOINTS[@]}"; do
  echo "→ $endpoint"
  for ((i = 1; i <= SAMPLES; i++)); do
    line="$(curl -sS -o /dev/null -w "%{time_total} %{http_code}" "${BASE_URL}${endpoint}" || echo "nan 000")"
    total_s="${line%% *}"
    code="${line##* }"

    total_ms="$(python3 - <<PY
import math
v = float("$total_s") if "$total_s" != "nan" else float('nan')
print("nan" if math.isnan(v) else f"{v * 1000.0:.3f}")
PY
)"

    printf "%s,%s,%s,%s\n" "$endpoint" "$i" "$code" "$total_ms" >> "$RAW_CSV"
  done
done

python3 - "$RAW_CSV" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

csv_path = sys.argv[1]
rows = defaultdict(list)
status_counts = defaultdict(lambda: defaultdict(int))

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        endpoint = row["endpoint"]
        status = row["http_code"]
        status_counts[endpoint][status] += 1
        try:
            v = float(row["total_ms"])
        except Exception:
            continue
        if math.isnan(v):
            continue
        rows[endpoint].append(v)


def percentile(values, p):
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return arr[0]
    idx = (len(arr) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    if lo == hi:
        return arr[lo]
    frac = idx - lo
    return arr[lo] + (arr[hi] - arr[lo]) * frac

print("\nLatency summary (ms):")
print(f"{'Endpoint':34} {'n':>4} {'p50':>10} {'p95':>10} {'max':>10} {'status':>18}")
print("-" * 92)

for endpoint, vals in rows.items():
    status = ",".join(f"{k}:{v}" for k, v in sorted(status_counts[endpoint].items()))
    print(
        f"{endpoint:34} {len(vals):4d} {percentile(vals,50):10.2f} "
        f"{percentile(vals,95):10.2f} {max(vals):10.2f} {status:>18}"
    )

print("\nRaw samples:", csv_path)
PY

