#!/usr/bin/env bash
# Logs GPU stats every 100ms (plus DCGM and CPU) in the background, into results/<gpu>/host-<hostname>/
# so two rentals of the same GPU type never overwrite each other's logs.
#   bash telemetry.sh start    # before sweep.py
#   bash telemetry.sh stop     # after
set -euo pipefail
cd "$(dirname "$0")"

FAM=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | grep -oE 'H100|B200' | tr 'A-Z' 'a-z' || echo other)
OUT="results/$FAM/host-$(hostname)"
PIDS="$OUT/.telemetry.pids"
mkdir -p "$OUT"

WANT="timestamp power.draw power.draw.instant power.draw.average power.limit enforced.power.limit
clocks.sm clocks.mem temperature.gpu temperature.memory utilization.gpu
clocks_event_reasons.sw_power_cap clocks_event_reasons.hw_slowdown
clocks_event_reasons.sw_thermal_slowdown clocks_event_reasons.hw_thermal_slowdown"

case "${1:-}" in
  start)
    if [ -f "$PIDS" ]; then echo "Already running. bash telemetry.sh stop first"; exit 1; fi
    FIELDS=""
    for f in $WANT; do
      if nvidia-smi -i 0 --query-gpu="$f" --format=csv,noheader >/dev/null 2>&1; then
        FIELDS="$FIELDS,$f"
      else
        echo "driver doesn't support $f, skipping it"
      fi
    done
    nohup nvidia-smi -i 0 --query-gpu="${FIELDS#,}" --format=csv -lms 100 >> "$OUT/telemetry.csv" 2>&1 &
    echo $! > "$PIDS"
    # an old DCGM (e.g. 3.x on a B200) installs fine but can't read the GPU, so test it first
    if command -v dcgmi >/dev/null && timeout 20 dcgmi dmon -i 0 -e 1002 -c 1 >/dev/null 2>&1; then
      # dcgmi dmon prints no time column, so stamp each line with the epoch time as it arrives.
      # $! is the stamping loop; when it's killed, dcgmi dies on its next write
      nohup dcgmi dmon -i 0 -e 155,156,100,101,150,140,1002,1004,1005 -d 100 2>&1 \
        | while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done >> "$OUT/dcgm.csv" &
      echo $! >> "$PIDS"
    else
      echo "DCGM not usable here, skipping it"
    fi
    nohup mpstat 1 >> "$OUT/mpstat.log" 2>&1 &
    echo $! >> "$PIDS"
    sleep 3
    for pid in $(cat "$PIDS"); do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "FAIL: one of the loggers died right away, check $OUT/"
        kill $(cat "$PIDS") 2>/dev/null || true; rm -f "$PIDS"; exit 1
      fi
    done
    if [ ! -s "$OUT/telemetry.csv" ]; then
      echo "FAIL: nothing in $OUT/telemetry.csv"; kill $(cat "$PIDS") 2>/dev/null || true; rm -f "$PIDS"; exit 1
    fi
    echo "Logging to $OUT/"
    ;;
  stop)
    if [ -f "$PIDS" ]; then kill $(cat "$PIDS") 2>/dev/null || true; rm -f "$PIDS"; fi
    echo "Stopped"
    ;;
  *)
    echo "usage: bash telemetry.sh start|stop"; exit 1
    ;;
esac
