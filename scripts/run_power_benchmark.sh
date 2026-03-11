#!/bin/bash

set -euo pipefail

MODE="${1:-both}"
MODEL_PATH="${2:-benchmarks/models/transformer_layer.mlpackage}"
TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
RESULTS_DIR="benchmarks/results/power-${TIMESTAMP}"
BENCH="./.build/release/espresso-bench"

usage() {
  cat <<'EOF'
Usage: scripts/run_power_benchmark.sh [ane|coreml|both] [model-path]

Runs `powermetrics` alongside `espresso-bench --sustained`.
  ane     Runs ANE direct only (`--ane-only`)
  coreml  Runs ANE + Core ML baseline
  both    Runs both modes sequentially
EOF
}

run_case() {
  local label="$1"
  shift

  local bench_args=("$@")
  local power_log="${RESULTS_DIR}/${label}-powermetrics.log"

  echo "=== ${label} ==="
  sudo powermetrics \
    --samplers cpu_power,gpu_power,ane_power \
    --sample-interval 1000 \
    -n 60 \
    >"${power_log}" 2>&1 &
  local power_pid=$!

  "${BENCH}" \
    --sustained \
    --warmup 10 \
    --iterations 100 \
    --model "${MODEL_PATH}" \
    --output "${RESULTS_DIR}/${label}" \
    "${bench_args[@]}"

  wait "${power_pid}" || true
  echo "powermetrics log: ${power_log}"
  echo
}

case "${MODE}" in
  ane|coreml|both)
    ;;
  *)
    usage
    exit 1
    ;;
esac

mkdir -p "${RESULTS_DIR}"
swift build -c release --target EspressoBench

if [[ "${MODE}" == "ane" || "${MODE}" == "both" ]]; then
  run_case "ane" --ane-only
fi

if [[ "${MODE}" == "coreml" || "${MODE}" == "both" ]]; then
  run_case "coreml"
fi
