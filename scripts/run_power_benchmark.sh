#!/bin/bash

set -euo pipefail

MODE="${1:-both}"
MODEL_PATH="${2:-benchmarks/models/transformer_layer.mlpackage}"
LAYERS="${LAYERS:-1}"
COREML_WEIGHT_MODE="${COREML_WEIGHT_MODE:-random}"
TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
RESULTS_DIR="benchmarks/results/power-${TIMESTAMP}"
BENCH="./.build/release/espresso-bench"
ENSURE_COREML_MODEL="./scripts/ensure_coreml_model.sh"

usage() {
  cat <<'EOF'
Usage: scripts/run_power_benchmark.sh [ane|coreml|both] [model-path]

Runs `powermetrics` alongside `espresso-bench --sustained`.
  ane     Runs ANE direct only (`--ane-only`)
  coreml  Runs ANE + Core ML baseline
  both    Runs both modes sequentially
Set `LAYERS` in the environment to benchmark a model with more than one layer.
EOF
}

prepare_coreml_model() {
  if [[ -f "${MODEL_PATH}" ]]; then
    return 0
  fi

  if [[ ! -x "${ENSURE_COREML_MODEL}" ]]; then
    echo "Core ML helper is missing or not executable: ${ENSURE_COREML_MODEL}" >&2
    exit 1
  fi

  echo "Preparing Core ML baseline model at ${MODEL_PATH}"
  "${ENSURE_COREML_MODEL}" \
    --output "${MODEL_PATH}" \
    --layers "${LAYERS}" \
    --weight-mode "${COREML_WEIGHT_MODE}"
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
    --layers "${LAYERS}" \
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

if [[ "${MODE}" == "coreml" || "${MODE}" == "both" ]]; then
  prepare_coreml_model
fi

if [[ "${MODE}" == "ane" || "${MODE}" == "both" ]]; then
  run_case "ane" --ane-only
fi

if [[ "${MODE}" == "coreml" || "${MODE}" == "both" ]]; then
  run_case "coreml"
fi
