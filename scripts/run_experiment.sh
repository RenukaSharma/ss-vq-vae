#!/usr/bin/env bash
# Generalized experiment runner for ss-VQ-VAE (DICTA 2023).
# No machine-specific paths. Set DATA_DIR and RESULTS_DIR in the environment.
#
# Usage:
#   export DATA_DIR=$HOME/data/mvtec
#   export RESULTS_DIR=$HOME/results/ss-vq-vae
#   ./scripts/run_experiment.sh mvtec vq_vae_net --category carpet --pretrain False
#
# Sweep (paper-style loops):
#   ./scripts/run_experiment.sh --sweep corrosion vq_vae_net \
#       --ratios 0,0.01,0.05,0.1,0.2 --seeds 0,1,2 --ablations A --pretrain False
#
# Optional extra flags after the dataset/net are forwarded to src/main.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT}/data}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results}"

SWEEP=0
RATIOS="0,0.01,0.05,0.1,0.2"
SEEDS="0"
ABLATIONS="A"
EXTRA=()

usage() {
  sed -n '2,16p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --sweep) SWEEP=1; shift ;;
    --ratios) RATIOS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --ablations) ABLATIONS="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    --*) EXTRA+=("$1"); shift
         if [[ $# -gt 0 && "$1" != --* && "$1" != mvtec && "$1" != corrosion && "$1" != malaria_dataset && "$1" != nanofibre ]]; then
           EXTRA+=("$1"); shift
         fi
         ;;
    *) break ;;
  esac
done

if [[ $# -lt 2 ]]; then
  echo "Need DATASET_NAME and NET_NAME." >&2
  usage
fi

DATASET="$1"
NET="$2"
shift 2
EXTRA+=("$@")

if [[ ! -d "$DATA_DIR" ]]; then
  echo "DATA_DIR does not exist: $DATA_DIR" >&2
  echo "Export DATA_DIR to your local dataset root." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

run_one() {
  local ratio="$1" seed="$2" ablation="$3"
  local xp
  xp="${RESULTS_DIR}/${DATASET}/ratio_${ratio}/seed_${seed}/ablation_${ablation}"
  mkdir -p "$xp"
  echo ">>> $DATASET $NET  ratio=${ratio} seed=${seed} ablation=${ablation}"
  echo "    xp=$xp"
  echo "    data=$DATA_DIR"
  ( cd "$ROOT/src" && python main.py "$DATASET" "$NET" "$xp" "$DATA_DIR" \
      --ratio_known_outlier "$ratio" \
      --seed "$seed" \
      --ablation_type "$ablation" \
      "${EXTRA[@]}" )
}

IFS=',' read -r -a RATIO_ARR <<< "$RATIOS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a ABL_ARR <<< "$ABLATIONS"

if [[ "$SWEEP" -eq 1 ]]; then
  for ratio in "${RATIO_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      for abl in "${ABL_ARR[@]}"; do
        run_one "$ratio" "$seed" "$abl"
      done
    done
  done
else
  run_one "${RATIO_ARR[0]}" "${SEED_ARR[0]}" "${ABL_ARR[0]}"
fi
