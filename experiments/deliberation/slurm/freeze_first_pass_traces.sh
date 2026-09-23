#!/bin/bash
#SBATCH --job-name=thesis_delib_freeze
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=analysis/final_thesis/deliberation/logs/%x_%j.out
#SBATCH --error=analysis/final_thesis/deliberation/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/final_thesis_deliberation_qwen3_4b
FIRST_PASS_PREDICTIONS="${FIRST_PASS_PREDICTIONS:-${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds/seed_42/validation_evaluations/budget_original_plus_dim2/predictions.jsonl}"
OUTPUT_FILE="${OUTPUT_ROOT}/first_pass_traces/first_pass_traces.jsonl"

mkdir -p analysis/final_thesis/deliberation/logs
mkdir -p "${OUTPUT_ROOT}/first_pass_traces"

if [[ ! -f "${FIRST_PASS_PREDICTIONS}" ]]; then
  echo "Missing first-pass predictions: ${FIRST_PASS_PREDICTIONS}" >&2
  exit 1
fi
if [[ -f "${OUTPUT_FILE}" ]]; then
  echo "Refusing to overwrite existing frozen traces: ${OUTPUT_FILE}" >&2
  exit 1
fi

python -u analysis/final_thesis/deliberation/freeze_first_pass_traces.py \
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl \
  --first_pass_predictions "${FIRST_PASS_PREDICTIONS}" \
  --output_file "${OUTPUT_FILE}"
