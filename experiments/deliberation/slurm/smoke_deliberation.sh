#!/bin/bash
#SBATCH --job-name=thesis_delib_smoke
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=analysis/final_thesis/deliberation/logs/%x_%j.out
#SBATCH --error=analysis/final_thesis/deliberation/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

mkdir -p analysis/final_thesis/deliberation/logs

FIRST_PASS_PREDICTIONS="${FIRST_PASS_PREDICTIONS:-${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds/seed_42/validation_evaluations/budget_original_plus_dim2/predictions.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Base}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds/seed_42/adapters/budget_original_plus_dim2}"

if [[ ! -f "${FIRST_PASS_PREDICTIONS}" ]]; then
  echo "Missing first-pass predictions: ${FIRST_PASS_PREDICTIONS}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Missing adapter_config.json in ${ADAPTER_DIR}" >&2
  exit 1
fi

python -u analysis/final_thesis/deliberation/smoke_test_deliberation.py \
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl \
  --first_pass_predictions "${FIRST_PASS_PREDICTIONS}" \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --records 2 \
  --run_model_smoke
