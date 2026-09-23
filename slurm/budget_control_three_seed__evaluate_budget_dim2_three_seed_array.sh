#!/bin/bash
#SBATCH --job-name=thesis_budget_dim2_val
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-5
#SBATCH --output=analysis/final_thesis/logs/%x_%A_%a.out
#SBATCH --error=analysis/final_thesis/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(
  budget_original_only
  budget_original_plus_dim2
  budget_original_only
  budget_original_plus_dim2
  budget_original_only
  budget_original_plus_dim2
)

SEEDS=(42 42 123 123 2026 2026)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds
ADAPTER_DIR="${OUTPUT_ROOT}/seed_${SEED}/adapters/${CONDITION}"
EVAL_OUTPUT="${OUTPUT_ROOT}/seed_${SEED}/validation_evaluations/${CONDITION}"

mkdir -p analysis/final_thesis/logs
mkdir -p "${EVAL_OUTPUT}"

if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Missing adapter_config.json in ${ADAPTER_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" && ! -f "${ADAPTER_DIR}/adapter_model.bin" ]]; then
  echo "Missing adapter weights in ${ADAPTER_DIR}" >&2
  exit 1
fi
if [[ -f "${EVAL_OUTPUT}/predictions.jsonl" ]]; then
  echo "Refusing to overwrite existing ${EVAL_OUTPUT}/predictions.jsonl" >&2
  exit 1
fi

python -u experiments/cogmath_reasoning_budget_control/static_validation.py

python -u experiments/cogmath_reasoning_budget_control/evaluate_budget_control.py \
  --model_name Qwen/Qwen3-4B-Base \
  --adapter_dir "${ADAPTER_DIR}" \
  --condition "${CONDITION}" \
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl \
  --output_dir "${EVAL_OUTPUT}" \
  --batch_size 8 \
  --max_new_tokens 512 \
  --warmup_generations 1 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --seed "${SEED}"
