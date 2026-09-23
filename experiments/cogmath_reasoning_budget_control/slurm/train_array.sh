#!/bin/bash
#SBATCH --job-name=cogmath_reason_budget_train
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=experiments/cogmath_reasoning_budget_control/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_reasoning_budget_control/logs/%x_%A_%a.err

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
  budget_original_plus_dim4
  budget_original_plus_dim2_dim4
  budget_original_plus_all
)

SOURCE_FILES=(
  experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl
  experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2.jsonl
  experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim4.jsonl
  experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2_dim4.jsonl
  experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_all.jsonl
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_FILE="${SOURCE_FILES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b

mkdir -p experiments/cogmath_reasoning_budget_control/logs
mkdir -p "${OUTPUT_ROOT}/adapters"

python -u experiments/cogmath_reasoning_budget_control/prepare_budget_control.py
python -u experiments/cogmath_reasoning_budget_control/static_validation.py

python -u experiments/cogmath_reasoning_budget_control/train_budget_control.py \
  --condition "${CONDITION}" \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_ROOT}/adapters/${CONDITION}" \
  --seed 42 \
  --max_steps 232 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
