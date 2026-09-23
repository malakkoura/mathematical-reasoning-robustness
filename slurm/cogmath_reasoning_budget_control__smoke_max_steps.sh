#!/bin/bash
#SBATCH --job-name=cogmath_reason_budget_smoke
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=experiments/cogmath_reasoning_budget_control/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_budget_control/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b
SMOKE_ID="${SLURM_JOB_ID:-local}"

mkdir -p experiments/cogmath_reasoning_budget_control/logs
mkdir -p "${OUTPUT_ROOT}/smoke/${SMOKE_ID}"

python -u experiments/cogmath_reasoning_budget_control/prepare_budget_control.py
python -u experiments/cogmath_reasoning_budget_control/static_validation.py

python -u experiments/cogmath_reasoning_budget_control/train_budget_control.py \
  --condition budget_original_plus_all \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_all.jsonl \
  --output_dir "${OUTPUT_ROOT}/smoke/${SMOKE_ID}/budget_original_plus_all" \
  --seed 42 \
  --max_steps 2 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --smoke_samples 32 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
