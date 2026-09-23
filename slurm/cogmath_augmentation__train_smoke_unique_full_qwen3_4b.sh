#!/bin/bash
#SBATCH --job-name=cogmath_aug_unique_4b_smoke
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=experiments/cogmath_augmentation/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_augmentation/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_augmentation_qwen3_4b_unique_full

mkdir -p experiments/cogmath_augmentation/logs
mkdir -p "${OUTPUT_ROOT}/smoke"

python -u experiments/cogmath_augmentation/train_lora_unique_full.py \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_augmentation/data_unique_full/train_original_only.jsonl \
  --condition original_only \
  --output_dir "${OUTPUT_ROOT}/smoke/original_only_32" \
  --seed 42 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --smoke_samples 32 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
