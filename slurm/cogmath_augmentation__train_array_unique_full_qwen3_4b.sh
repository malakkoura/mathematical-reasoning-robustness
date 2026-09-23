#!/bin/bash
#SBATCH --job-name=cogmath_aug_unique_4b_train
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-6
#SBATCH --output=experiments/cogmath_augmentation/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_augmentation/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(
  original_only
  original_plus_dim1
  original_plus_dim2
  original_plus_dim4
  original_plus_dim6
  original_plus_dim2_dim4
  original_plus_all
)

TRAIN_FILES=(
  train_original_only.jsonl
  train_original_plus_dim1.jsonl
  train_original_plus_dim2.jsonl
  train_original_plus_dim4.jsonl
  train_original_plus_dim6.jsonl
  train_original_plus_dim2_dim4.jsonl
  train_original_plus_all.jsonl
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_FILE="${TRAIN_FILES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_augmentation_qwen3_4b_unique_full

mkdir -p experiments/cogmath_augmentation/logs
mkdir -p "${OUTPUT_ROOT}/adapters"

python -u experiments/cogmath_augmentation/train_lora_unique_full.py \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file "experiments/cogmath_augmentation/data_unique_full/${TRAIN_FILE}" \
  --condition "${CONDITION}" \
  --output_dir "${OUTPUT_ROOT}/adapters/${CONDITION}" \
  --seed 42 \
  --epochs 2 \
  --learning_rate 2e-4 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
