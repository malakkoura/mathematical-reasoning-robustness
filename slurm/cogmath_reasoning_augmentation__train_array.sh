#!/bin/bash
#SBATCH --job-name=cogmath_reason_aug_4b_train
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=experiments/cogmath_reasoning_augmentation/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_reasoning_augmentation/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(
  reasoning_original_plus_dim1
  reasoning_original_plus_dim2
  reasoning_original_plus_dim4
  reasoning_original_plus_dim6
  reasoning_original_plus_dim2_dim4
  reasoning_original_plus_all
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_augmentation_qwen3_4b

mkdir -p experiments/cogmath_reasoning_augmentation/logs
mkdir -p "${OUTPUT_ROOT}/adapters"

python -u experiments/cogmath_reasoning_augmentation/prepare_reasoning_augmentation.py \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}

python -u experiments/cogmath_reasoning_augmentation/static_validation.py \
  --require-tokenizer \
  --require-ready

python -u experiments/cogmath_reasoning_augmentation/train_reasoning_augmentation.py \
  --condition "${CONDITION}" \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file "experiments/cogmath_reasoning_augmentation/data/train_${CONDITION}.jsonl" \
  --output_dir "${OUTPUT_ROOT}/adapters/${CONDITION}" \
  --seed 42 \
  --epochs 2 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
