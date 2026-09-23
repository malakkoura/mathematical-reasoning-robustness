#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_pb40
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=experiments/cogmath_reasoning_attention_control/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_attention_control/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_attention_control_qwen3_4b
DIAGNOSTIC_OUTPUT="${OUTPUT_ROOT}/diagnostics/prompt_bidir_original_only_40step"

mkdir -p experiments/cogmath_reasoning_attention_control/logs
mkdir -p "${OUTPUT_ROOT}/diagnostics"

python -u experiments/cogmath_reasoning_attention_control/static_validation.py \
  --require-causal-references \
  --require-runtime-diagnostics

python -u experiments/cogmath_reasoning_attention_control/train_attention_control.py \
  --condition prompt_bidir_original_only \
  --attention_type prompt_bidirectional \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl \
  --output_dir "${DIAGNOSTIC_OUTPUT}" \
  --seed 42 \
  --max_steps 40 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --warmup_steps 20 \
  --logging_steps 5 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
