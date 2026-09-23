#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_smoke
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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

mkdir -p experiments/cogmath_reasoning_attention_control/logs
mkdir -p "${OUTPUT_ROOT}/smoke"

python -u experiments/cogmath_reasoning_attention_control/prepare_attention_control.py
python -u experiments/cogmath_reasoning_attention_control/static_validation.py
python -u experiments/cogmath_reasoning_attention_control/test_attention_masks.py
python -u experiments/cogmath_reasoning_attention_control/runtime_attention_diagnostics.py \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl \
  --output_path experiments/cogmath_reasoning_attention_control/runtime_diagnostics_report.json \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --max_length 1536 \
  --batch_size 1 \
  --cached_decode_steps 5
python -u experiments/cogmath_reasoning_attention_control/static_validation.py \
  --require-causal-references \
  --require-runtime-diagnostics

python -u experiments/cogmath_reasoning_attention_control/train_attention_control.py \
  --condition causal_original_only \
  --attention_type causal \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl \
  --output_dir "${OUTPUT_ROOT}/smoke/causal_original_only_native_fix" \
  --seed 42 \
  --max_steps 2 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --smoke_samples 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing

python -u experiments/cogmath_reasoning_attention_control/train_attention_control.py \
  --condition prompt_bidir_original_only \
  --attention_type prompt_bidirectional \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl \
  --output_dir "${OUTPUT_ROOT}/smoke/prompt_bidir_original_only_native_fix" \
  --seed 42 \
  --max_steps 2 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --smoke_samples 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing

python -u experiments/cogmath_reasoning_attention_control/train_attention_control.py \
  --condition prompt_bidir_original_only \
  --attention_type prompt_bidirectional \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl \
  --output_dir "${OUTPUT_ROOT}/smoke/prompt_bidir_original_only_longer_diagnostic" \
  --seed 42 \
  --max_steps 8 \
  --learning_rate 2e-4 \
  --max_length 1536 \
  --smoke_samples 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --bf16 \
  --gradient_checkpointing
