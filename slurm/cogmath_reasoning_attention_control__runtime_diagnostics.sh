#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_runtime
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=experiments/cogmath_reasoning_attention_control/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_attention_control/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

mkdir -p experiments/cogmath_reasoning_attention_control/logs

python -u experiments/cogmath_reasoning_attention_control/prepare_attention_control.py
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
