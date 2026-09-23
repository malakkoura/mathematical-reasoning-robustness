#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_infsmoke
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

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_attention_control_qwen3_4b

mkdir -p experiments/cogmath_reasoning_attention_control/logs
mkdir -p "${OUTPUT_ROOT}/diagnostics"

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

python -u experiments/cogmath_reasoning_attention_control/inference_smoke_attention_control.py \
  --condition prompt_bidir_original_only \
  --model_name Qwen/Qwen3-4B-Base \
  --adapter_dir "${OUTPUT_ROOT}/adapters/prompt_bidir_original_only" \
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl \
  --output_path "${OUTPUT_ROOT}/diagnostics/prompt_bidir_original_only_cached_inference_smoke.json" \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --examples 2 \
  --smoke_new_tokens 96 \
  --full_eval_max_new_tokens 512 \
  --no-stop_on_eos
