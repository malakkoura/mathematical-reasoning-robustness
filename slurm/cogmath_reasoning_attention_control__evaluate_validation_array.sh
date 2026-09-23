#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_eval
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=experiments/cogmath_reasoning_attention_control/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_reasoning_attention_control/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(
  prompt_bidir_original_only
  prompt_bidir_original_plus_dim2
  prompt_bidir_original_plus_all
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_attention_control_qwen3_4b

mkdir -p experiments/cogmath_reasoning_attention_control/logs
mkdir -p "${OUTPUT_ROOT}/validation_evaluations"

python -u experiments/cogmath_reasoning_attention_control/static_validation.py \
  --require-causal-references \
  --require-runtime-diagnostics

python -u experiments/cogmath_reasoning_attention_control/evaluate_attention_control.py \
  --condition "${CONDITION}" \
  --attention_type prompt_bidirectional \
  --model_name Qwen/Qwen3-4B-Base \
  --adapter_dir "${OUTPUT_ROOT}/adapters/${CONDITION}" \
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl \
  --output_dir "${OUTPUT_ROOT}/validation_evaluations/${CONDITION}" \
  --batch_size 1 \
  --max_new_tokens 512 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --seed 42
