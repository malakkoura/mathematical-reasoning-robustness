#!/bin/bash
#SBATCH --job-name=cogmath_reason_lora_place_val
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-7
#SBATCH --output=experiments/cogmath_reasoning_lora_placement/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_reasoning_lora_placement/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(
  q_only_original_only
  q_only_plus_dim2
  qv_original_only
  qv_plus_dim2
  attention_all_original_only
  attention_all_plus_dim2
  mlp_only_original_only
  mlp_only_plus_dim2
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_lora_placement_qwen3_4b
ADAPTER_DIR="${OUTPUT_ROOT}/adapters/${CONDITION}"
EVAL_OUTPUT="${OUTPUT_ROOT}/validation_evaluations/${CONDITION}"

mkdir -p experiments/cogmath_reasoning_lora_placement/logs
mkdir -p "${EVAL_OUTPUT}"

if [[ -f "${EVAL_OUTPUT}/predictions.jsonl" ]]; then
  echo "Refusing to overwrite existing ${EVAL_OUTPUT}/predictions.jsonl" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Missing adapter_config.json in ${ADAPTER_DIR}" >&2
  exit 1
fi

python -u experiments/cogmath_reasoning_lora_placement/evaluate_lora_placement.py \
  --model_name Qwen/Qwen3-4B-Base \
  --adapter_dir "${ADAPTER_DIR}" \
  --condition "${CONDITION}" \
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl \
  --output_dir "${EVAL_OUTPUT}" \
  --batch_size 8 \
  --max_new_tokens 512 \
  --warmup_generations 1 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
