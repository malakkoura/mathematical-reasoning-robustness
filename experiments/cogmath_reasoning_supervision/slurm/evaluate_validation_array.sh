#!/bin/bash
#SBATCH --job-name=cogmath_reason_4b_val
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=experiments/cogmath_reasoning_supervision/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_reasoning_supervision/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

check_adapter() {
  local adapter_dir="$1"
  if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json in ${adapter_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${adapter_dir}/adapter_model.safetensors" && ! -f "${adapter_dir}/adapter_model.bin" ]]; then
    echo "Missing adapter weights in ${adapter_dir}" >&2
    exit 1
  fi
}

CONDITIONS=(
  base_reasoning_prompt
  direct_adapter_reasoning_prompt
  reasoning_adapter_reasoning_prompt
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_supervision_qwen3_4b
UNIQUE_FULL_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_augmentation_qwen3_4b_unique_full
EVAL_OUTPUT="${OUTPUT_ROOT}/validation_evaluations/${CONDITION}"

mkdir -p experiments/cogmath_reasoning_supervision/logs
mkdir -p "${EVAL_OUTPUT}"

if [[ -f "${EVAL_OUTPUT}/predictions.jsonl" ]]; then
  echo "Refusing to overwrite existing ${EVAL_OUTPUT}/predictions.jsonl" >&2
  exit 1
fi

COMMAND=(
  python -u experiments/cogmath_reasoning_supervision/evaluate_reasoning.py
  --model_name Qwen/Qwen3-4B-Base
  --condition "${CONDITION}"
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl
  --output_dir "${EVAL_OUTPUT}"
  --batch_size 8
  --max_new_tokens 256
  --warmup_generations 1
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
)

if [[ "${CONDITION}" == "direct_adapter_reasoning_prompt" ]]; then
  ADAPTER_DIR="${UNIQUE_FULL_ROOT}/adapters/original_only"
  check_adapter "${ADAPTER_DIR}"
  COMMAND+=(--adapter_dir "${ADAPTER_DIR}")
elif [[ "${CONDITION}" == "reasoning_adapter_reasoning_prompt" ]]; then
  ADAPTER_DIR="${OUTPUT_ROOT}/adapters/reasoning_original_only"
  check_adapter "${ADAPTER_DIR}"
  COMMAND+=(--adapter_dir "${ADAPTER_DIR}")
fi

"${COMMAND[@]}"

