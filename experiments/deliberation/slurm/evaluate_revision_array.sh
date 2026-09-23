#!/bin/bash
#SBATCH --job-name=thesis_delib_eval
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/final_thesis/deliberation/logs/%x_%A_%a.out
#SBATCH --error=analysis/final_thesis/deliberation/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(causal_revision bidir_trace_revision)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/final_thesis_deliberation_qwen3_4b
FIRST_PASS_FILE="${OUTPUT_ROOT}/first_pass_traces/first_pass_traces.jsonl"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Base}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds/seed_42/adapters/budget_original_plus_dim2}"
EVAL_OUTPUT="${OUTPUT_ROOT}/validation_evaluations/${CONDITION}"

mkdir -p analysis/final_thesis/deliberation/logs
mkdir -p "${EVAL_OUTPUT}"

if [[ ! -f "${FIRST_PASS_FILE}" ]]; then
  echo "Missing frozen first-pass traces: ${FIRST_PASS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Missing adapter_config.json in ${ADAPTER_DIR}" >&2
  exit 1
fi
if [[ -f "${EVAL_OUTPUT}/predictions.jsonl" ]]; then
  echo "Refusing to overwrite existing predictions: ${EVAL_OUTPUT}/predictions.jsonl" >&2
  exit 1
fi

python -u analysis/final_thesis/deliberation/evaluate_revision.py \
  --condition "${CONDITION}" \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --first_pass_file "${FIRST_PASS_FILE}" \
  --output_dir "${EVAL_OUTPUT}" \
  --max_new_tokens 128 \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory} \
  --seed 42 \
  --run_model_leakage_check
