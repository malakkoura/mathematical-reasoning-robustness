#!/bin/bash
#SBATCH --job-name=thesis_delib_analyze
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=analysis/final_thesis/deliberation/logs/%x_%j.out
#SBATCH --error=analysis/final_thesis/deliberation/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/final_thesis_deliberation_qwen3_4b
FIRST_PASS_FILE="${OUTPUT_ROOT}/first_pass_traces/first_pass_traces.jsonl"
CAUSAL="${OUTPUT_ROOT}/validation_evaluations/causal_revision/predictions.jsonl"
BIDIR="${OUTPUT_ROOT}/validation_evaluations/bidir_trace_revision/predictions.jsonl"
ANALYSIS_DIR="${OUTPUT_ROOT}/validation_analysis"

mkdir -p analysis/final_thesis/deliberation/logs
mkdir -p "${ANALYSIS_DIR}"

for path in "${FIRST_PASS_FILE}" "${CAUSAL}" "${BIDIR}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done
if [[ -f "${ANALYSIS_DIR}/summary.json" ]]; then
  echo "Refusing to overwrite existing analysis: ${ANALYSIS_DIR}/summary.json" >&2
  exit 1
fi

python -u analysis/final_thesis/deliberation/analyze_deliberation.py \
  --first_pass_file "${FIRST_PASS_FILE}" \
  --causal_predictions "${CAUSAL}" \
  --bidir_predictions "${BIDIR}" \
  --output_dir "${ANALYSIS_DIR}"
