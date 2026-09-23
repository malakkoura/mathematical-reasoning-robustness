#!/bin/bash
#SBATCH --job-name=cogmath_reason_budget_analysis
#SBATCH --time=00:30:00
#SBATCH --output=experiments/cogmath_reasoning_budget_control/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_budget_control/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_budget_control_qwen3_4b

mkdir -p experiments/cogmath_reasoning_budget_control/logs
mkdir -p "${OUTPUT_ROOT}/validation_analysis"

python -u experiments/cogmath_reasoning_budget_control/analyze_budget_control.py \
  --results_root "${OUTPUT_ROOT}/validation_evaluations" \
  --adapters_root "${OUTPUT_ROOT}/adapters" \
  --output_dir "${OUTPUT_ROOT}/validation_analysis"
