#!/bin/bash
#SBATCH --job-name=cogmath_reason_4b_analysis
#SBATCH --time=01:00:00
#SBATCH --output=experiments/cogmath_reasoning_supervision/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_supervision/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_supervision_qwen3_4b
UNIQUE_FULL_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_augmentation_qwen3_4b_unique_full

mkdir -p experiments/cogmath_reasoning_supervision/logs
mkdir -p "${OUTPUT_ROOT}/analysis"

python -u experiments/cogmath_reasoning_supervision/analyze_reasoning_supervision.py \
  --results_root "${OUTPUT_ROOT}/validation_evaluations" \
  --direct_reference_dir "${UNIQUE_FULL_ROOT}/validation_evaluations/original_only" \
  --output_dir "${OUTPUT_ROOT}/analysis/validation"

