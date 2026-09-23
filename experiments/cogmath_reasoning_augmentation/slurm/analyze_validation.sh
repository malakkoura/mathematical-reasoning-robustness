#!/bin/bash
#SBATCH --job-name=cogmath_reason_aug_4b_analysis
#SBATCH --time=01:00:00
#SBATCH --output=experiments/cogmath_reasoning_augmentation/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_augmentation/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_augmentation_qwen3_4b

mkdir -p experiments/cogmath_reasoning_augmentation/logs
mkdir -p "${OUTPUT_ROOT}/validation_analysis"

python -u experiments/cogmath_reasoning_augmentation/analyze_reasoning_augmentation.py \
  --results_root "${OUTPUT_ROOT}/validation_evaluations" \
  --output_dir "${OUTPUT_ROOT}/validation_analysis"

