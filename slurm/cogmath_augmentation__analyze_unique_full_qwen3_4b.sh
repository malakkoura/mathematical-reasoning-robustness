#!/bin/bash
#SBATCH --job-name=cogmath_aug_unique_4b_analysis
#SBATCH --partition=gpus24
#SBATCH --time=01:00:00
#SBATCH --output=experiments/cogmath_augmentation/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_augmentation/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_augmentation_qwen3_4b_unique_full

mkdir -p experiments/cogmath_augmentation/logs
mkdir -p "${OUTPUT_ROOT}/validation_analysis" "${OUTPUT_ROOT}/analysis"

python -u experiments/cogmath_augmentation/analyze_unique_full.py \
  --results_root "${OUTPUT_ROOT}/validation_evaluations" \
  --output_dir "${OUTPUT_ROOT}/validation_analysis" \
  --training_data_dir experiments/cogmath_augmentation/data_unique_full

python -u experiments/cogmath_augmentation/plot_experiment.py \
  --analysis_dir "${OUTPUT_ROOT}/validation_analysis" \
  --plot_dir "${OUTPUT_ROOT}/validation_plots"

python -u experiments/cogmath_augmentation/analyze_unique_full.py \
  --results_root "${OUTPUT_ROOT}/evaluations" \
  --output_dir "${OUTPUT_ROOT}/analysis" \
  --training_data_dir experiments/cogmath_augmentation/data_unique_full

python -u experiments/cogmath_augmentation/plot_experiment.py \
  --analysis_dir "${OUTPUT_ROOT}/analysis" \
  --plot_dir "${OUTPUT_ROOT}/plots"
