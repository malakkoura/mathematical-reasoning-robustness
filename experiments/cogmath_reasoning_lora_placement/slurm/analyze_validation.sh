#!/bin/bash
#SBATCH --job-name=cogmath_reason_lora_place_analysis
#SBATCH --time=00:30:00
#SBATCH --output=experiments/cogmath_reasoning_lora_placement/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_lora_placement/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_lora_placement_qwen3_4b
mkdir -p experiments/cogmath_reasoning_lora_placement/logs
mkdir -p "${OUTPUT_ROOT}/validation_analysis"

python -u experiments/cogmath_reasoning_lora_placement/static_validation.py --require-existing-references
python -u experiments/cogmath_reasoning_lora_placement/analyze_lora_placement.py \
  --output_dir "${OUTPUT_ROOT}/validation_analysis"
