#!/bin/bash
#SBATCH --job-name=cogmath_reason_lora_place_static
#SBATCH --time=00:20:00
#SBATCH --output=experiments/cogmath_reasoning_lora_placement/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_lora_placement/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
mkdir -p experiments/cogmath_reasoning_lora_placement/logs

python -u experiments/cogmath_reasoning_lora_placement/prepare_lora_placement.py
python -u experiments/cogmath_reasoning_lora_placement/static_validation.py --require-existing-references
