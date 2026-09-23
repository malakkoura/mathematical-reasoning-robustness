#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_analyze
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=experiments/cogmath_reasoning_attention_control/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_attention_control/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_reasoning_attention_control_qwen3_4b

mkdir -p experiments/cogmath_reasoning_attention_control/logs
mkdir -p "${OUTPUT_ROOT}/validation_analysis"

python -u experiments/cogmath_reasoning_attention_control/static_validation.py \
  --require-causal-references \
  --require-runtime-diagnostics
python -u experiments/cogmath_reasoning_attention_control/analyze_attention_control.py \
  --output_dir "${OUTPUT_ROOT}/validation_analysis"
