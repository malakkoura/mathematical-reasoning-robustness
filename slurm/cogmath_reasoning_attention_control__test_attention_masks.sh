#!/bin/bash
#SBATCH --job-name=cogmath_reason_attn_masks
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/cogmath_reasoning_attention_control/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_attention_control/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1

mkdir -p experiments/cogmath_reasoning_attention_control/logs
python -u experiments/cogmath_reasoning_attention_control/test_attention_masks.py
