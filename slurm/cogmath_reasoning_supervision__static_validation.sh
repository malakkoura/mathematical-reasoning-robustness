#!/bin/bash
#SBATCH --job-name=cogmath_reason_4b_static
#SBATCH --time=00:30:00
#SBATCH --output=experiments/cogmath_reasoning_supervision/logs/%x_%j.out
#SBATCH --error=experiments/cogmath_reasoning_supervision/logs/%x_%j.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

mkdir -p experiments/cogmath_reasoning_supervision/logs

python -u experiments/cogmath_reasoning_supervision/prepare_reasoning_data.py \
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}

python -u experiments/cogmath_reasoning_supervision/static_validation.py \
  --require-tokenizer

