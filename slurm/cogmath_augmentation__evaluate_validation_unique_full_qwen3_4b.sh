#!/bin/bash
#SBATCH --job-name=cogmath_aug_unique_4b_val
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=experiments/cogmath_augmentation/logs/%x_%A_%a.out
#SBATCH --error=experiments/cogmath_augmentation/logs/%x_%A_%a.err

set -euo pipefail

source ${CONDA_SH:?Set CONDA_SH to the path of conda.sh}
conda activate ${CONDA_ENV:-math-reasoning}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
export HF_DATASETS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/datasets
export TRANSFORMERS_CACHE=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}/transformers

CONDITIONS=(
  base_untuned
  original_only
  original_plus_dim1
  original_plus_dim2
  original_plus_dim4
  original_plus_dim6
  original_plus_dim2_dim4
  original_plus_all
)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}/outputs/cogmath_augmentation_qwen3_4b_unique_full

mkdir -p experiments/cogmath_augmentation/logs
mkdir -p "${OUTPUT_ROOT}/validation_evaluations/${CONDITION}"

COMMAND=(
  python -u experiments/cogmath_augmentation/evaluate_model_unique_full.py
  --model_name Qwen/Qwen3-4B-Base
  --condition "${CONDITION}"
  --eval_file experiments/cogmath_augmentation/validation_eval.jsonl
  --output_dir "${OUTPUT_ROOT}/validation_evaluations/${CONDITION}"
  --batch_size 8
  --max_new_tokens 32
  --warmup_generations 3
  --cache_dir ${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}
)

if [[ "${CONDITION}" != "base_untuned" ]]; then
  COMMAND+=(--adapter_dir "${OUTPUT_ROOT}/adapters/${CONDITION}")
fi

"${COMMAND[@]}"
