# Controlled CogMath GSM8K Augmentation Dataset

This directory contains the dataset-preparation stage for a controlled
CogMath GSM8K augmentation experiment using an internal grouped split.

No model training or model evaluation is performed here.

## Source

- Input: `data/CogMath/GSM8K.json`
- Base problem key: top-level CogMath key such as `GSM8K@0`
- Split label: `internal_grouped_split`
- Seed: `42`

## Conditions

- A: `untuned_baseline` — no training dataset
- B: `original_only`
- C: `original_plus_dim1`
- D: `original_plus_dim2`
- E: `original_plus_dim4`
- F: `original_plus_dim6`
- G: `original_plus_all`

The assigned split contains 923 train base IDs, 198
validation base IDs, and 198 test base IDs. The controlled
fine-tuning subset uses 920 of the assigned train base IDs:
the same 920 IDs appear in every training condition, with
two rows per eligible base ID, for 1840 rows per file.

## Generated Files

- `splits.json`: 1319 rows/entries
- `conditions.json`: 7 rows/entries
- `validation_eval.jsonl`: 990 rows/entries
- `test_eval.jsonl`: 990 rows/entries
- `train_original_only.jsonl`: 1840 rows/entries
- `train_original_plus_dim1.jsonl`: 1840 rows/entries
- `train_original_plus_dim2.jsonl`: 1840 rows/entries
- `train_original_plus_dim4.jsonl`: 1840 rows/entries
- `train_original_plus_dim6.jsonl`: 1840 rows/entries
- `train_original_plus_all.jsonl`: 1840 rows/entries

## Notes

- Split is by `base_id`; variants for a base problem never cross splits.
- The split is independent of variant availability: all IDs are sorted by the
  numeric suffix in `GSM8K@N`, shuffled with seed 42, and assigned
  923/198/198 before any training eligibility filtering.
- `splits.json` preserves all 923 assigned train IDs. Three assigned
  train IDs lacking Dimension 6 are excluded only from the controlled
  fine-tuning subset and are documented in `dataset_report.json`.
- Dimension 6 uses the variant-specific `inquiry answer`; the original answer is never assigned to Dimension 6.
- Training rows include `training_target` set to `Final answer: <final_answer>`.

## Model Configurations

The original Qwen3-1.7B configuration is preserved:

- Model: `Qwen/Qwen3-1.7B-Base`
- Cluster output root: `outputs/cogmath_augmentation`
- Existing SLURM scripts: `slurm/train_array.sh`, `slurm/train_large_array.sh`,
  `slurm/evaluate_array.sh`, `slurm/evaluate_validation_array.sh`,
  `slurm/train_smoke.sh`

Qwen3-4B is added as a separate configuration, not a replacement:

- Model: `Qwen/Qwen3-4B-Base`
- Cluster output root: `outputs/cogmath_augmentation_qwen3_4b`
- Config registry: `model_configs.json`
- SLURM scripts: `slurm/train_array_qwen3_4b.sh`,
  `slurm/train_large_array_qwen3_4b.sh`,
  `slurm/evaluate_array_qwen3_4b.sh`,
  `slurm/evaluate_validation_array_qwen3_4b.sh`,
  `slurm/train_smoke_qwen3_4b.sh`

The Qwen3-4B augmentation run uses the same prepared augmentation datasets as
the 1.7B run. It remains separate from the CogMath attention intervention and
does not use prompt-bidirectional attention.

A separate Qwen3-4B full-parameter smoke test is provided only to test whether
one optimizer step can fit on a 24 GB RTX 3090. It does not use PEFT or LoRA,
does not replace the LoRA augmentation experiment, uses 1 sample by default,
and writes only to:

`outputs/cogmath_augmentation_qwen3_4b_full_ft_smoke`

Run it with:

```bash
sbatch experiments/cogmath_augmentation/slurm/full_ft_smoke_qwen3_4b.sh
```

## Unique-Full Qwen3-4B Design

The unique-full setup is the redesign for future Qwen3-4B augmentation
analysis. It replaces the previous balanced, oversampled, repeated-original,
large-control, and full-exposure dataset designs for future comparisons. Do not
mix unique-full outputs with the older augmentation outputs.

- Model: `Qwen/Qwen3-4B-Base`
- Data directory: `experiments/cogmath_augmentation/data_unique_full`
- Output root: `outputs/cogmath_augmentation_qwen3_4b_unique_full`
- Preparation script: `prepare_unique_full.py`
- Static validation: `static_validation_unique_full.py`
- Training wrapper: `train_lora_unique_full.py`
- Evaluation wrapper: `evaluate_model_unique_full.py`
- Analysis script: `analyze_unique_full.py`

Unique-full training conditions:

- `original_only`
- `original_plus_dim1`
- `original_plus_dim2`
- `original_plus_dim4`
- `original_plus_dim6`
- `original_plus_dim2_dim4`
- `original_plus_all`

Every condition includes all valid unique source records for its listed forms.
Original examples are not repeated, no dimension is oversampled, and dataset
sizes are intentionally unequal. The analysis tables report training rows,
unique base IDs, original examples, and per-dimension augmentation examples so
comparisons are interpreted as the practical effect of adding all available
augmentation data.
