# CogMath Qwen3-4B Reasoning LoRA-Placement Experiment

This is a separate placement-at-fixed-rank ablation motivated by the controlled-budget Dim2 gain.

It does not regenerate datasets. It uses:

- `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl`
- `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2.jsonl`

It trains eight new adapters and reuses the two completed all-linear budget-control references:

- `budget_original_only` as `all_linear_original_only`
- `budget_original_plus_dim2` as `all_linear_plus_dim2`

## Output Root

`outputs/cogmath_reasoning_lora_placement_qwen3_4b`

Budget-control reference root:

`outputs/cogmath_reasoning_budget_control_qwen3_4b`

## Placements

- `q_only`: `q_proj`
- `qv`: `q_proj`, `v_proj`
- `attention_all`: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `mlp_only`: `gate_proj`, `up_proj`, `down_proj`
- `all_linear`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Training Array

- 0: `q_only_original_only`
- 1: `q_only_plus_dim2`
- 2: `qv_original_only`
- 3: `qv_plus_dim2`
- 4: `attention_all_original_only`
- 5: `attention_all_plus_dim2`
- 6: `mlp_only_original_only`
- 7: `mlp_only_plus_dim2`

All-linear adapters are not retrained.

## Fixed Configuration

- Model: `Qwen/Qwen3-4B-Base`
- `max_steps=232`
- `max_length=1536`
- seed 42
- learning rate `2e-4`
- batch size 1
- gradient accumulation 16
- LoRA rank 8, alpha 16, dropout 0.05
- BF16
- gradient checkpointing
- `use_cache=false`
- validation `max_new_tokens=512`

## Parameter Count Note

Rank is fixed at 8 for all placements. Target-module sets differ, so placement and total adapter capacity co-vary. Analysis reports trainable parameter count and percentage for every placement. Local reports include analytic estimates from the Qwen3-4B-Base config; cluster training writes exact PEFT `parameter_report.json` files.

Analytic trainable parameter estimates:

- `q_only`: 1,916,928
- `qv`: 2,949,120
- `attention_all`: 5,898,240
- `mlp_only`: 10,616,832
- `all_linear`: 16,515,072

## Validation and Analysis

The eight new adapters are evaluated on the existing 990 validation records. Analysis also reads the two all-linear budget-control validation prediction files without rerunning them.

Primary estimand:

`Dim2 augmentation gain = Dim2_accuracy(plus_dim2) - Dim2_accuracy(original_only)`

Bootstrap confidence intervals for cross-placement gain differences are not implemented; cross-placement gain differences should be treated as descriptive unless supported by the paired plus-Dim2 McNemar comparisons.

No held-out test evaluation script is included.
