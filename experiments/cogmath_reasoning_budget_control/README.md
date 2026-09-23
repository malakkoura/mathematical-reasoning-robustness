# CogMath Qwen3-4B Reasoning Budget-Control Experiment

This is a separate controlled training-budget follow-up to the reasoning-augmentation experiment.

It reuses the validated reasoning-supervised JSONL datasets from `experiments/cogmath_reasoning_augmentation/data/`. It does not regenerate transformed questions, rationales, prior adapters, prior evaluation outputs, or prior analysis outputs.

## Research Question

Under an equal optimizer-update budget, does targeted augmentation improve transformation-specific robustness while heterogeneous augmentation produces negative transfer?

## Output Root

`outputs/cogmath_reasoning_budget_control_qwen3_4b`

Subdirectories:

- `smoke/`
- `adapters/<condition>/`
- `validation_evaluations/<condition>/`
- `validation_analysis/`

## Conditions

- index 0: `budget_original_only` -> `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl`
- index 1: `budget_original_plus_dim2` -> `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2.jsonl`
- index 2: `budget_original_plus_dim4` -> `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim4.jsonl`
- index 3: `budget_original_plus_dim2_dim4` -> `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2_dim4.jsonl`
- index 4: `budget_original_plus_all` -> `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_all.jsonl`

No duplicate `_rep` records are created. The source datasets remain unchanged.

## Fixed Optimization Budget

All five conditions use `max_steps=232` as the stopping criterion.

This controls optimizer updates, not total example or sequence exposure. Transformers chunks each finite-dataset epoch into gradient-accumulation groups, and an epoch-ending incomplete group can still produce an optimizer step with fewer than 16 microbatches. The trainer records observed example presentations, observed effective epochs, observed non-padding token presentations, trainable parameter count, and final training loss.

Nominal full accumulation would be:

`232 optimizer steps * 16 gradient accumulation steps * batch size 1 = 3712 examples`

But the corrected pre-run presentation estimates are:

- `budget_original_only`: 3,692 examples, `3692 / 923 = 4.000` effective epochs
- `budget_original_plus_dim2`: 3,692 examples, `3692 / 1846 = 2.000` effective epochs
- `budget_original_plus_dim4`: 3,692 examples, `3692 / 1846 = 2.000` effective epochs
- `budget_original_plus_dim2_dim4`: 3,697 examples, `3697 / 2769 = 1.335` effective epochs
- `budget_original_plus_all`: 3,712 examples, `3712 / 4609 = 0.805` effective epochs

## Hyperparameters

- Model: `Qwen/Qwen3-4B-Base`
- Seed: 42
- Stopping criterion: `max_steps=232`
- Training max length: 1536
- Learning rate: `2e-4`
- Per-device train batch size: 1
- Gradient accumulation: 16
- BF16 enabled
- Gradient checkpointing enabled
- `use_cache=false`
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Validation

Validation uses the existing 990-record validation file and the same reasoning evaluation protocol:

- reasoning prompt;
- deterministic generation;
- `max_new_tokens=512`;
- same answer extraction and scoring;
- batch size 8 in the SLURM script.

No held-out test-set evaluation script is included.

## Scripts

- `prepare_budget_control.py`: writes metadata reports only.
- `train_budget_control.py`: trains one LoRA adapter with `max_steps`.
- `evaluate_budget_control.py`: validation-only reasoning evaluation wrapper.
- `analyze_budget_control.py`: summary, paired comparisons, Holm-adjusted dimension comparisons, and training-exposure CSVs.
- `static_validation.py`: source-hash, mapping, leakage, hyperparameter, script syntax and path isolation checks.

SLURM:

- `slurm/static_validation.sh`
- `slurm/smoke_max_steps.sh`
- `slurm/train_array.sh`
- `slurm/evaluate_validation_array.sh`
- `slurm/analyze_validation.sh`
