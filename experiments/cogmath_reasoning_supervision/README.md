# CogMath Qwen3-4B Reasoning-Supervision Diagnostic

This is a separate diagnostic experiment for Qwen3-4B on CogMath-GSM8K validation data. It does not modify the unique-full direct-answer augmentation experiment, the attention experiment, the LoRA-placement experiment, or their outputs.

The question is whether low absolute GSM8K accuracy is partly caused by the direct-answer-only training and evaluation protocol, and whether explicit GSM8K reasoning supervision improves original accuracy and robustness.

## Data

Source training file:

`experiments/cogmath_augmentation/data_unique_full/train_original_only.jsonl`

New reasoning training file:

`experiments/cogmath_reasoning_supervision/data/train_original_only_reasoning.jsonl`

The new file contains:

- 923 rows
- 923 unique base IDs
- Original GSM8K examples only
- No transformed training examples
- No validation or test base IDs

Each target is built directly from source `answer_raw`; no reasoning is generated or edited.

## Target Format

```text
Reasoning:
<answer_raw before the final GSM8K #### marker>

Final answer: <final_answer>
```

The final `####` marker is removed. The final answer is preserved as `Final answer: <gold>`.

## Training

The new reasoning adapter uses the same direct-answer LoRA hyperparameters wherever applicable:

- Model: `Qwen/Qwen3-4B-Base`
- Epochs: 2
- Learning rate: `2e-4`
- Seed: 42
- Per-device train batch size: 1
- Gradient accumulation steps: 16
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- BF16 enabled
- Gradient checkpointing enabled
- `use_cache=false` during training
- Completion-only loss

Configured max length is 768. The cluster tokenizer check found 2/923 examples exceed 512 tokens, max full sequence length is 554, and no examples exceed 768. The training code refuses to silently truncate.

## Evaluation

Validation only. Do not evaluate the test set for this diagnostic yet.

The reasoning prompt is:

```text
Solve the following maths problem step by step. Finish with exactly one line in this format:
Final answer: <answer>

Question:
{question}

Answer:
```

Generation uses greedy decoding with `max_new_tokens=256`. The scorer extracts the answer only after the final `Final answer:` marker and compares the numerical value to gold. Reasoning text itself is not scored.

Diagnostic validation conditions:

- `base_reasoning_prompt`: untuned Qwen3-4B-Base, reasoning prompt
- `direct_adapter_reasoning_prompt`: existing unique-full `original_only` direct-answer adapter, reasoning prompt
- `reasoning_adapter_reasoning_prompt`: new reasoning-supervised adapter, reasoning prompt
- `direct_adapter_direct_prompt_existing`: existing direct-answer `original_only` validation output, included by analysis without rerunning

## 512-Token Validation Diagnostic

The original reasoning evaluation uses `max_new_tokens=256`. A separate evaluation-only follow-up uses the same validation records, prompt template, scorer, batch size and adapters, but increases only the generation budget to 512.

This diagnostic writes to separate output locations and must not overwrite the 256-token predictions:

- 256-token root: `outputs/cogmath_reasoning_supervision_qwen3_4b/validation_evaluations`
- 512-token root: `outputs/cogmath_reasoning_supervision_qwen3_4b/validation_evaluations_512`
- 512 analysis: `outputs/cogmath_reasoning_supervision_qwen3_4b/validation_analysis_512`

512-token conditions:

- `base_reasoning_prompt_512`: untuned Qwen3-4B-Base, reasoning prompt, `max_new_tokens=512`
- `direct_adapter_reasoning_prompt_512`: existing unique-full `original_only` direct-answer adapter, reasoning prompt, `max_new_tokens=512`
- `reasoning_adapter_reasoning_prompt_512`: existing reasoning-supervised adapter, reasoning prompt, `max_new_tokens=512`

No model is retrained and no adapter is created for the 512-token diagnostic. The test set is not evaluated.

## Outputs

Cluster output root:

`outputs/cogmath_reasoning_supervision_qwen3_4b`

Subdirectories:

- `smoke/`
- `adapters/reasoning_original_only/`
- `validation_evaluations/<condition>/`
- `analysis/validation/`
- `logs/`

## Scripts

- `prepare_reasoning_data.py`: builds the reasoning JSONL and dataset report
- `train_reasoning_lora.py`: trains the new reasoning-supervised adapter
- `evaluate_reasoning.py`: runs reasoning-prompt validation evaluation
- `analyze_reasoning_supervision.py`: creates CSV summaries, paired comparisons, rescues and regressions
- `static_validation.py`: validates data, leakage, target consistency, syntax, tokenizer status and optional cluster artifacts

SLURM scripts:

- `slurm/static_validation.sh`
- `slurm/train_smoke_reasoning.sh`
- `slurm/train_reasoning_original_only.sh`
- `slurm/evaluate_validation_array.sh`
- `slurm/analyze_validation.sh`
- `slurm/evaluate_validation_512_array.sh`
- `slurm/analyze_validation_512.sh`

No test-evaluation SLURM script is included.
