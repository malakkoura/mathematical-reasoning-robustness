# CogMath Qwen3-4B Reasoning-Augmentation Experiment

This is a separate reasoning-supervised version of the unique-full CogMath augmentation experiment.

It does not modify the direct-answer augmentation experiment, existing reasoning-supervision results, the existing `reasoning_original_only` adapter, the attention experiment, or any test results.

## Status

Preparation is ready after explicitly excluding three audited invalid Dim6 reasoning records. The script materializes all seven reasoning datasets, and records the exclusions in `data/dataset_report.json`.

Materialized datasets:

- `reasoning_original_only`
- `reasoning_original_plus_dim1`
- `reasoning_original_plus_dim2`
- `reasoning_original_plus_dim4`
- `reasoning_original_plus_dim6`
- `reasoning_original_plus_dim2_dim4`
- `reasoning_original_plus_all`

Run `python -u experiments/cogmath_reasoning_augmentation/static_validation.py --require-tokenizer --require-ready` on the cluster before any full training.

## Dim6 Blockers

The audit explicitly excludes these Dim6 records:

- `GSM8K@855_dim6`: rationale says the transformed age setup does not yield an integer solution.
- `GSM8K@1134_dim6`: rationale concludes `5` more weeks, but prepared `final_answer` is `75`.
- `GSM8K@1155_dim6`: rationale says the total cassette time cannot be calculated because the third song length is missing.

No synthetic reasoning is generated, and these records are never silently dropped.

## Configuration

- Model: `Qwen/Qwen3-4B-Base`
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Epochs: 2
- Learning rate: `2e-4`
- Seed: 42
- Per-device train batch size: 1
- Gradient accumulation: 16
- BF16 enabled
- Gradient checkpointing enabled
- `use_cache=false`
- Training max length: 1536, based on the exact Imperial Qwen tokenizer audit: max sequence 1309, 4 examples over 768, 2 over 1024, 0 over 1536
- Validation generation: reasoning prompt, `max_new_tokens=512`

## Output Root

`outputs/cogmath_reasoning_augmentation_qwen3_4b`

Subdirectories:

- `smoke/`
- `adapters/<condition>/`
- `validation_evaluations/<condition>/`
- `validation_analysis/`

The existing `reasoning_original_only` adapter is reused from:

`outputs/cogmath_reasoning_supervision_qwen3_4b/adapters/reasoning_original_only`

## Training Array Mapping

- index 0: `reasoning_original_plus_dim1`
- index 1: `reasoning_original_plus_dim2`
- index 2: `reasoning_original_plus_dim4`
- index 3: `reasoning_original_plus_dim6`
- index 4: `reasoning_original_plus_dim2_dim4`
- index 5: `reasoning_original_plus_all`

The array script exists but starts with a `--require-ready` static validation gate.

## Scripts

- `prepare_reasoning_augmentation.py`: builds safe reasoning datasets and writes Dim6 audit blockers.
- `static_validation.py`: validates counts, leakage, readiness, tokenizer status and script syntax.
- `train_reasoning_augmentation.py`: trains one new LoRA adapter; the worst-case smoke path records exact selected sequence lengths for the longest examples.
- `evaluate_reasoning_augmentation.py`: uses the existing reasoning evaluator with `max_new_tokens=512`.
- `analyze_reasoning_augmentation.py`: writes summary, transfer, paired comparison, generation diagnostic and cross-protocol CSVs.

SLURM:

- `slurm/static_validation.sh`
- `slurm/train_smoke_worst_case.sh`
- `slurm/train_array.sh`
- `slurm/evaluate_validation_array.sh`
- `slurm/analyze_validation.sh`

No test-evaluation script is included.
