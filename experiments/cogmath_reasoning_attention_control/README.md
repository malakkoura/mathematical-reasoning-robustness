# CogMath Reasoning Attention-Control

This is a separate Qwen3-4B LoRA experiment for testing whether reasoning-augmentation effects depend on attention structure.

The design is a 2 x 3 validation experiment:

- Attention: `causal`, `prompt_bidirectional`
- Training data: `original_only`, `original_plus_dim2`, `original_plus_all`

The prompt-bidirectional condition makes problem/input tokens mutually visible, while answer tokens remain causal and can attend to the full prompt plus previous answer tokens. The prompt boundary is metadata from token-level `prompt_length`; no `<END_OF_INPUT>` marker is placed in model-visible text.

The causal training path deliberately uses the native Transformers `Trainer` pathway and Qwen's standard causal attention. The custom masked loss path is used only for `prompt_bidirectional`.

## Data

No training JSONL is copied or regenerated here. The experiment reuses:

- `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl`
- `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2.jsonl`
- `experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_all.jsonl`
- `experiments/cogmath_augmentation/validation_eval.jsonl`

The held-out test-evaluation JSONL is not accessed by this experiment.

## Causal Reuse

The causal side should reuse the completed budget-control adapters and validation predictions:

- `causal_original_only` -> `budget_original_only`
- `causal_original_plus_dim2` -> `budget_original_plus_dim2`
- `causal_original_plus_all` -> `budget_original_plus_all`

Run static validation with `--require-causal-references` on the cluster before relying on the reuse. It checks source data, row order, hyperparameters, LoRA config, max steps, max length, validation record order, and generation budget.

## Runtime Diagnostics

Before full training, run the runtime diagnostic on the cluster:

```bash
sbatch experiments/cogmath_reasoning_attention_control/slurm/runtime_diagnostics.sh
```

It compares the native budget-control-style causal path and the new attention-control causal path on the same initialized Qwen3-4B LoRA model and same tokenized batch. It also runs a model-level prompt-bidirectional leakage test by altering a future target token and confirming earlier target logits do not change under the actual eager Qwen forward pass.

For validation inference, prompt-bidirectional generation uses a memory-safe cached path: one bidirectional prompt prefill, then single-token causal decoding with `past_key_values`. It does not rebuild a full square prompt+generated 4D mask at every generated token.

Full prompt-bidirectional training is blocked by the training-array script unless:

- causal budget-reference compatibility passes;
- runtime causal equivalence passes;
- abstract mask tests pass;
- model-level future-target leakage passes;
- the prompt-bidirectional backend reports eager attention.

## Fixed Training Config

- Model: `Qwen/Qwen3-4B-Base`
- Seed: `42`
- Max sequence length: `1536`
- Max optimizer steps: `232`
- Learning rate: `2e-4`
- Batch size: `1`
- Gradient accumulation: `16`
- Warmup steps: `20`
- BF16: enabled
- Gradient checkpointing: enabled
- KV cache during training: disabled
- LoRA: rank `8`, alpha `16`, dropout `0.05`, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

Example presentations are reported using the same finite-dataset epoch-boundary accounting as the budget-control experiment. The experimental control is exactly `232` optimizer steps, not a claim that every condition processes exactly `232 * 16` examples.

## Cluster Commands

```bash
cd repository root
python -u experiments/cogmath_reasoning_attention_control/prepare_attention_control.py
python -u experiments/cogmath_reasoning_attention_control/runtime_attention_diagnostics.py \
  --model_name Qwen/Qwen3-4B-Base \
  --train_file experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl \
  --output_path experiments/cogmath_reasoning_attention_control/runtime_diagnostics_report.json \
  --cache_dir $HF_HOME \
  --max_length 1536 \
  --batch_size 1
python -u experiments/cogmath_reasoning_attention_control/static_validation.py \
  --require-causal-references \
  --require-runtime-diagnostics
sbatch experiments/cogmath_reasoning_attention_control/slurm/smoke_attention_control.sh
sbatch experiments/cogmath_reasoning_attention_control/slurm/inference_smoke_cached_prompt_bidir.sh
sbatch experiments/cogmath_reasoning_attention_control/slurm/train_prompt_bidir_array.sh
sbatch experiments/cogmath_reasoning_attention_control/slurm/evaluate_validation_array.sh
sbatch experiments/cogmath_reasoning_attention_control/slurm/analyze_validation.sh
```
