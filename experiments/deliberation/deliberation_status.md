# Deliberation Experiment Status

Held-out test records were not opened or summarized for this audit.

## Existing Related Code

| Area | Evidence | Reusable? | Status |
| --- | --- | --- | --- |
| Deliberation / second-pass revision | No existing dedicated deliberation, second-pass revision, self-correction, or review pipeline was found under `experiments/`, `analysis/`, `outputs/`, or `results/`. | No direct implementation to reuse. | Missing before this final-thesis add-on. |
| Prompt-bidirectional mask implementation | `experiments/cogmath_reasoning_attention_control/attention_masks.py` implements token-boundary prompt-bidirectional masks and cached decode masks. | Yes. Reused by `deliberation_utils.py`. | Available. |
| Runtime diagnostics for prompt-bidirectional attention | `experiments/cogmath_reasoning_attention_control/runtime_attention_diagnostics.py` and `inference_smoke_attention_control.py`. | Conceptually reusable; the deliberation smoke script adds a focused revision-span leakage check. | Available for attention-control experiments. |
| Qwen3-0.6B full-parameter attention follow-up | `experiments/cogmath_attention_full_ft/` and local validation outputs under `outputs/cogmath_attention_full_ft/evaluations/validation/{causal,prompt_bidirectional}`. Summaries identify `a private cluster model cache for Qwen3-0.6B-Base`. | Partially. It confirms the Qwen3-0.6B follow-up exists, but its local validation outputs are final-answer style, not rich stored reasoning traces. | Completed local validation outputs exist, but not suitable as the requested reasoning-trace source without weakening the design. |
| Qwen3-1.7B attention LoRA experiment | `experiments/cogmath_attention/` and local validation outputs under `outputs/cogmath_attention/evaluations/validation/{causal,prompt_bidirectional}`. | Not used for this deliberation design because the requested current thesis comparison centers on Qwen3-4B reasoning validation traces, and substituting Qwen3-1.7B would change the model family/size. | Existing older output. |
| Qwen3-4B reasoning/budget-control predictions | Expected under `outputs/cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds/...` after jobs `81232` and `81233`. | Yes, once available. These are the intended first-pass reasoning traces for the smallest clean validation diagnostic. | Not present locally; cluster jobs reported submitted by user. |

## Implemented Design

The new validation-only deliberation pipeline freezes one completed first-pass trace per validation record, then evaluates two matched second-pass revision conditions.

1. `first_pass_no_revision`: uses the frozen first-pass extracted answer and correctness directly.
2. `causal_revision`: re-encodes the identical revision prompt and first-pass trace using native causal attention, then greedily generates a revised answer.
3. `bidir_trace_revision`: re-encodes the identical revision prompt and first-pass trace with bidirectional attention only over the fixed prefix, then greedily generates revised answer tokens causally.

The fixed prefix contains:

```text
Original problem
First-pass reasoning trace
First-pass extracted answer
Revision instruction
```

The generated revised answer remains autoregressive. Prompt and first-pass tokens cannot attend to revised-answer tokens. Revised tokens cannot attend to future revised tokens.

## Model Choice

The implementation is configurable, but the default cluster wrappers use:

`Qwen/Qwen3-4B-Base`

with the seed-42 fixed-budget `budget_original_plus_dim2` adapter as the default first-pass/revision checkpoint. This is because the current thesis jobs and first-pass reasoning-trace source are Qwen3-4B budget-control runs. The repository does confirm a Qwen3-0.6B full-parameter attention follow-up exists, but it does not provide suitable stored reasoning traces for this requested deliberation experiment.

## Outputs To Be Created On Cluster

Default output root:

`outputs/final_thesis_deliberation_qwen3_4b`

Expected subdirectories:

| Path | Purpose |
| --- | --- |
| `first_pass_traces/first_pass_traces.jsonl` | Frozen first-pass validation traces. |
| `validation_evaluations/causal_revision/predictions.jsonl` | Causal revision predictions. |
| `validation_evaluations/bidir_trace_revision/predictions.jsonl` | Bidirectional-trace revision predictions. |
| `validation_analysis/` | Summaries, per-dimension metrics, and paired McNemar tests. |

## Missing Before Completion

- The intended first-pass Qwen3-4B validation prediction file must exist after cluster validation finishes.
- The seed/checkpoint chosen for deliberation must be confirmed if not using the default seed-42 `budget_original_plus_dim2` adapter.
- Full model-level smoke and full validation evaluation require Imperial GPU access.
- No held-out test evaluation should be run until the frozen test manifest is confirmed.

