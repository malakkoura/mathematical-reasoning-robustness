#!/usr/bin/env python3
"""Reasoning-prompt validation evaluation for the diagnostic experiment."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning_utils import (
    AUGMENTATION_DIR,
    ALL_REASONING_EVALUATION_CONDITIONS,
    BASE_REASONING_CONDITION,
    BASE_REASONING_CONDITION_512,
    MODEL_NAME,
    VALIDATION_EVAL_FILE,
    make_reasoning_prompt,
)

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import answers_match, extract_answer, read_jsonl, summarize_predictions, write_json, write_jsonl  # noqa: E402


FINAL_MARKER_RE = re.compile(r"final\s+answer\s*:", flags=re.IGNORECASE)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-4B with a reasoning prompt.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--condition", required=True, choices=ALL_REASONING_EVALUATION_CONDITIONS)
    parser.add_argument("--eval_file", type=Path, default=VALIDATION_EVAL_FILE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup_generations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_existing_output", action="store_true")
    return parser.parse_args(argv)


def batches(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]


def ensure_output_is_new(output_dir: Path, allow_existing_output: bool) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    if allow_existing_output:
        return
    if predictions_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {predictions_path}.")


def cuda_sync_if_needed(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def load_model_and_tokenizer(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        device_map="auto",
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()
    return model, tokenizer, torch


def generated_length_and_cap_status(token_ids: Any, eos_token_id: Optional[int], max_new_tokens: int) -> tuple[int, bool]:
    ids = [int(token) for token in token_ids.tolist()]
    if eos_token_id is not None and eos_token_id in ids:
        return ids.index(int(eos_token_id)) + 1, False
    return len(ids), len(ids) >= max_new_tokens


def extract_after_final_marker(text: str) -> tuple[bool, Optional[str]]:
    matches = list(FINAL_MARKER_RE.finditer(text))
    if not matches:
        return False, None
    tail = text[matches[-1].end():]
    return True, extract_answer(tail)


def generate_batch(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    rows: List[Dict[str, Any]],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    prompts = [make_reasoning_prompt(row["question"]) for row in rows]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    cuda_sync_if_needed(torch_module)
    start = time.perf_counter()
    with torch_module.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    cuda_sync_if_needed(torch_module)
    elapsed = time.perf_counter() - start

    prompt_width = encoded["input_ids"].shape[1]
    new_token_ids = generated[:, prompt_width:]
    generated_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    per_example_latency = elapsed / max(len(rows), 1)

    outputs = []
    for row, text, token_ids in zip(rows, generated_texts, new_token_ids):
        marker_produced, extracted = extract_after_final_marker(text)
        generated_count, hit_cap = generated_length_and_cap_status(token_ids, tokenizer.eos_token_id, max_new_tokens)
        outputs.append(
            {
                "condition": row.get("condition"),
                "record_id": row["record_id"],
                "base_id": row["base_id"],
                "dimension": row["dimension"],
                "variant_type": row["variant_type"],
                "question": row["question"],
                "gold_final_answer": row["final_answer"],
                "generated_reasoning": text,
                "generated_text": text,
                "final_answer_marker_produced": marker_produced,
                "missing_final_answer_marker": not marker_produced,
                "hit_max_new_tokens": hit_cap,
                "generated_token_count": generated_count,
                "generated_tokens": generated_count,
                "extracted_answer": extracted,
                "correct": marker_produced and answers_match(extracted, row["final_answer"]),
                "latency_seconds": per_example_latency,
                "tokens_per_second": generated_count / per_example_latency if per_example_latency > 0 else None,
            }
        )
    return outputs


def run_warmup(model: Any, tokenizer: Any, torch_module: Any, rows: List[Dict[str, Any]], count: int, max_new_tokens: int) -> None:
    if count <= 0 or not rows:
        return
    for _ in range(count):
        generate_batch(model, tokenizer, torch_module, [rows[0]], max_new_tokens)


def diagnostic_rates(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    cap_hits = sum(1 for row in predictions if row.get("hit_max_new_tokens"))
    missing_markers = sum(1 for row in predictions if row.get("missing_final_answer_marker"))
    return {
        "cap_hits": cap_hits,
        "cap_hit_rate": cap_hits / total if total else None,
        "missing_final_answer_markers": missing_markers,
        "missing_final_answer_marker_rate": missing_markers / total if total else None,
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.condition not in {BASE_REASONING_CONDITION, BASE_REASONING_CONDITION_512} and not args.adapter_dir:
        raise ValueError(f"{args.condition} requires --adapter_dir.")
    ensure_output_is_new(args.output_dir, args.allow_existing_output)

    rows = read_jsonl(args.eval_file)
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        row["condition"] = args.condition

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, torch_module = load_model_and_tokenizer(args)
    run_warmup(model, tokenizer, torch_module, rows, args.warmup_generations, args.max_new_tokens)

    predictions = []
    for batch in batches(rows, args.batch_size):
        predictions.extend(generate_batch(model, tokenizer, torch_module, batch, args.max_new_tokens))

    summary = summarize_predictions(predictions, args.condition)
    summary.update(
        {
            "model_name": args.model_name,
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "eval_file": str(args.eval_file),
            "prompt_template": "reasoning_utils.REASONING_PROMPT_TEMPLATE",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "warmup_generations": args.warmup_generations,
            "diagnostics": diagnostic_rates(predictions),
        }
    )
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Saved reasoning predictions to {args.output_dir / 'predictions.jsonl'}")
    print(f"Saved reasoning summary to {args.output_dir / 'summary.json'}")


def main(argv: Optional[List[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
