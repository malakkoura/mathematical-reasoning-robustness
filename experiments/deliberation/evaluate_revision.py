#!/usr/bin/env python3
"""Evaluate causal or bidirectional-trace second-pass revision on validation traces."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from deliberation_utils import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_ROOT,
    answers_match,
    assert_deliberation_mask_semantics,
    build_4d_attention_mask,
    build_prompt_bidir_decode_mask,
    ensure_new_predictions_path,
    extract_after_final_marker,
    generated_length_and_cap_status,
    make_revision_prompt,
    read_jsonl,
    summarize_revision_predictions,
    write_json,
    write_jsonl,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one deliberation revision condition.")
    parser.add_argument("--condition", required=True, choices=["causal_revision", "bidir_trace_revision"])
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--first_pass_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_existing_output", action="store_true")
    parser.add_argument("--run_model_leakage_check", action="store_true")
    return parser.parse_args(argv)


def cuda_sync_if_needed(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def model_forward_with_optional_cache_position(model: Any, kwargs: Dict[str, Any]):
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "dtype": torch.bfloat16,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "device_map": "auto",
    }
    if args.condition == "bidir_trace_revision":
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, torch


def causal_cached_generate_one(model: Any, tokenizer: Any, torch_module: Any, prompt_ids: List[int], max_new_tokens: int) -> tuple[str, List[int], float]:
    device = next(model.parameters()).device
    input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
    attention_mask = torch_module.ones_like(input_ids, dtype=torch_module.long)
    generated_ids: List[int] = []

    cuda_sync_if_needed(torch_module)
    start = time.perf_counter()
    with torch_module.no_grad():
        prefill = model_forward_with_optional_cache_position(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "cache_position": torch_module.arange(len(prompt_ids), device=device),
            },
        )
        past_key_values = prefill.past_key_values
        next_token = torch_module.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        del prefill
        for step in range(max_new_tokens):
            token_int = int(next_token.item())
            generated_ids.append(token_int)
            if tokenizer.eos_token_id is not None and token_int == int(tokenizer.eos_token_id):
                break
            if step == max_new_tokens - 1:
                break
            key_length = len(prompt_ids) + len(generated_ids)
            decode_attention_mask = torch_module.ones((1, key_length), dtype=torch_module.long, device=device)
            decode = model_forward_with_optional_cache_position(
                model,
                {
                    "input_ids": next_token,
                    "attention_mask": decode_attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": torch_module.tensor([[key_length - 1]], dtype=torch_module.long, device=device),
                    "use_cache": True,
                    "cache_position": torch_module.tensor([key_length - 1], dtype=torch_module.long, device=device),
                },
            )
            past_key_values = decode.past_key_values
            next_token = torch_module.argmax(decode.logits[:, -1, :], dim=-1, keepdim=True)
            del decode
    cuda_sync_if_needed(torch_module)
    elapsed = time.perf_counter() - start
    return tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids, elapsed


def bidir_trace_cached_generate_one(model: Any, tokenizer: Any, torch_module: Any, prompt_ids: List[int], max_new_tokens: int) -> tuple[str, List[int], float]:
    device = next(model.parameters()).device
    input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
    prompt_len = len(prompt_ids)
    generated_ids: List[int] = []

    cuda_sync_if_needed(torch_module)
    start = time.perf_counter()
    with torch_module.no_grad():
        prompt_attention_mask = torch_module.ones_like(input_ids, dtype=torch_module.long)
        prompt_lengths = torch_module.tensor([prompt_len], dtype=torch_module.long, device=device)
        prefill_mask = build_4d_attention_mask(
            prompt_lengths,
            prompt_attention_mask,
            "prompt_bidirectional",
            dtype=next(model.parameters()).dtype,
            device=device,
        )
        prefill = model_forward_with_optional_cache_position(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": prefill_mask,
                "use_cache": True,
                "cache_position": torch_module.arange(prompt_len, device=device),
            },
        )
        past_key_values = prefill.past_key_values
        next_token = torch_module.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        del prefill
        for step in range(max_new_tokens):
            token_int = int(next_token.item())
            generated_ids.append(token_int)
            if tokenizer.eos_token_id is not None and token_int == int(tokenizer.eos_token_id):
                break
            if step == max_new_tokens - 1:
                break
            key_length = prompt_len + len(generated_ids)
            decode_mask = build_prompt_bidir_decode_mask(
                batch_size=1,
                key_length=key_length,
                dtype=next(model.parameters()).dtype,
                device=device,
            )
            decode = model_forward_with_optional_cache_position(
                model,
                {
                    "input_ids": next_token,
                    "attention_mask": decode_mask,
                    "past_key_values": past_key_values,
                    "position_ids": torch_module.tensor([[key_length - 1]], dtype=torch_module.long, device=device),
                    "use_cache": True,
                    "cache_position": torch_module.tensor([key_length - 1], dtype=torch_module.long, device=device),
                },
            )
            past_key_values = decode.past_key_values
            next_token = torch_module.argmax(decode.logits[:, -1, :], dim=-1, keepdim=True)
            del decode
    cuda_sync_if_needed(torch_module)
    elapsed = time.perf_counter() - start
    return tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids, elapsed


def prediction_record(row: Dict[str, Any], condition: str, revision_prompt: str, generated_text: str, generated_ids: List[int], elapsed: float, tokenizer: Any, max_new_tokens: int) -> Dict[str, Any]:
    marker_present, extracted = extract_after_final_marker(generated_text)
    generated_count, cap_hit = generated_length_and_cap_status(generated_ids, tokenizer.eos_token_id, max_new_tokens)
    correct = marker_present and answers_match(extracted, row["gold_final_answer"])
    return {
        "condition": condition,
        "record_id": row["record_id"],
        "base_id": row["base_id"],
        "split": row.get("split"),
        "dimension": row.get("dimension"),
        "variant_type": row.get("variant_type"),
        "question": row["question"],
        "gold_final_answer": row["gold_final_answer"],
        "prompt": revision_prompt,
        "first_pass_trace": row["first_pass_trace"],
        "first_pass_extracted_answer": row.get("first_pass_extracted_answer"),
        "first_pass_correct": row.get("first_pass_correct"),
        "first_pass_generated_token_count": row.get("first_pass_generated_token_count"),
        "first_pass_hit_cap": row.get("first_pass_hit_cap"),
        "revised_output": generated_text,
        "generated_text": generated_text,
        "final_answer_marker_present": marker_present,
        "extracted_revised_answer": extracted,
        "extracted_answer": extracted,
        "correct": correct,
        "generated_token_count": generated_count,
        "hit_max_revision_tokens": cap_hit,
        "latency_seconds": elapsed,
    }


def model_level_future_leakage_check(model: Any, tokenizer: Any, torch_module: Any, row: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    """Small model-level check: changing a future generated token must not affect earlier revision logits.

    This uses a forced prefix under the bidirectional-trace mask. The earlier
    checked position is inside the generated/revision span, while the altered
    token is a later generated token.
    """
    prompt = make_revision_prompt(row)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    forced_tail = tokenizer("Final answer: 12345", add_special_tokens=False)["input_ids"]
    if len(forced_tail) < 4:
        raise RuntimeError("Unexpected tokenizer output for forced leakage check.")
    device = next(model.parameters()).device
    full_a = prompt_ids + forced_tail
    full_b = prompt_ids + forced_tail[:]
    full_b[-1] = (full_b[-1] + 1) % int(getattr(model.config, "vocab_size", max(full_b[-1] + 2, 10)))
    attention = torch_module.ones((1, len(full_a)), dtype=torch_module.long, device=device)
    prompt_lengths = torch_module.tensor([len(prompt_ids)], dtype=torch_module.long, device=device)
    mask = build_4d_attention_mask(prompt_lengths, attention, "prompt_bidirectional", dtype=next(model.parameters()).dtype, device=device)
    ids_a = torch_module.tensor([full_a], dtype=torch_module.long, device=device)
    ids_b = torch_module.tensor([full_b], dtype=torch_module.long, device=device)
    checked_index = len(prompt_ids) + 1
    with torch_module.no_grad():
        logits_a = model(input_ids=ids_a, attention_mask=mask, use_cache=False).logits[:, checked_index, :]
        logits_b = model(input_ids=ids_b, attention_mask=mask, use_cache=False).logits[:, checked_index, :]
    diff = torch_module.max(torch_module.abs(logits_a.float() - logits_b.float())).item()
    if diff > 1e-5:
        raise RuntimeError(f"Future revision-token leakage check failed; max logit delta={diff}")
    return {"future_token_leakage_max_abs_delta": diff, "passed": True}


def evaluate(args: argparse.Namespace) -> None:
    ensure_new_predictions_path(args.output_dir, args.allow_existing_output)
    rows = read_jsonl(args.first_pass_file)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No first-pass rows to evaluate.")
    assert_deliberation_mask_semantics(prefix_len=7, generated_len=5, pad_len=3)

    model, tokenizer, torch_module = load_model_and_tokenizer(args)
    if args.run_model_leakage_check:
        leakage = model_level_future_leakage_check(model, tokenizer, torch_module, rows[0], args.max_new_tokens)
    else:
        leakage = {"skipped": True, "reason": "not requested"}

    predictions = []
    for index, row in enumerate(rows, start=1):
        prompt = make_revision_prompt(row)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if args.condition == "causal_revision":
            generated_text, generated_ids, elapsed = causal_cached_generate_one(model, tokenizer, torch_module, prompt_ids, args.max_new_tokens)
        else:
            generated_text, generated_ids, elapsed = bidir_trace_cached_generate_one(model, tokenizer, torch_module, prompt_ids, args.max_new_tokens)
        predictions.append(prediction_record(row, args.condition, prompt, generated_text, generated_ids, elapsed, tokenizer, args.max_new_tokens))
        if index % 25 == 0:
            print(f"{args.condition}: evaluated {index}/{len(rows)} records", flush=True)

    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    summary = summarize_revision_predictions(predictions, args.condition)
    summary.update(
        {
            "model_name": args.model_name,
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "first_pass_file": str(args.first_pass_file),
            "max_new_tokens": args.max_new_tokens,
            "decoding": "manual greedy cached decode; identical revision prompt/token IDs across causal and bidirectional-trace conditions",
            "attention_semantics": {
                "causal_revision": "native causal attention over fixed review prefix and generated revision",
                "bidir_trace_revision": "bidirectional attention over fixed review prefix; autoregressive causal revision tokens",
            },
            "model_level_future_leakage_check": leakage,
            "test_set_used": False,
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(f"Wrote {len(predictions)} predictions to {args.output_dir / 'predictions.jsonl'}")


def main(argv: Optional[List[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
