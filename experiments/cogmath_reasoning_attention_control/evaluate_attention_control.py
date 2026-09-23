#!/usr/bin/env python3
"""Evaluate reasoning attention-control adapters on validation records."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from attention_control_utils import (
    ALL_CONDITIONS,
    ATTENTION_TYPE_BY_CONDITION,
    AUGMENTATION_DIR,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    VALIDATION_EVAL_FILE,
    make_reasoning_prompt,
)
from attention_masks import build_4d_attention_mask, build_prompt_bidir_decode_mask

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import answers_match, extract_answer, read_jsonl, summarize_predictions, write_json, write_jsonl  # noqa: E402


FINAL_MARKER_RE = re.compile(r"final\s+answer\s*:", flags=re.IGNORECASE)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one reasoning attention-control condition.")
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--attention_type", choices=["causal", "prompt_bidirectional"], default=None)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, default=VALIDATION_EVAL_FILE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--warmup_generations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_existing_output", action="store_true")
    return parser.parse_args(argv)


def ensure_output_is_new(output_dir: Path, allow_existing_output: bool) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    if not allow_existing_output and predictions_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {predictions_path}.")


def cuda_sync_if_needed(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def load_model_and_tokenizer(args: argparse.Namespace, attention_type: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if attention_type == "causal" else "right"

    model_kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "device_map": "auto",
    }
    if attention_type == "prompt_bidirectional":
        model_kwargs["attn_implementation"] = "eager"
    base = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model.config.use_cache = True
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
    return True, extract_answer(text[matches[-1].end():])


def causal_generate_batch(model: Any, tokenizer: Any, torch_module: Any, rows: List[Dict[str, Any]], max_new_tokens: int) -> List[Dict[str, Any]]:
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
    texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    latency = elapsed / max(len(rows), 1)
    return [prediction_record(row, text, token_ids, latency, tokenizer, max_new_tokens) for row, text, token_ids in zip(rows, texts, new_token_ids)]


def model_forward_with_optional_cache_position(model: Any, kwargs: Dict[str, Any]):
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def prompt_bidir_generate_one(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    row: Dict[str, Any],
    max_new_tokens: int,
    stop_on_eos: bool = True,
) -> Dict[str, Any]:
    prompt = make_reasoning_prompt(row["question"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
    prompt_length = len(prompt_ids)
    generated_ids: List[int] = []

    cuda_sync_if_needed(torch_module)
    start = time.perf_counter()
    with torch_module.no_grad():
        prompt_attention_mask = torch_module.ones_like(input_ids, dtype=torch_module.long)
        prompt_lengths = torch_module.tensor([prompt_length], dtype=torch_module.long, device=device)
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
                "cache_position": torch_module.arange(prompt_length, device=device),
            },
        )
        past_key_values = prefill.past_key_values
        next_token = torch_module.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        del prefill

        for step in range(max_new_tokens):
            token_int = int(next_token.item())
            generated_ids.append(token_int)
            if stop_on_eos and tokenizer.eos_token_id is not None and token_int == tokenizer.eos_token_id:
                break
            if step == max_new_tokens - 1:
                break

            key_length = prompt_length + len(generated_ids)
            decode_mask = build_prompt_bidir_decode_mask(
                batch_size=1,
                key_length=key_length,
                dtype=next(model.parameters()).dtype,
                device=device,
            )
            position_ids = torch_module.tensor([[key_length - 1]], dtype=torch_module.long, device=device)
            decode = model_forward_with_optional_cache_position(
                model,
                {
                    "input_ids": next_token,
                    "attention_mask": decode_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                    "use_cache": True,
                    "cache_position": torch_module.tensor([key_length - 1], dtype=torch_module.long, device=device),
                },
            )
            past_key_values = decode.past_key_values
            next_token = torch_module.argmax(decode.logits[:, -1, :], dim=-1, keepdim=True)
            del decode
    cuda_sync_if_needed(torch_module)
    elapsed = time.perf_counter() - start

    token_tensor = torch_module.tensor(generated_ids, dtype=torch_module.long)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    peak_allocated = torch_module.cuda.max_memory_allocated() if torch_module.cuda.is_available() else None
    peak_reserved = torch_module.cuda.max_memory_reserved() if torch_module.cuda.is_available() else None
    return prediction_record(
        row,
        text,
        token_tensor,
        elapsed,
        tokenizer,
        max_new_tokens,
        prompt_length=prompt_length,
        manual_decoding=True,
        generation_mode="prompt_bidirectional_prefill_cached_causal_decode",
        peak_cuda_memory_allocated_bytes=peak_allocated,
        peak_cuda_memory_reserved_bytes=peak_reserved,
    )


def prediction_record(
    row: Dict[str, Any],
    text: str,
    token_ids: Any,
    latency: float,
    tokenizer: Any,
    max_new_tokens: int,
    prompt_length: Optional[int] = None,
    manual_decoding: bool = False,
    generation_mode: Optional[str] = None,
    peak_cuda_memory_allocated_bytes: Optional[int] = None,
    peak_cuda_memory_reserved_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    marker_produced, extracted = extract_after_final_marker(text)
    generated_count, hit_cap = generated_length_and_cap_status(token_ids, tokenizer.eos_token_id, max_new_tokens)
    return {
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
        "latency_seconds": latency,
        "tokens_per_second": generated_count / latency if latency > 0 else None,
        "prompt_length": prompt_length,
        "manual_prompt_bidirectional_decoding": manual_decoding,
        "generation_mode": generation_mode,
        "boundary_marker_visible": False,
        "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
        "peak_cuda_memory_reserved_bytes": peak_cuda_memory_reserved_bytes,
    }


def diagnostic_rates(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    cap_hits = sum(1 for row in predictions if row.get("hit_max_new_tokens"))
    missing = sum(1 for row in predictions if row.get("missing_final_answer_marker"))
    return {
        "cap_hits": cap_hits,
        "cap_hit_rate": cap_hits / total if total else None,
        "missing_final_answer_markers": missing,
        "missing_final_answer_marker_rate": missing / total if total else None,
    }


def evaluate(args: argparse.Namespace) -> None:
    attention_type = args.attention_type or ATTENTION_TYPE_BY_CONDITION[args.condition]
    if attention_type != ATTENTION_TYPE_BY_CONDITION[args.condition]:
        raise ValueError(f"{args.condition} requires attention_type={ATTENTION_TYPE_BY_CONDITION[args.condition]}, got {attention_type}.")
    ensure_output_is_new(args.output_dir, args.allow_existing_output)

    rows = read_jsonl(args.eval_file)
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        row["condition"] = args.condition

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, torch_module = load_model_and_tokenizer(args, attention_type)

    predictions = []
    if rows and args.warmup_generations > 0:
        if attention_type == "causal":
            causal_generate_batch(model, tokenizer, torch_module, [rows[0]], args.max_new_tokens)
        else:
            if torch_module.cuda.is_available():
                torch_module.cuda.reset_peak_memory_stats()
            prompt_bidir_generate_one(model, tokenizer, torch_module, rows[0], args.max_new_tokens)

    if attention_type == "causal":
        for index in range(0, len(rows), args.batch_size):
            predictions.extend(causal_generate_batch(model, tokenizer, torch_module, rows[index:index + args.batch_size], args.max_new_tokens))
    else:
        for row in rows:
            if torch_module.cuda.is_available():
                torch_module.cuda.reset_peak_memory_stats()
            predictions.append(prompt_bidir_generate_one(model, tokenizer, torch_module, row, args.max_new_tokens))

    summary = summarize_predictions(predictions, args.condition)
    summary.update(
        {
            "model_name": args.model_name,
            "adapter_dir": str(args.adapter_dir),
            "eval_file": str(args.eval_file),
            "attention_type": attention_type,
            "prompt_template": "attention_control_utils.REASONING_PROMPT_TEMPLATE",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "warmup_generations": args.warmup_generations,
            "decoding": "prompt-bidirectional prompt prefill once, then cached single-query causal greedy decoding" if attention_type == "prompt_bidirectional" else "transformers.generate greedy causal",
            "diagnostics": diagnostic_rates(predictions),
        }
    )
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Saved predictions to {args.output_dir / 'predictions.jsonl'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")


def main(argv: Optional[List[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
