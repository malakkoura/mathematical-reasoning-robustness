#!/usr/bin/env python3
"""Cluster runtime diagnostics for attention-control correctness."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from attention_control_utils import (
    EXPERIMENT_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    SEED,
    TRAIN_MAX_LENGTH,
    make_reasoning_prompt,
    source_training_path_for_condition,
    write_json,
)
from attention_masks import build_4d_attention_mask, build_model_attention_mask, build_prompt_bidir_decode_mask, build_visibility_matrix
from train_attention_control import parameter_report, tokenize as attention_tokenize

sys.path.insert(0, str(EXPERIMENT_DIR.parents[1] / "experiments" / "cogmath_reasoning_budget_control"))
from train_budget_control import tokenize as budget_tokenize  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen runtime equivalence and leakage diagnostics.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", type=Path, default=source_training_path_for_condition("causal_original_only"))
    parser.add_argument("--output_path", type=Path, default=EXPERIMENT_DIR / "runtime_diagnostics_report.json")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_length", type=int, default=TRAIN_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cached_decode_steps", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rows(path: Path, seed: int, batch_size: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return rows[:batch_size]


def tensor_batch(torch_module: Any, encoded_rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    return {key: torch_module.tensor([row[key] for row in encoded_rows], dtype=torch_module.long) for key in keys}


def load_lora_model(args: argparse.Namespace, attention_type: str, force_eager: bool = False):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
    }
    if attention_type == "prompt_bidirectional" or force_eager:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = True
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES,
        ),
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer, torch


def supervised_positions(labels: Any, attention_mask: Any) -> List[int]:
    return [
        index
        for index, (label, mask) in enumerate(zip(labels.tolist(), attention_mask.tolist()))
        if int(mask) == 1 and int(label) != -100
    ]


def assert_true(name: str, value: bool, details: Any, failures: List[Dict[str, Any]]) -> None:
    if not value:
        failures.append({"check": name, "details": details})


def model_forward_with_optional_cache_position(model: Any, kwargs: Dict[str, Any], require_cache_position: bool = False):
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" not in str(exc):
            raise
        if require_cache_position:
            raise RuntimeError("This runtime diagnostic requires the model forward path to accept cache_position.") from exc
        kwargs = dict(kwargs)
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def causal_equivalence(args: argparse.Namespace) -> Dict[str, Any]:
    model, tokenizer, torch_module = load_lora_model(args, "causal")
    rows = load_rows(args.train_file, args.seed, args.batch_size)
    budget_encoded = [budget_tokenize(tokenizer, row["prompt"], row["training_target"], args.max_length, row["record_id"]) for row in rows]
    attention_encoded = [attention_tokenize(tokenizer, row["prompt"], row["training_target"], args.max_length, row["record_id"], include_prompt_length=True) for row in rows]
    failures: List[Dict[str, Any]] = []

    for index, (budget_row, attention_row) in enumerate(zip(budget_encoded, attention_encoded)):
        for key in ["input_ids", "labels", "attention_mask"]:
            assert_true(f"identical_{key}_{index}", budget_row[key] == attention_row[key], {"record_id": rows[index]["record_id"], "key": key}, failures)
        positions = supervised_positions(
            torch_module.tensor(attention_row["labels"], dtype=torch_module.long),
            torch_module.tensor(attention_row["attention_mask"], dtype=torch_module.long),
        )
        assert_true(
            f"prompt_boundary_matches_first_supervised_token_{index}",
            positions and attention_row["prompt_lengths"] == positions[0],
            {"record_id": rows[index]["record_id"], "prompt_lengths": attention_row["prompt_lengths"], "first_supervised": positions[0] if positions else None},
            failures,
        )

    batch = tensor_batch(torch_module, attention_encoded, ["input_ids", "labels", "attention_mask", "prompt_lengths"])
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    parameter_stats = parameter_report(model)
    native_mask = batch["attention_mask"]
    control_mask = build_model_attention_mask(batch["prompt_lengths"], batch["attention_mask"], "causal", dtype=next(model.parameters()).dtype, device=device)
    assert_true("causal_mask_is_native_2d_padding_mask", tuple(control_mask.shape) == tuple(native_mask.shape) and torch_module.equal(control_mask, native_mask), {"native_shape": tuple(native_mask.shape), "control_shape": tuple(control_mask.shape)}, failures)

    with torch_module.no_grad():
        native = model(input_ids=batch["input_ids"], attention_mask=native_mask, labels=batch["labels"])
        control = model(input_ids=batch["input_ids"], attention_mask=control_mask, labels=batch["labels"])

    loss_delta = float((native.loss - control.loss).abs().detach().cpu().item())
    max_logit_delta = float((native.logits - control.logits).abs().max().detach().cpu().item())
    assert_true("causal_loss_matches_native_pathway", loss_delta <= args.atol, {"loss_delta": loss_delta, "native_loss": float(native.loss.detach().cpu()), "control_loss": float(control.loss.detach().cpu())}, failures)
    assert_true("causal_logits_match_native_pathway", max_logit_delta <= args.atol, {"max_logit_delta": max_logit_delta}, failures)

    return {
        "passed": not failures,
        "failures": failures,
        "record_ids": [row["record_id"] for row in rows],
        "supervised_token_counts": [len(supervised_positions(torch_module.tensor(row["labels"]), torch_module.tensor(row["attention_mask"]))) for row in attention_encoded],
        "lora_target_modules": LORA_TARGET_MODULES,
        "parameter_report": parameter_stats,
        "native_loss": float(native.loss.detach().cpu()),
        "control_loss": float(control.loss.detach().cpu()),
        "loss_delta": loss_delta,
        "max_logit_delta": max_logit_delta,
        "attention_backend": getattr(model.base_model.model.config, "_attn_implementation", None),
        "position_ids": "not supplied; Qwen/Transformers derives native position_ids identically for both causal paths",
    }


def prompt_bidirectional_model_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    model, tokenizer, torch_module = load_lora_model(args, "prompt_bidirectional")
    rows = load_rows(args.train_file, args.seed, 1)
    encoded = attention_tokenize(tokenizer, rows[0]["prompt"], rows[0]["training_target"], args.max_length, rows[0]["record_id"], include_prompt_length=True)
    batch = tensor_batch(torch_module, [encoded], ["input_ids", "labels", "attention_mask", "prompt_lengths"])
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    positions = supervised_positions(batch["labels"][0].detach().cpu(), batch["attention_mask"][0].detach().cpu())
    failures: List[Dict[str, Any]] = []
    assert_true("enough_supervised_tokens_for_future_leakage_test", len(positions) >= 3, {"positions": positions[:10], "count": len(positions)}, failures)

    dtype = next(model.parameters()).dtype
    mask = build_4d_attention_mask(batch["prompt_lengths"], batch["attention_mask"], "prompt_bidirectional", dtype=dtype, device=device)
    visible = build_visibility_matrix(batch["prompt_lengths"], batch["attention_mask"], "prompt_bidirectional")
    prompt_len = int(batch["prompt_lengths"][0].item())
    valid_len = int(batch["attention_mask"][0].sum().item())
    future_position = positions[-1] if positions else valid_len - 1
    earlier_positions = positions[:-1][-3:] if len(positions) >= 4 else positions[:-1]
    changed_input_ids = batch["input_ids"].clone()
    vocab_size = int(model.get_input_embeddings().num_embeddings)
    replacement = int((changed_input_ids[0, future_position].item() + 17) % vocab_size)
    if replacement == int(changed_input_ids[0, future_position].item()):
        replacement = int((replacement + 1) % vocab_size)
    changed_input_ids[0, future_position] = replacement

    with torch_module.no_grad():
        original = model(input_ids=batch["input_ids"], attention_mask=mask, labels=batch["labels"], use_cache=False)
        changed = model(input_ids=changed_input_ids, attention_mask=mask, labels=batch["labels"], use_cache=False)

    for pos in earlier_positions:
        delta = float((original.logits[0, pos, :] - changed.logits[0, pos, :]).abs().max().detach().cpu().item())
        assert_true("future_target_token_does_not_change_earlier_target_logits", delta <= args.atol, {"earlier_position": pos, "future_position": future_position, "max_logit_delta": delta}, failures)

    checks = {
        "prompt_tokens_bidirectional": bool(visible[0, 0, max(prompt_len - 1, 0)].item()) if prompt_len else False,
        "prompt_tokens_cannot_see_output": not bool(visible[0, 0, prompt_len].item()) if prompt_len < valid_len else True,
        "output_tokens_see_prompt": bool(visible[0, prompt_len, 0].item()) if prompt_len < valid_len else False,
        "output_tokens_see_previous_output": bool(visible[0, min(prompt_len + 1, valid_len - 1), prompt_len].item()) if prompt_len + 1 < valid_len else True,
        "output_tokens_cannot_see_future_output": not bool(visible[0, prompt_len, prompt_len + 1].item()) if prompt_len + 1 < valid_len else True,
        "padding_masked": not bool(visible[0, 0, valid_len].item()) if valid_len < visible.shape[-1] else True,
        "mask_shape": tuple(mask.shape),
        "mask_min_negative": float(mask.min().detach().cpu().item()) < -1e20,
        "mask_max_zero": float(mask.max().detach().cpu().item()) == 0.0,
    }
    for name, value in checks.items():
        if isinstance(value, bool):
            assert_true(name, value, checks, failures)

    return {
        "passed": not failures,
        "failures": failures,
        "record_id": rows[0]["record_id"],
        "prompt_length": prompt_len,
        "valid_length": valid_len,
        "future_position_changed": future_position,
        "earlier_positions_checked": earlier_positions,
        "replacement_token_id": replacement,
        "original_loss": float(original.loss.detach().cpu()),
        "changed_loss": float(changed.loss.detach().cpu()),
        "visibility_checks": checks,
        "attention_backend": getattr(model.base_model.model.config, "_attn_implementation", None),
    }


def cache_sequence_length(past_key_values: Any) -> Optional[int]:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        value = past_key_values.get_seq_length()
        return int(value) if value is not None else None
    try:
        first_layer = past_key_values[0]
        key_tensor = first_layer[0]
        return int(key_tensor.shape[-2])
    except (TypeError, IndexError, AttributeError):
        return None


def compare_logits(torch_module: Any, reference_logits: Any, cached_logits: Any, top_k: int = 5) -> Dict[str, Any]:
    diff = (reference_logits - cached_logits).abs()
    reference_top = torch_module.topk(reference_logits, k=top_k, dim=-1).indices[0].detach().cpu().tolist()
    cached_top = torch_module.topk(cached_logits, k=top_k, dim=-1).indices[0].detach().cpu().tolist()
    reference_top1 = int(reference_top[0])
    cached_top1 = int(cached_top[0])
    return {
        "max_abs_logit_difference": float(diff.max().detach().cpu().item()),
        "mean_abs_logit_difference": float(diff.mean().detach().cpu().item()),
        "reference_top1_token": reference_top1,
        "cached_top1_token": cached_top1,
        "top1_agrees": reference_top1 == cached_top1,
        "reference_top5_tokens": [int(token) for token in reference_top],
        "cached_top5_tokens": [int(token) for token in cached_top],
        "top5_exact_agrees": set(reference_top) == set(cached_top),
        "reference_top1_in_cached_top5": reference_top1 in set(cached_top),
        "cached_top1_in_reference_top5": cached_top1 in set(reference_top),
    }


def full_prefix_next_logits(model: Any, torch_module: Any, input_ids: List[int], attention_type: str, prompt_length: int, dtype: Any, device: Any):
    tensor_ids = torch_module.tensor([input_ids], dtype=torch_module.long, device=device)
    attention = torch_module.ones_like(tensor_ids, dtype=torch_module.long)
    if attention_type == "prompt_bidirectional":
        prompt_lengths = torch_module.tensor([prompt_length], dtype=torch_module.long, device=device)
        mask = build_4d_attention_mask(prompt_lengths, attention, "prompt_bidirectional", dtype=dtype, device=device)
    else:
        mask = attention
    with torch_module.no_grad():
        output = model(input_ids=tensor_ids, attention_mask=mask, use_cache=False)
    logits = output.logits[:, -1, :].detach()
    del output
    return logits


def cached_prefill(model: Any, torch_module: Any, prompt_ids: List[int], attention_type: str, dtype: Any, device: Any):
    prompt_input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
    prompt_attention = torch_module.ones_like(prompt_input_ids, dtype=torch_module.long)
    if attention_type == "prompt_bidirectional":
        prompt_lengths = torch_module.tensor([len(prompt_ids)], dtype=torch_module.long, device=device)
        prefill_mask = build_4d_attention_mask(prompt_lengths, prompt_attention, "prompt_bidirectional", dtype=dtype, device=device)
    else:
        prefill_mask = prompt_attention
    with torch_module.no_grad():
        prefill = model_forward_with_optional_cache_position(
            model,
            {
                "input_ids": prompt_input_ids,
                "attention_mask": prefill_mask,
                "use_cache": True,
                "cache_position": torch_module.arange(len(prompt_ids), device=device),
            },
            require_cache_position=True,
        )
    return prefill


def cached_decode_step(model: Any, torch_module: Any, token_id: int, past_key_values: Any, attention_type: str, position: int, dtype: Any, device: Any):
    forced_token = torch_module.tensor([[token_id]], dtype=torch_module.long, device=device)
    if attention_type == "prompt_bidirectional":
        mask = build_prompt_bidir_decode_mask(batch_size=1, key_length=position + 1, dtype=dtype, device=device)
    else:
        mask = torch_module.ones((1, position + 1), dtype=torch_module.long, device=device)
    position_ids = torch_module.tensor([[position]], dtype=torch_module.long, device=device)
    cache_position = torch_module.tensor([position], dtype=torch_module.long, device=device)
    with torch_module.no_grad():
        output = model_forward_with_optional_cache_position(
            model,
            {
                "input_ids": forced_token,
                "attention_mask": mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "use_cache": True,
                "cache_position": cache_position,
            },
            require_cache_position=True,
        )
    return output, position_ids, cache_position, mask


def forced_cache_control(args: argparse.Namespace, attention_type: str) -> Dict[str, Any]:
    model, tokenizer, torch_module = load_lora_model(args, attention_type, force_eager=True)
    rows = load_rows(args.train_file, args.seed, 1)
    row = rows[0]
    prompt_ids = tokenizer(make_reasoning_prompt(row["question"]), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["training_target"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    steps = min(args.cached_decode_steps, len(target_ids))
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    failures: List[Dict[str, Any]] = []

    prefill = cached_prefill(model, torch_module, prompt_ids, attention_type, dtype, device)
    past_key_values = prefill.past_key_values
    cached_next_logits = prefill.logits[:, -1, :].detach()
    prefill_cache_length = cache_sequence_length(past_key_values)
    assert_true(
        f"{attention_type}_prefill_cache_length_matches_prompt_length",
        prefill_cache_length == len(prompt_ids),
        {"cache_length": prefill_cache_length, "prompt_length": len(prompt_ids)},
        failures,
    )
    del prefill

    step_reports = []
    for step in range(steps):
        prefix_ids = prompt_ids + target_ids[:step]
        expected_next_position = len(prompt_ids) + step
        before_cache_length = cache_sequence_length(past_key_values)
        reference_logits = full_prefix_next_logits(model, torch_module, prefix_ids, attention_type, len(prompt_ids), dtype, device)
        comparison = compare_logits(torch_module, reference_logits, cached_next_logits)
        step_report = {
            "step": step,
            "prefix_length": len(prefix_ids),
            "generated_length": step,
            "expected_next_token_position": expected_next_position,
            "cache_length_before_decode": before_cache_length,
            "cache_length_matches_prefix": before_cache_length == len(prefix_ids),
            **comparison,
        }
        step_reports.append(step_report)
        assert_true(
            f"{attention_type}_forced_step_top1_agrees",
            comparison["top1_agrees"],
            step_report,
            failures,
        )
        assert_true(
            f"{attention_type}_forced_step_top5_agrees",
            comparison["top5_exact_agrees"],
            step_report,
            failures,
        )
        assert_true(
            f"{attention_type}_kv_cache_length_before_decode_matches_prefix",
            before_cache_length == len(prefix_ids),
            step_report,
            failures,
        )

        decode, position_ids, cache_position, decode_mask = cached_decode_step(
            model,
            torch_module,
            target_ids[step],
            past_key_values,
            attention_type,
            expected_next_position,
            dtype,
            device,
        )
        past_key_values = decode.past_key_values
        cached_next_logits = decode.logits[:, -1, :].detach()
        after_cache_length = cache_sequence_length(past_key_values)
        step_report.update(
            {
                "fed_token_position": expected_next_position,
                "position_ids": position_ids.detach().cpu().tolist(),
                "cache_position": cache_position.detach().cpu().tolist(),
                "decode_mask_shape": tuple(decode_mask.shape),
                "decode_mask_shape_expected": (
                    (1, 1, 1, expected_next_position + 1)
                    if attention_type == "prompt_bidirectional"
                    else (1, expected_next_position + 1)
                ),
                "decode_mask_shape_matches_expected": tuple(decode_mask.shape) == (
                    (1, 1, 1, expected_next_position + 1)
                    if attention_type == "prompt_bidirectional"
                    else (1, expected_next_position + 1)
                ),
                "cache_length_after_decode": after_cache_length,
                "cache_length_after_decode_expected": expected_next_position + 1,
                "cache_length_after_decode_matches_expected": after_cache_length == expected_next_position + 1,
            }
        )
        assert_true(
            f"{attention_type}_position_ids_match_expected_position",
            position_ids.detach().cpu().tolist() == [[expected_next_position]],
            step_report,
            failures,
        )
        assert_true(
            f"{attention_type}_cache_position_matches_expected_position",
            cache_position.detach().cpu().tolist() == [expected_next_position],
            step_report,
            failures,
        )
        assert_true(
            f"{attention_type}_decode_mask_shape_matches_expected",
            step_report["decode_mask_shape_matches_expected"],
            step_report,
            failures,
        )
        assert_true(
            f"{attention_type}_kv_cache_length_after_decode_matches_expected",
            after_cache_length == expected_next_position + 1,
            step_report,
            failures,
        )
        del decode

    return {
        "passed": not failures,
        "failures": failures,
        "attention_type": attention_type,
        "record_id": row["record_id"],
        "steps_checked": steps,
        "step_reports": step_reports,
        "max_abs_logit_difference_over_steps": max((report["max_abs_logit_difference"] for report in step_reports), default=None),
        "mean_abs_logit_difference_over_steps": sum((report["mean_abs_logit_difference"] for report in step_reports), 0.0) / len(step_reports) if step_reports else None,
        "all_top1_agree": all(report["top1_agrees"] for report in step_reports),
        "all_top5_exact_agree": all(report["top5_exact_agrees"] for report in step_reports),
        "generation_cache_semantics": "Prompt-bidirectional prefill uses one 4D mask over prompt tokens; decode uses past_key_values with single-query masks over existing prompt+generated keys. Native causal control uses the same BF16/eager model with standard causal cache decoding.",
        "attention_backend": getattr(model.base_model.model.config, "_attn_implementation", None),
    }


def greedy_cache_control(args: argparse.Namespace, attention_type: str, steps: int = 32) -> Dict[str, Any]:
    model, tokenizer, torch_module = load_lora_model(args, attention_type, force_eager=True)
    row = load_rows(args.train_file, args.seed, 1)[0]
    prompt_ids = tokenizer(make_reasoning_prompt(row["question"]), add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    failures: List[Dict[str, Any]] = []

    prefill = cached_prefill(model, torch_module, prompt_ids, attention_type, dtype, device)
    past_key_values = prefill.past_key_values
    cached_next_logits = prefill.logits[:, -1, :].detach()
    del prefill

    cached_tokens: List[int] = []
    reference_tokens: List[int] = []
    per_step = []
    first_divergence = None
    for step in range(steps):
        reference_logits = full_prefix_next_logits(
            model,
            torch_module,
            prompt_ids + reference_tokens,
            attention_type,
            len(prompt_ids),
            dtype,
            device,
        )
        comparison = compare_logits(torch_module, reference_logits, cached_next_logits)
        cached_token = int(comparison["cached_top1_token"])
        reference_token = int(comparison["reference_top1_token"])
        cached_tokens.append(cached_token)
        reference_tokens.append(reference_token)
        if first_divergence is None and cached_token != reference_token:
            first_divergence = {
                "step": step,
                "cached_token": cached_token,
                "reference_token": reference_token,
                "cached_token_in_reference_top5": comparison["cached_top1_in_reference_top5"],
                "reference_token_in_cached_top5": comparison["reference_top1_in_cached_top5"],
                "comparison": comparison,
                "note": "Later logits are not valid equivalence comparisons because the independent greedy prefixes diverged at this step.",
            }
        per_step.append(
            {
                "step": step,
                "prefixes_identical_before_step": first_divergence is None or step <= first_divergence["step"],
                **comparison,
            }
        )

        expected_position = len(prompt_ids) + step
        decode, _, _, _ = cached_decode_step(
            model,
            torch_module,
            cached_token,
            past_key_values,
            attention_type,
            expected_position,
            dtype,
            device,
        )
        past_key_values = decode.past_key_values
        cached_next_logits = decode.logits[:, -1, :].detach()
        del decode

    if attention_type == "prompt_bidirectional":
        assert_true(
            f"{attention_type}_greedy_cached_tokens_match_full_prefix_tokens",
            cached_tokens == reference_tokens,
            {"cached_tokens": cached_tokens, "reference_tokens": reference_tokens, "per_step": per_step},
            failures,
        )
    return {
        "passed": not failures,
        "failures": failures,
        "attention_type": attention_type,
        "hard_gate": attention_type == "prompt_bidirectional",
        "diagnostic_role": (
            "hard_gate_prompt_bidirectional_cached_vs_full_prefix_greedy"
            if attention_type == "prompt_bidirectional"
            else "calibration_only_native_causal_independent_greedy"
        ),
        "record_id": row["record_id"],
        "steps": steps,
        "cached_greedy_token_ids": cached_tokens,
        "reference_full_prefix_greedy_token_ids": reference_tokens,
        "cached_greedy_text": tokenizer.decode(cached_tokens, skip_special_tokens=True),
        "reference_full_prefix_greedy_text": tokenizer.decode(reference_tokens, skip_special_tokens=True),
        "all_greedy_tokens_agree": cached_tokens == reference_tokens,
        "first_divergence": first_divergence,
        "comparison_valid_after_first_divergence": first_divergence is None,
        "per_step": per_step,
        "max_abs_logit_difference_over_steps": max((report["max_abs_logit_difference"] for report in per_step), default=None),
        "mean_abs_logit_difference_over_steps": sum((report["mean_abs_logit_difference"] for report in per_step), 0.0) / len(per_step) if per_step else None,
    }


def cache_controls(args: argparse.Namespace) -> Dict[str, Any]:
    causal_forced = forced_cache_control(args, "causal")
    prompt_forced = forced_cache_control(args, "prompt_bidirectional")
    causal_greedy = greedy_cache_control(args, "causal")
    prompt_greedy = greedy_cache_control(args, "prompt_bidirectional")
    failures: List[Dict[str, Any]] = []

    causal_max = causal_forced.get("max_abs_logit_difference_over_steps")
    prompt_max = prompt_forced.get("max_abs_logit_difference_over_steps")
    if causal_max is not None and prompt_max is not None:
        comparable_threshold = max(1e-3, (float(causal_max) * 2.0) + 1e-3)
        prompt_numerically_comparable_to_causal = float(prompt_max) <= comparable_threshold
    else:
        comparable_threshold = None
        prompt_numerically_comparable_to_causal = False

    for name, report in [
        ("native_causal_forced_cache_control", causal_forced),
        ("prompt_bidirectional_forced_cache_control", prompt_forced),
        ("prompt_bidirectional_greedy_cache_control", prompt_greedy),
    ]:
        if not report.get("passed"):
            failures.append({"check": name, "details": report.get("failures")})

    assert_true(
        "prompt_bidirectional_cache_numeric_drift_is_comparable_to_native_causal_control",
        prompt_numerically_comparable_to_causal,
        {"causal_max_abs": causal_max, "prompt_bidir_max_abs": prompt_max, "threshold": comparable_threshold},
        failures,
    )

    return {
        "passed": not failures,
        "failures": failures,
        "native_causal_forced_cache_control": causal_forced,
        "prompt_bidirectional_forced_cache_control": prompt_forced,
        "native_causal_greedy_cache_control": causal_greedy,
        "prompt_bidirectional_greedy_cache_control": prompt_greedy,
        "behavioural_acceptance_rule": {
            "forced_steps": "top-1 decisions must agree for steps 0 through cached_decode_steps",
            "greedy_steps": "cached greedy token sequence must match full-prefix greedy token sequence for 32 steps",
            "native_causal_greedy": "reported as calibration-only; independent trajectory divergence is not a hard gate because later logits have different prefixes",
            "numeric_drift": "prompt-bidirectional max absolute logit drift must be no more than about 2x native causal BF16/eager cache-control drift plus 1e-3",
            "reason": "BF16 eager cached decoding can differ numerically from full-prefix recomputation even when decisions are unchanged; native causal control calibrates this drift.",
            "prompt_bidir_numeric_threshold_used": comparable_threshold,
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    report = {
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "max_length": args.max_length,
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "causal_equivalence": causal_equivalence(args),
        "prompt_bidirectional_model_leakage": prompt_bidirectional_model_leakage(args),
        "cache_controls": cache_controls(args),
        "test_set_used": False,
    }
    report["passed"] = (
        report["causal_equivalence"]["passed"]
        and report["prompt_bidirectional_model_leakage"]["passed"]
        and report["cache_controls"]["passed"]
    )
    write_json(args.output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
