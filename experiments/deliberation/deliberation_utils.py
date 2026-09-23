#!/usr/bin/env python3
"""Shared utilities for the final-thesis deliberation validation experiment."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
AUGMENTATION_DIR = REPO_ROOT / "experiments" / "cogmath_augmentation"
ATTENTION_CONTROL_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_attention_control"

if str(AUGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(AUGMENTATION_DIR))
if str(ATTENTION_CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(ATTENTION_CONTROL_DIR))

from experiment_utils import answers_match, extract_answer, read_jsonl, write_json, write_jsonl  # noqa: E402
from attention_masks import build_4d_attention_mask, build_prompt_bidir_decode_mask, prompt_bidir_allowed_matrix  # noqa: E402


DEFAULT_VALIDATION_FILE = REPO_ROOT / "experiments" / "cogmath_augmentation" / "validation_eval.jsonl"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Base"
DEFAULT_OUTPUT_ROOT = Path("outputs/final_thesis_deliberation_qwen3_4b")
DEFAULT_FIRST_PASS_PREDICTIONS = (
    Path("outputs")
    / "cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds"
    / "seed_42"
    / "validation_evaluations"
    / "budget_original_plus_dim2"
    / "predictions.jsonl"
)
DEFAULT_ADAPTER_DIR = (
    Path("outputs")
    / "cogmath_reasoning_budget_control_qwen3_4b_final_thesis_seeds"
    / "seed_42"
    / "adapters"
    / "budget_original_plus_dim2"
)

REVISION_PROMPT_TEMPLATE = """You are reviewing a completed mathematical reasoning trace.

Original problem:
{question}

First-pass reasoning trace:
{first_pass_trace}

First-pass extracted answer: {first_pass_answer}

Review the reasoning trace. If the first-pass answer is correct, keep it. If it is incorrect, correct it.
Finish with exactly one line in this format:
Final answer: <answer>

Revised answer:
"""

FINAL_MARKER_RE = re.compile(r"final\s+answer\s*:", flags=re.IGNORECASE)


def make_revision_prompt(row: Dict[str, Any]) -> str:
    return REVISION_PROMPT_TEMPLATE.format(
        question=row["question"],
        first_pass_trace=row["first_pass_trace"],
        first_pass_answer=row.get("first_pass_extracted_answer") or "",
    )


def extract_after_final_marker(text: str) -> tuple[bool, Optional[str]]:
    matches = list(FINAL_MARKER_RE.finditer(text or ""))
    if not matches:
        return False, extract_answer(text)
    return True, extract_answer(text[matches[-1].end():])


def ensure_new_predictions_path(output_dir: Path, allow_existing_output: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    if pred_path.exists() and not allow_existing_output:
        raise FileExistsError(f"Refusing to overwrite existing {pred_path}.")


def generated_length_and_cap_status(token_ids: Sequence[int], eos_token_id: Optional[int], max_new_tokens: int) -> tuple[int, bool]:
    ids = [int(tok) for tok in token_ids]
    if eos_token_id is not None and int(eos_token_id) in ids:
        return ids.index(int(eos_token_id)) + 1, False
    return len(ids), len(ids) >= max_new_tokens


def first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def normalize_first_pass_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    trace = first_present(prediction, ["generated_reasoning", "generated_text", "model_output", "response", "output"])
    extracted = first_present(prediction, ["extracted_answer", "first_pass_extracted_answer"])
    if extracted is None and trace is not None:
        extracted = extract_answer(trace)
    return {
        "record_id": prediction.get("record_id") or prediction.get("id"),
        "first_pass_trace": trace or "",
        "first_pass_extracted_answer": extracted,
        "first_pass_correct": bool(prediction.get("correct") or prediction.get("is_correct")),
        "first_pass_generated_token_count": first_present(prediction, ["generated_token_count", "generated_tokens"]),
        "first_pass_hit_cap": bool(first_present(prediction, ["hit_max_new_tokens", "hit_cap"]) or False),
        "first_pass_source_condition": prediction.get("condition"),
    }


def load_validation_by_record_id(eval_file: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(eval_file)
    split_values = {row.get("split") for row in rows}
    if split_values != {"validation"}:
        raise ValueError(f"Expected validation-only records in {eval_file}, found splits={sorted(split_values)}")
    return {row["record_id"]: row for row in rows}


def build_first_pass_rows(eval_file: Path, predictions_file: Path, *, limit: int = 0) -> List[Dict[str, Any]]:
    eval_by_id = load_validation_by_record_id(eval_file)
    predictions = [normalize_first_pass_prediction(row) for row in read_jsonl(predictions_file)]
    out = []
    missing_eval = []
    duplicate_ids = []
    seen = set()
    for pred in predictions:
        record_id = pred["record_id"]
        if record_id in seen:
            duplicate_ids.append(record_id)
            continue
        seen.add(record_id)
        eval_row = eval_by_id.get(record_id)
        if eval_row is None:
            missing_eval.append(record_id)
            continue
        if not pred["first_pass_trace"]:
            raise ValueError(f"First-pass prediction for {record_id} has no generated trace/text.")
        gold = eval_row["final_answer"]
        first_pass_answer = pred["first_pass_extracted_answer"]
        out.append(
            {
                "record_id": record_id,
                "base_id": eval_row["base_id"],
                "split": eval_row.get("split"),
                "dimension": eval_row.get("dimension"),
                "variant_type": eval_row.get("variant_type"),
                "question": eval_row["question"],
                "gold_final_answer": gold,
                "first_pass_trace": pred["first_pass_trace"],
                "first_pass_extracted_answer": first_pass_answer,
                "first_pass_correct": answers_match(first_pass_answer, gold),
                "first_pass_generated_token_count": pred["first_pass_generated_token_count"],
                "first_pass_hit_cap": pred["first_pass_hit_cap"],
                "first_pass_source_condition": pred["first_pass_source_condition"],
            }
        )
        if limit and len(out) >= limit:
            break
    if duplicate_ids:
        raise ValueError(f"Duplicate prediction record IDs: {duplicate_ids[:10]}")
    if missing_eval:
        raise ValueError(f"Predictions include non-validation record IDs: {missing_eval[:10]}")
    return out


def summarize_first_pass(rows: Sequence[Dict[str, Any]], predictions_file: Path) -> Dict[str, Any]:
    dims = Counter(str(row["dimension"]) for row in rows)
    correct = sum(1 for row in rows if row["first_pass_correct"])
    return {
        "source_predictions": str(predictions_file),
        "rows": len(rows),
        "unique_record_ids": len({row["record_id"] for row in rows}),
        "unique_base_ids": len({row["base_id"] for row in rows}),
        "dimension_counts": dict(sorted(dims.items())),
        "first_pass_correct": correct,
        "first_pass_accuracy": correct / len(rows) if rows else None,
    }


def assert_deliberation_mask_semantics(prefix_len: int, generated_len: int, pad_len: int = 2) -> Dict[str, Any]:
    seq_len = prefix_len + generated_len + pad_len
    valid_len = prefix_len + generated_len
    matrix = prompt_bidir_allowed_matrix(seq_len, prefix_len, valid_len)
    failures = []
    for q in range(seq_len):
        for k in range(seq_len):
            allowed = matrix[q][k]
            if q >= valid_len or k >= valid_len:
                expected = False
            elif q < prefix_len:
                expected = k < prefix_len
            else:
                expected = k < prefix_len or k <= q
            if allowed != expected:
                failures.append({"query": q, "key": k, "allowed": allowed, "expected": expected})
    if failures:
        raise AssertionError(f"Deliberation mask semantics failed: {failures[:5]}")
    return {
        "prefix_len": prefix_len,
        "generated_len": generated_len,
        "pad_len": pad_len,
        "prompt_tokens_bidirectional": True,
        "prompt_tokens_cannot_attend_revision_tokens": True,
        "revision_tokens_causal": True,
        "padding_masked": True,
    }


def mcnemar_exact(rows_a: Sequence[bool], rows_b: Sequence[bool]) -> Dict[str, Any]:
    b = sum(1 for a, c in zip(rows_a, rows_b) if a and not c)
    c = sum(1 for a, c2 in zip(rows_a, rows_b) if (not a) and c2)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        cdf = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
        p = min(1.0, 2.0 * cdf)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "discordant": n, "exact_two_sided_p": p}


def summarize_revision_predictions(rows: Sequence[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    by_dim: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dim[str(row.get("dimension"))].append(row)
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    generated = [row.get("generated_token_count") for row in rows if isinstance(row.get("generated_token_count"), (int, float))]
    return {
        "condition": condition,
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "by_dimension": {
            dim: {
                "rows": len(dim_rows),
                "correct": sum(1 for row in dim_rows if row.get("correct")),
                "accuracy": sum(1 for row in dim_rows if row.get("correct")) / len(dim_rows) if dim_rows else None,
            }
            for dim, dim_rows in sorted(by_dim.items())
        },
        "corrections": sum(1 for row in rows if (not row.get("first_pass_correct")) and row.get("correct")),
        "regressions": sum(1 for row in rows if row.get("first_pass_correct") and not row.get("correct")),
        "stable_correct": sum(1 for row in rows if row.get("first_pass_correct") and row.get("correct")),
        "stable_incorrect": sum(1 for row in rows if (not row.get("first_pass_correct")) and not row.get("correct")),
        "generated_tokens": {
            "mean": sum(generated) / len(generated) if generated else None,
            "max": max(generated) if generated else None,
            "cap_hits": sum(1 for row in rows if row.get("hit_max_revision_tokens")),
        },
    }

