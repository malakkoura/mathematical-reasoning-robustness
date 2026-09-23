#!/usr/bin/env python3
"""Prepare original-only GSM8K reasoning targets for the diagnostic experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from reasoning_utils import (
    AUGMENTATION_DIR,
    DATASET_REPORT,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    REASONING_TARGET_TEMPLATE,
    REASONING_TRAIN_FILE,
    REASONING_TRAINING_CONDITION,
    SOURCE_TRAIN_FILE,
    TEST_EVAL_FILE,
    TRAIN_MAX_LENGTH,
    VALIDATION_EVAL_FILE,
    make_reasoning_prompt,
)

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import answers_match, extract_answer, read_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reasoning-supervised original-only training data.")
    parser.add_argument("--source_train_file", type=Path, default=SOURCE_TRAIN_FILE)
    parser.add_argument("--output_file", type=Path, default=REASONING_TRAIN_FILE)
    parser.add_argument("--report_file", type=Path, default=DATASET_REPORT)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--candidate_max_lengths", nargs="+", type=int, default=[512, 768, 1024, 1536, 2048])
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: List[int], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * pct
    low = int(index)
    high = min(low + 1, len(values) - 1)
    if low == high:
        return float(values[low])
    return values[low] + (values[high] - values[low]) * (index - low)


def length_summary(values: List[int]) -> Dict[str, Any]:
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def split_reasoning(answer_raw: str) -> str:
    if "####" not in answer_raw:
        raise ValueError("answer_raw does not contain GSM8K #### marker.")
    return answer_raw.split("####", 1)[0].rstrip()


def build_rows(source_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in source_rows:
        if int(row["dimension"]) != 0 or row.get("variant_type") != "original":
            raise ValueError(f"Expected original-only row, got {row.get('record_id')}.")
        final_from_raw = extract_answer(row["answer_raw"])
        if not answers_match(final_from_raw, row["final_answer"]):
            raise ValueError(
                f"{row['record_id']} answer_raw final answer {final_from_raw!r} disagrees with final_answer={row['final_answer']!r}."
            )
        reasoning = split_reasoning(row["answer_raw"])
        target = REASONING_TARGET_TEMPLATE.format(reasoning=reasoning, final_answer=row["final_answer"])
        if "####" in target:
            raise ValueError(f"{row['record_id']} still contains #### in reasoning training target.")
        if not target.endswith(f"Final answer: {row['final_answer']}"):
            raise ValueError(f"{row['record_id']} has malformed final-answer line.")

        rows.append(
            {
                **row,
                "source_record_id": row["record_id"],
                "record_id": f"{row['record_id']}_reasoning",
                "condition": REASONING_TRAINING_CONDITION,
                "training_target": target,
                "prompt": make_reasoning_prompt(row["question"]),
                "target_format": "reasoning_plus_final_answer_marker",
            }
        )
    return rows


def tokenizer_stats(rows: List[Dict[str, Any]], model_name: str, cache_dir: Optional[Path], candidate_max_lengths: List[int]) -> Dict[str, Any]:
    text_lengths = [len(row["prompt"] + row["training_target"]) for row in rows]
    whitespace_lengths = [len((row["prompt"] + row["training_target"]).split()) for row in rows]
    stats: Dict[str, Any] = {
        "character_lengths": length_summary(text_lengths),
        "whitespace_token_lengths": length_summary(whitespace_lengths),
        "tokenizer_status": "unavailable",
        "model_name": model_name,
        "candidate_max_lengths": candidate_max_lengths,
        "chosen_max_length": TRAIN_MAX_LENGTH,
        "chosen_max_length_status": "provisional_until_exact_tokenizer_check",
        "examples_over_candidate_lengths": {
            str(length): sum(1 for value in whitespace_lengths if value > length) for length in candidate_max_lengths
        },
    }
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        stats["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        return stats

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        eos = tokenizer.eos_token or ""
        token_lengths = []
        prompt_lengths = []
        target_lengths = []
        for row in rows:
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"]
            prompt_lengths.append(len(prompt_ids))
            target_lengths.append(len(target_ids))
            token_lengths.append(len(prompt_ids) + len(target_ids))
        chosen = next((length for length in candidate_max_lengths if max(token_lengths) <= length), None)
        stats.update(
            {
                "tokenizer_status": "available",
                "token_lengths": length_summary(token_lengths),
                "prompt_token_lengths": length_summary(prompt_lengths),
                "target_token_lengths": length_summary(target_lengths),
                "examples_over_candidate_lengths": {
                    str(length): sum(1 for value in token_lengths if value > length) for length in candidate_max_lengths
                },
                "examples_truncated_at_512": sum(1 for value in token_lengths if value > 512),
                f"examples_truncated_at_{TRAIN_MAX_LENGTH}": sum(1 for value in token_lengths if value > TRAIN_MAX_LENGTH),
                "chosen_max_length": chosen,
                "chosen_max_length_status": "exact" if chosen is not None else "no_candidate_large_enough",
            }
        )
    except Exception as exc:
        stats["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
    return stats


def leakage_report(train_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_ids = {row["base_id"] for row in train_rows}
    validation_ids = {row["base_id"] for row in read_jsonl(VALIDATION_EVAL_FILE)}
    test_ids = {row["base_id"] for row in read_jsonl(TEST_EVAL_FILE)}
    return {
        "train_validation_overlap": sorted(train_ids & validation_ids),
        "train_test_overlap": sorted(train_ids & test_ids),
        "validation_test_overlap": sorted(validation_ids & test_ids),
    }


def write_report(path: Path, report: Dict[str, Any]) -> None:
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    source_rows = read_jsonl(args.source_train_file)
    rows = build_rows(source_rows)
    record_ids = [row["record_id"] for row in rows]
    base_ids = [row["base_id"] for row in rows]

    if len(rows) != 923:
        raise AssertionError(f"Expected 923 reasoning rows, got {len(rows)}.")
    if len(set(base_ids)) != 923:
        raise AssertionError("Reasoning data does not contain 923 unique base IDs.")
    if len(set(record_ids)) != len(record_ids):
        raise AssertionError("Reasoning data has duplicate record IDs.")

    write_jsonl(args.output_file, rows)
    dimensions = Counter(int(row["dimension"]) for row in rows)
    report = {
        "experiment": "cogmath_reasoning_supervision_qwen3_4b",
        "model_name": args.model_name,
        "source_train_file": str(args.source_train_file),
        "output_file": str(args.output_file),
        "output_sha256": sha256(args.output_file),
        "row_count": len(rows),
        "unique_base_ids": len(set(base_ids)),
        "condition": REASONING_TRAINING_CONDITION,
        "dimension_counts": dict(sorted(dimensions.items())),
        "target_format": "Reasoning:\\n<answer_raw before ####>\\n\\nFinal answer: <final_answer>",
        "answer_raw_final_answer_agrees": True,
        "no_gsm8k_marker_in_training_target": all("####" not in row["training_target"] for row in rows),
        "all_targets_end_with_gold_final_answer": all(
            row["training_target"].endswith(f"Final answer: {row['final_answer']}") for row in rows
        ),
        "leakage": leakage_report(rows),
        "token_stats": tokenizer_stats(rows, args.model_name, args.cache_dir, args.candidate_max_lengths),
        "training_hyperparameters_preserved_from_direct_answer_where_applicable": {
            "epochs": 2,
            "learning_rate": 2e-4,
            "seed": 42,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": LORA_TARGET_MODULES,
            "bf16": True,
            "gradient_checkpointing": True,
            "use_cache": False,
        },
    }
    write_report(args.report_file, report)


if __name__ == "__main__":
    main()
