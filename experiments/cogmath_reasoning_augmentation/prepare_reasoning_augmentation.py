#!/usr/bin/env python3
"""Prepare and audit reasoning-supervised unique-full augmentation datasets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from reasoning_aug_utils import (
    CONDITION_DIMENSIONS,
    DATA_DIR,
    EVALUATION_CONDITIONS,
    EXPLICIT_DIM6_EXCLUSIONS,
    MODEL_NAME,
    REASONING_TARGET_TEMPLATE,
    SPLITS_FILE,
    TRAIN_MAX_LENGTH,
    TRAINING_CONDITIONS,
    make_reasoning_prompt,
    reasoning_training_path,
    source_training_path,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cogmath_augmentation"))
from experiment_utils import answers_match, extract_answer, read_jsonl, write_json, write_jsonl  # noqa: E402


BOXED_RE = re.compile(r"\\(?:boxed|fbox)\{([^{}]+)\}")
CALC_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
INVALID_TRACE_RE = re.compile(r"can't calculate|cannot calculate|not enough information|doesn't yield an integer", re.I)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reasoning augmentation data and audit Dim6 traces.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--write_data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate_max_lengths", nargs="+", type=int, default=[768, 1024, 1536, 2048])
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_reasoning(answer_raw: str) -> str:
    if "####" in answer_raw:
        return answer_raw.split("####", 1)[0].rstrip()
    return answer_raw.strip()


def numericish(value: Any) -> str:
    return str(value).strip().replace("\\", "").replace("$", "").replace(",", "").rstrip(".")


def relaxed_match(candidate: Optional[str], gold: Optional[str]) -> bool:
    if answers_match(candidate, gold):
        return True
    if answers_match(extract_answer(candidate), gold):
        return True
    c = numericish(candidate)
    g = numericish(gold)
    if c.endswith("%"):
        c = c[:-1]
    if g.endswith("%"):
        g = g[:-1]
    return c == g


def dominant_final_candidate(answer_raw: str) -> Optional[str]:
    if "####" in answer_raw:
        tail = answer_raw.rsplit("####", 1)[1].strip()
        return tail.splitlines()[0].strip() if tail else extract_answer(answer_raw)
    boxed = BOXED_RE.findall(answer_raw)
    if boxed:
        return extract_answer(boxed[-1])
    numbers = NUMBER_RE.findall(answer_raw)
    return numbers[-1] if numbers else extract_answer(answer_raw)


def safe_eval(expr: str) -> Optional[float]:
    allowed = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Load,
    }
    expr = expr.replace(",", "").replace("$", "").replace("%", "/100")
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in allowed:
                return None
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def check_calc_annotations(answer_raw: str) -> List[Dict[str, Any]]:
    failures = []
    for expr, result in CALC_RE.findall(answer_raw):
        value = safe_eval(expr)
        if value is None:
            continue
        try:
            target = float(numericish(result).rstrip("%"))
        except ValueError:
            continue
        if abs(value - target) > 1e-6:
            failures.append({"expr": expr, "encoded_result": result, "computed": value})
    return failures


def audit_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures = []
    answer_raw = row.get("answer_raw")
    if not answer_raw:
        failures.append({"reason": "missing_answer_raw"})
        return failures
    candidate = dominant_final_candidate(answer_raw)
    if not relaxed_match(candidate, row.get("final_answer")):
        failures.append({"reason": "answer_raw_final_disagrees_with_final_answer", "candidate": candidate, "final_answer": row.get("final_answer")})
    calc_failures = check_calc_annotations(answer_raw)
    if calc_failures:
        failures.append({"reason": "calculator_annotation_inconsistent", "examples": calc_failures[:5]})
    if int(row["dimension"]) == 6:
        if row.get("answer_changed") and row.get("answer_raw") == row.get("original_answer_raw"):
            failures.append({"reason": "stale_original_rationale_for_changed_dim6"})
        if INVALID_TRACE_RE.search(answer_raw):
            failures.append({"reason": "invalid_or_underspecified_dim6_trace"})
    return failures


def explicit_exclusion(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reason = EXPLICIT_DIM6_EXCLUSIONS.get(row["record_id"])
    if reason is None:
        return None
    return {
        "record_id": row["record_id"],
        "base_id": row["base_id"],
        "dimension": int(row["dimension"]),
        "reason": reason,
    }


def build_reasoning_rows(source_rows: List[Dict[str, Any]], condition: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    invalid = []
    excluded = []
    for row in source_rows:
        exclusion = explicit_exclusion(row)
        if exclusion is not None:
            excluded.append(exclusion)
            continue
        failures = audit_row(row)
        if failures:
            invalid.append({"record_id": row["record_id"], "base_id": row["base_id"], "dimension": row["dimension"], "failures": failures})
            continue
        target = REASONING_TARGET_TEMPLATE.format(reasoning=split_reasoning(row["answer_raw"]), final_answer=row["final_answer"])
        if "####" in target:
            invalid.append({"record_id": row["record_id"], "base_id": row["base_id"], "dimension": row["dimension"], "failures": [{"reason": "gsm8k_marker_remains"}]})
            continue
        rows.append(
            {
                **row,
                "source_condition": row["condition"],
                "source_record_id": row["record_id"],
                "record_id": f"{row['record_id']}_reasoning",
                "condition": condition,
                "prompt": make_reasoning_prompt(row["question"]),
                "training_target": target,
                "target_format": "reasoning_plus_final_answer_marker",
            }
        )
    return rows, invalid, excluded


def percentile(values: List[int], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (idx - lo)


def length_summary(values: List[int]) -> Dict[str, Any]:
    return {"min": min(values), "median": statistics.median(values), "p90": percentile(values, 0.9), "p95": percentile(values, 0.95), "p99": percentile(values, 0.99), "max": max(values)}


def tokenizer_stats(all_rows: List[Dict[str, Any]], model_name: str, cache_dir: Optional[Path], candidates: List[int]) -> Dict[str, Any]:
    char_lengths = [len(row["prompt"] + row["training_target"]) for row in all_rows]
    whitespace = [len((row["prompt"] + row["training_target"]).split()) for row in all_rows]
    stats: Dict[str, Any] = {
        "tokenizer_status": "unavailable",
        "character_lengths": length_summary(char_lengths) if char_lengths else {},
        "whitespace_token_lengths": length_summary(whitespace) if whitespace else {},
        "candidate_max_lengths": candidates,
        "chosen_max_length": TRAIN_MAX_LENGTH,
        "chosen_max_length_status": "provisional_until_cluster_tokenizer_check",
        "examples_over_candidate_lengths": {str(c): sum(v > c for v in whitespace) for c in candidates},
    }
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        eos = tokenizer.eos_token or ""
        lengths = [
            len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            + len(tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"])
            for row in all_rows
        ]
        chosen = next((c for c in candidates if max(lengths) <= c), None)
        stats.update(
            {
                "tokenizer_status": "available",
                "token_lengths": length_summary(lengths),
                "examples_over_candidate_lengths": {str(c): sum(v > c for v in lengths) for c in candidates},
                "chosen_max_length": chosen,
                "chosen_max_length_status": "exact" if chosen is not None else "no_candidate_large_enough",
            }
        )
    except Exception as exc:
        stats["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
    return stats


def leakage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_ids = {row["base_id"] for row in rows}
    with SPLITS_FILE.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    validation_ids = set(splits["validation"])
    test_ids = set(splits["test"])
    return {
        "train_validation_overlap": sorted(train_ids & validation_ids),
        "train_test_overlap": sorted(train_ids & test_ids),
        "validation_test_overlap": sorted(validation_ids & test_ids),
        "split_source": str(SPLITS_FILE),
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    condition_reports = {}
    all_valid_rows = []
    blocked_conditions = {}
    all_exclusions = []

    for condition in EVALUATION_CONDITIONS:
        source_rows = read_jsonl(source_training_path(condition))
        rows, invalid, excluded = build_reasoning_rows(source_rows, condition)
        invalid_dim6 = [entry for entry in invalid if int(entry["dimension"]) == 6]
        all_exclusions.extend({**entry, "condition": condition} for entry in excluded)
        if invalid_dim6:
            blocked_conditions[condition] = invalid_dim6
        else:
            path = reasoning_training_path(condition)
            if args.write_data:
                write_jsonl(path, rows)
            all_valid_rows.extend(rows)
        by_dim = Counter(int(row["dimension"]) for row in rows)
        source_by_dim = Counter(int(row["dimension"]) for row in source_rows)
        condition_reports[condition] = {
            "source_file": str(source_training_path(condition)),
            "source_rows": len(source_rows),
            "source_counts_by_dimension": dict(sorted(source_by_dim.items())),
            "valid_reasoning_rows": len(rows),
            "valid_counts_by_dimension": dict(sorted(by_dim.items())),
            "invalid_rows": invalid,
            "explicit_data_quality_exclusions": excluded,
            "blocked": condition in blocked_conditions,
            "output_file": str(reasoning_training_path(condition)) if condition not in blocked_conditions else None,
        }
        if condition not in blocked_conditions and args.write_data:
            condition_reports[condition]["sha256"] = sha256(reasoning_training_path(condition))

    report = {
        "experiment": "cogmath_reasoning_augmentation_qwen3_4b",
        "model_name": args.model_name,
        "training_max_length": TRAIN_MAX_LENGTH,
        "conditions": condition_reports,
        "explicit_data_quality_exclusions": all_exclusions,
        "expected_explicit_exclusions": [
            {"record_id": record_id, "dimension": 6, "reason": reason}
            for record_id, reason in sorted(EXPLICIT_DIM6_EXCLUSIONS.items())
        ],
        "blocked_conditions": blocked_conditions,
        "ready_for_six_condition_training": not blocked_conditions,
        "token_stats": tokenizer_stats(all_valid_rows, args.model_name, args.cache_dir, args.candidate_max_lengths) if all_valid_rows else {},
        "leakage": leakage(all_valid_rows) if all_valid_rows else {},
        "array_conditions_requested": TRAINING_CONDITIONS,
        "note": "Only explicit Dim6 data-quality exclusions are skipped; any other invalid reasoning trace blocks preparation.",
    }
    write_json(DATA_DIR / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
