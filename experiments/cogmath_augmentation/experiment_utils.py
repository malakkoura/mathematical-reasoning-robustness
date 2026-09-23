#!/usr/bin/env python3
"""Shared utilities for the CogMath augmentation experiment."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

PROMPT_TEMPLATE = """Solve the following maths problem. Return only the final answer in the requested format.

Question:
{question}

Answer:
"""

TRAINING_CONDITIONS: Dict[str, str] = {
    "original_only": "train_original_only.jsonl",
    "original_plus_dim1": "train_original_plus_dim1.jsonl",
    "original_plus_dim2": "train_original_plus_dim2.jsonl",
    "original_plus_dim4": "train_original_plus_dim4.jsonl",
    "original_plus_dim6": "train_original_plus_dim6.jsonl",
    "original_plus_all": "train_original_plus_all.jsonl",
    "original_only_large": "train_original_only_large.jsonl",
    "original_plus_all_full": "train_original_plus_all_full.jsonl",
}

EVALUATION_CONDITIONS: List[str] = [
    "base_untuned",
    "original_only",
    "original_plus_dim1",
    "original_plus_dim2",
    "original_plus_dim4",
    "original_plus_dim6",
    "original_plus_all",
    "original_only_large",
    "original_plus_all_full",
]

DIMENSION_LABELS = {
    0: "original",
    1: "dim1_paraphrasing",
    2: "dim2_word_scrambling",
    4: "dim4_irrelevant_information",
    6: "dim6_numerical_variation",
}

NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
FRACTION_RE = re.compile(r"-?\d+\s*/\s*\d+")
BOXED_RE = re.compile(r"\\(?:boxed|fbox)\{([^{}]+)\}")


def make_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def clean_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip().strip("`").strip()
    answer = answer.rstrip(".。;")
    return answer.strip() or None


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else str(text).strip()


def _text_after_answer_marker(text: str) -> Optional[str]:
    matches = list(
        re.finditer(
            r"(?:final\s+answer|the\s+answer\s+is|answer)\s*:?",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return None
    return text[matches[-1].end():].strip()


def extract_answer(text: Optional[str]) -> Optional[str]:
    """Extract a concise final answer from model output or gold text."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    if "####" in text:
        text = text.split("####")[-1].strip()

    marked = _text_after_answer_marker(text)
    candidate = marked if marked is not None else text
    candidate = clean_answer(_last_nonempty_line(candidate))
    if not candidate:
        return None

    boxed = BOXED_RE.findall(candidate)
    if boxed:
        return clean_answer(boxed[-1])

    fraction = FRACTION_RE.fullmatch(candidate)
    if fraction:
        return clean_answer(candidate.replace(" ", ""))

    numbers = NUMBER_RE.findall(candidate)
    if numbers:
        return clean_answer(numbers[-1])

    return clean_answer(candidate)


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    answer = clean_answer(answer)
    if answer is None:
        return None

    answer = answer.replace("$", "").replace(",", "").strip()
    answer = re.sub(r"\\\((.*?)\\\)", r"\1", answer)
    answer = re.sub(r"^\$(.*)\$$", r"\1", answer)
    answer = answer.rstrip(".")

    percent = answer.endswith("%")
    answer_no_percent = answer[:-1] if percent else answer
    answer_no_percent = answer_no_percent.strip()

    if FRACTION_RE.fullmatch(answer_no_percent):
        num, den = re.split(r"/", answer_no_percent)
        try:
            value = Decimal(num.strip()) / Decimal(den.strip())
            return f"{value.normalize()}%" if percent else str(value.normalize())
        except (InvalidOperation, ZeroDivisionError):
            return f"{answer_no_percent.replace(' ', '')}%" if percent else answer_no_percent.replace(" ", "")

    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", answer_no_percent)
    if number_match:
        try:
            value = Decimal(answer_no_percent)
            normalized = str(value.normalize())
            return f"{normalized}%" if percent else normalized
        except InvalidOperation:
            pass

    compact = re.sub(r"\s+", "", answer_no_percent.lower())
    return f"{compact}%" if percent else compact


def answers_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)


def dimension_key(dimension: int) -> str:
    return DIMENSION_LABELS.get(int(dimension), f"dim{dimension}")


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def accuracy(correct: int, total: int) -> Optional[float]:
    return correct / total if total else None


def summarize_predictions(rows: Sequence[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    by_dim: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dim[int(row["dimension"])].append(row)

    def metric(metric_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(metric_rows)
        correct = sum(1 for row in metric_rows if bool(row.get("correct")))
        return {"correct": correct, "total": total, "accuracy": accuracy(correct, total)}

    original_metric = metric(by_dim.get(0, []))
    dimension_metrics = {dimension_key(dim): metric(dim_rows) for dim, dim_rows in sorted(by_dim.items())}
    variant_dims = [1, 2, 4, 6]
    variant_accuracies = [
        dimension_metrics[dimension_key(dim)]["accuracy"]
        for dim in variant_dims
        if dimension_key(dim) in dimension_metrics and dimension_metrics[dimension_key(dim)]["accuracy"] is not None
    ]
    mean_variant_accuracy = mean(variant_accuracies)
    worst_dimension_accuracy = min(variant_accuracies) if variant_accuracies else None
    robustness_gap = (
        original_metric["accuracy"] - mean_variant_accuracy
        if original_metric["accuracy"] is not None and mean_variant_accuracy is not None
        else None
    )

    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[str(row["base_id"])].append(row)
    complete_bases = {
        base_id: base_rows
        for base_id, base_rows in by_base.items()
        if {int(row["dimension"]) for row in base_rows} >= {0, 1, 2, 4, 6}
    }
    matched_consistency_correct = sum(
        1 for base_rows in complete_bases.values() if all(bool(row.get("correct")) for row in base_rows)
    )
    complete_case_total = len(complete_bases)

    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    generated_tokens = [float(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None]
    tokens_per_second = [
        float(row["tokens_per_second"])
        for row in rows
        if row.get("tokens_per_second") not in (None, "") and math.isfinite(float(row["tokens_per_second"]))
    ]

    return {
        "condition": condition,
        "overall": metric(rows),
        "original": original_metric,
        "by_dimension": dimension_metrics,
        "mean_variant_accuracy": {
            "dimensions": [dimension_key(dim) for dim in variant_dims],
            "accuracy": mean_variant_accuracy,
            "denominator": sum(len(by_dim.get(dim, [])) for dim in variant_dims),
        },
        "worst_dimension_accuracy": worst_dimension_accuracy,
        "robustness_gap": robustness_gap,
        "matched_consistency_by_base_id": {
            "all_forms_correct": matched_consistency_correct,
            "complete_case_base_ids": complete_case_total,
            "rate": accuracy(matched_consistency_correct, complete_case_total),
        },
        "all_forms_correct_rate": accuracy(matched_consistency_correct, complete_case_total),
        "latency": {
            "average_latency_seconds": mean(latencies),
            "median_latency_seconds": median(latencies),
            "average_generated_tokens": mean(generated_tokens),
            "average_tokens_per_second": mean(tokens_per_second),
        },
    }
