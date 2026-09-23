#!/usr/bin/env python3
"""Analyze the separate 512-token validation diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from reasoning_utils import (
    AUGMENTATION_DIR,
    BASE_REASONING_CONDITION,
    BASE_REASONING_CONDITION_512,
    DIRECT_ADAPTER_REASONING_CONDITION,
    DIRECT_ADAPTER_REASONING_CONDITION_512,
    DIRECT_ORIGINAL_ONLY_ADAPTER_DIR,
    DIRECT_ORIGINAL_ONLY_VALIDATION_DIR,
    DIRECT_REFERENCE_CONDITION,
    OUTPUT_ROOT,
    REASONING_ADAPTER_CONDITION,
    REASONING_ADAPTER_CONDITION_512,
    REASONING_ADAPTER_DIR,
    REASONING_TRAIN_FILE,
)

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import read_jsonl, summarize_predictions, write_json  # noqa: E402


FORMS = [
    ("overall", None, "Overall"),
    ("original", 0, "Original"),
    ("dim1_paraphrasing", 1, "Dim1 paraphrasing"),
    ("dim2_word_scrambling", 2, "Dim2 word scrambling"),
    ("dim4_irrelevant_information", 4, "Dim4 irrelevant information"),
    ("dim6_numerical_variation", 6, "Dim6 numerical variation"),
]

SUMMARY_CONDITIONS = [
    DIRECT_REFERENCE_CONDITION,
    BASE_REASONING_CONDITION,
    BASE_REASONING_CONDITION_512,
    DIRECT_ADAPTER_REASONING_CONDITION,
    DIRECT_ADAPTER_REASONING_CONDITION_512,
    REASONING_ADAPTER_CONDITION,
    REASONING_ADAPTER_CONDITION_512,
]

PAIRED_256_512 = [
    (BASE_REASONING_CONDITION, BASE_REASONING_CONDITION_512),
    (DIRECT_ADAPTER_REASONING_CONDITION, DIRECT_ADAPTER_REASONING_CONDITION_512),
    (REASONING_ADAPTER_CONDITION, REASONING_ADAPTER_CONDITION_512),
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 512-token reasoning validation results.")
    parser.add_argument("--results_root_256", type=Path, default=Path(OUTPUT_ROOT) / "validation_evaluations")
    parser.add_argument("--results_root_512", type=Path, default=Path(OUTPUT_ROOT) / "validation_evaluations_512")
    parser.add_argument("--direct_reference_dir", type=Path, default=DIRECT_ORIGINAL_ONLY_VALIDATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=Path(OUTPUT_ROOT) / "validation_analysis_512")
    parser.add_argument("--train_file", type=Path, default=REASONING_TRAIN_FILE)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_if_present(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(args: argparse.Namespace, condition: str) -> List[Dict[str, Any]]:
    if condition == DIRECT_REFERENCE_CONDITION:
        path = args.direct_reference_dir / "predictions.jsonl"
    elif condition.endswith("_512"):
        path = args.results_root_512 / condition / "predictions.jsonl"
    else:
        path = args.results_root_256 / condition / "predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    for row in rows:
        row["condition"] = condition
    return rows


def training_stats(train_file: Path) -> Dict[str, Any]:
    rows = read_jsonl(train_file)
    return {
        "training_rows": len(rows),
        "training_unique_base_ids": len({row["base_id"] for row in rows}),
        "training_original_examples": sum(1 for row in rows if int(row["dimension"]) == 0),
    }


def condition_metadata(condition: str, train_stats: Dict[str, Any]) -> Dict[str, Any]:
    if condition in {BASE_REASONING_CONDITION, BASE_REASONING_CONDITION_512}:
        return {
            "prompt_protocol": "reasoning",
            "generation_budget": 512 if condition.endswith("_512") else 256,
            "adapter_protocol": "none",
            "adapter_dir": None,
            "training_rows": 0,
            "training_unique_base_ids": 0,
            "training_original_examples": 0,
            "trainable_parameters": 0,
            "trainable_percentage": 0.0,
        }
    if condition in {DIRECT_ADAPTER_REASONING_CONDITION, DIRECT_ADAPTER_REASONING_CONDITION_512, DIRECT_REFERENCE_CONDITION}:
        params = load_json_if_present(DIRECT_ORIGINAL_ONLY_ADAPTER_DIR / "parameter_report.json")
        return {
            "prompt_protocol": "direct_answer_existing" if condition == DIRECT_REFERENCE_CONDITION else "reasoning",
            "generation_budget": 32 if condition == DIRECT_REFERENCE_CONDITION else (512 if condition.endswith("_512") else 256),
            "adapter_protocol": "direct_answer_original_only",
            "adapter_dir": str(DIRECT_ORIGINAL_ONLY_ADAPTER_DIR),
            **train_stats,
            "trainable_parameters": params.get("trainable_parameters"),
            "trainable_percentage": params.get("trainable_percentage"),
        }
    if condition in {REASONING_ADAPTER_CONDITION, REASONING_ADAPTER_CONDITION_512}:
        params = load_json_if_present(REASONING_ADAPTER_DIR / "parameter_report.json")
        return {
            "prompt_protocol": "reasoning",
            "generation_budget": 512 if condition.endswith("_512") else 256,
            "adapter_protocol": "reasoning_original_only",
            "adapter_dir": str(REASONING_ADAPTER_DIR),
            **train_stats,
            "trainable_parameters": params.get("trainable_parameters"),
            "trainable_percentage": params.get("trainable_percentage"),
        }
    raise KeyError(condition)


def diagnostic_rates(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cap_values = [row.get("hit_max_new_tokens") for row in rows if row.get("hit_max_new_tokens") is not None]
    marker_values = [row.get("missing_final_answer_marker") for row in rows if row.get("missing_final_answer_marker") is not None]
    cap_hits = sum(1 for value in cap_values if value)
    missing_markers = sum(1 for value in marker_values if value)
    return {
        "cap_hits": cap_hits if cap_values else None,
        "cap_hit_rate": cap_hits / len(cap_values) if cap_values else None,
        "missing_final_answer_markers": missing_markers if marker_values else None,
        "missing_final_answer_marker_rate": missing_markers / len(marker_values) if marker_values else None,
    }


def row_for_summary(summary: Dict[str, Any], metadata: Dict[str, Any], predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_dim = summary["by_dimension"]
    return {
        "condition": summary["condition"],
        **metadata,
        "overall_accuracy": summary["overall"]["accuracy"],
        "overall_correct": summary["overall"]["correct"],
        "overall_total": summary["overall"]["total"],
        "original_accuracy": summary["original"]["accuracy"],
        "dim1_accuracy": by_dim.get("dim1_paraphrasing", {}).get("accuracy"),
        "dim2_accuracy": by_dim.get("dim2_word_scrambling", {}).get("accuracy"),
        "dim4_accuracy": by_dim.get("dim4_irrelevant_information", {}).get("accuracy"),
        "dim6_accuracy": by_dim.get("dim6_numerical_variation", {}).get("accuracy"),
        "mean_transformed_accuracy": summary["mean_variant_accuracy"]["accuracy"],
        "worst_dimension_accuracy": summary["worst_dimension_accuracy"],
        "robustness_gap": summary["robustness_gap"],
        "matched_consistency": summary["all_forms_correct_rate"],
        "complete_case_base_ids": summary["matched_consistency_by_base_id"]["complete_case_base_ids"],
        "average_generated_tokens": summary["latency"]["average_generated_tokens"],
        "average_latency_seconds": summary["latency"]["average_latency_seconds"],
        "average_tokens_per_second": summary["latency"]["average_tokens_per_second"],
        **diagnostic_rates(predictions),
    }


def exact_two_sided_binomial_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * lower_tail)


def holm_adjust(rows: List[Dict[str, Any]]) -> None:
    indexed = [(index, row["exact_mcnemar_p_value"]) for index, row in enumerate(rows)]
    ordered = sorted(indexed, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (index, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p_value"] = running


def keyed_predictions(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["record_id"]: row for row in rows}


def comparison_rows(
    left_condition: str,
    right_condition: str,
    left_rows: Sequence[Dict[str, Any]],
    right_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    left = keyed_predictions(left_rows)
    right = keyed_predictions(right_rows)
    shared_ids = sorted(set(left) & set(right))
    stats_rows = []
    rescues = []
    regressions = []

    for key, dimension, label in FORMS:
        ids = shared_ids if dimension is None else [rid for rid in shared_ids if int(left[rid]["dimension"]) == dimension]
        both_correct = left_only = right_only = both_wrong = 0
        for rid in ids:
            l_correct = bool(left[rid].get("correct"))
            r_correct = bool(right[rid].get("correct"))
            if l_correct and r_correct:
                both_correct += 1
            elif l_correct and not r_correct:
                left_only += 1
                regressions.append({"comparison": f"{left_condition}_vs_{right_condition}", "evaluation_form": key, "record": right[rid]})
            elif not l_correct and r_correct:
                right_only += 1
                rescues.append({"comparison": f"{left_condition}_vs_{right_condition}", "evaluation_form": key, "record": right[rid]})
            else:
                both_wrong += 1
        total = len(ids)
        left_acc = (both_correct + left_only) / total if total else None
        right_acc = (both_correct + right_only) / total if total else None
        stats_rows.append(
            {
                "comparison": f"{left_condition}_vs_{right_condition}",
                "left_condition": left_condition,
                "right_condition": right_condition,
                "evaluation_form": key,
                "label": label,
                "shared_records": total,
                "both_correct": both_correct,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "both_wrong": both_wrong,
                "left_accuracy": left_acc,
                "right_accuracy": right_acc,
                "accuracy_difference_right_minus_left": (right_acc - left_acc) if left_acc is not None else None,
                "exact_mcnemar_p_value": exact_two_sided_binomial_p(left_only, right_only),
                "holm_adjusted_p_value": None,
            }
        )
    dim_rows = [row for row in stats_rows if row["evaluation_form"] != "overall"]
    holm_adjust(dim_rows)
    by_form = {row["evaluation_form"]: row["holm_adjusted_p_value"] for row in dim_rows}
    for row in stats_rows:
        if row["evaluation_form"] in by_form:
            row["holm_adjusted_p_value"] = by_form[row["evaluation_form"]]
    return stats_rows, rescues, regressions


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = training_stats(args.train_file)
    predictions_by_condition = {}
    summary_rows = []

    for condition in SUMMARY_CONDITIONS:
        predictions = load_predictions(args, condition)
        if not predictions:
            continue
        predictions_by_condition[condition] = predictions
        summary = summarize_predictions(predictions, condition)
        write_json(args.output_dir / condition / "summary_recomputed.json", summary)
        summary_rows.append(row_for_summary(summary, condition_metadata(condition, train_stats), predictions))

    comparison_stats = []
    rescues = []
    regressions = []
    for left, right in PAIRED_256_512:
        if left not in predictions_by_condition or right not in predictions_by_condition:
            continue
        rows, pair_rescues, pair_regressions = comparison_rows(left, right, predictions_by_condition[left], predictions_by_condition[right])
        comparison_stats.extend(rows)
        rescues.extend(pair_rescues)
        regressions.extend(pair_regressions)

    summary_fields = [
        "condition", "prompt_protocol", "generation_budget", "adapter_protocol", "adapter_dir",
        "training_rows", "training_unique_base_ids", "training_original_examples",
        "trainable_parameters", "trainable_percentage", "overall_accuracy", "overall_correct",
        "overall_total", "original_accuracy", "dim1_accuracy", "dim2_accuracy", "dim4_accuracy",
        "dim6_accuracy", "mean_transformed_accuracy", "worst_dimension_accuracy", "robustness_gap",
        "matched_consistency", "complete_case_base_ids", "average_generated_tokens",
        "average_latency_seconds", "average_tokens_per_second", "cap_hits", "cap_hit_rate",
        "missing_final_answer_markers", "missing_final_answer_marker_rate",
    ]
    write_csv(args.output_dir / "results_summary_256_vs_512.csv", summary_rows, summary_fields)
    write_csv(
        args.output_dir / "paired_256_vs_512_mcnemar.csv",
        comparison_stats,
        [
            "comparison", "left_condition", "right_condition", "evaluation_form", "label",
            "shared_records", "both_correct", "left_only_correct", "right_only_correct", "both_wrong",
            "left_accuracy", "right_accuracy", "accuracy_difference_right_minus_left",
            "exact_mcnemar_p_value", "holm_adjusted_p_value",
        ],
    )
    write_jsonl(args.output_dir / "rescues_512_vs_256.jsonl", rescues)
    write_jsonl(args.output_dir / "regressions_512_vs_256.jsonl", regressions)
    write_json(
        args.output_dir / "analysis_note.json",
        {
            "purpose": "Compare existing 256-token reasoning evaluations with a separate 512-token validation-only diagnostic.",
            "only_intended_inference_change": "max_new_tokens 256 -> 512",
            "test_set_used": False,
            "latency_caution": "Do not interpret performance differences solely from latency because runs may use different hardware.",
            "conditions_requested": SUMMARY_CONDITIONS,
            "paired_256_vs_512": PAIRED_256_512,
            "results_root_256": str(args.results_root_256),
            "results_root_512": str(args.results_root_512),
        },
    )
    print(f"Analyzed {len(summary_rows)} 256/512 diagnostic conditions into {args.output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    analyze(parse_args(argv))


if __name__ == "__main__":
    main()

