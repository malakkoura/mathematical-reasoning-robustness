#!/usr/bin/env python3
"""Analyze unique-full Qwen3-4B augmentation predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from experiment_utils import dimension_key, read_jsonl, summarize_predictions, write_json
from unique_full_utils import PRIMARY_COMPARISONS, UNIQUE_FULL_DIR, UNIQUE_FULL_EVALUATION_CONDITIONS, UNIQUE_FULL_TRAINING_CONDITIONS


FORMS = [
    ("overall", None, "Overall"),
    ("original", 0, "Original"),
    ("dim1_paraphrasing", 1, "Dim1 paraphrasing"),
    ("dim2_word_scrambling", 2, "Dim2 word scrambling"),
    ("dim4_irrelevant_information", 4, "Dim4 irrelevant information"),
    ("dim6_numerical_variation", 6, "Dim6 numerical variation"),
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze unique-full Qwen3-4B augmentation predictions.")
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--training_data_dir", type=Path, default=UNIQUE_FULL_DIR)
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


def load_predictions(results_root: Path, condition: str) -> List[Dict[str, Any]]:
    path = results_root / condition / "predictions.jsonl"
    return read_jsonl(path) if path.exists() else []


def training_stats(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    stats = {}
    for condition, filename in UNIQUE_FULL_TRAINING_CONDITIONS.items():
        rows = read_jsonl(data_dir / filename)
        by_dim = Counter(int(row["dimension"]) for row in rows)
        stats[condition] = {
            "training_rows": len(rows),
            "training_unique_base_ids": len({row["base_id"] for row in rows}),
            "training_original_examples": by_dim.get(0, 0),
            "training_dim1_examples": by_dim.get(1, 0),
            "training_dim2_examples": by_dim.get(2, 0),
            "training_dim4_examples": by_dim.get(4, 0),
            "training_dim6_examples": by_dim.get(6, 0),
        }
    stats["base_untuned"] = {
        "training_rows": 0,
        "training_unique_base_ids": 0,
        "training_original_examples": 0,
        "training_dim1_examples": 0,
        "training_dim2_examples": 0,
        "training_dim4_examples": 0,
        "training_dim6_examples": 0,
    }
    return stats


def row_for_summary(summary: Dict[str, Any], train: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "condition": summary["condition"],
        **train,
        "overall_accuracy": summary["overall"]["accuracy"],
        "overall_correct": summary["overall"]["correct"],
        "overall_total": summary["overall"]["total"],
        "original_accuracy": summary["original"]["accuracy"],
        "mean_variant_accuracy": summary["mean_variant_accuracy"]["accuracy"],
        "mean_transformed_form_accuracy": summary["mean_variant_accuracy"]["accuracy"],
        "worst_dimension_accuracy": summary["worst_dimension_accuracy"],
        "robustness_gap": summary["robustness_gap"],
        "matched_consistency_all_forms_correct_rate": summary["all_forms_correct_rate"],
        "complete_case_base_ids": summary["matched_consistency_by_base_id"]["complete_case_base_ids"],
        "average_generated_tokens": summary["latency"]["average_generated_tokens"],
        "average_latency_seconds": summary["latency"]["average_latency_seconds"],
        "average_tokens_per_second": summary["latency"]["average_tokens_per_second"],
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


def comparison_rows(left_condition: str, right_condition: str, left_rows: Sequence[Dict[str, Any]], right_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                regressions.append({"comparison": f"{left_condition}_vs_{right_condition}", "evaluation_form": key, "left_condition": left_condition, "right_condition": right_condition, "record": right[rid]})
            elif not l_correct and r_correct:
                right_only += 1
                rescues.append({"comparison": f"{left_condition}_vs_{right_condition}", "evaluation_form": key, "left_condition": left_condition, "right_condition": right_condition, "record": right[rid]})
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
    output_dir = args.output_dir or args.results_root
    train_stats = training_stats(args.training_data_dir)
    summaries = []
    summary_rows = []
    by_variant_rows = []
    transfer_rows = []
    latency_rows = []
    predictions_by_condition = {}

    for condition in UNIQUE_FULL_EVALUATION_CONDITIONS:
        predictions = load_predictions(args.results_root, condition)
        if not predictions:
            continue
        predictions_by_condition[condition] = predictions
        summary = summarize_predictions(predictions, condition)
        summaries.append(summary)
        write_json(output_dir / condition / "summary_recomputed.json", summary)
        summary_rows.append(row_for_summary(summary, train_stats[condition]))
        for dim_label, metric in sorted(summary["by_dimension"].items()):
            by_variant_rows.append({"condition": condition, "evaluation_form": dim_label, **train_stats[condition], **metric})
        transfer_row = {"training_condition": condition, **train_stats[condition]}
        for dim_label, metric in sorted(summary["by_dimension"].items()):
            transfer_row[dim_label] = metric["accuracy"]
        transfer_rows.append(transfer_row)
        latency_rows.append({"condition": condition, **train_stats[condition], **summary["latency"]})

    comparison_stats = []
    rescues = []
    regressions = []
    for left, right in PRIMARY_COMPARISONS:
        if left not in predictions_by_condition or right not in predictions_by_condition:
            continue
        rows, pair_rescues, pair_regressions = comparison_rows(left, right, predictions_by_condition[left], predictions_by_condition[right])
        comparison_stats.extend(rows)
        rescues.extend(pair_rescues)
        regressions.extend(pair_regressions)

    write_csv(
        output_dir / "results_summary.csv",
        summary_rows,
        [
            "condition", "training_rows", "training_unique_base_ids", "training_original_examples",
            "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples",
            "overall_accuracy", "overall_correct", "overall_total", "original_accuracy",
            "mean_variant_accuracy", "mean_transformed_form_accuracy", "worst_dimension_accuracy", "robustness_gap",
            "matched_consistency_all_forms_correct_rate", "complete_case_base_ids",
            "average_generated_tokens", "average_latency_seconds", "average_tokens_per_second",
        ],
    )
    write_csv(output_dir / "results_by_variant.csv", by_variant_rows, ["condition", "evaluation_form", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "correct", "total", "accuracy"])
    write_csv(output_dir / "transfer_matrix.csv", transfer_rows, ["training_condition", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples"] + [dimension_key(dim) for dim in [0, 1, 2, 4, 6]])
    write_csv(output_dir / "latency_summary.csv", latency_rows, ["condition", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "average_latency_seconds", "median_latency_seconds", "average_generated_tokens", "average_tokens_per_second"])
    write_csv(output_dir / "paired_comparisons.csv", comparison_stats, ["comparison", "left_condition", "right_condition", "evaluation_form", "label", "shared_records", "both_correct", "left_only_correct", "right_only_correct", "both_wrong", "left_accuracy", "right_accuracy", "accuracy_difference_right_minus_left", "exact_mcnemar_p_value", "holm_adjusted_p_value"])
    write_jsonl(output_dir / "rescues.jsonl", rescues)
    write_jsonl(output_dir / "regressions.jsonl", regressions)
    write_json(
        output_dir / "analysis_note.json",
        {
            "interpretation": "Comparisons reflect the practical effect of adding all available augmentation data, including both transformation content and additional unique training exposure. Training dataset sizes are intentionally unequal.",
            "primary_comparisons": PRIMARY_COMPARISONS,
            "conditions_analyzed": [summary["condition"] for summary in summaries],
        },
    )
    print(f"Analyzed {len(summaries)} unique-full conditions into {output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    analyze(parse_args(argv))


if __name__ == "__main__":
    main()
