#!/usr/bin/env python3
"""Analyze Qwen3-4B reasoning-augmentation validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from reasoning_aug_utils import (
    AUGMENTATION_DIR,
    DIRECT_CONDITION_MAP,
    EVALUATION_CONDITIONS,
    OUTPUT_ROOT,
    PRIMARY_COMPARISONS,
    REASONING_SUPERVISION_OUTPUT_ROOT,
    adapter_dir_for,
    direct_validation_dir_for,
    reasoning_training_path,
)

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import dimension_key, read_jsonl, summarize_predictions, write_json  # noqa: E402


FORMS = [
    ("overall", None, "Overall"),
    ("original", 0, "Original"),
    ("dim1_paraphrasing", 1, "Dim1 paraphrasing"),
    ("dim2_word_scrambling", 2, "Dim2 word scrambling"),
    ("dim4_irrelevant_information", 4, "Dim4 irrelevant information"),
    ("dim6_numerical_variation", 6, "Dim6 numerical variation"),
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reasoning-augmentation validation results.")
    parser.add_argument("--results_root", type=Path, default=Path(OUTPUT_ROOT) / "validation_evaluations")
    parser.add_argument("--output_dir", type=Path, default=Path(OUTPUT_ROOT) / "validation_analysis")
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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(root: Path, condition: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(root / condition / "predictions.jsonl") if (root / condition / "predictions.jsonl").exists() else []
    for row in rows:
        row["condition"] = condition
    return rows


def training_stats(condition: str) -> Dict[str, Any]:
    path = reasoning_training_path(condition)
    if not path.exists() and condition == "reasoning_original_only":
        path = Path("experiments/cogmath_reasoning_supervision/data/train_original_only_reasoning.jsonl")
    if not path.exists():
        return {"training_rows": None, "training_unique_base_ids": None, "training_original_examples": None, "training_dim1_examples": None, "training_dim2_examples": None, "training_dim4_examples": None, "training_dim6_examples": None}
    rows = read_jsonl(path)
    by_dim = Counter(int(row["dimension"]) for row in rows)
    return {
        "training_rows": len(rows),
        "training_unique_base_ids": len({row["base_id"] for row in rows}),
        "training_original_examples": by_dim.get(0, 0),
        "training_dim1_examples": by_dim.get(1, 0),
        "training_dim2_examples": by_dim.get(2, 0),
        "training_dim4_examples": by_dim.get(4, 0),
        "training_dim6_examples": by_dim.get(6, 0),
    }


def diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    caps = [row.get("hit_max_new_tokens") for row in rows if row.get("hit_max_new_tokens") is not None]
    markers = [row.get("missing_final_answer_marker") for row in rows if row.get("missing_final_answer_marker") is not None]
    cap_hits = sum(1 for value in caps if value)
    missing = sum(1 for value in markers if value)
    return {
        "cap_hits": cap_hits if caps else None,
        "cap_hit_rate": cap_hits / len(caps) if caps else None,
        "missing_final_answer_markers": missing if markers else None,
        "missing_final_answer_marker_rate": missing / len(markers) if markers else None,
    }


def row_for_summary(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_dim = summary["by_dimension"]
    return {
        "condition": summary["condition"],
        **training_stats(summary["condition"]),
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
        **diagnostics(rows),
    }


def exact_two_sided_binomial_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm_adjust(rows: List[Dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["exact_mcnemar_p_value"])
    running = 0.0
    m = len(ordered)
    for rank, (idx, row) in enumerate(ordered, 1):
        running = max(running, min(1.0, (m - rank + 1) * row["exact_mcnemar_p_value"]))
        rows[idx]["holm_adjusted_p_value"] = running


def keyed(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["record_id"]: row for row in rows}


def comparison_rows(left_condition: str, right_condition: str, left_rows: Sequence[Dict[str, Any]], right_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    left = keyed(left_rows)
    right = keyed(right_rows)
    ids = sorted(set(left) & set(right))
    stats = []
    rescues = []
    regressions = []
    for form, dim, label in FORMS:
        form_ids = ids if dim is None else [rid for rid in ids if int(left[rid]["dimension"]) == dim]
        both = lo = ro = wrong = 0
        for rid in form_ids:
            lc, rc = bool(left[rid].get("correct")), bool(right[rid].get("correct"))
            if lc and rc:
                both += 1
            elif lc and not rc:
                lo += 1
                regressions.append({"comparison": f"{left_condition}_vs_{right_condition}", "evaluation_form": form, "record": right[rid]})
            elif not lc and rc:
                ro += 1
                rescues.append({"comparison": f"{left_condition}_vs_{right_condition}", "evaluation_form": form, "record": right[rid]})
            else:
                wrong += 1
        total = len(form_ids)
        la = (both + lo) / total if total else None
        ra = (both + ro) / total if total else None
        stats.append({"comparison": f"{left_condition}_vs_{right_condition}", "left_condition": left_condition, "right_condition": right_condition, "evaluation_form": form, "label": label, "shared_records": total, "both_correct": both, "left_only_correct": lo, "right_only_correct": ro, "both_wrong": wrong, "left_accuracy": la, "right_accuracy": ra, "accuracy_difference_right_minus_left": (ra - la) if la is not None else None, "exact_mcnemar_p_value": exact_two_sided_binomial_p(lo, ro), "holm_adjusted_p_value": None})
    dim_rows = [row for row in stats if row["evaluation_form"] != "overall"]
    holm_adjust(dim_rows)
    adjusted = {row["evaluation_form"]: row["holm_adjusted_p_value"] for row in dim_rows}
    for row in stats:
        if row["evaluation_form"] in adjusted:
            row["holm_adjusted_p_value"] = adjusted[row["evaluation_form"]]
    return stats, rescues, regressions


def direct_predictions(reasoning_condition: str) -> List[Dict[str, Any]]:
    path = direct_validation_dir_for(reasoning_condition) / "predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    direct_condition = DIRECT_CONDITION_MAP[reasoning_condition]
    for row in rows:
        row["condition"] = direct_condition
    return rows


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {condition: load_predictions(args.results_root, condition) for condition in EVALUATION_CONDITIONS}
    predictions = {condition: rows for condition, rows in predictions.items() if rows}
    summary_rows = []
    by_variant = []
    latency_rows = []
    transfer_rows = []
    for condition, rows in predictions.items():
        summary = summarize_predictions(rows, condition)
        write_json(args.output_dir / condition / "summary_recomputed.json", summary)
        summary_rows.append(row_for_summary(summary, rows))
        for dim_label, metric in sorted(summary["by_dimension"].items()):
            by_variant.append({"condition": condition, "evaluation_form": dim_label, **training_stats(condition), **metric})
        transfer = {"training_condition": condition, **training_stats(condition)}
        for dim_label, metric in sorted(summary["by_dimension"].items()):
            transfer[dim_label] = metric["accuracy"]
        transfer_rows.append(transfer)
        latency_rows.append({"condition": condition, **training_stats(condition), **summary["latency"], **diagnostics(rows)})

    comparisons = []
    rescues = []
    regressions = []
    for left, right in PRIMARY_COMPARISONS:
        if left in predictions and right in predictions:
            rows, r1, r2 = comparison_rows(left, right, predictions[left], predictions[right])
            comparisons.extend(rows); rescues.extend(r1); regressions.extend(r2)

    cross_rows = []
    for reasoning_condition, reasoning_rows in predictions.items():
        direct_rows = direct_predictions(reasoning_condition)
        if not direct_rows:
            continue
        direct_summary = summarize_predictions(direct_rows, DIRECT_CONDITION_MAP[reasoning_condition])
        reasoning_summary = summarize_predictions(reasoning_rows, reasoning_condition)
        direct_dim2 = direct_summary["by_dimension"].get("dim2_word_scrambling", {}).get("accuracy")
        reasoning_dim2 = reasoning_summary["by_dimension"].get("dim2_word_scrambling", {}).get("accuracy")
        cross_rows.append({
            "reasoning_condition": reasoning_condition,
            "direct_condition": DIRECT_CONDITION_MAP[reasoning_condition],
            "direct_overall_accuracy": direct_summary["overall"]["accuracy"],
            "reasoning_overall_accuracy": reasoning_summary["overall"]["accuracy"],
            "direct_dim2_accuracy": direct_dim2,
            "reasoning_dim2_accuracy": reasoning_dim2,
            "reasoning_minus_direct_dim2_accuracy": (reasoning_dim2 - direct_dim2) if direct_dim2 is not None and reasoning_dim2 is not None else None,
            "direct_robustness_gap": direct_summary["robustness_gap"],
            "reasoning_robustness_gap": reasoning_summary["robustness_gap"],
        })
    if {"reasoning_original_only", "reasoning_original_plus_dim2"} <= set(predictions):
        r0 = summarize_predictions(predictions["reasoning_original_only"], "reasoning_original_only")["by_dimension"].get("dim2_word_scrambling", {}).get("accuracy")
        r2 = summarize_predictions(predictions["reasoning_original_plus_dim2"], "reasoning_original_plus_dim2")["by_dimension"].get("dim2_word_scrambling", {}).get("accuracy")
        cross_rows.append({"reasoning_condition": "augmentation_gain_dim2_reasoning", "direct_condition": "augmentation_gain_dim2_direct", "direct_overall_accuracy": None, "reasoning_overall_accuracy": None, "direct_dim2_accuracy": None, "reasoning_dim2_accuracy": None, "reasoning_minus_direct_dim2_accuracy": None, "direct_robustness_gap": None, "reasoning_robustness_gap": None, "reasoning_dim2_gain_over_original_only": (r2 - r0) if r0 is not None and r2 is not None else None})

    summary_fields = ["condition", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "overall_accuracy", "overall_correct", "overall_total", "original_accuracy", "dim1_accuracy", "dim2_accuracy", "dim4_accuracy", "dim6_accuracy", "mean_transformed_accuracy", "worst_dimension_accuracy", "robustness_gap", "matched_consistency", "complete_case_base_ids", "average_generated_tokens", "average_latency_seconds", "average_tokens_per_second", "cap_hits", "cap_hit_rate", "missing_final_answer_markers", "missing_final_answer_marker_rate"]
    write_csv(args.output_dir / "results_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "results_by_variant.csv", by_variant, ["condition", "evaluation_form", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "correct", "total", "accuracy"])
    write_csv(args.output_dir / "transfer_matrix.csv", transfer_rows, ["training_condition", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples"] + [dimension_key(dim) for dim in [0, 1, 2, 4, 6]])
    write_csv(args.output_dir / "latency_generation_diagnostics.csv", latency_rows, ["condition", "training_rows", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "average_latency_seconds", "median_latency_seconds", "average_generated_tokens", "average_tokens_per_second", "cap_hits", "cap_hit_rate", "missing_final_answer_markers", "missing_final_answer_marker_rate"])
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons, ["comparison", "left_condition", "right_condition", "evaluation_form", "label", "shared_records", "both_correct", "left_only_correct", "right_only_correct", "both_wrong", "left_accuracy", "right_accuracy", "accuracy_difference_right_minus_left", "exact_mcnemar_p_value", "holm_adjusted_p_value"])
    write_csv(args.output_dir / "cross_protocol_direct_vs_reasoning.csv", cross_rows, sorted({key for row in cross_rows for key in row}))
    write_jsonl(args.output_dir / "rescues.jsonl", rescues)
    write_jsonl(args.output_dir / "regressions.jsonl", regressions)
    write_json(args.output_dir / "analysis_note.json", {"test_set_used": False, "max_new_tokens": 512, "reasoning_supervision_output_root": REASONING_SUPERVISION_OUTPUT_ROOT, "primary_comparisons": PRIMARY_COMPARISONS})
    print(f"Analyzed {len(summary_rows)} reasoning-augmentation conditions into {args.output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    analyze(parse_args(argv))


if __name__ == "__main__":
    main()

