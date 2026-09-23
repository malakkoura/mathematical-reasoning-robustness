#!/usr/bin/env python3
"""Analyze validation results for the reasoning budget-control experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from budget_control_utils import (
    AUGMENTATION_DIR,
    BUDGET_CONDITIONS,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    OUTPUT_ROOT,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PRIMARY_COMPARISONS,
    composition,
    source_training_path,
    trainer_epoch_group_example_counts,
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
    parser = argparse.ArgumentParser(description="Analyze budget-control validation results.")
    parser.add_argument("--results_root", type=Path, default=Path(OUTPUT_ROOT) / "validation_evaluations")
    parser.add_argument("--adapters_root", type=Path, default=Path(OUTPUT_ROOT) / "adapters")
    parser.add_argument("--output_dir", type=Path, default=Path(OUTPUT_ROOT) / "validation_analysis")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    extra_keys = sorted({key for row in rows for key in row} - set(fieldnames))
    if extra_keys:
        raise ValueError(f"{path} rows contain fields not in fieldnames: {extra_keys}")
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
    path = root / condition / "predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    for row in rows:
        row["condition"] = condition
    return rows


def training_stats(condition: str, adapters_root: Path) -> Dict[str, Any]:
    rows = read_jsonl(source_training_path(condition))
    comp = composition(rows)
    exposure = load_json(adapters_root / condition / "training_exposure.json")
    metrics = load_json(adapters_root / condition / "training_metrics.json")
    estimated = trainer_epoch_group_example_counts(comp["rows"])
    examples_presented = exposure.get("examples_presented", estimated["estimated_example_presentations"])
    effective_epochs = exposure.get("effective_epochs", estimated["effective_epochs"])
    return {
        "source_dataset_size": comp["rows"],
        "training_unique_base_ids": comp["unique_base_ids"],
        "training_original_examples": comp["counts_by_dimension"].get("0", 0),
        "training_dim1_examples": comp["counts_by_dimension"].get("1", 0),
        "training_dim2_examples": comp["counts_by_dimension"].get("2", 0),
        "training_dim4_examples": comp["counts_by_dimension"].get("4", 0),
        "training_dim6_examples": comp["counts_by_dimension"].get("6", 0),
        "optimizer_steps": exposure.get("actual_completed_optimizer_steps", metrics.get("actual_completed_optimizer_steps", MAX_STEPS)),
        "max_steps": MAX_STEPS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "nominal_full_accumulation_examples": estimated["nominal_full_accumulation_examples"],
        "estimated_example_presentations": estimated["estimated_example_presentations"],
        "examples_presented": examples_presented,
        "example_presentation_count_source": exposure.get("example_presentation_count_source", "estimated_transformers_epoch_boundary_grouping"),
        "effective_epochs": effective_epochs,
        "observed_nonpadding_tokens_processed": exposure.get("observed_nonpadding_tokens_processed"),
        "estimated_nonpadding_tokens_processed": exposure.get("estimated_nonpadding_tokens_processed"),
        "measured_dataset_nonpadding_tokens": exposure.get("measured_dataset_nonpadding_tokens"),
        "mean_nonpadding_tokens_per_example": exposure.get("mean_nonpadding_tokens_per_example"),
        "trainable_parameter_count": exposure.get("trainable_parameter_count"),
        "final_training_loss": exposure.get("final_training_loss", metrics.get("train_loss")),
    }


def diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cap_hits = sum(1 for row in rows if row.get("hit_max_new_tokens"))
    missing = sum(1 for row in rows if row.get("missing_final_answer_marker"))
    total = len(rows)
    return {
        "cap_hits": cap_hits,
        "cap_hit_rate": cap_hits / total if total else None,
        "missing_final_answer_markers": missing,
        "missing_final_answer_marker_rate": missing / total if total else None,
    }


def row_for_summary(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], adapters_root: Path) -> Dict[str, Any]:
    by_dim = summary["by_dimension"]
    return {
        "condition": summary["condition"],
        **training_stats(summary["condition"], adapters_root),
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
        stats.append(
            {
                "comparison": f"{left_condition}_vs_{right_condition}",
                "left_condition": left_condition,
                "right_condition": right_condition,
                "evaluation_form": form,
                "label": label,
                "shared_records": total,
                "both_correct": both,
                "left_only_correct": lo,
                "right_only_correct": ro,
                "rescues": ro,
                "regressions": lo,
                "both_wrong": wrong,
                "left_accuracy": la,
                "right_accuracy": ra,
                "accuracy_difference_right_minus_left": (ra - la) if la is not None else None,
                "percentage_point_difference_right_minus_left": 100 * (ra - la) if la is not None else None,
                "exact_mcnemar_p_value": exact_two_sided_binomial_p(lo, ro),
                "holm_adjusted_p_value": None,
            }
        )
    dim_rows = [row for row in stats if row["evaluation_form"] != "overall"]
    holm_adjust(dim_rows)
    adjusted = {row["evaluation_form"]: row["holm_adjusted_p_value"] for row in dim_rows}
    for row in stats:
        if row["evaluation_form"] in adjusted:
            row["holm_adjusted_p_value"] = adjusted[row["evaluation_form"]]
    return stats, rescues, regressions


def interpretation_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition = {row["condition"]: row for row in summary_rows}
    baseline = by_condition.get("budget_original_only")
    if not baseline:
        return []
    rows = []
    for condition, row in by_condition.items():
        if condition == "budget_original_only":
            continue
        rows.append(
            {
                "condition": condition,
                "dim2_change_vs_original_only_pp": 100 * (row["dim2_accuracy"] - baseline["dim2_accuracy"]) if row.get("dim2_accuracy") is not None and baseline.get("dim2_accuracy") is not None else None,
                "dim4_change_vs_original_only_pp": 100 * (row["dim4_accuracy"] - baseline["dim4_accuracy"]) if row.get("dim4_accuracy") is not None and baseline.get("dim4_accuracy") is not None else None,
                "original_change_vs_original_only_pp": 100 * (row["original_accuracy"] - baseline["original_accuracy"]) if row.get("original_accuracy") is not None and baseline.get("original_accuracy") is not None else None,
                "robustness_gap_change_vs_original_only_pp": 100 * (row["robustness_gap"] - baseline["robustness_gap"]) if row.get("robustness_gap") is not None and baseline.get("robustness_gap") is not None else None,
                "interpretation_note": "A smaller robustness gap can be caused by higher transformed accuracy, lower original accuracy, or both; inspect original_change_vs_original_only_pp alongside transformed-dimension changes.",
            }
        )
    return rows


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {condition: load_predictions(args.results_root, condition) for condition in BUDGET_CONDITIONS}
    predictions = {condition: rows for condition, rows in predictions.items() if rows}
    summary_rows = []
    by_variant = []
    exposure_rows = []
    comparisons = []
    rescues = []
    regressions = []

    for condition, rows in predictions.items():
        summary = summarize_predictions(rows, condition)
        write_json(args.output_dir / condition / "summary_recomputed.json", summary)
        stats = training_stats(condition, args.adapters_root)
        exposure_rows.append({"condition": condition, "dimension_composition": json.dumps({k: stats[k] for k in ["training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples"]}), **stats})
        summary_rows.append(row_for_summary(summary, rows, args.adapters_root))
        for dim_label, metric in sorted(summary["by_dimension"].items()):
            by_variant.append({"condition": condition, "evaluation_form": dim_label, **metric})

    for left, right in PRIMARY_COMPARISONS:
        if left in predictions and right in predictions:
            rows, r1, r2 = comparison_rows(left, right, predictions[left], predictions[right])
            comparisons.extend(rows)
            rescues.extend(r1)
            regressions.extend(r2)

    summary_fields = ["condition", "source_dataset_size", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "optimizer_steps", "max_steps", "gradient_accumulation_steps", "nominal_full_accumulation_examples", "estimated_example_presentations", "examples_presented", "example_presentation_count_source", "effective_epochs", "measured_dataset_nonpadding_tokens", "mean_nonpadding_tokens_per_example", "observed_nonpadding_tokens_processed", "estimated_nonpadding_tokens_processed", "trainable_parameter_count", "final_training_loss", "overall_accuracy", "overall_correct", "overall_total", "original_accuracy", "dim1_accuracy", "dim2_accuracy", "dim4_accuracy", "dim6_accuracy", "mean_transformed_accuracy", "worst_dimension_accuracy", "robustness_gap", "matched_consistency", "complete_case_base_ids", "average_generated_tokens", "average_latency_seconds", "average_tokens_per_second", "cap_hits", "cap_hit_rate", "missing_final_answer_markers", "missing_final_answer_marker_rate"]
    write_csv(args.output_dir / "results_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "results_by_variant.csv", by_variant, ["condition", "evaluation_form", "correct", "total", "accuracy"])
    write_csv(args.output_dir / "training_exposure.csv", exposure_rows, ["condition", "source_dataset_size", "dimension_composition", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "optimizer_steps", "max_steps", "gradient_accumulation_steps", "nominal_full_accumulation_examples", "estimated_example_presentations", "examples_presented", "example_presentation_count_source", "effective_epochs", "measured_dataset_nonpadding_tokens", "mean_nonpadding_tokens_per_example", "estimated_nonpadding_tokens_processed", "observed_nonpadding_tokens_processed", "trainable_parameter_count", "final_training_loss"])
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons, ["comparison", "left_condition", "right_condition", "evaluation_form", "label", "shared_records", "both_correct", "left_only_correct", "right_only_correct", "rescues", "regressions", "both_wrong", "left_accuracy", "right_accuracy", "accuracy_difference_right_minus_left", "percentage_point_difference_right_minus_left", "exact_mcnemar_p_value", "holm_adjusted_p_value"])
    write_csv(args.output_dir / "interpretation_checks.csv", interpretation_rows(summary_rows), ["condition", "dim2_change_vs_original_only_pp", "dim4_change_vs_original_only_pp", "original_change_vs_original_only_pp", "robustness_gap_change_vs_original_only_pp", "interpretation_note"])
    write_jsonl(args.output_dir / "rescues.jsonl", rescues)
    write_jsonl(args.output_dir / "regressions.jsonl", regressions)
    write_json(args.output_dir / "analysis_note.json", {"test_set_used": False, "max_new_tokens": MAX_NEW_TOKENS, "primary_comparison": "budget_original_only_vs_budget_original_plus_dim2", "primary_comparisons": PRIMARY_COMPARISONS})
    print(f"Analyzed {len(summary_rows)} budget-control conditions into {args.output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    analyze(parse_args(argv))


if __name__ == "__main__":
    main()
