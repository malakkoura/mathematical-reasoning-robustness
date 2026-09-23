#!/usr/bin/env python3
"""Analyze LoRA-placement validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from placement_utils import (
    ALL_ANALYSIS_CONDITIONS,
    AUGMENTATION_DIR,
    BUDGET_REFERENCE_CONDITION,
    CROSS_PLACEMENT_PLUS_DIM2_COMPARISONS,
    MAX_NEW_TOKENS,
    OUTPUT_ROOT,
    PLACEMENTS,
    PRIMARY_WITHIN_PLACEMENT_COMPARISONS,
    adapter_dir_for,
    condition_placement,
    condition_variant,
    parameter_estimates,
    source_training_path,
    validation_dir_for,
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LoRA-placement validation results.")
    parser.add_argument("--placement_results_root", type=Path, default=Path(OUTPUT_ROOT) / "validation_evaluations")
    parser.add_argument("--budget_results_root", type=Path, default=None)
    parser.add_argument("--placement_adapters_root", type=Path, default=Path(OUTPUT_ROOT) / "adapters")
    parser.add_argument("--budget_adapters_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path(OUTPUT_ROOT) / "validation_analysis")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    extra = sorted({key for row in rows for key in row} - set(fieldnames))
    if extra:
        raise ValueError(f"{path} rows contain fields not in fieldnames: {extra}")
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


def condition_results_dir(condition: str, placement_results_root: Path, budget_results_root: Optional[Path]) -> Path:
    if condition in BUDGET_REFERENCE_CONDITION:
        root = budget_results_root if budget_results_root is not None else validation_dir_for(condition).parent
        return root / BUDGET_REFERENCE_CONDITION[condition]
    return placement_results_root / condition


def condition_adapter_dir(condition: str, placement_adapters_root: Path, budget_adapters_root: Optional[Path]) -> Path:
    if condition in BUDGET_REFERENCE_CONDITION:
        root = budget_adapters_root if budget_adapters_root is not None else adapter_dir_for(condition).parent
        return root / BUDGET_REFERENCE_CONDITION[condition]
    return placement_adapters_root / condition


def load_predictions(condition: str, placement_results_root: Path, budget_results_root: Optional[Path]) -> List[Dict[str, Any]]:
    path = condition_results_dir(condition, placement_results_root, budget_results_root) / "predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    for row in rows:
        row["condition"] = condition
    return rows


def training_stats(condition: str, placement_adapters_root: Path, budget_adapters_root: Optional[Path]) -> Dict[str, Any]:
    placement = condition_placement(condition)
    adapter_dir = condition_adapter_dir(condition, placement_adapters_root, budget_adapters_root)
    rows = read_jsonl(source_training_path(condition))
    parameter_report = load_json(adapter_dir / "parameter_report.json")
    exposure = load_json(adapter_dir / "training_exposure.json")
    estimates = parameter_estimates()[placement]
    return {
        "placement": placement,
        "variant": condition_variant(condition),
        "target_modules": ",".join(PLACEMENTS[placement]),
        "source_dataset_rows": len(rows),
        "source_unique_base_ids": len({row["base_id"] for row in rows}),
        "trainable_parameters": parameter_report.get("trainable_parameters", exposure.get("trainable_parameter_count", estimates["estimated_trainable_parameters"])),
        "trainable_percentage": parameter_report.get("trainable_percentage", exposure.get("trainable_percentage", estimates["estimated_trainable_percentage"])),
        "parameter_count_source": "parameter_report.json" if parameter_report else "analytic_estimate_pending_cluster_parameter_report",
        "optimizer_steps": exposure.get("actual_completed_optimizer_steps"),
        "examples_presented": exposure.get("examples_presented"),
        "effective_epochs": exposure.get("effective_epochs"),
        "observed_nonpadding_tokens_processed": exposure.get("observed_nonpadding_tokens_processed"),
        "final_training_loss": exposure.get("final_training_loss"),
    }


def diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    cap_hits = sum(1 for row in rows if row.get("hit_max_new_tokens"))
    missing = sum(1 for row in rows if row.get("missing_final_answer_marker"))
    return {
        "cap_hits": cap_hits,
        "cap_hit_rate": cap_hits / total if total else None,
        "missing_final_answer_markers": missing,
        "missing_final_answer_marker_rate": missing / total if total else None,
    }


def summary_row(condition: str, rows: Sequence[Dict[str, Any]], placement_adapters_root: Path, budget_adapters_root: Optional[Path]) -> Dict[str, Any]:
    summary = summarize_predictions(rows, condition)
    by_dim = summary["by_dimension"]
    return {
        "condition": condition,
        **training_stats(condition, placement_adapters_root, budget_adapters_root),
        "overall_accuracy": summary["overall"]["accuracy"],
        "original_accuracy": summary["original"]["accuracy"],
        "dim1_accuracy": by_dim.get("dim1_paraphrasing", {}).get("accuracy"),
        "dim2_accuracy": by_dim.get("dim2_word_scrambling", {}).get("accuracy"),
        "dim4_accuracy": by_dim.get("dim4_irrelevant_information", {}).get("accuracy"),
        "dim6_accuracy": by_dim.get("dim6_numerical_variation", {}).get("accuracy"),
        "mean_transformed_accuracy": summary["mean_variant_accuracy"]["accuracy"],
        "worst_dimension_accuracy": summary["worst_dimension_accuracy"],
        "robustness_gap": summary["robustness_gap"],
        "matched_consistency": summary["all_forms_correct_rate"],
        "average_generated_tokens": summary["latency"]["average_generated_tokens"],
        **diagnostics(rows),
    }


def exact_two_sided_binomial_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm_adjust(rows: List[Dict[str, Any]], p_key: str = "exact_mcnemar_p_value") -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1][p_key])
    running = 0.0
    m = len(ordered)
    for rank, (idx, row) in enumerate(ordered, 1):
        running = max(running, min(1.0, (m - rank + 1) * row[p_key]))
        rows[idx]["holm_adjusted_p_value"] = running


def keyed(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["record_id"]: row for row in rows}


def paired_comparison(left_condition: str, right_condition: str, left_rows: Sequence[Dict[str, Any]], right_rows: Sequence[Dict[str, Any]], forms: Sequence[Tuple[str, Optional[int], str]] = FORMS) -> List[Dict[str, Any]]:
    left = keyed(left_rows)
    right = keyed(right_rows)
    ids = sorted(set(left) & set(right))
    out = []
    for form, dim, label in forms:
        form_ids = ids if dim is None else [rid for rid in ids if int(left[rid]["dimension"]) == dim]
        both = lo = ro = wrong = 0
        for rid in form_ids:
            lc = bool(left[rid].get("correct"))
            rc = bool(right[rid].get("correct"))
            if lc and rc:
                both += 1
            elif lc and not rc:
                lo += 1
            elif not lc and rc:
                ro += 1
            else:
                wrong += 1
        total = len(form_ids)
        la = (both + lo) / total if total else None
        ra = (both + ro) / total if total else None
        out.append({
            "comparison": f"{left_condition}_vs_{right_condition}",
            "left_condition": left_condition,
            "right_condition": right_condition,
            "placement_left": condition_placement(left_condition),
            "placement_right": condition_placement(right_condition),
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
        })
    return out


def gain_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition = {row["condition"]: row for row in summary_rows}
    out = []
    for placement in PLACEMENTS:
        left = by_condition.get(f"{placement}_original_only")
        right = by_condition.get(f"{placement}_plus_dim2")
        if not left or not right:
            continue
        out.append({
            "placement": placement,
            "target_modules": right["target_modules"],
            "trainable_parameters": right["trainable_parameters"],
            "trainable_percentage": right["trainable_percentage"],
            "parameter_count_source": right["parameter_count_source"],
            "original_only_dim2": left["dim2_accuracy"],
            "plus_dim2_dim2": right["dim2_accuracy"],
            "dim2_gain": right["dim2_accuracy"] - left["dim2_accuracy"] if left["dim2_accuracy"] is not None and right["dim2_accuracy"] is not None else None,
            "dim2_gain_pp": 100 * (right["dim2_accuracy"] - left["dim2_accuracy"]) if left["dim2_accuracy"] is not None and right["dim2_accuracy"] is not None else None,
            "overall_before": left["overall_accuracy"],
            "overall_after": right["overall_accuracy"],
            "overall_gain": right["overall_accuracy"] - left["overall_accuracy"],
            "original_gain": right["original_accuracy"] - left["original_accuracy"],
            "dim1_gain": right["dim1_accuracy"] - left["dim1_accuracy"],
            "dim4_gain": right["dim4_accuracy"] - left["dim4_accuracy"],
            "dim6_gain": right["dim6_accuracy"] - left["dim6_accuracy"],
            "mean_transformed_gain": right["mean_transformed_accuracy"] - left["mean_transformed_accuracy"],
            "worst_dimension_gain": right["worst_dimension_accuracy"] - left["worst_dimension_accuracy"],
            "robustness_gap_change": right["robustness_gap"] - left["robustness_gap"],
            "matched_consistency_gain": right["matched_consistency"] - left["matched_consistency"],
        })
    return out


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {condition: load_predictions(condition, args.placement_results_root, args.budget_results_root) for condition in ALL_ANALYSIS_CONDITIONS}
    predictions = {condition: rows for condition, rows in predictions.items() if rows}
    summary_rows = [summary_row(condition, rows, args.placement_adapters_root, args.budget_adapters_root) for condition, rows in predictions.items()]
    for condition, rows in predictions.items():
        write_json(args.output_dir / condition / "summary_recomputed.json", summarize_predictions(rows, condition))

    within = []
    for left, right in PRIMARY_WITHIN_PLACEMENT_COMPARISONS:
        if left in predictions and right in predictions:
            within.extend(paired_comparison(left, right, predictions[left], predictions[right]))
    dim2_within = [row for row in within if row["evaluation_form"] == "dim2_word_scrambling"]
    holm_adjust(dim2_within)
    adjusted = {row["comparison"]: row["holm_adjusted_p_value"] for row in dim2_within}
    for row in within:
        if row["comparison"] in adjusted and row["evaluation_form"] == "dim2_word_scrambling":
            row["holm_adjusted_p_value"] = adjusted[row["comparison"]]

    cross = []
    dim2_only = [("dim2_word_scrambling", 2, "Dim2 word scrambling")]
    for left, right in CROSS_PLACEMENT_PLUS_DIM2_COMPARISONS:
        if left in predictions and right in predictions:
            cross.extend(paired_comparison(left, right, predictions[left], predictions[right], dim2_only))
    holm_adjust(cross)

    summary_fields = ["condition", "placement", "variant", "target_modules", "source_dataset_rows", "source_unique_base_ids", "trainable_parameters", "trainable_percentage", "parameter_count_source", "optimizer_steps", "examples_presented", "effective_epochs", "observed_nonpadding_tokens_processed", "final_training_loss", "overall_accuracy", "original_accuracy", "dim1_accuracy", "dim2_accuracy", "dim4_accuracy", "dim6_accuracy", "mean_transformed_accuracy", "worst_dimension_accuracy", "robustness_gap", "matched_consistency", "average_generated_tokens", "cap_hits", "cap_hit_rate", "missing_final_answer_markers", "missing_final_answer_marker_rate"]
    gain_fields = ["placement", "target_modules", "trainable_parameters", "trainable_percentage", "parameter_count_source", "original_only_dim2", "plus_dim2_dim2", "dim2_gain", "dim2_gain_pp", "overall_before", "overall_after", "overall_gain", "original_gain", "dim1_gain", "dim4_gain", "dim6_gain", "mean_transformed_gain", "worst_dimension_gain", "robustness_gap_change", "matched_consistency_gain"]
    comparison_fields = ["comparison", "left_condition", "right_condition", "placement_left", "placement_right", "evaluation_form", "label", "shared_records", "both_correct", "left_only_correct", "right_only_correct", "rescues", "regressions", "both_wrong", "left_accuracy", "right_accuracy", "accuracy_difference_right_minus_left", "percentage_point_difference_right_minus_left", "exact_mcnemar_p_value", "holm_adjusted_p_value"]
    write_csv(args.output_dir / "results_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "dim2_augmentation_gains_by_placement.csv", gain_rows(summary_rows), gain_fields)
    write_csv(args.output_dir / "within_placement_paired_comparisons.csv", within, comparison_fields)
    write_csv(args.output_dir / "cross_placement_plus_dim2_comparisons.csv", cross, comparison_fields)
    write_json(args.output_dir / "analysis_note.json", {
        "test_set_used": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "all_linear_references": BUDGET_REFERENCE_CONDITION,
        "bootstrap_ci_status": "not_implemented; cross-placement gain differences are descriptive unless paired McNemar comparisons are reported for plus-Dim2 predictions.",
        "interpretation_note": "Placement and adapter capacity co-vary because r=8 is fixed while target-module sets differ.",
    })
    print(f"Analyzed {len(summary_rows)} LoRA-placement conditions into {args.output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    analyze(parse_args(argv))


if __name__ == "__main__":
    main()
