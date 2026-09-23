#!/usr/bin/env python3
"""Analyze validation results for the reasoning attention-control experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from attention_control_utils import (
    ALL_CONDITIONS,
    ATTENTION_TYPE_BY_CONDITION,
    BUDGET_CONDITION_BY_CAUSAL,
    FORMS,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    OUTPUT_ROOT,
    adapter_dir_for,
    composition,
    evaluation_dir_for,
    source_training_path_for_condition,
    trainer_epoch_group_example_counts,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cogmath_augmentation"))
from experiment_utils import read_jsonl, summarize_predictions, write_json  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reasoning attention-control validation results.")
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


def load_predictions(condition: str) -> List[Dict[str, Any]]:
    path = evaluation_dir_for(condition) / "predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    for row in rows:
        row["condition"] = condition
        row["conceptual_condition"] = condition
        if condition in BUDGET_CONDITION_BY_CAUSAL:
            row["reused_budget_condition"] = BUDGET_CONDITION_BY_CAUSAL[condition]
    return rows


def training_stats(condition: str) -> Dict[str, Any]:
    rows = read_jsonl(source_training_path_for_condition(condition))
    comp = composition(rows)
    adapter_dir = adapter_dir_for(condition)
    exposure = load_json(adapter_dir / "training_exposure.json")
    metrics = load_json(adapter_dir / "training_metrics.json")
    estimated = trainer_epoch_group_example_counts(comp["rows"])
    examples_presented = exposure.get("examples_presented", estimated["estimated_example_presentations"])
    effective_epochs = exposure.get("effective_epochs", estimated["effective_epochs"])
    return {
        "attention_type": ATTENTION_TYPE_BY_CONDITION[condition],
        "reused_budget_condition": BUDGET_CONDITION_BY_CAUSAL.get(condition),
        "source_dataset_size": comp["rows"],
        "training_unique_base_ids": comp["unique_base_ids"],
        "training_original_examples": comp["counts_by_dimension"].get("0", 0),
        "training_dim1_examples": comp["counts_by_dimension"].get("1", 0),
        "training_dim2_examples": comp["counts_by_dimension"].get("2", 0),
        "training_dim4_examples": comp["counts_by_dimension"].get("4", 0),
        "training_dim6_examples": comp["counts_by_dimension"].get("6", 0),
        "optimizer_steps": exposure.get("actual_completed_optimizer_steps", metrics.get("actual_completed_optimizer_steps", MAX_STEPS)),
        "max_steps": MAX_STEPS,
        "estimated_example_presentations": estimated["estimated_example_presentations"],
        "examples_presented": examples_presented,
        "example_presentation_count_source": exposure.get("example_presentation_count_source", "estimated_transformers_epoch_boundary_grouping"),
        "effective_epochs": effective_epochs,
        "measured_dataset_nonpadding_tokens": exposure.get("measured_dataset_nonpadding_tokens"),
        "mean_nonpadding_tokens_per_example": exposure.get("mean_nonpadding_tokens_per_example"),
        "observed_nonpadding_tokens_processed": exposure.get("observed_nonpadding_tokens_processed"),
        "estimated_nonpadding_tokens_processed": exposure.get("estimated_nonpadding_tokens_processed"),
        "trainable_parameter_count": exposure.get("trainable_parameter_count"),
        "final_training_loss": exposure.get("final_training_loss", metrics.get("train_loss")),
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


def row_for_summary(condition: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_predictions(rows, condition)
    by_dim = summary["by_dimension"]
    return {
        "condition": condition,
        **training_stats(condition),
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
    for rank, (index, row) in enumerate(ordered, 1):
        running = max(running, min(1.0, (m - rank + 1) * row["exact_mcnemar_p_value"]))
        rows[index]["holm_adjusted_p_value"] = running


def keyed(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["record_id"]: row for row in rows}


def comparison_rows(left_condition: str, right_condition: str, left_rows: Sequence[Dict[str, Any]], right_rows: Sequence[Dict[str, Any]], family: str) -> List[Dict[str, Any]]:
    left = keyed(left_rows)
    right = keyed(right_rows)
    ids = sorted(set(left) & set(right))
    stats = []
    for form, dim, label in FORMS:
        form_ids = ids if dim is None else [record_id for record_id in ids if int(left[record_id]["dimension"]) == dim]
        both = left_only = right_only = both_wrong = 0
        for record_id in form_ids:
            lc = bool(left[record_id].get("correct"))
            rc = bool(right[record_id].get("correct"))
            if lc and rc:
                both += 1
            elif lc:
                left_only += 1
            elif rc:
                right_only += 1
            else:
                both_wrong += 1
        total = len(form_ids)
        left_accuracy = (both + left_only) / total if total else None
        right_accuracy = (both + right_only) / total if total else None
        stats.append(
            {
                "comparison_family": family,
                "comparison": f"{left_condition}_vs_{right_condition}",
                "left_condition": left_condition,
                "right_condition": right_condition,
                "evaluation_form": form,
                "label": label,
                "shared_records": total,
                "both_correct": both,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "both_wrong": both_wrong,
                "left_accuracy": left_accuracy,
                "right_accuracy": right_accuracy,
                "accuracy_difference_right_minus_left": (right_accuracy - left_accuracy) if left_accuracy is not None else None,
                "percentage_point_difference_right_minus_left": 100 * (right_accuracy - left_accuracy) if left_accuracy is not None else None,
                "exact_mcnemar_p_value": exact_two_sided_binomial_p(left_only, right_only),
                "holm_adjusted_p_value": None,
            }
        )
    dim_rows = [row for row in stats if row["evaluation_form"] != "overall"]
    holm_adjust(dim_rows)
    adjusted = {row["evaluation_form"]: row["holm_adjusted_p_value"] for row in dim_rows}
    for row in stats:
        if row["evaluation_form"] in adjusted:
            row["holm_adjusted_p_value"] = adjusted[row["evaluation_form"]]
    return stats


def interaction_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition = {row["condition"]: row for row in summary_rows}
    rows = []
    for metric in ["overall_accuracy", "original_accuracy", "dim2_accuracy", "mean_transformed_accuracy", "robustness_gap"]:
        causal_original = by_condition.get("causal_original_only", {}).get(metric)
        causal_aug = by_condition.get("causal_original_plus_dim2", {}).get(metric)
        bidir_original = by_condition.get("prompt_bidir_original_only", {}).get(metric)
        bidir_aug = by_condition.get("prompt_bidir_original_plus_dim2", {}).get(metric)
        if None not in {causal_original, causal_aug, bidir_original, bidir_aug}:
            rows.append(
                {
                    "metric": metric,
                    "causal_dim2_gain": causal_aug - causal_original,
                    "prompt_bidir_dim2_gain": bidir_aug - bidir_original,
                    "interaction_difference_prompt_bidir_minus_causal": (bidir_aug - bidir_original) - (causal_aug - causal_original),
                    "note": "Descriptive difference-in-differences; inspect paired comparisons before interpretation.",
                }
            )
    return rows


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {condition: load_predictions(condition) for condition in ALL_CONDITIONS}
    predictions = {condition: rows for condition, rows in predictions.items() if rows}

    summary_rows = []
    by_variant = []
    comparisons = []
    for condition, rows in predictions.items():
        summary = summarize_predictions(rows, condition)
        write_json(args.output_dir / condition / "summary_recomputed.json", summary)
        summary_rows.append(row_for_summary(condition, rows))
        for dim_label, metric in sorted(summary["by_dimension"].items()):
            by_variant.append({"condition": condition, "evaluation_form": dim_label, **metric})

    pairs: List[Tuple[str, str, str]] = [
        ("causal_original_only", "prompt_bidir_original_only", "attention_effect"),
        ("causal_original_plus_dim2", "prompt_bidir_original_plus_dim2", "attention_effect"),
        ("causal_original_plus_all", "prompt_bidir_original_plus_all", "attention_effect"),
        ("causal_original_only", "causal_original_plus_dim2", "augmentation_within_causal"),
        ("causal_original_only", "causal_original_plus_all", "augmentation_within_causal"),
        ("causal_original_plus_dim2", "causal_original_plus_all", "augmentation_within_causal"),
        ("prompt_bidir_original_only", "prompt_bidir_original_plus_dim2", "augmentation_within_prompt_bidir"),
        ("prompt_bidir_original_only", "prompt_bidir_original_plus_all", "augmentation_within_prompt_bidir"),
        ("prompt_bidir_original_plus_dim2", "prompt_bidir_original_plus_all", "augmentation_within_prompt_bidir"),
    ]
    for left, right, family in pairs:
        if left in predictions and right in predictions:
            comparisons.extend(comparison_rows(left, right, predictions[left], predictions[right], family))

    summary_fields = ["condition", "attention_type", "reused_budget_condition", "source_dataset_size", "training_unique_base_ids", "training_original_examples", "training_dim1_examples", "training_dim2_examples", "training_dim4_examples", "training_dim6_examples", "optimizer_steps", "max_steps", "estimated_example_presentations", "examples_presented", "example_presentation_count_source", "effective_epochs", "measured_dataset_nonpadding_tokens", "mean_nonpadding_tokens_per_example", "observed_nonpadding_tokens_processed", "estimated_nonpadding_tokens_processed", "trainable_parameter_count", "final_training_loss", "overall_accuracy", "overall_correct", "overall_total", "original_accuracy", "dim1_accuracy", "dim2_accuracy", "dim4_accuracy", "dim6_accuracy", "mean_transformed_accuracy", "worst_dimension_accuracy", "robustness_gap", "matched_consistency", "complete_case_base_ids", "average_generated_tokens", "average_latency_seconds", "average_tokens_per_second", "cap_hits", "cap_hit_rate", "missing_final_answer_markers", "missing_final_answer_marker_rate"]
    comparison_fields = ["comparison_family", "comparison", "left_condition", "right_condition", "evaluation_form", "label", "shared_records", "both_correct", "left_only_correct", "right_only_correct", "both_wrong", "left_accuracy", "right_accuracy", "accuracy_difference_right_minus_left", "percentage_point_difference_right_minus_left", "exact_mcnemar_p_value", "holm_adjusted_p_value"]
    write_csv(args.output_dir / "results_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "results_by_variant.csv", by_variant, ["condition", "evaluation_form", "correct", "total", "accuracy"])
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons, comparison_fields)
    write_csv(args.output_dir / "interaction_checks.csv", interaction_rows(summary_rows), ["metric", "causal_dim2_gain", "prompt_bidir_dim2_gain", "interaction_difference_prompt_bidir_minus_causal", "note"])
    write_json(args.output_dir / "analysis_note.json", {"test_set_used": False, "max_new_tokens": MAX_NEW_TOKENS, "conditions_analyzed": sorted(predictions)})
    print(f"Analyzed {len(summary_rows)} attention-control conditions into {args.output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    analyze(parse_args(argv))


if __name__ == "__main__":
    main()
