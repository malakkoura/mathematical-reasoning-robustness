#!/usr/bin/env python3
"""Analyze first-pass, causal-revision, and bidirectional-trace revision outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from deliberation_utils import mcnemar_exact, read_jsonl, summarize_revision_predictions, write_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deliberation validation predictions.")
    parser.add_argument("--first_pass_file", type=Path, required=True)
    parser.add_argument("--causal_predictions", type=Path, required=True)
    parser.add_argument("--bidir_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_existing_output", action="store_true")
    return parser.parse_args(argv)


def load_by_id(path: Path) -> Dict[str, dict]:
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        rid = row["record_id"]
        if rid in out:
            raise ValueError(f"Duplicate record_id in {path}: {rid}")
        out[rid] = row
    return out


def first_pass_as_predictions(first_pass_rows: List[dict]) -> List[dict]:
    out = []
    for row in first_pass_rows:
        out.append(
            {
                "condition": "first_pass_no_revision",
                "record_id": row["record_id"],
                "base_id": row["base_id"],
                "split": row.get("split"),
                "dimension": row.get("dimension"),
                "variant_type": row.get("variant_type"),
                "question": row.get("question"),
                "gold_final_answer": row.get("gold_final_answer"),
                "first_pass_trace": row.get("first_pass_trace"),
                "first_pass_extracted_answer": row.get("first_pass_extracted_answer"),
                "first_pass_correct": row.get("first_pass_correct"),
                "revised_output": "",
                "extracted_revised_answer": row.get("first_pass_extracted_answer"),
                "correct": row.get("first_pass_correct"),
                "generated_token_count": 0,
                "hit_max_revision_tokens": False,
            }
        )
    return out


def write_summary_csv(path: Path, summaries: List[dict]) -> None:
    fieldnames = ["condition", "rows", "correct", "accuracy", "corrections", "regressions", "stable_correct", "stable_incorrect", "mean_generated_tokens", "max_generated_tokens", "cap_hits"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "condition": summary["condition"],
                    "rows": summary["rows"],
                    "correct": summary["correct"],
                    "accuracy": summary["accuracy"],
                    "corrections": summary["corrections"],
                    "regressions": summary["regressions"],
                    "stable_correct": summary["stable_correct"],
                    "stable_incorrect": summary["stable_incorrect"],
                    "mean_generated_tokens": summary["generated_tokens"]["mean"],
                    "max_generated_tokens": summary["generated_tokens"]["max"],
                    "cap_hits": summary["generated_tokens"]["cap_hits"],
                }
            )


def write_dimension_csv(path: Path, summaries: List[dict]) -> None:
    dims = sorted({dim for summary in summaries for dim in summary["by_dimension"]})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "dimension", "rows", "correct", "accuracy"])
        writer.writeheader()
        for summary in summaries:
            for dim in dims:
                stats = summary["by_dimension"].get(dim, {"rows": 0, "correct": 0, "accuracy": None})
                writer.writerow({"condition": summary["condition"], "dimension": dim, **stats})


def paired_rows(ids: List[str], a: Dict[str, dict], b: Dict[str, dict]) -> tuple[List[bool], List[bool]]:
    return [bool(a[rid]["correct"]) for rid in ids], [bool(b[rid]["correct"]) for rid in ids]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists() and not args.allow_existing_output:
        raise FileExistsError(f"Refusing to overwrite existing analysis in {args.output_dir}.")

    first_rows = first_pass_as_predictions(read_jsonl(args.first_pass_file))
    causal_rows = read_jsonl(args.causal_predictions)
    bidir_rows = read_jsonl(args.bidir_predictions)
    first = {row["record_id"]: row for row in first_rows}
    causal = load_by_id(args.causal_predictions)
    bidir = load_by_id(args.bidir_predictions)
    ids = sorted(set(first) & set(causal) & set(bidir))
    if len(ids) != len(first) or len(ids) != len(causal) or len(ids) != len(bidir):
        raise ValueError(
            "Prediction sets are not matched: "
            f"first={len(first)} causal={len(causal)} bidir={len(bidir)} intersection={len(ids)}"
        )

    summaries = [
        summarize_revision_predictions(first_rows, "first_pass_no_revision"),
        summarize_revision_predictions(causal_rows, "causal_revision"),
        summarize_revision_predictions(bidir_rows, "bidir_trace_revision"),
    ]
    comparisons = {}
    for name, a, b in [
        ("first_pass_vs_causal_revision", first, causal),
        ("first_pass_vs_bidir_trace_revision", first, bidir),
        ("causal_revision_vs_bidir_trace_revision", causal, bidir),
    ]:
        a_vec, b_vec = paired_rows(ids, a, b)
        comparisons[name] = mcnemar_exact(a_vec, b_vec)

    summary = {"rows": len(ids), "summaries": summaries, "paired_mcnemar": comparisons, "test_set_used": False}
    write_json(args.output_dir / "summary.json", summary)
    write_summary_csv(args.output_dir / "condition_summary.csv", summaries)
    write_dimension_csv(args.output_dir / "by_dimension.csv", summaries)
    with (args.output_dir / "paired_mcnemar.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "a_correct_b_wrong", "a_wrong_b_correct", "discordant", "exact_two_sided_p"])
        writer.writeheader()
        for comparison, stats in comparisons.items():
            writer.writerow({"comparison": comparison, **stats})
    print(f"Wrote deliberation analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
