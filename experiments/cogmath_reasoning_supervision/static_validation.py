#!/usr/bin/env python3
"""Static validation for the reasoning-supervision diagnostic."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning_utils import (
    AUGMENTATION_DIR,
    DATASET_REPORT,
    DIRECT_ORIGINAL_ONLY_ADAPTER_DIR,
    DIRECT_ORIGINAL_ONLY_VALIDATION_DIR,
    EXPERIMENT_DIR,
    LORA_TARGET_MODULES,
    OUTPUT_ROOT,
    REASONING_ADAPTER_DIR,
    REASONING_TRAIN_FILE,
    REASONING_TRAINING_CONDITION,
    SOURCE_TRAIN_FILE,
    TEST_EVAL_FILE,
    TRAIN_MAX_LENGTH,
    UNIQUE_FULL_OUTPUT_ROOT,
    VALIDATION_EVAL_FILE,
)

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import answers_match, extract_answer, read_jsonl, write_json  # noqa: E402
from prepare_reasoning_data import build_rows  # noqa: E402


EXPECTED_ROWS = 923
EXPECTED_MAX_LENGTH = TRAIN_MAX_LENGTH


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the reasoning-supervision diagnostic setup.")
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--strict-adapter-check", action="store_true")
    parser.add_argument("--strict-direct-reference-check", action="store_true")
    parser.add_argument("--write-report", type=Path, default=EXPERIMENT_DIR / "static_validation_report.json")
    return parser.parse_args(argv)


def check(name: str, passed: bool, details: Any, failures: List[str]) -> Dict[str, Any]:
    if not passed:
        failures.append(name)
    return {"name": name, "passed": passed, "details": details}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def adapter_files_present(path: Path) -> bool:
    if not path.exists():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    return has_config and has_weights


def dataset_checks() -> Dict[str, Any]:
    rows = read_jsonl(REASONING_TRAIN_FILE)
    source_rows = read_jsonl(SOURCE_TRAIN_FILE)
    rebuilt_rows = build_rows(source_rows)
    source_by_record = {row["record_id"]: row for row in source_rows}
    base_ids = [row["base_id"] for row in rows]
    record_ids = [row["record_id"] for row in rows]

    answer_mismatches = []
    trace_mismatches = []
    malformed_targets = []
    targets_with_marker = []
    for row in rows:
        source = source_by_record.get(row.get("source_record_id"))
        if source is None:
            trace_mismatches.append({"record_id": row["record_id"], "reason": "missing_source_record"})
            continue
        if source.get("answer_raw") != row.get("answer_raw") or source.get("question") != row.get("question"):
            trace_mismatches.append({"record_id": row["record_id"], "reason": "source_content_mismatch"})
        if not answers_match(extract_answer(row.get("answer_raw")), row.get("final_answer")):
            answer_mismatches.append(row["record_id"])
        if "####" in row.get("training_target", ""):
            targets_with_marker.append(row["record_id"])
        if not row.get("training_target", "").endswith(f"Final answer: {row.get('final_answer')}"):
            malformed_targets.append(row["record_id"])

    validation_ids = {row["base_id"] for row in read_jsonl(VALIDATION_EVAL_FILE)}
    test_ids = {row["base_id"] for row in read_jsonl(TEST_EVAL_FILE)}
    train_ids = set(base_ids)
    return {
        "row_count": len(rows),
        "unique_base_ids": len(set(base_ids)),
        "duplicate_record_ids": len(record_ids) - len(set(record_ids)),
        "condition_values": sorted({row.get("condition") for row in rows}),
        "dimension_values": sorted({int(row.get("dimension")) for row in rows}),
        "answer_mismatches": answer_mismatches[:20],
        "trace_mismatches": trace_mismatches[:20],
        "targets_with_gsm8k_marker": targets_with_marker[:20],
        "malformed_targets": malformed_targets[:20],
        "train_validation_overlap": sorted(train_ids & validation_ids),
        "train_test_overlap": sorted(train_ids & test_ids),
        "validation_test_overlap": sorted(validation_ids & test_ids),
        "deterministic_rebuild_matches": rows == rebuilt_rows,
    }


def syntax_checks() -> Dict[str, Any]:
    py_status = {}
    for path in sorted(EXPERIMENT_DIR.glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            py_status[str(path)] = "ok"
        except SyntaxError as exc:
            py_status[str(path)] = str(exc)

    json_status = {}
    for path in sorted(EXPERIMENT_DIR.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            json_status[str(path)] = "ok"
        except json.JSONDecodeError as exc:
            json_status[str(path)] = str(exc)

    bash_status = {}
    for path in sorted((EXPERIMENT_DIR / "slurm").glob("*.sh")):
        completed = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        bash_status[str(path)] = {"returncode": completed.returncode, "stderr": completed.stderr}

    return {"python": py_status, "json": json_status, "bash": bash_status}


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []

    dataset = dataset_checks()
    checks.append(check("exactly 923 reasoning-training records", dataset["row_count"] == EXPECTED_ROWS, dataset, failures))
    checks.append(check("unique training base IDs", dataset["unique_base_ids"] == EXPECTED_ROWS, dataset["unique_base_ids"], failures))
    checks.append(check("no duplicate reasoning record IDs", dataset["duplicate_record_ids"] == 0, dataset["duplicate_record_ids"], failures))
    checks.append(check("only original dimension used", dataset["dimension_values"] == [0], dataset["dimension_values"], failures))
    checks.append(check("reasoning condition exact", dataset["condition_values"] == [REASONING_TRAINING_CONDITION], dataset["condition_values"], failures))
    checks.append(check("reasoning trace corresponds to source record", not dataset["trace_mismatches"], dataset["trace_mismatches"], failures))
    checks.append(check("answer_raw final answer agrees with final_answer", not dataset["answer_mismatches"], dataset["answer_mismatches"], failures))
    checks.append(check("no #### marker remains in training_target", not dataset["targets_with_gsm8k_marker"], dataset["targets_with_gsm8k_marker"], failures))
    checks.append(check("all targets end with Final answer: <gold>", not dataset["malformed_targets"], dataset["malformed_targets"], failures))
    checks.append(check("deterministic dataset generation", dataset["deterministic_rebuild_matches"], dataset["deterministic_rebuild_matches"], failures))
    checks.append(
        check(
            "no train/validation/test leakage",
            not dataset["train_validation_overlap"] and not dataset["train_test_overlap"] and not dataset["validation_test_overlap"],
            {
                "train_validation_overlap": dataset["train_validation_overlap"],
                "train_test_overlap": dataset["train_test_overlap"],
                "validation_test_overlap": dataset["validation_test_overlap"],
            },
            failures,
        )
    )

    dataset_report = read_json(DATASET_REPORT)
    token_stats = dataset_report.get("token_stats", {})
    tokenizer_available = token_stats.get("tokenizer_status") == "available"
    examples_over_configured = token_stats.get("examples_over_candidate_lengths", {}).get(str(EXPECTED_MAX_LENGTH))
    exact_no_configured_truncation = tokenizer_available and examples_over_configured == 0
    checks.append(
        check(
            "tokenizer length check available when required",
            tokenizer_available or not args.require_tokenizer,
            token_stats,
            failures,
        )
    )
    checks.append(
        check(
            "no silent sequence truncation at configured max_length",
            exact_no_configured_truncation or not args.require_tokenizer,
            {"configured_max_length": EXPECTED_MAX_LENGTH, "token_stats": token_stats},
            failures,
        )
    )

    checks.append(
        check(
            "output root is separate from existing unique-full experiment",
            OUTPUT_ROOT != UNIQUE_FULL_OUTPUT_ROOT and "cogmath_reasoning_supervision_qwen3_4b" in OUTPUT_ROOT,
            {"output_root": OUTPUT_ROOT, "unique_full_output_root": UNIQUE_FULL_OUTPUT_ROOT},
            failures,
        )
    )
    checks.append(
        check(
            "LoRA modules match direct-answer all-linear setup",
            LORA_TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            LORA_TARGET_MODULES,
            failures,
        )
    )

    syntax = syntax_checks()
    checks.append(check("python syntax parses", all(value == "ok" for value in syntax["python"].values()), syntax["python"], failures))
    checks.append(check("json files parse", all(value == "ok" for value in syntax["json"].values()), syntax["json"], failures))
    checks.append(check("slurm bash syntax", all(value["returncode"] == 0 for value in syntax["bash"].values()), syntax["bash"], failures))

    adapter_presence = {
        "direct_original_only_adapter": adapter_files_present(DIRECT_ORIGINAL_ONLY_ADAPTER_DIR),
        "reasoning_adapter": adapter_files_present(REASONING_ADAPTER_DIR),
    }
    checks.append(
        check(
            "adapter files present when strict adapter check requested",
            all(adapter_presence.values()) if args.strict_adapter_check else True,
            {"strict_adapter_check": args.strict_adapter_check, "adapter_presence": adapter_presence},
            failures,
        )
    )
    direct_reference = {
        "dir": str(DIRECT_ORIGINAL_ONLY_VALIDATION_DIR),
        "predictions_jsonl": (DIRECT_ORIGINAL_ONLY_VALIDATION_DIR / "predictions.jsonl").exists(),
        "summary_json": (DIRECT_ORIGINAL_ONLY_VALIDATION_DIR / "summary.json").exists(),
    }
    checks.append(
        check(
            "existing direct-answer validation reference present when required",
            (direct_reference["predictions_jsonl"] and direct_reference["summary_json"]) if args.strict_direct_reference_check else True,
            {"strict_direct_reference_check": args.strict_direct_reference_check, **direct_reference},
            failures,
        )
    )

    report = {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "notes": [
            "Normal local validation permits missing transformers/tokenizer and missing cluster adapters.",
            "Use --require-tokenizer on the cluster before training to confirm exact Qwen token lengths.",
            "No test evaluation script is created for this diagnostic.",
        ],
    }
    write_json(args.write_report, report)
    return report


def main(argv: Optional[List[str]] = None) -> None:
    report = validate(parse_args(argv))
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
