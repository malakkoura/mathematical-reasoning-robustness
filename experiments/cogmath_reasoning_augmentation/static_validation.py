#!/usr/bin/env python3
"""Static validation for the reasoning-augmentation experiment."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning_aug_utils import (
    DATA_DIR,
    EVALUATION_CONDITIONS,
    EXPLICIT_DIM6_EXCLUSIONS,
    EXPERIMENT_DIR,
    OUTPUT_ROOT,
    REASONING_SUPERVISION_OUTPUT_ROOT,
    TRAINING_CONDITIONS,
    TRAIN_MAX_LENGTH,
    VALIDATION_EVAL_FILE,
    reasoning_training_path,
    source_training_path,
)


EXPECTED_COUNTS = {
    "reasoning_original_only": {0: 923},
    "reasoning_original_plus_dim1": {0: 923, 1: 923},
    "reasoning_original_plus_dim2": {0: 923, 2: 923},
    "reasoning_original_plus_dim4": {0: 923, 4: 923},
    "reasoning_original_plus_dim6": {0: 923, 6: 920},
    "reasoning_original_plus_dim2_dim4": {0: 923, 2: 923, 4: 923},
    "reasoning_original_plus_all": {0: 923, 1: 923, 2: 923, 4: 923, 6: 920},
}

EXPECTED_VALID_COUNTS = {
    "reasoning_original_only": {0: 923},
    "reasoning_original_plus_dim1": {0: 923, 1: 923},
    "reasoning_original_plus_dim2": {0: 923, 2: 923},
    "reasoning_original_plus_dim4": {0: 923, 4: 923},
    "reasoning_original_plus_dim6": {0: 923, 6: 917},
    "reasoning_original_plus_dim2_dim4": {0: 923, 2: 923, 4: 923},
    "reasoning_original_plus_all": {0: 923, 1: 923, 2: 923, 4: 923, 6: 917},
}

EXPECTED_ROWS = {
    "reasoning_original_only": 923,
    "reasoning_original_plus_dim1": 1846,
    "reasoning_original_plus_dim2": 1846,
    "reasoning_original_plus_dim4": 1846,
    "reasoning_original_plus_dim6": 1840,
    "reasoning_original_plus_dim2_dim4": 2769,
    "reasoning_original_plus_all": 4609,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reasoning-augmentation setup.")
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--write-report", type=Path, default=EXPERIMENT_DIR / "static_validation_report.json")
    return parser.parse_args(argv)


def check(name: str, passed: bool, details: Any, failures: List[str]) -> Dict[str, Any]:
    if not passed:
        failures.append(name)
    return {"name": name, "passed": passed, "details": details}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["record_id"])
    return ids


def read_jsonl_source_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ids.append(row.get("source_record_id", row["record_id"]))
    return ids


def shell_array_items(script: str, name: str) -> List[str]:
    marker = f"{name}=("
    if marker not in script:
        return []
    body = script.split(marker, 1)[1].split(")", 1)[0]
    return [line.strip().strip('"').strip("'") for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]


def bash_n(path: Path) -> Dict[str, Any]:
    completed = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
    return {"returncode": completed.returncode, "stderr": completed.stderr}


def syntax() -> Dict[str, Any]:
    py = {}
    for path in sorted(EXPERIMENT_DIR.glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            py[str(path)] = "ok"
        except SyntaxError as exc:
            py[str(path)] = str(exc)
    js = {}
    for path in sorted(EXPERIMENT_DIR.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            js[str(path)] = "ok"
        except json.JSONDecodeError as exc:
            js[str(path)] = str(exc)
    sh = {str(path): bash_n(path) for path in sorted((EXPERIMENT_DIR / "slurm").glob("*.sh"))}
    return {"python": py, "json": js, "bash": sh}


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []
    report = read_json(DATA_DIR / "dataset_report.json")
    blocked = report.get("blocked_conditions", {})

    counts = {
        condition: {int(k): v for k, v in details["source_counts_by_dimension"].items()}
        for condition, details in report["conditions"].items()
    }
    valid_counts = {
        condition: {int(k): v for k, v in details["valid_counts_by_dimension"].items()}
        for condition, details in report["conditions"].items()
    }
    valid_rows = {condition: details["valid_reasoning_rows"] for condition, details in report["conditions"].items()}
    explicit = report.get("explicit_data_quality_exclusions", [])
    explicit_by_condition = {
        condition: {
            entry["record_id"]: entry["reason"]
            for entry in report["conditions"][condition].get("explicit_data_quality_exclusions", [])
        }
        for condition in report["conditions"]
    }
    checks.append(check("source row/dimension counts match unique-full inputs", counts == EXPECTED_COUNTS, counts, failures))
    checks.append(check("post-exclusion row counts exact", valid_rows == EXPECTED_ROWS, valid_rows, failures))
    checks.append(check("post-exclusion dimension counts exact", valid_counts == EXPECTED_VALID_COUNTS, valid_counts, failures))
    checks.append(check("no validation/test leakage in materialized safe rows", not report.get("leakage", {}).get("train_validation_overlap") and not report.get("leakage", {}).get("train_test_overlap"), report.get("leakage"), failures))
    checks.append(check("Dim1/Dim2/Dim4 reasoning conditions unblocked", not any(blocked.get(c) for c in ["reasoning_original_plus_dim1", "reasoning_original_plus_dim2", "reasoning_original_plus_dim4", "reasoning_original_plus_dim2_dim4"]), blocked, failures))
    checks.append(check("no unexpected invalid reasoning traces remain", not blocked, blocked, failures))
    checks.append(
        check(
            "explicit Dim6 exclusions documented exactly",
            explicit_by_condition.get("reasoning_original_plus_dim6") == EXPLICIT_DIM6_EXCLUSIONS
            and explicit_by_condition.get("reasoning_original_plus_all") == EXPLICIT_DIM6_EXCLUSIONS
            and len(explicit) == 6,
            explicit_by_condition,
            failures,
        )
    )

    omissions = {}
    for condition in EVALUATION_CONDITIONS:
        source_ids = set(read_jsonl_ids(source_training_path(condition)))
        materialized_ids = set(read_jsonl_source_ids(reasoning_training_path(condition))) if reasoning_training_path(condition).exists() else set()
        omissions[condition] = sorted(source_ids - materialized_ids)
    expected_omissions = {
        condition: sorted(EXPLICIT_DIM6_EXCLUSIONS) if condition in {"reasoning_original_plus_dim6", "reasoning_original_plus_all"} else []
        for condition in EVALUATION_CONDITIONS
    }
    checks.append(check("no unintended source records excluded", omissions == expected_omissions, omissions, failures))

    checks.append(check("six-condition training is ready when required", report.get("ready_for_six_condition_training") if args.require_ready else True, report.get("ready_for_six_condition_training"), failures))
    resolved_config = read_json(EXPERIMENT_DIR / "resolved_configs" / "reasoning_augmentation.json")
    train_array_script = (EXPERIMENT_DIR / "slurm" / "train_array.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "train_array.sh").exists() else ""
    smoke_script = (EXPERIMENT_DIR / "slurm" / "train_smoke_worst_case.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "train_smoke_worst_case.sh").exists() else ""
    eval_script = (EXPERIMENT_DIR / "slurm" / "evaluate_validation_array.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "evaluate_validation_array.sh").exists() else ""
    checks.append(
        check(
            "training max_length configured consistently",
            report.get("training_max_length") == TRAIN_MAX_LENGTH
            and resolved_config.get("training_max_length") == TRAIN_MAX_LENGTH
            and f"--max_length {TRAIN_MAX_LENGTH}" in train_array_script
            and f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script,
            {
                "constant": TRAIN_MAX_LENGTH,
                "dataset_report": report.get("training_max_length"),
                "resolved_config": resolved_config.get("training_max_length"),
                "train_array_contains": f"--max_length {TRAIN_MAX_LENGTH}" in train_array_script,
                "smoke_contains": f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script,
            },
            failures,
        )
    )
    checks.append(
        check(
            "validation generation remains 512 tokens",
            resolved_config.get("max_new_tokens_validation") == 512
            and "--max_new_tokens 512" in eval_script,
            {"resolved_config": resolved_config.get("max_new_tokens_validation"), "eval_script_contains": "--max_new_tokens 512" in eval_script},
            failures,
        )
    )

    token = report.get("token_stats", {})
    tokenizer_ok = token.get("tokenizer_status") == "available"
    no_trunc = tokenizer_ok and token.get("examples_over_candidate_lengths", {}).get(str(TRAIN_MAX_LENGTH)) == 0
    checks.append(check("tokenizer check available when required", tokenizer_ok or not args.require_tokenizer, token, failures))
    checks.append(check("no sequence truncation at chosen max_length when tokenizer required", no_trunc or not args.require_tokenizer, {"max_length": TRAIN_MAX_LENGTH, "token_stats": token}, failures))

    array_script = train_array_script
    checks.append(check("training array mappings exact", shell_array_items(array_script, "CONDITIONS") == TRAINING_CONDITIONS, shell_array_items(array_script, "CONDITIONS"), failures))
    checks.append(check("validation evaluation maps all seven conditions", shell_array_items(eval_script, "CONDITIONS") == EVALUATION_CONDITIONS, shell_array_items(eval_script, "CONDITIONS"), failures))
    checks.append(check("no test evaluation access", "test_eval" not in eval_script and "test_eval" not in array_script, {"eval_script": "evaluate_validation_array.sh"}, failures))
    checks.append(check("output root isolated", "cogmath_reasoning_augmentation_qwen3_4b" in OUTPUT_ROOT and OUTPUT_ROOT != REASONING_SUPERVISION_OUTPUT_ROOT, {"output_root": OUTPUT_ROOT}, failures))
    checks.append(check("validation record count is 990", sum(1 for _ in VALIDATION_EVAL_FILE.open()) == 990, str(VALIDATION_EVAL_FILE), failures))

    syn = syntax()
    checks.append(check("python syntax parses", all(v == "ok" for v in syn["python"].values()), syn["python"], failures))
    checks.append(check("json files parse", all(v == "ok" for v in syn["json"].values()), syn["json"], failures))
    checks.append(check("slurm bash syntax", all(v["returncode"] == 0 for v in syn["bash"].values()), syn["bash"], failures))

    materialized = {condition: reasoning_training_path(condition).exists() for condition in EVALUATION_CONDITIONS}
    checks.append(check("all required training datasets are materialized", all(materialized.values()), materialized, failures))

    out = {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "ready_for_six_condition_training": report.get("ready_for_six_condition_training"),
        "blocked_conditions": blocked,
        "note": "Use --require-ready before training; it passes only when explicit exclusions and all materialized counts match expectations.",
    }
    with args.write_report.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    report = validate(parse_args(argv))
    print(json.dumps({"passed": report["passed"], "failures": report["failures"], "ready_for_six_condition_training": report["ready_for_six_condition_training"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
