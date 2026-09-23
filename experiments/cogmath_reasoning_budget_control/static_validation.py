#!/usr/bin/env python3
"""Static validation for the reasoning budget-control experiment."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from budget_control_utils import (
    BUDGET_CONDITIONS,
    EXPECTED_DIMENSION_COUNTS,
    EXPLICIT_DIM6_EXCLUSIONS,
    EXPERIMENT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    MODEL_NAME,
    OUTPUT_ROOT,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    REASONING_AUG_DIR,
    SEED,
    SOURCE_CONDITION_BY_BUDGET,
    SOURCE_REPORT_FILE,
    SPLITS_FILE,
    TRAIN_MAX_LENGTH,
    VALIDATION_EVAL_FILE,
    WARMUP_STEPS,
    budget_dataset_report,
    read_jsonl,
    sha256_file,
    source_training_path,
    trainer_epoch_group_example_counts,
    write_json,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reasoning budget-control setup.")
    parser.add_argument("--write-report", type=Path, default=EXPERIMENT_DIR / "static_validation_report.json")
    return parser.parse_args(argv)


def check(name: str, passed: bool, details: Any, failures: List[str]) -> Dict[str, Any]:
    if not passed:
        failures.append(name)
    return {"name": name, "passed": passed, "details": details}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def shell_array_items(script: str, name: str = "CONDITIONS") -> List[str]:
    marker = f"{name}=("
    if marker not in script:
        return []
    body = script.split(marker, 1)[1].split(")", 1)[0]
    return [line.strip().strip('"').strip("'") for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]


def source_hashes_from_reasoning_report() -> Dict[str, str]:
    report = read_json(SOURCE_REPORT_FILE)
    return {condition: details["sha256"] for condition, details in report["conditions"].items()}


def source_condition_details() -> Dict[str, Dict[str, Any]]:
    details = {}
    for condition in BUDGET_CONDITIONS:
        rows = read_jsonl(source_training_path(condition))
        record_counts = Counter(row["record_id"] for row in rows)
        details[condition] = {
            "rows": len(rows),
            "unique_base_ids": len({row["base_id"] for row in rows}),
            "counts_by_dimension": {dim: sum(1 for row in rows if int(row["dimension"]) == dim) for dim in [0, 1, 2, 4, 6]},
            "duplicate_record_ids": sorted(record_id for record_id, count in record_counts.items() if count > 1),
            "record_ids": {row["record_id"] for row in rows},
            "base_ids": {row["base_id"] for row in rows},
        }
    return details


def script_text(name: str) -> str:
    path = EXPERIMENT_DIR / "slurm" / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []
    dataset_report = budget_dataset_report()
    resolved_config = read_json(EXPERIMENT_DIR / "resolved_configs" / "budget_control.json") if (EXPERIMENT_DIR / "resolved_configs" / "budget_control.json").exists() else {}
    source_hashes = source_hashes_from_reasoning_report()
    details = source_condition_details()
    splits = read_json(SPLITS_FILE)
    validation_ids = set(splits["validation"])
    split_test_ids = set(splits["test"])

    checks.append(check("condition mapping exact", SOURCE_CONDITION_BY_BUDGET == {
        "budget_original_only": "reasoning_original_only",
        "budget_original_plus_dim2": "reasoning_original_plus_dim2",
        "budget_original_plus_dim4": "reasoning_original_plus_dim4",
        "budget_original_plus_dim2_dim4": "reasoning_original_plus_dim2_dim4",
        "budget_original_plus_all": "reasoning_original_plus_all",
    }, SOURCE_CONDITION_BY_BUDGET, failures))
    checks.append(check("expected dimension membership", {c: {k: v for k, v in d["counts_by_dimension"].items() if v} for c, d in details.items()} == EXPECTED_DIMENSION_COUNTS, {c: d["counts_by_dimension"] for c, d in details.items()}, failures))
    checks.append(check("all source datasets have 923 base IDs", all(d["unique_base_ids"] == 923 for d in details.values()), {c: d["unique_base_ids"] for c, d in details.items()}, failures))
    expected_presentations = {
        condition: trainer_epoch_group_example_counts(details[condition]["rows"])["estimated_example_presentations"]
        for condition in BUDGET_CONDITIONS
    }
    checks.append(check("pre-run example presentation estimates account for incomplete accumulation groups", expected_presentations == {
        "budget_original_only": 3692,
        "budget_original_plus_dim2": 3692,
        "budget_original_plus_dim4": 3692,
        "budget_original_plus_dim2_dim4": 3697,
        "budget_original_plus_all": 3712,
    }, expected_presentations, failures))
    checks.append(check("no duplicate record IDs in source datasets", all(not d["duplicate_record_ids"] for d in details.values()), {c: d["duplicate_record_ids"] for c, d in details.items()}, failures))
    checks.append(check("existing source dataset hashes unchanged", all(sha256_file(source_training_path(c)) == source_hashes[SOURCE_CONDITION_BY_BUDGET[c]] for c in BUDGET_CONDITIONS), {c: sha256_file(source_training_path(c)) for c in BUDGET_CONDITIONS}, failures))
    checks.append(check("no generated dataset jsonl files in new experiment", not list((EXPERIMENT_DIR / "data").glob("*.jsonl")) if (EXPERIMENT_DIR / "data").exists() else True, str(EXPERIMENT_DIR / "data"), failures))

    all_train_base_ids = set().union(*(d["base_ids"] for d in details.values()))
    checks.append(check("no validation leakage", not (all_train_base_ids & validation_ids), sorted(all_train_base_ids & validation_ids), failures))
    checks.append(check("no split-test leakage", not (all_train_base_ids & split_test_ids), sorted(all_train_base_ids & split_test_ids), failures))

    all_records = set().union(*(d["record_ids"] for d in details.values()))
    excluded_present = sorted(set(EXPLICIT_DIM6_EXCLUSIONS) & all_records)
    checks.append(check("known invalid Dim6 records remain excluded", not excluded_present and details["budget_original_plus_all"]["counts_by_dimension"][6] == 917, {"present_invalid_records": excluded_present, "dim6_count_in_all": details["budget_original_plus_all"]["counts_by_dimension"][6]}, failures))

    checks.append(check("resolved config uses max_steps stopping", resolved_config.get("max_steps") == MAX_STEPS and resolved_config.get("stopping_criterion") == "max_steps", resolved_config, failures))
    checks.append(check("training max_length is 1536", resolved_config.get("training_max_length") == TRAIN_MAX_LENGTH == 1536, resolved_config.get("training_max_length"), failures))
    checks.append(check("validation max_new_tokens is 512", resolved_config.get("validation_max_new_tokens") == MAX_NEW_TOKENS == 512, resolved_config.get("validation_max_new_tokens"), failures))
    checks.append(check("same model and seed", resolved_config.get("model_name") == MODEL_NAME and resolved_config.get("seed") == SEED, {"model": resolved_config.get("model_name"), "seed": resolved_config.get("seed")}, failures))
    checks.append(check("LoRA config fixed", resolved_config.get("lora") == {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES}, resolved_config.get("lora"), failures))
    checks.append(check("training hyperparameters fixed", resolved_config.get("learning_rate") == LEARNING_RATE and resolved_config.get("per_device_train_batch_size") == PER_DEVICE_TRAIN_BATCH_SIZE and resolved_config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION_STEPS and resolved_config.get("warmup_steps") == WARMUP_STEPS, resolved_config, failures))

    train_script = script_text("train_array.sh")
    smoke_script = script_text("smoke_max_steps.sh")
    eval_script = script_text("evaluate_validation_array.sh")
    all_script_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in EXPERIMENT_DIR.rglob("*")
        if path.is_file() and path.name != "static_validation_report.json" and path.suffix in {".py", ".sh", ".md", ".json"}
    )
    forbidden_eval_path = "test" + "_eval"
    checks.append(check("training array maps all five conditions exactly", shell_array_items(train_script) == BUDGET_CONDITIONS, shell_array_items(train_script), failures))
    checks.append(check("validation array maps all five conditions exactly", shell_array_items(eval_script) == BUDGET_CONDITIONS, shell_array_items(eval_script), failures))
    checks.append(check("all five array training jobs use max_steps 232", f"--max_steps {MAX_STEPS}" in train_script, {"train_array": f"--max_steps {MAX_STEPS}" in train_script}, failures))
    checks.append(check("smoke uses short max_steps stopping", "--max_steps 2" in smoke_script, {"smoke": "--max_steps 2" in smoke_script}, failures))
    checks.append(check("all training jobs use max_length 1536", f"--max_length {TRAIN_MAX_LENGTH}" in train_script and f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script, {"train": f"--max_length {TRAIN_MAX_LENGTH}" in train_script, "smoke": f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script}, failures))
    checks.append(check("validation script uses max_new_tokens 512", f"--max_new_tokens {MAX_NEW_TOKENS}" in eval_script, eval_script, failures))
    checks.append(check("no test-eval path referenced", forbidden_eval_path not in all_script_text, "checked without opening held-out evaluation data", failures))
    checks.append(check("separate output root", "cogmath_reasoning_budget_control_qwen3_4b" in OUTPUT_ROOT and "cogmath_reasoning_augmentation_qwen3_4b" not in OUTPUT_ROOT, OUTPUT_ROOT, failures))
    checks.append(check("validation record count is 990", sum(1 for _ in VALIDATION_EVAL_FILE.open()) == 990, str(VALIDATION_EVAL_FILE), failures))

    syn = syntax()
    checks.append(check("python syntax parses", all(v == "ok" for v in syn["python"].values()), syn["python"], failures))
    checks.append(check("json files parse", all(v == "ok" for v in syn["json"].values()), syn["json"], failures))
    checks.append(check("slurm bash syntax", all(v["returncode"] == 0 for v in syn["bash"].values()), syn["bash"], failures))

    out = {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "dataset_report": dataset_report,
        "note": "Static validation reads split IDs but does not open or evaluate the held-out evaluation file.",
    }
    write_json(args.write_report, out)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    report = validate(parse_args(argv))
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
