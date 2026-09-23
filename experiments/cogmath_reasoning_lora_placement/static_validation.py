#!/usr/bin/env python3
"""Static validation for the LoRA-placement experiment."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from placement_utils import (
    ALL_ANALYSIS_CONDITIONS,
    BUDGET_REFERENCE_CONDITION,
    BUDGET_OUTPUT_ROOT,
    EXPECTED_DIMENSION_COUNTS,
    EXPERIMENT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    MODEL_NAME,
    NEW_TRAINING_CONDITIONS,
    OUTPUT_ROOT,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PLACEMENTS,
    SEED,
    SOURCE_DATASET_BY_VARIANT,
    TRAIN_MAX_LENGTH,
    VALIDATION_EVAL_FILE,
    WARMUP_STEPS,
    adapter_dir_for,
    condition_placement,
    condition_variant,
    dataset_report,
    parameter_estimates,
    read_jsonl,
    source_training_path,
    validation_dir_for,
    write_json,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LoRA-placement experiment setup.")
    parser.add_argument("--require-existing-references", action="store_true")
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


def source_details() -> Dict[str, Any]:
    out = {}
    for variant, path in SOURCE_DATASET_BY_VARIANT.items():
        rows = read_jsonl(path)
        counts = Counter(row["record_id"] for row in rows)
        by_dim = Counter(int(row["dimension"]) for row in rows)
        out[variant] = {
            "rows": len(rows),
            "unique_base_ids": len({row["base_id"] for row in rows}),
            "counts_by_dimension": {dim: by_dim.get(dim, 0) for dim in [0, 1, 2, 4, 6] if by_dim.get(dim, 0)},
            "duplicate_record_ids": sorted(record_id for record_id, count in counts.items() if count > 1),
            "base_ids": {row["base_id"] for row in rows},
        }
    return out


def public_source_details(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: {field: value for field, value in detail.items() if field != "base_ids"}
        for key, detail in details.items()
    }


def script_text(name: str) -> str:
    path = EXPERIMENT_DIR / "slurm" / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def compatible_budget_reference(condition: str) -> Dict[str, Any]:
    ref_condition = BUDGET_REFERENCE_CONDITION[condition]
    adapter_dir = adapter_dir_for(condition)
    validation_dir = validation_dir_for(condition)
    resolved = read_json(adapter_dir / "resolved_config.json") if (adapter_dir / "resolved_config.json").exists() else {}
    summary = read_json(validation_dir / "summary.json") if (validation_dir / "summary.json").exists() else {}
    return {
        "condition": condition,
        "reference_condition": ref_condition,
        "adapter_dir_exists": adapter_dir.exists(),
        "predictions_exist": (validation_dir / "predictions.jsonl").exists(),
        "model_name": resolved.get("model_name"),
        "max_steps": resolved.get("max_steps"),
        "max_length": resolved.get("max_length"),
        "seed": resolved.get("seed"),
        "learning_rate": resolved.get("learning_rate"),
        "lora": resolved.get("lora"),
        "eval_max_new_tokens": summary.get("max_new_tokens"),
        "compatible": (
            resolved.get("model_name") == MODEL_NAME
            and resolved.get("max_steps") == MAX_STEPS
            and resolved.get("max_length") == TRAIN_MAX_LENGTH
            and resolved.get("seed") == SEED
            and resolved.get("learning_rate") == LEARNING_RATE
            and resolved.get("lora", {}).get("rank") == LORA_RANK
            and resolved.get("lora", {}).get("alpha") == LORA_ALPHA
            and resolved.get("lora", {}).get("dropout") == LORA_DROPOUT
            and resolved.get("lora", {}).get("target_modules") == PLACEMENTS["all_linear"]
            and summary.get("max_new_tokens") == MAX_NEW_TOKENS
            and (validation_dir / "predictions.jsonl").exists()
        ),
    }


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []
    report = dataset_report()
    resolved_config = read_json(EXPERIMENT_DIR / "resolved_configs" / "lora_placement.json") if (EXPERIMENT_DIR / "resolved_configs" / "lora_placement.json").exists() else {}
    details = source_details()
    train_script = script_text("train_array.sh")
    smoke_script = script_text("smoke_placement.sh")
    eval_script = script_text("evaluate_validation_array.sh")
    analysis_script = script_text("analyze_validation.sh")

    checks.append(check("placement target modules exact", resolved_config.get("placements") == PLACEMENTS, resolved_config.get("placements"), failures))
    checks.append(check("new training array mapping exact", shell_array_items(train_script) == NEW_TRAINING_CONDITIONS, shell_array_items(train_script), failures))
    checks.append(check("validation array mapping exact", shell_array_items(eval_script) == NEW_TRAINING_CONDITIONS, shell_array_items(eval_script), failures))
    checks.append(check("only eight new adapters are trained", len(NEW_TRAINING_CONDITIONS) == 8 and not any(c.startswith("all_linear") for c in NEW_TRAINING_CONDITIONS), NEW_TRAINING_CONDITIONS, failures))
    checks.append(check("expected source dimension counts", {k: v["counts_by_dimension"] for k, v in details.items()} == EXPECTED_DIMENSION_COUNTS, public_source_details(details), failures))
    checks.append(check("no duplicate source IDs", all(not v["duplicate_record_ids"] for v in details.values()), {k: v["duplicate_record_ids"] for k, v in details.items()}, failures))
    checks.append(check("all source datasets have 923 base IDs", all(v["unique_base_ids"] == 923 for v in details.values()), {k: v["unique_base_ids"] for k, v in details.items()}, failures))
    checks.append(check("condition-to-dataset mapping preserves matched pairs", all(source_training_path(c) == SOURCE_DATASET_BY_VARIANT[condition_variant(c)] for c in NEW_TRAINING_CONDITIONS), {c: str(source_training_path(c)) for c in NEW_TRAINING_CONDITIONS}, failures))

    checks.append(check("training scripts use max_steps 232", f"--max_steps {MAX_STEPS}" in train_script and "--max_steps 2" in smoke_script, {"train": f"--max_steps {MAX_STEPS}" in train_script, "smoke": "--max_steps 2" in smoke_script}, failures))
    checks.append(check("training scripts use max_length 1536", f"--max_length {TRAIN_MAX_LENGTH}" in train_script and f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script, {"train": f"--max_length {TRAIN_MAX_LENGTH}" in train_script, "smoke": f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script}, failures))
    checks.append(check("fixed hyperparameters", resolved_config.get("model_name") == MODEL_NAME and resolved_config.get("seed") == SEED and resolved_config.get("learning_rate") == LEARNING_RATE and resolved_config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION_STEPS and resolved_config.get("per_device_train_batch_size") == PER_DEVICE_TRAIN_BATCH_SIZE and resolved_config.get("warmup_steps") == WARMUP_STEPS, resolved_config, failures))
    checks.append(check("validation max_new_tokens 512", resolved_config.get("validation_max_new_tokens") == MAX_NEW_TOKENS and f"--max_new_tokens {MAX_NEW_TOKENS}" in eval_script, {"config": resolved_config.get("validation_max_new_tokens"), "script": f"--max_new_tokens {MAX_NEW_TOKENS}" in eval_script}, failures))
    checks.append(check("output roots are separate", "cogmath_reasoning_lora_placement_qwen3_4b" in OUTPUT_ROOT and OUTPUT_ROOT != BUDGET_OUTPUT_ROOT, {"output_root": OUTPUT_ROOT, "budget_root": BUDGET_OUTPUT_ROOT}, failures))
    checks.append(check("validation record count is 990", sum(1 for _ in VALIDATION_EVAL_FILE.open()) == 990, str(VALIDATION_EVAL_FILE), failures))

    params = parameter_estimates()
    checks.append(check("analytic trainable parameter counts available", all(params[p]["estimated_trainable_parameters"] > 0 for p in PLACEMENTS), params, failures))

    references = {condition: compatible_budget_reference(condition) for condition in BUDGET_REFERENCE_CONDITION}
    checks.append(check("all-linear reference compatibility when required", all(v["compatible"] for v in references.values()) or not args.require_existing_references, references, failures))

    all_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in EXPERIMENT_DIR.rglob("*")
        if path.is_file() and path.name != "static_validation_report.json" and path.suffix in {".py", ".sh", ".md", ".json"}
    )
    forbidden = "test" + "_eval"
    checks.append(check("no held-out evaluation path referenced", forbidden not in all_text, "checked without opening held-out evaluation data", failures))
    checks.append(check("analysis job has no evaluation/training call", "train_lora_placement.py" not in analysis_script and "evaluate_lora_placement.py" not in analysis_script, analysis_script, failures))

    syn = syntax()
    checks.append(check("python syntax parses", all(v == "ok" for v in syn["python"].values()), syn["python"], failures))
    checks.append(check("json files parse", all(v == "ok" for v in syn["json"].values()), syn["json"], failures))
    checks.append(check("slurm bash syntax", all(v["returncode"] == 0 for v in syn["bash"].values()), syn["bash"], failures))

    out = {"passed": not failures, "failures": failures, "checks": checks, "dataset_report": report, "all_linear_reference_checks": references}
    write_json(args.write_report, out)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    report = validate(parse_args(argv))
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
