#!/usr/bin/env python3
"""Static validation for the reasoning attention-structure control."""

from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from attention_control_utils import (
    ALL_CONDITIONS,
    ATTENTION_TYPE_BY_CONDITION,
    BUDGET_CONDITION_BY_CAUSAL,
    BUDGET_OUTPUT_ROOT,
    CAUSAL_CONDITIONS,
    EXPECTED_DIMENSION_COUNTS,
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
    PROMPT_BIDIR_TRAINING_CONDITIONS,
    SEED,
    SOURCE_CONDITION_BY_ATTENTION_CONDITION,
    SOURCE_CONDITION_BY_TRAINING_DATA,
    SOURCE_REPORT_FILE,
    SPLITS_FILE,
    TRAINING_DATASETS,
    TRAINING_DATA_BY_CONDITION,
    TRAIN_MAX_LENGTH,
    VALIDATION_EVAL_FILE,
    WARMUP_STEPS,
    dataset_report,
    read_json,
    read_jsonl,
    row_order_sha256,
    sha256_file,
    source_training_path_for_condition,
    source_training_path_for_data,
    trainer_epoch_group_example_counts,
    write_json,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reasoning attention-control setup.")
    parser.add_argument("--write-report", type=Path, default=EXPERIMENT_DIR / "static_validation_report.json")
    parser.add_argument("--require-causal-references", action="store_true")
    parser.add_argument("--require-runtime-diagnostics", action="store_true")
    return parser.parse_args(argv)


def check(name: str, passed: bool, details: Any, failures: List[str]) -> Dict[str, Any]:
    if not passed:
        failures.append(name)
    return {"name": name, "passed": passed, "details": details}


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


def source_report_hashes() -> Dict[str, str]:
    report = read_json(SOURCE_REPORT_FILE)
    return {condition: details["sha256"] for condition, details in report["conditions"].items()}


def source_details() -> Dict[str, Dict[str, Any]]:
    details = {}
    for training_data in TRAINING_DATASETS:
        rows = read_jsonl(source_training_path_for_data(training_data))
        counts = Counter(int(row["dimension"]) for row in rows)
        details[training_data] = {
            "rows": len(rows),
            "unique_base_ids": len({row["base_id"] for row in rows}),
            "counts_by_dimension": {dimension: counts.get(dimension, 0) for dimension in [0, 1, 2, 4, 6]},
            "duplicate_record_ids": sorted(record_id for record_id, count in Counter(row["record_id"] for row in rows).items() if count > 1),
            "base_ids": {row["base_id"] for row in rows},
            "record_ids": [row["record_id"] for row in rows],
        }
    return details


def shuffled_record_ids(path: Path, seed: int) -> List[str]:
    rows = read_jsonl(path)
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return [row["record_id"] for row in rows]


def causal_reference_checks(require: bool) -> Dict[str, Any]:
    details = {}
    failures = []
    for causal_condition, budget_condition in BUDGET_CONDITION_BY_CAUSAL.items():
        adapter_dir = Path(BUDGET_OUTPUT_ROOT) / "adapters" / budget_condition
        eval_dir = Path(BUDGET_OUTPUT_ROOT) / "validation_evaluations" / budget_condition
        config_path = adapter_dir / "resolved_config.json"
        ids_path = adapter_dir / "training_ids.json"
        summary_path = eval_dir / "summary.json"
        predictions_path = eval_dir / "predictions.jsonl"
        condition_details: Dict[str, Any] = {
            "causal_condition": causal_condition,
            "budget_condition": budget_condition,
            "adapter_dir": str(adapter_dir),
            "evaluation_dir": str(eval_dir),
            "status": "not_checked_missing_cluster_outputs",
        }
        if config_path.exists() and ids_path.exists() and summary_path.exists() and predictions_path.exists():
            config = read_json(config_path)
            ids = read_json(ids_path)
            summary = read_json(summary_path)
            expected_ids = shuffled_record_ids(source_training_path_for_condition(causal_condition), SEED)
            predictions = read_jsonl(predictions_path)
            validation_ids = [row["record_id"] for row in read_jsonl(VALIDATION_EVAL_FILE)]
            checks = {
                "model_name": config.get("model_name") == MODEL_NAME,
                "max_steps": config.get("max_steps") == MAX_STEPS,
                "learning_rate": config.get("learning_rate") == LEARNING_RATE,
                "max_length": config.get("max_length") == TRAIN_MAX_LENGTH,
                "seed": config.get("seed") == SEED,
                "batch_and_accumulation": config.get("per_device_train_batch_size") == PER_DEVICE_TRAIN_BATCH_SIZE and config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION_STEPS,
                "warmup": config.get("warmup_steps") == WARMUP_STEPS,
                "lora": config.get("lora") == {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES},
                "source_file": Path(config.get("train_file", "")).name == source_training_path_for_condition(causal_condition).name,
                "training_row_order": ids.get("record_ids") == expected_ids,
                "validation_max_new_tokens": summary.get("max_new_tokens") == MAX_NEW_TOKENS,
                "validation_record_ids": [row["record_id"] for row in predictions] == validation_ids,
            }
            condition_details.update({"status": "checked", "checks": checks, "passed": all(checks.values())})
            if not all(checks.values()):
                failures.append(causal_condition)
        else:
            condition_details["missing_paths"] = [str(path) for path in [config_path, ids_path, summary_path, predictions_path] if not path.exists()]
            if require:
                failures.append(causal_condition)
        details[causal_condition] = condition_details
    return {"passed": not failures, "failures": failures, "details": details}


def runtime_diagnostics_checks(require: bool) -> Dict[str, Any]:
    report_path = EXPERIMENT_DIR / "runtime_diagnostics_report.json"
    if not report_path.exists():
        return {
            "passed": not require,
            "status": "pending_missing_runtime_diagnostics_report",
            "report_path": str(report_path),
            "required": require,
        }
    report = read_json(report_path)
    causal = report.get("causal_equivalence", {})
    leakage = report.get("prompt_bidirectional_model_leakage", {})
    cache_controls = report.get("cache_controls", {})
    causal_forced = cache_controls.get("native_causal_forced_cache_control", {})
    prompt_forced = cache_controls.get("prompt_bidirectional_forced_cache_control", {})
    causal_greedy = cache_controls.get("native_causal_greedy_cache_control", {})
    prompt_greedy = cache_controls.get("prompt_bidirectional_greedy_cache_control", {})
    backend = leakage.get("attention_backend")
    checks = {
        "runtime_report_passed": report.get("passed") is True,
        "causal_equivalence_passed": causal.get("passed") is True,
        "causal_loss_delta_zero_tolerance": causal.get("loss_delta") is not None and causal.get("loss_delta") <= 1e-5,
        "causal_logit_delta_zero_tolerance": causal.get("max_logit_delta") is not None and causal.get("max_logit_delta") <= 1e-5,
        "model_level_future_target_leakage_passed": leakage.get("passed") is True,
        "cache_controls_passed": cache_controls.get("passed") is True,
        "native_causal_forced_cache_control_passed": causal_forced.get("passed") is True,
        "native_causal_greedy_calibration_report_present": causal_greedy.get("diagnostic_role") == "calibration_only_native_causal_independent_greedy",
        "prompt_bidirectional_cache_control_passed": prompt_forced.get("passed") is True and prompt_greedy.get("passed") is True,
        "cached_generation_checked_multiple_steps": prompt_forced.get("steps_checked", 0) >= 5 and causal_forced.get("steps_checked", 0) >= 5,
        "prompt_bidirectional_positions_and_cache_lengths_checked": all(
            row.get("position_ids") == [[row.get("expected_next_token_position")]]
            and row.get("cache_position") == [row.get("expected_next_token_position")]
            and row.get("decode_mask_shape_matches_expected")
            and row.get("cache_length_matches_prefix")
            and row.get("cache_length_after_decode_matches_expected")
            for row in prompt_forced.get("step_reports", [])
        ),
        "prompt_bidirectional_backend_eager": backend == "eager",
        "test_set_not_used": report.get("test_set_used") is False,
    }
    return {
        "passed": all(checks.values()),
        "status": "checked",
        "report_path": str(report_path),
        "checks": checks,
        "causal_summary": {
            "native_loss": causal.get("native_loss"),
            "control_loss": causal.get("control_loss"),
            "loss_delta": causal.get("loss_delta"),
            "max_logit_delta": causal.get("max_logit_delta"),
        },
        "prompt_bidirectional_summary": {
            "original_loss": leakage.get("original_loss"),
            "changed_loss": leakage.get("changed_loss"),
            "attention_backend": backend,
            "visibility_checks": leakage.get("visibility_checks"),
        },
        "cached_generation_summary": {
            "behavioural_acceptance_rule": cache_controls.get("behavioural_acceptance_rule"),
            "native_causal_forced": {
                "steps_checked": causal_forced.get("steps_checked"),
                "max_abs_logit_difference_over_steps": causal_forced.get("max_abs_logit_difference_over_steps"),
                "mean_abs_logit_difference_over_steps": causal_forced.get("mean_abs_logit_difference_over_steps"),
                "all_top1_agree": causal_forced.get("all_top1_agree"),
            },
            "prompt_bidirectional_forced": {
                "steps_checked": prompt_forced.get("steps_checked"),
                "max_abs_logit_difference_over_steps": prompt_forced.get("max_abs_logit_difference_over_steps"),
                "mean_abs_logit_difference_over_steps": prompt_forced.get("mean_abs_logit_difference_over_steps"),
                "all_top1_agree": prompt_forced.get("all_top1_agree"),
                "step_reports": prompt_forced.get("step_reports"),
            },
            "greedy": {
                "native_causal_all_greedy_tokens_agree": causal_greedy.get("all_greedy_tokens_agree"),
                "native_causal_diagnostic_role": causal_greedy.get("diagnostic_role"),
                "native_causal_first_divergence": causal_greedy.get("first_divergence"),
                "prompt_bidirectional_all_greedy_tokens_agree": prompt_greedy.get("all_greedy_tokens_agree"),
                "prompt_bidirectional_diagnostic_role": prompt_greedy.get("diagnostic_role"),
            },
        },
    }


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []
    report = dataset_report()
    config_path = EXPERIMENT_DIR / "resolved_configs" / "attention_control.json"
    config = read_json(config_path) if config_path.exists() else {}
    source_hashes = source_report_hashes()
    details = source_details()
    splits = read_json(SPLITS_FILE)
    validation_ids = set(splits["validation"])
    split_test_ids = set(splits["test"])

    checks.append(check("source mapping exact", SOURCE_CONDITION_BY_TRAINING_DATA == {"original_only": "reasoning_original_only", "original_plus_dim2": "reasoning_original_plus_dim2", "original_plus_all": "reasoning_original_plus_all"}, SOURCE_CONDITION_BY_TRAINING_DATA, failures))
    checks.append(check("condition grid exact", ALL_CONDITIONS == CAUSAL_CONDITIONS + PROMPT_BIDIR_TRAINING_CONDITIONS, ALL_CONDITIONS, failures))
    checks.append(check("dimension counts exact", {name: {k: v for k, v in data["counts_by_dimension"].items() if v} for name, data in details.items()} == EXPECTED_DIMENSION_COUNTS, {name: data["counts_by_dimension"] for name, data in details.items()}, failures))
    checks.append(check("source hashes unchanged", all(sha256_file(source_training_path_for_data(data)) == source_hashes[SOURCE_CONDITION_BY_TRAINING_DATA[data]] for data in TRAINING_DATASETS), {data: sha256_file(source_training_path_for_data(data)) for data in TRAINING_DATASETS}, failures))
    checks.append(check("no duplicate record IDs", all(not data["duplicate_record_ids"] for data in details.values()), {name: data["duplicate_record_ids"] for name, data in details.items()}, failures))
    all_train_base_ids = set().union(*(data["base_ids"] for data in details.values()))
    checks.append(check("no validation leakage", not (all_train_base_ids & validation_ids), sorted(all_train_base_ids & validation_ids), failures))
    checks.append(check("no split-test leakage", not (all_train_base_ids & split_test_ids), sorted(all_train_base_ids & split_test_ids), failures))
    checks.append(check("validation record count is 990", sum(1 for line in VALIDATION_EVAL_FILE.open() if line.strip()) == 990, str(VALIDATION_EVAL_FILE), failures))

    expected_presentations = {name: trainer_epoch_group_example_counts(data["rows"])["estimated_example_presentations"] for name, data in details.items()}
    checks.append(check("example exposure estimates use incomplete accumulation groups", expected_presentations == {"original_only": 3692, "original_plus_dim2": 3692, "original_plus_all": 3712}, expected_presentations, failures))
    checks.append(check("resolved config fixed", config.get("model_name") == MODEL_NAME and config.get("max_steps") == MAX_STEPS and config.get("training_max_length") == TRAIN_MAX_LENGTH and config.get("validation_max_new_tokens") == MAX_NEW_TOKENS and config.get("seed") == SEED, config, failures))
    checks.append(check("LoRA config fixed", config.get("lora") == {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES}, config.get("lora"), failures))
    checks.append(check("optimizer config fixed", config.get("learning_rate") == LEARNING_RATE and config.get("per_device_train_batch_size") == PER_DEVICE_TRAIN_BATCH_SIZE and config.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION_STEPS and config.get("warmup_steps") == WARMUP_STEPS, config, failures))

    mask_tests = subprocess.run(["python3", str(EXPERIMENT_DIR / "test_attention_masks.py")], text=True, capture_output=True)
    checks.append(check("deterministic mask tests pass", mask_tests.returncode == 0, {"stdout": mask_tests.stdout, "stderr": mask_tests.stderr}, failures))

    train_script = (EXPERIMENT_DIR / "slurm" / "train_prompt_bidir_array.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "train_prompt_bidir_array.sh").exists() else ""
    eval_script = (EXPERIMENT_DIR / "slurm" / "evaluate_validation_array.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "evaluate_validation_array.sh").exists() else ""
    smoke_script = (EXPERIMENT_DIR / "slurm" / "smoke_attention_control.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "smoke_attention_control.sh").exists() else ""
    runtime_script = (EXPERIMENT_DIR / "slurm" / "runtime_diagnostics.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "runtime_diagnostics.sh").exists() else ""
    inference_smoke_script = (EXPERIMENT_DIR / "slurm" / "inference_smoke_cached_prompt_bidir.sh").read_text(encoding="utf-8") if (EXPERIMENT_DIR / "slurm" / "inference_smoke_cached_prompt_bidir.sh").exists() else ""
    training_code = (EXPERIMENT_DIR / "train_attention_control.py").read_text(encoding="utf-8")
    evaluation_code = (EXPERIMENT_DIR / "evaluate_attention_control.py").read_text(encoding="utf-8")
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in EXPERIMENT_DIR.rglob("*") if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"} and path.name != "static_validation_report.json")
    forbidden_eval_path = "test" + "_eval"
    checks.append(check("training array trains only prompt-bidirectional models", shell_array_items(train_script) == PROMPT_BIDIR_TRAINING_CONDITIONS, shell_array_items(train_script), failures))
    checks.append(check("validation array evaluates only prompt-bidirectional models", shell_array_items(eval_script) == PROMPT_BIDIR_TRAINING_CONDITIONS, shell_array_items(eval_script), failures))
    checks.append(check("smoke covers causal and prompt-bidirectional", "causal_original_only" in smoke_script and "prompt_bidir_original_only" in smoke_script, smoke_script, failures))
    checks.append(check("scripts use max_steps 232 and max_length 1536", f"--max_steps {MAX_STEPS}" in train_script and f"--max_length {TRAIN_MAX_LENGTH}" in train_script and f"--max_length {TRAIN_MAX_LENGTH}" in smoke_script, {"train": train_script, "smoke": smoke_script}, failures))
    checks.append(check("full training requires runtime diagnostics gate", "--require-runtime-diagnostics" in train_script and "--require-causal-references" in train_script, train_script, failures))
    checks.append(check("runtime diagnostics slurm invokes model-level checks", "runtime_attention_diagnostics.py" in runtime_script and "--max_length 1536" in runtime_script, runtime_script, failures))
    runtime_code = (EXPERIMENT_DIR / "runtime_attention_diagnostics.py").read_text(encoding="utf-8")
    checks.append(check("runtime diagnostics check native causal and prompt-bidir cache controls", "--cached_decode_steps 5" in inference_smoke_script and "native_causal_forced_cache_control" in runtime_code and "prompt_bidirectional_forced_cache_control" in runtime_code, inference_smoke_script, failures))
    checks.append(check("prompt-bidirectional evaluation uses cached decode", "prompt_bidirectional_prefill_cached_causal_decode" in evaluation_code and "use_cache\": True" in evaluation_code and "use_cache=False)" not in evaluation_code, "cached generation path present", failures))
    checks.append(check("causal training uses native Trainer path", "trainer_cls = Trainer if attention_type == \"causal\" else MaskedTrainer" in training_code and "include_prompt_lengths=(attention_type == \"prompt_bidirectional\")" in training_code, "native causal path is selected conditionally", failures))
    checks.append(check("evaluation uses max_new_tokens 512", f"--max_new_tokens {MAX_NEW_TOKENS}" in eval_script, eval_script, failures))
    checks.append(check("no held-out evaluation path referenced", forbidden_eval_path not in all_text, "checked by string without opening held-out eval data", failures))
    checks.append(check("separate output root", "cogmath_reasoning_attention_control_qwen3_4b" in OUTPUT_ROOT and "cogmath_reasoning_budget_control_qwen3_4b" not in OUTPUT_ROOT, OUTPUT_ROOT, failures))

    causal_refs = causal_reference_checks(args.require_causal_references)
    checks.append(check("causal budget references compatible or explicitly pending", causal_refs["passed"], causal_refs, failures))
    runtime_diags = runtime_diagnostics_checks(args.require_runtime_diagnostics)
    checks.append(check("cluster runtime diagnostics pass or explicitly pending", runtime_diags["passed"], runtime_diags, failures))

    syn = syntax()
    checks.append(check("python syntax parses", all(value == "ok" for value in syn["python"].values()), syn["python"], failures))
    checks.append(check("json files parse", all(value == "ok" for value in syn["json"].values()), syn["json"], failures))
    checks.append(check("slurm bash syntax", all(value["returncode"] == 0 for value in syn["bash"].values()), syn["bash"], failures))

    out = {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "dataset_report": report,
        "note": "Static validation reads split IDs for leakage checks but does not open the held-out evaluation file.",
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
