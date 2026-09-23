#!/usr/bin/env python3
"""Static checks for the separate 512-token reasoning validation diagnostic."""

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
    BASE_REASONING_CONDITION,
    BASE_REASONING_CONDITION_512,
    DIRECT_ADAPTER_REASONING_CONDITION,
    DIRECT_ADAPTER_REASONING_CONDITION_512,
    DIRECT_ORIGINAL_ONLY_ADAPTER_DIR,
    EXPERIMENT_DIR,
    OUTPUT_ROOT,
    REASONING_ADAPTER_CONDITION,
    REASONING_ADAPTER_CONDITION_512,
    REASONING_ADAPTER_DIR,
    REASONING_PROMPT_TEMPLATE,
    VALIDATION_EVAL_FILE,
)

sys.path.insert(0, str(AUGMENTATION_DIR))
from experiment_utils import read_jsonl, write_json  # noqa: E402


EVAL_512_SCRIPT = EXPERIMENT_DIR / "slurm" / "evaluate_validation_512_array.sh"
ANALYSIS_512_SCRIPT = EXPERIMENT_DIR / "slurm" / "analyze_validation_512.sh"
ANALYSIS_512_PY = EXPERIMENT_DIR / "analyze_reasoning_supervision_512.py"
EVAL_PY = EXPERIMENT_DIR / "evaluate_reasoning.py"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the 512-token reasoning evaluation setup.")
    parser.add_argument("--strict-results-check", action="store_true")
    parser.add_argument("--write-report", type=Path, default=EXPERIMENT_DIR / "static_validation_512_report.json")
    return parser.parse_args(argv)


def check(name: str, passed: bool, details: Any, failures: List[str]) -> Dict[str, Any]:
    if not passed:
        failures.append(name)
    return {"name": name, "passed": passed, "details": details}


def bash_n(path: Path) -> Dict[str, Any]:
    completed = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
    return {"returncode": completed.returncode, "stderr": completed.stderr}


def adapter_files_present(path: Path) -> bool:
    if not path.exists():
        return False
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    )


def prediction_ids(path: Path) -> List[str]:
    return [row["record_id"] for row in read_jsonl(path)] if path.exists() else []


def strict_result_details() -> Dict[str, Any]:
    eval_ids = [row["record_id"] for row in read_jsonl(VALIDATION_EVAL_FILE)]
    pairs = [
        (BASE_REASONING_CONDITION, BASE_REASONING_CONDITION_512),
        (DIRECT_ADAPTER_REASONING_CONDITION, DIRECT_ADAPTER_REASONING_CONDITION_512),
        (REASONING_ADAPTER_CONDITION, REASONING_ADAPTER_CONDITION_512),
    ]
    details = {"validation_record_count": len(eval_ids), "pairs": {}}
    root_256 = Path(OUTPUT_ROOT) / "validation_evaluations"
    root_512 = Path(OUTPUT_ROOT) / "validation_evaluations_512"
    for cond_256, cond_512 in pairs:
        ids_256 = prediction_ids(root_256 / cond_256 / "predictions.jsonl")
        ids_512 = prediction_ids(root_512 / cond_512 / "predictions.jsonl")
        details["pairs"][f"{cond_256}_vs_{cond_512}"] = {
            "has_256_predictions": bool(ids_256),
            "has_512_predictions": bool(ids_512),
            "same_as_validation_order_256": ids_256 == eval_ids,
            "same_as_validation_order_512": ids_512 == eval_ids,
            "same_256_512_order": ids_256 == ids_512,
        }
    return details


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []

    eval_script_text = EVAL_512_SCRIPT.read_text(encoding="utf-8")
    analysis_script_text = ANALYSIS_512_SCRIPT.read_text(encoding="utf-8")
    eval_py_text = EVAL_PY.read_text(encoding="utf-8")
    validation_rows = read_jsonl(VALIDATION_EVAL_FILE)

    output_256 = Path(OUTPUT_ROOT) / "validation_evaluations"
    output_512 = Path(OUTPUT_ROOT) / "validation_evaluations_512"
    analysis_512 = Path(OUTPUT_ROOT) / "validation_analysis_512"
    checks.append(
        check(
            "512 results cannot overwrite 256 results",
            output_512 != output_256
            and "validation_evaluations_512" in eval_script_text
            and "validation_evaluations/${CONDITION}" not in eval_script_text,
            {"output_256": str(output_256), "output_512": str(output_512)},
            failures,
        )
    )
    checks.append(
        check(
            "512 analysis directory is separate",
            "validation_analysis_512" in analysis_script_text and analysis_512 != Path(OUTPUT_ROOT) / "analysis" / "validation",
            {"analysis_512": str(analysis_512)},
            failures,
        )
    )
    checks.append(check("same validation record count available", len(validation_rows) == 990, len(validation_rows), failures))
    checks.append(
        check(
            "same prompt template and evaluator are used",
            "make_reasoning_prompt" in eval_py_text
            and "extract_after_final_marker" in eval_py_text
            and "answers_match" in eval_py_text
            and "Final answer: <answer>" in REASONING_PROMPT_TEMPLATE,
            {"evaluator": str(EVAL_PY)},
            failures,
        )
    )
    checks.append(
        check(
            "same adapters are referenced",
            "/adapters/original_only" in eval_script_text
            and "/adapters/reasoning_original_only" in eval_script_text
            and str(DIRECT_ORIGINAL_ONLY_ADAPTER_DIR).endswith("/adapters/original_only")
            and str(REASONING_ADAPTER_DIR).endswith("/adapters/reasoning_original_only"),
            {
                "direct_adapter": str(DIRECT_ORIGINAL_ONLY_ADAPTER_DIR),
                "reasoning_adapter": str(REASONING_ADAPTER_DIR),
            },
            failures,
        )
    )
    checks.append(
        check(
            "only inference generation budget changes to 512",
            "--max_new_tokens 512" in eval_script_text
            and "--batch_size 8" in eval_script_text
            and "--warmup_generations 1" in eval_script_text
            and "--eval_file experiments/cogmath_augmentation/validation_eval.jsonl" in eval_script_text,
            {"eval_script": str(EVAL_512_SCRIPT)},
            failures,
        )
    )
    checks.append(
        check(
            "no test-set access in 512 jobs",
            "test_eval.jsonl" not in eval_script_text
            and "test_eval.jsonl" not in analysis_script_text
            and "evaluate_test" not in eval_script_text,
            {"eval_script": str(EVAL_512_SCRIPT), "analysis_script": str(ANALYSIS_512_SCRIPT)},
            failures,
        )
    )

    syntax_details: Dict[str, Any] = {}
    for path in [EVAL_PY, ANALYSIS_512_PY, Path(__file__).resolve()]:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            syntax_details[str(path)] = "ok"
        except SyntaxError as exc:
            syntax_details[str(path)] = str(exc)
    checks.append(check("python syntax parses", all(value == "ok" for value in syntax_details.values()), syntax_details, failures))

    json_details = {}
    for path in sorted((EXPERIMENT_DIR / "resolved_configs").glob("*512*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            json_details[str(path)] = "ok"
        except json.JSONDecodeError as exc:
            json_details[str(path)] = str(exc)
    checks.append(check("512 json configs parse", all(value == "ok" for value in json_details.values()), json_details, failures))

    bash_details = {str(path): bash_n(path) for path in [EVAL_512_SCRIPT, ANALYSIS_512_SCRIPT]}
    checks.append(check("512 slurm bash syntax", all(item["returncode"] == 0 for item in bash_details.values()), bash_details, failures))

    adapter_presence = {
        "direct_adapter_present": adapter_files_present(DIRECT_ORIGINAL_ONLY_ADAPTER_DIR),
        "reasoning_adapter_present": adapter_files_present(REASONING_ADAPTER_DIR),
    }
    checks.append(
        check(
            "adapter presence is not required for local static validation",
            True,
            adapter_presence,
            failures,
        )
    )

    result_details = strict_result_details()
    strict_passed = all(
        pair["same_as_validation_order_256"] and pair["same_as_validation_order_512"] and pair["same_256_512_order"]
        for pair in result_details["pairs"].values()
    )
    checks.append(
        check(
            "strict result record IDs/order match validation and 256 outputs",
            strict_passed if args.strict_results_check else True,
            {"strict_results_check": args.strict_results_check, **result_details},
            failures,
        )
    )

    report = {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "conditions_512": [
            BASE_REASONING_CONDITION_512,
            DIRECT_ADAPTER_REASONING_CONDITION_512,
            REASONING_ADAPTER_CONDITION_512,
        ],
        "notes": [
            "This is evaluation-only; no training files or adapters are modified.",
            "Use --strict-results-check after 512 predictions are complete.",
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
