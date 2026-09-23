#!/usr/bin/env python3
"""Static validation for the unique-full Qwen3-4B augmentation setup."""

from __future__ import annotations

import hashlib
import ast
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from experiment_utils import make_prompt, read_jsonl
from train_lora import tokenize_prompt_completion
from unique_full_utils import CONDITION_DIMENSIONS, OUTPUT_ROOT, UNIQUE_FULL_DIR, UNIQUE_FULL_TRAINING_CONDITIONS


EXPERIMENT_DIR = Path(__file__).resolve().parent
LEGACY_OUTPUT_ASSIGNMENTS = {
    "OUTPUT_ROOT=outputs/cogmath_augmentation\n",
    "OUTPUT_ROOT=outputs/cogmath_augmentation_qwen3_4b\n",
}


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_json() -> Dict[str, Any]:
    files = [UNIQUE_FULL_DIR / "conditions.json", UNIQUE_FULL_DIR / "dataset_report.json"]
    for path in files:
        json.loads(path.read_text(encoding="utf-8"))
    for filename in UNIQUE_FULL_TRAINING_CONDITIONS.values():
        rows = read_jsonl(UNIQUE_FULL_DIR / filename)
        if not rows:
            raise AssertionError(f"{filename} has no rows.")
    return {"passed": True, "files": [str(path) for path in files]}


def check_python_syntax() -> Dict[str, Any]:
    files = [
        "unique_full_utils.py",
        "prepare_unique_full.py",
        "train_lora_unique_full.py",
        "evaluate_model_unique_full.py",
        "static_validation_unique_full.py",
        "analyze_unique_full.py",
    ]
    for filename in files:
        path = EXPERIMENT_DIR / filename
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {"passed": True, "files": files}


def check_datasets() -> Dict[str, Any]:
    splits = json.loads((EXPERIMENT_DIR / "splits.json").read_text(encoding="utf-8"))
    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    tokenizer = FakeTokenizer()
    result = {}

    for condition, filename in UNIQUE_FULL_TRAINING_CONDITIONS.items():
        path = UNIQUE_FULL_DIR / filename
        rows = read_jsonl(path)
        record_ids = [row["record_id"] for row in rows]
        duplicate_record_ids = sorted(rid for rid, count in Counter(record_ids).items() if count > 1)
        by_dimension = Counter(int(row["dimension"]) for row in rows)
        allowed = set(CONDITION_DIMENSIONS[condition])
        unintended = sorted({int(row["dimension"]) for row in rows} - allowed)
        validation_overlap = sorted({row["base_id"] for row in rows} & validation)
        test_overlap = sorted({row["base_id"] for row in rows} & test)
        non_train = sorted({row["base_id"] for row in rows} - train)
        rep_ids = sorted(row["record_id"] for row in rows if "_rep" in row["record_id"])
        repeated_originals = [
            base_id
            for base_id, count in Counter(row["base_id"] for row in rows if int(row["dimension"]) == 0).items()
            if count > 1
        ]
        target_errors = [
            row["record_id"]
            for row in rows
            if row.get("training_target") != f"Final answer: {row.get('final_answer')}"
        ]

        for row in rows[: min(len(rows), 20)]:
            encoded = tokenize_prompt_completion(tokenizer, make_prompt(row["question"]), row["training_target"], max_length=8192)
            if not any(label != -100 for label in encoded["labels"]):
                raise AssertionError(f"{row['record_id']} has no completion labels after tokenization.")

        checks = {
            "duplicate_record_ids": duplicate_record_ids,
            "repeated_original_base_ids": repeated_originals,
            "rep_record_ids": rep_ids,
            "validation_overlap": validation_overlap,
            "test_overlap": test_overlap,
            "non_train_base_ids": non_train,
            "unintended_dimensions": unintended,
            "target_error_count": len(target_errors),
        }
        if any(checks.values()):
            raise AssertionError(f"{condition} failed static checks: {checks}")

        result[condition] = {
            "file": str(path),
            "rows": len(rows),
            "unique_base_ids": len({row["base_id"] for row in rows}),
            "counts_by_dimension": {f"dim{dim}": by_dimension.get(dim, 0) for dim in [0, 1, 2, 4, 6]},
            "sha256": sha256(path),
            **checks,
        }

    if set(CONDITION_DIMENSIONS["original_plus_dim2_dim4"]) != {0, 2, 4}:
        raise AssertionError("original_plus_dim2_dim4 mapping is wrong.")
    if set(CONDITION_DIMENSIONS["original_plus_all"]) != {0, 1, 2, 4, 6}:
        raise AssertionError("original_plus_all mapping is wrong.")
    return {"passed": True, "conditions": result}


def check_deterministic_regeneration() -> Dict[str, Any]:
    report = json.loads((UNIQUE_FULL_DIR / "dataset_report.json").read_text(encoding="utf-8"))
    mismatches = {}
    for filename, info in report["output_files"].items():
        current = sha256(UNIQUE_FULL_DIR / filename)
        if current != info["sha256"]:
            mismatches[filename] = {"report": info["sha256"], "current": current}
    if mismatches:
        raise AssertionError(f"Dataset hashes no longer match dataset_report.json: {mismatches}")
    return {"passed": True, "checked_files": sorted(report["output_files"])}


def check_slurm() -> Dict[str, Any]:
    scripts = [
        "train_smoke_unique_full_qwen3_4b.sh",
        "train_array_unique_full_qwen3_4b.sh",
        "evaluate_validation_unique_full_qwen3_4b.sh",
        "evaluate_test_unique_full_qwen3_4b.sh",
        "analyze_unique_full_qwen3_4b.sh",
    ]
    result = {}
    for script in scripts:
        path = EXPERIMENT_DIR / "slurm" / script
        completed = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
        result[script] = {"returncode": completed.returncode, "stderr": completed.stderr}
        if completed.returncode != 0:
            raise AssertionError(f"Shell syntax failed for {script}: {completed.stderr}")
        text = path.read_text(encoding="utf-8")
        if OUTPUT_ROOT not in text:
            raise AssertionError(f"{script} does not use the unique-full output root.")
        for legacy_assignment in LEGACY_OUTPUT_ASSIGNMENTS:
            if legacy_assignment in text:
                raise AssertionError(f"{script} collides with legacy output assignment {legacy_assignment.strip()}.")
    return {"passed": True, "scripts": result}


def main() -> None:
    report = {
        "python_syntax": check_python_syntax(),
        "json": check_json(),
        "datasets": check_datasets(),
        "deterministic_regeneration": check_deterministic_regeneration(),
        "slurm": check_slurm(),
        "output_root": OUTPUT_ROOT,
        "legacy_output_collision": False,
    }
    out = UNIQUE_FULL_DIR / "static_validation_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
