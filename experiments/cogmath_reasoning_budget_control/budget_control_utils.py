#!/usr/bin/env python3
"""Constants and helpers for the reasoning-augmentation budget-control experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
REASONING_AUG_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_augmentation"
REASONING_SUPERVISION_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_supervision"
AUGMENTATION_DIR = REPO_ROOT / "experiments" / "cogmath_augmentation"

MODEL_NAME = "Qwen/Qwen3-4B-Base"
OUTPUT_ROOT = "outputs/cogmath_reasoning_budget_control_qwen3_4b"
TRAIN_MAX_LENGTH = 1536
MAX_STEPS = 232
MAX_NEW_TOKENS = 512
SEED = 42

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
LOGGING_STEPS = 5
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

VALIDATION_EVAL_FILE = AUGMENTATION_DIR / "validation_eval.jsonl"
SPLITS_FILE = AUGMENTATION_DIR / "splits.json"
SOURCE_DATA_DIR = REASONING_AUG_DIR / "data"
SOURCE_REPORT_FILE = SOURCE_DATA_DIR / "dataset_report.json"

EXPLICIT_DIM6_EXCLUSIONS = {
    "GSM8K@855_dim6": "transformed setup does not yield the required valid integer-age solution",
    "GSM8K@1134_dim6": "reasoning concludes 5 while prepared final_answer is 75",
    "GSM8K@1155_dim6": "reasoning states the transformed problem cannot be calculated",
}

BUDGET_CONDITIONS: List[str] = [
    "budget_original_only",
    "budget_original_plus_dim2",
    "budget_original_plus_dim4",
    "budget_original_plus_dim2_dim4",
    "budget_original_plus_all",
]

SOURCE_CONDITION_BY_BUDGET: Dict[str, str] = {
    "budget_original_only": "reasoning_original_only",
    "budget_original_plus_dim2": "reasoning_original_plus_dim2",
    "budget_original_plus_dim4": "reasoning_original_plus_dim4",
    "budget_original_plus_dim2_dim4": "reasoning_original_plus_dim2_dim4",
    "budget_original_plus_all": "reasoning_original_plus_all",
}

EXPECTED_DIMENSION_COUNTS: Dict[str, Dict[int, int]] = {
    "budget_original_only": {0: 923},
    "budget_original_plus_dim2": {0: 923, 2: 923},
    "budget_original_plus_dim4": {0: 923, 4: 923},
    "budget_original_plus_dim2_dim4": {0: 923, 2: 923, 4: 923},
    "budget_original_plus_all": {0: 923, 1: 923, 2: 923, 4: 923, 6: 917},
}

PRIMARY_COMPARISONS: List[Tuple[str, str]] = [
    ("budget_original_only", "budget_original_plus_dim2"),
    ("budget_original_only", "budget_original_plus_dim4"),
    ("budget_original_only", "budget_original_plus_dim2_dim4"),
    ("budget_original_only", "budget_original_plus_all"),
    ("budget_original_plus_dim2", "budget_original_plus_dim2_dim4"),
    ("budget_original_plus_dim2", "budget_original_plus_all"),
    ("budget_original_plus_dim4", "budget_original_plus_dim2_dim4"),
    ("budget_original_plus_dim2_dim4", "budget_original_plus_all"),
]

REASONING_PROMPT_TEMPLATE = """Solve the following maths problem step by step. Finish with exactly one line in this format:
Final answer: <answer>

Question:
{question}

Answer:
"""


def source_training_path(condition: str) -> Path:
    return SOURCE_DATA_DIR / f"train_{SOURCE_CONDITION_BY_BUDGET[condition]}.jsonl"


def adapter_dir_for(condition: str) -> Path:
    return Path(OUTPUT_ROOT) / "adapters" / condition


def make_reasoning_prompt(question: str) -> str:
    return REASONING_PROMPT_TEMPLATE.format(question=question)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_rows(condition: str) -> List[Dict[str, Any]]:
    return read_jsonl(source_training_path(condition))


def composition(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dim = Counter(int(row["dimension"]) for row in rows)
    return {
        "rows": len(rows),
        "unique_base_ids": len({row["base_id"] for row in rows}),
        "counts_by_dimension": {str(dim): by_dim.get(dim, 0) for dim in [0, 1, 2, 4, 6] if by_dim.get(dim, 0)},
    }


def nominal_full_accumulation_examples(
    max_steps: int = MAX_STEPS,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
) -> int:
    return max_steps * gradient_accumulation_steps * per_device_train_batch_size


def trainer_epoch_group_example_counts(
    row_count: int,
    max_steps: int = MAX_STEPS,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
) -> Dict[str, Any]:
    """Estimate Trainer example presentations with epoch-local accumulation groups.

    Transformers chunks each finite-dataset epoch into gradient-accumulation groups.
    If an epoch ends with an incomplete group, that group still performs an optimizer
    step with fewer microbatches. With max_steps, training reiterates epochs until
    the requested optimizer-step count is reached.
    """
    if row_count <= 0:
        return {
            "estimated_example_presentations": 0,
            "effective_epochs": None,
            "optimizer_steps_per_epoch": 0,
            "incomplete_accumulation_groups_per_epoch": 0,
            "nominal_full_accumulation_examples": nominal_full_accumulation_examples(max_steps, gradient_accumulation_steps, per_device_train_batch_size),
            "shortfall_vs_nominal_full_accumulation": 0,
        }
    full_batches, last_batch_examples = divmod(row_count, per_device_train_batch_size)
    microbatch_sizes = [per_device_train_batch_size] * full_batches
    if last_batch_examples:
        microbatch_sizes.append(last_batch_examples)
    groups = [
        sum(microbatch_sizes[index:index + gradient_accumulation_steps])
        for index in range(0, len(microbatch_sizes), gradient_accumulation_steps)
    ]
    full_epochs, remainder_steps = divmod(max_steps, len(groups))
    examples = full_epochs * sum(groups) + sum(groups[:remainder_steps])
    nominal = nominal_full_accumulation_examples(max_steps, gradient_accumulation_steps, per_device_train_batch_size)
    return {
        "estimated_example_presentations": examples,
        "effective_epochs": examples / row_count,
        "optimizer_steps_per_epoch": len(groups),
        "incomplete_accumulation_groups_per_epoch": sum(1 for count in groups if count < gradient_accumulation_steps * per_device_train_batch_size),
        "last_group_examples_per_epoch": groups[-1],
        "nominal_full_accumulation_examples": nominal,
        "shortfall_vs_nominal_full_accumulation": nominal - examples,
        "example_presentation_estimate_note": "Estimated using Transformers finite-dataset epoch-local gradient-accumulation grouping; runtime training_exposure.json records observed dataset item presentations.",
    }


def expected_effective_epochs(row_count: int, max_steps: int = MAX_STEPS) -> float:
    estimate = trainer_epoch_group_example_counts(row_count, max_steps)
    return float(estimate["effective_epochs"]) if estimate["effective_epochs"] is not None else 0.0


def budget_dataset_report() -> Dict[str, Any]:
    conditions = {}
    for condition in BUDGET_CONDITIONS:
        path = source_training_path(condition)
        rows = read_jsonl(path)
        details = composition(rows)
        details.update(
            {
                "budget_condition": condition,
                "source_condition": SOURCE_CONDITION_BY_BUDGET[condition],
                "source_file": str(path),
                "source_sha256": sha256_file(path),
                "max_steps": MAX_STEPS,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                **trainer_epoch_group_example_counts(len(rows)),
            }
        )
        conditions[condition] = details
    return {
        "experiment": "cogmath_reasoning_budget_control_qwen3_4b",
        "model_name": MODEL_NAME,
        "output_root": OUTPUT_ROOT,
        "source_data_dir": str(SOURCE_DATA_DIR),
        "training_max_length": TRAIN_MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "validation_max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "note": "The optimizer update budget is controlled with max_steps=232. Example and token exposure may still differ because Transformers can step on incomplete accumulation groups at finite-dataset epoch boundaries and examples have different lengths.",
        "conditions": conditions,
    }
