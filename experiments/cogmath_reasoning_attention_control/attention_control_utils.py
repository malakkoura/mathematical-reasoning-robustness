#!/usr/bin/env python3
"""Constants and helpers for the reasoning attention-structure control."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
AUGMENTATION_DIR = REPO_ROOT / "experiments" / "cogmath_augmentation"
REASONING_AUG_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_augmentation"
BUDGET_CONTROL_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_budget_control"

MODEL_NAME = "Qwen/Qwen3-4B-Base"
OUTPUT_ROOT = "outputs/cogmath_reasoning_attention_control_qwen3_4b"
BUDGET_OUTPUT_ROOT = "outputs/cogmath_reasoning_budget_control_qwen3_4b"

TRAIN_MAX_LENGTH = 1536
MAX_STEPS = 232
MAX_NEW_TOKENS = 512
SEED = 42
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_STEPS = 20
LOGGING_STEPS = 5
BF16 = True
GRADIENT_CHECKPOINTING = True
USE_CACHE = False

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

VALIDATION_EVAL_FILE = AUGMENTATION_DIR / "validation_eval.jsonl"
SPLITS_FILE = AUGMENTATION_DIR / "splits.json"
SOURCE_DATA_DIR = REASONING_AUG_DIR / "data"
SOURCE_REPORT_FILE = SOURCE_DATA_DIR / "dataset_report.json"

TRAINING_DATASETS = ["original_only", "original_plus_dim2", "original_plus_all"]
ATTENTION_TYPES = ["causal", "prompt_bidirectional"]

PROMPT_BIDIR_TRAINING_CONDITIONS = [
    "prompt_bidir_original_only",
    "prompt_bidir_original_plus_dim2",
    "prompt_bidir_original_plus_all",
]
CAUSAL_CONDITIONS = [
    "causal_original_only",
    "causal_original_plus_dim2",
    "causal_original_plus_all",
]
ALL_CONDITIONS = CAUSAL_CONDITIONS + PROMPT_BIDIR_TRAINING_CONDITIONS

SOURCE_CONDITION_BY_TRAINING_DATA = {
    "original_only": "reasoning_original_only",
    "original_plus_dim2": "reasoning_original_plus_dim2",
    "original_plus_all": "reasoning_original_plus_all",
}
SOURCE_CONDITION_BY_ATTENTION_CONDITION = {
    "causal_original_only": "reasoning_original_only",
    "causal_original_plus_dim2": "reasoning_original_plus_dim2",
    "causal_original_plus_all": "reasoning_original_plus_all",
    "prompt_bidir_original_only": "reasoning_original_only",
    "prompt_bidir_original_plus_dim2": "reasoning_original_plus_dim2",
    "prompt_bidir_original_plus_all": "reasoning_original_plus_all",
}
TRAINING_DATA_BY_CONDITION = {
    "causal_original_only": "original_only",
    "causal_original_plus_dim2": "original_plus_dim2",
    "causal_original_plus_all": "original_plus_all",
    "prompt_bidir_original_only": "original_only",
    "prompt_bidir_original_plus_dim2": "original_plus_dim2",
    "prompt_bidir_original_plus_all": "original_plus_all",
}
ATTENTION_TYPE_BY_CONDITION = {
    "causal_original_only": "causal",
    "causal_original_plus_dim2": "causal",
    "causal_original_plus_all": "causal",
    "prompt_bidir_original_only": "prompt_bidirectional",
    "prompt_bidir_original_plus_dim2": "prompt_bidirectional",
    "prompt_bidir_original_plus_all": "prompt_bidirectional",
}
BUDGET_CONDITION_BY_CAUSAL = {
    "causal_original_only": "budget_original_only",
    "causal_original_plus_dim2": "budget_original_plus_dim2",
    "causal_original_plus_all": "budget_original_plus_all",
}
CAUSAL_CONDITION_BY_BUDGET = {value: key for key, value in BUDGET_CONDITION_BY_CAUSAL.items()}

EXPECTED_DIMENSION_COUNTS = {
    "original_only": {0: 923},
    "original_plus_dim2": {0: 923, 2: 923},
    "original_plus_all": {0: 923, 1: 923, 2: 923, 4: 923, 6: 917},
}

FORMS = [
    ("overall", None, "Overall"),
    ("original", 0, "Original"),
    ("dim1_paraphrasing", 1, "Dim1 paraphrasing"),
    ("dim2_word_scrambling", 2, "Dim2 word scrambling"),
    ("dim4_irrelevant_information", 4, "Dim4 irrelevant information"),
    ("dim6_numerical_variation", 6, "Dim6 numerical variation"),
]

REASONING_PROMPT_TEMPLATE = """Solve the following maths problem step by step. Finish with exactly one line in this format:
Final answer: <answer>

Question:
{question}

Answer:
"""


def source_training_path_for_data(training_data: str) -> Path:
    return SOURCE_DATA_DIR / f"train_{SOURCE_CONDITION_BY_TRAINING_DATA[training_data]}.jsonl"


def source_training_path_for_condition(condition: str) -> Path:
    return SOURCE_DATA_DIR / f"train_{SOURCE_CONDITION_BY_ATTENTION_CONDITION[condition]}.jsonl"


def adapter_dir_for(condition: str) -> Path:
    if condition in BUDGET_CONDITION_BY_CAUSAL:
        return Path(BUDGET_OUTPUT_ROOT) / "adapters" / BUDGET_CONDITION_BY_CAUSAL[condition]
    return Path(OUTPUT_ROOT) / "adapters" / condition


def evaluation_dir_for(condition: str) -> Path:
    if condition in BUDGET_CONDITION_BY_CAUSAL:
        return Path(BUDGET_OUTPUT_ROOT) / "validation_evaluations" / BUDGET_CONDITION_BY_CAUSAL[condition]
    return Path(OUTPUT_ROOT) / "validation_evaluations" / condition


def make_reasoning_prompt(question: str) -> str:
    return REASONING_PROMPT_TEMPLATE.format(question=question)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def row_order_sha256(record_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for record_id in record_ids:
        digest.update(str(record_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_report() -> Dict[str, Any]:
    source_datasets = {}
    for training_data in TRAINING_DATASETS:
        path = source_training_path_for_data(training_data)
        rows = read_jsonl(path)
        details = composition(rows)
        details.update(
            {
                "training_data": training_data,
                "source_condition": SOURCE_CONDITION_BY_TRAINING_DATA[training_data],
                "source_file": str(path),
                "source_sha256": sha256_file(path),
                "row_order_sha256": row_order_sha256(row["record_id"] for row in rows),
                **trainer_epoch_group_example_counts(len(rows)),
            }
        )
        source_datasets[training_data] = details

    return {
        "experiment": "cogmath_reasoning_attention_control_qwen3_4b",
        "research_question": "Does prompt-bidirectional input attention interact with reasoning-augmentation data under matched Qwen3-4B LoRA fine-tuning?",
        "model_name": MODEL_NAME,
        "output_root": OUTPUT_ROOT,
        "budget_control_output_root_for_causal_reuse": BUDGET_OUTPUT_ROOT,
        "source_data_dir": str(SOURCE_DATA_DIR),
        "training_max_length": TRAIN_MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "validation_max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "attention_types": ATTENTION_TYPES,
        "training_datasets": TRAINING_DATASETS,
        "conditions": {
            condition: {
                "attention_type": ATTENTION_TYPE_BY_CONDITION[condition],
                "training_data": TRAINING_DATA_BY_CONDITION[condition],
                "source_condition": SOURCE_CONDITION_BY_ATTENTION_CONDITION[condition],
                "source_file": str(source_training_path_for_condition(condition)),
                "adapter_dir": str(adapter_dir_for(condition)),
                "validation_evaluation_dir": str(evaluation_dir_for(condition)),
                "causal_reference_condition": BUDGET_CONDITION_BY_CAUSAL.get(condition),
            }
            for condition in ALL_CONDITIONS
        },
        "source_datasets": source_datasets,
        "methodological_notes": [
            "The prompt-bidirectional condition changes only the model attention mask: prompt/input tokens are mutually visible; generated answer tokens remain causal.",
            "The model-visible prompt does not contain an explicit boundary marker. The boundary is the token-level prompt length.",
            "Causal adapters and validation predictions are reused from the budget-control experiment only after static validation confirms exact source-data, row-order, and hyperparameter compatibility.",
            "No CogMath test-evaluation JSONL is opened by preparation or static validation.",
        ],
    }
