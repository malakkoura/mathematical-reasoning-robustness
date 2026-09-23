#!/usr/bin/env python3
"""Constants for the Qwen3-4B reasoning-augmentation experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
AUGMENTATION_DIR = REPO_ROOT / "experiments" / "cogmath_augmentation"
SUPERVISION_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_supervision"
DATA_DIR = EXPERIMENT_DIR / "data"

MODEL_NAME = "Qwen/Qwen3-4B-Base"
OUTPUT_ROOT = "outputs/cogmath_reasoning_augmentation_qwen3_4b"
DIRECT_OUTPUT_ROOT = "outputs/cogmath_augmentation_qwen3_4b_unique_full"
REASONING_SUPERVISION_OUTPUT_ROOT = "outputs/cogmath_reasoning_supervision_qwen3_4b"

VALIDATION_EVAL_FILE = AUGMENTATION_DIR / "validation_eval.jsonl"
SPLITS_FILE = AUGMENTATION_DIR / "splits.json"
TRAIN_MAX_LENGTH = 1536

EXPLICIT_DIM6_EXCLUSIONS = {
    "GSM8K@855_dim6": "transformed setup does not yield the required valid integer-age solution",
    "GSM8K@1134_dim6": "reasoning concludes 5 while prepared final_answer is 75",
    "GSM8K@1155_dim6": "reasoning states the transformed problem cannot be calculated",
}

SOURCE_TRAINING_FILES: Dict[str, str] = {
    "reasoning_original_only": "train_original_only.jsonl",
    "reasoning_original_plus_dim1": "train_original_plus_dim1.jsonl",
    "reasoning_original_plus_dim2": "train_original_plus_dim2.jsonl",
    "reasoning_original_plus_dim4": "train_original_plus_dim4.jsonl",
    "reasoning_original_plus_dim6": "train_original_plus_dim6.jsonl",
    "reasoning_original_plus_dim2_dim4": "train_original_plus_dim2_dim4.jsonl",
    "reasoning_original_plus_all": "train_original_plus_all.jsonl",
}

TRAINING_CONDITIONS = [
    "reasoning_original_plus_dim1",
    "reasoning_original_plus_dim2",
    "reasoning_original_plus_dim4",
    "reasoning_original_plus_dim6",
    "reasoning_original_plus_dim2_dim4",
    "reasoning_original_plus_all",
]

EVALUATION_CONDITIONS = [
    "reasoning_original_only",
    *TRAINING_CONDITIONS,
]

CONDITION_DIMENSIONS = {
    "reasoning_original_only": [0],
    "reasoning_original_plus_dim1": [0, 1],
    "reasoning_original_plus_dim2": [0, 2],
    "reasoning_original_plus_dim4": [0, 4],
    "reasoning_original_plus_dim6": [0, 6],
    "reasoning_original_plus_dim2_dim4": [0, 2, 4],
    "reasoning_original_plus_all": [0, 1, 2, 4, 6],
}

PRIMARY_COMPARISONS: List[Tuple[str, str]] = [
    ("reasoning_original_only", "reasoning_original_plus_dim1"),
    ("reasoning_original_only", "reasoning_original_plus_dim2"),
    ("reasoning_original_only", "reasoning_original_plus_dim4"),
    ("reasoning_original_only", "reasoning_original_plus_dim6"),
    ("reasoning_original_only", "reasoning_original_plus_dim2_dim4"),
    ("reasoning_original_only", "reasoning_original_plus_all"),
    ("reasoning_original_plus_dim2", "reasoning_original_plus_dim2_dim4"),
    ("reasoning_original_plus_dim4", "reasoning_original_plus_dim2_dim4"),
    ("reasoning_original_plus_dim2_dim4", "reasoning_original_plus_all"),
]

DIRECT_CONDITION_MAP = {
    "reasoning_original_only": "original_only",
    "reasoning_original_plus_dim1": "original_plus_dim1",
    "reasoning_original_plus_dim2": "original_plus_dim2",
    "reasoning_original_plus_dim4": "original_plus_dim4",
    "reasoning_original_plus_dim6": "original_plus_dim6",
    "reasoning_original_plus_dim2_dim4": "original_plus_dim2_dim4",
    "reasoning_original_plus_all": "original_plus_all",
}

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

REASONING_PROMPT_TEMPLATE = """Solve the following maths problem step by step. Finish with exactly one line in this format:
Final answer: <answer>

Question:
{question}

Answer:
"""

REASONING_TARGET_TEMPLATE = """Reasoning:
{reasoning}

Final answer: {final_answer}"""


def source_training_path(condition: str) -> Path:
    return AUGMENTATION_DIR / "data_unique_full" / SOURCE_TRAINING_FILES[condition]


def reasoning_training_path(condition: str) -> Path:
    return DATA_DIR / f"train_{condition}.jsonl"


def adapter_dir_for(condition: str) -> Optional[Path]:
    if condition == "reasoning_original_only":
        return Path(REASONING_SUPERVISION_OUTPUT_ROOT) / "adapters" / "reasoning_original_only"
    if condition in TRAINING_CONDITIONS:
        return Path(OUTPUT_ROOT) / "adapters" / condition
    raise KeyError(condition)


def direct_validation_dir_for(reasoning_condition: str) -> Path:
    return Path(DIRECT_OUTPUT_ROOT) / "validation_evaluations" / DIRECT_CONDITION_MAP[reasoning_condition]


def make_reasoning_prompt(question: str) -> str:
    return REASONING_PROMPT_TEMPLATE.format(question=question)
