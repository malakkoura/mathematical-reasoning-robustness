#!/usr/bin/env python3
"""Constants for the unique-full Qwen3-4B augmentation experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


EXPERIMENT_DIR = Path(__file__).resolve().parent
UNIQUE_FULL_DIR = EXPERIMENT_DIR / "data_unique_full"
OUTPUT_ROOT = "outputs/cogmath_augmentation_qwen3_4b_unique_full"
MODEL_NAME = "Qwen/Qwen3-4B-Base"

UNIQUE_FULL_TRAINING_CONDITIONS: Dict[str, str] = {
    "original_only": "train_original_only.jsonl",
    "original_plus_dim1": "train_original_plus_dim1.jsonl",
    "original_plus_dim2": "train_original_plus_dim2.jsonl",
    "original_plus_dim4": "train_original_plus_dim4.jsonl",
    "original_plus_dim6": "train_original_plus_dim6.jsonl",
    "original_plus_dim2_dim4": "train_original_plus_dim2_dim4.jsonl",
    "original_plus_all": "train_original_plus_all.jsonl",
}

UNIQUE_FULL_EVALUATION_CONDITIONS: List[str] = [
    "base_untuned",
    *UNIQUE_FULL_TRAINING_CONDITIONS.keys(),
]

CONDITION_DIMENSIONS = {
    "original_only": [0],
    "original_plus_dim1": [0, 1],
    "original_plus_dim2": [0, 2],
    "original_plus_dim4": [0, 4],
    "original_plus_dim6": [0, 6],
    "original_plus_dim2_dim4": [0, 2, 4],
    "original_plus_all": [0, 1, 2, 4, 6],
}

ARRAY_CONDITIONS = [
    "original_only",
    "original_plus_dim1",
    "original_plus_dim2",
    "original_plus_dim4",
    "original_plus_dim6",
    "original_plus_dim2_dim4",
    "original_plus_all",
]

PRIMARY_COMPARISONS = [
    ("original_only", "original_plus_dim1"),
    ("original_only", "original_plus_dim2"),
    ("original_only", "original_plus_dim4"),
    ("original_only", "original_plus_dim6"),
    ("original_only", "original_plus_dim2_dim4"),
    ("original_only", "original_plus_all"),
    ("original_plus_dim2", "original_plus_dim2_dim4"),
    ("original_plus_dim4", "original_plus_dim2_dim4"),
    ("original_plus_dim2_dim4", "original_plus_all"),
]
