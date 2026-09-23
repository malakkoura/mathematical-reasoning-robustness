#!/usr/bin/env python3
"""Constants and prompts for the Qwen3-4B reasoning-supervision diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
AUGMENTATION_DIR = REPO_ROOT / "experiments" / "cogmath_augmentation"
DATA_DIR = EXPERIMENT_DIR / "data"

MODEL_NAME = "Qwen/Qwen3-4B-Base"
TRAIN_MAX_LENGTH = 768
OUTPUT_ROOT = "outputs/cogmath_reasoning_supervision_qwen3_4b"
UNIQUE_FULL_OUTPUT_ROOT = "outputs/cogmath_augmentation_qwen3_4b_unique_full"

SOURCE_TRAIN_FILE = AUGMENTATION_DIR / "data_unique_full" / "train_original_only.jsonl"
REASONING_TRAIN_FILE = DATA_DIR / "train_original_only_reasoning.jsonl"
DATASET_REPORT = DATA_DIR / "dataset_report.json"
VALIDATION_EVAL_FILE = AUGMENTATION_DIR / "validation_eval.jsonl"
TEST_EVAL_FILE = AUGMENTATION_DIR / "test_eval.jsonl"

REASONING_TRAINING_CONDITION = "reasoning_original_only"
REASONING_ADAPTER_CONDITION = "reasoning_adapter_reasoning_prompt"
BASE_REASONING_CONDITION = "base_reasoning_prompt"
DIRECT_ADAPTER_REASONING_CONDITION = "direct_adapter_reasoning_prompt"
DIRECT_REFERENCE_CONDITION = "direct_adapter_direct_prompt_existing"

BASE_REASONING_CONDITION_512 = "base_reasoning_prompt_512"
DIRECT_ADAPTER_REASONING_CONDITION_512 = "direct_adapter_reasoning_prompt_512"
REASONING_ADAPTER_CONDITION_512 = "reasoning_adapter_reasoning_prompt_512"

REASONING_EVALUATION_CONDITIONS = [
    BASE_REASONING_CONDITION,
    DIRECT_ADAPTER_REASONING_CONDITION,
    REASONING_ADAPTER_CONDITION,
]

REASONING_EVALUATION_CONDITIONS_512 = [
    BASE_REASONING_CONDITION_512,
    DIRECT_ADAPTER_REASONING_CONDITION_512,
    REASONING_ADAPTER_CONDITION_512,
]

ALL_REASONING_EVALUATION_CONDITIONS = [
    *REASONING_EVALUATION_CONDITIONS,
    *REASONING_EVALUATION_CONDITIONS_512,
]

ANALYSIS_CONDITIONS = [
    BASE_REASONING_CONDITION,
    DIRECT_ADAPTER_REASONING_CONDITION,
    REASONING_ADAPTER_CONDITION,
    DIRECT_REFERENCE_CONDITION,
]

DIRECT_ORIGINAL_ONLY_ADAPTER_DIR = Path(UNIQUE_FULL_OUTPUT_ROOT) / "adapters" / "original_only"
DIRECT_ORIGINAL_ONLY_VALIDATION_DIR = Path(UNIQUE_FULL_OUTPUT_ROOT) / "validation_evaluations" / "original_only"
REASONING_ADAPTER_DIR = Path(OUTPUT_ROOT) / "adapters" / "reasoning_original_only"

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

PRIMARY_COMPARISONS = [
    (BASE_REASONING_CONDITION, DIRECT_ADAPTER_REASONING_CONDITION),
    (BASE_REASONING_CONDITION, REASONING_ADAPTER_CONDITION),
    (DIRECT_ADAPTER_REASONING_CONDITION, REASONING_ADAPTER_CONDITION),
    (DIRECT_REFERENCE_CONDITION, BASE_REASONING_CONDITION),
    (DIRECT_REFERENCE_CONDITION, DIRECT_ADAPTER_REASONING_CONDITION),
    (DIRECT_REFERENCE_CONDITION, REASONING_ADAPTER_CONDITION),
]


def make_reasoning_prompt(question: str) -> str:
    return REASONING_PROMPT_TEMPLATE.format(question=question)


def adapter_dir_for(condition: str) -> Optional[Path]:
    if condition in {BASE_REASONING_CONDITION, BASE_REASONING_CONDITION_512}:
        return None
    if condition in {DIRECT_ADAPTER_REASONING_CONDITION, DIRECT_ADAPTER_REASONING_CONDITION_512}:
        return DIRECT_ORIGINAL_ONLY_ADAPTER_DIR
    if condition in {REASONING_ADAPTER_CONDITION, REASONING_ADAPTER_CONDITION_512}:
        return REASONING_ADAPTER_DIR
    raise KeyError(f"Unknown reasoning evaluation condition: {condition}")


def condition_description() -> Dict[str, str]:
    return {
        BASE_REASONING_CONDITION: "Untuned Qwen3-4B-Base with the reasoning prompt and 256-token generation.",
        DIRECT_ADAPTER_REASONING_CONDITION: "Existing direct-answer original_only LoRA adapter evaluated with the reasoning prompt.",
        REASONING_ADAPTER_CONDITION: "New original-only reasoning-supervised LoRA adapter evaluated with the reasoning prompt.",
        BASE_REASONING_CONDITION_512: "Untuned Qwen3-4B-Base with the reasoning prompt and 512-token generation.",
        DIRECT_ADAPTER_REASONING_CONDITION_512: "Existing direct-answer original_only LoRA adapter evaluated with the reasoning prompt and 512-token generation.",
        REASONING_ADAPTER_CONDITION_512: "Reasoning-supervised LoRA adapter evaluated with the reasoning prompt and 512-token generation.",
        DIRECT_REFERENCE_CONDITION: "Existing direct-answer original_only validation result, included by analysis without rerunning.",
    }
