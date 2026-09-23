#!/usr/bin/env python3
"""Constants for the Qwen3-4B reasoning LoRA-placement experiment."""

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
REASONING_SUPERVISION_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_supervision"
BUDGET_CONTROL_DIR = REPO_ROOT / "experiments" / "cogmath_reasoning_budget_control"

MODEL_NAME = "Qwen/Qwen3-4B-Base"
OUTPUT_ROOT = "outputs/cogmath_reasoning_lora_placement_qwen3_4b"
BUDGET_OUTPUT_ROOT = "outputs/cogmath_reasoning_budget_control_qwen3_4b"

TRAIN_MAX_LENGTH = 1536
MAX_STEPS = 232
MAX_NEW_TOKENS = 512
SEED = 42
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_STEPS = 20
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

VALIDATION_EVAL_FILE = AUGMENTATION_DIR / "validation_eval.jsonl"
SPLITS_FILE = AUGMENTATION_DIR / "splits.json"
SOURCE_DATA_DIR = REASONING_AUG_DIR / "data"

PLACEMENTS: Dict[str, List[str]] = {
    "q_only": ["q_proj"],
    "qv": ["q_proj", "v_proj"],
    "attention_all": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp_only": ["gate_proj", "up_proj", "down_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

NEW_TRAINING_CONDITIONS: List[str] = [
    "q_only_original_only",
    "q_only_plus_dim2",
    "qv_original_only",
    "qv_plus_dim2",
    "attention_all_original_only",
    "attention_all_plus_dim2",
    "mlp_only_original_only",
    "mlp_only_plus_dim2",
]

ALL_ANALYSIS_CONDITIONS: List[str] = [
    *NEW_TRAINING_CONDITIONS,
    "all_linear_original_only",
    "all_linear_plus_dim2",
]

BUDGET_REFERENCE_CONDITION = {
    "all_linear_original_only": "budget_original_only",
    "all_linear_plus_dim2": "budget_original_plus_dim2",
}

SOURCE_DATASET_BY_VARIANT = {
    "original_only": SOURCE_DATA_DIR / "train_reasoning_original_only.jsonl",
    "plus_dim2": SOURCE_DATA_DIR / "train_reasoning_original_plus_dim2.jsonl",
}

PRIMARY_WITHIN_PLACEMENT_COMPARISONS: List[Tuple[str, str]] = [
    (f"{placement}_original_only", f"{placement}_plus_dim2")
    for placement in PLACEMENTS
]

CROSS_PLACEMENT_PLUS_DIM2_COMPARISONS: List[Tuple[str, str]] = [
    ("q_only_plus_dim2", "qv_plus_dim2"),
    ("qv_plus_dim2", "attention_all_plus_dim2"),
    ("attention_all_plus_dim2", "mlp_only_plus_dim2"),
    ("attention_all_plus_dim2", "all_linear_plus_dim2"),
    ("mlp_only_plus_dim2", "all_linear_plus_dim2"),
    ("qv_plus_dim2", "all_linear_plus_dim2"),
]

EXPECTED_DIMENSION_COUNTS = {
    "original_only": {0: 923},
    "plus_dim2": {0: 923, 2: 923},
}

QWEN3_4B_BASE_CONFIG = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}


def condition_placement(condition: str) -> str:
    if condition.startswith("all_linear_"):
        return "all_linear"
    if condition.startswith("attention_all_"):
        return "attention_all"
    if condition.startswith("mlp_only_"):
        return "mlp_only"
    if condition.startswith("q_only_"):
        return "q_only"
    if condition.startswith("qv_"):
        return "qv"
    raise KeyError(condition)


def condition_variant(condition: str) -> str:
    return "plus_dim2" if condition.endswith("plus_dim2") else "original_only"


def source_training_path(condition: str) -> Path:
    return SOURCE_DATASET_BY_VARIANT[condition_variant(condition)]


def adapter_dir_for(condition: str) -> Path:
    if condition in BUDGET_REFERENCE_CONDITION:
        return Path(BUDGET_OUTPUT_ROOT) / "adapters" / BUDGET_REFERENCE_CONDITION[condition]
    return Path(OUTPUT_ROOT) / "adapters" / condition


def validation_dir_for(condition: str) -> Path:
    if condition in BUDGET_REFERENCE_CONDITION:
        return Path(BUDGET_OUTPUT_ROOT) / "validation_evaluations" / BUDGET_REFERENCE_CONDITION[condition]
    return Path(OUTPUT_ROOT) / "validation_evaluations" / condition


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


def trainer_epoch_group_example_counts(row_count: int) -> Dict[str, Any]:
    microbatches = [1] * row_count
    groups = [sum(microbatches[index:index + GRADIENT_ACCUMULATION_STEPS]) for index in range(0, row_count, GRADIENT_ACCUMULATION_STEPS)]
    full_epochs, remainder_steps = divmod(MAX_STEPS, len(groups))
    examples = full_epochs * sum(groups) + sum(groups[:remainder_steps])
    nominal = MAX_STEPS * GRADIENT_ACCUMULATION_STEPS * PER_DEVICE_TRAIN_BATCH_SIZE
    return {
        "estimated_example_presentations": examples,
        "effective_epochs": examples / row_count,
        "optimizer_steps_per_epoch": len(groups),
        "last_group_examples_per_epoch": groups[-1],
        "nominal_full_accumulation_examples": nominal,
        "shortfall_vs_nominal_full_accumulation": nominal - examples,
    }


def lora_params_for_module(module: str) -> int:
    cfg = QWEN3_4B_BASE_CONFIG
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    q_out = cfg["num_attention_heads"] * head_dim
    kv_out = cfg["num_key_value_heads"] * head_dim
    intermediate = cfg["intermediate_size"]
    dims = {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    in_features, out_features = dims[module]
    return LORA_RANK * (in_features + out_features) * cfg["num_hidden_layers"]


def estimated_lora_trainable_params(modules: List[str]) -> int:
    return sum(lora_params_for_module(module) for module in modules)


def estimated_base_parameter_count() -> int:
    cfg = QWEN3_4B_BASE_CONFIG
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    q_out = cfg["num_attention_heads"] * head_dim
    kv_out = cfg["num_key_value_heads"] * head_dim
    intermediate = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    embeddings = cfg["vocab_size"] * hidden
    attention = hidden * q_out + hidden * kv_out + hidden * kv_out + q_out * hidden
    mlp = hidden * intermediate + hidden * intermediate + intermediate * hidden
    norms = hidden + hidden + head_dim + head_dim
    final_norm = hidden
    return embeddings + layers * (attention + mlp + norms) + final_norm


def parameter_estimates() -> Dict[str, Dict[str, Any]]:
    base = estimated_base_parameter_count()
    out = {}
    for placement, modules in PLACEMENTS.items():
        trainable = estimated_lora_trainable_params(modules)
        out[placement] = {
            "target_modules": modules,
            "estimated_trainable_parameters": trainable,
            "estimated_total_parameters_with_adapter": base + trainable,
            "estimated_trainable_percentage": 100.0 * trainable / (base + trainable),
            "parameter_count_note": "Analytic estimate from Qwen3-4B-Base config and PEFT LoRA A/B matrices at r=8; cluster smoke/training writes exact PEFT parameter_report.json.",
        }
    return out


def dataset_report() -> Dict[str, Any]:
    datasets = {}
    for variant, path in SOURCE_DATASET_BY_VARIANT.items():
        rows = read_jsonl(path)
        datasets[variant] = {
            "source_file": str(path),
            "source_sha256": sha256_file(path),
            **composition(rows),
            **trainer_epoch_group_example_counts(len(rows)),
        }
    return {
        "experiment": "cogmath_reasoning_lora_placement_qwen3_4b",
        "model_name": MODEL_NAME,
        "output_root": OUTPUT_ROOT,
        "budget_reference_output_root": BUDGET_OUTPUT_ROOT,
        "max_steps": MAX_STEPS,
        "training_max_length": TRAIN_MAX_LENGTH,
        "validation_max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "placements": parameter_estimates(),
        "datasets": datasets,
        "new_training_conditions": NEW_TRAINING_CONDITIONS,
        "analysis_conditions": ALL_ANALYSIS_CONDITIONS,
        "note": "Placement and adapter capacity co-vary because r=8 is fixed across target-module sets.",
    }
