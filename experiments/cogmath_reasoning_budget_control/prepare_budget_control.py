#!/usr/bin/env python3
"""Write resolved metadata for the reasoning budget-control experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from budget_control_utils import (
    BUDGET_CONDITIONS,
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
    SEED,
    TRAIN_MAX_LENGTH,
    WARMUP_STEPS,
    budget_dataset_report,
    write_json,
)


def write_outputs() -> None:
    report = budget_dataset_report()
    write_json(Path(__file__).resolve().parent / "dataset_report.json", report)
    write_json(
        Path(__file__).resolve().parent / "resolved_configs" / "budget_control.json",
        {
            "experiment": "cogmath_reasoning_budget_control_qwen3_4b",
            "model_name": MODEL_NAME,
            "output_root": OUTPUT_ROOT,
            "conditions": BUDGET_CONDITIONS,
            "max_steps": MAX_STEPS,
            "stopping_criterion": "max_steps",
            "training_max_length": TRAIN_MAX_LENGTH,
            "validation_max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "bf16": True,
            "gradient_checkpointing": True,
            "use_cache": False,
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
            },
            "methodological_note": "Optimizer updates are controlled exactly with max_steps=232. Example presentations are not assumed to equal max_steps * gradient_accumulation_steps because Transformers can perform optimizer steps on incomplete accumulation groups at finite-dataset epoch boundaries.",
        },
    )


def main(argv: Optional[List[str]] = None) -> None:
    _ = argv
    write_outputs()
    print(json.dumps({"wrote": ["dataset_report.json", "resolved_configs/budget_control.json"], "max_steps": MAX_STEPS, "max_length": TRAIN_MAX_LENGTH}, indent=2))


if __name__ == "__main__":
    main()
