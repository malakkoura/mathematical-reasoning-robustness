#!/usr/bin/env python3
"""Write resolved metadata for the LoRA-placement experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from placement_utils import (
    ALL_ANALYSIS_CONDITIONS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    MODEL_NAME,
    NEW_TRAINING_CONDITIONS,
    OUTPUT_ROOT,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PLACEMENTS,
    SEED,
    TRAIN_MAX_LENGTH,
    WARMUP_STEPS,
    dataset_report,
    write_json,
)


def main(argv: Optional[List[str]] = None) -> None:
    _ = argv
    experiment_dir = Path(__file__).resolve().parent
    report = dataset_report()
    write_json(experiment_dir / "dataset_report.json", report)
    write_json(
        experiment_dir / "resolved_configs" / "lora_placement.json",
        {
            "experiment": "cogmath_reasoning_lora_placement_qwen3_4b",
            "model_name": MODEL_NAME,
            "output_root": OUTPUT_ROOT,
            "new_training_conditions": NEW_TRAINING_CONDITIONS,
            "analysis_conditions": ALL_ANALYSIS_CONDITIONS,
            "placements": PLACEMENTS,
            "max_steps": MAX_STEPS,
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
            "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
            "all_linear_reference": "Reused from completed budget-control budget_original_only and budget_original_plus_dim2 outputs.",
        },
    )
    print(json.dumps({"wrote": ["dataset_report.json", "resolved_configs/lora_placement.json"], "new_training_conditions": len(NEW_TRAINING_CONDITIONS)}, indent=2))


if __name__ == "__main__":
    main()
