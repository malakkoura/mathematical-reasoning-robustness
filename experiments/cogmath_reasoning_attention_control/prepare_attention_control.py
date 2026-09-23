#!/usr/bin/env python3
"""Materialize metadata for the reasoning attention-structure control."""

from __future__ import annotations

from attention_control_utils import (
    ALL_CONDITIONS,
    ATTENTION_TYPES,
    BF16,
    EXPERIMENT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    MODEL_NAME,
    OUTPUT_ROOT,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PROMPT_BIDIR_TRAINING_CONDITIONS,
    SEED,
    TRAINING_DATASETS,
    TRAIN_MAX_LENGTH,
    USE_CACHE,
    WARMUP_STEPS,
    dataset_report,
    write_json,
)


def main() -> None:
    report = dataset_report()
    config = {
        "experiment": "cogmath_reasoning_attention_control_qwen3_4b",
        "model_name": MODEL_NAME,
        "output_root": OUTPUT_ROOT,
        "conditions": ALL_CONDITIONS,
        "training_conditions_requiring_new_training": PROMPT_BIDIR_TRAINING_CONDITIONS,
        "training_datasets": TRAINING_DATASETS,
        "attention_types": ATTENTION_TYPES,
        "stopping_criterion": "max_steps",
        "max_steps": MAX_STEPS,
        "training_max_length": TRAIN_MAX_LENGTH,
        "validation_max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "logging_steps": LOGGING_STEPS,
        "bf16": BF16,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "use_cache": USE_CACHE,
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
        },
        "prompt_bidirectional_mask": {
            "model_visible_boundary_marker": False,
            "boundary_source": "token-level prompt_length",
            "prompt_tokens": "bidirectional among prompt/input tokens only",
            "answer_tokens": "causal over generated answer tokens while attending to the full prompt",
            "kv_cache_training": False,
            "kv_cache_generation": False,
        },
        "training_gate": {
            "causal_path": "native Transformers Trainer and native Qwen causal attention, matching budget-control training",
            "prompt_bidirectional_path": "custom compute_loss only for additive 4D prompt-bidirectional mask under eager attention",
            "full_training_requires": [
                "causal budget-reference compatibility",
                "runtime causal-equivalence diagnostic",
                "abstract mask tests",
                "model-level future-target leakage diagnostic",
                "prompt-bidirectional eager backend confirmation",
            ],
        },
    }
    write_json(EXPERIMENT_DIR / "dataset_report.json", report)
    write_json(EXPERIMENT_DIR / "resolved_configs" / "attention_control.json", config)
    print("Wrote reasoning attention-control metadata.")


if __name__ == "__main__":
    main()
