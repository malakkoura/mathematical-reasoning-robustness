#!/usr/bin/env python3
"""LoRA training wrapper for the unique-full Qwen3-4B augmentation setup."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from train_lora import train
from unique_full_utils import MODEL_NAME, UNIQUE_FULL_TRAINING_CONDITIONS


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a unique-full Qwen3-4B LoRA adapter.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--condition", required=True, choices=sorted(UNIQUE_FULL_TRAINING_CONDITIONS))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
