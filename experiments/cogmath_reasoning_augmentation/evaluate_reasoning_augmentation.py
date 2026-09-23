#!/usr/bin/env python3
"""Evaluate reasoning-augmentation adapters with the 512-token reasoning protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from reasoning_aug_utils import EVALUATION_CONDITIONS, MODEL_NAME, SUPERVISION_DIR, VALIDATION_EVAL_FILE

sys.path.insert(0, str(SUPERVISION_DIR))
from evaluate_reasoning import evaluate  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a reasoning-augmentation condition.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--condition", required=True, choices=EVALUATION_CONDITIONS)
    parser.add_argument("--eval_file", type=Path, default=VALIDATION_EVAL_FILE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup_generations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_existing_output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()

