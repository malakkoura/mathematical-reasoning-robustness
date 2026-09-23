#!/usr/bin/env python3
"""Evaluation wrapper for the unique-full Qwen3-4B augmentation setup."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from evaluate_model import evaluate
from unique_full_utils import MODEL_NAME, UNIQUE_FULL_EVALUATION_CONDITIONS


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-4B unique-full augmentation conditions.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--condition", required=True, choices=UNIQUE_FULL_EVALUATION_CONDITIONS)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--warmup_generations", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
