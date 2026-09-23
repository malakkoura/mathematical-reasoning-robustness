#!/usr/bin/env python3
"""Freeze one first-pass validation trace per record for deliberation evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from deliberation_utils import (
    DEFAULT_FIRST_PASS_PREDICTIONS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VALIDATION_FILE,
    build_first_pass_rows,
    summarize_first_pass,
    write_json,
    write_jsonl,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze first-pass validation traces for deliberation.")
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--first_pass_predictions", type=Path, default=DEFAULT_FIRST_PASS_PREDICTIONS)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_ROOT / "first_pass_traces" / "first_pass_traces.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected_count", type=int, default=990)
    parser.add_argument("--allow_existing_output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.output_file.exists() and not args.allow_existing_output:
        raise FileExistsError(f"Refusing to overwrite existing {args.output_file}.")
    rows = build_first_pass_rows(args.eval_file, args.first_pass_predictions, limit=args.limit)
    if not rows:
        raise ValueError("No first-pass rows were frozen.")
    if not args.limit and args.expected_count and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} frozen validation traces, got {len(rows)}.")
    write_jsonl(args.output_file, rows)
    manifest = summarize_first_pass(rows, args.first_pass_predictions)
    manifest.update(
        {
            "eval_file": str(args.eval_file),
            "output_file": str(args.output_file),
            "test_set_used": False,
            "trace_policy": "one completed first-pass validation trace per record; no held-out test records read",
        }
    )
    write_json(args.output_file.with_suffix(".manifest.json"), manifest)
    print(f"Wrote {len(rows)} frozen first-pass traces to {args.output_file}")


if __name__ == "__main__":
    main()
