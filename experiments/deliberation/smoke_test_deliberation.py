#!/usr/bin/env python3
"""Smoke checks for the deliberation experiment.

By default this runs local, model-free checks over a tiny synthetic prediction
file. On the cluster, pass --run_model_smoke with a real first-pass prediction
file to exercise model loading and the model-level leakage check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from deliberation_utils import DEFAULT_FIRST_PASS_PREDICTIONS, DEFAULT_MODEL_NAME, DEFAULT_VALIDATION_FILE, assert_deliberation_mask_semantics


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test deliberation setup.")
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--first_pass_predictions", type=Path, default=DEFAULT_FIRST_PASS_PREDICTIONS)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--run_model_smoke", action="store_true")
    parser.add_argument("--records", type=int, default=2)
    return parser.parse_args(argv)


def run(cmd: List[str]) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def make_synthetic_predictions(eval_file: Path, output_path: Path, records: int) -> None:
    rows = []
    with eval_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= records:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            trace = f"Synthetic first-pass reasoning for {row['record_id']}.\nFinal answer: {row['final_answer']}"
            pred = {
                "record_id": row["record_id"],
                "condition": "synthetic_first_pass",
                "generated_text": trace,
                "extracted_answer": row["final_answer"],
                "correct": True,
                "generated_token_count": 8,
                "hit_max_new_tokens": False,
            }
            f.write(json.dumps(pred) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    assert_deliberation_mask_semantics(prefix_len=8, generated_len=5, pad_len=2)
    run([sys.executable, SCRIPT_DIR / "test_deliberation_masks.py"])

    with tempfile.TemporaryDirectory(prefix="deliberation_smoke_") as tmp:
        tmp_path = Path(tmp)
        if args.run_model_smoke:
            first_pass_predictions = args.first_pass_predictions
        else:
            first_pass_predictions = tmp_path / "synthetic_first_pass_predictions.jsonl"
            make_synthetic_predictions(args.eval_file, first_pass_predictions, args.records)

        frozen = tmp_path / "first_pass_traces.jsonl"
        run(
            [
                sys.executable,
                SCRIPT_DIR / "freeze_first_pass_traces.py",
                "--eval_file",
                args.eval_file,
                "--first_pass_predictions",
                first_pass_predictions,
                "--output_file",
                frozen,
                "--limit",
                str(args.records),
            ]
        )

        if args.run_model_smoke:
            for condition in ["causal_revision", "bidir_trace_revision"]:
                cmd = [
                    sys.executable,
                    SCRIPT_DIR / "evaluate_revision.py",
                    "--condition",
                    condition,
                    "--model_name",
                    args.model_name,
                    "--first_pass_file",
                    frozen,
                    "--output_dir",
                    tmp_path / condition,
                    "--max_new_tokens",
                    "32",
                    "--limit",
                    str(args.records),
                    "--run_model_leakage_check",
                ]
                if args.adapter_dir:
                    cmd.extend(["--adapter_dir", args.adapter_dir])
                if args.cache_dir:
                    cmd.extend(["--cache_dir", args.cache_dir])
                run(cmd)
        else:
            print("Skipped model smoke; pass --run_model_smoke on the cluster to test model loading/generation.")
    print("deliberation smoke checks passed")


if __name__ == "__main__":
    main()
