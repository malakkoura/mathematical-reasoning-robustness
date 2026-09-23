#!/usr/bin/env python3
"""GPU smoke test for cached prompt-bidirectional inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from attention_control_utils import MAX_NEW_TOKENS, MODEL_NAME, OUTPUT_ROOT, VALIDATION_EVAL_FILE, write_json
from evaluate_attention_control import load_model_and_tokenizer, prompt_bidir_generate_one


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test cached prompt-bidirectional inference memory.")
    parser.add_argument("--condition", default="prompt_bidir_original_only", choices=["prompt_bidir_original_only", "prompt_bidir_original_plus_dim2", "prompt_bidir_original_plus_all"])
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--adapter_dir", type=Path, default=Path(OUTPUT_ROOT) / "adapters" / "prompt_bidir_original_only")
    parser.add_argument("--eval_file", type=Path, default=VALIDATION_EVAL_FILE)
    parser.add_argument("--output_path", type=Path, default=Path(OUTPUT_ROOT) / "diagnostics" / "prompt_bidir_cached_inference_smoke.json")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--smoke_new_tokens", type=int, default=96)
    parser.add_argument("--full_eval_max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--stop_on_eos", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.full_eval_max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("This smoke must preserve the full evaluation max_new_tokens=512 configuration.")

    rows = read_jsonl(args.eval_file)[: args.examples]
    for row in rows:
        row["condition"] = args.condition

    model_args = argparse.Namespace(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    model, tokenizer, torch_module = load_model_and_tokenizer(model_args, "prompt_bidirectional")
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    predictions = [
        prompt_bidir_generate_one(
            model,
            tokenizer,
            torch_module,
            row,
            max_new_tokens=args.smoke_new_tokens,
            stop_on_eos=args.stop_on_eos,
        )
        for row in rows
    ]
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    report = {
        "passed": True,
        "condition": args.condition,
        "model_name": args.model_name,
        "adapter_dir": str(args.adapter_dir),
        "eval_file": str(args.eval_file),
        "test_set_used": False,
        "generation_mode": "prompt-bidirectional prompt prefill once, cached single-query causal decode",
        "full_validation_max_new_tokens_preserved": args.full_eval_max_new_tokens,
        "smoke_new_tokens": args.smoke_new_tokens,
        "stop_on_eos": args.stop_on_eos,
        "examples": [
            {
                "record_id": row["record_id"],
                "generated_tokens": row["generated_tokens"],
                "hit_max_new_tokens": row["hit_max_new_tokens"],
                "latency_seconds": row["latency_seconds"],
                "peak_cuda_memory_allocated_bytes": row["peak_cuda_memory_allocated_bytes"],
                "peak_cuda_memory_reserved_bytes": row["peak_cuda_memory_reserved_bytes"],
            }
            for row in predictions
        ],
        "peak_cuda_memory_allocated_bytes": torch_module.cuda.max_memory_allocated() if torch_module.cuda.is_available() else None,
        "peak_cuda_memory_reserved_bytes": torch_module.cuda.max_memory_reserved() if torch_module.cuda.is_available() else None,
    }
    write_json(args.output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
