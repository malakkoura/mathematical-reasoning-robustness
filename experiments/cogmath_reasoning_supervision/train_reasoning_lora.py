#!/usr/bin/env python3
"""Train the reasoning-supervised original-only Qwen3-4B LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning_utils import (
    LORA_TARGET_MODULES,
    MODEL_NAME,
    REASONING_TRAIN_FILE,
    REASONING_TRAINING_CONDITION,
    TRAIN_MAX_LENGTH,
    make_reasoning_prompt,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reasoning-supervised Qwen3-4B LoRA adapter.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", type=Path, default=REASONING_TRAIN_FILE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=TRAIN_MAX_LENGTH)
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_existing_output", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Training file is empty.")
    for index, row in enumerate(rows):
        if row.get("condition") != REASONING_TRAINING_CONDITION:
            raise ValueError(f"Row {index} has condition={row.get('condition')!r}.")
        if not row.get("question"):
            raise ValueError(f"Row {index} is missing question.")
        if not row.get("training_target", "").endswith(f"Final answer: {row.get('final_answer')}"):
            raise ValueError(f"Row {index} has malformed reasoning training_target.")
        if "####" in row.get("training_target", ""):
            raise ValueError(f"Row {index} still contains GSM8K #### marker.")


def load_training_rows(path: Path, seed: int, smoke_samples: int = 0) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    validate_rows(rows)
    if smoke_samples:
        rows = rows[:smoke_samples]
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    return rows


def tokenize_prompt_completion(tokenizer: Any, prompt: str, completion: str, max_length: int, record_id: str) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids[:]

    if len(input_ids) > max_length:
        raise ValueError(
            f"{record_id} requires {len(input_ids)} tokens with the reasoning target, exceeding max_length={max_length}. "
            "Increase max_length; reasoning targets are never silently truncated."
        )

    attention_mask = [1] * len(input_ids)
    pad_id = tokenizer.pad_token_id
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids.extend([pad_id] * pad_len)
        attention_mask.extend([0] * pad_len)
        labels.extend([-100] * pad_len)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ReasoningDataset:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        prompt = row.get("prompt") or make_reasoning_prompt(row["question"])
        encoded = tokenize_prompt_completion(self.tokenizer, prompt, row["training_target"], self.max_length, row["record_id"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class LossHistoryCallback:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update({key: value for key, value in logs.items() if key in {"loss", "learning_rate", "grad_norm"}})
        if "loss" in record:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def package_versions() -> Dict[str, str]:
    versions = {}
    for package in ["torch", "transformers", "peft", "accelerate", "datasets", "tokenizers"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def parameter_report(model: Any) -> Dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": (100.0 * trainable / total) if total else None,
    }


def resolved_config(args: argparse.Namespace, row_count: int) -> Dict[str, Any]:
    return {
        "experiment": "cogmath_reasoning_supervision_qwen3_4b",
        "model_name": args.model_name,
        "condition": REASONING_TRAINING_CONDITION,
        "train_file": str(args.train_file),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "smoke_samples": args.smoke_samples,
        "row_count": row_count,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_cache": False,
        "warmup_steps": args.warmup_steps,
        "tracking": "disabled",
        "prompt_template": "reasoning_utils.REASONING_PROMPT_TEMPLATE",
        "supervision_field": "training_target",
        "no_silent_truncation": True,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": LORA_TARGET_MODULES,
        },
    }


def ensure_output_is_new(output_dir: Path, allow_existing_output: bool) -> None:
    if allow_existing_output:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output_dir={output_dir}. "
            "Choose a new directory or pass --allow_existing_output explicitly."
        )


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    class _CallbackBridge(TrainerCallback):
        def __init__(self, callback: LossHistoryCallback) -> None:
            self.callback = callback

        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            self.callback.on_log(args, state, control, logs, **kwargs)

    ensure_output_is_new(args.output_dir, args.allow_existing_output)
    set_seed(args.seed)
    rows = load_training_rows(args.train_file, args.seed, args.smoke_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.output_dir / "resolved_config.json", resolved_config(args, len(rows)))
    write_json(args.output_dir / "training_ids.json", {"row_count": len(rows), "record_ids": [row["record_id"] for row in rows]})
    write_json(args.output_dir / "package_versions.json", package_versions())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    write_json(args.output_dir / "parameter_report.json", parameter_report(model))

    dataset = ReasoningDataset(rows, tokenizer, args.max_length)
    save_strategy = "steps" if args.save_steps else "epoch"
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps else 500,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[_CallbackBridge(LossHistoryCallback(args.output_dir / "loss_history.jsonl"))],
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    write_json(args.output_dir / "training_metrics.json", train_result.metrics)


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
