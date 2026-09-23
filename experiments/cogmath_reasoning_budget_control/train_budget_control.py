#!/usr/bin/env python3
"""Train one Qwen3-4B reasoning LoRA adapter with a fixed optimizer-step budget."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from budget_control_utils import (
    BUDGET_CONDITIONS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_STEPS,
    MODEL_NAME,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SEED,
    TRAIN_MAX_LENGTH,
    WARMUP_STEPS,
    composition,
    source_training_path,
    trainer_epoch_group_example_counts,
    write_json,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one budget-control reasoning adapter.")
    parser.add_argument("--condition", required=True, choices=BUDGET_CONDITIONS)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=TRAIN_MAX_LENGTH)
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
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
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    seen = set()
    for index, row in enumerate(rows):
        record_id = row["record_id"]
        if record_id in seen:
            raise ValueError(f"Duplicate record_id in source data: {record_id}")
        seen.add(record_id)
        target = row.get("training_target", "")
        if not target.startswith("Reasoning:\n"):
            raise ValueError(f"Row {index} has unexpected reasoning target prefix.")
        if "\n\nFinal answer: " not in target:
            raise ValueError(f"Row {index} lacks final-answer marker in training target.")
        if not target.endswith(f"Final answer: {row.get('final_answer')}"):
            raise ValueError(f"Row {index} has malformed final-answer target.")


def load_rows(path: Path, seed: int, smoke_samples: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    validate_rows(rows)
    if smoke_samples:
        rows = rows[:smoke_samples]
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return rows


def tokenize(tokenizer: Any, prompt: str, target: str, max_length: int, record_id: str) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        raise ValueError(f"{record_id} needs {len(input_ids)} tokens, exceeding max_length={max_length}; no silent truncation.")
    labels = [-100] * len(prompt_ids) + target_ids[:]
    attention_mask = [1] * len(input_ids)
    pad_len = max_length - len(input_ids)
    if pad_len:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class Dataset:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.observed_example_presentations = 0
        self.observed_nonpadding_tokens = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        encoded = tokenize(self.tokenizer, row["prompt"], row["training_target"], self.max_length, row["record_id"])
        self.observed_example_presentations += 1
        self.observed_nonpadding_tokens += sum(encoded["attention_mask"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class LossHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if logs:
            rec = {"step": state.global_step, "epoch": state.epoch}
            rec.update({k: v for k, v in logs.items() if k in {"loss", "learning_rate", "grad_norm"}})
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


def package_versions() -> Dict[str, str]:
    out = {}
    for package in ["torch", "transformers", "peft", "accelerate", "datasets", "tokenizers"]:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not_installed"
    return out


def parameter_report(model: Any) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_percentage": 100.0 * trainable / total if total else None}


def ensure_new(path: Path, allow: bool) -> None:
    if not allow and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output_dir={path}.")


def token_exposure(rows: List[Dict[str, Any]], tokenizer: Any, estimated_example_presentations: int) -> Dict[str, Any]:
    eos = tokenizer.eos_token or ""
    lengths = []
    for row in rows:
        length = (
            len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            + len(tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"])
        )
        lengths.append(length)
    mean_length = sum(lengths) / len(lengths) if lengths else None
    return {
        "measured_dataset_nonpadding_tokens": sum(lengths),
        "mean_nonpadding_tokens_per_example": mean_length,
        "max_nonpadding_tokens": max(lengths) if lengths else None,
        "estimated_nonpadding_tokens_processed": mean_length * estimated_example_presentations if mean_length is not None else None,
        "token_exposure_note": "Pre-run token exposure is estimated from mean tokenized sequence length times estimated example presentations. Runtime training_exposure.json records observed non-padding token presentations when dataloader_num_workers=0.",
    }


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    class Bridge(TrainerCallback):
        def __init__(self, cb: LossHistory) -> None:
            self.cb = cb

        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            self.cb.on_log(args, state, control, logs, **kwargs)

    ensure_new(args.output_dir, args.allow_existing_output)
    train_file = args.train_file or source_training_path(args.condition)
    rows = load_rows(train_file, args.seed, args.smoke_samples)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    presentation_estimate = trainer_epoch_group_example_counts(
        len(rows),
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
    )
    exposure = {
        **composition(rows),
        "max_steps": args.max_steps,
        **presentation_estimate,
        **token_exposure(rows, tokenizer, int(presentation_estimate["estimated_example_presentations"])),
    }
    write_json(args.output_dir / "resolved_config.json", {
        "experiment": "cogmath_reasoning_budget_control_qwen3_4b",
        "condition": args.condition,
        "model_name": args.model_name,
        "train_file": str(train_file),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "smoke_samples": args.smoke_samples,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_cache": False,
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES},
        "training_exposure_planned": exposure,
    })
    write_json(args.output_dir / "training_ids.json", {"row_count": len(rows), "record_ids": [row["record_id"] for row in rows]})
    write_json(args.output_dir / "package_versions.json", package_versions())

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=LORA_TARGET_MODULES))
    params = parameter_report(model)
    write_json(args.output_dir / "parameter_report.json", params)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps if args.save_steps else 500,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    train_dataset = Dataset(rows, tokenizer, args.max_length)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, callbacks=[Bridge(LossHistory(args.output_dir / "loss_history.jsonl"))])
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = dict(result.metrics)
    metrics["actual_completed_optimizer_steps"] = trainer.state.global_step
    metrics["final_training_loss"] = metrics.get("train_loss")
    observed_examples = train_dataset.observed_example_presentations
    final_exposure = {
        **exposure,
        "actual_completed_optimizer_steps": trainer.state.global_step,
        "observed_example_presentations": observed_examples,
        "examples_presented": observed_examples,
        "example_presentation_count_source": "observed_dataset_getitem_count_with_dataloader_num_workers_0",
        "observed_effective_epochs": observed_examples / len(rows) if rows else None,
        "effective_epochs": observed_examples / len(rows) if rows else None,
        "observed_nonpadding_tokens_processed": train_dataset.observed_nonpadding_tokens,
        "estimated_vs_observed_example_difference": observed_examples - int(exposure["estimated_example_presentations"]),
        "trainable_parameter_count": params["trainable_parameters"],
        "final_training_loss": metrics.get("final_training_loss"),
    }
    write_json(args.output_dir / "training_metrics.json", metrics)
    write_json(args.output_dir / "training_exposure.json", final_exposure)


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
