#!/usr/bin/env python3
"""Train one reasoning-augmentation LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning_aug_utils import LORA_TARGET_MODULES, MODEL_NAME, TRAINING_CONDITIONS, TRAIN_MAX_LENGTH, reasoning_training_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen3-4B reasoning-augmentation LoRA adapter.")
    parser.add_argument("--condition", required=True, choices=TRAINING_CONDITIONS)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=TRAIN_MAX_LENGTH)
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--longest_samples", type=int, default=0)
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
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_rows(rows: List[Dict[str, Any]], condition: str) -> None:
    for index, row in enumerate(rows):
        if row.get("condition") != condition:
            raise ValueError(f"Row {index} has condition={row.get('condition')!r}, expected {condition!r}.")
        if "####" in row.get("training_target", ""):
            raise ValueError(f"Row {index} still contains ####.")
        if not row.get("training_target", "").endswith(f"Final answer: {row.get('final_answer')}"):
            raise ValueError(f"Row {index} has malformed training target.")


def load_rows(path: Path, condition: str, seed: int, smoke_samples: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    validate_rows(rows, condition)
    if smoke_samples:
        rows = rows[:smoke_samples]
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return rows


def select_longest_rows(rows: List[Dict[str, Any]], tokenizer: Any, count: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return rows
    eos = tokenizer.eos_token or ""
    ranked = []
    for row in rows:
        length = (
            len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            + len(tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"])
        )
        ranked.append((length, row["record_id"], row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in ranked[:count]]


def sequence_length_report(rows: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, Any]:
    eos = tokenizer.eos_token or ""
    lengths = []
    for row in rows:
        length = (
            len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            + len(tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"])
        )
        lengths.append({"record_id": row["record_id"], "source_record_id": row.get("source_record_id"), "base_id": row.get("base_id"), "dimension": row.get("dimension"), "sequence_length": length})
    return {
        "row_count": len(lengths),
        "max_sequence_length": max((item["sequence_length"] for item in lengths), default=0),
        "selected_lengths_descending": sorted(lengths, key=lambda item: (-item["sequence_length"], item["record_id"])),
    }


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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        encoded = tokenize(self.tokenizer, row["prompt"], row["training_target"], self.max_length, row["record_id"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class LossHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if logs and "loss" in logs:
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
    train_file = args.train_file or reasoning_training_path(args.condition)
    rows = load_rows(train_file, args.condition, args.seed, args.smoke_samples)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows = select_longest_rows(rows, tokenizer, args.longest_samples)
    seq_report = sequence_length_report(rows, tokenizer)
    write_json(args.output_dir / "resolved_config.json", {
        "experiment": "cogmath_reasoning_augmentation_qwen3_4b",
        "condition": args.condition,
        "model_name": args.model_name,
        "train_file": str(train_file),
        "row_count": len(rows),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "smoke_samples": args.smoke_samples,
        "longest_samples": args.longest_samples,
        "selected_sequence_lengths": seq_report,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_cache": False,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": LORA_TARGET_MODULES},
    })
    write_json(args.output_dir / "training_ids.json", {"row_count": len(rows), "record_ids": [row["record_id"] for row in rows], "selected_sequence_lengths": seq_report})
    write_json(args.output_dir / "package_versions.json", package_versions())
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=LORA_TARGET_MODULES))
    write_json(args.output_dir / "parameter_report.json", parameter_report(model))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps if args.save_steps else 500,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset(rows, tokenizer, args.max_length), callbacks=[Bridge(LossHistory(args.output_dir / "loss_history.jsonl"))])
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    write_json(args.output_dir / "training_metrics.json", result.metrics)


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
