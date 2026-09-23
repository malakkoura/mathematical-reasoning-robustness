#!/usr/bin/env python3
"""Train one reasoning LoRA adapter with causal or prompt-bidirectional attention."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from attention_control_utils import (
    ALL_CONDITIONS,
    ATTENTION_TYPE_BY_CONDITION,
    BF16,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
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
    row_order_sha256,
    source_training_path_for_condition,
    trainer_epoch_group_example_counts,
    write_json,
)
from attention_masks import build_model_attention_mask


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one reasoning attention-control adapter.")
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--attention_type", choices=["causal", "prompt_bidirectional"], default=None)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=TRAIN_MAX_LENGTH)
    parser.add_argument("--smoke_samples", type=int, default=0)
    parser.add_argument("--longest_samples", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=BF16)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=GRADIENT_CHECKPOINTING)
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
            raise ValueError(f"Duplicate record_id in training data: {record_id}")
        seen.add(record_id)
        target = row.get("training_target", "")
        if not target.startswith("Reasoning:\n"):
            raise ValueError(f"Row {index} has unexpected reasoning target prefix.")
        if "\n\nFinal answer: " not in target:
            raise ValueError(f"Row {index} lacks final-answer marker.")
        if not target.endswith(f"Final answer: {row.get('final_answer')}"):
            raise ValueError(f"Row {index} has malformed final answer.")
        if "<END_OF_INPUT>" in row.get("prompt", ""):
            raise ValueError(f"Row {index} unexpectedly contains a visible boundary marker.")


def load_rows(path: Path, seed: int, smoke_samples: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    validate_rows(rows)
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


def tokenize(tokenizer: Any, prompt: str, target: str, max_length: int, record_id: str, include_prompt_length: bool = False) -> Dict[str, List[int] | int]:
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
    encoded: Dict[str, List[int] | int] = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    if include_prompt_length:
        encoded["prompt_lengths"] = len(prompt_ids)
    return encoded


def sequence_length_report(rows: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, Any]:
    eos = tokenizer.eos_token or ""
    lengths = []
    for row in rows:
        length = (
            len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            + len(tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"])
        )
        lengths.append({"record_id": row["record_id"], "base_id": row["base_id"], "dimension": row["dimension"], "sequence_length": length})
    return {
        "row_count": len(lengths),
        "max_sequence_length": max((item["sequence_length"] for item in lengths), default=0),
        "selected_lengths_descending": sorted(lengths, key=lambda item: (-item["sequence_length"], item["record_id"]))[:20],
    }


class Dataset:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int, include_prompt_lengths: bool = False) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_prompt_lengths = include_prompt_lengths
        self.observed_example_presentations = 0
        self.observed_nonpadding_tokens = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        encoded = tokenize(self.tokenizer, row["prompt"], row["training_target"], self.max_length, row["record_id"], include_prompt_length=self.include_prompt_lengths)
        self.observed_example_presentations += 1
        self.observed_nonpadding_tokens += sum(encoded["attention_mask"])  # type: ignore[arg-type]
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class LossHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if logs:
            rec = {"step": state.global_step, "epoch": state.epoch}
            rec.update({key: value for key, value in logs.items() if key in {"loss", "learning_rate", "grad_norm"}})
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
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_percentage": 100.0 * trainable / total if total else None}


def ensure_new(path: Path, allow_existing_output: bool) -> None:
    if not allow_existing_output and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output_dir={path}.")


def token_exposure(rows: List[Dict[str, Any]], tokenizer: Any, estimated_example_presentations: int) -> Dict[str, Any]:
    eos = tokenizer.eos_token or ""
    lengths = [
        len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        + len(tokenizer(row["training_target"] + eos, add_special_tokens=False)["input_ids"])
        for row in rows
    ]
    mean_length = sum(lengths) / len(lengths) if lengths else None
    return {
        "measured_dataset_nonpadding_tokens": sum(lengths),
        "mean_nonpadding_tokens_per_example": mean_length,
        "max_nonpadding_tokens": max(lengths) if lengths else None,
        "estimated_nonpadding_tokens_processed": mean_length * estimated_example_presentations if mean_length is not None else None,
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

    class MaskedTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any):
            prompt_lengths = inputs.pop("prompt_lengths")
            labels = inputs.pop("labels")
            base_attention_mask = inputs.pop("attention_mask")
            dtype = next(model.parameters()).dtype
            model_mask = build_model_attention_mask(prompt_lengths, base_attention_mask, attention_type, dtype=dtype, device=base_attention_mask.device)
            outputs = model(**inputs, attention_mask=model_mask, labels=labels, use_cache=False)
            if outputs.loss is None or not torch.isfinite(outputs.loss).item():
                raise RuntimeError(f"Non-finite loss for {args.condition}.")
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    attention_type = args.attention_type or ATTENTION_TYPE_BY_CONDITION[args.condition]
    if attention_type != ATTENTION_TYPE_BY_CONDITION[args.condition]:
        raise ValueError(f"{args.condition} requires attention_type={ATTENTION_TYPE_BY_CONDITION[args.condition]}, got {attention_type}.")

    ensure_new(args.output_dir, args.allow_existing_output)
    train_file = args.train_file or source_training_path_for_condition(args.condition)
    rows = load_rows(train_file, args.seed, args.smoke_samples)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows = select_longest_rows(rows, tokenizer, args.longest_samples)
    seq_report = sequence_length_report(rows, tokenizer)

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
    write_json(
        args.output_dir / "resolved_config.json",
        {
            "experiment": "cogmath_reasoning_attention_control_qwen3_4b",
            "condition": args.condition,
            "attention_type": attention_type,
            "model_name": args.model_name,
            "train_file": str(train_file),
            "row_count": len(rows),
            "row_order_sha256": row_order_sha256(row["record_id"] for row in rows),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "smoke_samples": args.smoke_samples,
            "longest_samples": args.longest_samples,
            "selected_sequence_lengths": seq_report,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_steps": args.warmup_steps,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_cache": False,
            "completion_only_loss": True,
            "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES},
            "masking": "standard causal mask" if attention_type == "causal" else "additive prompt-bidirectional 4D mask under eager attention",
            "trainer_path": "native_transformers_trainer" if attention_type == "causal" else "custom_prompt_bidirectional_compute_loss",
            "training_exposure_planned": exposure,
        },
    )
    write_json(args.output_dir / "training_ids.json", {"row_count": len(rows), "record_ids": [row["record_id"] for row in rows]})
    write_json(args.output_dir / "package_versions.json", package_versions())

    model_kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "cache_dir": str(args.cache_dir) if args.cache_dir else None}
    if attention_type == "prompt_bidirectional":
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
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
    train_dataset = Dataset(rows, tokenizer, args.max_length, include_prompt_lengths=(attention_type == "prompt_bidirectional"))
    trainer_cls = Trainer if attention_type == "causal" else MaskedTrainer
    trainer = trainer_cls(model=model, args=training_args, train_dataset=train_dataset, callbacks=[Bridge(LossHistory(args.output_dir / "loss_history.jsonl"))])
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = dict(result.metrics)
    metrics["actual_completed_optimizer_steps"] = trainer.state.global_step
    final_exposure = {
        **exposure,
        "actual_completed_optimizer_steps": trainer.state.global_step,
        "observed_example_presentations": train_dataset.observed_example_presentations,
        "examples_presented": train_dataset.observed_example_presentations,
        "example_presentation_count_source": "observed_dataset_getitem_count_with_dataloader_num_workers_0",
        "observed_effective_epochs": train_dataset.observed_example_presentations / len(rows) if rows else None,
        "effective_epochs": train_dataset.observed_example_presentations / len(rows) if rows else None,
        "observed_nonpadding_tokens_processed": train_dataset.observed_nonpadding_tokens,
        "trainable_parameter_count": params["trainable_parameters"],
        "final_training_loss": metrics.get("train_loss"),
    }
    write_json(args.output_dir / "training_metrics.json", metrics)
    write_json(args.output_dir / "training_exposure.json", final_exposure)


def main(argv: Optional[List[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
