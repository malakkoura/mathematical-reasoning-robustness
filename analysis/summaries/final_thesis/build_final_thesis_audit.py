#!/usr/bin/env python3
"""Build thesis handoff audit files without touching experiment outputs.

This script is intentionally read-only with respect to experiments/, outputs/,
and results/. It writes only under analysis/final_thesis/.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "final_thesis"
PUBLIC = OUT / "public_llms"
TOKEN = OUT / "token_audit"
MANIFESTS = OUT / "manifests"
QUAL = OUT / "qualitative"

CLUSTER_ROOT = Path(".")
CLUSTER_OUTPUTS = CLUSTER_ROOT / "outputs"
MODEL_QWEN4B = "Qwen/Qwen3-4B-Base"


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100*x:.2f}%"


def summarize_training_file(path: Path) -> dict:
    rows = list(read_jsonl(path))
    dims = Counter(str(r.get("dimension")) for r in rows)
    base_ids = {r.get("base_id") for r in rows}
    seq_chars = [len((r.get("prompt") or "") + (r.get("training_target") or "")) for r in rows]
    seq_words = [len(((r.get("prompt") or "") + (r.get("training_target") or "")).split()) for r in rows]
    return {
        "path": str(path.relative_to(REPO)),
        "rows": len(rows),
        "unique_base_ids": len(base_ids),
        "dimension_counts": dict(sorted(dims.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999)),
        "median_prompt_plus_target_chars": median(seq_chars) if seq_chars else None,
        "max_prompt_plus_target_chars": max(seq_chars) if seq_chars else None,
        "median_prompt_plus_target_whitespace_tokens": median(seq_words) if seq_words else None,
        "max_prompt_plus_target_whitespace_tokens": max(seq_words) if seq_words else None,
    }


def build_experiment_status() -> list[dict]:
    validation_rows = count_jsonl(REPO / "experiments" / "cogmath_augmentation" / "validation_eval.jsonl")
    attention_validation_rows = count_jsonl(REPO / "experiments" / "cogmath_attention" / "validation_eval.jsonl")
    rows: list[dict] = []

    def add(**kwargs):
        rows.append(kwargs)

    # Existing local augmentation validation summaries.
    aug_summary_root = REPO / "outputs" / "cogmath_augmentation" / "validation_analysis"
    for condition in [
        "base_untuned",
        "original_only",
        "original_plus_dim1",
        "original_plus_dim2",
        "original_plus_dim4",
        "original_plus_dim6",
        "original_plus_all",
        "original_only_large",
        "original_plus_all_full",
    ]:
        summary = aug_summary_root / condition / "summary_recomputed.json"
        pred = REPO / "outputs" / "cogmath_augmentation" / "validation_evaluations" / condition / "predictions.jsonl"
        add(
            experiment_model=f"augmentation LoRA validation / {condition}",
            dataset_split="CogMath-GSM8K validation",
            seed="42",
            checkpoint_status="not locally verifiable",
            prediction_status="raw predictions present" if pred.exists() else "summary only locally; raw predictions absent",
            eval_record_count=str(read_json(summary)["overall"]["total"]) if summary.exists() else "missing",
            output_path=str((aug_summary_root / condition).relative_to(REPO)),
            status="complete-summary-only" if summary.exists() else "missing",
            exact_reason="summary_recomputed.json exists, but adapter checkpoints/raw prediction files are not present locally" if summary.exists() else "no local summary",
        )

    # Older attention local validation outputs.
    for root, model_label, expected_rows in [
        (REPO / "outputs" / "cogmath_attention" / "evaluations" / "validation", "attention LoRA Qwen3-1.7B", attention_validation_rows),
        (REPO / "outputs" / "cogmath_attention_full_ft" / "evaluations" / "validation", "attention full-parameter Qwen3-0.6B", attention_validation_rows),
    ]:
        for condition in ["causal", "prompt_bidirectional"]:
            pred = root / condition / "predictions.jsonl"
            summary = root / condition / "summary.json"
            actual = count_jsonl(pred)
            add(
                experiment_model=f"{model_label} / {condition}",
                dataset_split="CogMath attention validation",
                seed="42",
                checkpoint_status="not locally verifiable",
                prediction_status="predictions.jsonl readable" if actual is not None else "missing",
                eval_record_count=f"{actual}/{expected_rows}" if actual is not None else f"missing/{expected_rows}",
                output_path=str((root / condition).relative_to(REPO)),
                status="complete-local-validation" if actual == expected_rows and summary.exists() else "incomplete-or-unverified",
                exact_reason="local validation predictions and summary match expected row count" if actual == expected_rows and summary.exists() else "missing summary/predictions or row-count mismatch",
            )

    # Qwen3-4B reasoning experiments expected on cluster.
    expected_groups = [
        ("reasoning supervision", "cogmath_reasoning_supervision_qwen3_4b", ["reasoning_original_only"], "validation_evaluations", ["base_reasoning_prompt", "direct_adapter_reasoning_prompt", "reasoning_adapter_reasoning_prompt"]),
        ("reasoning supervision 512", "cogmath_reasoning_supervision_qwen3_4b", ["reasoning_original_only"], "validation_evaluations_512", ["base_reasoning_prompt_512", "direct_adapter_reasoning_prompt_512", "reasoning_adapter_reasoning_prompt_512"]),
        ("reasoning augmentation", "cogmath_reasoning_augmentation_qwen3_4b", ["reasoning_original_plus_dim1", "reasoning_original_plus_dim2", "reasoning_original_plus_dim4", "reasoning_original_plus_dim6", "reasoning_original_plus_dim2_dim4", "reasoning_original_plus_all"], "validation_evaluations", ["reasoning_original_only", "reasoning_original_plus_dim1", "reasoning_original_plus_dim2", "reasoning_original_plus_dim4", "reasoning_original_plus_dim6", "reasoning_original_plus_dim2_dim4", "reasoning_original_plus_all"]),
        ("budget control fixed-step", "cogmath_reasoning_budget_control_qwen3_4b", ["budget_original_only", "budget_original_plus_dim2", "budget_original_plus_dim4", "budget_original_plus_dim2_dim4", "budget_original_plus_all"], "validation_evaluations", ["budget_original_only", "budget_original_plus_dim2", "budget_original_plus_dim4", "budget_original_plus_dim2_dim4", "budget_original_plus_all"]),
        ("attention control fixed-step", "cogmath_reasoning_attention_control_qwen3_4b", ["prompt_bidir_original_only", "prompt_bidir_original_plus_dim2", "prompt_bidir_original_plus_all"], "validation_evaluations", ["prompt_bidir_original_only", "prompt_bidir_original_plus_dim2", "prompt_bidir_original_plus_all"]),
    ]
    for label, root_name, train_conditions, eval_dir, eval_conditions in expected_groups:
        root = CLUSTER_OUTPUTS / root_name
        for condition in sorted(set(train_conditions) | set(eval_conditions)):
            adapter = root / "adapters" / condition
            pred = root / eval_dir / condition / "predictions.jsonl"
            local_pred = REPO / "outputs" / root_name / eval_dir / condition / "predictions.jsonl"
            actual = count_jsonl(local_pred)
            add(
                experiment_model=f"{label} / {condition} / {MODEL_QWEN4B}",
                dataset_split="CogMath-GSM8K validation",
                seed="42",
                checkpoint_status="present on cluster mount" if adapter.exists() else "not present locally; cluster not mounted",
                prediction_status="local prediction copy readable" if actual is not None else "not present locally; cluster not mounted",
                eval_record_count=f"{actual}/990" if actual is not None else "unverified/990",
                output_path=str(pred.parent),
                status="complete-local-copy" if actual == 990 else "incomplete-local-audit",
                exact_reason="local prediction copy has 990 rows" if actual == 990 else "no readable local validation prediction copy; must verify on Imperial or rsync outputs back",
            )

    return rows


def write_experiment_status(rows: list[dict]) -> None:
    path = OUT / "experiment_status.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "experiment_model",
        "dataset_split",
        "seed",
        "checkpoint_status",
        "prediction_status",
        "eval_record_count",
        "output_path",
        "status",
        "exact_reason",
    ]
    with path.open("w") as f:
        f.write("# Experiment Status Audit\n\n")
        f.write("Held-out test files were not opened or summarized in this audit.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            vals = [str(row.get(h, "")).replace("|", "\\|").replace("\n", " ") for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")


def build_public_llm_audit() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary_path in sorted((REPO / "results").glob("*_summary_*.json")):
        if summary_path.name.endswith("_summary_all_models.json"):
            continue
        if any(marker in summary_path.name for marker in ["mock_check", "judge_update_check", "smoke_test"]):
            continue
        try:
            data = read_json(summary_path)
        except Exception:
            continue
        if not isinstance(data, dict) or "model" not in data or "rows" not in data:
            continue
        if str(data.get("model", "")).startswith("model_"):
            continue
        pred_path = summary_path.with_name(summary_path.name.replace("_summary_", "_predictions_").replace(".json", ".jsonl"))
        actual_rows = count_jsonl(pred_path)
        first = None
        if pred_path.exists():
            try:
                first = next(read_jsonl(pred_path))
            except StopIteration:
                first = None
        model = data.get("model", "")
        rows.append(
            {
                "run": summary_path.name.split("_summary_")[0],
                "provider": data.get("provider"),
                "model": model,
                "model_category": data.get("model_category"),
                "rows_expected_from_summary": data.get("rows"),
                "prediction_rows_actual": actual_rows,
                "counts_match": actual_rows == data.get("rows"),
                "accuracy": data.get("accuracy"),
                "correct": data.get("correct"),
                "errors": data.get("errors"),
                "model_errors": data.get("model_errors"),
                "judge_errors": data.get("judge_errors"),
                "missing_extracted_answer": data.get("extracted_answer_missing"),
                "source_datasets": ";".join(sorted((data.get("by_source_dataset") or {}).keys())),
                "dimensions": ";".join(sorted((data.get("by_dimension") or {}).keys())),
                "variant_types": ";".join(sorted((data.get("by_variant_type") or {}).keys())),
                "prompt_template_sample": (first or {}).get("prompt", "")[:240] if first else "",
                "split_recorded": (first or {}).get("split") if first else None,
                "decoding_recorded": "not recorded in prediction JSONL",
                "prediction_path": str(pred_path.relative_to(REPO)) if pred_path.exists() else "",
                "summary_path": str(summary_path.relative_to(REPO)),
                "reliability_note": "usable for thesis only as 30-example public/API baseline audit unless methods document the sampling and OpenRouter decoding settings",
            }
        )

    csv_path = PUBLIC / "public_llm_audit.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    md = PUBLIC / "public_llm_audit.md"
    with md.open("w") as f:
        f.write("# Public/API LLM Audit\n\n")
        f.write(f"Found {len(rows)} model/run summary files with matching prediction-file checks where available.\n\n")
        f.write("Important caveat: these are mostly `*_arch_30_*` runs with 30 examples per domain, not the 990-record CogMath validation protocol used by the fine-tuning experiments. Decoding parameters are not recorded in the prediction JSONL files I inspected.\n\n")
        f.write("| model | runs | row counts ok | min rows | max rows |\n| --- | ---: | ---: | ---: | ---: |\n")
        for model, model_rows in sorted(by_model.items()):
            counts = [r["prediction_rows_actual"] for r in model_rows if r["prediction_rows_actual"] is not None]
            ok = sum(1 for r in model_rows if r["counts_match"])
            f.write(f"| {model} | {len(model_rows)} | {ok}/{len(model_rows)} | {min(counts) if counts else 'n/a'} | {max(counts) if counts else 'n/a'} |\n")


def build_token_audit() -> None:
    TOKEN.mkdir(parents=True, exist_ok=True)
    datasets = {
        "budget_original_only": REPO / "experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl",
        "budget_original_plus_dim2": REPO / "experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2.jsonl",
        "budget_original_plus_all": REPO / "experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_all.jsonl",
        "attention_prompt_bidir_original_only": REPO / "experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_only.jsonl",
        "attention_prompt_bidir_original_plus_dim2": REPO / "experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_dim2.jsonl",
        "attention_prompt_bidir_original_plus_all": REPO / "experiments/cogmath_reasoning_augmentation/data/train_reasoning_original_plus_all.jsonl",
    }
    budget_static = read_json(REPO / "experiments/cogmath_reasoning_budget_control/static_validation_report.json")
    attn_static = read_json(REPO / "experiments/cogmath_reasoning_attention_control/static_validation_report.json")

    exposure_estimates = {}
    for report in [budget_static, attn_static]:
        for check in report.get("checks", []):
            if "example" in check.get("name", "") and isinstance(check.get("details"), dict):
                exposure_estimates.update(check["details"])

    rows = []
    for condition, path in datasets.items():
        s = summarize_training_file(path)
        key = condition.replace("attention_prompt_bidir_", "").replace("budget_", "")
        rows.append(
            {
                "condition": condition,
                "source_file": s["path"],
                "rows": s["rows"],
                "unique_base_ids": s["unique_base_ids"],
                "dimension_counts": json.dumps(s["dimension_counts"], sort_keys=True),
                "optimizer_steps": 232,
                "gradient_accumulation_steps": 16,
                "estimated_example_presentations_from_static_validation": exposure_estimates.get(key) or exposure_estimates.get(condition.replace("budget_", "")),
                "configured_max_length": 1536,
                "tokenizer_exact_available_locally": False,
                "median_prompt_plus_target_chars": s["median_prompt_plus_target_chars"],
                "max_prompt_plus_target_chars": s["max_prompt_plus_target_chars"],
                "median_prompt_plus_target_whitespace_tokens": s["median_prompt_plus_target_whitespace_tokens"],
                "max_prompt_plus_target_whitespace_tokens": s["max_prompt_plus_target_whitespace_tokens"],
                "audit_note": "Exact Qwen nonpadding/padded token parity requires Imperial tokenizer/runtime logs; local audit verifies file identity/counts and static presentation estimates.",
            }
        )
    csv_path = TOKEN / "training_token_exposure_audit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gen_rows = []
    for pred in sorted((REPO / "outputs").glob("cogmath_attention*/evaluations/validation/*/predictions.jsonl")):
        if "/test/" in str(pred):
            continue
        vals = [r.get("generated_tokens") for r in read_jsonl(pred)]
        vals = [v for v in vals if isinstance(v, (int, float))]
        gen_rows.append(
            {
                "prediction_file": str(pred.relative_to(REPO)),
                "rows": count_jsonl(pred),
                "mean_generated_tokens": mean(vals) if vals else None,
                "max_generated_tokens": max(vals) if vals else None,
                "cap_hit_count": sum(1 for v in vals if v >= 64),
                "max_new_tokens_in_summary": read_json(pred.with_name("summary.json")).get("max_new_tokens") if pred.with_name("summary.json").exists() else None,
            }
        )
    if gen_rows:
        with (TOKEN / "validation_generation_token_audit.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gen_rows[0].keys()))
            writer.writeheader()
            writer.writerows(gen_rows)

    with (TOKEN / "token_audit.md").open("w") as f:
        f.write("# Token And Exposure Audit\n\n")
        f.write("The fixed-budget Qwen3-4B experiments control optimizer steps at exactly 232. They must not be described as processing exactly 3,712 examples because finite dataset boundaries can create incomplete accumulation groups. Static validation currently estimates presentations as recorded in `training_token_exposure_audit.csv`.\n\n")
        f.write("Exact Qwen tokenizer nonpadding-token counts and runtime padded-token counts are not reproducible on this local machine because `transformers` and the checkpoint tokenizer are unavailable here. Run the cluster static validations and training wrappers to populate exact runtime exposure logs before thesis reporting.\n")


def build_frozen_test_manifest() -> None:
    manifest = {
        "status": "frozen_pending_user_confirmation",
        "created_by": "analysis/final_thesis/build_final_thesis_audit.py",
        "held_out_test_records_opened_in_this_audit": False,
        "test_file_paths_declared_but_not_read": [
            "experiments/cogmath_augmentation/test_eval.jsonl",
            "experiments/cogmath_attention/test_eval.jsonl",
        ],
        "primary_validation_comparison_before_test": {
            "model": MODEL_QWEN4B,
            "conditions": ["budget_original_only", "budget_original_plus_dim2"],
            "experiment": "cogmath_reasoning_budget_control_qwen3_4b",
            "optimizer_steps": 232,
            "seed_policy": "three seeds requested; local repo currently defines seed 42 only",
            "primary_metric": "Dim2 word-scrambling validation accuracy and matched problem-level robustness metrics",
            "paired_test": "exact McNemar on matched validation/test record correctness",
            "confidence_interval": "95% cluster bootstrap by base_id, seed=10000 reps",
            "prompt": "reasoning prompt from experiments/cogmath_reasoning_supervision/reasoning_utils.py",
            "decoding": "greedy, max_new_tokens=512",
            "extraction": "Final answer marker extraction from reasoning evaluation utilities",
        },
        "do_not_access_test_until": "User confirms this manifest and the selected checkpoint set.",
    }
    write_json(MANIFESTS / "frozen_test_manifest.json", manifest)
    with (MANIFESTS / "frozen_test_manifest.md").open("w") as f:
        f.write("# Frozen Test Manifest\n\n")
        f.write("Held-out test records were not opened or summarized while creating this manifest.\n\n")
        f.write(f"Primary model checkpoint ID: `{MODEL_QWEN4B}`\n\n")
        f.write("Primary comparison: `budget_original_only` vs `budget_original_plus_dim2` under the fixed 232-optimizer-step Qwen3-4B reasoning budget-control protocol.\n\n")
        f.write("Status: pending user confirmation before any held-out test access.\n")


def build_blocker_notes() -> None:
    (OUT / "matched_analysis_blocker.md").write_text(
        "# Matched Analysis Status\n\n"
        "The requested matched final analysis cannot be completed from the local checkout because validation prediction JSONL files for the Qwen3-4B fixed-budget reasoning conditions are not present locally. Required conditions include `budget_original_only`, `budget_original_plus_dim2`, `budget_original_plus_all`, and prompt-bidirectional attention-control validation predictions.\n\n"
        "Once those validation predictions are copied back under `outputs/`, run the final paired analysis over validation only. Held-out test remains locked behind the frozen manifest.\n"
    )
    QUAL.mkdir(parents=True, exist_ok=True)
    with (QUAL / "dim2_sample_template.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_type",
                "base_id",
                "record_id",
                "question",
                "gold_answer",
                "original_only_response",
                "original_only_extracted",
                "original_only_correct",
                "original_plus_dim2_response",
                "original_plus_dim2_extracted",
                "original_plus_dim2_correct",
                "human_error_label",
                "notes",
            ],
        )
        writer.writeheader()
    (QUAL / "dim2_sample_blocker.md").write_text(
        "# Dim2 Qualitative Sample Status\n\n"
        "The qualitative Dim2 rescue/failure/regression sample was not populated because the required fixed-budget Qwen3-4B validation predictions are not present locally. A blank coding template was created at `dim2_sample_template.csv`; no labels were invented.\n"
    )


def build_public_commands() -> None:
    commands = {
        "upload_analysis_files": "rsync -av analysis/final_thesis/ analysis/final_thesis/",
        "budget_static_validation": "python -u experiments/cogmath_reasoning_budget_control/static_validation.py",
        "attention_static_validation": "python -u experiments/cogmath_reasoning_attention_control/static_validation.py --require-causal-references --require-runtime-diagnostics",
        "submit_primary_three_seed_training": "sbatch analysis/final_thesis/slurm/train_budget_dim2_three_seed_array.sh",
        "submit_primary_three_seed_validation": "sbatch analysis/final_thesis/slurm/evaluate_budget_dim2_three_seed_array.sh",
        "submit_budget_training_seed42_existing_script": "sbatch experiments/cogmath_reasoning_budget_control/slurm/train_array.sh",
        "submit_attention_prompt_bidir_training": "sbatch experiments/cogmath_reasoning_attention_control/slurm/train_prompt_bidir_array.sh",
        "submit_budget_validation": "sbatch experiments/cogmath_reasoning_budget_control/slurm/evaluate_validation_array.sh",
        "submit_attention_validation": "sbatch experiments/cogmath_reasoning_attention_control/slurm/evaluate_validation_array.sh",
        "monitor_jobs": "squeue -u "$USER"",
        "copy_back_relevant_outputs": "rsync -av $USER@$IMPERIAL_LOGIN:outputs/cogmath_reasoning_budget_control_qwen3_4b/ outputs/cogmath_reasoning_budget_control_qwen3_4b/",
    }
    write_json(OUT / "cluster_commands.json", commands)


def main() -> None:
    for path in [OUT, PUBLIC, TOKEN, MANIFESTS, QUAL]:
        path.mkdir(parents=True, exist_ok=True)
    status_rows = build_experiment_status()
    write_experiment_status(status_rows)
    build_public_llm_audit()
    build_token_audit()
    build_frozen_test_manifest()
    build_blocker_notes()
    build_public_commands()
    write_json(
        OUT / "audit_summary.json",
        {
            "status_rows": len(status_rows),
            "slurm_available_locally": False,
            "held_out_test_records_opened": False,
            "exact_model_checkpoint_name_primary": MODEL_QWEN4B,
            "new_files_root": str(OUT.relative_to(REPO)),
        },
    )
    print(f"Wrote final thesis audit files under {OUT.relative_to(REPO)}")
    print(f"Primary model checkpoint ID: {MODEL_QWEN4B}")
    print("Held-out test records opened: false")


if __name__ == "__main__":
    main()
