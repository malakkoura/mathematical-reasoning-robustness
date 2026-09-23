#!/usr/bin/env python3
"""Prepare unique-full CogMath augmentation training datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from prepare_dataset import (
    DIMENSIONS,
    SOURCE_PATH,
    base_id_number,
    dimension_obj,
    extract_final_answer,
    load_source,
    make_original_record,
    make_variant_record,
)
from unique_full_utils import CONDITION_DIMENSIONS, UNIQUE_FULL_DIR, UNIQUE_FULL_TRAINING_CONDITIONS


SPLITS_PATH = Path(__file__).resolve().parent / "splits.json"


def read_splits() -> Dict[str, List[str]]:
    with SPLITS_PATH.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    for key in ["train", "validation", "test"]:
        if key not in splits or not isinstance(splits[key], list):
            raise ValueError(f"{SPLITS_PATH} lacks split list {key!r}.")
    return splits


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def invalid_record(reason: str, base_id: str, dimension: int, field: str) -> Dict[str, Any]:
    return {"reason": reason, "base_id": base_id, "dimension": dimension, "field": field}


def original_validity(base_id: str, item: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    invalid = []
    if not item.get("ori_question"):
        invalid.append(invalid_record("missing_or_empty_original_question", base_id, 0, "ori_question"))
    if not item.get("ori_answer"):
        invalid.append(invalid_record("missing_or_empty_original_answer", base_id, 0, "ori_answer"))
    elif extract_final_answer(item.get("ori_answer")) is None:
        invalid.append(invalid_record("unparseable_original_answer", base_id, 0, "ori_answer"))
    return not invalid, invalid


def variant_validity(base_id: str, item: Dict[str, Any], dimension: int) -> Tuple[bool, List[Dict[str, Any]]]:
    invalid = []
    original_ok, original_invalid = original_validity(base_id, item)
    if not original_ok:
        invalid.extend(
            {**entry, "reason": f"invalid_original_needed_for_dim{dimension}"} for entry in original_invalid
        )
    obj = dimension_obj(item, dimension)
    if not obj:
        invalid.append(invalid_record("missing_dimension_object", base_id, dimension, f"Dimension {dimension}"))
    elif not obj.get("inquiry"):
        invalid.append(invalid_record("missing_or_empty_variant_question", base_id, dimension, "inquiry"))
    if dimension == 6:
        if obj:
            if not obj.get("inquiry answer"):
                invalid.append(invalid_record("missing_or_empty_variant_answer", base_id, dimension, "inquiry answer"))
            elif extract_final_answer(obj.get("inquiry answer")) is None:
                invalid.append(invalid_record("unparseable_variant_answer", base_id, dimension, "inquiry answer"))
    return not invalid, invalid


def valid_record_for_dimension(base_id: str, item: Dict[str, Any], dimension: int, condition: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if dimension == 0:
        ok, invalid = original_validity(base_id, item)
        if not ok:
            return None, invalid
        return make_original_record(base_id, item, "train", condition), []
    ok, invalid = variant_validity(base_id, item, dimension)
    if not ok:
        return None, invalid
    record = make_variant_record(base_id, item, dimension, "train", condition)
    if record is None:
        return None, [invalid_record("variant_builder_returned_none", base_id, dimension, f"Dimension {dimension}")]
    return record, []


def condition_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
    return (base_id_number(row["base_id"]), int(row["dimension"]), str(row["record_id"]))


def validate_rows(rows: Sequence[Dict[str, Any]], condition: str, allowed_dimensions: Sequence[int], train_base_ids: set[str], validation_base_ids: set[str], test_base_ids: set[str]) -> None:
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        duplicates = [rid for rid, count in Counter(record_ids).items() if count > 1]
        raise AssertionError(f"{condition} has duplicate record IDs: {duplicates[:10]}")
    for row in rows:
        if row["condition"] != condition:
            raise AssertionError(f"{row['record_id']} condition={row['condition']!r}, expected {condition!r}.")
        if int(row["dimension"]) not in set(allowed_dimensions):
            raise AssertionError(f"{condition} contains unintended dimension {row['dimension']}.")
        if row["base_id"] not in train_base_ids:
            raise AssertionError(f"{condition} contains non-train base_id {row['base_id']}.")
        if row["base_id"] in validation_base_ids or row["base_id"] in test_base_ids:
            raise AssertionError(f"{condition} leaks validation/test base_id {row['base_id']}.")
        if "_rep" in row["record_id"]:
            raise AssertionError(f"{condition} introduced replica record ID {row['record_id']}.")
        if row.get("replicate_id") is not None:
            raise AssertionError(f"{condition} has non-null replicate_id for {row['record_id']}.")
        if not row.get("question"):
            raise AssertionError(f"{condition} has empty question for {row['record_id']}.")
        if row.get("training_target") != f"Final answer: {row['final_answer']}":
            raise AssertionError(f"{condition} has invalid training_target for {row['record_id']}.")
    original_rows = [row for row in rows if int(row["dimension"]) == 0]
    original_base_ids = [row["base_id"] for row in original_rows]
    if len(original_base_ids) != len(set(original_base_ids)):
        raise AssertionError(f"{condition} has repeated original rows.")


def build_condition_rows(data: Dict[str, Dict[str, Any]], train_base_ids: Sequence[str], condition: str, dimensions: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    invalid = []
    for base_id in sorted(train_base_ids, key=base_id_number):
        item = data[base_id]
        for dimension in dimensions:
            record, failures = valid_record_for_dimension(base_id, item, dimension, condition)
            if record is not None:
                rows.append(record)
            invalid.extend(failures)
    rows.sort(key=condition_sort_key)
    return rows, invalid


def grouped_invalid(invalid: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for entry in invalid:
        key_tuple = (entry["reason"], entry["base_id"], entry["dimension"], entry["field"])
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        dim_key = f"dim{entry['dimension']}"
        reason = entry["reason"]
        grouped.setdefault(dim_key, {}).setdefault(reason, {"count": 0, "examples": []})
        grouped[dim_key][reason]["count"] += 1
        if len(grouped[dim_key][reason]["examples"]) < 20:
            grouped[dim_key][reason]["examples"].append(entry)
    return grouped


def dataset_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts_by_dimension = Counter(int(row["dimension"]) for row in rows)
    counts_by_variant_type = Counter(row["variant_type"] for row in rows)
    return {
        "rows": len(rows),
        "unique_base_ids": len({row["base_id"] for row in rows}),
        "counts_by_dimension": {f"dim{dim}": counts_by_dimension.get(dim, 0) for dim in [0, 1, 2, 4, 6]},
        "counts_by_variant_type": dict(sorted(counts_by_variant_type.items())),
    }


def main() -> None:
    data = load_source()
    splits = read_splits()
    train_base_ids = splits["train"]
    train_set = set(splits["train"])
    validation_set = set(splits["validation"])
    test_set = set(splits["test"])

    UNIQUE_FULL_DIR.mkdir(parents=True, exist_ok=True)
    all_invalid: Dict[str, List[Dict[str, Any]]] = {}
    files = {}
    condition_stats = {}

    for condition, filename in UNIQUE_FULL_TRAINING_CONDITIONS.items():
        dimensions = CONDITION_DIMENSIONS[condition]
        rows, invalid = build_condition_rows(data, train_base_ids, condition, dimensions)
        validate_rows(rows, condition, dimensions, train_set, validation_set, test_set)
        path = UNIQUE_FULL_DIR / filename
        write_jsonl(path, rows)
        all_invalid[condition] = invalid
        files[filename] = {"rows": len(rows), "sha256": sha256(path)}
        condition_stats[condition] = dataset_stats(rows)

    conditions_payload = {
        key: {
            "name": key,
            "training_file": UNIQUE_FULL_TRAINING_CONDITIONS[key],
            "included_dimensions": CONDITION_DIMENSIONS[key],
            "description": "Unique-full Qwen3-4B augmentation condition with every valid source record for the listed dimensions.",
        }
        for key in UNIQUE_FULL_TRAINING_CONDITIONS
    }
    write_json(UNIQUE_FULL_DIR / "conditions.json", conditions_payload)

    invalid_flat = [entry for entries in all_invalid.values() for entry in entries]
    report = {
        "experiment": "cogmath_augmentation_qwen3_4b_unique_full",
        "model_name": "Qwen/Qwen3-4B-Base",
        "source_path": str(SOURCE_PATH),
        "split_source": str(SPLITS_PATH),
        "train_base_ids_assigned": len(train_base_ids),
        "validation_base_ids": len(validation_set),
        "test_base_ids": len(test_set),
        "data_rule": "all valid unique records for each condition; no repeats, oversampling, subsampling or artificial replicas",
        "condition_stats": condition_stats,
        "output_files": files,
        "invalid_or_unavailable_by_condition": {
            condition: grouped_invalid(entries) for condition, entries in all_invalid.items()
        },
        "invalid_or_unavailable_overall": grouped_invalid(invalid_flat),
        "array_conditions": list(UNIQUE_FULL_TRAINING_CONDITIONS),
    }
    write_json(UNIQUE_FULL_DIR / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
