#!/usr/bin/env python3
"""Prompt-bidirectional attention masks with token-level boundaries."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence


AttentionType = Literal["causal", "prompt_bidirectional"]


def validate_attention_type(attention_type: str) -> AttentionType:
    if attention_type not in {"causal", "prompt_bidirectional"}:
        raise ValueError(f"Unknown attention_type={attention_type!r}.")
    return attention_type  # type: ignore[return-value]


def causal_allowed_matrix(seq_len: int, valid_len: int) -> List[List[bool]]:
    return [[query < valid_len and key < valid_len and key <= query for key in range(seq_len)] for query in range(seq_len)]


def prompt_bidir_allowed_matrix(seq_len: int, prompt_len: int, valid_len: int) -> List[List[bool]]:
    if prompt_len < 0 or prompt_len > seq_len:
        raise ValueError("prompt_len must be between 0 and seq_len.")
    if valid_len < 0 or valid_len > seq_len:
        raise ValueError("valid_len must be between 0 and seq_len.")
    if prompt_len > valid_len:
        raise ValueError("prompt_len cannot exceed valid_len.")
    matrix = []
    for query in range(seq_len):
        row = []
        for key in range(seq_len):
            if query >= valid_len or key >= valid_len:
                row.append(False)
            elif query < prompt_len:
                row.append(key < prompt_len)
            else:
                row.append(key < prompt_len or key <= query)
        matrix.append(row)
    return matrix


def allowed_matrix(seq_len: int, prompt_len: int, valid_len: int, attention_type: AttentionType) -> List[List[bool]]:
    attention_type = validate_attention_type(attention_type)
    if attention_type == "causal":
        return causal_allowed_matrix(seq_len, valid_len)
    return prompt_bidir_allowed_matrix(seq_len, prompt_len, valid_len)


def infer_prompt_lengths_from_labels(labels: Sequence[Sequence[int]], ignore_index: int = -100) -> List[int]:
    lengths = []
    for row in labels:
        first_label = next((index for index, value in enumerate(row) if int(value) != ignore_index), len(row))
        lengths.append(first_label)
    return lengths


def _require_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for tensor attention masks.") from exc
    return torch


def build_visibility_matrix(prompt_lengths, attention_mask, attention_type: AttentionType):
    torch = _require_torch()
    attention_type = validate_attention_type(attention_type)
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be 2D [batch, seq], got {tuple(attention_mask.shape)}")

    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape
    prompt_lengths = torch.as_tensor(prompt_lengths, dtype=torch.long, device=device)
    if tuple(prompt_lengths.shape) != (batch_size,):
        raise ValueError(f"prompt_lengths must have shape ({batch_size},), got {tuple(prompt_lengths.shape)}")
    valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if torch.any(prompt_lengths < 0) or torch.any(prompt_lengths > valid_lengths):
        raise ValueError("Each prompt length must be between 0 and that example's valid token length.")

    positions = torch.arange(seq_len, device=device)
    query_pos = positions.view(1, seq_len, 1)
    key_pos = positions.view(1, 1, seq_len)
    query_valid = query_pos < valid_lengths.view(batch_size, 1, 1)
    key_valid = key_pos < valid_lengths.view(batch_size, 1, 1)

    if attention_type == "causal":
        return query_valid & key_valid & (key_pos <= query_pos)

    prompt_len = prompt_lengths.view(batch_size, 1, 1)
    query_is_prompt = query_pos < prompt_len
    key_is_prompt = key_pos < prompt_len
    prompt_query_visibility = query_is_prompt & key_is_prompt
    answer_query_visibility = (~query_is_prompt) & (key_is_prompt | (key_pos <= query_pos))
    return query_valid & key_valid & (prompt_query_visibility | answer_query_visibility)


def build_4d_attention_mask(prompt_lengths, attention_mask, attention_type: AttentionType, dtype=None, device: Optional[object] = None):
    torch = _require_torch()
    if device is not None:
        attention_mask = attention_mask.to(device)
    if dtype is None:
        dtype = torch.float32
    visible = build_visibility_matrix(prompt_lengths, attention_mask, attention_type)
    additive = torch.zeros(visible.shape, dtype=dtype, device=visible.device)
    return additive.masked_fill(~visible, torch.finfo(dtype).min)[:, None, :, :]


def build_prompt_bidir_decode_mask(batch_size: int, key_length: int, dtype=None, device: Optional[object] = None):
    """Return the single-query decode mask after prompt-bidirectional prefill.

    During cached generation, the current generated token may attend to every
    cached prompt token, every previous generated token, and itself. Future
    generated tokens do not exist in the cache, so the required additive mask is
    all zeros with shape [batch, 1, 1, key_length].
    """
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    return torch.zeros((batch_size, 1, 1, key_length), dtype=dtype, device=device)


def build_model_attention_mask(prompt_lengths, attention_mask, attention_type: AttentionType, dtype=None, device: Optional[object] = None):
    attention_type = validate_attention_type(attention_type)
    if attention_type == "causal":
        return attention_mask.to(device) if device is not None else attention_mask
    return build_4d_attention_mask(prompt_lengths, attention_mask, attention_type, dtype=dtype, device=device)
