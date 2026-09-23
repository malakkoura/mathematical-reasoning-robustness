#!/usr/bin/env python3
"""Unit checks for deliberation attention-mask semantics."""

from __future__ import annotations

from deliberation_utils import assert_deliberation_mask_semantics, prompt_bidir_allowed_matrix


def test_fixed_prefix_bidirectional_and_revision_causal() -> None:
    report = assert_deliberation_mask_semantics(prefix_len=4, generated_len=3, pad_len=2)
    assert report["prompt_tokens_bidirectional"]
    assert report["prompt_tokens_cannot_attend_revision_tokens"]
    assert report["revision_tokens_causal"]
    assert report["padding_masked"]


def test_future_revision_tokens_are_invisible() -> None:
    prefix_len = 5
    generated_len = 4
    matrix = prompt_bidir_allowed_matrix(prefix_len + generated_len, prefix_len, prefix_len + generated_len)
    first_revision = prefix_len
    future_revision = prefix_len + 2
    assert not matrix[first_revision][future_revision]
    assert matrix[future_revision][first_revision]


def test_prompt_cannot_see_revision_tokens() -> None:
    prefix_len = 6
    generated_len = 4
    matrix = prompt_bidir_allowed_matrix(prefix_len + generated_len, prefix_len, prefix_len + generated_len)
    assert not matrix[0][prefix_len]
    assert not matrix[prefix_len - 1][prefix_len + generated_len - 1]


def main() -> None:
    test_fixed_prefix_bidirectional_and_revision_causal()
    test_future_revision_tokens_are_invisible()
    test_prompt_cannot_see_revision_tokens()
    print("deliberation mask tests passed")


if __name__ == "__main__":
    main()
