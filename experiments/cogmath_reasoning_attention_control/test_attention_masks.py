#!/usr/bin/env python3
"""Deterministic mask tests for reasoning attention-control semantics."""

from __future__ import annotations

import unittest

from attention_masks import allowed_matrix, infer_prompt_lengths_from_labels, prompt_bidir_allowed_matrix


class PureMaskTests(unittest.TestCase):
    def test_prompt_bidirectional_visibility(self) -> None:
        matrix = prompt_bidir_allowed_matrix(seq_len=6, prompt_len=3, valid_len=5)

        self.assertTrue(matrix[0][2], "prompt tokens can see later prompt tokens")
        self.assertFalse(matrix[0][3], "prompt tokens cannot see answer tokens")
        self.assertTrue(matrix[3][0], "answer tokens can see the full prompt")
        self.assertTrue(matrix[4][3], "answer tokens can see prior answer tokens")
        self.assertFalse(matrix[3][4], "answer tokens cannot see future answer tokens")
        self.assertFalse(matrix[0][5], "padding keys are never visible")
        self.assertFalse(any(matrix[5]), "padding queries have no visible keys")

    def test_causal_baseline_differs_only_on_prompt_prompt_edges(self) -> None:
        causal = allowed_matrix(seq_len=5, prompt_len=3, valid_len=5, attention_type="causal")
        prompt_bidir = allowed_matrix(seq_len=5, prompt_len=3, valid_len=5, attention_type="prompt_bidirectional")

        self.assertFalse(causal[0][2])
        self.assertTrue(prompt_bidir[0][2])
        self.assertEqual(causal[3][4], prompt_bidir[3][4])
        self.assertEqual(causal[4][3], prompt_bidir[4][3])

    def test_variable_prompt_lengths(self) -> None:
        first = prompt_bidir_allowed_matrix(seq_len=5, prompt_len=2, valid_len=4)
        second = prompt_bidir_allowed_matrix(seq_len=5, prompt_len=4, valid_len=4)

        self.assertFalse(first[0][2])
        self.assertTrue(first[2][0])
        self.assertTrue(second[0][3])
        self.assertFalse(second[3][4])

    def test_prompt_lengths_come_from_label_boundary(self) -> None:
        labels = [
            [-100, -100, -100, 10, 11],
            [-100, -100, 20, 21, -100],
        ]
        self.assertEqual(infer_prompt_lengths_from_labels(labels), [3, 2])

    def test_no_boundary_marker_needed_for_generation_slice(self) -> None:
        prompt_ids = [101, 102, 103]
        generated = [101, 102, 103, 201, 202]
        prompt_length = len(prompt_ids)
        self.assertEqual(generated[prompt_length:], [201, 202])


if __name__ == "__main__":
    unittest.main()
