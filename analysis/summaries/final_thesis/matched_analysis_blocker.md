# Matched Analysis Status

The requested matched final analysis cannot be completed from the local checkout because validation prediction JSONL files for the Qwen3-4B fixed-budget reasoning conditions are not present locally. Required conditions include `budget_original_only`, `budget_original_plus_dim2`, `budget_original_plus_all`, and prompt-bidirectional attention-control validation predictions.

Once those validation predictions are copied back under `outputs/`, run the final paired analysis over validation only. Held-out test remains locked behind the frozen manifest.
