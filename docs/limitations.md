# Limitations

- Results are validation-set development results.
- Dim2 is interpreted as numerical-answer recovery under word scrambling in this project, not CogMath's original unsolvable classification task.
- LoRA placement and trainable parameter count are partly confounded.
- Prompt-bidirectional attention and deliberation were tested implementations, not proofs that those ideas cannot help under other setups.
- Raw predictions are omitted from the public release to avoid redistributing benchmark text; exact reported numbers are preserved in summaries.
