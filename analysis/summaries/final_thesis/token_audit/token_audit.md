# Token And Exposure Audit

The fixed-budget Qwen3-4B experiments control optimizer steps at exactly 232. They must not be described as processing exactly 3,712 examples because finite dataset boundaries can create incomplete accumulation groups. Static validation currently estimates presentations as recorded in `training_token_exposure_audit.csv`.

Exact Qwen tokenizer nonpadding-token counts and runtime padded-token counts are not reproducible on this local machine because `transformers` and the checkpoint tokenizer are unavailable here. Run the cluster static validations and training wrappers to populate exact runtime exposure logs before thesis reporting.
