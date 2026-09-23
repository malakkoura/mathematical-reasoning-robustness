# Reproducibility

Run `bash scripts/validate_release.sh` for lightweight validation. Run `bash scripts/reproduce_analysis.sh` to regenerate compact tables from saved summaries.

Full training/evaluation requires authorised data, `Qwen/Qwen3-4B-Base`, a CUDA GPU environment, and the SLURM wrappers in `slurm/`. Set `PROJECT_ROOT`, `HF_HOME`, `CONDA_SH`, and `CONDA_ENV` for your cluster before submitting jobs.

No command in this release should access held-out test records by default.
