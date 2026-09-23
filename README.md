# Mathematical Reasoning Robustness

This repository contains the public, lightweight release for Malak Koura's MSc thesis project at Imperial College London on mathematical reasoning robustness under controlled transformations of GSM8K-style word problems from CogMath. The primary model for the final experiments is `Qwen/Qwen3-4B-Base`, used with LoRA fine-tuning and validation-set analyses.

## Research Question

Can targeted supervision on transformed mathematical problems improve robustness to input transformations, and do architectural or inference-time interventions such as LoRA placement, prompt-bidirectional attention, or second-pass deliberation add further benefit?

## Evaluated Dimensions

The core benchmark is the GSM8K-based portion of CogMath, evaluated on five forms:

- Dim0: original problem.
- Dim1: paraphrasing.
- Dim2: word scrambling.
- Dim4: irrelevant or redundant information.
- Dim6: numerical variation.

Important: in this project the prepared Dim2 examples retain the original numerical answer as the target. Dim2 therefore measures numerical-answer recovery under word scrambling. It is not treated as CogMath's original `unsolvable` classification task.

Controlled evaluation uses 990 validation records derived from 198 matched base problems: original, Dim1, Dim2, Dim4, and Dim6 forms for each base problem.

## Experimental Sequence

1. Reasoning-supervision and generation-budget diagnostic.
2. Baseline diagnosis across transformation dimensions.
3. Targeted and broader augmentation.
4. Fixed optimiser-step budget comparison.
5. Three-seed replication of original-only versus original-plus-Dim2.
6. LoRA-placement comparison.
7. Causal versus prompt-bidirectional attention.
8. Second-pass deliberation.
9. Final statistical and matched-prediction analysis.

## Main Findings

Stored validation summaries show that reasoning-format generation substantially improves over direct-answer baselines, but Dim2 word scrambling remains the hardest transformation. Under a fixed 232 optimiser-step budget, targeted Dim2 augmentation improves Dim2 validation accuracy relative to original-only training: in the three-seed replication, mean Dim2 accuracy is 0.1145 for original-only and 0.2441 for original-plus-Dim2. The full per-seed table is included in `analysis/tables/three_seed_summary.csv`. LoRA-placement results show that placement and trainable capacity are partly confounded. Prompt-bidirectional attention and second-pass deliberation did not provide an additional benefit in these implementations; the negative results are included rather than hidden.

All reported results are validation-set development results. The held-out test set was not accessed for this release.

## Repository Structure

- `experiments/`: final experiment code, configurations, static validators, and SLURM wrappers.
- `analysis/summaries/`: lightweight verified summaries and statistical outputs copied from the authoritative cluster staging bundle.
- `analysis/tables/`: compact tables regenerated from summaries.
- `data/manifests/`: dataset reports and row-count manifests; raw benchmark records are not redistributed.
- `docs/`: experiment map, result provenance, reproducibility notes, data/licensing notes, and limitations.
- `scripts/`: lightweight validation and analysis reproduction commands.
- `slurm/`: sanitized cluster job wrappers configured through environment variables.
- `tests/`: lightweight mask/unit tests.

## Installation

Python 3.10+ is recommended. For lightweight validation and summary analysis:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For model training/evaluation on a GPU cluster, use the package versions documented in `environment.yml` and the saved `package_versions.json` files under `analysis/summaries/`.

## Data Preparation

Raw GSM8K/CogMath records are not redistributed here. See `data/README.md` and `docs/data_and_licensing.md` for source access, expected row counts, and preparation scripts. The dataset reports in `data/manifests/` document the final prepared splits and explicit exclusions.

## Lightweight Validation

```bash
bash scripts/validate_release.sh
```

This compiles Python files, checks shell syntax, runs lightweight mask tests, verifies summary files, and searches for likely credential strings and private paths. It does not train, evaluate, download models, or inspect held-out test records.

## Reproduce Reported Analysis Tables

```bash
bash scripts/reproduce_analysis.sh
```

This regenerates compact CSV tables from existing lightweight summaries into `analysis/reproduced/` unless that directory already exists.

## SLURM Reproduction

Set environment variables for your cluster rather than editing scripts in place:

```bash
export PROJECT_ROOT="$(pwd)"
export HF_HOME="$HOME/.cache/huggingface"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV=math-reasoning
```

Then submit the relevant wrappers in `slurm/`, for example the three-seed primary comparison wrappers:

```bash
sbatch slurm/budget_control_three_seed__train_budget_dim2_three_seed_array.sh
sbatch slurm/budget_control_three_seed__evaluate_budget_dim2_three_seed_array.sh
```

## Licensing and Citation

No software licence is included in this release. Third-party datasets and models, including GSM8K, CogMath, and Qwen3, remain subject to their original terms. Cite this release using `CITATION.cff`.

## Contact

For thesis examination or access to non-public artifacts such as trained adapters or full validation prediction files, contact the repository owner through the thesis submission channel.
