#!/usr/bin/env bash
set -euo pipefail
python3 -m compileall -q experiments analysis/scripts tests
find . -path './.git' -prune -o -name '*.sh' -print | while read -r script; do bash -n "$script"; done
PYTHONPATH=experiments/cogmath_reasoning_attention_control python3 tests/cogmath_reasoning_attention_control/test_attention_masks.py
PYTHONPATH=experiments/deliberation:experiments/cogmath_reasoning_attention_control:experiments/cogmath_augmentation python3 tests/deliberation/test_deliberation_masks.py
test -f analysis/tables/key_results.csv
test -f docs/results_manifest.md
if rg -n --hidden '(api[_-]?key|secret|password|BEGIN .*PRIVATE KEY|HF_TOKEN|OPENAI_API_KEY|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})' . --glob '!analysis/summaries/**' --glob '!docs/results_manifest.md' --glob '!scripts/validate_release.sh'; then
  echo 'Potential credential-like string found; inspect above.' >&2
  exit 1
fi
if rg -n '(/Users/|/home/|/vol/|mk5223|shell1|vm-biomedia)' . --glob '!scripts/validate_release.sh'; then
  echo 'Private absolute path or username found; inspect above.' >&2
  exit 1
fi
python3 - <<'PY'
from pathlib import Path
assert not any(Path('.').rglob('test_eval.jsonl')), 'held-out test file present in release'
print('release validation checks passed')
PY
