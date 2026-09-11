#!/usr/bin/env bash
# Reproduce every number in the report. Assumes data/raw/twcs.csv exists
# (see README) and that data/golden.jsonl -- the hand-labelled set -- is present.
set -e
export PYTHONIOENCODING=utf-8

# Windows Git Bash often has py/python3 but not `python` on PATH.
PY=$(command -v python || command -v python3 || command -v py) \
  || { echo "No python found on PATH."; exit 1; }

"$PY" src/prep.py            # raw csv       -> data/threads.jsonl
"$PY" src/sample_golden.py   #               -> data/golden_candidates.jsonl
"$PY" src/evaluate.py        # golden.jsonl  -> outputs/metrics.json, predictions.jsonl
"$PY" src/human_ceiling.py   # brand's own replies through the same judge
"$PY" src/significance.py    # bootstrap CIs on agent-vs-simple
"$PY" src/agreement.py       # judge vs human kappa
"$PY" src/grounding_ablation.py --report   # grounding fix ablation (uses committed ratings)
"$PY" src/error_analysis.py  # the failures quoted in the report
