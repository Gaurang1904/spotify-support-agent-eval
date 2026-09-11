# Reproduce every number in the report. PowerShell equivalent of run_all.sh.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

python src/prep.py            # raw csv       -> data/threads.jsonl
python src/sample_golden.py   #               -> data/golden_candidates.jsonl
python src/evaluate.py        # golden.jsonl  -> outputs/metrics.json, predictions.jsonl
python src/human_ceiling.py   # brand's own replies through the same judge
python src/significance.py    # bootstrap CIs on agent-vs-simple
python src/agreement.py       # judge vs human kappa
python src/grounding_ablation.py --report   # grounding fix ablation (uses committed ratings)
python src/error_analysis.py  # the failures quoted in the report
