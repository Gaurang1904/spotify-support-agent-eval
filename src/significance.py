"""Bootstrap confidence intervals on the agent-vs-simple deltas.

At n=147 a 4-point accuracy gap and a 9-vs-4 escalation-miss gap may be noise.
Paired bootstrap over examples (both systems scored on the same resampled rows,
so the pairing is preserved), 10,000 resamples.

A delta whose 95% interval straddles zero is reported as "no measured
difference", not as a win.

    python src/significance.py
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
import random
from pathlib import Path

from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "outputs" / "predictions.jsonl"
OUT = ROOT / "outputs" / "significance.json"
B = 10000
SEED = 11


def ci(vals, lo=2.5, hi=97.5):
    vals = sorted(vals)
    return (round(vals[int(len(vals) * lo / 100)], 4),
            round(vals[int(len(vals) * hi / 100)], 4))


def main():
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8")]
    n = len(rows)
    rng = random.Random(SEED)

    def acc(rs, sysname):
        return sum(r[sysname]["intent"] == r["gold_intent"] for r in rs) / len(rs)

    def mf1(rs, sysname):
        return f1_score([r["gold_intent"] for r in rs],
                        [r[sysname]["intent"] for r in rs],
                        average="macro", zero_division=0)

    def missed(rs, sysname):
        return sum(r["gold_escalate"] and not r[sysname]["escalate"] for r in rs)

    obs = {
        "accuracy": {s: round(acc(rows, s), 4) for s in ("simple", "agent")},
        "macro_f1": {s: round(mf1(rows, s), 4) for s in ("simple", "agent")},
        "missed_escalations": {s: missed(rows, s) for s in ("simple", "agent")},
    }

    deltas = {"accuracy": [], "macro_f1": [], "missed_escalations": []}
    for _ in range(B):
        samp = [rows[rng.randrange(n)] for _ in range(n)]
        deltas["accuracy"].append(acc(samp, "agent") - acc(samp, "simple"))
        deltas["macro_f1"].append(mf1(samp, "agent") - mf1(samp, "simple"))
        deltas["missed_escalations"].append(
            missed(samp, "agent") - missed(samp, "simple"))

    res = {"n": n, "resamples": B, "observed": obs, "agent_minus_simple": {}}
    for k, v in deltas.items():
        lo, hi = ci(v)
        res["agent_minus_simple"][k] = {
            "observed": round(obs[k]["agent"] - obs[k]["simple"], 4),
            "ci95": [lo, hi],
            "crosses_zero": lo <= 0 <= hi,
            "p_agent_worse": round(sum(d < 0 for d in v) / B, 4),
        }

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    for k, d in res["agent_minus_simple"].items():
        verdict = "NO MEASURED DIFFERENCE" if d["crosses_zero"] else "real difference"
        print(f"{k:20} delta={d['observed']:+.4f}  95% CI {d['ci95']}  -> {verdict}")


if __name__ == "__main__":
    main()
