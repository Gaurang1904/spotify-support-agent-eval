"""How much does the LLM judge agree with the human rater?

Without this number the judge scores in the report mean nothing. Reports exact
agreement, quadratically-weighted Cohen's kappa (ordinal scale, so a 2-vs-1
disagreement should not cost the same as 2-vs-0), Spearman rho, and the judge's
directional bias -- whether it is systematically softer or harsher than a human.

A human-vs-human number is not available (one rater), which is stated as a
limitation in the report rather than papered over.
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from judge import judge_reply

ROOT = Path(__file__).resolve().parent.parent
RATINGS = ROOT / "data" / "human_reply_ratings.jsonl"
PRED = ROOT / "outputs" / "predictions.jsonl"
OUT = ROOT / "outputs" / "agreement.json"


def main():
    if not RATINGS.exists():
        sys.exit("No human ratings -- run: python src/label_replies.py")
    human = [json.loads(l) for l in open(RATINGS, encoding="utf-8")]

    # Reuse the judge scores from the eval run where possible (cached anyway).
    ref = {}
    if PRED.exists():
        for r in (json.loads(l) for l in open(PRED, encoding="utf-8")):
            ref[r["thread_id"]] = r

    h, j, rows = [], [], []
    for d in human:
        row = ref.get(d["thread_id"], {})
        s = judge_reply(d["customer_msg"], d["reply"],
                        row.get("brand_reply"))
        h.append(d["human_verdict"])
        j.append(s["verdict"])
        rows.append({**d, "judge_verdict": s["verdict"], "judge_total": s["total"],
                     "judge_why": s["why"]})

    exact = sum(a == b for a, b in zip(h, j)) / len(h)
    within1 = sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(h)
    res = {
        "n": len(h),
        "exact_agreement": round(exact, 3),
        "within_1_agreement": round(within1, 3),
        "quadratic_weighted_kappa": round(
            cohen_kappa_score(h, j, weights="quadratic"), 3),
        "spearman_rho": round(float(spearmanr(h, j).statistic), 3),
        "judge_mean": round(sum(j) / len(j), 2),
        "human_mean": round(sum(h) / len(h), 2),
        "judge_bias": round((sum(j) - sum(h)) / len(h), 2),
        "confusion_human_judge": {f"h{a}_j{b}": c for (a, b), c
                                  in Counter(zip(h, j)).most_common()},
        "by_system": {},
    }
    for sysname in {d["system"] for d in human}:
        idx = [i for i, d in enumerate(human) if d["system"] == sysname]
        res["by_system"][sysname] = {
            "n": len(idx),
            "human_mean": round(sum(h[i] for i in idx) / len(idx), 2),
            "judge_mean": round(sum(j[i] for i in idx) / len(idx), 2),
        }

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    (ROOT / "outputs" / "agreement_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
