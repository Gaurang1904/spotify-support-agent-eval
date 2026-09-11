"""Score the brand's OWN replies with the same judge, as a control.

Without this, "12% of the agent's replies are sendable as-is" has no
denominator. If SpotifyCares' real human agents also score low under this
rubric, the number is telling you about the judge's strictness, not about the
agent. If they score high, the gap is real and worth reporting.

The brand's reply is passed as the candidate with NO historical context, so
the judge cannot simply reward it for matching itself.

    python src/human_ceiling.py
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
from pathlib import Path

from judge import judge_reply

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden.jsonl"
OUT = ROOT / "outputs" / "human_ceiling.json"


def main():
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    rows = [r for r in rows if r.get("intent") != "ambiguous" and r["brand_reply"]]

    scored = []
    for i, r in enumerate(rows, 1):
        scored.append(judge_reply(r["customer_msg"], r["brand_reply"]))
        if i % 25 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    n = len(scored)
    res = {
        "n": n,
        **{k: round(sum(s[k] for s in scored) / n, 2)
           for k in ("grounded", "helpful", "tone", "safe", "total", "verdict")},
        "pct_sendable": round(sum(s["verdict"] == 2 for s in scored) / n, 3),
        "pct_must_not_send": round(sum(s["verdict"] == 0 for s in scored) / n, 3),
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")
    print("\nRead this as the ceiling: these are real human support replies "
          "scored by the same rubric the agent is scored by.")


if __name__ == "__main__":
    main()
