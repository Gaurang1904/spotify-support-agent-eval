"""Pull the agent's actual failures out of outputs/predictions.jsonl.

Prints, per failure mode, real examples to paste into the report. No new
metrics -- this is a reading tool.

    python src/error_analysis.py
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "outputs" / "predictions.jsonl"


def show(title, rows, n=4):
    print("\n" + "=" * 72)
    print(f"{title}  (n={len(rows)})")
    print("=" * 72)
    for r in rows[:n]:
        a = r["agent"]
        print(f"\n  CUSTOMER : {r['customer_msg'][:180]}")
        print(f"  GOLD     : {r['gold_intent']} / {'escalate' if r['gold_escalate'] else 'auto'}")
        print(f"  AGENT    : {a['intent']} / {'escalate' if a['escalate'] else 'auto'}"
              f" (conf {a.get('confidence')})  reason: {a['reason'][:80]}")
        print(f"  REPLY    : {a['reply'][:200]}")
        if a.get("judge"):
            j = a["judge"]
            print(f"  JUDGE    : verdict={j['verdict']} total={j['total']} -- {j['why'][:110]}")


def main():
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8")]

    show("A. Missed escalations (auto-handled a case a human needed)",
         [r for r in rows if r["gold_escalate"] and not r["agent"]["escalate"]])
    show("B. Over-escalation (escalated something it could have answered)",
         [r for r in rows if not r["gold_escalate"] and r["agent"]["escalate"]])
    show("C. Judge says must-not-send",
         [r for r in rows if r["agent"].get("judge", {}).get("verdict") == 0])
    show("D. Intent wrong but confidence high",
         [r for r in rows if r["agent"]["intent"] != r["gold_intent"]
          and (r["agent"].get("confidence") or 0) >= 0.8])
    show("E. Empty or degenerate reply",
         [r for r in rows if len(r["agent"]["reply"]) < 25])

    conf = Counter((r["gold_intent"], r["agent"]["intent"])
                   for r in rows if r["gold_intent"] != r["agent"]["intent"])
    print("\n" + "=" * 72)
    print("Top intent confusions (gold -> predicted)")
    for (g, p), c in conf.most_common(10):
        print(f"  {c:3d}  {g:22} -> {p}")


if __name__ == "__main__":
    main()
