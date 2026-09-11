"""Blind human rating of draft replies -- the ground truth for the LLM judge.

Run after evaluate.py. Replies from all three systems are pooled, shuffled and
shown without saying which system wrote them, so the rating is not biased
toward "the fancy one".

    python src/label_replies.py --n 60

Same 3-point scale the judge uses:
    2 = send as-is    1 = needs an edit    0 = must not send
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "outputs" / "predictions.jsonl"
OUT = ROOT / "data" / "human_reply_ratings.jsonl"
SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of replies to rate")
    args = ap.parse_args()

    if not PRED.exists():
        sys.exit("Run python src/evaluate.py first.")
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8")]

    pool = [
        {"thread_id": r["thread_id"], "system": s, "customer_msg": r["customer_msg"],
         "reply": r[s]["reply"]}
        for r in rows for s in ("trivial", "simple", "agent") if r[s]["reply"]
    ]
    random.Random(SEED).shuffle(pool)

    done = {}
    if OUT.exists():
        done = {(d["thread_id"], d["system"]): d
                for d in (json.loads(l) for l in open(OUT, encoding="utf-8"))}

    todo = [p for p in pool if (p["thread_id"], p["system"]) not in done][:max(
        0, args.n - len(done))]
    if not todo:
        print(f"{len(done)} ratings already in {OUT}.")
        return

    print("2 = send as-is | 1 = needs an edit | 0 = must not send | s = skip | q = quit")
    for p in todo:
        print("\n" + "=" * 72)
        print(f"[{len(done)}/{args.n}]")
        print(f"CUSTOMER: {p['customer_msg']}")
        print(f"REPLY   : {p['reply']}")
        v = ""
        while v not in ("0", "1", "2", "s", "q"):
            v = input("rating: ").strip().lower()
        if v == "q":
            break
        if v == "s":
            continue
        done[(p["thread_id"], p["system"])] = {**p, "human_verdict": int(v)}
        with open(OUT, "w", encoding="utf-8") as f:
            for d in done.values():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n{len(done)} ratings in {OUT}")


if __name__ == "__main__":
    main()
