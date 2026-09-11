"""Does a one-line prompt change fix the grounding failure? (report failure 1)

The evaluated system is frozen. This runs the SAME agent with `grounding_patch`
on and off over a subset, so the fix can be measured without invalidating the
main run or the 60 human reply ratings pinned to it.

The metric here is deliberately NOT the LLM judge. That judge failed its own
validity check (kappa 0.100 against a human, inverted system ranking), so no
quality score it produces can carry an argument. Instead the metric is binary
and human-rated:

    C = contradicts the retrieved cases (says available when they say not, etc.)
    I = ignores a retrieved answer that was right there
    G = grounded: consistent with the retrieved cases, or honestly says it
        does not know

A contradiction check survives the judge's invalidity because it asks a
question of fact, not of taste.

    python src/grounding_ablation.py --build    # generate both replies per case
    python src/grounding_ablation.py --rate     # blind human rating
    python src/grounding_ablation.py --report   # before/after
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from agent import Agent
from retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden.jsonl"
PAIRS = ROOT / "outputs" / "ablation_pairs.jsonl"
RATINGS = ROOT / "data" / "ablation_ratings.jsonl"
SEED = 3

# The two cases that exposed the failure in a live demo. Pinned so the ablation
# is always tested on the examples the report quotes.
DEMO = [
    "why isn't the new Taylor Swift album on spotify in the UK?",
    "hey Spotify, how do I access your live chat? Used it last month and it was amazing",
]


def build(n=24):
    """Only cases where retrieval actually fired — otherwise there is no
    grounding to obey and the patch has nothing to act on."""
    gold = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    r = Retriever(exclude_ids=[g["thread_id"] for g in gold])
    base, patched = Agent(r), Agent(r, grounding_patch=True)

    pool = []
    for g in gold:
        if g["intent"] == "ambiguous":
            continue
        hits = r.search(g["customer_msg"], k=3)
        if hits and hits[0][0] >= 0.35:
            pool.append((g["customer_msg"], g["thread_id"]))
    random.Random(SEED).shuffle(pool)
    cases = [(m, "demo") for m in DEMO] + pool[:n]

    rows = []
    for i, (msg, tid) in enumerate(cases, 1):
        b, p = base.handle(msg), patched.handle(msg)
        rows.append({
            "id": i, "thread_id": tid, "customer_msg": msg,
            "retrieved": [h["reply"] for h in b["retrieved"]],
            "base_reply": b["reply"], "patched_reply": p["reply"],
        })
        print(f"{i}/{len(cases)}", flush=True)

    PAIRS.parent.mkdir(exist_ok=True)
    with open(PAIRS, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {PAIRS} ({len(rows)} cases, {len(rows)*2} replies to rate)")


def rate():
    rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
    items = [(r["id"], v, r) for r in rows for v in ("base", "patched")]
    random.Random(SEED).shuffle(items)  # blind: variant order scrambled

    done = {}
    if RATINGS.exists():
        done = {(d["id"], d["variant"]): d
                for d in (json.loads(l) for l in open(RATINGS, encoding="utf-8"))}

    print("c = contradicts retrieved | i = ignores a retrieved answer | "
          "g = grounded | s = skip | q = quit\n")
    for cid, variant, row in items:
        if (cid, variant) in done:
            continue
        print("=" * 72)
        print(f"[{len(done)}/{len(items)}]")
        print(f"CUSTOMER : {row['customer_msg']}")
        print("RETRIEVED:")
        for h in row["retrieved"]:
            print(f"   - {h[:120]}")
        print(f"\nREPLY    : {row[variant + '_reply'] or '(empty)'}")
        v = ""
        while v not in ("c", "i", "g", "s", "q"):
            v = input("rating: ").strip().lower()
            if v not in ("c", "i", "g", "s", "q"):
                print("  ? c / i / g / s / q")
        if v == "q":
            break
        if v == "s":
            continue
        done[(cid, variant)] = {"id": cid, "variant": variant, "rating": v,
                                "customer_msg": row["customer_msg"],
                                "reply": row[variant + "_reply"]}
        with open(RATINGS, "w", encoding="utf-8") as f:
            for d in done.values():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n{len(done)} ratings in {RATINGS}")


def report():
    rows = [json.loads(l) for l in open(RATINGS, encoding="utf-8")]
    out = {}
    for variant in ("base", "patched"):
        rs = [r["rating"] for r in rows if r["variant"] == variant]
        if not rs:
            continue
        c = Counter(rs)
        out[variant] = {
            "n": len(rs),
            "contradicts": c["c"], "ignores": c["i"], "grounded": c["g"],
            "ungrounded_rate": round((c["c"] + c["i"]) / len(rs), 3),
        }
    print(json.dumps(out, indent=2))
    if len(out) == 2:
        d = out["patched"]["ungrounded_rate"] - out["base"]["ungrounded_rate"]
        print(f"\nungrounded rate: base {out['base']['ungrounded_rate']:.3f} "
              f"-> patched {out['patched']['ungrounded_rate']:.3f}  ({d:+.3f})")
        print(f"n={out['base']['n']} per variant -- far too small to be "
              f"conclusive, and reported as such.")
    (ROOT / "outputs" / "ablation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    if a.build:
        build(a.n)
    elif a.rate:
        rate()
    elif a.report:
        report()
    else:
        ap.print_help()
