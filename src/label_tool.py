"""Terminal labelling tool for the golden set. Resumable.

    python src/label_tool.py            # label / continue
    python src/label_tool.py --review   # walk back through what you labelled
    python src/label_tool.py --stats

Per example you give two things: the intent (1-8) and the decision
(a = auto-handle, e = escalate). Label the decision as what a good human
support lead would want, not what Spotify happened to do.

The intent is asked BLIND -- what Spotify actually replied is revealed only
after you have committed to an intent, because it is evidence for the routing
call but contamination for the intent label.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from taxonomy import ESCALATION_POLICY, INTENTS, LABELS

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data" / "golden_candidates.jsonl"
GOLD = ROOT / "data" / "golden.jsonl"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def write(done, cands):
    """Rewrite the whole file after every label, so a crash costs nothing."""
    with open(GOLD, "w", encoding="utf-8") as f:
        for c in cands:
            if c["thread_id"] in done:
                f.write(json.dumps(done[c["thread_id"]], ensure_ascii=False) + "\n")


def banner():
    print("\n" + "=" * 72)
    for i, k in enumerate(LABELS, 1):
        print(f"  {i}  {k:22} {INTENTS[k][:44]}")
    print("  a = auto-handle    e = escalate    x = ambiguous    s = skip    q = quit")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    cands = read_jsonl(CAND)
    done = {r["thread_id"]: r for r in read_jsonl(GOLD)}

    if args.stats:
        from collections import Counter
        rows = list(done.values())
        amb = [r for r in rows if r["intent"] == "ambiguous"]
        clear = [r for r in rows if r["intent"] != "ambiguous"]
        print(f"labelled {len(rows)}/{len(cands)}")
        print(Counter(r["intent"] for r in rows).most_common())
        print(f"ambiguous: {len(amb)}/{len(rows)} = {len(amb)/max(len(rows),1):.1%}")
        if clear:
            print("escalate rate:",
                  round(sum(bool(r["escalate"]) for r in clear) / len(clear), 3))
        secs = [r["seconds"] for r in rows if r.get("seconds")]
        if secs:
            secs_sorted = sorted(secs)
            print(f"median {secs_sorted[len(secs)//2]:.0f}s/item, "
                  f"total {sum(secs)/60:.0f} min over {len(secs)} timed items")
        return

    todo = cands if args.review else [c for c in cands if c["thread_id"] not in done]
    if not todo:
        print(f"All {len(done)} labelled. Run --stats, or --review to revise.")
        return

    print(ESCALATION_POLICY)
    for n, c in enumerate(todo, 1):
        banner()
        prev = done.get(c["thread_id"])
        print(f"[{len(done)}/{len(cands)} done]  slice={c['slice']}  turns={c['n_turns']}")
        print(f"\nCUSTOMER: {c['customer_msg']}\n")

        # Intent is labelled BLIND. Showing Spotify's own reply here would
        # anchor the intent label to their routing habits, and since these
        # labels are the ground truth every agreement number is computed
        # against, that contamination would silently inflate all of them.
        t0 = time.monotonic()
        intent = None
        while intent is None:
            raw = input("intent 1-8 (x=ambiguous, s/q): ").strip().lower()
            if raw == "q":
                print(f"saved {len(done)} labels."); return
            if raw == "s":
                intent = "SKIP"
            elif raw == "x":
                # Escape hatch, on purpose. Forcing a guess on a genuinely
                # unreadable tweet puts noise in the ground truth; recording it
                # as ambiguous makes that noise a measurable rate instead.
                intent = "ambiguous"
            elif raw.isdigit() and 1 <= int(raw) <= len(LABELS):
                intent = LABELS[int(raw) - 1]
            else:
                print(f"  ? need 1-{len(LABELS)}, x=ambiguous, s=skip, q=quit")
        if intent == "SKIP":
            continue
        if intent == "ambiguous":
            done[c["thread_id"]] = {
                "thread_id": c["thread_id"], "slice": c["slice"],
                "customer_msg": c["customer_msg"], "brand_reply": c["brand_reply"],
                "intent": "ambiguous", "escalate": None, "note": "",
                "seconds": round(time.monotonic() - t0, 1),
            }
            write(done, cands)
            continue

        # Reveal it now. For the routing call, "did a human have to take this
        # backstage?" is genuine evidence rather than contamination.
        print(f"\n(spotify actually said: {c['brand_reply'][:160]})")
        if prev:
            print(f"(your earlier label: {prev['intent']} / "
                  f"{'escalate' if prev['escalate'] else 'auto'})")
        dec = ""
        while dec not in ("a", "e"):
            dec = input("a=auto  e=escalate: ").strip().lower()
            if dec not in ("a", "e"):
                print("  ? need a (auto-handle) or e (escalate)")
        note = input("note (optional, enter to skip): ").strip()

        done[c["thread_id"]] = {
            "thread_id": c["thread_id"], "slice": c["slice"],
            "customer_msg": c["customer_msg"], "brand_reply": c["brand_reply"],
            "intent": intent, "escalate": dec == "e", "note": note,
            "seconds": round(time.monotonic() - t0, 1),
        }
        write(done, cands)

    print(f"\ndone. {len(done)} labels in {GOLD}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
