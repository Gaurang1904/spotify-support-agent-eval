"""Model-assisted labelling: audit pre-labels instead of typing every one.

    python src/audit_tool.py            # audit / continue
    python src/audit_tool.py --stats    # progress + correction rate

Each example shows the model's proposed intent and route. Press Enter to
accept, or type a correction:

    5        change intent to 5, keep the proposed route
    e        keep the proposed intent, change route to escalate
    5e       change both
    s        skip    q  save & quit

Every row still passes under a human's eye, and each one is recorded with
whether the human changed it. That correction rate is the honest measure of
how much of this golden set is actually the model's opinion -- it goes in the
report, and it is the number an interviewer should ask about.

Trade-off vs label_tool.py, stated plainly: that tool asks for the intent
blind, so its labels are uncontaminated. This one shows you a proposal first,
which anchors you. It is roughly 3x faster and strictly weaker evidence.
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse
import json
from collections import Counter
from pathlib import Path

from taxonomy import ESCALATION_POLICY, INTENTS, LABELS

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data" / "golden_candidates.jsonl"
SEED = ROOT / "data" / "model_labels.jsonl"
GOLD = ROOT / "data" / "golden.jsonl"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def parse(raw, intent, escalate):
    """Return (intent, escalate, ok). Empty input accepts the proposal."""
    raw = raw.strip().lower()
    if not raw:
        return intent, escalate, True
    for tok in (raw[i] for i in range(len(raw))):
        if tok.isdigit() and 1 <= int(tok) <= len(LABELS):
            intent = LABELS[int(tok) - 1]
        elif tok == "a":
            escalate = False
        elif tok == "e":
            escalate = True
        else:
            return intent, escalate, False
    return intent, escalate, True


def save(done, cands):
    order = [c["thread_id"] for c in cands if c["thread_id"] in done]
    with open(GOLD, "w", encoding="utf-8") as f:
        for tid in order:
            f.write(json.dumps(done[tid], ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    cands = read_jsonl(CAND)
    seeds = {r["thread_id"]: r for r in read_jsonl(SEED)}
    done = {r["thread_id"]: r for r in read_jsonl(GOLD)}

    if args.stats:
        audited = [r for r in done.values() if r.get("protocol") == "audited"]
        changed = sum(r.get("changed_from_model", False) for r in audited)
        print(f"labelled {len(done)}/{len(cands)}  (audited: {len(audited)})")
        print(Counter(r["intent"] for r in done.values()).most_common())
        print("escalate rate:",
              round(sum(r["escalate"] for r in done.values()) / max(len(done), 1), 3))
        if audited:
            print(f"correction rate: {changed}/{len(audited)} "
                  f"= {changed / len(audited):.1%}")
        return

    if not seeds:
        sys.exit("No data/model_labels.jsonl -- run: python src/prelabel.py")

    todo = [c for c in cands if c["thread_id"] not in done]
    if not todo:
        print(f"All {len(done)} done. Run --stats.")
        return

    print(ESCALATION_POLICY)
    print("Enter = accept.  '5' = intent 5.  'e' = escalate.  '5e' = both.  s/q.\n")

    for c in todo:
        seed = seeds.get(c["thread_id"])
        if not seed:
            continue
        intent, escalate = seed["intent"], bool(seed["escalate"])

        print("=" * 72)
        for i, k in enumerate(LABELS, 1):
            print(f"  {i}  {k:22} {INTENTS[k][:44]}")
        print("=" * 72)
        print(f"[{len(done)}/{len(cands)}]  slice={c['slice']}")
        print(f"\nCUSTOMER: {c['customer_msg']}")
        print(f"\n(spotify actually said: {c['brand_reply'][:160]})")
        print(f"\nPROPOSED: {intent}  /  "
              f"{'ESCALATE' if escalate else 'auto-handle'}")

        while True:
            raw = input("enter=accept, or correct: ").strip().lower()
            if raw == "q":
                save(done, cands)
                print(f"saved {len(done)} labels.")
                return
            if raw == "s":
                intent = None
                break
            new_i, new_e, ok = parse(raw, intent, escalate)
            if ok:
                changed = (new_i != intent) or (new_e != escalate)
                intent, escalate = new_i, new_e
                break
            print("  ? use a digit 1-8 and/or a/e, e.g. '5', 'e', '5e'")

        if intent is None:
            continue

        done[c["thread_id"]] = {
            "thread_id": c["thread_id"], "slice": c["slice"],
            "customer_msg": c["customer_msg"], "brand_reply": c["brand_reply"],
            "intent": intent, "escalate": escalate, "note": "",
            "protocol": "audited", "changed_from_model": changed,
            "model_intent": seed["intent"], "model_escalate": bool(seed["escalate"]),
        }
        save(done, cands)

    save(done, cands)
    print(f"\ndone. {len(done)} labels in {GOLD}")


if __name__ == "__main__":
    assert parse("", "content_request", False) == ("content_request", False, True)
    assert parse("5e", "other", False)[:2] == ("content_request", True)
    assert parse("z", "other", False)[2] is False
    main()
