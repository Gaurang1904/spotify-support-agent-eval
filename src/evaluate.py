"""Evaluation harness: 3 systems x 3 tasks, on the hand-labelled golden set.

    python src/evaluate.py                 # full run
    python src/evaluate.py --limit 30      # smoke run
    python src/evaluate.py --no-judge      # metrics only, no LLM judging

Writes outputs/predictions.jsonl and outputs/metrics.json, prints the table
that goes in the report.
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score

import baselines
from agent import Agent
from judge import judge_reply
from retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "golden.jsonl"
OUTD = ROOT / "outputs"


def load_gold(limit=None, path=GOLD):
    """Rows the human marked `ambiguous` are dropped from scoring and reported
    as a rate. Forcing a guess on an unreadable tweet would put noise straight
    into the ground truth; excluding them makes that noise measurable instead."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    usable = [r for r in rows if r.get("intent") != "ambiguous"]
    if len(usable) < len(rows):
        print(f"ambiguous rows excluded: {len(rows) - len(usable)}/{len(rows)} "
              f"= {(len(rows) - len(usable)) / len(rows):.1%}")
    return usable[:limit] if limit else usable


def intent_metrics(y_true, y_pred):
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    return {
        "accuracy": round(acc, 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0), 3),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 3),
    }


def route_metrics(y_true, y_pred):
    """`escalate=True` is the positive class. The error that actually costs
    money is a missed escalation: the bot auto-answered a case a human needed."""
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "escalation_precision": round(prec, 3),
        "escalation_recall": round(rec, 3),
        "escalation_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "missed_escalations": fn,
        "missed_escalation_rate": round(fn / len(y_true), 3),
        "auto_handled": tn + fn,
        "automation_rate": round((tn + fn) / len(y_true), 3),
    }


def run_systems(gold, retriever):
    msgs = [g["customer_msg"] for g in gold]
    y_intent = [g["intent"] for g in gold]

    out = {"trivial": baselines.trivial(msgs, y_intent),
           "simple": baselines.simple(msgs, y_intent, retriever)}

    agent = Agent(retriever)
    res = [agent.handle(m) for m in msgs]
    out["agent"] = {
        "intent": [r["intent"] for r in res],
        "escalate": [r["escalate"] for r in res],
        "reply": [r["reply"] for r in res],
        "reason": [r["reason"] for r in res],
        "confidence": [r["confidence"] for r in res],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gold", default=str(GOLD), help="labels file to evaluate against")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        sys.exit(f"No {gold_path} -- run: python src/label_tool.py")

    gold = load_gold(args.limit, gold_path)
    retriever = Retriever(exclude_ids=[g["thread_id"] for g in gold])
    print(f"golden={len(gold)}  retrieval index={len(retriever.docs)}")

    preds = run_systems(gold, retriever)
    y_intent = [g["intent"] for g in gold]
    y_route = [g["escalate"] for g in gold]
    rand = [i for i, g in enumerate(gold) if g["slice"] == "random"]

    metrics = {"n": len(gold), "n_random_slice": len(rand), "systems": {}}
    for name, p in preds.items():
        m = {"intent": intent_metrics(y_intent, p["intent"]),
             "routing": route_metrics(y_route, p["escalate"])}
        if rand:
            m["intent_random_slice"] = intent_metrics(
                [y_intent[i] for i in rand], [p["intent"][i] for i in rand])
        metrics["systems"][name] = m

    if not args.no_judge:
        for name, p in preds.items():
            print(f"judging {name}...", flush=True)
            scored = [judge_reply(g["customer_msg"], r, g["brand_reply"])
                      for g, r in zip(gold, p["reply"])]
            p["judge"] = scored
            n = len(scored)
            metrics["systems"][name]["reply"] = {
                **{k: round(sum(s[k] for s in scored) / n, 2)
                   for k in ("grounded", "helpful", "tone", "safe", "total", "verdict")},
                "pct_sendable": round(sum(s["verdict"] == 2 for s in scored) / n, 3),
                "pct_must_not_send": round(sum(s["verdict"] == 0 for s in scored) / n, 3),
                "judge_parse_fail": sum(not s["parsed"] for s in scored),
            }

    OUTD.mkdir(exist_ok=True)
    with open(OUTD / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, g in enumerate(gold):
            row = {"thread_id": g["thread_id"], "slice": g["slice"],
                   "customer_msg": g["customer_msg"],
                   "gold_intent": g["intent"], "gold_escalate": g["escalate"],
                   "brand_reply": g["brand_reply"]}
            for name, p in preds.items():
                row[name] = {k: v[i] for k, v in p.items()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # A --no-judge run must not clobber a full run's metrics: it would silently
    # delete every reply score and the next reader would not know why.
    out_name = "metrics_nojudge.json" if args.no_judge else "metrics.json"
    (OUTD / out_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    hdr = f"{'system':10}{'intent acc':>12}{'macro F1':>10}{'esc recall':>12}{'missed esc':>12}{'auto %':>9}{'reply/8':>9}{'sendable':>10}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, m in metrics["systems"].items():
        r = m.get("reply", {})
        print(f"{name:10}{m['intent']['accuracy']:>12.3f}{m['intent']['macro_f1']:>10.3f}"
              f"{m['routing']['escalation_recall']:>12.3f}"
              f"{m['routing']['missed_escalations']:>12d}"
              f"{m['routing']['automation_rate']:>9.3f}"
              f"{r.get('total', float('nan')):>9.2f}{r.get('pct_sendable', float('nan')):>10.3f}")
    print(f"\ngold intent distribution: {Counter(y_intent).most_common()}")
    print(f"wrote {OUTD/out_name} and {OUTD/'predictions.jsonl'}")


if __name__ == "__main__":
    main()
