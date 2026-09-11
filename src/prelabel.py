"""Model pre-labels for the candidate pool.

NOT the golden set. Two uses:
  1. Bootstrap: lets the pipeline be smoke-tested before a human has labelled.
  2. Second annotator: comparing these to the human labels gives a
     model-vs-human intent agreement number, which is the closest thing to an
     inter-annotator score available with a single human rater.

Uses the larger judge model, not the agent model, so it is not simply the
agent grading its own homework.

    python src/prelabel.py
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
from pathlib import Path

from llm import JUDGE_MODEL, chat_json
from taxonomy import ESCALATION_POLICY, INTENTS, LABELS

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data" / "golden_candidates.jsonl"
OUT = ROOT / "data" / "model_labels.jsonl"

PROMPT = f"""Label this Spotify support tweet.

Intents:
{chr(10).join(f"- {k}: {v}" for k, v in INTENTS.items())}

{ESCALATION_POLICY}

Return ONLY JSON: {{"intent":"<label>","escalate":true|false}}"""


def main():
    rows = [json.loads(l) for l in open(CAND, encoding="utf-8")]
    with open(OUT, "w", encoding="utf-8") as f:
        for i, c in enumerate(rows, 1):
            out = chat_json([{"role": "system", "content": PROMPT},
                             {"role": "user", "content": c["customer_msg"]}],
                            model=JUDGE_MODEL, max_tokens=80)
            intent = out.get("intent") if out.get("intent") in LABELS else "other"
            f.write(json.dumps({
                "thread_id": c["thread_id"], "slice": c["slice"],
                "customer_msg": c["customer_msg"], "brand_reply": c["brand_reply"],
                "intent": intent, "escalate": bool(out.get("escalate", True)),
                "note": "model prelabel",
            }, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
