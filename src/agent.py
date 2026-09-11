"""The support agent: classify -> retrieve -> draft -> route.

One LLM call per message. The retrieved history is injected as exemplars, so
the model is grounded in how SpotifyCares actually resolved similar cases
rather than inventing Spotify policy.
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
import json
import re
from pathlib import Path

from llm import chat_json
from retrieval import Retriever
from taxonomy import ESCALATION_POLICY, INTENTS, LABELS, policy_escalates

PII = re.compile(
    r"(__email__|[\w.+-]+@[\w-]+\.\w+|\b(?:\d[ -]?){13,16}\b|\b\d{10,}\b)", re.I)

SYSTEM = f"""You are a customer support agent for Spotify, replying on Twitter/X.

Classify the customer's message into exactly one intent:
{chr(10).join(f"- {k}: {v}" for k, v in INTENTS.items())}

{ESCALATION_POLICY}

Reply rules, taken from how SpotifyCares actually writes:
- 1 to 3 sentences, under 280 characters, warm and plain, at most one emoji.
- Never invent policy, prices, dates, or refunds. If the answer needs account
  data you do not have, say what you need instead of guessing.
- If escalating, the reply should hand off gracefully; do not promise outcomes.
- Never ask the customer to post personal data publicly.

Return ONLY a JSON object:
{{"intent": "<one label>", "confidence": <0.0-1.0>,
  "escalate": true|false, "reason": "<one short sentence>",
  "reply": "<the draft reply>"}}"""

# Candidate fix for the grounding failure (report failure mode 1), kept OFF by
# default so the evaluated system stays frozen. Measured by grounding_ablation.py.
GROUNDING_PATCH = """

CRITICAL — the past cases above are FACTS about what this brand can and cannot
do, not writing samples. They outrank anything you believe:
- If they say the brand does NOT have something, the brand does not have it.
  Never tell the customer it is available.
- If one of them answers the question, use that answer, including any link.
- If they do not cover the question, say what you do not know. Do not fill the
  gap from your own knowledge."""


def _exemplars(hits):
    if not hits:
        return "(no similar past case found)"
    return "\n\n".join(
        f"Past customer: {d['customer_msg']}\nSpotify replied: {d['first_reply']}"
        for _, d in hits)


class Agent:
    def __init__(self, retriever: Retriever, k: int = 3, model: str | None = None,
                 grounding_patch: bool = False):
        self.r, self.k, self.model = retriever, k, model
        self.system = SYSTEM + (GROUNDING_PATCH if grounding_patch else "")

    def handle(self, message: str) -> dict:
        hits = self.r.search(message, k=self.k)
        user = (f"Similar past cases from this brand's history:\n{_exemplars(hits)}\n\n"
                f"---\nNew customer message: {message}\n\nJSON:")
        out = chat_json(
            [{"role": "system", "content": self.system},
             {"role": "user", "content": user}],
            model=self.model, max_tokens=400)

        intent = out.get("intent") if out.get("intent") in LABELS else "other"
        conf = out.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else 0.0
        reply = str(out.get("reply") or "").strip()[:400]
        escalate = bool(out.get("escalate"))
        reason = str(out.get("reason") or "").strip()

        # Hard guardrails the model does not get a vote on. These are safety
        # rails, not accuracy tricks -- each maps to a line of the policy.
        if PII.search(message):
            escalate, reason = True, "Customer posted personal data publicly."
        elif not out:
            escalate, reason = True, "Model returned unparseable output."
        elif intent in ("account_access", "billing_subscription", "family_plan"):
            if not escalate:
                escalate = True
                reason = f"{intent} needs account access the agent does not have."
        elif conf < 0.5 and not escalate:
            escalate, reason = True, "Low classification confidence."
        elif not hits and not escalate and policy_escalates(intent):
            escalate, reason = True, "No similar resolved case in brand history."

        return {"intent": intent, "confidence": conf, "escalate": escalate,
                "reason": reason or "n/a", "reply": reply,
                "retrieved": [{"sim": round(s, 3), "thread_id": d["thread_id"],
                               "reply": d["first_reply"]} for s, d in hits]}


def show(out, show_sources=False):
    print(f"\n  intent    : {out['intent']}  (confidence {out['confidence']})")
    print(f"  decision  : {'ESCALATE' if out['escalate'] else 'auto-handle'}")
    print(f"  reason    : {out['reason']}")
    print(f"  reply     : {out['reply'] or '(empty)'}")
    if show_sources:
        for r in out["retrieved"]:
            print(f"    [{r['sim']}] {r['reply'][:90]}")


if __name__ == "__main__":
    # python src/agent.py                 -> self-check on two fixed messages
    # python src/agent.py "some tweet"    -> answer that one message
    # python src/agent.py -i              -> interactive, one tweet per line
    args = sys.argv[1:]
    a = Agent(Retriever())

    if args and args[0] in ("-i", "--interactive"):
        print("Type a customer tweet. Blank line or Ctrl-C to quit.\n")
        while True:
            try:
                msg = input("customer> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg:
                break
            show(a.handle(msg), show_sources=True)
            print()
    elif args:
        show(a.handle(" ".join(args)), show_sources=True)
    else:
        for m in ["my premium got charged twice this month, i want a refund",
                  "why isn't the new Kendrick album on spotify in the UK?"]:
            print(f"\nIN : {m}")
            out = a.handle(m)
            show(out)
        assert out["intent"] in LABELS
        print("\nself-check ok")
