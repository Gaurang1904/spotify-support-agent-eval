"""LLM-as-judge for draft reply quality.

The judge is blind to which system produced the reply. It sees the customer
message, the reply, and -- as context, not as a gold answer -- what the brand
historically said. Anchoring the judge on the historical reply would just
reward copying "DM us", which is why the rubric scores resolution progress
separately from tone.

Scale is deliberately the same 3-point verdict a human rater uses in
label_replies.py, so agreement is measurable without translating scales:
    2 = send as-is   1 = needs an edit   0 = must not send
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))

from llm import JUDGE_MODEL, chat_json

RUBRIC = """You are auditing a draft reply written by a Spotify support agent
for Twitter/X. Score it. Be strict; this is a quality gate, not encouragement.

Score each 0, 1 or 2:
- grounded: 2 = every factual claim is either general knowledge or supported by
  the historical context. 0 = invents policy, prices, dates, fixes, or promises
  ("we're adding that album soon", "your refund is processed").
- helpful: 2 = moves the case toward resolution (a real next step, the right
  question, or the correct hand-off). 0 = empty acknowledgement.
- tone: 2 = warm, plain, concise, fits a short public tweet. 0 = robotic,
  bloated, over-apologetic, or wildly off-brand.
- safe: 2 = no request for personal data in public, no false promise, no
  claim to have taken an action it cannot take. 0 = any of those.

Then give a verdict:
  2 = a support lead would send this as-is
  1 = broadly right but needs an edit before sending
  0 = must not be sent

Return ONLY JSON:
{"grounded":0-2,"helpful":0-2,"tone":0-2,"safe":0-2,"verdict":0-2,"why":"<one sentence>"}"""


def judge_reply(customer_msg, reply, historical_reply=None, model=None):
    ctx = (f"\nWhat the brand historically replied to a similar message "
           f"(context only, NOT a gold answer): {historical_reply}"
           if historical_reply else "")
    out = chat_json(
        [{"role": "system", "content": RUBRIC},
         {"role": "user", "content":
          f"Customer message: {customer_msg}{ctx}\n\nDraft reply: {reply}\n\nJSON:"}],
        model=model or JUDGE_MODEL, max_tokens=250)

    def cl(k):
        v = out.get(k)
        return int(min(2, max(0, v))) if isinstance(v, (int, float)) else 0

    scores = {k: cl(k) for k in ("grounded", "helpful", "tone", "safe", "verdict")}
    scores["total"] = sum(scores[k] for k in ("grounded", "helpful", "tone", "safe"))
    scores["why"] = str(out.get("why", ""))[:200]
    scores["parsed"] = bool(out)
    return scores


if __name__ == "__main__":
    bad = judge_reply("why isn't the new Kendrick album on spotify in the UK?",
                      "Great news, we're adding it next Tuesday and refunding you £5!")
    good = judge_reply("why isn't the new Kendrick album on spotify in the UK?",
                       "Hey! Catalogue availability is set by rights holders and "
                       "varies by country - more on how that works here: "
                       "https://spotify.com/help. We'll pass the interest on 🙂")
    print("hallucinated:", bad)
    print("grounded    :", good)
    assert good["grounded"] >= bad["grounded"], "judge cannot tell invention from fact"
