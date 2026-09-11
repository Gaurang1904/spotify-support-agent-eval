"""Two baselines the agent has to beat.

trivial -- the "would a cron job do this?" floor:
    intent  = always the majority class of the training fold
    route   = always escalate (a real, and sadly common, support policy)
    reply   = one fixed canned string

simple -- a strong non-LLM system:
    intent  = TF-IDF + logistic regression, trained on the golden labels
              (5-fold cross-validated, so it never predicts its own row)
    route   = intent-prior rule from taxonomy.py
    reply   = verbatim copy of the historical reply to the most similar past case

Note the simple baseline is *advantaged*: it trains on the same hand labels it
is scored against, while the agent is zero-shot. See the report.
"""
import sys, pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parent))
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from taxonomy import policy_escalates

CANNED = ("Hey there, help's here! Can you DM us your account's email address "
          "and username? We'll take a look backstage 🙂")


def trivial(messages, y_intent):
    """Majority intent, always escalate, one canned reply."""
    top = Counter(y_intent).most_common(1)[0][0]
    return {
        "intent": [top] * len(messages),
        "escalate": [True] * len(messages),
        "reply": [CANNED] * len(messages),
        "reason": ["Fixed policy: everything goes to a human."] * len(messages),
    }


def simple(messages, y_intent, retriever, folds=5):
    """Cross-validated TF-IDF + logistic regression, plus nearest-neighbour reply."""
    clf = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1,
                        stop_words="english"),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    # Classes with fewer members than `folds` cannot be stratified; drop the
    # fold count to the rarest class size rather than silently reshuffling.
    n = min(folds, min(Counter(y_intent).values()))
    if n < 2:
        pred = [Counter(y_intent).most_common(1)[0][0]] * len(messages)
    else:
        pred = list(cross_val_predict(
            clf, messages, y_intent,
            cv=StratifiedKFold(n, shuffle=True, random_state=0)))

    replies, reasons = [], []
    for m, p in zip(messages, pred):
        hits = retriever.search(m, k=1)
        replies.append(hits[0][1]["first_reply"] if hits else CANNED)
        reasons.append(f"Rule: intent '{p}' is "
                       f"{'not ' if not policy_escalates(p) else ''}account-bound.")
    return {
        "intent": pred,
        "escalate": [policy_escalates(p) for p in pred],
        "reply": replies,
        "reason": reasons,
    }


if __name__ == "__main__":
    msgs = ["charged twice", "add this song", "cant log in", "app crashes",
            "charged again", "add another song", "login broken", "app freezes"]
    ys = ["billing_subscription", "content_request", "account_access",
          "playback_error"] * 2
    t = trivial(msgs, ys)
    assert len(set(t["intent"])) == 1 and all(t["escalate"])
    print("trivial ok:", t["intent"][0])
