# Decision log

15 non-obvious decisions and why. Ordered by how much they shaped the result,
which puts the ones the data forced on me first and the up-front design choices
last.

Decisions that produced *findings* rather than design changes — the judge
failing its validity check, the human-ceiling inversion, the iPhone X sampling
skew, the underpowered null — live in the report, where they are evidence
rather than bookkeeping.

---

1. **I ran a model-assisted labelling protocol, measured that it failed, and
   threw the labels away.** `audit_tool.py` pre-labels each example and asks a
   human to accept or correct it. Over the first 51 rows the recorded
   correction rate was **0.0%** — while a spot-check of those same rows found
   four clear errors (two "my account was hacked" tweets filed as
   `billing_subscription` because the tweet also said "premium"; a Spanish
   family-plan request auto-handled against my own always-escalate policy; a
   press enquiry filed as `praise_chatter`). The flag was measuring the accept
   key, not judgement. Worse, the pre-labeller (qwen2.5:7b) is also the judge,
   which would have made "judge agrees with human" a measurement of qwen
   against qwen. The 51 rows are kept in `data/discarded_audit_labels.jsonl` as
   evidence rather than deleted, and the golden set was rebuilt blind.

2. **Intent is labelled blind; the brand's actual reply is revealed only after
   the intent is committed.** My first version showed it up front, reasoning
   that "the brand asked for a DM" is evidence a human was needed. That was
   wrong for the intent label: these labels are the ground truth every
   agreement number is measured against, so anchoring them to Spotify's routing
   habits would have inflated all of them at once with no way to detect it.
   Split the difference — blind for intent, revealed for the auto-vs-escalate
   call where it is evidence rather than contamination. Model pre-labels are
   never shown.

3. **`ambiguous` is a first-class label, not a skip.** Some tweets cannot be
   classified from the text alone. Forcing a guess injects noise directly into
   the ground truth; recording it makes that noise a published rate (2.0%).
   Ambiguous rows are excluded from scoring and the harness prints the share.

4. **The dead guardrail stays in the code, documented as dead.** The router
   escalates when self-reported confidence drops below 0.5. Across 147 examples
   it fired twice, both on JSON parse failures another rule already caught, and
   the signal is inverted where it is most assertive (accuracy 0.167 at
   confidence 1.0 against 0.704 at 0.9). I kept the rule and wrote down that it
   does nothing, because "we shipped a safety rule that never fires" is the
   finding; deleting it would have destroyed the evidence.

5. **The labelling tool records seconds per item.** A median time is the
   cheapest available evidence that a set was read rather than clicked through
   — and if the median had come out at three seconds, the report would have had
   to say so. It was 12s over 44 minutes.

6. **Golden set cut from 220 to 150, entirely out of the stratified slice.**
   150 is the brief's floor. The random slice stayed at 50 because it is the
   only unbiased estimator of live performance; shrinking it to save effort
   would have cost the one number in the report that is not flattering.

7. **No `how_to` intent — "how do I..." questions route to `playback_error`.**
   Found while labelling: "How do you clear the queue on the desktop app?" is
   not a fault, so the class definition technically excludes it, but `other` is
   for junk. I kept 8 classes and stated the rule, because the taxonomy was
   merged on "would an agent do the same thing for these?" and the answer is
   the same either way: link the help article, auto-handle. The cost is that
   `playback_error` is really "anything about using the player".

8. **Mid-conversation tweets are left in the corpus, and I measured how many.**
   A thread root is an inbound tweet with no parent in the subgraph — but if
   the real parent was deleted or the exchange began in DMs, a mid-conversation
   tweet becomes a root and the brand's reply answers context the corpus does
   not contain. Keyword probe: **3.4% of the corpus, 5.9% of the golden
   candidates.** Left in rather than filtered, because they are real production
   traffic; an agent that only works on clean first contact is not the thing
   being evaluated. It does cap how high any reply metric can honestly go.

9. **Brand = SpotifyCares, chosen by measuring deflection before committing.**
   52% of AppleSupport replies and 82% of TMobileHelp replies are pure "DM us",
   leaving almost no resolution text to ground on. SpotifyCares is 31%
   deflection with a help-article link in 50% of replies, at 43k replies.
   AmazonHelp has more volume but near-templated replies and logistics-generic
   intents.

10. **The escalation policy is anchored to what the brand actually did.** Every
    intent marked always-escalate is one SpotifyCares itself answered with "DM
    us your account email" — a human working backstage. That makes the policy
    defensible from data rather than from my opinion, and gives the routing
    metric a referent that is not just my taste.

11. **The golden set is two disjoint slices: 100 stratified + 50 uniform
    random.** Stratification is the only way to get `family_plan` and web-player
    cases represented at all, but it inflates per-class averages relative to
    production traffic. The random slice is the honest estimator. Both are
    reported, and the gap between them is a section of the report.

12. **Golden thread ids are excluded from the retrieval index.** Otherwise the
    agent retrieves the exact case it is being scored on and the reply metric
    measures copying rather than generalisation.

13. **Deterministic guardrails override the model on routing.** PII in the
    message, unparseable output, an account-bound intent, or zero retrieval
    hits all force escalation regardless of what the model said. A 3B model's
    `escalate` field is not something to bet a customer's billing dispute on.
    This is also the cause of the agent's dominant error — see the report's
    over-escalation failure mode. The guardrail trades precision for safety on
    purpose.

14. **The judge is a bigger model than the agent, and shares the human's
    scale.** qwen2.5:7b judges llama3.2:3b — a judge no stronger than the
    system it grades is a rubber stamp. It scores on the same 3-point verdict
    a human rater uses, so agreement is a direct kappa with no scale
    translation. The brand's historical reply is given to the judge as
    *context, not gold*; anchoring on it would have rewarded copying "DM us".

15. **The simple baseline is deliberately advantaged.** It trains on the golden
    labels via 5-fold cross-validation while the agent is zero-shot. If a bag
    of words with access to the answers matches the LLM, that is worth knowing
    — and it did.
