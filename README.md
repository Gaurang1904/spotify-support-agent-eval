# Spotify support agent — Hiver SDE intern take-home

An AI support agent for **@SpotifyCares**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming customer message into one of 8 intents,
drafts a reply grounded in how Spotify historically resolved similar cases, and
decides whether to auto-handle or escalate — with a stated reason.

The interesting part is the evaluation, not the agent. See
[Report](#report) and [reports/DECISIONS.md](reports/DECISIONS.md).

---

## Reproduce the headline numbers (< 15 min)

Runs entirely on a local model via **Ollama** — no API key, no cost. Cached LLM
responses ship with the repo, so the headline table is a re-read, not a re-run.

```bash
pip install -r requirements.txt
```

```bash
mkdir -p data/raw && curl -L -o data/raw/twcs.csv https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv
```

```bash
ollama pull llama3.2:3b && ollama pull qwen2.5:7b-instruct && cp .env.example .env
```

```bash
bash run_all.sh
```

On Windows PowerShell use `.
un_all.ps1` instead.

Step 2 is a 516 MB download (~2 min) — a Hugging Face mirror of the Kaggle csv,
so no Kaggle login is needed. Step 3 pulls the agent model and the judge model;
skip it if you point `.env` at OpenAI or Anthropic instead.

`run_all.sh` = `prep.py` → `sample_golden.py` → `evaluate.py` → `agreement.py`.
Outputs land in `outputs/metrics.json`, `outputs/predictions.jsonl`,
`outputs/agreement.json`.

**Any OpenAI-compatible endpoint works.** Edit `.env`:

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
JUDGE_MODEL=gpt-4.1
```

Delete `outputs/llm_cache.db` to force fresh generations (~50 min on an
RTX 4060 for the full 150-example run).

### Rebuild the golden set from scratch

```bash
python src/label_tool.py     # or: python src/audit_tool.py (faster, weaker)
python src/evaluate.py
python src/label_replies.py --n 60
python src/agreement.py
```

`label_tool.py` walks the 150 candidates asking for intent + route and is
resumable. `label_replies.py` shows drafted replies with the system name hidden
and asks for a verdict on the same 3-point scale the judge uses.

---

## What is in here

| Path | What it is |
|---|---|
| `src/prep.py` | Raw csv → `data/threads.jsonl` (28,197 SpotifyCares cases) |
| `src/taxonomy.py` | The 8 intents + the escalation policy, both derived from data |
| `src/sample_golden.py` | Stratified + random candidate sampling, frozen seed |
| `src/label_tool.py` | Resumable CLI for hand-labelling the golden set, intent asked blind |
| `src/audit_tool.py` | Faster model-assisted protocol: accept/correct pre-labels, records correction rate |
| `src/retrieval.py` | TF-IDF nearest-neighbour over resolved brand history |
| `src/agent.py` | classify → retrieve → draft → route, one LLM call + guardrails |
| `src/baselines.py` | Trivial and simple baselines |
| `src/judge.py` | LLM-as-judge rubric (4 dimensions + a 3-point verdict) |
| `src/label_replies.py` | Blind human rating of replies, same scale as the judge |
| `src/agreement.py` | Judge-vs-human kappa, Spearman, directional bias |
| `src/evaluate.py` | The harness — every number in the report comes from here |
| `src/human_ceiling.py` | Scores the brand's OWN replies with the same judge — the control that reframed the headline |
| `src/significance.py` | Paired bootstrap CIs on the agent-vs-simple deltas |
| `src/error_analysis.py` | Pulls real failures out of the predictions file |
| `src/prelabel.py` | Model pre-labels — a second annotator, *not* the golden set |
| `data/golden.jsonl` | The hand-labelled golden set |
| `data/discarded_audit_labels.jsonl` | A failed model-assisted labelling run, kept as evidence (decision 1) |
| `reports/DECISIONS.md` | The 15-item decision log |

---

## Report

### Problem framing: what "good" means for SpotifyCares

Spotify's Twitter support is a **triage desk, not a resolution desk**. Roughly a
third of its replies are "DM us your account email"; the agent has no account
access, no billing system, no ability to issue a refund. So the job is not
"resolve the ticket". The job is:

- put the message in the right lane,
- answer the ones that are genuinely answerable in public — catalogue
  questions, feature feedback, standard troubleshooting — without inventing
  anything,
- and hand the rest to a human *cleanly and early*, with a reason a support
  lead can audit.

That makes the cost function asymmetric. Over-escalating wastes an agent's
minute. Under-escalating means a bot confidently tells someone their refund is
processed. **Missed escalations are the metric that matters**, and I report
them as a raw count, not folded into an F1.

#### What I deliberately did not build

- **No multi-turn dialogue.** The agent sees the opening message only. Every
  case in this dataset starts with one public tweet; conversation state moves
  into DMs the dataset does not contain.
- **No fine-tuning.** With 150 labels, a fine-tune measures the golden set, not
  the task.
- **No embeddings / vector DB.** TF-IDF over 28k short tweets is competitive
  here, keeps product names and misspellings a small embedding model
  smooths away, and is one dependency lighter.
- **No sentiment or priority model.** Anger matters only insofar as it changes
  routing, and the escalation policy already covers it.
- **No PII scrubbing pipeline.** The dataset is already redacted to
  `__email__`; the agent detects the residue and escalates rather than rewriting
  it.
- **No production serving.** No API, no queue, no retry semantics — scaffolding
  around an unproven core.

### The taxonomy

TF-IDF + KMeans (k=14) over 28,197 opening messages, then merged by hand on the
test *"would a support agent do the same thing for these?"*:

`playback_error` · `account_access` · `billing_subscription` · `family_plan` ·
`content_request` · `feature_feedback` · `praise_chatter` · `other`

The escalation policy is anchored to the brand's own behaviour: every intent
marked always-escalate is one Spotify itself answered with "DM us your account
email" — a human working backstage (decision 10).

### The golden set

150 examples, hand-labelled blind with intent **and** route, in two disjoint slices:

- **stratified (100)** — proportional across the 14 clusters with a floor of 8,
  so `family_plan` and `web player` are actually present;
- **random (50)** — a uniform draw, untouched by stratification.

Golden thread ids are excluded from the retrieval index so the agent cannot
retrieve the case it is being scored on.

### Baselines

| | intent | routing | reply |
|---|---|---|---|
| **trivial** | majority class | always escalate | one canned string |
| **simple** | TF-IDF + logistic regression, 5-fold CV | intent-prior rule | verbatim reply of the nearest past case |
| **agent** | LLM, zero-shot | LLM + deterministic guardrails | LLM grounded on top-3 retrieved cases |

The simple baseline is deliberately advantaged — it trains on the golden labels
the agent never sees.

### Results

150 hand-labelled examples, 3 marked `ambiguous` and excluded, **n = 147**
(random slice n = 48). Every number below comes from `src/evaluate.py`.

| system | intent acc | macro F1 | kappa | esc. recall | missed esc. | automation | reply /8 | sendable |
|---|---|---|---|---|---|---|---|---|
| trivial | 0.224 | 0.046 | 0.00 | 1.000 | 0 | 0.0% | 1.31 | 2.0% |
| simple | 0.544 | 0.497 | 0.45 | 0.905 | 4 | 30.6% | 3.37 | 4.8% |
| **agent** | **0.585** | **0.496** | **0.51** | 0.786 | 9 | 57.8% | **4.52** | **12.9%** |
| *human (SpotifyCares)* | — | — | — | — | — | — | *4.52* | *8.2%* |

**The agent does not beat the simple baseline.** Paired bootstrap, 10,000
resamples (`src/significance.py`):

| delta (agent − simple) | observed | 95% CI | verdict |
|---|---|---|---|
| accuracy | +0.041 | [−0.068, +0.150] | no measured difference |
| macro F1 | **−0.0004** | [−0.131, +0.119] | no measured difference |
| missed escalations | +5 | [−1, +11] | no measured difference |

At n = 147, an LLM with retrieval and a TF-IDF + logistic-regression classifier
are **statistically indistinguishable** on every intent and routing metric. The
one thing the agent demonstrably buys is automation rate — 57.8% against 30.6%
— and that is a threshold choice, not model quality.

**This is a null, not a proof of equivalence, and the distinction matters.**
The accuracy interval is [−0.068, +0.150]: half-width ~0.11, so this test can
only resolve differences larger than about 11 accuracy points. A real 5-point
advantage for the agent would be invisible at this sample size. The honest
claim is *"no difference detectable at n = 147"*, not *"no difference exists"*.
Detecting a 5-point effect at this variance would need roughly 800-1000
labelled examples — five to seven times the golden set, which is the main
argument for spending the next week labelling rather than modelling.

Both beat the trivial baseline decisively, which mostly proves the task is
non-trivial rather than that either system is good.

**Reproducibility.** Every number above comes from the run that the shipped
`outputs/llm_cache.db` reproduces exactly. Ollama is not deterministic even at
`temperature=0`, so a reviewer who deletes the cache and regenerates should
expect roughly one example in 147 to flip: observed spread across runs was
accuracy 0.585-0.592, macro F1 0.496-0.500, reply score 4.52-4.56. That is far
inside every confidence interval here, but it is not zero and it is not hidden.

### Judge validity

The judge was checked against 60 replies rated blind by a human on the same
3-point scale (`src/label_replies.py`, `src/agreement.py`). It failed.

| metric | value |
|---|---|
| exact agreement | 0.333 |
| quadratic-weighted kappa | **0.100** |
| Spearman rho | 0.176 |
| judge bias | −0.75 (judge 0.52 vs human 1.27) |

Kappa at 0.10 is chance. The usual fallback — *"the judge is harsh but
consistent, so it still ranks systems correctly"* — does not hold either:

| system | human mean | judge mean |
|---|---|---|
| trivial | 1.17 | 0.17 |
| agent | 1.18 | 0.82 |
| simple | **1.40** | 0.56 |

The human ranks simple first; the judge ranks agent first. **The ordering is
inverted**, so no reply-quality claim in this report rests on the judge.

A second, independent control says the same thing. SpotifyCares' own human
replies, scored by the identical rubric (`src/human_ceiling.py`), get **4.52/8
with 8.2% sendable, and 35.4% rated "must not send."** The agent scores **4.52/8
with 12.9% sendable** -- identical on the mean, better on sendability. So
"12.9% of replies are sendable" was never a fact
about the agent — it is a fact about the rubric.

Caveat: n = 60, split 18/17/25 across systems, so the per-system means are
within noise. The kappa is the load-bearing number; the ranking inversion is
strong but not conclusive.

**Why this generalises beyond one project.** LLM-as-judge is the default way
teams evaluate support-reply quality, and it is usually deployed without a
human-agreement check or a human-ceiling control. This project ran both, on
real production support data, and the method failed both: chance-level
agreement, inverted ranking, and human agents scored below a 3B model. Neither
control is expensive — 60 blind ratings and one extra scoring pass. Any team
grading generated replies with an LLM judge and no human control should assume
their numbers are measuring the rubric until they have checked.

### Failure analysis

**1. `other` is undetectable — 24 gold examples, 1 correct.**
The model has no concept of "this tweet contains no classifiable request." The
24 `other` rows scatter into `account_access` (8), `praise_chatter` (6),
`feature_feedback` (3), `content_request` (3), `billing_subscription` (3).

> "Answer my DM! Urgent, thanks." → predicted `praise_chatter`, confidence 1.0

*Hypothesis:* the prompt offers 8 positive categories and one residual, and an
instruction-tuned model asked to choose a label will always find a positive one
it likes. A `null`-first prompt, or a confidence gate before classification,
would likely fix this. It is 16% of the golden set.

**2. Confidence is anti-calibrated, so the confidence guardrail is dead code.**
The agent reports a confidence and the router escalates anything below 0.5.
It has never fired on a real case:

| stated confidence | n | actual accuracy |
|---|---|---|
| 1.0 | 12 | **0.167** |
| 0.9 | 108 | 0.704 |
| 0.8 | 25 | 0.240 |
| 0.0 (parse failure) | 2 | — |

Only 2 of 147 examples land below 0.5, and both are JSON parse failures already
caught by a different rule. Worse, the signal is **inverted at the top**: the
model is least accurate exactly when it claims certainty. 61 of 147 predictions
are wrong at confidence >= 0.8.

*Hypothesis:* a 3B instruction-tuned model asked to emit a confidence emits a
plausible-looking number, not a calibrated one — it has learned that support
replies sound confident. Any design that gates on self-reported confidence is
resting on nothing. The fix is an external signal: retrieval similarity,
classifier margin, or agreement between two prompts.

**3. Broken-app complaints read as product opinion — 10 cases,
`playback_error` -> `feature_feedback`.** The single largest confusion.

> "Please fix the desktop app? Endless spinning, everything, all platforms —
> weeks now. PLEASE FIX" -> `feature_feedback` / auto-handle, confidence 0.8

*Hypothesis:* imperative "please fix / please add" phrasing dominates the
model's decision over the symptom described. The retrieved exemplars do not
help, because past agents replied to both with the same soothing template.

**4. Over-escalation is the dominant routing error — 29 cases, 20% of the set —
and it asks for personal data in public.** Escalation precision is 0.532:
nearly half of everything sent to a human did not need to be. The deterministic
guardrail causes it — any misclassification into an account-bound intent forces
escalation, so a classification error becomes a routing error.

> Spotify's own advertisement, appearing in the inbound stream -> predicted
> `billing_subscription` -> escalate -> *"Sorry, we can't confirm the 99p offer.
> Can you DM us your account's email address?"*

The reply above requests an email address on a public timeline. This is the
failure with real-world cost, and the PII guardrail does not catch it: the
regex checks the **customer's** message for personal data and never the
**agent's own reply**.

**5. Three empty replies, and the judge hallucinated violations in all of
them.** The agent returned an empty string three times. The judge scored one
4/8 with the comment *"requests personal data in public and falsely promises
action"* — describing a reply that does not exist. This is the judge-validity failure caught in a single example, and it is why the kappa matters more
than any score the judge produced.

### What is misleading about my headline number?

The headline is "0.585 intent accuracy, 57.8% automation." Six reasons not to
believe it.

**1. It is not better than a bag of words — and I cannot prove it is not
worse.** The +0.041 accuracy gap has a 95% CI of [−0.068, +0.150]. Reporting
the point estimate as a win would be reporting noise. But the reverse
overclaim is just as available and I want to name it: **the null is
underpowered.** At n = 147 the test resolves nothing smaller than ~11 accuracy
points, so "statistically indistinguishable" means the experiment could not
see a difference, not that none is there. Anyone quoting this result as
"LLMs don't beat TF-IDF" is overreading it exactly as badly as quoting 0.585
as a win.

**2. The stratified slice flatters it.** On the uniform-random slice — the only
unbiased estimate of live traffic — the agent's macro F1 drops from **0.496 to
0.393**, a 21% fall. Stratification guaranteed rare classes a seat, which makes
per-class averages look healthier than production would.

**3. It partly measures one week of news.** Eight of 150 examples are "optimise
the app for iPhone X" — 5.3% of the golden set against 1.9% of the corpus, a
**2.8x over-representation**, 7 of the 8 in the stratified slice. TWCS was
collected as the iPhone X shipped, KMeans found the cluster, the floor rule
seated it. `feature_feedback` accuracy is partly iPhone-X accuracy.

**4. The reply scores measure the rubric, not the replies.** Judge-human kappa
is 0.100 and the judge ranks the systems in the wrong order. The same judge
rates real SpotifyCares agents *below* the bot. Every reply number in this
report should be read as "what this 7B judge thinks", not "what is good."

**5. One of the four guardrails has never fired.** The router escalates when
self-reported confidence drops below 0.5. Across 147 examples that fired twice,
both on JSON parse failures another rule already caught. The safety
architecture is thinner than it looks on paper, and the report's automation
rate is produced by three rules, not four.

**6. Accuracy is the wrong metric for the actual cost function.** The agent
auto-handled 9 cases a human needed (6.1%) — including a customer reporting a
compromised account and one whose tweet carried self-harm-adjacent language.
Meanwhile 19% of the set was escalated unnecessarily. A single accuracy figure
prices those identically. They are not.

**7. The ground truth is one person's opinion.** 150 labels, one annotator, no
inter-annotator agreement number. The taxonomy was derived by clustering, then
the same person who defined the clusters assigned the labels, and the agent
retrieves with the same TF-IDF representation the clusters came from. Some part
of 0.585 is agreement with a clustering artifact. Median labelling time was 12s
per item over 44 minutes, and 2.0% were marked ambiguous — those numbers are
published so the ground truth can be judged, not assumed.

An honest one-line summary: **an LLM support agent, evaluated properly, could
not be shown to beat logistic regression on this brand — and the instrument
built to judge its replies does not agree with a human.**

### With one more week

Ordered by what would change a conclusion, not by what is most interesting.

1. **Fix the judge, or drop it.** Kappa 0.100 makes every reply metric
   unusable. First move is a rubric rewritten against the 60 human ratings as a
   dev set, then re-measure agreement. If a rewritten rubric cannot clear ~0.6,
   report reply quality by human rating alone and delete the judge.
2. **A second annotator on 50 examples.** Every caveat in this report bottoms
   out at "one person decided." Even one more rater converts that from an
   admission into a measured bound. Failing that, a blind self-retest after 48
   hours for intra-rater kappa.
3. **Replace self-reported confidence with a signal that means something.**
   The current gate is provably dead (failure 2), so there is nothing to tune
   until it is replaced. Retrieval similarity is the cheapest candidate — it is
   already computed — followed by agreement between two differently-worded
   prompts. Once a real signal exists, sweep *its* threshold and plot
   automation rate against missed escalations, then choose the operating point
   deliberately instead of inheriting it from a rule. That curve is the thing
   a support lead actually needs, and this report cannot currently draw it.
4. **Kill the over-escalation.** 20% of traffic escalated needlessly, because
   any misclassification into an account-bound intent forces a hand-off.
   Escalation precision 0.532 means a human's time is wasted roughly as often
   as it is well spent.
5. **Add a `null` / no-request class and re-prompt for it.** `other` is 16% of
   traffic and the agent gets 1 of 24 right. Two-stage classification —
   "is there a request at all?" then "which kind?" — is the obvious fix.
6. **Scan the agent's own output for PII requests.** The guardrail checks the
   customer's message and never the reply. A regex on the draft is an hour's
   work and closes the highest-real-cost failure in the set.
7. **Model swap as an ablation.** Run the 7B as the agent against the 3B, same
   harness, same golden set. If the harness cannot detect a capability
   difference, the harness is what needs fixing.
8. **Handle images.** Several tweets carry their payload in a screenshot the
   pipeline cannot read. Cheapest honest fix is a rule that escalates
   image-only messages rather than guessing.

---

## Borrowed / cited

- Dataset: `thoughtvector/customer-support-on-twitter` (Kaggle), pulled from the
  `SunidhiSriram/twcs` Hugging Face mirror of the same csv.
- Models: `llama3.2:3b` and `qwen2.5:7b-instruct` via [Ollama](https://ollama.com).
- Libraries: pandas, scikit-learn (TF-IDF, KMeans, logistic regression, Cohen's
  kappa, macro-F1), scipy (Spearman), openai-python — used purely as an
  OpenAI-compatible HTTP client, including against Ollama.
- The thread-reconstruction approach (following `in_response_to_tweet_id`) is
  the standard one in public notebooks on this dataset; the implementation here
  is my own.
- LLM-as-judge framing follows the now-common pattern from the MT-Bench line of
  work (Zheng et al., 2023); the rubric and the 3-point verdict scale are mine.
- Code in this repo was written with an AI coding assistant. Every decision in
  `reports/DECISIONS.md` is one I can defend and modify live.
