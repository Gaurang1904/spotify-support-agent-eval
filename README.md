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

Spotify's Twitter support is a **triage desk, not a resolution desk**. A third
of its replies are "DM us your account email"; the agent has no account access,
no billing system, no ability to issue a refund. The job is not to resolve the
ticket. It is to:

- put the message in the right lane,
- answer the ones that are genuinely answerable in public — catalogue
  questions, feature feedback, standard troubleshooting — without inventing
  anything,
- and hand the rest to a human *cleanly and early*, with a reason a support
  lead can audit.

The cost function is asymmetric: over-escalating wastes a minute of an agent's
time, under-escalating means a bot tells someone their refund is processed.
**Missed escalations are reported as a raw count**, not folded into an F1.

#### What I deliberately did not build

- **No multi-turn dialogue.** The agent sees the opening tweet only;
  conversation state moves into DMs the dataset does not contain.
- **No fine-tuning.** With 150 labels a fine-tune measures the golden set.
- **No embeddings / vector DB.** TF-IDF over 28k short tweets is competitive,
  keeps product names and misspellings, and is one dependency lighter.
- **No sentiment or priority model.** Anger matters only where it changes
  routing, which the escalation policy already covers.
- **No PII scrubbing.** The dataset is pre-redacted; the agent escalates on the
  residue rather than rewriting it.
- **No production serving.** Scaffolding around an unproven core.

### The taxonomy

TF-IDF + KMeans (k=14) over 28,197 opening messages, merged by hand on the
test *"would a support agent do the same thing for these?"*: `playback_error` ·
`account_access` · `billing_subscription` · `family_plan` · `content_request` ·
`feature_feedback` · `praise_chatter` · `other`. The escalation policy is
anchored to the brand's own behaviour — every always-escalate intent is one
Spotify itself answered with "DM us your account email" (decision 10).

### The golden set

150 examples, hand-labelled blind with intent **and** route, in two disjoint
slices: **stratified (100)**, proportional across the 14 clusters with a floor
of 8 so rare intents appear at all, and **random (50)**, a uniform draw. Golden
thread ids are excluded from the retrieval index so the agent cannot retrieve
the case it is scored on. 2.0% were marked ambiguous and dropped; median
labelling time 12s over 44 minutes.

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

**This is a null, not proof of equivalence.** Half-width ~0.11, so the test
resolves nothing under ~11 accuracy points; a real 5-point advantage would be
invisible. Detecting one would need roughly 800-1000 labels — the main argument
for spending next week labelling rather than modelling.

Both beat the trivial baseline decisively, which proves the task is non-trivial
rather than that either system is good.

**Reproducibility.** Every number comes from the run the shipped
`outputs/llm_cache.db` reproduces exactly. Ollama is not deterministic even at
`temperature=0`; deleting the cache moves about one example in 147 (observed:
accuracy 0.585-0.592, reply 4.52-4.56) — inside every CI here, but not zero.

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

**This generalises.** LLM-as-judge is the default way teams grade support
replies, usually with no human-agreement check and no human-ceiling control.
Both controls here are cheap — 60 blind ratings and one extra scoring pass —
and the method failed both. Absent them, assume such numbers measure the
rubric.

### Failure analysis

**1. The model ignores its own retrieval. Found by demoing, not by the harness.**
This is the core required feature — a reply "grounded in how that brand has
historically resolved similar issues" — and it silently does not work.
Retrieval is fine; the correct past cases come back with good similarity. The
model treats them as *tone* samples and discards their *content*.

> **customer:** "why isn't the new Taylor Swift album on spotify in the UK?"
> **retrieved [0.719]:** "We'd love to have all of Taylor's stuff available, but
> we have some info about content here..."
> **retrieved [0.626]:** "Fingers crossed we'll be able to have it soon..."
> **agent:** *"The new Taylor Swift album **is available** on Spotify in the UK."*

Every exemplar says the brand does not have it. The agent asserted the
opposite as fact. A customer acting on that reply goes looking for something
that is not there.

> **customer:** "how do I access your live chat? Used it last month and it was
> amazing"
> **retrieved [0.829]:** "you can also reach out to our Chat team here [link]"
> **agent:** `praise_chatter`, confidence 1.0 — *"We're glad to hear you had a
> great experience!"* No link, no answer.

The top hit contained exactly the answer. The model replied to the sentiment
instead.

*Hypothesis:* one JSON call collapses classification, routing and drafting into
a single generation, so retrieved text enters as style rather than evidence.
This is the mechanism behind the judge's `grounded` sub-score of 1.13/2, a
number that sat here unexplained until a demo produced one. 147 evaluated
examples missed it; two minutes of demoing did not.

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

*Hypothesis:* a 3B model asked for a confidence emits a plausible number, not
a calibrated one. Gating on self-reported confidence rests on nothing; the fix
is an external signal — retrieval similarity, classifier margin, or two-prompt
agreement.

**3. `other` is undetectable — 24 gold examples, 1 correct.**
The model has no concept of "this tweet contains no classifiable request." The
24 `other` rows scatter into `account_access` (8), `praise_chatter` (6),
`feature_feedback` (3), `content_request` (3), `billing_subscription` (3).

> "Answer my DM! Urgent, thanks." -> predicted `praise_chatter`, confidence 1.0

*Hypothesis:* offered 8 positive categories and one residual, the model always
finds a positive one it likes. A two-stage "is there a request at all?" check
would likely fix it. 16% of the golden set.

**4. Broken-app complaints read as product opinion — 10 cases,
`playback_error` -> `feature_feedback`.** The single largest confusion.

> "Please fix the desktop app? Endless spinning, everything, all platforms —
> weeks now. PLEASE FIX" -> `feature_feedback` / auto-handle, confidence 0.8

*Hypothesis:* imperative "please fix / please add" phrasing outweighs the
symptom described, and the exemplars do not help because past agents answered
both with the same soothing template.

**5. Over-escalation — 29 cases, 20% of the set — and it asks for personal
data in public.** Escalation precision 0.532: nearly half of what reaches a
human did not need to. The guardrail causes it — any misclassification into an
account-bound intent forces escalation, turning a classification error into a
routing error.

> Spotify's own advertisement, appearing in the inbound stream -> predicted
> `billing_subscription` -> escalate -> *"Sorry, we can't confirm the 99p offer.
> Can you DM us your account's email address?"*

The reply above requests an email address on a public timeline. This is the
failure with real-world cost, and the PII guardrail does not catch it: the
regex checks the **customer's** message for personal data and never the
**agent's own reply**.

#### Grounding ablation: I tried the obvious fix and it did not work

The obvious fix is to tell the model the retrieved cases are facts.
`agent.py` carries it as `GROUNDING_PATCH`, **off by default so the evaluated
system stays frozen**; `grounding_ablation.py` runs both variants over 26 cases
where retrieval fired, including both demo cases.

The metric is deliberately **not** the LLM judge — at kappa 0.100 nothing it
emits can carry an argument. Each reply was hand-rated blind to variant, on a
question of fact: **C** contradicts the retrieved cases · **I** ignores an
answer that was right there · **G** grounded, or honestly unsure.

| | contradicts | ignores | grounded | **ungrounded rate** |
|---|---|---|---|---|
| frozen agent | 3 | 11 | 12 | **0.538** |
| + grounding patch | 2 | 13 | 11 | **0.577** |

Paired over the same 26 cases: the patch **fixed 3, broke 4, changed nothing on
19**. Delta +0.038, 95% CI [−0.154, +0.231] — no measured effect, and the
sample is far too small to claim one.

It did work on the headline case:

> **frozen:** "The new Taylor Swift album **is available** on Spotify in the UK."
> **patched:** "We'd love to have all of Taylor's stuff available, but we have
> some info about content here: [link]"

But it bought that with evasion elsewhere — contradictions fell 3 to 2 while
*ignoring a retrieved answer* rose 11 to 13. Told not to fill gaps from its own
knowledge, a 3B model hedges rather than reading more carefully.

**The number that matters is the baseline, not the delta: 54% of the frozen
agent's replies are ungrounded.** More than half contradict their retrieval or
waste an answer sitting in it. It is the worst number in this report and it is
invisible to intent accuracy.

Prompting is not the fix. Next is structural: a separate retrieval-reading call
that must quote the span it relies on before any draft is written, making
grounding a checkable step rather than a hope.

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

**5. The metrics are blind to the worst failure they could have.** The agent
told a customer an album was available when all three retrieved cases said it
was not — and intent accuracy scores that example **correct**, because the
intent really is `content_request`. The judge is invalid at kappa 0.100. Two
minutes of demoing found what 147 automated examples missed; assume the same of
failures nobody has demoed yet.

**6. Two of the four guardrails have never fired.** The confidence gate
(escalate below 0.5) fired twice in 147, both on JSON parse failures another
rule already caught. The "no similar resolved case" gate fired **zero** times,
because `Retriever.search` returns any hit with similarity above 0 and the gate
only tests whether the list is empty — so a top hit at 0.33 counts as
grounding. On the golden set 25% of cases have a top similarity at or below
0.35 and 65% at or below 0.50, and the gate passed all of them. Live example:
*"do you sell headphones"* retrieved three irrelevant cases (0.41 / 0.33 /
0.33) and the agent invented a Spotify store — *"we do have a wide selection of
audio equipment and accessories available for purchase on our website"*. The
safety architecture is half the size it appears, and the automation rate comes
from two rules, not four. Both dead gates want the same fix: a similarity
floor, which is next-week item 5.

**7. Accuracy is the wrong metric for the actual cost function.** The agent
auto-handled 9 cases a human needed (6.1%) — including a customer reporting a
compromised account and one whose tweet carried self-harm-adjacent language.
Meanwhile 19% of the set was escalated unnecessarily. A single accuracy figure
prices those identically. They are not.

**8. The ground truth is one person's opinion.** 150 labels, one annotator, no
inter-annotator agreement number. The taxonomy was derived by clustering, then
the same person who defined the clusters assigned the labels, and the agent
retrieves with the same TF-IDF representation the clusters came from. Some part
of 0.585 is agreement with a clustering artifact. Median labelling time was 12s
per item over 44 minutes, and 2.0% were marked ambiguous — those numbers are
published so the ground truth can be judged, not assumed.

One line: **an LLM support agent, evaluated properly, could not be shown to
beat logistic regression here; the instrument built to judge its replies does
not agree with a human; and more than half its replies ignore or contradict the
evidence it retrieved.**

### With one more week

Ordered by what would change a conclusion, not by what is most interesting.

1. **Make grounding structural, not prompted.** The patch failed (54%
   ungrounded either way). Next: a separate retrieval-reading call that must
   quote the span it relies on, then a check that the draft asserts nothing
   outside it. That check needs no LLM judge, which is the point.
2. **Fix the judge or drop it.** Kappa 0.100 makes every reply metric unusable.
   Rewrite the rubric against the 60 human ratings as a dev set and re-measure;
   if it cannot clear ~0.6, report human ratings alone and delete the judge.
3. **A second annotator on 50 examples.** Every caveat here bottoms out at "one
   person decided." Failing that, a blind self-retest after 48 hours for
   intra-rater kappa.
4. **More labels, for power.** The headline null resolves nothing under ~11
   accuracy points. 800-1000 labels would make the agent-vs-baseline comparison
   mean something.
5. **Replace self-reported confidence.** The gate is provably dead, so there is
   nothing to tune until it is replaced — retrieval similarity is already
   computed. Then sweep *its* threshold and plot automation against missed
   escalations, the curve a support lead actually needs and this report cannot
   draw.
6. **Add a `null` / no-request class.** `other` is 16% of traffic at 1 of 24
   correct. Two-stage — "is there a request?" then "which kind?"
7. **Scan the agent's own output for PII requests.** The guardrail checks the
   customer's message, never the reply. An hour's work, closes the
   highest-real-cost failure in the set.
8. **Model swap as an ablation, and handle images.** Run the 7B as the agent on
   the same harness; if it cannot detect the capability difference, the harness
   needs fixing. Several tweets carry their payload in a screenshot — escalate
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
