"""Sample golden-set candidates and freeze the train/eval split.

Two disjoint slices, both written to data/golden_candidates.jsonl:

  stratified (n=170) -- proportional over 14 TF-IDF/KMeans clusters with a floor
      of 8 per cluster, so rare-but-real intents (web player, Discover Weekly,
      family plan) are actually represented.
  random (n=50) -- uniform draw from the same pool, untouched by stratification.

The random slice exists so the report can show what the stratified headline
number hides: stratified over-weights small clusters, so it is NOT an estimate
of live production accuracy. The random slice is.

Golden threads are excluded from the retrieval index (see agent.py) to stop the
agent copying the exact reply it is being scored against.
"""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
THREADS = ROOT / "data" / "threads.jsonl"
OUT = ROOT / "data" / "golden_candidates.jsonl"
SEED = 20260910
# 150 total = the brief's floor. Cut from the stratified slice, not the random
# one: the random slice is the only honest production estimator and 50 is
# already thin.
N_STRAT, N_RANDOM, K = 100, 50, 14


def load(path=THREADS):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    rng = random.Random(SEED)
    threads = load()

    # Only messages with enough text to label meaningfully.
    pool = [t for t in threads if len(t["customer_msg"]) >= 15]

    vec = TfidfVectorizer(max_features=20000, stop_words="english",
                          ngram_range=(1, 2), min_df=5)
    X = vec.fit_transform([t["customer_msg"] for t in pool])
    labels = KMeans(n_clusters=K, random_state=0, n_init=4).fit_predict(X)

    by_cluster: dict[int, list[int]] = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(i)

    # Proportional allocation with a floor, then trim the largest to hit N.
    quota = {c: max(8, round(N_STRAT * len(ix) / len(pool)))
             for c, ix in by_cluster.items()}
    while sum(quota.values()) > N_STRAT:
        quota[max(quota, key=quota.get)] -= 1

    picked: set[int] = set()
    for c, n in quota.items():
        picked.update(rng.sample(by_cluster[c], min(n, len(by_cluster[c]))))

    rest = [i for i in range(len(pool)) if i not in picked]
    rnd = set(rng.sample(rest, N_RANDOM))

    rows = []
    for i in sorted(picked | rnd):
        t = pool[i]
        rows.append({
            "thread_id": t["thread_id"],
            "slice": "stratified" if i in picked else "random",
            "cluster": int(labels[i]),
            "customer_msg": t["customer_msg"],
            "brand_reply": t["first_reply"],
            "n_turns": t["n_turns"],
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"pool={len(pool)} candidates={len(rows)} "
          f"(stratified={sum(r['slice']=='stratified' for r in rows)}, "
          f"random={sum(r['slice']=='random' for r in rows)})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
