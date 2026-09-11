"""TF-IDF nearest-neighbour retrieval over the brand's resolved history.

Deliberately not embeddings: sklearn is already a dependency, the corpus is 28k
short tweets, and character+word TF-IDF handles the misspellings and product
names ("spotifiy", "Connect", "Discover Weekly") that a small general-purpose
embedding model tends to smooth away. Swap-in point is `Retriever.search`.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent.parent
THREADS = ROOT / "data" / "threads.jsonl"

# Replies that resolve nothing -- never worth retrieving as an exemplar.
_JUNK = ("we've sent a reply over to your dm", "let's carry on chatting there")


class Retriever:
    def __init__(self, exclude_ids=(), path=THREADS, min_reply_len=40):
        exclude = set(exclude_ids)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.docs = [
            r for r in rows
            if r["thread_id"] not in exclude
            and len(r["first_reply"]) >= min_reply_len
            and not any(j in r["first_reply"].lower() for j in _JUNK)
        ]
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   ngram_range=(1, 2), min_df=2)
        self.X = self.vec.fit_transform([d["customer_msg"] for d in self.docs])

    def search(self, query: str, k: int = 3):
        sims = linear_kernel(self.vec.transform([query]), self.X).ravel()
        top = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.docs[i]) for i in top if sims[i] > 0]


if __name__ == "__main__":
    r = Retriever()
    print(f"index size: {len(r.docs)}")
    hits = r.search("my premium got charged twice this month")
    assert hits, "retriever returned nothing for an obvious query"
    for s, d in hits:
        print(f"\n{s:.3f}  C: {d['customer_msg'][:100]}\n       A: {d['first_reply'][:120]}")
