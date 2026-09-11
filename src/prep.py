"""Build the brand's conversation corpus from the raw TWCS csv.

Output: data/threads.jsonl, one object per customer support case.

Brand choice (see reports/DECISIONS.md): SpotifyCares. High volume (43k agent
replies), only ~31% of replies are pure "DM us" deflections, and ~50% carry a
help-article link -- i.e. the historical replies actually contain the resolution
we want to ground on. Airlines and telcos in this dataset deflect far more.
"""
import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "twcs.csv"
OUT = ROOT / "data" / "threads.jsonl"

HANDLE = re.compile(r"@\w+")
WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Unescape entities, drop @handles and trailing agent signatures."""
    t = html.unescape(str(text))
    t = HANDLE.sub("", t)
    t = re.sub(r"\s+[-–—/]\s?[A-Z]{1,3}\s*$", "", t)  # "-JP" agent sign-off
    return WS.sub(" ", t).strip()


def load_brand_subgraph(brand: str) -> pd.DataFrame:
    df = pd.read_csv(
        RAW,
        dtype=str,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text",
                 "in_response_to_tweet_id"],
    )
    df["inbound"] = df["inbound"] == "True"

    # Grow the id set outward from the brand's own tweets along parent/child
    # links until it stops growing. Conversations here are short, so this
    # converges in a handful of passes.
    ids = set(df.loc[df.author_id == brand, "tweet_id"])
    parent = df["in_response_to_tweet_id"]
    for _ in range(10):
        before = len(ids)
        in_set = df["tweet_id"].isin(ids)
        ids |= set(parent[in_set].dropna())            # walk up
        ids |= set(df.loc[parent.isin(ids), "tweet_id"])  # walk down
        if len(ids) == before:
            break
    return df[df["tweet_id"].isin(ids)].copy()


def build_threads(sub: pd.DataFrame, brand: str) -> list[dict]:
    by_id = {r.tweet_id: r for r in sub.itertuples(index=False)}
    children: dict[str, list[str]] = {}
    for r in sub.itertuples(index=False):
        if isinstance(r.in_response_to_tweet_id, str):
            children.setdefault(r.in_response_to_tweet_id, []).append(r.tweet_id)

    # A case starts at an inbound tweet with no parent in our subgraph.
    roots = [
        r.tweet_id for r in sub.itertuples(index=False)
        if r.inbound and not (isinstance(r.in_response_to_tweet_id, str)
                              and r.in_response_to_tweet_id in by_id)
    ]

    threads = []
    for root in roots:
        turns, node, seen = [], root, set()
        while node and node in by_id and node not in seen:
            seen.add(node)
            r = by_id[node]
            turns.append({
                "author": brand if not r.inbound else "customer",
                "inbound": bool(r.inbound),
                "text": clean(r.text),
                "created_at": r.created_at,
            })
            kids = [k for k in children.get(node, []) if k in by_id]
            node = kids[0] if kids else None  # main line only; branches are rare

        agent_turns = [t for t in turns if not t["inbound"]]
        if not agent_turns or not turns[0]["text"]:
            continue  # unanswered or empty -- no resolution to learn from
        threads.append({
            "thread_id": root,
            "created_at": turns[0]["created_at"],
            "customer_msg": turns[0]["text"],
            "first_reply": agent_turns[0]["text"],
            "turns": turns,
            "n_turns": len(turns),
            "n_agent_turns": len(agent_turns),
        })
    return threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    sub = load_brand_subgraph(args.brand)
    threads = build_threads(sub, args.brand)
    threads.sort(key=lambda t: int(t["thread_id"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    lens = pd.Series([t["n_turns"] for t in threads])
    print(f"brand={args.brand} subgraph_tweets={len(sub)} threads={len(threads)}")
    print(f"turns per thread: mean={lens.mean():.1f} median={lens.median():.0f} max={lens.max()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
