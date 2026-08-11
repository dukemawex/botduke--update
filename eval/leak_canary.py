"""Leak canary: does date-bounded research keep post-resolution news out?

A backtest on resolved questions is worthless if today's search results contain
the answer. This runs each question twice — bounded to a simulated forecast date
and unbounded — so the difference is visible before any score is trusted.

    python eval/leak_canary.py --corpus eval/corpus.jsonl --n 20 --lead-days 30
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, random, sys
import requests

URL = "https://nimble-retriever.webit.live/search"


def search(query: str, end_date: str | None = None, n: int = 6) -> list[dict]:
    payload = {"query": query, "max_results": n, "deep_search": False, "focus": "general"}
    if end_date:
        payload["end_date"] = end_date
    try:
        r = requests.post(
            URL, json=payload, timeout=45,
            headers={"Authorization": f"Bearer {os.environ['NIMBLE_API_KEY']}",
                     "Content-Type": "application/json"},
        )
    except requests.RequestException:
        return []
    return ((r.json() or {}).get("results") or []) if r.status_code == 200 else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eval/corpus.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--lead-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.corpus)]
    rows = [r for r in rows if r.get("actual_resolve_time")]
    random.seed(a.seed)
    for q in random.sample(rows, min(a.n, len(rows))):
        rt = dt.datetime.fromisoformat(q["actual_resolve_time"].replace("Z", "+00:00"))
        cutoff = (rt - dt.timedelta(days=a.lead_days)).date().isoformat()
        bounded, live = search(q["title"], end_date=cutoff), search(q["title"])
        print("=" * 100)
        print(f"Q: {q['title'][:95]}")
        print(f"   resolved {str(q['resolution']).upper()} on {rt.date()} | cutoff {cutoff} "
              f"| bounded={len(bounded)} unbounded={len(live)}")
        for it in bounded[:4]:
            print("   [bounded  ]", str(it.get("title"))[:85])
        for it in live[:3]:
            print("   [unbounded]", str(it.get("title"))[:85])


if __name__ == "__main__":
    main()
