"""Build a corpus of resolved Metaculus questions.

The list endpoint does NOT serialize `resolution` — it is only present on the
detail endpoint — so this runs list-then-hydrate. Requires METACULUS_TOKEN.

    python eval/fetch_corpus.py --months 18 --limit 1500 --out eval/corpus.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import requests

API = "https://www.metaculus.com/api"
_local = threading.local()


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Token {os.environ['METACULUS_TOKEN']}",
            "Accept": "application/json",
        })
        _local.s = s
    return s


def list_ids(qtype: str, months: int, limit: int) -> list[int]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
    ids, offset = [], 0
    while len(ids) < limit:
        r = session().get(f"{API}/posts/", timeout=60, params={
            "statuses": "resolved", "forecast_type": qtype, "limit": 100,
            "offset": offset, "order_by": "-actual_resolve_time",
        })
        if r.status_code != 200:
            print(f"  list {qtype} offset={offset}: HTTP {r.status_code}", file=sys.stderr)
            break
        results = r.json().get("results", [])
        if not results:
            break
        stop = False
        for p in results:
            rt = p.get("actual_resolve_time") or (p.get("question") or {}).get("actual_resolve_time")
            if rt:
                try:
                    if datetime.fromisoformat(rt.replace("Z", "+00:00")) < cutoff:
                        stop = True
                        continue
                except ValueError:
                    pass
            ids.append(p["id"])
        if stop or not r.json().get("next"):
            break
        offset += 100
        time.sleep(0.2)
    return ids[:limit]


def hydrate(post_id: int) -> dict | None:
    for attempt in range(3):
        try:
            r = session().get(f"{API}/posts/{post_id}/", timeout=60, params={"with_cp": "true"})
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        post = r.json()
        q = post.get("question") or {}
        res = q.get("resolution")
        if res in (None, "annulled", "ambiguous", ""):
            return None
        agg = ((q.get("aggregations") or {}).get("recency_weighted") or {})
        latest = agg.get("latest") or {}
        return {
            "id": post_id,
            "type": q.get("type"),
            "title": post.get("title"),
            "description": q.get("description"),
            "resolution_criteria": q.get("resolution_criteria"),
            "fine_print": q.get("fine_print"),
            "unit": q.get("unit"),
            "options": q.get("options"),
            "scaling": q.get("scaling"),
            "open_time": q.get("open_time"),
            "scheduled_close_time": q.get("scheduled_close_time"),
            "actual_resolve_time": q.get("actual_resolve_time"),
            "resolution": res,
            "community_latest": (latest.get("centers") or [None])[0],
            "nr_forecasters": post.get("nr_forecasters"),
            "url": f"https://www.metaculus.com/questions/{post_id}/",
        }
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--types", default="binary,multiple_choice,numeric")
    ap.add_argument("--out", default="corpus.jsonl")
    a = ap.parse_args()

    all_ids: list[int] = []
    for t in a.types.split(","):
        got = list_ids(t, a.months, a.limit)
        print(f"listed {len(got):5d} {t}", file=sys.stderr, flush=True)
        all_ids += got

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(hydrate, all_ids):
            done += 1
            if row:
                rows.append(row)
            if done % 100 == 0:
                print(f"  hydrated {done}/{len(all_ids)} -> {len(rows)} usable", file=sys.stderr, flush=True)

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"wrote {len(rows)} questions -> {a.out}  {by_type}")


if __name__ == "__main__":
    main()
