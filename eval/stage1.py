"""Stage 1: date-bounded research + ensemble over the resolved corpus.

Expensive and cached. Produces stage1.jsonl, which replay.py then sweeps for free.
Every search is bounded to the simulated forecast date so the backtest cannot
read the answer. Resumable: rerunning skips questions already in the output.

    python eval/stage1.py --corpus eval/corpus.jsonl --lead-days 30 --members 5
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import complete, extract_probability

NIMBLE_URL = "https://nimble-retriever.webit.live/search"


def nimble(query: str, end_date: str, n: int = 6) -> list[dict]:
    try:
        r = requests.post(NIMBLE_URL, timeout=45,
            headers={"Authorization": f"Bearer {os.environ['NIMBLE_API_KEY']}",
                     "Content-Type": "application/json"},
            json={"query": query, "max_results": n, "deep_search": False,
                  "focus": "general", "end_date": end_date})
    except requests.RequestException:
        return []
    return ((r.json() or {}).get("results") or []) if r.status_code == 200 else []


def make_queries(question: str, as_of: str) -> list[str]:
    """Raw question titles retrieve badly. Rewrite into search queries first."""
    out = complete(
        f"Today is {as_of}. Turn this forecasting question into 3 web search "
        f"queries that would surface the relevant evidence as of that date. "
        f"Return only the 3 queries, one per line, no numbering.\n\n{question}",
        temperature=0.2, max_tokens=700)
    qs = [l.strip(" -•\t") for l in (out or "").splitlines() if l.strip()][:3]
    return qs or [question]


# main.py maps "number of sources that returned content" onto this ladder.
# The harness reuses it so legacy() sees the same quality scale it sees live,
# otherwise the A/B silently disables legacy's weak-research shrink.
_QUALITY_LADDER = {0: 0.40, 1: 0.50, 2: 0.65, 3: 0.78, 4: 0.85, 5: 0.90, 6: 0.93}


def _domain(url: str) -> str:
    u = (url or "").split("//")[-1]
    return u.split("/")[0].lower().removeprefix("www.")


def research(question: str, as_of: str) -> tuple[str, float]:
    blocks, domains = [], set()
    for q in make_queries(question, as_of):
        for item in nimble(q, as_of):
            title = (item.get("title") or "").strip()
            desc = (item.get("description") or item.get("content") or "").strip()
            if not (title or desc):
                continue
            blocks.append(f"- {title}: {desc[:400]}")
            d = _domain(item.get("url") or "")
            domains.add(d or title[:24].lower())
    text = "\n".join(blocks[:18])
    # Count distinct sources, not raw snippet volume. Snippet count saturates
    # at 18 on every question and makes the quality signal constant.
    quality = _QUALITY_LADDER.get(min(len(domains), 6), 0.93)
    return text, round(quality, 3)


PROMPT = """You are a careful superforecaster. Today is {as_of}.

Question: {title}

Resolution criteria: {criteria}

Research (nothing published after {as_of}):
{research}

Give the base rate for this class of event, the main evidence for and against,
and what the status quo outcome is if nothing changes. Then state your answer
on the final line in exactly this form:
Probability: NN%"""


def forecast_one(row: dict, lead_days: int, members: int) -> dict | None:
    rt = row.get("actual_resolve_time")
    if not rt or row.get("type") != "binary":
        return None
    resolve = dt.datetime.fromisoformat(rt.replace("Z", "+00:00"))
    as_of_dt = resolve - dt.timedelta(days=lead_days)
    open_dt = None
    if row.get("open_time"):
        open_dt = dt.datetime.fromisoformat(row["open_time"].replace("Z", "+00:00"))
        if as_of_dt <= open_dt:
            as_of_dt = open_dt + dt.timedelta(days=1)
    if as_of_dt >= resolve:
        return None
    as_of = as_of_dt.date().isoformat()

    try:
        res_text, quality = research(row["title"], as_of)
        prompt = PROMPT.format(as_of=as_of, title=row["title"],
                               criteria=(row.get("resolution_criteria") or "")[:1500],
                               research=res_text or "(no usable results)")
        def one_member(i: int):
            try:
                return extract_probability(
                    complete(prompt, temperature=0.2 + 0.2 * i, max_tokens=2500))
            except Exception as e:
                print(f"  member fail {row['id']}: {e}", file=sys.stderr)
                return None

        # Ensemble members are independent; running them sequentially made each
        # question cost members x latency for no reason.
        with ThreadPoolExecutor(max_workers=members) as mex:
            probs = [p for p in mex.map(one_member, range(members)) if p is not None]
        if not probs:
            return None
    except Exception as e:
        print(f"  question fail {row['id']}: {e}", file=sys.stderr)
        return None

    return {
        "id": row["id"], "title": row["title"], "url": row.get("url"),
        "as_of": as_of, "member_probs": probs, "research_quality": quality,
        "research_chars": len(res_text),
        "days_to_close": (resolve - as_of_dt).total_seconds() / 86400.0,
        "resolved_yes": str(row["resolution"]).lower() == "yes",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eval/corpus.jsonl")
    ap.add_argument("--out", default="eval/stage1.jsonl")
    ap.add_argument("--lead-days", type=int, default=30)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    rows = [json.loads(l) for l in open(a.corpus)]
    todo = [r for r in rows if r["id"] not in done and r.get("type") == "binary"]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(done)} cached, {len(todo)} to run", file=sys.stderr)

    n = 0
    with open(a.out, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(lambda r: forecast_one(r, a.lead_days, a.members), todo):
            n += 1
            if res:
                f.write(json.dumps(res) + "\n")
                f.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr, flush=True)
    print("stage1 complete")


if __name__ == "__main__":
    main()
