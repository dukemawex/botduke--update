"""Two-stage replay.

Stage 1 (expensive, once per question): research + ensemble -> member probabilities,
        spread, research quality. Cached to disk. This is what Azure credits buy.
Stage 2 (free, unlimited): push cached stage-1 output through a calibration policy
        and score it. Sweep thousands of configs for zero tokens.

    python eval/replay.py sweep --stage1 eval/stage1.jsonl
"""
from __future__ import annotations
import argparse, json, itertools, statistics, sys
from dataclasses import replace
from calibration import Inputs, V2Config, legacy, v2
from score import summarize


def load_stage1(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_inputs(row: dict) -> Inputs:
    probs = row["member_probs"]
    return Inputs(
        p=statistics.median(probs),
        spread=(max(probs) - min(probs)) if len(probs) > 1 else 0.0,
        research_quality=row.get("research_quality", 0.7),
        days_to_close=row["days_to_close"],
        consistent=row.get("consistent", True),
        community=row.get("community"),
        base_rate=row.get("base_rate"),
    )


def evaluate(rows: list[dict], policy) -> tuple:
    pairs, baseline = [], []
    for row in rows:
        p = policy(to_inputs(row))
        pairs.append((p, bool(row["resolved_yes"])))
        baseline.append(row.get("community") or 0.5)
    return summarize(pairs, baseline), pairs


def cmd_compare(rows: list[dict]) -> None:
    for name, pol in (("legacy", legacy), ("v2", lambda i: v2(i))):
        s, _ = evaluate(rows, pol)
        print(f"{name:10s} {s}")


def cmd_sweep(rows: list[dict]) -> None:
    grid = {
        "default_base_rate": [0.25, 0.35, 0.45],
        "max_total_shrink": [0.20, 0.35, 0.50],
        "horizon_weight": [0.0, 0.25, 0.5],
        "extremize_strength": [0.0, 0.15, 0.3],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = replace(V2Config(), **dict(zip(keys, combo)))
        s, _ = evaluate(rows, lambda i, c=cfg: v2(i, c))
        results.append((s.mean_log_score, dict(zip(keys, combo)), s))
    results.sort(key=lambda r: -r[0])
    base, _ = evaluate(rows, legacy)
    print(f"{'legacy':<62s} {base}")
    print("-" * 110)
    for _, cfg, s in results[:8]:
        print(f"{json.dumps(cfg):<62s} {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "sweep"])
    ap.add_argument("--stage1", default="eval/stage1.jsonl")
    a = ap.parse_args()
    rows = load_stage1(a.stage1)
    if not rows:
        sys.exit("empty stage1 file")
    {"compare": cmd_compare, "sweep": cmd_sweep}[a.cmd](rows)


if __name__ == "__main__":
    main()
