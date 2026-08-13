"""Two-stage replay with deduplication, bootstrap CIs, and held-out validation.

Stage 1 (expensive, once per question): research + ensemble -> member probabilities,
        spread, research quality. Cached to disk. This is what provider credits buy.
Stage 2 (free, unlimited): push cached stage-1 output through a calibration policy
        and score it. Sweep configs for zero tokens.

    python eval/replay.py compare --stage1 eval/stage1.jsonl
    python eval/replay.py validate --stage1 eval/stage1.jsonl
    python eval/replay.py clean --stage1 eval/stage1.jsonl --out eval/stage1.clean.jsonl
"""
from __future__ import annotations
import argparse, json, itertools, statistics, sys, random
from dataclasses import replace
from calibration import Inputs, V2Config, legacy, v2
from score import summarize, bootstrap_metric_delta


def load_stage1(path: str, dedupe: bool = True) -> list[dict]:
    """Load successful stage-one rows; retries can append duplicate question IDs."""
    rows = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if dedupe and row.get("id") in seen:
                continue
            if dedupe:
                seen.add(row.get("id"))
            rows.append(row)
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


def sweep_results(rows: list[dict]):
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
    return results


def cmd_compare(rows: list[dict]) -> None:
    print(f"deduplicated rows={len(rows)}")
    for name, pol in (("legacy", legacy), ("v2", lambda i: v2(i))):
        s, _ = evaluate(rows, pol)
        print(f"{name:10s} {s}")


def cmd_sweep(rows: list[dict]) -> None:
    base, _ = evaluate(rows, legacy)
    print(f"{'legacy':<62s} {base}")
    print("-" * 110)
    for _, cfg, s in sweep_results(rows)[:8]:
        print(f"{json.dumps(cfg):<62s} {s}")


def _policy_for(cfg: dict):
    c = replace(V2Config(), **cfg)
    return lambda i: v2(i, c)


def _describe_delta(rows: list[dict], a, b, seed: int) -> dict:
    _, pairs_a = evaluate(rows, a)
    _, pairs_b = evaluate(rows, b)
    log = []
    brier = []
    for (pa, y), (pb, _) in zip(pairs_a, pairs_b):
        from score import binary_log_score, brier as brier_score
        log.append(binary_log_score(pb, y) - binary_log_score(pa, y))
        brier.append(brier_score(pb, y) - brier_score(pa, y))
    return {
        "n": len(rows),
        "log_score_delta_b_minus_a": bootstrap_metric_delta(log, seed=seed),
        "brier_delta_b_minus_a": bootstrap_metric_delta(brier, seed=seed + 1),
    }


def cmd_validate(rows: list[dict]) -> None:
    if len(rows) < 20:
        sys.exit("need at least 20 deduplicated rows for held-out validation")
    ordered = sorted(rows, key=lambda r: (r.get("as_of") or "", str(r.get("id"))))
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * 0.80)))
    train, holdout = ordered[:cut], ordered[cut:]
    ranked = sweep_results(train)
    best_cfg = ranked[0][1]
    best = _policy_for(best_cfg)
    print(json.dumps({"dataset": len(rows), "train": len(train), "holdout": len(holdout),
                      "split": "chronological 80/20", "best_train_config": best_cfg}, indent=2))
    for label, subset in (("train", train), ("holdout", holdout), ("all", ordered)):
        legacy_s, _ = evaluate(subset, legacy)
        v2_s, _ = evaluate(subset, lambda i: v2(i))
        best_s, _ = evaluate(subset, best)
        print(f"{label:8s} legacy {legacy_s}")
        print(f"{label:8s} v2-default {v2_s}")
        print(f"{label:8s} v2-best-train {best_s}")
    print("bootstrap 95% CIs: positive log delta means V2 is better; negative Brier delta means V2 is better")
    for label, subset in (("holdout", holdout), ("all", ordered)):
        print(label, "default-vs-legacy", json.dumps(_describe_delta(subset, legacy, lambda i: v2(i), 20260813)))
        print(label, "best-vs-legacy", json.dumps(_describe_delta(subset, legacy, best, 20260814)))


def cmd_clean(rows: list[dict], out: str) -> None:
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} unique rows to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "sweep", "validate", "clean"])
    ap.add_argument("--stage1", default="eval/stage1.jsonl")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load_stage1(a.stage1)
    if not rows:
        sys.exit("empty stage1 file")
    if a.cmd == "compare": cmd_compare(rows)
    elif a.cmd == "sweep": cmd_sweep(rows)
    elif a.cmd == "validate": cmd_validate(rows)
    else: cmd_clean(rows, a.out or a.stage1 + ".clean.jsonl")


if __name__ == "__main__":
    main()
