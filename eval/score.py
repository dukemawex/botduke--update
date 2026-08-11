"""Scoring for offline forecast replay. No network, no LLM calls."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

EPS = 1e-6


def _clip(p: float, lo: float = EPS, hi: float = 1 - EPS) -> float:
    return min(max(p, lo), hi)


def binary_log_score(p: float, resolved_yes: bool) -> float:
    """Natural-log score. Higher is better. log(1)=0 is perfect."""
    p = _clip(p)
    return math.log(p if resolved_yes else 1 - p)


def brier(p: float, resolved_yes: bool) -> float:
    """Lower is better."""
    return (p - (1.0 if resolved_yes else 0.0)) ** 2


def mc_log_score(probs: Sequence[float], correct_index: int) -> float:
    total = sum(probs) or 1.0
    return math.log(_clip(probs[correct_index] / total))


def numeric_log_score(percentiles: dict[float, float], outcome: float) -> float:
    """Approximate continuous log score from a percentile ladder.

    Builds a piecewise-uniform density between the declared percentiles and
    returns log(density at outcome). Tails get the nearest interior slab width,
    which is pessimistic but consistent across variants.
    """
    pts = sorted(percentiles.items())
    if len(pts) < 2:
        return math.log(EPS)
    slabs = []
    for (q0, v0), (q1, v1) in zip(pts, pts[1:]):
        width = v1 - v0
        mass = q1 - q0
        if width <= 0 or mass <= 0:
            continue
        slabs.append((v0, v1, mass / width))
    if not slabs:
        return math.log(EPS)
    for lo, hi, dens in slabs:
        if lo <= outcome <= hi:
            return math.log(_clip(dens, EPS, 1e9))
    # outside the ladder: charge the outer tail mass over the nearest slab width
    if outcome < slabs[0][0]:
        tail_mass, (lo, hi, _) = pts[0][0], slabs[0]
    else:
        tail_mass, (lo, hi, _) = 1.0 - pts[-1][0], slabs[-1]
    width = max(hi - lo, EPS)
    return math.log(_clip(max(tail_mass, EPS) / width, EPS, 1e9))


@dataclass
class Summary:
    n: int
    mean_log_score: float
    mean_brier: float
    baseline_relative: float   # peer-score proxy, x100
    calibration_error: float   # weighted |empirical - stated|

    def __str__(self) -> str:
        return (
            f"n={self.n}  log={self.mean_log_score:+.4f}  brier={self.mean_brier:.4f}  "
            f"vs-baseline={self.baseline_relative:+.2f}  cal_err={self.calibration_error:.4f}"
        )


def calibration_error(pairs: Iterable[tuple[float, bool]], bins: int = 10) -> float:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    pairs = list(pairs)
    if not pairs:
        return 0.0
    for p, y in pairs:
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((p, y))
    total = len(pairs)
    err = 0.0
    for b in buckets:
        if not b:
            continue
        stated = sum(p for p, _ in b) / len(b)
        empirical = sum(1 for _, y in b if y) / len(b)
        err += (len(b) / total) * abs(empirical - stated)
    return err


def summarize(
    pairs: Sequence[tuple[float, bool]],
    baseline: Sequence[float] | None = None,
) -> Summary:
    """`baseline` is a comparison forecast per question (peer-score proxy).

    Metaculus peer score is your log score minus the average log score of every
    other forecaster, x100. We cannot see other bots offline, so we score against
    a fixed baseline instead. It ranks variants correctly; it is not the real number.
    """
    if not pairs:
        return Summary(0, 0.0, 0.0, 0.0, 0.0)
    logs = [binary_log_score(p, y) for p, y in pairs]
    briers = [brier(p, y) for p, y in pairs]
    if baseline is None:
        baseline = [0.5] * len(pairs)
    base_logs = [binary_log_score(b, y) for b, (_, y) in zip(baseline, pairs)]
    rel = 100.0 * (sum(logs) - sum(base_logs)) / len(pairs)
    return Summary(
        n=len(pairs),
        mean_log_score=sum(logs) / len(logs),
        mean_brier=sum(briers) / len(briers),
        baseline_relative=rel,
        calibration_error=calibration_error(pairs),
    )
