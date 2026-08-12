"""Aggregation + calibration policies, isolated so the harness can A/B them.

LEGACY reproduces main.py@main exactly (six sequential pulls toward 0.5).
V2 replaces it with one bounded shrink in log-odds space toward an explicit prior.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---- legacy constants, copied from main.py --------------------------------
_MAX_EXTREMIZE_STRENGTH = 1.3
_HIGH_SPREAD_SHRINK_THRESH = 0.25
_MED_SPREAD_SHRINK_THRESH = 0.15
_WEAK_RESEARCH_PRIOR_WT = 0.40


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def extremize(p: float, strength: float = 0.3) -> float:
    o = p / (1 - p)
    e = o ** (1 + strength)
    return min(max(e / (1 + e), 0.01), 0.99)


@dataclass
class Inputs:
    """Everything the post-processing chain is allowed to see."""
    p: float                              # ensemble aggregate before calibration
    spread: float                         # dispersion across ensemble members
    research_quality: float               # 0..1
    days_to_close: float
    consistent: bool = True
    community: Optional[float] = None
    base_rate: Optional[float] = None     # reference-class prior, V2 only
    minibench: bool = False


# ---- LEGACY ---------------------------------------------------------------

def legacy(i: Inputs) -> float:
    p = i.p
    if i.spread >= _HIGH_SPREAD_SHRINK_THRESH:
        p = 0.55 * p + 0.45 * 0.5
    elif i.spread >= _MED_SPREAD_SHRINK_THRESH:
        p = 0.75 * p + 0.25 * 0.5
    if not i.consistent:
        p = 0.5 * p + 0.5 * 0.5
    if i.research_quality < 0.6:
        p = (1 - _WEAK_RESEARCH_PRIOR_WT) * p + _WEAK_RESEARCH_PRIOR_WT * 0.5
    if i.community is not None:
        p = i.research_quality * p + (1 - i.research_quality) * i.community
    p = extremize(p, strength=0.3)
    d = i.days_to_close
    if d > 365:
        p = 0.20 * p + 0.80 * 0.5
    elif d > 180:
        p = 0.40 * p + 0.60 * 0.5
    elif d > 90:
        p = 0.65 * p + 0.35 * 0.5
    return min(max(p, 0.03), 0.97)


# ---- V2 -------------------------------------------------------------------

@dataclass
class V2Config:
    """Every number here is a sweep axis. Nothing is hand-picked and kept."""
    default_base_rate: float = 0.32       # measured: 60/190 YES on the resolved corpus
    max_total_shrink: float = 0.35        # hard cap on how much signal we discard
    spread_weight: float = 0.6            # how much dispersion contributes to shrink
    quality_weight: float = 0.5           # how much thin research contributes
    horizon_weight: float = 0.25          # how much a distant close date contributes
    horizon_halflife_days: float = 240.0
    inconsistent_penalty: float = 0.25
    community_weight: float = 0.0         # 0 = ignore; AIB usually hides it anyway
    extremize_strength: float = 0.0       # earn this on the harness before enabling
    clip: tuple[float, float] = (0.02, 0.98)


def _shrink_fraction(i: Inputs, c: V2Config) -> float:
    """One number in [0, max_total_shrink]: how much to discount our own signal."""
    frac = 0.0
    frac += c.spread_weight * min(i.spread / 0.30, 1.0)
    frac += c.quality_weight * max(0.0, 0.75 - i.research_quality) / 0.75
    horizon = 1.0 - math.exp(-max(i.days_to_close, 0.0) / c.horizon_halflife_days)
    frac += c.horizon_weight * horizon
    if not i.consistent:
        frac += c.inconsistent_penalty
    # squash the sum into the cap instead of letting terms stack past it
    return c.max_total_shrink * (1.0 - math.exp(-frac))


def v2(i: Inputs, c: V2Config = V2Config()) -> float:
    prior = i.base_rate if i.base_rate is not None else c.default_base_rate
    w = _shrink_fraction(i, c)
    # shrink in log-odds toward the prior, not toward 0.5 in probability space
    z = (1.0 - w) * logit(i.p) + w * logit(prior)
    p = sigmoid(z)
    if c.community_weight > 0 and i.community is not None:
        p = sigmoid((1 - c.community_weight) * logit(p) + c.community_weight * logit(i.community))
    if c.extremize_strength > 0:
        p = extremize(p, strength=c.extremize_strength)
    return min(max(p, c.clip[0]), c.clip[1])


POLICIES = {"legacy": lambda i: legacy(i), "v2": lambda i: v2(i)}
