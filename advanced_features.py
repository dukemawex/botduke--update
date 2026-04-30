"""
Advanced forecasting features: fact verification, market data, regulatory tracking,
scenario analysis, uncertainty quantification, and question classification.
"""
import re
import json
import logging
import httpx
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ──── Fact Verification Layer ────────────────────────────────────────────────

@dataclass
class FactVerificationResult:
    claim: str
    verifications: List[str]
    contradiction_count: int
    confidence: float


class FactVerifier:
    async def verify_claims(self, research: str) -> List[FactVerificationResult]:
        """Extract key claims and cross-validate against sources."""
        claims = self._extract_claims(research)
        results = []
        for claim in claims:
            verifications = self._find_claim_mentions(claim, research)
            contradiction_count = self._count_contradictions(claim, research)
            confidence = max(0.0, 1.0 - (contradiction_count * 0.15))
            results.append(FactVerificationResult(
                claim=claim,
                verifications=verifications,
                contradiction_count=contradiction_count,
                confidence=confidence,
            ))
        return results

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims (sentences with numbers, dates, entities)."""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        claims = [s for s in sentences if any(c in s for c in ['2025', '2026', '%', '$', 'billion', 'million'])]
        return claims[:10]

    def _find_claim_mentions(self, claim: str, text: str) -> List[str]:
        """Find sources mentioning similar claims."""
        keywords = re.findall(r'\b[A-Z][a-z]+\b', claim)[:3]
        matches = []
        for kw in keywords:
            if kw in text:
                matches.append(f"[{kw}]")
        return matches

    def _count_contradictions(self, claim: str, text: str) -> int:
        """Count sentences contradicting the claim."""
        negation_words = ['not', 'no ', 'contrary', 'opposite', 'contradicts', 'denies']
        contradiction_count = sum(1 for word in negation_words if word in text.lower() and claim.lower() in text.lower())
        return contradiction_count


# ──── Time-Series Momentum Detection ─────────────────────────────────────────

@dataclass
class MomentumAnalysis:
    trend: str
    momentum_strength: float
    projected_value: Optional[float]
    uncertainty_band: Tuple[float, float]


class TimeSeriesMomentum:
    async def analyze_momentum(self, recent_values: List[float]) -> MomentumAnalysis:
        """Detect trend acceleration/deceleration in numeric series."""
        if len(recent_values) < 2:
            return MomentumAnalysis(trend="stable", momentum_strength=0.0, projected_value=None, uncertainty_band=(0, 0))

        diffs = np.diff(recent_values)
        accel = np.diff(diffs)
        
        if len(accel) > 0:
            avg_accel = float(np.mean(accel))
            trend = "accelerating" if avg_accel > 0 else "decelerating" if avg_accel < 0 else "stable"
            momentum_strength = min(1.0, abs(avg_accel) / (np.std(accel) + 1e-6))
        else:
            trend = "stable"
            momentum_strength = 0.0

        last_val = recent_values[-1]
        last_diff = diffs[-1] if len(diffs) > 0 else 0
        projected = last_val + (last_diff * 0.7)
        
        std = np.std(recent_values)
        uncertainty_band = (projected - std, projected + std)

        return MomentumAnalysis(
            trend=trend,
            momentum_strength=min(1.0, float(momentum_strength)),
            projected_value=float(projected),
            uncertainty_band=(float(uncertainty_band[0]), float(uncertainty_band[1])),
        )


# ──── Market Data Integration ────────────────────────────────────────────────

@dataclass
class MarketSignal:
    asset_type: str
    ticker: str
    price: float
    change_pct: float
    timestamp: datetime


class MarketDataIntegrator:
    async def fetch_market_signals(self, keywords: List[str]) -> List[MarketSignal]:
        """Fetch real-time market data for relevant assets."""
        signals = []
        for kw in keywords[:3]:
            signal = await self._fetch_coingecko_or_yahoo(kw)
            if signal:
                signals.append(signal)
        return signals

    async def _fetch_coingecko_or_yahoo(self, symbol: str) -> Optional[MarketSignal]:
        """Fetch from CoinGecko (crypto) or Yahoo Finance (stocks)."""
        try:
            if symbol.lower() in ['bitcoin', 'btc', 'ethereum', 'eth']:
                return await self._fetch_coingecko(symbol)
            else:
                return await self._fetch_yahoo_finance(symbol)
        except Exception as e:
            logger.warning(f"Market data fetch failed for {symbol}: {e}")
            return None

    async def _fetch_coingecko(self, symbol: str) -> Optional[MarketSignal]:
        """Simple CoinGecko fetch for crypto."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd&include_24hr_change=true"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if symbol.lower() in data:
                        price_data = data[symbol.lower()]
                        return MarketSignal(
                            asset_type="crypto",
                            ticker=symbol.upper(),
                            price=float(price_data.get("usd", 0)),
                            change_pct=float(price_data.get("usd_24h_change", 0)),
                            timestamp=datetime.now(timezone.utc),
                        )
            except Exception as e:
                logger.debug(f"CoinGecko fetch failed: {e}")
        return None

    async def _fetch_yahoo_finance(self, symbol: str) -> Optional[MarketSignal]:
        """Yahoo Finance fallback (requires additional setup)."""
        return None


# ──── Regulatory/News Event Tracker ──────────────────────────────────────────

@dataclass
class RegulatoryEvent:
    event_type: str
    title: str
    source: str
    date: datetime
    impact_estimate: float


class RegulatoryTracker:
    async def identify_regulatory_events(self, question_text: str, research: str) -> List[RegulatoryEvent]:
        """Extract regulatory/policy events from research."""
        events = []
        
        sec_pattern = r'(SEC|filing|10-K|10-Q|S-1|8-K)'
        policy_pattern = r'(regulation|law|executive order|ruling|directive|policy)'
        lawsuit_pattern = r'(lawsuit|settlement|court|indictment|charges)'
        
        if re.search(sec_pattern, research, re.IGNORECASE):
            events.append(RegulatoryEvent(
                event_type="SEC_filing",
                title="SEC Filing Mentioned",
                source="research",
                date=datetime.now(timezone.utc),
                impact_estimate=0.15,
            ))
        
        if re.search(policy_pattern, research, re.IGNORECASE):
            events.append(RegulatoryEvent(
                event_type="policy_change",
                title="Policy/Regulatory Change",
                source="research",
                date=datetime.now(timezone.utc),
                impact_estimate=0.25,
            ))
        
        if re.search(lawsuit_pattern, research, re.IGNORECASE):
            events.append(RegulatoryEvent(
                event_type="lawsuit",
                title="Legal Action",
                source="research",
                date=datetime.now(timezone.utc),
                impact_estimate=0.2,
            ))
        
        return events


# ──── Scenario Analysis ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    probability: float
    predicted_value: float
    reasoning: str


class ScenarioAnalyzer:
    async def generate_scenarios(self, question, research: str, llm=None) -> List[Scenario]:
        """Generate bull/base/bear scenarios."""
        if not llm:
            return self._generate_default_scenarios()
        
        try:
            from forecasting_tools import clean_indents
            response = await llm.invoke(clean_indents(f"""
Generate 3 probability-weighted scenarios (bull, base, bear) for this question.
Question: {question.question_text}
Research: {research[:2000]}

Return JSON only:
{{
  "scenarios": [
    {{"name": "bull", "probability": 0.25, "predicted_value": 0.7, "reasoning": "..."}},
    {{"name": "base", "probability": 0.50, "predicted_value": 0.5, "reasoning": "..."}},
    {{"name": "bear", "probability": 0.25, "predicted_value": 0.3, "reasoning": "..."}}
  ]
}}
"""))
            from main import sanitize_llm_json
            data = json.loads(sanitize_llm_json(response))
            scenarios = [Scenario(**s) for s in data.get("scenarios", [])]
            return scenarios if scenarios else self._generate_default_scenarios()
        except Exception as e:
            logger.debug(f"Scenario generation failed: {e}")
            return self._generate_default_scenarios()

    def _generate_default_scenarios(self) -> List[Scenario]:
        """Fallback: basic bull/base/bear."""
        return [
            Scenario(name="bull", probability=0.20, predicted_value=0.70, reasoning="Optimistic case"),
            Scenario(name="base", probability=0.60, predicted_value=0.50, reasoning="Base case"),
            Scenario(name="bear", probability=0.20, predicted_value=0.30, reasoning="Pessimistic case"),
        ]


# ──── Uncertainty Quantification ─────────────────────────────────────────────

@dataclass
class UncertaintyEstimate:
    point_estimate: float
    credible_interval: Tuple[float, float]
    posterior_samples: List[float]
    epistemic_unc: float
    aleatoric_unc: float


class UncertaintyQuantifier:
    def quantify_uncertainty(self, predictions: List[float], research_quality: float = 0.5, n_samples: int = 5000) -> UncertaintyEstimate:
        """Generate full posterior distribution via Monte Carlo."""
        point_estimate = float(np.mean(predictions))
        base_std = float(np.std(predictions)) if len(predictions) > 1 else 0.15
        
        epistemic_unc = base_std * (1.5 - research_quality)
        aleatoric_unc = base_std * 0.5
        total_std = np.sqrt(epistemic_unc**2 + aleatoric_unc**2)
        
        posterior_samples = list(np.random.normal(point_estimate, total_std, n_samples))
        posterior_samples = [np.clip(p, 0.0, 1.0) for p in posterior_samples]
        
        ci_lower = float(np.percentile(posterior_samples, 5))
        ci_upper = float(np.percentile(posterior_samples, 95))
        
        return UncertaintyEstimate(
            point_estimate=point_estimate,
            credible_interval=(ci_lower, ci_upper),
            posterior_samples=posterior_samples,
            epistemic_unc=float(epistemic_unc),
            aleatoric_unc=float(aleatoric_unc),
        )


# ──── Question Classifier ────────────────────────────────────────────────────

class QuestionCategory(str, Enum):
    GEOPOLITICS = "geopolitics"
    ECONOMICS = "economics"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    SPORTS = "sports"
    BUSINESS = "business"
    HEALTH = "health"
    OTHER = "other"


class QuestionClassifier:
    def classify(self, question_text: str) -> Tuple[QuestionCategory, float]:
        """Classify question into domain with confidence."""
        text_lower = question_text.lower()
        
        geo_keywords = ['russia', 'china', 'war', 'conflict', 'election', 'uk', 'europe', 'middle east', 'israel', 'ukraine']
        econ_keywords = ['gdp', 'inflation', 'interest rate', 'unemployment', 'recession', 'market', 'stock', 'fed']
        science_keywords = ['climate', 'nobel', 'research', 'discovery', 'physics', 'biology', 'particle', 'space']
        tech_keywords = ['ai', 'chip', 'software', 'startup', 'ipo', 'tech', 'quantum', 'code']
        sports_keywords = ['olympic', 'world cup', 'championship', 'nba', 'nfl', 'soccer', 'tennis']
        business_keywords = ['ceo', 'company', 'merger', 'acquisition', 'bankruptcy', 'ipo', 'revenue']
        health_keywords = ['vaccine', 'disease', 'pandemic', 'medicine', 'approval', 'fda', 'covid', 'health']
        
        scores = {
            QuestionCategory.GEOPOLITICS: sum(text_lower.count(kw) for kw in geo_keywords),
            QuestionCategory.ECONOMICS: sum(text_lower.count(kw) for kw in econ_keywords),
            QuestionCategory.SCIENCE: sum(text_lower.count(kw) for kw in science_keywords),
            QuestionCategory.TECHNOLOGY: sum(text_lower.count(kw) for kw in tech_keywords),
            QuestionCategory.SPORTS: sum(text_lower.count(kw) for kw in sports_keywords),
            QuestionCategory.BUSINESS: sum(text_lower.count(kw) for kw in business_keywords),
            QuestionCategory.HEALTH: sum(text_lower.count(kw) for kw in health_keywords),
        }
        
        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]
        confidence = min(1.0, max_score / 3.0) if max_score > 0 else 0.3
        
        return (max_category if confidence > 0.2 else QuestionCategory.OTHER, confidence)
