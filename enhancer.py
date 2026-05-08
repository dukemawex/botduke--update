import json
import logging
import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Optional, Tuple

from forecasting_tools import GeneralLlm, clean_indents

logger = logging.getLogger(__name__)


@dataclass
class ResearchSynthesis:
    context_summary: str
    signal_strength: float
    directional_bias: float


class ForecastingEnhancer:
    LAPLACE_NUMERATOR_SMOOTHING = 1.0
    LAPLACE_DENOMINATOR_SMOOTHING = 2.0
    MAX_RESEARCH_CONTEXT_LENGTH = 12000

    # Minibench calibration thresholds from task specification.
    MINIBENCH_SIGNAL_THRESHOLD = 0.80
    MINIBENCH_HIGH_MEDIAN_THRESHOLD = 0.70
    MINIBENCH_LOW_MEDIAN_THRESHOLD = 0.15
    MINIBENCH_HIGH_EXTREMIZED_VALUE = 0.90
    MINIBENCH_LOW_EXTREMIZED_VALUE = 0.03

    _DOMAIN_FRAMEWORKS = {
        "ai_safety": (
            "AI SAFETY MENTAL MODEL:\n"
            "- Focus on capability thresholds, eval evidence, and deployment incentives.\n"
            "- Separate lab claims from independent validation.\n"
            "- Weight governance, regulation, and misuse pathways explicitly."
        ),
        "geopolitics": (
            "GEOPOLITICS MENTAL MODEL:\n"
            "- Use actor incentives, coalition constraints, and escalation ladders.\n"
            "- Prioritize official actions over rhetoric.\n"
            "- Distinguish short-term signaling from durable policy changes."
        ),
        "finance": (
            "FINANCE MENTAL MODEL:\n"
            "- Ground forecasts in base rates, regime shifts, and liquidity conditions.\n"
            "- Separate leading indicators from lagging indicators.\n"
            "- Stress-test upside/downside tails and reflexivity effects."
        ),
        "general": (
            "GENERAL ANALYTICAL MODEL:\n"
            "- Start with base rates, then update with strongest evidence.\n"
            "- Track key assumptions and dominant uncertainty drivers."
        ),
    }

    def route_domain_framework(self, question_text: str, classified_domain: Optional[str] = None) -> Tuple[str, str]:
        text = (question_text or "").lower()
        domain = (classified_domain or "").lower()
        if any(k in text for k in ("ai safety", "alignment", "agi", "frontier model", "model eval", "misuse")):
            route = "ai_safety"
        elif "geopolitics" in domain or any(k in text for k in ("geopolitic", "war", "sanction", "treaty", "election", "conflict")):
            route = "geopolitics"
        elif domain in ("economics", "business") or any(k in text for k in ("finance", "stock", "bond", "inflation", "rate cut", "gdp", "bank")):
            route = "finance"
        else:
            route = "general"
        return route, self._DOMAIN_FRAMEWORKS[route]

    @staticmethod
    def _sanitize_json(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"(?<=\d)_(?=\d)", "", cleaned)
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    @classmethod
    def _heuristic_synthesis(cls, research_text: str) -> ResearchSynthesis:
        text = (research_text or "").lower()
        support_terms = ("confirms", "consistent", "aligned", "corroborated", "supports", "evidence suggests")
        conflict_terms = ("contradict", "unclear", "mixed", "disputed", "inconsistent", "conflict")
        positive_terms = ("increase", "rise", "accelerate", "upside", "strong")
        negative_terms = ("decrease", "fall", "slowdown", "downside", "weak")

        support = sum(text.count(t) for t in support_terms)
        conflict = sum(text.count(t) for t in conflict_terms)
        pos = sum(text.count(t) for t in positive_terms)
        neg = sum(text.count(t) for t in negative_terms)

        # Laplace smoothing (+1 numerator, +2 denominator) prevents 0/0 and overconfident
        # extremes when only a few support/conflict cues are detected.
        signal = cls._clip01(
            (support + cls.LAPLACE_NUMERATOR_SMOOTHING)
            / (support + conflict + cls.LAPLACE_DENOMINATOR_SMOOTHING)
        )
        direction = 0.0 if (pos + neg) == 0 else float((pos - neg) / (pos + neg))
        summary = "Heuristic synthesis used due to synthesis-node unavailability."
        return ResearchSynthesis(
            context_summary=summary,
            signal_strength=signal,
            directional_bias=float(min(1.0, max(-1.0, direction))),
        )

    async def synthesize_research(self, question_text: str, aggregated_research: str) -> ResearchSynthesis:
        prompt = clean_indents(f"""
You are GPT-5.5 Online acting as a research synthesizer for forecasting.
Synthesize all research sources and estimate directional confidence consistency.

Question:
{question_text}

Aggregated Research:
{(aggregated_research or "")[:self.MAX_RESEARCH_CONTEXT_LENGTH]}

Return ONLY JSON:
{{
  "context_summary": "3-6 concise sentences with key qualitative context",
  "signal_strength": 0.0,
  "directional_bias": 0.0
}}

Rules:
- signal_strength must be in [0.0, 1.0] and reflect cross-source consistency.
- directional_bias must be in [-1.0, 1.0], where positive means upward/YES pressure.
- Do not include markdown/code fences.
""")

        for model_name in ("openrouter/openai/gpt-5.5-online", "openrouter/openai/gpt-5.5"):
            try:
                llm = GeneralLlm(model=model_name, temperature=0)
                raw = await llm.invoke(prompt)
                data = json.loads(self._sanitize_json(raw))
                summary = str(data.get("context_summary", "")).strip()
                signal = self._clip01(float(data.get("signal_strength", 0.5)))
                bias = float(data.get("directional_bias", 0.0))
                bias = float(min(1.0, max(-1.0, bias)))
                if summary:
                    return ResearchSynthesis(
                        context_summary=summary,
                        signal_strength=signal,
                        directional_bias=bias,
                    )
            except Exception as e:
                logger.warning(f"Research synthesis failed for {model_name}: {e}")
        return self._heuristic_synthesis(aggregated_research)

    def median_probability(self, probabilities: Iterable[float]) -> float:
        values = []
        for p in probabilities:
            try:
                f = float(p)
                if math.isfinite(f):
                    values.append(self._clip01(f))
            except Exception:
                continue
        if not values:
            logger.warning("Median protocol received no valid probabilities; defaulting to 0.5")
            return 0.5
        return float(median(values))

    def apply_minibench_extremization(self, median_forecast: float, signal_strength: float) -> float:
        m = self._clip01(float(median_forecast))
        s = self._clip01(float(signal_strength))
        if s > self.MINIBENCH_SIGNAL_THRESHOLD:
            if m >= self.MINIBENCH_HIGH_MEDIAN_THRESHOLD:
                return self.MINIBENCH_HIGH_EXTREMIZED_VALUE
            if m <= self.MINIBENCH_LOW_MEDIAN_THRESHOLD:
                return self.MINIBENCH_LOW_EXTREMIZED_VALUE
        return m
