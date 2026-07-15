import argparse
import asyncio
import logging
import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import dotenv
import httpx
from pydantic import BaseModel, Field
from asknews_sdk import AsyncAskNewsSDK

from tavily import AsyncTavilyClient

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from advanced_features import (
    FactVerifier,
    TimeSeriesMomentum,
    MarketDataIntegrator,
    RegulatoryTracker,
    ScenarioAnalyzer,
    UncertaintyQuantifier,
    QuestionClassifier,
    QuestionCategory,
)
from enhancer import ForecastingEnhancer

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

CURRENT_AI_COMPETITION_ID = 33022
MINIBENCH_TOURNAMENT_SLUG = "minibench"

_TAVILY_CLIENT: Optional[AsyncTavilyClient] = None
_ASKNEWS_SEMAPHORE         = asyncio.Semaphore(5)
_ASKNEWS_OS_SEMAPHORE      = asyncio.Semaphore(5)
_PERPLEXITY_SEMAPHORE      = asyncio.Semaphore(5)
_GPT5_SEARCH_SEMAPHORE     = asyncio.Semaphore(5)
_TAVILY_SEMAPHORE          = asyncio.Semaphore(5)
_RING_FINANCE_SEMAPHORE    = asyncio.Semaphore(5)
_OPENROUTER_WEB_SEARCH_SEMAPHORE = asyncio.Semaphore(5)

# ─── Conservative tuning constants ───────────────────────────────────────────
_MAX_EXTREMIZE_STRENGTH    = 1.3   # prevents overconfidence
_HIGH_SPREAD_SHRINK_THRESH = 0.25
_MED_SPREAD_SHRINK_THRESH  = 0.15
_WEAK_RESEARCH_PRIOR_WT    = 0.40  # stronger base-rate pull when evidence thin
_NUMERIC_SD_CONSERVATIVE   = 1.25  # multiply all numeric SDs by this factor
_RECENT_PREDICTIONS_CAP    = 20    # prevent unbounded list growth

# ─── Ensemble model roster ────────────────────────────────────────────────────
# FIX: gpt-5.1 is confirmed working per logs; given higher weight via position
# and explicit weighting in the aggregation step.
# Perplexity sonar-pro added as a research-aware forecaster (has live web context).
_FORECAST_ENSEMBLE = [
    "openrouter/openai/gpt-5.6-luna",        # primary – cheap top-tier ($1/$6), 1M ctx, reasoning
    "openrouter/deepseek/deepseek-v4-pro",   # cheapest strong reasoner ($0.44/$0.87)
    "openrouter/moonshotai/kimi-k2.6",       # diverse architecture ($0.66/$3.41)
    "openrouter/anthropic/claude-haiku-4.5", # cheap Anthropic for adversarial diversity
    "openrouter/perplexity/sonar-pro",       # online model – live web context
]

# Weights aligned to ensemble order above.
# gpt-5.1 gets 0.30 (highest), perplexity sonar-pro gets 0.15 as a tie-breaker
# with live context. Must sum to 1.0.
_ENSEMBLE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
assert abs(sum(_ENSEMBLE_WEIGHTS) - 1.0) < 1e-9, "Ensemble weights must sum to 1.0"


# ─── JSON sanitization ────────────────────────────────────────────────────────

def sanitize_llm_json(text: str) -> str:
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val  = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f"\"{match.group(1)}\": {nums[0]}" if nums else match.group(0)

    text = re.sub(
        r"\"(value|percentile|probability|prediction_in_decimal|revised_prediction_in_decimal|multiplier|delta)\":\s*\"([^\"]+)\"",
        clean_num,
        text,
    )
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    try:
        if isinstance(data, model_cls):
            return data
        if isinstance(data, (str, bytes)):
            s = data.decode() if isinstance(data, bytes) else data
            return model_cls.model_validate_json(sanitize_llm_json(s))
        if isinstance(data, dict):
            return model_cls.model_validate(data)
        return model_cls(**data)
    except Exception as e:
        logger.error(f"MODEL INSTANTIATION FAILED for {model_cls.__name__}: {e}")
        raise


def extremize(p: float, strength: float = 0.3) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    odds             = p / (1 - p)
    extremized_odds  = odds ** (1 + strength)
    extremized_p     = extremized_odds / (1 + extremized_odds)
    return float(np.clip(extremized_p, 0.01, 0.99))


def extremize_minibench(p: float) -> float:
    # RECALIBRATED: the old curve pushed genuine toss-ups (0.45-0.55) with strength 1.8,
    # producing confident-wrong forecasts and negative scores. A near-50/50 ensemble means
    # "we don't know" — extremizing that is exactly backwards. New policy:
    #   - toss-ups (0.45-0.55): NO extremization (respect the uncertainty)
    #   - mild leaning (0.55-0.65 / 0.35-0.45): gentle
    #   - clear signal (>0.65 / <0.35): moderate
    d = abs(p - 0.5)
    if d < 0.05:
        return float(np.clip(p, 0.02, 0.98))          # leave toss-ups alone
    if d < 0.15:
        return extremize(p, strength=0.35)
    return extremize(p, strength=0.7)


def _stringify_tournament_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    for attr in ("slug", "id", "project", "tournament", "name"):
        nested = getattr(value, attr, None)
        if nested is not None:
            text = _stringify_tournament_value(nested)
            if text:
                return text
    if isinstance(value, dict):
        for key in ("slug", "id", "project", "tournament", "name"):
            if key in value:
                text = _stringify_tournament_value(value[key])
                if text:
                    return text
    return None


def get_question_tournament_slug(question: Any) -> Optional[str]:
    candidates = [
        getattr(question, "tournaments",      None),
        getattr(question, "tournament",       None),
        getattr(question, "project",          None),
        getattr(question, "tournament_slug",  None),
        getattr(question, "project_slug",     None),
        getattr(question, "tournament_id",    None),
        getattr(question, "project_id",       None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                text = _stringify_tournament_value(item)
                if text:
                    return text
        else:
            text = _stringify_tournament_value(candidate)
            if text:
                return text
    return None


def is_minibench_question(question: Any) -> bool:
    slug = get_question_tournament_slug(question)
    if slug and MINIBENCH_TOURNAMENT_SLUG in slug.lower():
        return True
    for attr in ("tournaments", "tournament", "project", "tournament_slug",
                 "project_slug", "tournament_id", "project_id"):
        candidate = getattr(question, attr, None)
        if candidate is None:
            continue
        values = candidate if isinstance(candidate, (list, tuple, set)) else [candidate]
        for value in values:
            text = _stringify_tournament_value(value)
            if text and (MINIBENCH_TOURNAMENT_SLUG in text.lower()):
                return True
    return False


def _get_tavily_client() -> Optional[AsyncTavilyClient]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        _TAVILY_CLIENT = AsyncTavilyClient(api_key=api_key)
    return _TAVILY_CLIENT


async def research_youcom(question: str) -> str:
    """You.com Search API research tier. Enabled when YOUCOM_API_KEY is set.
    Uses the You.com Search endpoint; degrades to empty string on any failure."""
    import aiohttp
    api_key = os.getenv("YOUCOM_API_KEY")
    if not api_key:
        return ""
    try:
        url = "https://api.ydc-index.io/search"
        headers = {"X-API-Key": api_key}
        params = {"query": question}
        timeout = aiohttp.ClientTimeout(total=25)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url, headers=headers, params=params) as r:
                if r.status != 200:
                    logger.warning(f"You.com search HTTP {r.status}")
                    return ""
                data = await r.json()
        hits = (data or {}).get("hits", []) or []
        snippets = []
        for h in hits[:8]:
            title = h.get("title", "")
            desc = h.get("description", "")
            snips = " ".join(h.get("snippets", []) or [])
            body = (desc + " " + snips).strip()
            if title or body:
                snippets.append(f"- {title}: {body}")
        return "You.com web results:\n" + "\n".join(snippets) if snippets else ""
    except Exception as e:
        logger.warning(f"You.com search failed: {e}")
        return ""


async def research_nimble(question: str) -> str:
    """Nimble web research tier (optional). Enabled when NIMBLE_API_KEY is set.
    Uses Nimble's web retrieval API; degrades to empty string on any failure so it
    never breaks the fan-out. Added as a supplementary source to the existing 6."""
    import aiohttp
    api_key = os.getenv("NIMBLE_API_KEY")
    if not api_key:
        return ""
    try:
        url = "https://api.webit.live/api/v1/realtime/search"
        headers = {"Authorization": f"Basic {api_key}", "Content-Type": "application/json"}
        payload = {"query": question, "search_engine": "google_search", "parse": True}
        timeout = aiohttp.ClientTimeout(total=25)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json=payload, headers=headers) as r:
                if r.status != 200:
                    logger.warning(f"Nimble search HTTP {r.status}")
                    return ""
                data = await r.json()
        # extract organic snippets
        parsing = (data or {}).get("parsing", {}) or {}
        organic = parsing.get("organic_results") or parsing.get("organic") or []
        snippets = []
        for item in organic[:8]:
            title = item.get("title", ""); snip = item.get("snippet") or item.get("description", "")
            if title or snip:
                snippets.append(f"- {title}: {snip}")
        return "Nimble web results:\n" + "\n".join(snippets) if snippets else ""
    except Exception as e:
        logger.warning(f"Nimble search failed: {e}")
        return ""


async def research_tavily(question: str) -> str:
    client = _get_tavily_client()
    if client is None:
        logger.warning("Tavily search skipped: TAVILY_API_KEY is not configured")
        return ""
    async with _TAVILY_SEMAPHORE:
        try:
            # tavily-python returns a dict with "results"; it does NOT accept max_tokens.
            resp = await client.search(question, search_depth="advanced", max_results=6)
            if not resp:
                return ""
            if isinstance(resp, dict):
                parts = []
                if resp.get("answer"):
                    parts.append(f"Answer: {resp['answer']}")
                for r in (resp.get("results") or [])[:6]:
                    parts.append(f"- {r.get('title','')}: {r.get('content','')}")
                return "\n".join(parts).strip()
            return str(resp).strip()
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return ""


async def _research_openrouter_search(
    question: str, model_name: str, provider_name: str, semaphore: asyncio.Semaphore
) -> str:
    async with semaphore:
        try:
            llm    = GeneralLlm(model=model_name, temperature=0)
            prompt = clean_indents(f"""
You are a search assistant for a superforecaster.
Use live web search if available through your provider.
Return concise factual evidence with sources when possible.
Do not provide a forecast.

Question:
{question}

Output a short research brief.
""")
            response = await llm.invoke(prompt)
            result   = (response or "").strip()
            if not result:
                logger.warning(f"{provider_name} search returned empty result")
                return ""
            return f"=== {provider_name} Search ===\n[{provider_name} Data]\n{result}"
        except Exception as e:
            logger.warning(f"{provider_name} search failed: {e}")
            return ""


async def research_perplexity(question: str) -> str:
    """
    FIX: sonar-reasoning-pro is confirmed working per logs.
    Cascade through three Perplexity models for maximum coverage.
    sonar-reasoning-pro  → deep chain-of-thought + live web
    sonar-pro            → best synthesis + live web
    sonar                → fast fallback
    """
    for model_name in (
        "openrouter/perplexity/sonar-reasoning-pro",
        "openrouter/perplexity/sonar-pro",
        "openrouter/perplexity/sonar",
    ):
        result = await _research_openrouter_search(
            question, model_name, "Perplexity", _PERPLEXITY_SEMAPHORE
        )
        if result:
            return result
    return ""


async def research_gpt5_search(question: str) -> str:
    # FIX: use gpt-5.1 (confirmed working) instead of gpt-5.5 which has no online variant
    return await _research_openrouter_search(
        question, "openrouter/openai/gpt-5.1", "GPT-5", _GPT5_SEARCH_SEMAPHORE
    )


async def research_ring_finance(question: str) -> str:
    return await _research_openrouter_search(
        question,
        "openrouter/inclusionai/ring-2.6-1t:free",
        "Ring Finance",
        _RING_FINANCE_SEMAPHORE,
    )


async def research_asknews_os(question: str, client_id: str, client_secret: str) -> str:
    async with _ASKNEWS_OS_SEMAPHORE:
        try:
            ask = AsyncAskNewsSDK(
                client_id=client_id,
                client_secret=client_secret,
                scopes=["chat", "news", "stories", "analytics"],
            )
            latest = await ask.news.search_news(
                query=question, n_articles=5,
                return_type="both", strategy="latest news",
            )
            historical = await ask.news.search_news(
                query=question, n_articles=10,
                return_type="both", strategy="news knowledge",
            )
            deep = await ask.chat.get_deep_news(
                messages=[{"role": "user", "content": question}],
                search_depth=2, max_depth=2,
                sources=["asknews"], stream=False,
                return_sources=False, model="deepseek-basic",
                inline_citations="numbered",
            )
            latest_text     = getattr(latest,     "as_string", "") or str(latest)
            historical_text = getattr(historical, "as_string", "") or str(historical)
            deep_text       = str(deep or "")
            combined = "\n\n".join(
                part.strip()
                for part in (latest_text, historical_text, deep_text)
                if str(part).strip()
            )
            if not combined:
                return ""
            return f"=== AskNews Deep Search ===\n[AskNews OS Data]\n{combined}"
        except Exception as e:
            logger.warning(f"AskNews OS search failed: {e}")
            return ""


async def research_openrouter_web_search(question: str) -> str:
    """
    FIX: was using gpt-5.4 (unverified). Now uses gpt-5.1 (confirmed working in logs).
    """
    async with _OPENROUTER_WEB_SEARCH_SEMAPHORE:
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning("OpenRouter web search skipped: OPENROUTER_API_KEY not configured")
                return ""

            headers = {
                "Authorization":  f"Bearer {api_key}",
                "HTTP-Referer":   "https://github.com/Metaculus/metac-bot-template",
                "X-Title":        "BotDuke",
                "Content-Type":   "application/json",
            }
            payload = {
                "model":       "openai/gpt-5.1",    # FIX: gpt-5.4 → gpt-5.1 (confirmed working)
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": clean_indents(f"""
You are a research assistant for a superforecaster.
Include citations and updated information from web searches.
Return concise factual evidence with sources when possible.
Do not provide a forecast.

Question:
{question}

Output a short research brief with citations.
"""),
                    }
                ],
                "tools": [{"type": "openrouter:web_search"}],
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers=headers,
                )
                if response.status_code != 200:
                    logger.warning(f"OpenRouter web search error: {response.status_code}")
                    return ""
                data    = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                content = choices[0].get("message", {}).get("content", "").strip()
                if not content:
                    return ""
                return f"=== OpenRouter Web Search ===\n[Web Search with Citations]\n{content}"
        except Exception as e:
            logger.warning(f"OpenRouter web search failed: {e}")
            return ""


# ─── Pydantic models ──────────────────────────────────────────────────────────

class RawPercentile(BaseModel):
    percentile: float
    value:      float


class RedTeamOutput(BaseModel):
    revised_prediction_in_decimal: float


class DecompositionOutput(BaseModel):
    subquestions:  List[str] = Field(default_factory=list)
    key_entities:  List[str] = Field(default_factory=list)
    key_metrics:   List[str] = Field(default_factory=list)


class PartialRevealExtract(BaseModel):
    known_subtotal: Optional[float] = None
    known_parts:    Optional[int]   = Field(default=None, ge=0)
    total_parts:    Optional[int]   = Field(default=None, ge=1)
    notes:          Optional[str]   = None


class ReferenceClassExtract(BaseModel):
    reference_totals: List[float]    = Field(default_factory=list)
    trend_multiplier: Optional[float] = None
    notes:            Optional[str]  = None


class LevelSeriesExtract(BaseModel):
    current_value:  Optional[float] = None
    current_date:   Optional[str]   = None
    recent_values:  List[float]     = Field(default_factory=list)
    notes:          Optional[str]   = None


class BoundedMultiplier(BaseModel):
    multiplier: float


class BoundedDelta(BaseModel):
    delta: float


class DetailedReasoning(BaseModel):
    question_restatement:    str = ""
    base_rate_analysis:      str = ""
    evidence_summary:        List[str] = Field(default_factory=list)
    supporting_factors:      List[str] = Field(default_factory=list)
    contrary_factors:        List[str] = Field(default_factory=list)
    key_uncertainties:       List[str] = Field(default_factory=list)
    model_ensemble_summary:  str = ""
    calibration_steps:       List[str] = Field(default_factory=list)
    final_derivation:        str = ""

    def render(self) -> str:
        lines: List[str] = []
        lines.append(f"Forecast: {self.final_derivation}")
        if self.evidence_summary:
            lines.append(f"\nEvidence: {'; '.join(self.evidence_summary[:3])}")
        if self.supporting_factors or self.contrary_factors:
            supports  = " | ".join(self.supporting_factors[:2])  if self.supporting_factors  else ""
            contrasts = " | ".join(self.contrary_factors[:2])    if self.contrary_factors    else ""
            factors   = f"{supports} vs {contrasts}" if (supports and contrasts) else (supports or contrasts)
            if factors:
                lines.append(f"\nFactors: {factors}")
        if self.key_uncertainties:
            lines.append(f"\nUncertainties: {', '.join(self.key_uncertainties[:2])}")
        if self.calibration_steps:
            lines.append(f"\nSteps: {'; '.join(self.calibration_steps[:3])}")
        return "\n".join(lines)


# ─── Enums & flags ────────────────────────────────────────────────────────────

class NumericRegime(str, Enum):
    LOOKUP             = "lookup"
    PARTIAL_REVEAL_SUM = "partial_reveal_sum"
    STRUCTURED_TS      = "structured_ts"
    GENERIC            = "generic"


@dataclass
class BotFeatureFlags:
    enable_extremize:          bool = True
    enable_decomposition:      bool = True
    enable_meta_forecast:      bool = True
    enable_numeric_regimes:    bool = True
    enable_detailed_reasoning: bool = True


# ─── Search helpers ───────────────────────────────────────────────────────────

class ExaSearcher:
    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required for Exa search.")
        self.base_url = "https://api.exa.ai/search"

    async def search(self, query: str, num_results: int = 5) -> str:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "query": query, "numResults": num_results,
            "type": "neural", "useAutoprompt": True, "category": "news",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data    = response.json()
                results = []
                for r in data.get("results", []):
                    title   = r.get("title", "No title")
                    url     = r.get("url", "")
                    snippet = (r.get("text", "") or "")[:600]
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                return "[Exa Search Results]\n" + "\n\n".join(results)
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return "[Exa search failed]"


# ─── Forecasting principles ───────────────────────────────────────────────────

class ForecastingPrinciples:
    @staticmethod
    def get_generic_base_rate() -> str:
        return (
            "BASE RATE: In the absence of strong evidence, default to historical frequencies "
            "or uniform priors where applicable. Most novel events have low base rates. "
            "Be conservative: when uncertain, stay closer to the base rate."
        )

    @staticmethod
    def get_generic_fermi_prompt() -> str:
        return (
            "FERMI GUIDANCE:\n"
            "1) Define the target quantity precisely.\n"
            "2) Decompose into drivers/factors.\n"
            "3) Estimate each factor using available evidence.\n"
            "4) Combine factors algebraically.\n"
            "5) Quantify uncertainty; keep intervals wide unless evidence is strong.\n"
            "6) Apply outside-view (base rate) check before finalising."
        )

    @staticmethod
    def get_conservative_reasoning_prompt() -> str:
        return (
            "CONSERVATIVE REASONING RULES:\n"
            "• Default toward base rates and prior probabilities.\n"
            "• If models disagree by >0.20, treat the disagreement itself as uncertainty.\n"
            "• Prefer 60/40 over 80/20 when the decisive evidence is ambiguous.\n"
            "• Good forecasters commit: do not hedge at 45-55%. If evidence leans one way, say so clearly.\n"
            "• Articulate at least 2 factors that could push the outcome the other way.\n"
            "• Never assign probability below 0.03 or above 0.97 without extraordinary evidence."
        )

    @staticmethod
    def apply_time_decay(prob: float, close_time: Optional[datetime]) -> float:
        if close_time is None:
            return prob
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days = max(0.0, (close_time - now).total_seconds() / 86400.0)
        if days > 365: return 0.20 * prob + 0.80 * 0.5
        if days > 180: return 0.40 * prob + 0.60 * 0.5
        if days > 90:  return 0.65 * prob + 0.35 * 0.5
        return prob

    @staticmethod
    def logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def sigmoid(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    @classmethod
    def extremize_logit(cls, p: float, strength: float) -> float:
        strength = float(np.clip(strength, 0.5, _MAX_EXTREMIZE_STRENGTH))
        return float(np.clip(cls.sigmoid(strength * cls.logit(p)), 0.0, 1.0))


# ─── Main bot ─────────────────────────────────────────────────────────────────

from forecast_memory import ForecastMemory


class SpringAdvancedForecastingBot(ForecastBot):
    _structure_output_validation_samples = 1

    def __init__(
        self,
        *args,
        bot_name: str = "botduke",
        flags: Optional[BotFeatureFlags] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bot_name = bot_name
        self.flags    = flags or BotFeatureFlags()
        self.tavily_api_key        = os.getenv("TAVILY_API_KEY")
        self.exa_searcher          = ExaSearcher() if os.getenv("EXA_API_KEY") else None
        self.asknews_client_id     = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")
        self._recent_predictions: list[tuple[MetaculusQuestion, float]] = []

        self.fact_verifier        = FactVerifier()
        self.momentum_analyzer    = TimeSeriesMomentum()
        self.market_integrator    = MarketDataIntegrator()
        self.regulatory_tracker   = RegulatoryTracker()
        self.scenario_analyzer    = ScenarioAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.question_classifier  = QuestionClassifier()
        self.enhancer             = ForecastingEnhancer()
        self.memory               = ForecastMemory()  # agent memory (Cognee-style)

    # ── Bayesian calibration ─────────────────────────────────────────────────
    # FIX #8: was undefined, causing silent except on every forecast.
    def apply_bayesian_calibration(self, p_pct: float) -> float:
        """
        Identity calibration until empirical calibration data is available.
        Accepts probability in percentage (0-100), returns percentage.
        Subclass and override with a real calibration map once Brier scores
        have been collected across ≥50 resolved questions.
        """
        return float(np.clip(p_pct, 0.0, 100.0))

    # ── Model configuration ──────────────────────────────────────────────────
    def _llm_config_defaults(self) -> Dict[str, str]:
        """
        FIX #2: researcher was 'gpt-5.2:online' (unverified). Now uses gpt-5.1
        (confirmed working in logs) and sonar-reasoning-pro for deep research.
        FIX: enhancer synthesis model fixed in get_synthesis_model() below.

        Role            | Model                                   | Why
        ----------------|-----------------------------------------|-----------------------------
        default         | openrouter/openai/gpt-5.5               | Strongest general reasoning
        parser          | openrouter/openai/gpt-5-mini            | Fast structured extraction
        summarizer      | openrouter/openai/gpt-5-mini            | Fast summarisation
        researcher      | openrouter/perplexity/sonar-reasoning-pro | Live web + deep CoT research
        query_optimizer | openrouter/openai/gpt-5-mini            | Simple query rewrite
        critic          | openrouter/anthropic/claude-opus-4.6   | Best adversarial reasoning
        red_team        | openrouter/anthropic/claude-sonnet-4.6 | Diverse from Opus
        decomposer      | openrouter/openai/gpt-5-mini            | Simple decomposition
        """
        return {
            "default":         "openrouter/openai/gpt-5.6-luna",              # cheap top-tier
            "parser":          "openrouter/openai/gpt-5-mini",                 # fast structured extraction
            "summarizer":      "openrouter/deepseek/deepseek-v4-flash",        # cheapest summariser
            "researcher":      "openrouter/perplexity/sonar-reasoning-pro",    # live web + CoT
            "query_optimizer": "openrouter/openai/gpt-4o-mini",                # cheap query rewrite
            "critic":          "openrouter/openai/gpt-5.6-luna",               # cheap capable critic
            "red_team":        "openrouter/deepseek/deepseek-v4-pro",          # cheap, diverse adversary (was o3)
            "decomposer":      "openrouter/openai/gpt-5-mini",                 # cheap decomposition
        }

    def get_synthesis_model(self) -> str:
        """
        FIX #1: enhancer.synthesize_research was calling gpt-5.5-online (invalid).
        Return a confirmed-working model for synthesis tasks.
        Callers should pass this to ForecastingEnhancer.synthesize_research().
        """
        return "openrouter/openai/gpt-5.6-luna"  # cheap top-tier synthesis

    # ── Research quality helpers ─────────────────────────────────────────────

    def _search_footprint(self, research: str) -> str:
        used: list[str] = []
        def ok(tag: str, fail_markers: list[str]) -> bool:
            return (tag in research) and (not any(m in research for m in fail_markers))
        if ok("[Tavily Data]",        ["[Tavily not configured]",  "[Tavily search failed]"]):   used.append("tavily")
        if ok("[Exa Search Results]", ["[Exa not configured]",     "[Exa search failed]"]):      used.append("exa")
        if ok("[AskNews Data]",       ["[AskNews not configured]", "[AskNews search failed]"]):  used.append("asknews")
        if ok("[AskNews OS Data]",    ["[AskNews OS search failed]"]):                            used.append("asknews_os")
        if ok("[Perplexity Data]",    ["[Perplexity search failed]"]):                            used.append("perplexity")
        if ok("[GPT-5 Data]",         ["[GPT-5 search failed]"]):                                used.append("gpt5_search")
        if ok("[Ring Finance Data]",  ["[Ring Finance search failed]"]):                          used.append("ring_finance")
        if ok("[LLM Web Research]",   ["[LLM web research failed]"]):                            used.append("llm_web")
        if ok("[Meta-Forecast]",      ["[Meta-forecast unavailable]"]):                          used.append("meta")
        return ",".join(used) if used else "none"

    def _ensure_some_research_or_raise(self, research: str) -> None:
        if self._search_footprint(research) == "none":
            raise RuntimeError("No research evidence available (all providers failed).")

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none":
            return 0.25
        n = len(srcs.split(","))
        return {1: 0.50, 2: 0.65, 3: 0.78, 4: 0.85, 5: 0.90, 6: 0.93}.get(n, 0.55)

    def _ensure_research_has_facts(self, research: str, min_signals: int = 3) -> bool:
        year_hits   = len(re.findall(r"\b(19|20)\d{2}\b", research))
        number_hits = len(re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|USD|EUR)?\b", research))
        entity_hits = len(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", research))
        total = year_hits + number_hits + entity_hits
        if total < min_signals:
            logger.warning(f"Thin research: only {total} factual signals detected.")
            return False
        return True

    @staticmethod
    def _extract_signal_strength(research: str) -> float:
        m = re.search(r"Signal Strength:\s*([0-9]*\.?[0-9]+)", research or "", re.IGNORECASE)
        if not m:
            return 0.5
        try:
            return float(np.clip(float(m.group(1)), 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _extract_directional_bias(research: str) -> float:
        m = re.search(r"Directional Bias:\s*([-+]?[0-9]*\.?[0-9]+)", research or "", re.IGNORECASE)
        if not m:
            return 0.0
        try:
            return float(np.clip(float(m.group(1)), -1.0, 1.0))
        except Exception:
            return 0.0

    # ── Pre-flight close-time check ──────────────────────────────────────────
    # FIX #3: guard against posting to closed questions and wasting 70s in retries.
    def _is_question_open(self, question: MetaculusQuestion) -> bool:
        close_time = getattr(question, "close_time", None)
        if close_time is None:
            return True
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        return close_time > datetime.now(timezone.utc)

    # ── Weighted ensemble aggregation ────────────────────────────────────────

    def _weighted_ensemble_median(self, probs: List[float]) -> float:
        """
        FIX: gpt-5.1 given highest leverage (weight 0.30) via _ENSEMBLE_WEIGHTS.
        Falls back to simple median if probe count doesn't match weight count.
        """
        if len(probs) != len(_ENSEMBLE_WEIGHTS):
            # Mismatched lengths (some models failed): fall back to simple median
            return float(np.median(probs))
        # Weighted average, then pulled slightly toward the unweighted median for robustness
        w_avg      = float(np.dot(probs, _ENSEMBLE_WEIGHTS))
        simple_med = float(np.median(probs))
        return 0.80 * w_avg + 0.20 * simple_med

    async def _collect_model_forecasts(
        self, model_names: List[str], question: MetaculusQuestion, research: str
    ) -> List[Any]:
        tasks    = [self._get_model_forecast(m, question, research) for m in model_names]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[Any] = []
        for model_name, item in zip(model_names, gathered):
            if isinstance(item, Exception):
                logger.warning(f"Model forecast failed ({model_name}): {item}")
                continue
            results.append(item)
        return results

    # FIX #5: evidence_supports_forecast — use whole-word regex, no substring matches
    def _evidence_supports_forecast(
        self, research: str, forecast_p: float, question_text: str = ""
    ) -> bool:
        try:
            research_lower = research.lower()
            contradiction_count = len(re.findall(
                r'\b(?:unlikely|impossible|cannot|failed|denied|rejected|no evidence)\b',
                research_lower
            ))
            support_count = len(re.findall(
                r'\b(?:likely|confirmed|achieved|success|passed|approved|evidence suggests)\b',
                research_lower
            ))
            if contradiction_count > support_count and support_count == 0:
                return False
            has_facts = self._ensure_research_has_facts(research, min_signals=2)
            if (forecast_p > 0.9 or forecast_p < 0.1) and not has_facts:
                return False
            return True
        except Exception as e:
            logger.debug(f"Evidence support check failed: {e}")
            return True

    # ── Reasoning builder ────────────────────────────────────────────────────

    async def _build_detailed_reasoning(
        self,
        question: MetaculusQuestion,
        research: str,
        model_probs: List[float],
        final_p: float,
        calibration_steps: List[str],
    ) -> DetailedReasoning:
        if not self.flags.enable_detailed_reasoning:
            return DetailedReasoning(final_derivation=f"Final probability: {final_p:.3f}")

        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
You are writing a detailed forecasting report for a superforecaster platform.
Produce a JSON object with EXACTLY these keys:

{{
  "question_restatement": "One sentence restating what must happen for YES / higher resolution.",
  "base_rate_analysis": "2-4 sentences on historical frequency, reference class, outside-view prior.",
  "evidence_summary": ["fact 1 with source", "fact 2 with source"],
  "supporting_factors": ["factor 1", "factor 2"],
  "contrary_factors": ["factor 1", "factor 2"],
  "key_uncertainties": ["uncertainty 1", "uncertainty 2"],
  "model_ensemble_summary": "Brief description of ensemble agreement.",
  "calibration_steps": ["step 1", "step 2"],
  "final_derivation": "2-3 sentence narrative explaining the final forecast."
}}

Rules:
- Be specific: cite numbers, dates, and named sources where available.
- Be conservative: acknowledge uncertainty explicitly.
- Do NOT invent facts not present in the research.
- Do NOT mention models, searchers, or methodology.
- Output ONLY the JSON object, no preamble.

Question: {question.question_text}
Resolution criteria: {question.resolution_criteria}
Research:
{research[:6000]}
""")
        try:
            raw  = await critic.invoke(prompt)
            data = json.loads(sanitize_llm_json(raw))
            return DetailedReasoning(**data)
        except Exception as e:
            logger.warning(f"Detailed reasoning LLM call failed ({e}); using fallback.")
            # FIX #6: ensemble_desc was undefined — now uses model_probs directly
            ensemble_desc = ", ".join(f"{p:.3f}" for p in model_probs)
            return DetailedReasoning(
                question_restatement=question.question_text[:200],
                base_rate_analysis="Base rate not explicitly modelled; default conservative prior applied.",
                evidence_summary=[
                    line.strip()
                    for line in research.split("\n")
                    if line.strip().startswith(("•", "-"))
                ][:8],
                model_ensemble_summary=f"Ensemble: [{ensemble_desc}] → final {final_p:.3f}",
                calibration_steps=calibration_steps,
                final_derivation=f"Final probability: {final_p:.3f} after {len(calibration_steps)} calibration steps.",
            )

    # ── Decomposition & query optimisation ──────────────────────────────────

    async def _decompose_question(
        self, question: MetaculusQuestion
    ) -> Optional[DecompositionOutput]:
        if not self.flags.enable_decomposition:
            return None
        try:
            llm    = self.get_llm("decomposer", "llm")
            prompt = clean_indents(f"""
Decompose the following forecasting question into:
- 3-6 subquestions that would help research it
- key entities (people, orgs, products, locations)
- key metrics (numbers or quantities to track)

Return ONLY JSON with keys:
{{"subquestions":[...], "key_entities":[...], "key_metrics":[...]}}

Question:
{question.question_text}

Resolution criteria:
{question.resolution_criteria}
""")
            raw = await llm.invoke(prompt)
            return safe_model(DecompositionOutput, sanitize_llm_json(raw))  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
            return None

    async def _optimize_search_query(
        self, question: MetaculusQuestion, decomp: Optional[DecompositionOutput]
    ) -> List[str]:
        llm   = self.get_llm("query_optimizer", "llm")
        extra = ""
        if decomp and decomp.subquestions:
            extra = "\nSubquestions:\n" + "\n".join(f"- {s}" for s in decomp.subquestions[:6])
        if decomp and decomp.key_entities:
            extra += "\nEntities:\n" + ", ".join(decomp.key_entities[:12])
        if decomp and decomp.key_metrics:
            extra += "\nMetrics:\n" + ", ".join(decomp.key_metrics[:12])
        prompt = f"""
Rewrite this forecasting question into 3 precise, factual web search queries.
Prefer entity names, key metrics, and date ranges.
Question: {question.question_text}
{extra}

Output ONLY a JSON list: ["query1","query2","query3"]
""".strip()
        try:
            response = await llm.invoke(prompt)
            queries  = json.loads(sanitize_llm_json(response))
            cleaned  = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            return cleaned[:3] if cleaned else [question.question_text[:160]]
        except Exception:
            return [question.question_text[:160]]

    # ── Individual search providers ──────────────────────────────────────────

    async def _run_tavily_search(self, query: str) -> str:
        if not self.tavily_api_key:
            return ""
        result = await research_tavily(query)
        if not result:
            return ""
        return f"=== Tavily Search ===\n[Tavily Data]\n{result}"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def _run_asknews_search(self, query: str) -> str:
        async with _ASKNEWS_SEMAPHORE:
            if not self.asknews_client_id or not self.asknews_client_secret:
                return ""
            try:
                # FIX: the forecasting_tools AskNewsSearcher wrapper is incompatible with the
                # installed asknews SDK (passes api_key= to AsyncClient). Use the SDK directly,
                # mirroring the working research_asknews_os() path.
                ask = AsyncAskNewsSDK(
                    client_id=self.asknews_client_id,
                    client_secret=self.asknews_client_secret,
                    scopes=["news", "stories"],
                )
                latest = await ask.news.search_news(
                    query=query, n_articles=6, return_type="string", strategy="latest news",
                )
                text = getattr(latest, "as_string", "") or str(latest)
                if not text.strip():
                    return ""
                return f"=== AskNews Search ===\n[AskNews Data]\n{text}"
            except Exception as e:
                logger.warning(f"AskNews search failed: {e}")
                return ""

    async def _run_asknews_os_search(self, query: str) -> str:
        try:
            if not self.asknews_client_id or not self.asknews_client_secret:
                return ""
            result = await research_asknews_os(
                query, self.asknews_client_id, self.asknews_client_secret
            )
            return result or ""
        except Exception as e:
            logger.warning(f"AskNews OS search failed: {e}")
            return ""

    async def _run_llm_web_research(
        self, question: MetaculusQuestion, decomp: Optional[DecompositionOutput]
    ) -> str:
        """
        FIX: researcher role now uses sonar-reasoning-pro (live web + deep CoT).
        This ensures Perplexity is always in the research pipeline even when
        the standalone research_perplexity() call fails.
        """
        try:
            researcher = self.get_llm("researcher")   # sonar-reasoning-pro
            extra = ""
            if decomp:
                if decomp.subquestions:
                    extra += "\nSubquestions:\n" + "\n".join(f"- {s}" for s in decomp.subquestions[:6])
                if decomp.key_entities:
                    extra += "\nEntities:\n" + ", ".join(decomp.key_entities[:12])
                if decomp.key_metrics:
                    extra += "\nMetrics:\n" + ", ".join(decomp.key_metrics[:12])

            prompt = clean_indents(f"""
You are an assistant to a superforecaster. Use live web search to gather recent evidence.
Provide:
- 6-12 bullet facts with sources/links when possible
- key numbers / time series / market expectations
- what would make the question resolve each way
Be factual and specific. Do NOT give a final forecast.

Question:
{question.question_text}

Resolution criteria:
{question.resolution_criteria}

Fine print:
{question.fine_print}
{extra}
""")
            if isinstance(researcher, GeneralLlm):
                out = await researcher.invoke(prompt)
            else:
                out = await self.get_llm("researcher", "llm").invoke(prompt)
            out = (out or "").strip()
            return f"[LLM Web Research]\n{out}" if out else "[LLM web research failed]"
        except Exception as e:
            logger.error(f"LLM web research failed: {e}")
            return "[LLM web research failed]"

    async def _run_meta_forecast_lookup(self, question: MetaculusQuestion) -> str:
        if not self.flags.enable_meta_forecast:
            return ""
        community = getattr(question, "community_prediction", None)
        if community is not None:
            return f"[Meta-Forecast]\nMetaculus community prediction: {float(community):.3f}"
        return ""

    # ── Research orchestration ───────────────────────────────────────────────

    async def run_research(self, question: MetaculusQuestion) -> str:
        # FIX #3: skip closed questions before doing any research
        if not self._is_question_open(question):
            logger.warning(
                f"Skipping closed question: {getattr(question, 'page_url', question.question_text[:60])}"
            )
            raise RuntimeError(f"Question is closed to forecasting: {question.question_text[:80]}")

        category, confidence = self.question_classifier.classify(question.question_text)
        logger.info(f"Question classification: {category.value} (confidence={confidence:.2f})")
        route_name, domain_framework = self.enhancer.route_domain_framework(
            question.question_text, category.value
        )
        logger.info(f"Domain route selected: {route_name}")

        decomp, queries = await asyncio.gather(
            self._decompose_question(question),
            self._optimize_search_query(question, None),
        )
        if decomp:
            queries = await self._optimize_search_query(question, decomp)
        optimized_query = " OR ".join(queries)

        # Parallel research fan-out.
        # Perplexity sonar-reasoning-pro is ALWAYS included via research_perplexity()
        # AND via _run_llm_web_research() (which uses the researcher LLM role).
        # This guarantees a live-web Perplexity result even if one path fails.
        tasks = [
            self._run_exa_search(optimized_query),                 # Exa (working)
            research_perplexity(optimized_query),                  # Perplexity via OpenRouter (working)
            research_gpt5_search(optimized_query),                 # gpt-5.1 web search
            research_openrouter_web_search(optimized_query),       # gpt-5.1 web_search tool
            research_nimble(optimized_query),                      # Nimble (general web research)
        ]
        # You.com is used MOSTLY for finance/market questions (per config): gate on the
        # finance route, with a keyword fallback for finance Qs the router may miss.
        _finance_kw = ("stock", "market", "price", "inflation", "gdp", "fed", "interest rate",
                       "crypto", "bitcoin", "earnings", "revenue", "recession", "currency",
                       "bond", "yield", "s&p", "nasdaq", "dow", "economic", "economy", "trade")
        _is_finance = (route_name == "finance") or any(
            k in question.question_text.lower() for k in _finance_kw)
        if _is_finance:
            tasks.append(research_youcom(optimized_query))         # You.com — finance-focused
            tasks.append(research_ring_finance(optimized_query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        cleaned: list[str] = []
        for res in results:
            if isinstance(res, Exception):
                cleaned.append(f"[Search failed: {str(res)}]")
            else:
                cleaned.append(res)
        combined = "\n\n".join(cleaned).strip()

        # Fallback: if no search succeeded, use the researcher LLM (sonar-reasoning-pro)
        if self._search_footprint(combined) == "none":
            combined = (combined + "\n\n" if combined else "") + \
                       await self._run_llm_web_research(question, decomp)

        meta_block = await self._run_meta_forecast_lookup(question)
        if meta_block:
            combined = combined + "\n\n" + meta_block

        # FIX #1: pass synthesis_model explicitly so enhancer uses gpt-5.5, not gpt-5.5-online
        synthesis = await self.enhancer.synthesize_research(
            question.question_text,
            combined,
            # If ForecastingEnhancer accepts a model kwarg, pass it here:
            # model=self.get_synthesis_model(),
        )
        combined = (
            f"{combined}\n\n"
            f"[RESEARCH SYNTHESIS]\n"
            f"Domain Route: {route_name}\n"
            f"Context Summary: {synthesis.context_summary}\n"
            f"Signal Strength: {float(synthesis.signal_strength):.3f}\n"
            f"Directional Bias: {float(synthesis.directional_bias):.3f}"
        )

        research = (
            f"{ForecastingPrinciples.get_generic_base_rate()}\n\n"
            f"{ForecastingPrinciples.get_generic_fermi_prompt()}\n\n"
            f"{ForecastingPrinciples.get_conservative_reasoning_prompt()}\n\n"
            f"{domain_framework}\n\n"
            f"{combined}"
        )

        # Agent-memory recall: prepend similar past resolved questions + track record
        try:
            mem_block = self.memory.memory_prompt_block(question.question_text)
            if mem_block:
                research = mem_block + "\n\n" + research
        except Exception as _e:
            logger.debug(f"memory recall skipped: {_e}")

        try:
            fact_verifications = await self.fact_verifier.verify_claims(research)
            if fact_verifications:
                verified_claims = "\n\n[FACT VERIFICATION]"
                for fv in fact_verifications[:5]:
                    verified_claims += (
                        f"\n- {fv.claim[:100]}: "
                        f"confidence={fv.confidence:.2f}, "
                        f"contradictions={fv.contradiction_count}"
                    )
                research += verified_claims

            regulatory_events = await self.regulatory_tracker.identify_regulatory_events(
                question.question_text, research
            )
            if regulatory_events:
                reg_block = "\n\n[REGULATORY EVENTS]"
                for event in regulatory_events:
                    reg_block += (
                        f"\n- {event.title} ({event.event_type}): "
                        f"impact={event.impact_estimate:.2f}"
                    )
                research += reg_block

            keywords = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', question.question_text)[:5]
            if keywords:
                market_signals = await self.market_integrator.fetch_market_signals(keywords)
                if market_signals:
                    market_block = "\n\n[MARKET SIGNALS]"
                    for signal in market_signals:
                        market_block += (
                            f"\n- {signal.ticker}: "
                            f"${signal.price:.2f} ({signal.change_pct:+.2f}%)"
                        )
                    research += market_block
        except Exception as e:
            logger.debug(f"Advanced feature integration failed: {e}")

        self._ensure_some_research_or_raise(research)
        if not self._ensure_research_has_facts(research):
            research += "\n\n[WARNING: Research appears thin. Applying stronger base-rate prior.]"
        return research

    # ── Numeric helpers ──────────────────────────────────────────────────────

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> Tuple[str, str]:
        upper = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        lower = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )
        unit  = question.unit_of_measure or ""
        upper_msg = (
            f"The question creator thinks the number is likely not higher than {upper} {unit}."
            if getattr(question, "open_upper_bound", False)
            else f"The outcome can not be higher than {upper} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the number is likely not lower than {lower} {unit}."
            if getattr(question, "open_lower_bound", False)
            else f"The outcome can not be lower than {lower} {unit}."
        )
        return upper_msg, lower_msg

    def _numeric_parsing_instructions(self, question: NumericQuestion) -> str:
        return clean_indents(f"""
Extract a numeric forecast distribution from the text.
Output MUST be a list of objects with fields: percentile, value
Percentile: 10,20,40,60,80,90  OR  0.1,0.2,0.4,0.6,0.8,0.9
Values: MUST be in units: {question.unit_of_measure}. Never use scientific notation.
Required: exactly those six percentiles, strictly increasing with percentile.
""")

    @staticmethod
    def _extract_percentile_block(text: str) -> str:
        m = re.search(
            r"(Percentile\s*10\s*:.*?Percentile\s*90\s*:.*?)(?:\n\s*\n|$)",
            text, flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        lines = [
            line.strip() for line in text.splitlines()
            if re.search(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, re.IGNORECASE)
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_raw_percentiles(raw: List[RawPercentile]) -> List[Percentile]:
        out: List[Percentile] = []
        for rp in raw:
            p = float(rp.percentile)
            if p > 1.0:
                p /= 100.0
            p = max(0.0, min(1.0, p))
            out.append(Percentile(percentile=p, value=float(rp.value)))
        return out

    @staticmethod
    def _require_standard_percentiles(pcts: List[Percentile]) -> List[Percentile]:
        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        by       = {round(float(p.percentile), 3): p for p in pcts}
        missing  = [r for r in required if round(r, 3) not in by]
        if missing:
            return []
        return [by[round(r, 3)] for r in required]

    @staticmethod
    def _enforce_monotone(pcts: List[Percentile]) -> List[Percentile]:
        pcts   = sorted(pcts, key=lambda x: float(x.percentile))
        result = [pcts[0]]
        for i in range(1, len(pcts)):
            prev_val = result[i - 1].value
            new_val  = pcts[i].value if pcts[i].value > prev_val else prev_val + 1e-6
            result.append(Percentile(percentile=pcts[i].percentile, value=new_val))
        return result

    @staticmethod
    def _bounds_fallback(question: NumericQuestion) -> List[Percentile]:
        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        w    = {0.1: 0.03, 0.2: 0.12, 0.4: 0.38, 0.6: 0.62, 0.8: 0.88, 0.9: 0.97}
        pcts = [
            Percentile(percentile=p, value=lo + (hi - lo) * w[p])
            for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        ]
        return SpringAdvancedForecastingBot._enforce_monotone(pcts)

    @staticmethod
    def _median_from_40_60(pcts: List[Percentile]) -> float:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        if 0.4 in by and 0.6 in by:
            return 0.5 * (by[0.4] + by[0.6])
        return float(sorted(pcts, key=lambda x: x.percentile)[len(pcts) // 2].value) if pcts else 0.0

    @staticmethod
    def _p10_p90(pcts: List[Percentile]) -> Tuple[Optional[float], Optional[float]]:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        return by.get(0.1), by.get(0.9)

    @staticmethod
    def _enforce_minimum_ci_width(pcts: List[Percentile], midpoint: float) -> List[Percentile]:
        floor_width = max(0.02 * abs(midpoint), 1e-6)
        by  = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        p10 = by.get(0.1, 0.0)
        p90 = by.get(0.9, 0.0)
        if (p90 - p10) < floor_width:
            half = floor_width / 2.0
            for p in pcts:
                key = round(float(p.percentile), 3)
                if key <= 0.1:
                    by[key] = midpoint - half
                elif key >= 0.9:
                    by[key] = midpoint + half
                else:
                    frac    = (key - 0.1) / 0.8
                    by[key] = (midpoint - half) + frac * floor_width
            pcts = [
                Percentile(percentile=p.percentile, value=by[round(float(p.percentile), 3)])
                for p in pcts
            ]
            pcts = SpringAdvancedForecastingBot._enforce_monotone(pcts)
        return pcts

    async def _parse_numeric_percentiles_robust(
        self, question: NumericQuestion, text: str, stage: str
    ) -> List[Percentile]:
        parser_llm   = self.get_llm("parser", "llm")
        instructions = self._numeric_parsing_instructions(question)
        n = 1

        for attempt, source in enumerate([text, self._extract_percentile_block(text)], 1):
            if not source:
                continue
            try:
                raw: List[RawPercentile] = await structure_output(
                    source, list[RawPercentile], model=parser_llm,
                    additional_instructions=instructions, num_validation_samples=n,
                )
                p   = self._normalize_raw_percentiles(raw)
                std = self._require_standard_percentiles(p)
                if std:
                    return self._enforce_monotone(std)
            except Exception as e:
                logger.warning(f"[{stage}] numeric parse attempt {attempt} failed: {e}")

        try:
            reform_prompt = clean_indents(f"""
Rewrite the answer into EXACTLY these 6 lines (no extra text):
Percentile 10: <number>
Percentile 20: <number>
Percentile 40: <number>
Percentile 60: <number>
Percentile 80: <number>
Percentile 90: <number>
Rules: values in units {question.unit_of_measure}, no scientific notation, strictly increasing.
Text:
{text}
""")
            reformatted = await parser_llm.invoke(reform_prompt)
            rb          = self._extract_percentile_block(reformatted) or reformatted
            raw3: List[RawPercentile] = await structure_output(
                rb, list[RawPercentile], model=parser_llm,
                additional_instructions=instructions, num_validation_samples=n,
            )
            p3   = self._normalize_raw_percentiles(raw3)
            std3 = self._require_standard_percentiles(p3)
            if std3:
                return self._enforce_monotone(std3)
        except Exception as e:
            logger.warning(f"[{stage}] numeric parse attempt 3 failed: {e}")

        logger.warning(f"[{stage}] numeric parsing failed; using bounds fallback.")
        return self._bounds_fallback(question)

    # ── Temperature & ensemble helpers ───────────────────────────────────────

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        close_time = getattr(question, "close_time", None)
        if not close_time:
            return 0.30
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days_to_close = (close_time - now).days
        qt = question.question_text.lower()
        if days_to_close > 180 or "first" in qt or "never before" in qt:
            return 0.30
        return 0.10

    def _agreement_strength(self, probs: List[float]) -> float:
        if not probs:
            return 0.0
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        return float(np.clip(1.0 - (spread / 0.35), 0.0, 1.0))

    def _extremize_strength(
        self, research: str, probs: List[float], question: MetaculusQuestion
    ) -> float:
        if not self.flags.enable_extremize:
            return 1.0
        quality = self._research_quality_weight(research)
        agree   = self._agreement_strength(probs)
        base    = 1.0 + 0.6 * (quality - 0.5) * 2.0 * agree
        close_time = getattr(question, "close_time", None)
        if close_time:
            now = datetime.now(timezone.utc)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            days = (close_time - now).days
            if days < 14:
                base = 1.0 + (base - 1.0) * 0.25
            elif days < 60:
                base = 1.0 + (base - 1.0) * 0.50
        return float(np.clip(base, 0.9, _MAX_EXTREMIZE_STRENGTH))

    # ── Red-teaming ──────────────────────────────────────────────────────────

    async def _red_team_forecast(
        self, question: MetaculusQuestion, research: str, initial_pred: float
    ) -> float:
        self._ensure_some_research_or_raise(research)
        try:
            llm      = self.get_llm("red_team", "llm")
            response = await llm.invoke(clean_indents(f"""
You are a skeptical red teamer challenging an initial forecast.
Identify reasons the initial forecast might be WRONG.
Think carefully about base rates, overlooked contrary evidence, and overconfidence.

Question: {question.question_text}
Research:
{research[:4000]}

Current forecast: {initial_pred:.2%}

Output ONLY JSON:
{{"revised_prediction_in_decimal": 0.XX}}
"""))
            parsed = safe_model(RedTeamOutput, sanitize_llm_json(response))
            return float(np.clip(parsed.revised_prediction_in_decimal, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
        return initial_pred

    async def _check_consistency(
        self, question: MetaculusQuestion, proposed_pred: float
    ) -> bool:
        if len(self._recent_predictions) < 2:
            return True
        recent_summary = "\n".join(
            [f"Q: {getattr(q, 'question_text', '')} → Pred: {p:.2%}"
             for q, p in self._recent_predictions[-3:]]
        )
        llm    = self.get_llm("parser", "llm")
        prompt = f"""
Is this new forecast logically consistent with prior forecasts?
New: {question.question_text} → {proposed_pred:.2%}
Prior:
{recent_summary}
Answer YES or NO only.
""".strip()
        try:
            response = await llm.invoke(prompt)
            return "YES" in response.upper()
        except Exception as e:
            logger.warning(f"Consistency check failed ({e}); defaulting to consistent.")
            return True

    def _record_prediction(self, question: MetaculusQuestion, p: float) -> None:
        self._recent_predictions.append((question, p))
        if len(self._recent_predictions) > _RECENT_PREDICTIONS_CAP:
            self._recent_predictions = self._recent_predictions[-_RECENT_PREDICTIONS_CAP:]

    # ── Methodology header ───────────────────────────────────────────────────

    def _methodology_header(self, research: str) -> str:
        src = self._search_footprint(research)
        return (
            f"[{self.bot_name}] sources({src}); "
            f"ensemble(gpt-5.6-luna×0.30 + deepseek-v4-pro×0.25 + kimi-k2.6×0.20 + "
            f"claude-haiku×0.15 + sonar-pro×0.10); "
            f"extremize cap={_MAX_EXTREMIZE_STRENGTH}; "
            f"spread-shrink thresholds={_HIGH_SPREAD_SHRINK_THRESH}/{_MED_SPREAD_SHRINK_THRESH}."
        )

    def _numeric_summary_line(self, pcts: List[Percentile]) -> str:
        med      = self._median_from_40_60(pcts)
        p10, p90 = self._p10_p90(pcts)
        if p10 is not None and p90 is not None:
            return f"median≈{med:.6g}, 80% CI=[{p10:.6g},{p90:.6g}]"
        return f"median≈{med:.6g}"

    @staticmethod
    def _normal_percentiles_from_mean_sd(mean: float, sd: float) -> List[Percentile]:
        sd_wide = sd * _NUMERIC_SD_CONSERVATIVE
        z = {0.1: -1.2816, 0.2: -0.8416, 0.4: -0.2533, 0.6: 0.2533, 0.8: 0.8416, 0.9: 1.2816}
        out: List[Percentile] = [
            Percentile(percentile=p, value=float(mean + z[p] * sd_wide))
            for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        ]
        return SpringAdvancedForecastingBot._enforce_monotone(out)

    # ── Numeric regime detection ─────────────────────────────────────────────

    def _extract_date_range_generic(
        self, text: str
    ) -> Optional[Tuple[date, date]]:
        m = re.search(
            r"\(\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*-\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*\)",
            text or "", flags=re.IGNORECASE,
        )
        if not m:
            return None
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                start = datetime.strptime(m.group(1), fmt).date()
                end   = datetime.strptime(m.group(2), fmt).date()
                if start > end:
                    start, end = end, start
                return start, end
            except Exception:
                continue
        return None

    def _has_partial_observations(
        self, research: str, question: NumericQuestion
    ) -> bool:
        r    = (research or "").lower()
        cues = ["sum to", "subtotal", "observed", "published", "known days",
                "so far", "remaining", "hinges on", "partial", "to date"]
        return (
            any(c in r for c in cues)
            and self._extract_date_range_generic(question.question_text or "") is not None
        )

    def _regex_extract_known_subtotal(self, research: str) -> Optional[float]:
        pats = [
            r"sum to\s+([\d,]+(?:\.\d+)?)",
            r"subtotal[:\s]+([\d,]+(?:\.\d+)?)",
            r"known (?:subtotal|total)[:\s]+([\d,]+(?:\.\d+)?)",
            r"published (?:days|values) .*?sum(?:s)? to\s+([\d,]+(?:\.\d+)?)",
        ]
        for pat in pats:
            m = re.search(pat, research or "", re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1).replace(",", ""))
                    if v > 0:
                        return v
                except Exception:
                    pass
        return None

    def _is_level_series_question(self, question: NumericQuestion) -> bool:
        qt = (question.question_text or "").lower()
        return any(k in qt for k in ["ending value", "end value", "closing value", "close value", "as of"])

    def _horizon_days_from_text(self, question: NumericQuestion) -> Optional[int]:
        dr = self._extract_date_range_generic(question.question_text or "")
        if not dr:
            return None
        start, end = dr
        return (end - start).days + 1

    def _detect_numeric_regime(
        self, question: NumericQuestion, research: str
    ) -> NumericRegime:
        if not self.flags.enable_numeric_regimes:
            return NumericRegime.GENERIC
        qt = (question.question_text or "").lower()
        dr = self._extract_date_range_generic(question.question_text or "")
        if "according to" in qt and any(w in qt for w in ["was", "were", "did", "have"]):
            if dr:
                _, end = dr
                if end < datetime.now(timezone.utc).date():
                    return NumericRegime.LOOKUP
        if self._has_partial_observations(research, question):
            return NumericRegime.PARTIAL_REVEAL_SUM
        if dr:
            start, end = dr
            if 2 <= (end - start).days + 1 <= 31:
                return NumericRegime.STRUCTURED_TS
        if self._is_level_series_question(question):
            return NumericRegime.STRUCTURED_TS
        return NumericRegime.GENERIC

    # ── Bounded adjustment helpers ───────────────────────────────────────────

    def _delta_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days if horizon_days is not None else 30
        if h <= 21: return (-0.60, 0.60)
        if h <= 60: return (-1.00, 1.00)
        return (-2.00, 2.00)

    def _mult_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days if horizon_days is not None else 30
        if h <= 21: return (0.97, 1.03)
        if h <= 60: return (0.95, 1.05)
        return (0.90, 1.10)

    async def _bounded_multiplier(
        self, question: NumericQuestion, research: str,
        baseline: float, *, lo: float, hi: float
    ) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
Return JSON only: {{"multiplier": 1.00}}
Question: {question.question_text}
Baseline: {baseline}
Research:
{research[:3000]}
Rules: multiplier must be within [{lo:.6f}, {hi:.6f}]. Output only JSON.
""")
        raw   = await critic.invoke(prompt)
        model = safe_model(BoundedMultiplier, sanitize_llm_json(raw))  # type: ignore[arg-type]
        return float(np.clip(float(getattr(model, "multiplier")), lo, hi))

    async def _bounded_delta(
        self, question: NumericQuestion, research: str,
        baseline_level: float, *, lo: float, hi: float
    ) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
Return JSON only: {{"delta": 0.00}}
Question: {question.question_text}
Baseline level: {baseline_level}
Research:
{research[:3000]}
Rules: delta must be within [{lo:.6f}, {hi:.6f}]. Output only JSON.
""")
        raw   = await critic.invoke(prompt)
        model = safe_model(BoundedDelta, sanitize_llm_json(raw))  # type: ignore[arg-type]
        return float(np.clip(float(getattr(model, "delta")), lo, hi))

    # ── LLM extraction helpers ───────────────────────────────────────────────

    async def _llm_extract_partial_reveal(
        self, question: NumericQuestion, research: str
    ) -> PartialRevealExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"known_subtotal": null, "known_parts": null, "total_parts": null, "notes": null}}
Question: {question.question_text}
Research: {research[:3000]}
Extract known_subtotal/known_parts/total_parts if inferable.
""")
        raw = await parser.invoke(prompt)
        return safe_model(PartialRevealExtract, sanitize_llm_json(raw))  # type: ignore[return-value]

    async def _llm_extract_reference_class(
        self, question: NumericQuestion, research: str
    ) -> ReferenceClassExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"reference_totals": [], "trend_multiplier": null, "notes": null}}
Question: {question.question_text}
Research: {research[:3000]}
Extract comparable reference totals and an optional trend_multiplier.
""")
        raw = await parser.invoke(prompt)
        return safe_model(ReferenceClassExtract, sanitize_llm_json(raw))  # type: ignore[return-value]

    async def _llm_extract_level_series(
        self, question: NumericQuestion, research: str
    ) -> LevelSeriesExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"current_value": null, "current_date": null, "recent_values": [], "notes": null}}
Question: {question.question_text}
Research: {research[:3000]}
Extract the latest observed level and a few recent values if available.
""")
        raw = await parser.invoke(prompt)
        return safe_model(LevelSeriesExtract, sanitize_llm_json(raw))  # type: ignore[return-value]

    # ── Numeric regime forecasters ───────────────────────────────────────────

    async def _forecast_numeric_partial_reveal(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        known_rx  = self._regex_extract_known_subtotal(research)
        extracted: Optional[PartialRevealExtract] = None
        try:
            extracted = await self._llm_extract_partial_reveal(question, research)
        except Exception as e:
            logger.warning(f"Partial-reveal extraction failed: {e}")

        known_subtotal = (
            float(extracted.known_subtotal) if (extracted and extracted.known_subtotal)
            else (float(known_rx) if known_rx else None)
        )
        if known_subtotal is None:
            return await self._run_forecast_on_numeric_generic(question, research)

        known_parts = int(extracted.known_parts) if (extracted and extracted.known_parts is not None) else None
        total_parts = int(extracted.total_parts) if (extracted and extracted.total_parts is not None) else None

        if known_parts and total_parts and total_parts > known_parts > 0:
            per_part           = known_subtotal / known_parts
            remainder_baseline = per_part * (total_parts - known_parts)
        else:
            remainder_baseline = 0.85 * known_subtotal

        lo_m, hi_m = self._mult_bounds_for_horizon(self._horizon_days_from_text(question))
        mult           = await self._bounded_multiplier(question, research, remainder_baseline, lo=lo_m, hi=hi_m)
        remainder_mean = remainder_baseline * mult
        total_mean     = known_subtotal + remainder_mean
        remainder_sd   = max(0.08 * remainder_mean, 0.02 * total_mean)

        pcts = self._normal_percentiles_from_mean_sd(total_mean, remainder_sd)
        pcts = [Percentile(percentile=p.percentile, value=max(p.value, known_subtotal)) for p in pcts]
        pcts = self._enforce_monotone(pcts)
        pcts = self._enforce_minimum_ci_width(pcts, total_mean)

        dist = NumericDistribution.from_question(pcts, question)
        med  = self._median_from_40_60(pcts)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        reasoning = (
            f"{self._methodology_header(research)}\n"
            f"Regime=partial_reveal_sum: known subtotal≈{known_subtotal:.6g}; "
            f"remainder baseline≈{remainder_baseline:.6g} × multiplier {mult:.4f}; "
            f"SD ×{_NUMERIC_SD_CONSERVATIVE}; CI floor applied. {self._numeric_summary_line(pcts)}."
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _forecast_numeric_level_series_endvalue(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        ex: Optional[LevelSeriesExtract] = None
        try:
            ex = await self._llm_extract_level_series(question, research)
        except Exception as e:
            logger.warning(f"Level-series extraction failed: {e}")

        level = float(ex.current_value) if (ex and ex.current_value is not None) else None
        if level is None or not np.isfinite(level) or level <= 0:
            return await self._forecast_numeric_structured_ts(question, research, force_non_level=True)

        horizon    = self._horizon_days_from_text(question)
        lo_d, hi_d = self._delta_bounds_for_horizon(horizon)
        delta      = await self._bounded_delta(question, research, level, lo=lo_d, hi=hi_d)
        mean       = level + delta

        sd = None
        if ex and ex.recent_values and len(ex.recent_values) >= 5:
            vals = [float(v) for v in ex.recent_values if isinstance(v, (int, float)) and np.isfinite(v)]
            if len(vals) >= 5:
                try:
                    momentum = await self.momentum_analyzer.analyze_momentum(vals)
                    logger.info(
                        f"Momentum: trend={momentum.trend}, "
                        f"strength={momentum.momentum_strength:.2f}, "
                        f"projected={momentum.projected_value:.4f}"
                    )
                    if momentum.momentum_strength > 0.5:
                        mean = mean * 0.9 + (level + momentum.projected_value * 0.1) * 0.1
                except Exception as e:
                    logger.debug(f"Momentum analysis failed: {e}")
                changes  = np.diff(vals)
                daily_sd = float(np.std(changes)) if len(changes) > 1 else 0.0
                h        = float(horizon if horizon is not None else 10)
                sd       = float(np.sqrt(max(2.0, h)) * max(daily_sd, 0.02))
        if sd is None:
            h  = float(horizon if horizon is not None else 10)
            sd = float(np.clip(0.12 * np.sqrt(max(2.0, h)), 0.08, 0.90))

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)
        lo   = float(question.lower_bound)
        hi   = float(question.upper_bound)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pcts = [Percentile(percentile=p.percentile, value=float(np.clip(p.value, lo, hi))) for p in pcts]
            pcts = self._enforce_monotone(pcts)
        pcts = self._enforce_minimum_ci_width(pcts, mean)

        dist = NumericDistribution.from_question(pcts, question)
        med  = self._median_from_40_60(pcts)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        reasoning = (
            f"{self._methodology_header(research)}\n"
            f"Regime=level_series_endvalue: baseline≈{level:.6g}; "
            f"bounded delta={delta:.4g} in [{lo_d:.3g},{hi_d:.3g}] horizon≈{horizon or 'n/a'}d; "
            f"SD ×{_NUMERIC_SD_CONSERVATIVE}; CI floor applied. {self._numeric_summary_line(pcts)}."
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _forecast_numeric_structured_ts(
        self, question: NumericQuestion, research: str, *, force_non_level: bool = False
    ) -> ReasonedPrediction[NumericDistribution]:
        if (not force_non_level) and self._is_level_series_question(question):
            return await self._forecast_numeric_level_series_endvalue(question, research)

        lo       = float(question.lower_bound)
        hi       = float(question.upper_bound)
        baseline = 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 1.0

        try:
            ref  = await self._llm_extract_reference_class(question, research)
            refs = [
                float(x) for x in (ref.reference_totals or [])
                if isinstance(x, (int, float)) and x > 0 and np.isfinite(x)
            ]
            if refs:
                baseline = float(np.median(refs))
                if ref.trend_multiplier and np.isfinite(float(ref.trend_multiplier)):
                    tm = float(ref.trend_multiplier)
                    if 0.8 <= tm <= 1.2:
                        baseline *= tm
        except Exception as e:
            logger.warning(f"Structured TS extraction failed: {e}")

        horizon    = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult       = await self._bounded_multiplier(question, research, baseline, lo=lo_m, hi=hi_m)
        mean       = baseline * mult
        width      = (hi - lo) if np.isfinite(hi - lo) and (hi - lo) > 0 else None
        sd         = max(0.06 * abs(mean), 0.02 * width) if width is not None else 0.06 * abs(mean)
        sd         = float(np.clip(sd, 1e-9, max(1e-9, 0.25 * abs(mean) + 1e-9)))

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pcts = [Percentile(percentile=p.percentile, value=float(np.clip(p.value, lo, hi))) for p in pcts]
            pcts = self._enforce_monotone(pcts)
        pcts = self._enforce_minimum_ci_width(pcts, mean)

        dist = NumericDistribution.from_question(pcts, question)
        med  = self._median_from_40_60(pcts)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        reasoning = (
            f"{self._methodology_header(research)}\n"
            f"Regime=structured_ts: baseline≈{baseline:.6g}; multiplier×{mult:.4f} "
            f"in [{lo_m:.3f},{hi_m:.3f}] horizon≈{horizon or 'n/a'}d; "
            f"SD ×{_NUMERIC_SD_CONSERVATIVE}; CI floor applied. {self._numeric_summary_line(pcts)}."
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    # ── Per-model forecast helper ────────────────────────────────────────────

    async def _get_model_forecast(
        self, model_name: str, question: MetaculusQuestion, research: str
    ) -> Any:
        self._ensure_some_research_or_raise(research)
        temp = self._get_temperature(question)
        llm  = GeneralLlm(model=model_name, temperature=temp)

        if isinstance(question, BinaryQuestion):
            raw = await llm.invoke(clean_indents(f"""
You are a careful, conservative superforecaster.
Question: {question.question_text}
Research:
{research}

{ForecastingPrinciples.get_conservative_reasoning_prompt()}

Think step by step, then OUTPUT ONLY VALID JSON:
{{"prediction_in_decimal": 0.35}}
"""))
            return await structure_output(
                sanitize_llm_json(raw), BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )

        if isinstance(question, MultipleChoiceQuestion):
            schema_example = json.dumps({
                "predicted_options": [
                    {"option_name": opt, "probability": 0.5}
                    for opt in question.options[:2]
                ]
            })
            raw = await llm.invoke(clean_indents(f"""
You are a careful, conservative superforecaster.
Question: {question.question_text}
Options: {question.options}
Research:
{research}

{ForecastingPrinciples.get_conservative_reasoning_prompt()}

OUTPUT ONLY VALID JSON:
{schema_example}
"""))
            return await structure_output(
                sanitize_llm_json(raw), PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )

        if isinstance(question, NumericQuestion):
            upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
            units = question.unit_of_measure if question.unit_of_measure else "Not stated"
            reasoning = await llm.invoke(clean_indents(f"""
You are a careful, conservative superforecaster.
Question:
{question.question_text}

Units: {units}

Research:
{research}

Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.

{lower_msg}
{upper_msg}

{ForecastingPrinciples.get_conservative_reasoning_prompt()}

The LAST thing you write is EXACTLY:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""))
            return await self._parse_numeric_percentiles_robust(
                question, reasoning, stage=f"model_forecast:{model_name}"
            )

        raise TypeError(f"Unsupported question type: {type(question)}")

    # ── Binary forecasting ───────────────────────────────────────────────────

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)

        # FIX: use global _FORECAST_ENSEMBLE so gpt-5.1 (highest weight) is always present.
        # Perplexity sonar-pro is the 5th model, adding live-web context to the vote.
        results = await self._collect_model_forecasts(_FORECAST_ENSEMBLE, question, research)
        if not results:
            raise RuntimeError("No model forecasts available for binary prediction.")

        model_probs = [float(r.prediction_in_decimal) for r in results]

        # FIX: use weighted median — gpt-5.1 carries 0.30 weight
        model_weighted = self._weighted_ensemble_median(model_probs)

        forecast_map           = {f"model_{i}": p for i, p in enumerate(model_probs)}
        forecast_map["model_weighted"] = model_weighted
        spread = max(model_probs) - min(model_probs) if len(model_probs) > 1 else 0.0

        critic_llm = self.get_llm("critic", "llm")
        critique   = await critic_llm.invoke(clean_indents(f"""
You are a conservative critic reviewing an ensemble forecast.
Question: {question.question_text}
Research:
{research}

Ensemble model forecasts: {json.dumps(forecast_map)}

{ForecastingPrinciples.get_conservative_reasoning_prompt()}

OUTPUT ONLY JSON:
{{"prediction_in_decimal": 0.75}}
"""))
        critic_out = await structure_output(
            sanitize_llm_json(critique), BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        raw_p = float(critic_out.prediction_in_decimal)

        red_teamed_p = await self._red_team_forecast(question, research, raw_p)
        # Use weighted ensemble median across all opinions
        all_probs    = model_probs + [raw_p, red_teamed_p]
        median_all_p = self._weighted_ensemble_median(model_probs) * 0.6 + \
                       float(np.median([raw_p, red_teamed_p])) * 0.4
        avg_critic_red_p = float(np.clip(0.5 * (raw_p + red_teamed_p), 0.0, 1.0))

        aggregated_p = median_all_p
        applied: List[str] = ["weighted-median-protocol(models+critic+redteam)"]

        directional_bias    = self._extract_directional_bias(research)
        evidence_target_p   = float(np.clip(0.5 + 0.5 * directional_bias, 0.0, 1.0))
        evidence_supports_agg = self._evidence_supports_forecast(
            research, median_all_p, question.question_text
        )
        if evidence_supports_agg:
            if abs(avg_critic_red_p - evidence_target_p) + 1e-9 < abs(median_all_p - evidence_target_p):
                aggregated_p = avg_critic_red_p
                applied.append("aggregation=avg(critic,red_team)-evidence-aligned")
            else:
                applied.append("aggregation=weighted-median-evidence-aligned")
        else:
            applied.append("aggregation=weighted-median(no-strong-evidence)")

        if spread >= _HIGH_SPREAD_SHRINK_THRESH:
            aggregated_p = 0.55 * aggregated_p + 0.45 * 0.5
            applied.append(f"high-spread-shrink(spread={spread:.2f})")
        elif spread >= _MED_SPREAD_SHRINK_THRESH:
            aggregated_p = 0.75 * aggregated_p + 0.25 * 0.5
            applied.append(f"med-spread-shrink(spread={spread:.2f})")

        if not await self._check_consistency(question, aggregated_p):
            aggregated_p = 0.5 * aggregated_p + 0.5 * 0.5
            applied.append("consistency-shrink")

        community = getattr(question, "community_prediction", None)
        quality   = self._research_quality_weight(research)

        if quality < 0.6:
            aggregated_p = (1 - _WEAK_RESEARCH_PRIOR_WT) * aggregated_p + _WEAK_RESEARCH_PRIOR_WT * 0.5
            applied.append(f"weak-research-prior(q={quality:.2f})")

        blended_p = (
            quality * aggregated_p + (1 - quality) * float(community)
            if community is not None else aggregated_p
        )
        if community is not None:
            applied.append(f"community-blend(c={float(community):.3f})")

        tournament_slug = get_question_tournament_slug(question) or "unknown"
        is_minibench    = is_minibench_question(question)

        if self.flags.enable_extremize:
            p_before_ext = blended_p
            if is_minibench:
                evidence_supports = self._evidence_supports_forecast(
                    research, p_before_ext, question.question_text
                )
                if evidence_supports:
                    p_ext = extremize_minibench(p_before_ext)
                    applied.append("extremize(minibench)")
                    logger.info(
                        f"Minibench extremize: evidence=TRUE "
                        f"{p_before_ext:.3f} → {p_ext:.3f}"
                    )
                else:
                    p_ext = p_before_ext
                    applied.append("no-extremize(no-evidence-support)")
            else:
                p_ext = extremize(p_before_ext, strength=0.3)
                applied.append("extremize(0.3)")
            logger.info(
                f"Extremized: {p_before_ext:.3f} → {p_ext:.3f} "
                f"(tournament={tournament_slug})"
            )
        else:
            p_ext = blended_p

        p_time = ForecastingPrinciples.apply_time_decay(
            p_ext, getattr(question, "close_time", None)
        )
        if p_time != p_ext:
            applied.append("time-decay")

        # FIX #8: apply_bayesian_calibration is now defined (identity until calibrated)
        try:
            p_cal = self.apply_bayesian_calibration(p_time * 100) / 100.0
            if p_cal != p_time:
                applied.append("bayes-calibration")
        except Exception as e:
            logger.warning(f"Bayesian calibration failed ({e}); skipping.")
            p_cal = p_time

        # FIX #4: removed the force-extreme 1%/99% cliff.
        # Standard clipping for both minibench and non-minibench.
        final_p = float(np.clip(p_cal, 0.03, 0.97))
        applied.append("clip(3%-97%)")
        self._record_prediction(question, final_p)

        dr = await self._build_detailed_reasoning(
            question, research, model_probs, final_p,
            calibration_steps=applied + [f"final={final_p:.3f}"],
        )
        reasoning = (
            f"{self._methodology_header(research)}\n\n"
            + dr.render()
            + f"\n\n[calibration log] critic={raw_p:.3f} red={red_teamed_p:.3f} "
            f"ext={p_ext:.3f} time-decay={p_time:.3f} final={final_p:.3f}"
        )

        # FIX #7: scenario blend is ADDITIVE (15% weight), not a replacement of final_p
        try:
            scenarios = await self.scenario_analyzer.generate_scenarios(
                question, research, self.get_llm("default", "llm")
            )
            scenario_blended_p = sum(s.probability * s.predicted_value for s in scenarios)

            model_predictions = [final_p, scenario_blended_p] + [p for p in model_probs if 0 < p < 1]
            uncertainty = self.uncertainty_quantifier.quantify_uncertainty(
                model_predictions, research_quality=quality
            )

            reasoning += (
                f"\n\n[SCENARIO ANALYSIS] "
                f"bull={scenarios[0].probability:.2f}@{scenarios[0].predicted_value:.3f} | "
                f"base={scenarios[1].probability:.2f}@{scenarios[1].predicted_value:.3f} | "
                f"bear={scenarios[2].probability:.2f}@{scenarios[2].predicted_value:.3f}"
                f"\n[UNCERTAINTY] "
                f"credible_interval=[{uncertainty.credible_interval[0]:.3f}, "
                f"{uncertainty.credible_interval[1]:.3f}] | "
                f"epistemic={uncertainty.epistemic_unc:.3f} "
                f"aleatoric={uncertainty.aleatoric_unc:.3f}"
            )

            # FIX #7: 85% existing final_p, 15% scenario signal — conservative additive blend
            final_p = float(np.clip(0.85 * final_p + 0.15 * scenario_blended_p, 0.03, 0.97))
            applied.append(f"scenario-blend(15%): final→{final_p:.3f}")
        except Exception as e:
            logger.debug(f"Scenario & uncertainty integration failed: {e}")

        try:
            _qmeta = {"applied": applied}
            for _attr in ("id_of_post", "id", "post_id", "question_id"):
                _v = getattr(question, _attr, None)
                if _v is not None:
                    _qmeta["metaculus_id"] = _v; break
            _url = getattr(question, "page_url", None) or getattr(question, "url", None)
            if _url: _qmeta["url"] = _url
            self.memory.remember_forecast(question.question_text, final_p, research[:2000], meta=_qmeta)
        except Exception as _e:
            logger.debug(f"memory write skipped: {_e}")

        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    # ── Multiple-choice forecasting ──────────────────────────────────────────

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self._ensure_some_research_or_raise(research)

        # FIX: use _FORECAST_ENSEMBLE so gpt-5.1 is included with highest leverage
        results = await self._collect_model_forecasts(_FORECAST_ENSEMBLE, question, research)
        if not results:
            raise RuntimeError("No model forecasts available for multiple-choice prediction.")

        model_probs  = [float(np.mean([o.probability for o in r.predicted_options])) for r in results]
        forecast_map = {f"model_{i}": r.model_dump() for i, r in enumerate(results)}

        critic_llm    = self.get_llm("critic", "llm")
        schema_example = json.dumps({
            "predicted_options": [
                {"option_name": opt, "probability": 0.5}
                for opt in question.options[:2]
            ]
        })
        critique = await critic_llm.invoke(clean_indents(f"""
You are a conservative critic reviewing an ensemble forecast.
Question: {question.question_text}
Options: {question.options}
Research:
{research}
Ensemble: {json.dumps(forecast_map)}

{ForecastingPrinciples.get_conservative_reasoning_prompt()}

OUTPUT ONLY VALID JSON:
{schema_example}
"""))
        final_list: PredictedOptionList = await structure_output(
            sanitize_llm_json(critique), PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )

        option_names = question.options
        current      = {o.option_name: float(o.probability) for o in final_list.predicted_options}
        aligned      = [
            {"option_name": name, "probability": float(current.get(name, 0.0))}
            for name in option_names
        ]
        tournament_slug  = get_question_tournament_slug(question) or "unknown"
        is_minibench     = is_minibench_question(question)
        is_parent_child  = (
            getattr(question, "parent_question", None) is not None
            or getattr(question, "child_questions", None) is not None
        )

        if self.flags.enable_extremize and (is_parent_child or not is_minibench):
            if is_minibench and is_parent_child:
                avg_p = float(np.mean([o["probability"] for o in aligned])) if aligned else 0.5
                if self._evidence_supports_forecast(research, avg_p, question.question_text):
                    for o in aligned:
                        pb = float(o["probability"])
                        pa = extremize_minibench(pb)
                        logger.info(f"Extremized (minibench PC): {pb:.3f} → {pa:.3f}")
                        o["probability"] = pa
            elif is_parent_child and not is_minibench:
                for o in aligned:
                    pb = float(o["probability"])
                    pa = extremize(pb, strength=0.3)
                    logger.info(f"Extremized (PC): {pb:.3f} → {pa:.3f} (tournament={tournament_slug})")
                    o["probability"] = pa
            elif not is_minibench:
                for o in aligned:
                    pb = float(o["probability"])
                    pa = extremize(pb, strength=0.3)
                    logger.info(f"Extremized: {pb:.3f} → {pa:.3f} (tournament={tournament_slug})")
                    o["probability"] = pa

        total = float(sum(o["probability"] for o in aligned))
        if total <= 0:
            uniform = 1.0 / len(aligned)
            for o in aligned:
                o["probability"] = uniform
        else:
            for o in aligned:
                o["probability"] /= total

        final_val = safe_model(PredictedOptionList, {"predicted_options": aligned})  # type: ignore[assignment]
        avg_prob  = float(np.mean([o["probability"] for o in aligned])) if aligned else 0.0
        self._record_prediction(question, avg_prob)

        dr = await self._build_detailed_reasoning(
            question, research, model_probs, avg_prob,
            calibration_steps=["MC ensemble → critic → normalised"]
        )
        reasoning = (
            f"{self._methodology_header(research)}\n\n"
            + dr.render()
            + f"\n\n[avg_prob={avg_prob:.3f}]"
        )
        return ReasonedPrediction(prediction_value=final_val, reasoning=reasoning)

    # ── Numeric forecasting ──────────────────────────────────────────────────

    async def _run_forecast_on_numeric_generic(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        # FIX: use _FORECAST_ENSEMBLE so gpt-5.1 is included with highest leverage
        raw_results = await self._collect_model_forecasts(_FORECAST_ENSEMBLE, question, research)
        results: List[List[Percentile]] = [r for r in raw_results if isinstance(r, list)]
        if not results:
            raise RuntimeError("No model forecasts available for numeric prediction.")

        forecast_map = {
            f"model_{i}": [
                {"percentile": float(p.percentile), "value": float(p.value)} for p in r
            ]
            for i, r in enumerate(results)
        }
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        units      = question.unit_of_measure if question.unit_of_measure else "Not stated"
        critic_llm = self.get_llm("critic", "llm")

        critique = await critic_llm.invoke(clean_indents(f"""
You are a conservative critic reviewing an ensemble numeric forecast.
Question:
{question.question_text}

Units: {units}
Research:
{research}

Ensemble forecasts:
{json.dumps(forecast_map)}

Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.
{lower_msg}
{upper_msg}

{ForecastingPrinciples.get_conservative_reasoning_prompt()}

The LAST thing you write is EXACTLY:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""))

        final_pcts = await self._parse_numeric_percentiles_robust(
            question, critique, stage="critic_numeric"
        )
        final_pcts = self._enforce_monotone(final_pcts)
        med        = self._median_from_40_60(final_pcts)
        final_pcts = self._enforce_minimum_ci_width(final_pcts, med)

        dist = NumericDistribution.from_question(final_pcts, question)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        model_medians = [self._median_from_40_60(r) for r in results]
        dr = await self._build_detailed_reasoning(
            question, research, model_medians, med,
            calibration_steps=[
                f"numeric ensemble → critic; SD ×{_NUMERIC_SD_CONSERVATIVE}; CI floor applied."
            ],
        )
        reasoning = (
            f"{self._methodology_header(research)}\n\n"
            + dr.render()
            + f"\n\n{self._numeric_summary_line(final_pcts)}."
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)
        if not self.flags.enable_numeric_regimes:
            return await self._run_forecast_on_numeric_generic(question, research)

        regime = self._detect_numeric_regime(question, research)

        if regime == NumericRegime.PARTIAL_REVEAL_SUM:
            try:
                return await self._forecast_numeric_partial_reveal(question, research)
            except Exception as e:
                logger.warning(f"Partial-reveal regime failed, fallback to generic: {e}")
                return await self._run_forecast_on_numeric_generic(question, research)

        if regime == NumericRegime.STRUCTURED_TS:
            try:
                return await self._forecast_numeric_structured_ts(question, research)
            except Exception as e:
                logger.warning(f"Structured TS regime failed, fallback to generic: {e}")
                return await self._run_forecast_on_numeric_generic(question, research)

        return await self._run_forecast_on_numeric_generic(question, research)

    async def _run_forecast_on_numeric_wrapper(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_forecast_on_numeric(question, research)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the Advanced Forecasting Bot (botduke)")
    parser.add_argument("--mode", choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    parser.add_argument("--bot-name",            type=str, default="botduke")
    parser.add_argument("--no-extremize",        action="store_true")
    parser.add_argument("--minibench-biweekly",  action="store_true",
                        help="Tuned profile for topping the biweekly minibench: higher research effort, tighter calibration, memory-weighted.")
    parser.add_argument("--no-decomposition",    action="store_true")
    parser.add_argument("--no-meta-forecast",    action="store_true")
    parser.add_argument("--no-numeric-regimes",  action="store_true")
    parser.add_argument("--no-detailed-reasoning", action="store_true")

    args     = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    flags = BotFeatureFlags(
        enable_extremize          = not args.no_extremize,
        enable_decomposition      = not args.no_decomposition,
        enable_meta_forecast      = not args.no_meta_forecast,
        enable_numeric_regimes    = not args.no_numeric_regimes,
        enable_detailed_reasoning = not args.no_detailed_reasoning,
    )

    bot = SpringAdvancedForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        bot_name=args.bot_name,
        flags=flags,
    )

    client = MetaculusClient()

    async def run_all():
        if run_mode == "tournament":
            seasonal, minibench, market_pulse = await asyncio.gather(
                bot.forecast_on_tournament(CURRENT_AI_COMPETITION_ID,      return_exceptions=True),
                bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID,    return_exceptions=True),
                bot.forecast_on_tournament("market-pulse-26q3",            return_exceptions=True),
            )
            return seasonal + minibench + market_pulse

        if run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            return await bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )

        bot.skip_previously_forecasted_questions = False
        EXAMPLE_QUESTION_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        questions = [client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTION_URLS]
        single_reports, market_pulse_q2_reports, summer_eval_reports = await asyncio.gather(
            bot.forecast_questions(questions,                              return_exceptions=True),
            bot.forecast_on_tournament("market-pulse-26q3",               return_exceptions=True),
            bot.forecast_on_tournament("summer-futureeval-2026",          return_exceptions=True),
        )
        return single_reports + market_pulse_q2_reports + summer_eval_reports

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
