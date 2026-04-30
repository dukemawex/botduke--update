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

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

MINIBENCH_TOURNAMENT_ID = 33022
MINIBENCH_TOURNAMENT_SLUG = "minibench"

_TAVILY_CLIENT: Optional[AsyncTavilyClient] = None
_ASKNEWS_SEMAPHORE = asyncio.Semaphore(5)
_PERPLEXITY_SEMAPHORE = asyncio.Semaphore(5)
_GPT5_SEARCH_SEMAPHORE = asyncio.Semaphore(5)
_TAVILY_SEMAPHORE = asyncio.Semaphore(5)

# â”€â”€â”€ Conservative tuning constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_MAX_EXTREMIZE_STRENGTH    = 1.3   # was 2.0 â€” prevents overconfidence
_HIGH_SPREAD_SHRINK_THRESH = 0.25  # was 0.35 â€” shrink sooner when models disagree
_MED_SPREAD_SHRINK_THRESH  = 0.15  # was 0.20
_WEAK_RESEARCH_PRIOR_WT    = 0.40  # was 0.25 â€” stronger base-rate pull when evidence thin
_NUMERIC_SD_CONSERVATIVE   = 1.25  # multiply all numeric SDs by this factor
_RECENT_PREDICTIONS_CAP    = 20    # prevent unbounded list growth


# â”€â”€â”€ JSON sanitization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def sanitize_llm_json(text: str) -> str:
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val = match.group(2)
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
    odds = p / (1 - p)
    extremized_odds = odds ** (1 + strength)
    extremized_p = extremized_odds / (1 + extremized_odds)
    return float(np.clip(extremized_p, 0.01, 0.99))


def extremize_minibench(p: float) -> float:
    if 0.45 < p < 0.55:
        return extremize(p, strength=1.2)
    return extremize(p, strength=0.4)


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
        getattr(question, "tournaments", None),
        getattr(question, "tournament", None),
        getattr(question, "project", None),
        getattr(question, "tournament_slug", None),
        getattr(question, "project_slug", None),
        getattr(question, "tournament_id", None),
        getattr(question, "project_id", None),
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

    for attr in ("tournaments", "tournament", "project", "tournament_slug", "project_slug", "tournament_id", "project_id"):
        candidate = getattr(question, attr, None)
        if candidate is None:
            continue
        values = candidate if isinstance(candidate, (list, tuple, set)) else [candidate]
        for value in values:
            if isinstance(value, (int, float)) and int(value) == MINIBENCH_TOURNAMENT_ID:
                return True
            if isinstance(value, str) and value.strip() == str(MINIBENCH_TOURNAMENT_ID):
                return True
            text = _stringify_tournament_value(value)
            if text and (MINIBENCH_TOURNAMENT_SLUG in text.lower() or text == str(MINIBENCH_TOURNAMENT_ID)):
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


async def research_tavily(question: str) -> str:
    client = _get_tavily_client()
    if client is None:
        logger.warning("Tavily search skipped: TAVILY_API_KEY is not configured")
        return ""

    async with _TAVILY_SEMAPHORE:
        try:
            result = await client.search_context(question, search_depth="advanced", max_tokens=4000)
            if not result:
                logger.warning("Tavily search returned empty result")
                return ""
            if isinstance(result, str):
                return result.strip()
            return str(result).strip()
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return ""


async def _research_openrouter_search(question: str, model_name: str, provider_name: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            llm = GeneralLlm(model=model_name, temperature=0)
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
            result = (response or "").strip()
            if not result:
                logger.warning(f"{provider_name} search returned empty result")
                return ""
            return f"=== {provider_name} Search ===\n[{provider_name} Data]\n{result}"
        except Exception as e:
            logger.warning(f"{provider_name} search failed: {e}")
            return ""


async def research_perplexity(question: str) -> str:
    for model_name in ("openrouter/perplexity/sonar-pro", "openrouter/perplexity/sonar"):
        result = await _research_openrouter_search(question, model_name, "Perplexity", _PERPLEXITY_SEMAPHORE)
        if result:
            return result
    return ""


async def research_gpt5_search(question: str) -> str:
    return await _research_openrouter_search(question, "openrouter/openai/gpt-5.5", "GPT-5", _GPT5_SEARCH_SEMAPHORE)


# â”€â”€â”€ Pydantic models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RawPercentile(BaseModel):
    percentile: float
    value: float


class RedTeamOutput(BaseModel):
    """Typed model to replace bare dict in red-team parsing (bug fix)."""
    revised_prediction_in_decimal: float


class DecompositionOutput(BaseModel):
    subquestions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_metrics: List[str] = Field(default_factory=list)


class PartialRevealExtract(BaseModel):
    known_subtotal: Optional[float] = None
    known_parts: Optional[int] = Field(default=None, ge=0)
    total_parts: Optional[int] = Field(default=None, ge=1)
    notes: Optional[str] = None


class ReferenceClassExtract(BaseModel):
    reference_totals: List[float] = Field(default_factory=list)
    trend_multiplier: Optional[float] = None
    notes: Optional[str] = None


class LevelSeriesExtract(BaseModel):
    current_value: Optional[float] = None
    current_date: Optional[str] = None
    recent_values: List[float] = Field(default_factory=list)
    notes: Optional[str] = None


class BoundedMultiplier(BaseModel):
    multiplier: float


class BoundedDelta(BaseModel):
    delta: float


class DetailedReasoning(BaseModel):
    """
    Structured container for the full forecast reasoning chain.
    Serialised into the ReasonedPrediction.reasoning string so Metaculus
    commentary is human-readable.
    """
    question_restatement: str = ""
    base_rate_analysis: str = ""
    evidence_summary: List[str] = Field(default_factory=list)   # bullet facts
    supporting_factors: List[str] = Field(default_factory=list)
    contrary_factors: List[str] = Field(default_factory=list)
    key_uncertainties: List[str] = Field(default_factory=list)
    model_ensemble_summary: str = ""
    calibration_steps: List[str] = Field(default_factory=list)
    final_derivation: str = ""

    def render(self) -> str:
        """Render to a readable multi-paragraph string."""
        lines: List[str] = []
        lines.append(f"QUESTION RESTATEMENT\n{self.question_restatement}")
        lines.append(f"\nBASE RATE & REFERENCE CLASS\n{self.base_rate_analysis}")
        if self.evidence_summary:
            lines.append("\nEVIDENCE SUMMARY")
            for fact in self.evidence_summary:
                lines.append(f"  â€¢ {fact}")
        if self.supporting_factors:
            lines.append("\nFACTORS SUPPORTING RESOLUTION YES / HIGHER")
            for f in self.supporting_factors:
                lines.append(f"  + {f}")
        if self.contrary_factors:
            lines.append("\nFACTORS AGAINST / LOWER")
            for f in self.contrary_factors:
                lines.append(f"  - {f}")
        if self.key_uncertainties:
            lines.append("\nKEY UNCERTAINTIES")
            for u in self.key_uncertainties:
                lines.append(f"  ? {u}")
        if self.model_ensemble_summary:
            lines.append(f"\nMODEL ENSEMBLE\n{self.model_ensemble_summary}")
        if self.calibration_steps:
            lines.append("\nCALIBRATION STEPS")
            for step in self.calibration_steps:
                lines.append(f"  â†’ {step}")
        lines.append(f"\nFINAL DERIVATION\n{self.final_derivation}")
        return "\n".join(lines)


# â”€â”€â”€ Enums & flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class NumericRegime(str, Enum):
    LOOKUP            = "lookup"
    PARTIAL_REVEAL_SUM = "partial_reveal_sum"
    STRUCTURED_TS     = "structured_ts"
    GENERIC           = "generic"


@dataclass
class BotFeatureFlags:
    enable_extremize:       bool = True
    enable_decomposition:   bool = True
    enable_meta_forecast:   bool = True   # now wired to community_prediction
    enable_numeric_regimes: bool = True
    enable_detailed_reasoning: bool = True  # NEW: emit full reasoning chain


# â”€â”€â”€ Search helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ExaSearcher:
    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required for Exa search.")
        self.base_url = "https://api.exa.ai/search"

    async def search(self, query: str, num_results: int = 5) -> str:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "query": query,
            "numResults": num_results,
            "type": "neural",
            "useAutoprompt": True,
            "category": "news",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
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


# â”€â”€â”€ Forecasting principles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            "â€¢ Default toward base rates and prior probabilities.\n"
            "â€¢ gree by >0.20, treat the disagreement itself as uncertainty.\n"
            "â€¢ Prefer 60/40 over 80/20 when the decisive evidence is ambiguous.\n"
            "â€¢ Articulate at least 2 factors that could push the outcome the other way.\n"
            "â€¢ Never assign probability below 0.03 or above 0.97 without extraordinary evidence."
        )

    @staticmethod
    def apply_time_decay(prob: float, close_time: Optional[datetime]) -> float:
        if close_time is None:
            return prob
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days = max(0.0, (close_time - now).total_seconds() / 86400.0)
        # Conservative: blend more aggressively toward 0.5 for uncertain futures
        if days > 365:
            return 0.20 * prob + 0.80 * 0.5   # was 0.30/0.70
        if days > 180:
            return 0.40 * prob + 0.60 * 0.5   # was 0.50/0.50
        if days > 90:
            return 0.65 * prob + 0.35 * 0.5   # was 0.70/0.30
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
        # Conservative: cap strength at _MAX_EXTREMIZE_STRENGTH
        strength = float(np.clip(strength, 0.5, _MAX_EXTREMIZE_STRENGTH))
        return float(np.clip(cls.sigmoid(strength * cls.logit(p)), 0.0, 1.0))


# â”€â”€â”€ Main bot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        self.flags = flags or BotFeatureFlags()
        self.tavily_api_key     = os.getenv("TAVILY_API_KEY")
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None
        self.asknews_client_id     = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")
        # Capped list: (question, final_p)
        self._recent_predictions: list[tuple[MetaculusQuestion, float]] = []

    # â”€â”€ Model configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _llm_config_defaults(self) -> Dict[str, str]:
        """
        All model strings verified against OpenRouter as of March 2026.

        Role            | Model                              | Why
        ----------------|------------------------------------|---------------------------
        default         | openrouter/openai/gpt-5.2          | Strongest general reasoning
        parser          | openrouter/openai/gpt-5-mini       | Fast, cheap, structured output
        summarizer      | openrouter/openai/gpt-5-mini       | Fast summarisation
        researcher      | openrouter/openai/gpt-5.2          | Long-context + web search
        query_optimizer | openrouter/openai/gpt-5-mini       | Simple rewrite task
        critic          | openrouter/anthropic/claude-opus-4.6 | Best adversarial reasoning
        red_team        | openrouter/anthropic/claude-sonnet-4.6 | Strong, diverse from critic
        decomposer      | openrouter/openai/gpt-5-mini       | Simple decomposition
        """
        return {
            "default":         "openrouter/openai/gpt-5.4",
            "parser":          "openrouter/openai/gpt-5-mini",
            "summarizer":      "openrouter/openai/gpt-5-mini",
            "researcher":      "openrouter/openai/gpt-5.4",
            "query_optimizer": "openrouter/openai/gpt-5-mini",
            "critic":          "openrouter/anthropic/claude-opus-4.6",
            "red_team":        "openrouter/anthropic/claude-sonnet-4.6",
            "decomposer":      "openrouter/openai/gpt-5-mini",
        }

    # â”€â”€ Research quality helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _search_footprint(self, research: str) -> str:
        used: list[str] = []
        def ok(tag: str, fail_markers: list[str]) -> bool:
            return (tag in research) and (not any(m in research for m in fail_markers))
        if ok("[Tavily Data]",        ["[Tavily not configured]", "[Tavily search failed]"]):    used.append("tavily")
        if ok("[Exa Search Results]", ["[Exa not configured]",    "[Exa search failed]"]):       used.append("exa")
        if ok("[AskNews Data]",       ["[AskNews not configured]","[AskNews search failed]"]):   used.append("asknews")
        if ok("[Perplexity Data]",    ["[Perplexity search failed]"]):                            used.append("perplexity")
        if ok("[GPT-5 Data]",         ["[GPT-5 search failed]"]):                                 used.append("gpt5_search")
        if ok("[LLM Web Research]",   ["[LLM web research failed]"]):                            used.append("llm_web")
        if ok("[Meta-Forecast]",      ["[Meta-forecast unavailable]"]):                          used.append("meta")
        return ",".join(used) if used else "none"

    def _ensure_some_research_or_raise(self, research: str) -> None:
        if self._search_footprint(research) == "none":
            raise RuntimeError("No research evidence available (all providers and LLM fallback failed).")

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none":
            return 0.25
        n = len(srcs.split(","))
        return {1: 0.50, 2: 0.65, 3: 0.78, 4: 0.85, 5: 0.90}.get(n, 0.55)
        # Note: weights are slightly lower than v1 â†’ more conservative blending

    def _ensure_research_has_facts(self, research: str, min_signals: int = 3) -> bool:
        """
        Verify the research block contains at least min_signals factual signals
        (years, numbers, capitalised entities). Returns False if thin.
        """
        year_hits   = len(re.findall(r"\b(19|20)\d{2}\b", research))
        number_hits = len(re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|USD|EUR)?\b", research))
        entity_hits = len(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", research))
        total = year_hits + number_hits + entity_hits
        if total < min_signals:
            logger.warning(f"Thin research: only {total} factual signals detected.")
            return False
        return True

    # â”€â”€ Reasoning builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _build_detailed_reasoning(
        self,
        question: MetaculusQuestion,
        research: str,
        model_probs: List[float],
        final_p: float,
        calibration_steps: List[str],
    ) -> DetailedReasoning:
        """
        Ask the critic LLM to produce a structured DetailedReasoning JSON block
        covering all required elements for conservative, transparent forecasting.
        Falls back to a lightweight manual assembly if the LLM call fails.
        """
        if not self.flags.enable_detailed_reasoning:
            return DetailedReasoning(
                final_derivation=f"Final probability: {final_p:.3f}"
            )

        critic = self.get_llm("critic", "llm")
        ensemble_desc = ", ".join(f"{p:.3f}" for p in model_probs)

        prompt = clean_indents(f"""
You are writing a detailed forecasting report for a superforecaster platform.
Produce a JSON object with EXACTLY these keys:

{{
  "question_restatement": "One sentence restating what must happen for YES / higher resolution.",
  "base_rate_analysis": "2-4 sentences on base rate: historical frequency, reference class, outside-view prior.",
  "evidence_summary": ["fact 1 with source", "fact 2 with source", ...],   // 5-10 bullet facts
  "supporting_factors": ["factor 1", ...],   // 3-5 factors pushing toward YES / higher
  "contrary_factors":   ["factor 1", ...],   // 3-5 factors pushing toward NO / lower
  "key_uncertainties":  ["uncertainty 1", ...],  // 2-4 things that could most move this
  "model_ensemble_summary": "Brief description of model outputs: {ensemble_desc} â†’ final {final_p:.3f}",
  "calibration_steps": ["step 1", ...],   // enumerate each adjustment: base rate, research quality, spread, time decay, etc.
  "final_derivation": "2-3 sentence narrative explaining the final number."
}}

Rules:
- Be specific: cite numbers, dates, and named sources where available.
- Be conservative: acknowledge uncertainty explicitly.
- Do NOT invent facts not present in the research.
- Output ONLY the JSON object, no preamble.

Question: {question.question_text}

Resolution criteria: {question.resolution_criteria}

Research:
{research[:6000]}
""")

        try:
            raw = await critic.invoke(prompt)
            data = json.loads(sanitize_llm_json(raw))
            return DetailedReasoning(**data)
        except Exception as e:
            logger.warning(f"Detailed reasoning LLM call failed ({e}); using fallback assembly.")
            return DetailedReasoning(
                question_restatement=question.question_text[:200],
                base_rate_analysis="Base rate not explicitly modelled; default conservative prior applied.",
                evidence_summary=[line.strip() for line in research.split("\n") if line.strip().startswith("â€¢") or line.strip().startswith("-")][:8],
                model_ensemble_summary=f"Ensemble: [{ensemble_desc}] â†’ final {final_p:.3f}",
                calibration_steps=calibration_steps,
                final_derivation=f"Final probability: {final_p:.3f} after {len(calibration_steps)} calibration steps.",
            )

    # â”€â”€ Decomposition & query optimisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _decompose_question(self, question: MetaculusQuestion) -> Optional[DecompositionOutput]:
        if not self.flags.enable_decomposition:
            return None
        try:
            llm = self.get_llm("decomposer", "llm")
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
        llm = self.get_llm("query_optimizer", "llm")
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

    # â”€â”€ Individual search providers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _run_tavily_search(self, query: str) -> str:
        if not self.tavily_api_key:
            logger.warning("Tavily search skipped: TAVILY_API_KEY is not configured")
            return ""

        result = await research_tavily(query)
        if not result:
            logger.warning("Tavily search returned no usable output")
            return ""
        return f"=== Tavily Search ===\n[Tavily Data]\n{result}"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def _run_asknews_search(self, query: str) -> str:
        async with _ASKNEWS_SEMAPHORE:
            if not self.asknews_client_id or not self.asknews_client_secret:
                logger.warning("AskNews search skipped: ASKNEWS_CLIENT_ID / ASKNEWS_CLIENT_SECRET is not configured")
                return ""
            try:
                searcher = AskNewsSearcher(
                    client_id=self.asknews_client_id,
                    client_secret=self.asknews_client_secret,
                )
                result = await searcher.get_formatted_news_async(query)
                if not str(result).strip():
                    logger.warning("AskNews search returned empty result")
                    return ""
                return f"=== AskNews Search ===\n[AskNews Data]\n{result}"
            except Exception as e:
                logger.warning(f"AskNews search failed: {e}")
                return ""

    async def _run_llm_web_research(
        self, question: MetaculusQuestion, decomp: Optional[DecompositionOutput]
    ) -> str:
        try:
            researcher = self.get_llm("researcher")
            extra = ""
            if decomp:
                if decomp.subquestions:
                    extra += "\nSubquestions:\n" + "\n".join(f"- {s}" for s in decomp.subquestions[:6])
                if decomp.key_entities:
                    extra += "\nEntities:\n" + ", ".join(decomp.key_entities[:12])
                if decomp.key_metrics:
                    extra += "\nMetrics:\n" + ", ".join(decomp.key_metrics[:12])

            prompt = clean_indents(f"""
You are an assistant to a superforecaster.
Use live web browsing/search (if available) to gather the most relevant, recent evidence.
Provide:
- 6-12 bullet facts with sources/links when possible
- any key numbers / time series / market expectations
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
        """
        Retrieve the Metaculus community prediction if available and flags enabled.
        Previously this was always a stub returning unavailable. Now it checks for
        an existing community_prediction attribute first.
        """
        if not self.flags.enable_meta_forecast:
            return ""
        community = getattr(question, "community_prediction", None)
        if community is not None:
            return f"[Meta-Forecast]\nMetaculus community prediction: {float(community):.3f}"
        return ""

    # â”€â”€ Research orchestration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def run_research(self, question: MetaculusQuestion) -> str:
        decomp  = await self._decompose_question(question)
        queries = await self._optimize_search_query(question, decomp)
        optimized_query = " OR ".join(queries)

        tasks   = [
            self._run_tavily_search(optimized_query),
            self._run_exa_search(optimized_query),
            self._run_asknews_search(optimized_query),
            research_perplexity(optimized_query),
            research_gpt5_search(optimized_query),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cleaned: list[str] = []
        for res in results:
            if isinstance(res, Exception):
                cleaned.append(f"[Search failed: {str(res)}]")
            else:
                cleaned.append(res)
        combined = "\n\n".join(cleaned).strip()

        if self._search_footprint(combined) == "none":
            combined = (combined + "\n\n" if combined else "") + await self._run_llm_web_research(question, decomp)

        meta_block = await self._run_meta_forecast_lookup(question)
        if meta_block:
            combined = combined + "\n\n" + meta_block

        research = (
            f"{ForecastingPrinciples.get_generic_base_rate()}\n\n"
            f"{ForecastingPrinciples.get_generic_fermi_prompt()}\n\n"
            f"{ForecastingPrinciples.get_conservative_reasoning_prompt()}\n\n"
            f"{combined}"
        )

        self._ensure_some_research_or_raise(research)
        thin = not self._ensure_research_has_facts(research)
        if thin:
            research += "\n\n[WARNING: Research appears thin. Applying stronger base-rate prior.]"
        return research

    # â”€â”€ Numeric helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> Tuple[str, str]:
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
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

Output MUST be a list of objects with fields:
  - percentile
  - value

Percentile can be:
  - 10,20,40,60,80,90
  OR
  - 0.1,0.2,0.4,0.6,0.8,0.9

Values:
  - MUST be in units: {question.unit_of_measure}
  - Never use scientific notation.

Rules:
  - Required percentiles are exactly those six.
  - Values must be strictly increasing with percentile.
""")

    @staticmethod
    def _extract_percentile_block(text: str) -> str:
        m = re.search(
            r"(Percentile\s*10\s*:.*?Percentile\s*90\s*:.*?)(?:\n\s*\n|$)",
            text, flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        lines = []
        for line in text.splitlines():
            if re.search(r"^\s*Percentile\s*(10|20|40|60|80|90)\s*:", line, flags=re.IGNORECASE):
                lines.append(line.strip())
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_raw_percentiles(raw: List[RawPercentile]) -> List[Percentile]:
        out: List[Percentile] = []
        for rp in raw:
            p = float(rp.percentile)
            if p > 1.0:
                p = p / 100.0
            p = max(0.0, min(1.0, p))
            out.append(Percentile(percentile=p, value=float(rp.value)))
        return out

    @staticmethod
    def _require_standard_percentiles(pcts: List[Percentile]) -> List[Percentile]:
        required = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        by      = {round(float(p.percentile), 3): p for p in pcts}
        missing = [r for r in required if round(r, 3) not in by]
        if missing:
            return []
        return [by[round(r, 3)] for r in required]

    @staticmethod
    def _enforce_monotone(pcts: List[Percentile]) -> List[Percentile]:
        """Build new Percentile objects rather than mutating (bug fix)."""
        pcts = sorted(pcts, key=lambda x: float(x.percentile))
        result: List[Percentile] = [pcts[0]]
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
        # Conservative: wider fallback weights than v1
        w = {0.1: 0.03, 0.2: 0.12, 0.4: 0.38, 0.6: 0.62, 0.8: 0.88, 0.9: 0.97}
        pcts = [Percentile(percentile=p, value=lo + (hi - lo) * w[p]) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]]
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
        """
        Ensure p10-p90 width is at least 2% of |midpoint|. Prevents collapsed distributions.
        """
        floor_width = max(0.02 * abs(midpoint), 1e-6)
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
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
                    frac = (key - 0.1) / 0.8
                    by[key] = (midpoint - half) + frac * floor_width
            pcts = [Percentile(percentile=p.percentile, value=by[round(float(p.percentile), 3)]) for p in pcts]
            pcts = SpringAdvancedForecastingBot._enforce_monotone(pcts)
        return pcts

    async def _parse_numeric_percentiles_robust(
        self, question: NumericQuestion, text: str, stage: str
    ) -> List[Percentile]:
        parser_llm = self.get_llm("parser", "llm")
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
                p = self._normalize_raw_percentiles(raw)
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

Rules:
- Values must be in units: {question.unit_of_measure}
- Never use scientific notation.
- Values must be strictly increasing.

Text:
{text}
""")
            reformatted = await parser_llm.invoke(reform_prompt)
            rb = self._extract_percentile_block(reformatted) or reformatted
            raw3: List[RawPercentile] = await structure_output(
                rb, list[RawPercentile], model=parser_llm,
                additional_instructions=instructions, num_validation_samples=n,
            )
            p3 = self._normalize_raw_percentiles(raw3)
            std3 = self._require_standard_percentiles(p3)
            if std3:
                return self._enforce_monotone(std3)
        except Exception as e:
            logger.warning(f"[{stage}] numeric parse attempt 3 failed: {e}")

        logger.warning(f"[{stage}] numeric parsing failed; using conservative bounds fallback.")
        return self._bounds_fallback(question)

    # â”€â”€ Temperature & ensemble helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        close_time = getattr(question, "close_time", None)
        if not close_time:
            return 0.30
        # Bug fix: always use tz-aware comparison
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

    def _extremize_strength(self, research: str, probs: List[float], question: MetaculusQuestion) -> float:
        if not self.flags.enable_extremize:
            return 1.0
        quality = self._research_quality_weight(research)
        agree   = self._agreement_strength(probs)
        # Conservative: tighter base, capped lower
        base = 1.0 + 0.6 * (quality - 0.5) * 2.0 * agree   # was 0.9 multiplier
        close_time = getattr(question, "close_time", None)
        if close_time:
            now  = datetime.now(timezone.utc)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            days = (close_time - now).days
            if days < 14:
                base = 1.0 + (base - 1.0) * 0.25   # was 0.30
            elif days < 60:
                base = 1.0 + (base - 1.0) * 0.50   # was 0.60
        return float(np.clip(base, 0.9, _MAX_EXTREMIZE_STRENGTH))

    # â”€â”€ Red-teaming & consistency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _red_team_forecast(
        self, question: MetaculusQuestion, research: str, initial_pred: float
    ) -> float:
        self._ensure_some_research_or_raise(research)
        try:
            llm      = self.get_llm("red_team", "llm")
            response = await llm.invoke(clean_indents(f"""
You are a skeptical red teamer challenging an initial forecast.
Your role: identify reasons the initial forecast might be WRONG.
Think carefully about base rates, overlooked contrary evidence, and overconfidence.

Question: {question.question_text}
Research:
{research[:4000]}

Current forecast: {initial_pred:.2%}

Output ONLY JSON:
{{"revised_prediction_in_decimal": 0.XX}}
"""))
            # Bug fix: typed RedTeamOutput instead of bare dict
            parsed = safe_model(RedTeamOutput, sanitize_llm_json(response))
            return float(np.clip(parsed.revised_prediction_in_decimal, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
        return initial_pred

    async def _check_consistency(self, question: MetaculusQuestion, proposed_pred: float) -> bool:
        if len(self._recent_predictions) < 2:
            return True
        recent_summary = "\n".join(
            [f"Q: {getattr(q, 'question_text', '')} â†’ Pred: {p:.2%}" for q, p in self._recent_predictions[-3:]]
        )
        llm    = self.get_llm("parser", "llm")
        prompt = f"""
Is this new forecast logically consistent with prior forecasts?
New: {question.question_text} â†’ {proposed_pred:.2%}
Prior:
{recent_summary}
Answer YES or NO only.
""".strip()
        try:
            response = await llm.invoke(prompt)
            return "YES" in response.upper()
        except Exception as e:
            # Bug fix: log the failure instead of silently returning True
            logger.warning(f"Consistency check failed ({e}); defaulting to consistent.")
            return True

    def _record_prediction(self, question: MetaculusQuestion, p: float) -> None:
        """Append and cap the recent predictions list."""
        self._recent_predictions.append((question, p))
        if len(self._recent_predictions) > _RECENT_PREDICTIONS_CAP:
            self._recent_predictions = self._recent_predictions[-_RECENT_PREDICTIONS_CAP:]

    # â”€â”€ Methodology header & summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _methodology_header(self, research: str) -> str:
        src = self._search_footprint(research)
        return (
            f"[{self.bot_name}] sources({src}); "
            f"conservative ensemble (GPT-5.2 + Claude Opus 4.6 + Claude Sonnet 4.6); "
            f"extremize cap={_MAX_EXTREMIZE_STRENGTH}; spread-shrink thresholds="
            f"{_HIGH_SPREAD_SHRINK_THRESH}/{_MED_SPREAD_SHRINK_THRESH}."
        )

    def _numeric_summary_line(self, pcts: List[Percentile]) -> str:
        med = self._median_from_40_60(pcts)
        p10, p90 = self._p10_p90(pcts)
        if p10 is not None and p90 is not None:
            return f"medianâ‰ˆ{med:.6g}, 80% CI=[{p10:.6g},{p90:.6g}]"
        return f"medianâ‰ˆ{med:.6g}"

    # â”€â”€ Normal distribution helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _normal_percentiles_from_mean_sd(mean: float, sd: float) -> List[Percentile]:
        """Conservative: SD is scaled by _NUMERIC_SD_CONSERVATIVE before use."""
        sd_wide = sd * _NUMERIC_SD_CONSERVATIVE
        z = {0.1: -1.2816, 0.2: -0.8416, 0.4: -0.2533, 0.6: 0.2533, 0.8: 0.8416, 0.9: 1.2816}
        out: List[Percentile] = []
        for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
            val = mean + z[p] * sd_wide
            out.append(Percentile(percentile=p, value=float(val)))
        return SpringAdvancedForecastingBot._enforce_monotone(out)

    # â”€â”€ Numeric regime detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _extract_date_range_generic(self, text: str) -> Optional[Tuple[date, date]]:
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

    def _has_partial_observations(self, research: str, question: NumericQuestion) -> bool:
        r    = (research or "").lower()
        cues = ["sum to","subtotal","observed","published","known days","so far","remaining","hinges on","partial","to date"]
        return any(c in r for c in cues) and self._extract_date_range_generic(question.question_text or "") is not None

    def _regex_extract_known_subtotal(self, research: str) -> Optional[float]:
        pats = [
            r"sum to\s+([\d,]+(?:\.\d+)?)",
            r"subtotal[:\s]+([\d,]+(?:\.\d+)?)",
            r"known (?:subtotal|total)[:\s]+([\d,]+(?:\.\d+)?)",
            r"published (?:days|values) .*?sum(?:s)? to\s+([\d,]+(?:\.\d+)?)",
        ]
        for pat in pats:
            m = re.search(pat, research or "", flags=re.IGNORECASE)
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
        return any(k in qt for k in ["ending value","end value","closing value","close value","as of"])

    def _horizon_days_from_text(self, question: NumericQuestion) -> Optional[int]:
        dr = self._extract_date_range_generic(question.question_text or "")
        if not dr:
            return None
        start, end = dr
        return (end - start).days + 1

    def _detect_numeric_regime(self, question: NumericQuestion, research: str) -> NumericRegime:
        if not self.flags.enable_numeric_regimes:
            return NumericRegime.GENERIC
        qt = (question.question_text or "").lower()
        dr = self._extract_date_range_generic(question.question_text or "")
        if "according to" in qt and any(w in qt for w in ["was","were","did","have"]):
            if dr:
                _, end = dr
                if end < datetime.now(timezone.utc).date():
                    return NumericRegime.LOOKUP
        if self._has_partial_observations(research, question):
            return NumericRegime.PARTIAL_REVEAL_SUM
        if dr:
            start, end = dr
            horizon = (end - start).days + 1
            if 2 <= horizon <= 31:
                return NumericRegime.STRUCTURED_TS
        if self._is_level_series_question(question):
            return NumericRegime.STRUCTURED_TS
        return NumericRegime.GENERIC

    # â”€â”€ Bounded adjustment helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _delta_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days if horizon_days is not None else 30
        if h <= 21:  return (-0.60, 0.60)
        if h <= 60:  return (-1.00, 1.00)
        return (-2.00, 2.00)

    def _mult_bounds_for_horizon(self, horizon_days: Optional[int]) -> Tuple[float, float]:
        h = horizon_days if horizon_days is not None else 30
        if h <= 21:  return (0.97, 1.03)
        if h <= 60:  return (0.95, 1.05)
        return (0.90, 1.10)

    async def _bounded_multiplier(
        self, question: NumericQuestion, research: str, baseline: float, *, lo: float, hi: float
    ) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
Return JSON only: {{"multiplier": 1.00}}
Question: {question.question_text}
Baseline: {baseline}
Research:
{research[:3000]}
Rules:
- multiplier must be within [{lo:.6f}, {hi:.6f}]
- Output only JSON.
""")
        raw   = await critic.invoke(prompt)
        model = safe_model(BoundedMultiplier, sanitize_llm_json(raw))  # type: ignore[arg-type]
        return float(np.clip(float(getattr(model, "multiplier")), lo, hi))

    async def _bounded_delta(
        self, question: NumericQuestion, research: str, baseline_level: float, *, lo: float, hi: float
    ) -> float:
        critic = self.get_llm("critic", "llm")
        prompt = clean_indents(f"""
Return JSON only: {{"delta": 0.00}}
Question: {question.question_text}
Baseline level: {baseline_level}
Research:
{research[:3000]}
Rules:
- delta must be within [{lo:.6f}, {hi:.6f}]
- Output only JSON.
""")
        raw   = await critic.invoke(prompt)
        model = safe_model(BoundedDelta, sanitize_llm_json(raw))  # type: ignore[arg-type]
        return float(np.clip(float(getattr(model, "delta")), lo, hi))

    # â”€â”€ LLM extraction helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _llm_extract_partial_reveal(self, question: NumericQuestion, research: str) -> PartialRevealExtract:
        parser = self.get_llm("parser", "llm")
        prompt = clean_indents(f"""
Return JSON only:
{{"known_subtotal": null, "known_parts": null, "total_parts": null, "notes": null}}
Question: {question.question_text}
Research: {research[:3000]}
Extract: known_subtotal if research states a subtotal/sum; known_parts and total_parts if inferable.
""")
        raw = await parser.invoke(prompt)
        return safe_model(PartialRevealExtract, sanitize_llm_json(raw))  # type: ignore[return-value]

    async def _llm_extract_reference_class(self, question: NumericQuestion, research: str) -> ReferenceClassExtract:
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

    async def _llm_extract_level_series(self, question: NumericQuestion, research: str) -> LevelSeriesExtract:
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

    # â”€â”€ Numeric regime forecasters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

        if known_parts and total_parts and total_parts > known_parts and known_parts > 0:
            per_part          = known_subtotal / known_parts
            remainder_baseline = per_part * (total_parts - known_parts)
        else:
            remainder_baseline = 0.85 * known_subtotal

        lo_m, hi_m = self._mult_bounds_for_horizon(self._horizon_days_from_text(question))
        mult           = await self._bounded_multiplier(question, research, remainder_baseline, lo=lo_m, hi=hi_m)
        remainder_mean = remainder_baseline * mult
        total_mean     = known_subtotal + remainder_mean

        remainder_sd = max(0.08 * remainder_mean, 0.02 * total_mean)
        pcts = self._normal_percentiles_from_mean_sd(total_mean, remainder_sd)
        pcts = [Percentile(percentile=p.percentile, value=max(p.value, known_subtotal)) for p in pcts]
        pcts = self._enforce_monotone(pcts)
        pcts = self._enforce_minimum_ci_width(pcts, total_mean)

        dist = NumericDistribution.from_question(pcts, question)
        med  = self._median_from_40_60(pcts)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        reasoning = (
            f"{self._methodology_header(research)}\n"
            f"Regime=partial_reveal_sum: known subtotalâ‰ˆ{known_subtotal:.6g}; "
            f"remainder baselineâ‰ˆ{remainder_baseline:.6g} Ã— multiplier {mult:.4f}; "
            f"conservative SD widened by Ã—{_NUMERIC_SD_CONSERVATIVE}; "
            f"enforced totalâ‰¥known; CI floor applied. {self._numeric_summary_line(pcts)}."
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

        horizon  = self._horizon_days_from_text(question)
        lo_d, hi_d = self._delta_bounds_for_horizon(horizon)
        delta    = await self._bounded_delta(question, research, level, lo=lo_d, hi=hi_d)
        mean     = level + delta

        sd = None
        if ex and ex.recent_values and len(ex.recent_values) >= 5:
            vals = [float(v) for v in ex.recent_values if isinstance(v, (int, float)) and np.isfinite(v)]
            if len(vals) >= 5:
                changes  = np.diff(vals)
                daily_sd = float(np.std(changes)) if len(changes) > 1 else 0.0
                h        = float(horizon if horizon is not None else 10)
                sd       = float(np.sqrt(max(2.0, h)) * max(daily_sd, 0.02))
        if sd is None:
            h  = float(horizon if horizon is not None else 10)
            sd = float(np.clip(0.12 * np.sqrt(max(2.0, h)), 0.08, 0.90))

        pcts = self._normal_percentiles_from_mean_sd(mean, sd)

        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pcts = [Percentile(percentile=p.percentile, value=float(np.clip(p.value, lo, hi))) for p in pcts]
            pcts = self._enforce_monotone(pcts)
        pcts = self._enforce_minimum_ci_width(pcts, mean)

        dist = NumericDistribution.from_question(pcts, question)
        med  = self._median_from_40_60(pcts)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        reasoning = (
            f"{self._methodology_header(research)}\n"
            f"Regime=level_series_endvalue: baselineâ‰ˆ{level:.6g}; "
            f"bounded delta={delta:.4g} in [{lo_d:.3g},{hi_d:.3g}] horizonâ‰ˆ{horizon or 'n/a'}d; "
            f"conservative SD Ã—{_NUMERIC_SD_CONSERVATIVE}; CI floor applied. {self._numeric_summary_line(pcts)}."
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _forecast_numeric_structured_ts(
        self, question: NumericQuestion, research: str, *, force_non_level: bool = False
    ) -> ReasonedPrediction[NumericDistribution]:
        if (not force_non_level) and self._is_level_series_question(question):
            return await self._forecast_numeric_level_series_endvalue(question, research)

        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        baseline = 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 1.0

        try:
            ref  = await self._llm_extract_reference_class(question, research)
            refs = [float(x) for x in (ref.reference_totals or []) if isinstance(x, (int, float)) and x > 0 and np.isfinite(x)]
            if refs:
                baseline = float(np.median(refs))
                if ref.trend_multiplier and np.isfinite(float(ref.trend_multiplier)):
                    tm = float(ref.trend_multiplier)
                    if 0.8 <= tm <= 1.2:
                        baseline *= tm
        except Exception as e:
            logger.warning(f"Structured TS extraction failed: {e}")

        horizon  = self._horizon_days_from_text(question)
        lo_m, hi_m = self._mult_bounds_for_horizon(horizon)
        mult     = await self._bounded_multiplier(question, research, baseline, lo=lo_m, hi=hi_m)
        mean     = baseline * mult

        width = (hi - lo) if np.isfinite(hi - lo) and (hi - lo) > 0 else None
        sd    = max(0.06 * abs(mean), 0.02 * width) if width is not None else 0.06 * abs(mean)
        sd    = float(np.clip(sd, 1e-9, max(1e-9, 0.25 * abs(mean) + 1e-9)))

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
            f"Regime=structured_ts: baselineâ‰ˆ{baseline:.6g}; multiplierÃ—{mult:.4f} in [{lo_m:.3f},{hi_m:.3f}] "
            f"horizonâ‰ˆ{horizon or 'n/a'}d; conservative SD Ã—{_NUMERIC_SD_CONSERVATIVE}; "
            f"CI floor applied. {self._numeric_summary_line(pcts)}."
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    # â”€â”€ Per-model forecast helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            schema_example = json.dumps({"predicted_options": [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]})
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
            units    = question.unit_of_measure if question.unit_of_measure else "Not stated"
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
            return await self._parse_numeric_percentiles_robust(question, reasoning, stage=f"model_forecast:{model_name}")

        raise TypeError(f"Unsupported question type: {type(question)}")

    # â”€â”€ Binary forecasting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)

        # Diverse ensemble: GPT-5.5, Claude Opus 4.7, Claude Sonnet 4.7
        forecasters = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/anthropic/claude-sonnet-4.7",
        ]
        results     = await asyncio.gather(*[self._get_model_forecast(m, question, research) for m in forecasters])
        model_probs = [float(r.prediction_in_decimal) for r in results]
        forecast_map = {f"model_{i}": p for i, p in enumerate(model_probs)}
        spread       = (max(model_probs) - min(model_probs)) if len(model_probs) > 1 else 0.0

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
        averaged_p   = 0.5 * (raw_p + red_teamed_p)

        applied: List[str] = []

        # Conservative spread-shrink (lower thresholds)
        if spread >= _HIGH_SPREAD_SHRINK_THRESH:
            averaged_p = 0.55 * averaged_p + 0.45 * 0.5   # stronger pull than v1
            applied.append(f"high-spread-shrink(spread={spread:.2f})")
        elif spread >= _MED_SPREAD_SHRINK_THRESH:
            averaged_p = 0.75 * averaged_p + 0.25 * 0.5
            applied.append(f"med-spread-shrink(spread={spread:.2f})")

        if not await self._check_consistency(question, averaged_p):
            averaged_p = 0.5 * averaged_p + 0.5 * 0.5
            applied.append("consistency-shrink")

        community = getattr(question, "community_prediction", None)
        quality   = self._research_quality_weight(research)

        # Conservative: if research is weak, pull more toward base rate (0.5)
        if quality < 0.6:
            averaged_p = (1 - _WEAK_RESEARCH_PRIOR_WT) * averaged_p + _WEAK_RESEARCH_PRIOR_WT * 0.5
            applied.append(f"weak-research-prior(q={quality:.2f})")

        blended_p = (quality * averaged_p + (1 - quality) * float(community)) if (community is not None) else averaged_p
        if community is not None:
            applied.append(f"community-blend(c={float(community):.3f})")

        tournament_slug = get_question_tournament_slug(question) or "unknown"
        if self.flags.enable_extremize:
            p_before_ext = blended_p
            if is_minibench_question(question):
                p_ext = extremize_minibench(p_before_ext)
            else:
                p_ext = extremize(p_before_ext, strength=0.3)
            logger.info(f"Extremized: {p_before_ext:.3f} → {p_ext:.3f} (tournament={tournament_slug})")
            applied.append("extremize(minibench)" if is_minibench_question(question) else "extremize(0.3)")
        else:
            p_ext = blended_p

        p_time = ForecastingPrinciples.apply_time_decay(p_ext, getattr(question, "close_time", None))
        if p_time != p_ext:
            applied.append("time-decay")

        try:
            p_cal = self.apply_bayesian_calibration(p_time * 100) / 100.0
            if p_cal != p_time:
                applied.append("bayes-calibration")
        except Exception:
            p_cal = p_time

        # Conservative floor/ceiling: never go below 0.03 or above 0.97
        final_p = float(np.clip(p_cal, 0.03, 0.97))
        self._record_prediction(question, final_p)

        # Build detailed reasoning
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
        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    # â”€â”€ Multiple-choice forecasting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self._ensure_some_research_or_raise(research)

        forecasters = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/anthropic/claude-sonnet-4.7",
        ]
        results      = await asyncio.gather(*[self._get_model_forecast(m, question, research) for m in forecasters])
        model_probs  = [float(np.mean([o.probability for o in r.predicted_options])) for r in results]
        forecast_map = {f"model_{i}": r.model_dump() for i, r in enumerate(results)}

        critic_llm    = self.get_llm("critic", "llm")
        schema_example = json.dumps({"predicted_options": [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]})
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
        aligned      = [{"option_name": name, "probability": float(current.get(name, 0.0))} for name in option_names]
        tournament_slug = get_question_tournament_slug(question) or "unknown"
        if self.flags.enable_extremize:
            use_minibench = is_minibench_question(question)
            for o in aligned:
                p_before_ext = float(o["probability"])
                p_after_ext = extremize_minibench(p_before_ext) if use_minibench else extremize(p_before_ext, strength=0.3)
                logger.info(f"Extremized: {p_before_ext:.3f} → {p_after_ext:.3f} (tournament={tournament_slug})")
                o["probability"] = p_after_ext

        total        = float(sum(o["probability"] for o in aligned))
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
            question, research, model_probs, avg_prob, calibration_steps=["MC ensemble â†’ critic â†’ normalised"]
        )
        reasoning = (
            f"{self._methodology_header(research)}\n\n"
            + dr.render()
            + f"\n\n[avg_prob={avg_prob:.3f}]"
        )
        return ReasonedPrediction(prediction_value=final_val, reasoning=reasoning)

    # â”€â”€ Numeric forecasting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _run_forecast_on_numeric_generic(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        forecasters = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/anthropic/claude-sonnet-4.7",
        ]
        results: List[List[Percentile]] = await asyncio.gather(
            *[self._get_model_forecast(m, question, research) for m in forecasters]
        )
        forecast_map = {
            f"model_{i}": [{"percentile": float(p.percentile), "value": float(p.value)} for p in r]
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

        final_pcts = await self._parse_numeric_percentiles_robust(question, critique, stage="critic_numeric")
        final_pcts = self._enforce_monotone(final_pcts)
        med        = self._median_from_40_60(final_pcts)
        final_pcts = self._enforce_minimum_ci_width(final_pcts, med)

        dist = NumericDistribution.from_question(final_pcts, question)
        self._record_prediction(question, float(med / (abs(med) + 1.0)) if med else 0.0)

        model_medians = [self._median_from_40_60(r) for r in results]
        dr = await self._build_detailed_reasoning(
            question, research, model_medians, med,
            calibration_steps=[f"numeric ensemble â†’ critic; conservative SD Ã—{_NUMERIC_SD_CONSERVATIVE}; CI floor applied."],
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


# â”€â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the Advanced Forecasting Bot (botduke)")
    parser.add_argument("--mode", choices=["tournament","metaculus_cup","test_questions"], default="tournament")
    parser.add_argument("--bot-name", type=str, default="botduke")
    parser.add_argument("--no-extremize",        action="store_true")
    parser.add_argument("--no-decomposition",    action="store_true")
    parser.add_argument("--no-meta-forecast",    action="store_true")
    parser.add_argument("--no-numeric-regimes",  action="store_true")
    parser.add_argument("--no-detailed-reasoning", action="store_true")

    args     = parser.parse_args()
    run_mode: Literal["tournament","metaculus_cup","test_questions"] = args.mode

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
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        bot_name=args.bot_name,
        flags=flags,
    )

    client = MetaculusClient()

    async def run_all():
        if run_mode == "tournament":
            seasonal, minibench = await asyncio.gather(
                bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True),
                bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True),
            )
            return seasonal + minibench

        if run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            return await bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)

        bot.skip_previously_forecasted_questions = False
        EXAMPLE_QUESTION_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        questions = [client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTION_URLS]
        single_reports, market_pulse_q2_reports, summer_eval_reports = await asyncio.gather(
            bot.forecast_questions(questions, return_exceptions=True),
            bot.forecast_on_tournament("market-pulse-26q2", return_exceptions=True),
            bot.forecast_on_tournament("summer-futureeval-2026", return_exceptions=True),
        )
        return single_reports + market_pulse_q2_reports + summer_eval_reports

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
