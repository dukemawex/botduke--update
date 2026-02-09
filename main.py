import argparse
import asyncio
import logging
import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any, Dict, List, Optional, Tuple

import numpy as np
import dotenv
from pydantic import BaseModel, Field
import httpx

from tavily import TavilyClient

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
    Percentile,  # forecasting_tools.Percentile expects percentile in [0,1]
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


# ==========================================
# 🛡️ DATA SANITIZATION UTILITIES
# ==========================================

def sanitize_llm_json(text: str) -> str:
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f"\"{match.group(1)}\": {nums[0]}" if nums else match.group(0)

    text = re.sub(
        r"\"(value|percentile|probability|prediction_in_decimal|revised_prediction_in_decimal)\":\s*\"([^\"]+)\"",
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
            clean_data = sanitize_llm_json(s)
            return model_cls.model_validate_json(clean_data)
        if isinstance(data, dict):
            return model_cls.model_validate(data)
        return model_cls(**data)
    except Exception as e:
        logger.error(f"❌ MODEL INSTANTIATION FAILED for {model_cls.__name__}: {e}")
        raise


# ==========================================
# ✅ RAW MODELS TO AVOID Percentile VALIDATION ERRORS
# ==========================================

class RawPercentile(BaseModel):
    """
    Accepts percentiles as 10/20/... or 0.1/0.2/... and values as floats.
    We'll normalize to forecasting_tools.Percentile (percentile in [0,1]) later.
    """
    percentile: float = Field(..., description="Percentile as 10/20/... or 0.1/0.2/...")
    value: float


# ==========================================
# 🔍 EXA SEARCH CLIENT
# ==========================================

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
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippet = (r.get("text", "") or "")[:600]
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                return "[Exa Search Results]\n" + "\n\n".join(results)
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return "[Exa search failed]"


# ==========================================
# 🧠 FORECASTING PRINCIPLES
# ==========================================

class ForecastingPrinciples:
    @staticmethod
    def get_generic_base_rate() -> str:
        return (
            "BASE RATE: In the absence of strong evidence, default to historical frequencies "
            "or uniform priors where applicable. Most novel events have low base rates."
        )

    @staticmethod
    def get_generic_fermi_prompt() -> str:
        return """
FERMI GUIDANCE:
1) Define the target quantity precisely.
2) Decompose into drivers/factors.
3) Estimate each factor using available evidence.
4) Combine factors algebraically.
5) Quantify uncertainty; keep intervals wide unless evidence is strong.
""".strip()

    @staticmethod
    def apply_time_decay(prob: float, close_time: Optional[datetime]) -> float:
        if close_time is None:
            return prob
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days = (close_time - now).days
        if days > 365:
            return 0.3 * prob + 0.7 * 0.5
        elif days > 180:
            return 0.5 * prob + 0.5 * 0.5
        elif days > 90:
            return 0.7 * prob + 0.3 * 0.5
        else:
            return prob


# ==========================================
# 🤖 SPRING ADVANCED FORECASTING BOT
# ==========================================

class SpringAdvancedForecastingBot(ForecastBot):
    """
    Fixes in this revision:
      - NO hard failure if Tavily/Exa/AskNews not configured.
      - Research ALWAYS runs using whatever is available:
          Tavily / Exa / AskNews / (fallback) the bot's configured "researcher" LLM.
      - "Live web search" requirement is satisfied either via Tavily/Exa/AskNews OR via the web-enabled researcher model.
    """
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None
        self.asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")
        self._recent_predictions: list[tuple[MetaculusQuestion, float]] = []

    def _llm_config_defaults(self) -> Dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4o-mini",
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "openrouter/openai/gpt-4o-search-preview",
            "query_optimizer": "openrouter/openai/gpt-4o-mini",
            "critic": "openrouter/openai/gpt-5",
            "red_team": "openrouter/openai/gpt-4o",
        }

    # ------------------------------------------
    # Research footprint + gating (soft)
    # ------------------------------------------

    def _search_footprint(self, research: str) -> str:
        used: list[str] = []

        def ok(tag: str, fail_markers: list[str]) -> bool:
            return (tag in research) and (not any(m in research for m in fail_markers))

        if ok("[Tavily Data]", ["[Tavily not configured]", "[Tavily search failed]"]):
            used.append("tavily")
        if ok("[Exa Search Results]", ["[Exa not configured]", "[Exa search failed]"]):
            used.append("exa")
        if ok("[AskNews Data]", ["[AskNews not configured]", "[AskNews search failed]"]):
            used.append("asknews")
        if ok("[LLM Web Research]", ["[LLM web research failed]"]):
            used.append("llm_web")

        return ",".join(used) if used else "none"

    def _ensure_some_research_or_raise(self, research: str) -> None:
        """
        We still enforce: do not forecast without *some* research.
        But now "some research" can be from Tavily/Exa/AskNews OR the researcher LLM.
        """
        if self._search_footprint(research) == "none":
            raise RuntimeError(
                "No research evidence available (all providers failed, and LLM web research failed)."
            )

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none":
            return 0.25
        n = len(srcs.split(","))
        return {1: 0.55, 2: 0.75, 3: 0.85, 4: 0.90}.get(n, 0.6)

    # ------------------------------------------
    # Research
    # ------------------------------------------

    async def _optimize_search_query(self, question: MetaculusQuestion) -> List[str]:
        llm = self.get_llm("query_optimizer", "llm")
        prompt = f"""
Rewrite this forecasting question into 3 precise, factual web search queries.
Prefer entity names, key metrics, and date ranges.
Question: {question.question_text}
Output ONLY a JSON list: ["query1","query2","query3"]
""".strip()
        try:
            response = await llm.invoke(prompt)
            queries = json.loads(sanitize_llm_json(response))
            cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            return cleaned[:3] if cleaned else [question.question_text[:160]]
        except Exception:
            return [question.question_text[:160]]

    async def _run_tavily_search(self, query: str) -> str:
        if not self.tavily:
            return "[Tavily not configured]"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(query=query, search_depth="advanced", max_results=6),
            )
            context = "\n".join(
                [f"Source: {r.get('url','')}\nContent: {r.get('content','')}" for r in response.get("results", [])]
            )
            return f"[Tavily Data]\n{context}" if context.strip() else "[Tavily search failed]"
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=6)

    async def _run_asknews_search(self, query: str) -> str:
        if not self.asknews_client_id or not self.asknews_client_secret:
            return "[AskNews not configured]"
        try:
            searcher = AskNewsSearcher(
                client_id=self.asknews_client_id,
                client_secret=self.asknews_client_secret,
            )
            result = await searcher.call_preconfigured_version("asknews/news-summaries", query)
            return f"[AskNews Data]\n{result}" if str(result).strip() else "[AskNews search failed]"
        except Exception as e:
            logger.error(f"AskNews search failed: {e}")
            return "[AskNews search failed]"

    async def _run_llm_web_research(self, question: MetaculusQuestion) -> str:
        """
        Fallback research using the configured 'researcher' LLM.
        This is used when external APIs aren't configured or fail.
        """
        try:
            researcher = self.get_llm("researcher")
            prompt = clean_indents(
                f"""
You are an assistant to a superforecaster.
Use live web browsing/search (if available to your model) to gather the most relevant, recent evidence.
Provide:
- 6-12 bullet facts with sources/links when possible
- any key numbers / time series / market expectations
- what would make the question resolve each way (if applicable)
Do NOT give a final forecast.

Question:
{question.question_text}

Resolution criteria:
{question.resolution_criteria}

Fine print:
{question.fine_print}
"""
            )
            if isinstance(researcher, GeneralLlm):
                out = await researcher.invoke(prompt)
            else:
                out = await self.get_llm("researcher", "llm").invoke(prompt)
            out = (out or "").strip()
            if not out:
                return "[LLM web research failed]"
            return f"[LLM Web Research]\n{out}"
        except Exception as e:
            logger.error(f"LLM web research failed: {e}")
            return "[LLM web research failed]"

    async def run_research(self, question: MetaculusQuestion) -> str:
        queries = await self._optimize_search_query(question)
        optimized_query = " OR ".join(queries)

        tasks = [
            self._run_tavily_search(optimized_query),
            self._run_exa_search(optimized_query),
            self._run_asknews_search(optimized_query),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cleaned: list[str] = []
        for res in results:
            if isinstance(res, Exception):
                cleaned.append(f"[Search failed: {str(res)}]")
            else:
                cleaned.append(res)

        combined = "\n\n".join(cleaned).strip()

        # If external providers aren't usable, fallback to LLM web research
        if self._search_footprint(combined) == "none":
            combined = (combined + "\n\n" if combined else "") + await self._run_llm_web_research(question)

        base_rate = ForecastingPrinciples.get_generic_base_rate()
        fermi = ForecastingPrinciples.get_generic_fermi_prompt()

        research = f"""{base_rate}

{fermi}

{combined}"""

        # Now enforce: we got *something* usable (providers OR LLM web research)
        self._ensure_some_research_or_raise(research)
        return research

    # ------------------------------------------
    # Numeric helpers
    # ------------------------------------------

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> Tuple[str, str]:
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        unit = question.unit_of_measure or ""

        if getattr(question, "open_upper_bound", False):
            upper_msg = f"The question creator thinks the number is likely not higher than {upper} {unit}."
        else:
            upper_msg = f"The outcome can not be higher than {upper} {unit}."

        if getattr(question, "open_lower_bound", False):
            lower_msg = f"The question creator thinks the number is likely not lower than {lower} {unit}."
        else:
            lower_msg = f"The outcome can not be lower than {lower} {unit}."

        return upper_msg, lower_msg

    def _numeric_parsing_instructions(self, question: NumericQuestion) -> str:
        return clean_indents(
            f"""
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
"""
        )

    @staticmethod
    def _extract_percentile_block(text: str) -> str:
        m = re.search(
            r"(Percentile\s*10\s*:.*?Percentile\s*90\s*:.*?)(?:\n\s*\n|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
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
        by = {round(float(p.percentile), 3): p for p in pcts}
        missing = [r for r in required if round(r, 3) not in by]
        if missing:
            return []
        return [by[round(r, 3)] for r in required]

    @staticmethod
    def _enforce_monotone(pcts: List[Percentile]) -> List[Percentile]:
        pcts = sorted(pcts, key=lambda x: float(x.percentile))
        for i in range(1, len(pcts)):
            if pcts[i].value <= pcts[i - 1].value:
                pcts[i].value = pcts[i - 1].value + 1e-6
        return pcts

    @staticmethod
    def _bounds_fallback(question: NumericQuestion) -> List[Percentile]:
        lo = float(question.lower_bound)
        hi = float(question.upper_bound)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        w = {0.1: 0.05, 0.2: 0.15, 0.4: 0.40, 0.6: 0.60, 0.8: 0.85, 0.9: 0.95}
        pcts = [Percentile(percentile=p, value=lo + (hi - lo) * w[p]) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]]
        return SpringAdvancedForecastingBot._enforce_monotone(pcts)

    @staticmethod
    def _median_from_40_60(pcts: List[Percentile]) -> float:
        by = {round(float(p.percentile), 3): float(p.value) for p in pcts}
        if 0.4 in by and 0.6 in by:
            return 0.5 * (by[0.4] + by[0.6])
        return float(sorted(pcts, key=lambda x: x.percentile)[len(pcts) // 2].value) if pcts else 0.0

    async def _parse_numeric_percentiles_robust(self, question: NumericQuestion, text: str, stage: str) -> List[Percentile]:
        parser_llm = self.get_llm("parser", "llm")
        instructions = self._numeric_parsing_instructions(question)
        numeric_validation_samples = 1  # avoid sampled-output mismatch

        try:
            raw1: List[RawPercentile] = await structure_output(
                text,
                list[RawPercentile],
                model=parser_llm,
                additional_instructions=instructions,
                num_validation_samples=numeric_validation_samples,
            )
            p1 = self._normalize_raw_percentiles(raw1)
            std1 = self._require_standard_percentiles(p1)
            if std1:
                return self._enforce_monotone(std1)
        except Exception as e:
            logger.warning(f"[{stage}] numeric parse attempt 1 failed: {e}")

        block = self._extract_percentile_block(text)
        if block:
            try:
                raw2: List[RawPercentile] = await structure_output(
                    block,
                    list[RawPercentile],
                    model=parser_llm,
                    additional_instructions=instructions,
                    num_validation_samples=numeric_validation_samples,
                )
                p2 = self._normalize_raw_percentiles(raw2)
                std2 = self._require_standard_percentiles(p2)
                if std2:
                    return self._enforce_monotone(std2)
            except Exception as e:
                logger.warning(f"[{stage}] numeric parse attempt 2 failed: {e}")

        try:
            reform_prompt = clean_indents(
                f"""
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
"""
            )
            reformatted = await parser_llm.invoke(reform_prompt)
            rb = self._extract_percentile_block(reformatted) or reformatted
            raw3: List[RawPercentile] = await structure_output(
                rb,
                list[RawPercentile],
                model=parser_llm,
                additional_instructions=instructions,
                num_validation_samples=numeric_validation_samples,
            )
            p3 = self._normalize_raw_percentiles(raw3)
            std3 = self._require_standard_percentiles(p3)
            if std3:
                return self._enforce_monotone(std3)
        except Exception as e:
            logger.warning(f"[{stage}] numeric parse attempt 3 failed: {e}")

        logger.warning(f"[{stage}] numeric parsing failed; using bounds fallback.")
        return self._bounds_fallback(question)

    # ------------------------------------------
    # Forecasting
    # ------------------------------------------

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        if not getattr(question, "close_time", None):
            return 0.35
        days_to_close = (question.close_time - datetime.now(timezone.utc)).days
        qt = question.question_text.lower()
        if days_to_close > 180 or "first" in qt or "never before" in qt:
            return 0.35
        return 0.1

    async def _get_model_forecast(self, model_name: str, question: MetaculusQuestion, research: str) -> Any:
        # Ensure we have *some* research (providers or LLM web research)
        self._ensure_some_research_or_raise(research)

        temp = self._get_temperature(question)
        llm = GeneralLlm(model=model_name, temperature=temp)

        if isinstance(question, BinaryQuestion):
            raw = await llm.invoke(
                clean_indents(
                    f"""
Question: {question.question_text}

Research:
{research}

OUTPUT ONLY VALID JSON:
{{"prediction_in_decimal": 0.35}}
"""
                )
            )
            return await structure_output(
                sanitize_llm_json(raw),
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )

        if isinstance(question, MultipleChoiceQuestion):
            example_opts = [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]
            schema_example = json.dumps({"predicted_options": example_opts})
            raw = await llm.invoke(
                clean_indents(
                    f"""
Question: {question.question_text}
Options: {question.options}

Research:
{research}

OUTPUT ONLY VALID JSON:
{schema_example}
"""
                )
            )
            return await structure_output(
                sanitize_llm_json(raw),
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )

        if isinstance(question, NumericQuestion):
            upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
            units = question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"
            reasoning = await llm.invoke(
                clean_indents(
                    f"""
You are a professional forecaster.

Question:
{question.question_text}

Units for answer: {units}

Research:
{research}

Today is {datetime.now().strftime("%Y-%m-%d")}.

{lower_msg}
{upper_msg}

The LAST thing you write is EXACTLY:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""
                )
            )
            return await self._parse_numeric_percentiles_robust(question, reasoning, stage=f"model_forecast:{model_name}")

        raise TypeError(f"Unsupported question type: {type(question)}")

    async def _red_team_forecast(self, question: MetaculusQuestion, research: str, initial_pred: float) -> float:
        self._ensure_some_research_or_raise(research)

        try:
            llm = self.get_llm("red_team", "llm")
            response = await llm.invoke(
                clean_indents(
                    f"""
You are a skeptical red teamer.

Question: {question.question_text}
Research:
{research}

Current forecast: {initial_pred:.2%}

Output ONLY JSON:
{{"revised_prediction_in_decimal": 0.XX}}
"""
                )
            )
            parsed = await structure_output(
                sanitize_llm_json(response),
                dict,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
            )
            if isinstance(parsed, dict) and "revised_prediction_in_decimal" in parsed:
                val = float(parsed["revised_prediction_in_decimal"])
                return float(np.clip(val, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
        return initial_pred

    async def _check_consistency(self, question: MetaculusQuestion, proposed_pred: float) -> bool:
        if len(self._recent_predictions) < 2:
            return True
        recent_summary = "\n".join(
            [f"Q: {getattr(q, 'question_text', '')} → Pred: {p:.2%}" for q, p in self._recent_predictions[-3:]]
        )
        llm = self.get_llm("parser", "llm")
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
        except Exception:
            return True

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        self._ensure_some_research_or_raise(research)

        forecasters = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.5-sonnet",
        ]

        results = await asyncio.gather(*[self._get_model_forecast(m, question, research) for m in forecasters])
        forecast_map: Dict[str, float] = {
            f"model_{i}": float(r.prediction_in_decimal) for i, r in enumerate(results)
        }

        vals = list(forecast_map.values()) or [0.5]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0

        critic_llm = self.get_llm("critic", "llm")
        critique = await critic_llm.invoke(
            clean_indents(
                f"""
Question: {question.question_text}

Research:
{research}

Ensemble model forecasts:
{json.dumps(forecast_map)}

OUTPUT ONLY JSON:
{{"prediction_in_decimal": 0.75}}
"""
            )
        )
        critic_out = await structure_output(
            sanitize_llm_json(critique),
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        raw_p = float(critic_out.prediction_in_decimal)

        red_teamed_p = await self._red_team_forecast(question, research, raw_p)
        averaged_p = 0.5 * (raw_p + red_teamed_p)

        if spread >= 0.35:
            averaged_p = 0.6 * averaged_p + 0.4 * 0.5
        elif spread >= 0.20:
            averaged_p = 0.8 * averaged_p + 0.2 * 0.5

        if not await self._check_consistency(question, averaged_p):
            averaged_p = 0.5 * averaged_p + 0.5 * 0.5

        community = getattr(question, "community_prediction", None)
        quality = self._research_quality_weight(research)
        blended_p = (quality * averaged_p + (1 - quality) * float(community)) if (community is not None) else averaged_p

        final_p = ForecastingPrinciples.apply_time_decay(blended_p, getattr(question, "close_time", None))
        final_p = self.apply_bayesian_calibration(final_p * 100) / 100.0
        final_p = float(np.clip(final_p, 0.01, 0.99))

        self._recent_predictions.append((question, final_p))

        comment = (
            f"final={final_p:.3f} critic={raw_p:.3f} red={red_teamed_p:.3f} "
            f"spread={spread:.3f} search={self._search_footprint(research)}"
        )
        return ReasonedPrediction(prediction_value=final_p, reasoning=comment)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self._ensure_some_research_or_raise(research)

        forecasters = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.5-sonnet",
        ]
        results = await asyncio.gather(*[self._get_model_forecast(m, question, research) for m in forecasters])
        forecast_map = {f"model_{i}": r.model_dump() for i, r in enumerate(results)}

        critic_llm = self.get_llm("critic", "llm")
        example_opts = [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]
        schema_example = json.dumps({"predicted_options": example_opts})

        critique = await critic_llm.invoke(
            clean_indents(
                f"""
Question: {question.question_text}
Options: {question.options}

Research:
{research}

Ensemble model forecasts:
{json.dumps(forecast_map)}

OUTPUT ONLY JSON:
{schema_example}
"""
            )
        )
        final_list: PredictedOptionList = await structure_output(
            sanitize_llm_json(critique),
            PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )

        # Align + normalize
        option_names = question.options
        current = {o.option_name: float(o.probability) for o in final_list.predicted_options}
        aligned = [{"option_name": name, "probability": float(current.get(name, 0.0))} for name in option_names]
        total = float(sum(o["probability"] for o in aligned))
        if total <= 0:
            uniform = 1.0 / len(aligned)
            for o in aligned:
                o["probability"] = uniform
        else:
            for o in aligned:
                o["probability"] /= total

        final_val = safe_model(PredictedOptionList, {"predicted_options": aligned})
        avg_prob = float(np.mean([o["probability"] for o in aligned])) if aligned else 0.0
        self._recent_predictions.append((question, avg_prob))

        return ReasonedPrediction(prediction_value=final_val, reasoning=f"search={self._search_footprint(research)}")

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self._ensure_some_research_or_raise(research)

        forecasters = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.5-sonnet",
        ]
        results: List[List[Percentile]] = await asyncio.gather(*[self._get_model_forecast(m, question, research) for m in forecasters])

        forecast_map = {
            f"model_{i}": [{"percentile": float(p.percentile), "value": float(p.value)} for p in r]
            for i, r in enumerate(results)
        }

        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        units = question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"
        critic_llm = self.get_llm("critic", "llm")

        critique = await critic_llm.invoke(
            clean_indents(
                f"""
You are a professional forecaster.

Question:
{question.question_text}

Units for answer: {units}

Research:
{research}

Ensemble forecasts:
{json.dumps(forecast_map)}

Today is {datetime.now().strftime("%Y-%m-%d")}.

{lower_msg}
{upper_msg}

The LAST thing you write is EXACTLY:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""
            )
        )

        final_pcts = await self._parse_numeric_percentiles_robust(question, critique, stage="critic_numeric")
        final_pcts = self._enforce_monotone(final_pcts)

        dist = NumericDistribution.from_question(final_pcts, question)
        median_proxy = self._median_from_40_60(final_pcts)

        self._recent_predictions.append((question, float(median_proxy / (abs(median_proxy) + 1.0)) if median_proxy else 0.0))

        return ReasonedPrediction(
            prediction_value=dist,
            reasoning=f"search={self._search_footprint(research)} median≈{median_proxy:g}",
        )


# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the General-Purpose Advanced Forecasting Bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    bot = SpringAdvancedForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    client = MetaculusClient()

    async def run_all():
        if run_mode == "tournament":
            seasonal_task = bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
            minibench_task = bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
            seasonal, minibench = await asyncio.gather(seasonal_task, minibench_task)
            return seasonal + minibench

        if run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            return await bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )

        # test_questions: include Market Pulse 26Q1 + sample URLs
        bot.skip_previously_forecasted_questions = False

        EXAMPLE_QUESTION_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        questions = [client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTION_URLS]

        single_reports_task = bot.forecast_questions(questions, return_exceptions=True)
        market_pulse_task = bot.forecast_on_tournament("market-pulse-26q1", return_exceptions=True)

        single_reports, market_pulse_reports = await asyncio.gather(single_reports_task, market_pulse_task)
        return single_reports + market_pulse_reports

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
