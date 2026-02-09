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
from pydantic import BaseModel
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
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

# Load environment variables
dotenv.load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ==========================================
# 🛡️ DATA SANITIZATION UTILITIES
# ==========================================

def sanitize_llm_json(text: str) -> str:
    # remove digit_underscore_digit patterns (e.g., 1_000 -> 1000)
    text = re.sub(r"(?<=\d)_(?=\d)", "", text)

    # coerce some numeric strings into numbers
    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f"\"{match.group(1)}\": {nums[0]}" if nums else match.group(0)

    text = re.sub(r"\"(value|percentile|probability|prediction_in_decimal)\":\s*\"([^\"]+)\"", clean_num, text)

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    """
    Robustly coerce `data` into `model_cls`.
    Supports:
      - already-instantiated model
      - JSON string/bytes
      - dict
      - kwargs-like objects
    """
    try:
        if isinstance(data, model_cls):
            return data
        if isinstance(data, (str, bytes)):
            s = data.decode() if isinstance(data, bytes) else data
            clean_data = sanitize_llm_json(s)
            return model_cls.model_validate_json(clean_data)
        if isinstance(data, dict):
            return model_cls.model_validate(data)
        return model_cls(**data)  # last resort
    except Exception as e:
        logger.error(f"❌ MODEL INSTANTIATION FAILED for {model_cls.__name__}: {e}")
        raise


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
                    snippet = (r.get("text", "") or "")[:500]
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
Decompose the problem into independent factors whose probabilities or values can be estimated.
Multiply or combine these factors logically.
Account for uncertainty in each step.
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
    Updated numeric handling to match the SpringTemplateBot2026 contract:
      - Forecasting LLM produces reasoning + a strict percentile block (10/20/40/60/80/90)
      - Explicit formatting constraints (units, no scientific notation, increasing percentiles)
      - Parser LLM extracts list[Percentile] with unit normalization instructions
      - NumericDistribution.from_question(...) builds the final distribution
    """
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize external clients only if keys are present
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
    # Search footprint + brief comments
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

        return ",".join(used) if used else "none"

    def _research_quality_weight(self, research: str) -> float:
        srcs = self._search_footprint(research)
        if srcs == "none":
            return 0.25
        n = len(srcs.split(","))
        return {1: 0.55, 2: 0.75, 3: 0.85}.get(n, 0.6)

    def _brief_binary_comment(
        self,
        forecast_map: Dict[str, float],
        raw_p: float,
        red_teamed_p: float,
        final_p: float,
        research: str,
    ) -> str:
        vals = list(forecast_map.values()) or [0.5]
        med = float(np.median(vals))
        spread = float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0
        srcs = self._search_footprint(research)
        return (
            f"final={final_p:.3f} critic={raw_p:.3f} red={red_teamed_p:.3f} "
            f"med={med:.3f} spread={spread:.3f} search={srcs}"
        )

    def _brief_mcq_comment(self, research: str) -> str:
        return f"search={self._search_footprint(research)}"

    def _brief_numeric_comment(self, research: str) -> str:
        return f"search={self._search_footprint(research)}"

    # ------------------------------------------
    # Calibration
    # ------------------------------------------

    def apply_bayesian_calibration(self, estimate_pct: float) -> float:
        p = estimate_pct / 100.0

        if p >= 0.99:
            p = 0.96 + 0.01 * min((p - 0.99) / 0.01, 1.0)
        elif p >= 0.95:
            p = 0.92 + 0.04 * ((p - 0.95) / 0.04)
        elif p >= 0.90:
            p = 0.88 + 0.04 * ((p - 0.90) / 0.05)
        elif p <= 0.01:
            p = 0.008 + 0.022 * (p / 0.01)
        elif p <= 0.05:
            p = 0.03 + 0.05 * ((p - 0.01) / 0.04)
        elif p <= 0.10:
            p = 0.08 + 0.04 * ((p - 0.05) / 0.05)

        p = np.clip(p, 0.005, 0.995)
        return round(float(p * 100), 2)

    # ------------------------------------------
    # Research
    # ------------------------------------------

    async def _optimize_search_query(self, question: MetaculusQuestion) -> List[str]:
        llm = self.get_llm("query_optimizer", "llm")
        prompt = f"""
Rewrite this forecasting question into 3 precise, factual search queries for news/reports.
Focus on entities, dates, and measurable outcomes.
Question: {question.question_text}
Output ONLY a JSON list: ["query1", "query2", "query3"]
""".strip()
        try:
            response = await llm.invoke(prompt)
            queries = json.loads(sanitize_llm_json(response))
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:3]
        except Exception:
            return [question.question_text[:150]]

    async def _run_tavily_search(self, query: str) -> str:
        if not self.tavily:
            return "[Tavily not configured]"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(query=query, search_depth="advanced", max_results=5),
            )
            context = "\n".join(
                [f"Source: {r['url']}\nContent: {r['content']}" for r in response.get("results", [])]
            )
            return f"[Tavily Data]\n{context}"
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=5)

    async def _run_asknews_search(self, query: str) -> str:
        if not self.asknews_client_id or not self.asknews_client_secret:
            return "[AskNews not configured]"
        try:
            searcher = AskNewsSearcher(
                client_id=self.asknews_client_id,
                client_secret=self.asknews_client_secret,
            )
            result = await searcher.call_preconfigured_version("asknews/news-summaries", query)
            return f"[AskNews Data]\n{result}"
        except Exception as e:
            logger.error(f"AskNews search failed: {e}")
            return "[AskNews search failed]"

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

        combined = "\n\n".join(cleaned)

        base_rate = ForecastingPrinciples.get_generic_base_rate()
        fermi = ForecastingPrinciples.get_generic_fermi_prompt()

        return f"""{base_rate}

{fermi}

{combined}"""

    # ------------------------------------------
    # Helpers: numeric bounds + numeric parsing instructions
    # ------------------------------------------

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> Tuple[str, str]:
        # Mirrors the template logic (nominal bounds preferred for messaging)
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound

        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        unit = question.unit_of_measure or ""

        if getattr(question, "open_upper_bound", False):
            upper_msg = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit}."
        else:
            upper_msg = f"The outcome can not be higher than {upper_bound_number} {unit}."

        if getattr(question, "open_lower_bound", False):
            lower_msg = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit}."
        else:
            lower_msg = f"The outcome can not be lower than {lower_bound_number} {unit}."

        return upper_msg, lower_msg

    def _numeric_parsing_instructions(self, question: NumericQuestion) -> str:
        return clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers you parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure},
              so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units.
              For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output,
              but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            - IMPORTANT: the required percentiles are exactly 10, 20, 40, 60, 80, and 90.
            """
        )

    def _enforce_numeric_percentiles_monotone(self, pcts: List[Percentile]) -> List[Percentile]:
        # Sort by percentile, and ensure increasing values
        pcts = sorted(pcts, key=lambda x: x.percentile)
        for i in range(1, len(pcts)):
            if pcts[i].value <= pcts[i - 1].value:
                pcts[i].value = pcts[i - 1].value + 1e-6
        return pcts

    def _require_standard_numeric_percentiles(self, pcts: List[Percentile]) -> List[Percentile]:
        # Require the exact set used by the template bot
        required = [10, 20, 40, 60, 80, 90]
        by_pct = {int(p.percentile): p for p in pcts if int(p.percentile) in required}
        missing = [r for r in required if r not in by_pct]
        if missing:
            raise ValueError(f"Missing required percentiles: {missing}. Got: {sorted(by_pct.keys())}")
        return [by_pct[r] for r in required]

    def _pseudo_median_from_40_60(self, pcts: List[Percentile]) -> float:
        by_pct = {int(p.percentile): float(p.value) for p in pcts}
        if 40 in by_pct and 60 in by_pct:
            return 0.5 * (by_pct[40] + by_pct[60])
        # fallback: use 60 if present else 40 else first
        if 60 in by_pct:
            return by_pct[60]
        if 40 in by_pct:
            return by_pct[40]
        return float(pcts[len(pcts) // 2].value) if pcts else 0.0

    # ------------------------------------------
    # Forecasting
    # ------------------------------------------

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        if not getattr(question, "close_time", None):
            return 0.4
        days_to_close = (question.close_time - datetime.now(timezone.utc)).days
        qt = question.question_text.lower()
        if days_to_close > 180 or "first" in qt or "never before" in qt:
            return 0.4
        return 0.1

    async def _get_model_forecast(self, model_name: str, question: MetaculusQuestion, research: str) -> Any:
        """
        Returns:
          - BinaryQuestion -> BinaryPrediction
          - MultipleChoiceQuestion -> PredictedOptionList
          - NumericQuestion -> list[Percentile] (standard 10/20/40/60/80/90), parsed from a formatted text block
        """
        temp = self._get_temperature(question)
        llm = GeneralLlm(model=model_name, temperature=temp)

        if isinstance(question, BinaryQuestion):
            schema_example = '{"prediction_in_decimal": 0.35}'
            out_type = BinaryPrediction
            prompt = clean_indents(
                f"""
                Question: {question.question_text}
                Research: {research}

                Apply forecasting best practices:
                - Start from general priors
                - Decompose complex problems
                - Avoid over-updating on recent news
                - Favor structural stability
                - Quantify uncertainty

                OUTPUT ONLY VALID JSON:
                {schema_example}
                """
            )
            raw = await llm.invoke(prompt)
            return await structure_output(
                sanitize_llm_json(raw),
                out_type,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )

        if isinstance(question, MultipleChoiceQuestion):
            example_opts = [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]
            schema_example = json.dumps({"predicted_options": example_opts})
            out_type = PredictedOptionList
            prompt = clean_indents(
                f"""
                Question: {question.question_text}
                Options: {question.options}
                Research: {research}

                Apply forecasting best practices:
                - Start from general priors
                - Decompose complex problems
                - Avoid over-updating on recent news
                - Favor structural stability
                - Quantify uncertainty
                - Leave some probability mass on unexpected outcomes

                OUTPUT ONLY VALID JSON:
                {schema_example}
                """
            )
            raw = await llm.invoke(prompt)
            return await structure_output(
                sanitize_llm_json(raw),
                out_type,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )

        # Numeric: adopt template-style formatted percentile block + parser extraction
        if isinstance(question, NumericQuestion):
            upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
            units = question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"
            prompt = clean_indents(
                f"""
                You are a professional forecaster.

                Your question is:
                {question.question_text}

                Background:
                {question.background_info}

                Resolution criteria:
                {question.resolution_criteria}

                {question.fine_print}

                Units for answer: {units}

                Research:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                {lower_msg}
                {upper_msg}

                Formatting Instructions:
                - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
                - Never use scientific notation.
                - Always start with a smaller number (more negative if negative) and then increase from there.
                  The value for percentile 10 should always be less than the value for percentile 20, and so on.

                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The outcome if nothing changed.
                (c) The outcome if the current trend continued.
                (d) The expectations of experts and markets.
                (e) A brief description of an unexpected scenario that results in a low outcome.
                (f) A brief description of an unexpected scenario that results in a high outcome.

                You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

                The last thing you write is your final answer as:
                "
                Percentile 10: XX (lowest number value)
                Percentile 20: XX
                Percentile 40: XX
                Percentile 60: XX
                Percentile 80: XX
                Percentile 90: XX (highest number value)
                "
                """
            )

            reasoning = await llm.invoke(prompt)

            parsed: List[Percentile] = await structure_output(
                reasoning,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=self._numeric_parsing_instructions(question),
                num_validation_samples=self._structure_output_validation_samples,
            )

            parsed = self._require_standard_numeric_percentiles(parsed)
            parsed = self._enforce_numeric_percentiles_monotone(parsed)
            return parsed

        raise TypeError(f"Unsupported question type: {type(question)}")

    async def _red_team_forecast(self, question: MetaculusQuestion, research: str, initial_pred: float) -> float:
        try:
            llm = self.get_llm("red_team", "llm")
            prompt = clean_indents(
                f"""
                You are a skeptical red teamer challenging this forecast: {initial_pred:.2%}.
                Question: {question.question_text}
                Research: {research}

                Identify 3 strongest reasons why this forecast is TOO HIGH or TOO LOW.
                Then output ONLY: {{"revised_prediction_in_decimal": 0.XX}}
                Avoid markdown. Only JSON.
                """
            )
            response = await llm.invoke(prompt)
            revised = await structure_output(
                sanitize_llm_json(response),
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            return revised.prediction_in_decimal
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
            return initial_pred

    async def _verify_claims(self, draft_reasoning: str, research: str) -> str:
        # Keep but don’t inflate output; we won’t append to reasoning (only to research internally)
        try:
            llm = self.get_llm("parser", "llm")
            extract_prompt = f"List up to 3 key factual claims in this reasoning:\n{draft_reasoning}"
            claims_response = await llm.invoke(extract_prompt)
            claims = [c.strip() for c in claims_response.split("\n") if c.strip()][:3]

            verified: list[str] = []
            for claim in claims:
                verification = await self._run_tavily_search(f"Verify: {claim}")
                verified.append(f"Claim: {claim}\nEvidence: {verification[:250]}")
            return "\n\n".join(verified)
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return ""

    async def _check_consistency(self, question: MetaculusQuestion, proposed_pred: float) -> bool:
        if len(self._recent_predictions) < 2:
            return True

        recent_summary = "\n".join(
            [
                f"Q: {getattr(q, 'question_text', getattr(q, 'text', ''))} → Pred: {p:.2%}"
                for q, p in self._recent_predictions[-3:]
            ]
        )
        llm = self.get_llm("parser", "llm")
        prompt = f"""
Is this new forecast logically consistent with prior forecasts?
New: {question.question_text} → {proposed_pred:.2%}
Prior: {recent_summary}
Answer YES or NO only.
""".strip()
        try:
            response = await llm.invoke(prompt)
            return "YES" in response.upper()
        except Exception:
            return True

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        forecasters = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.5-sonnet",
        ]

        tasks = [self._get_model_forecast(m, question, research) for m in forecasters]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        forecast_map: Dict[str, float] = {
            f"model_{i}": (r.prediction_in_decimal if r else 0.5) for i, r in enumerate(results)
        }

        # disagreement-based shrinkage
        vals = list(forecast_map.values()) or [0.5]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0

        critic_llm = self.get_llm("critic", "llm")
        schema_example = '{"prediction_in_decimal": 0.75}'
        prompt = clean_indents(
            f"""
Question: {question.question_text}
Research: {research}
Ensemble Forecasts: {json.dumps(forecast_map)}

Apply forecasting best practices:
- Start from general priors
- Decompose complex problems
- Avoid recency/salience bias
- Favor structural stability
- Ensure logical consistency

OUTPUT ONLY VALID JSON:
{schema_example}
"""
        )
        critique = await critic_llm.invoke(prompt)
        critic_out = await structure_output(
            sanitize_llm_json(critique),
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        raw_p = critic_out.prediction_in_decimal

        red_teamed_p = await self._red_team_forecast(question, research, raw_p)
        averaged_p = (raw_p + red_teamed_p) / 2.0

        # shrink if forecasters disagree
        if spread >= 0.35:
            averaged_p = 0.6 * averaged_p + 0.4 * 0.5
        elif spread >= 0.20:
            averaged_p = 0.8 * averaged_p + 0.2 * 0.5

        verification = await self._verify_claims(critique, research)
        if verification:
            research += f"\n\n[VERIFICATION]\n{verification}"

        if not await self._check_consistency(question, averaged_p):
            logger.warning("Inconsistency detected; pulling toward 50%")
            averaged_p = 0.5 * averaged_p + 0.5 * 0.5

        community = getattr(question, "community_prediction", None)
        research_quality = self._research_quality_weight(research)
        if community is not None:
            blended_p = research_quality * averaged_p + (1 - research_quality) * community
        else:
            blended_p = averaged_p

        final_p = ForecastingPrinciples.apply_time_decay(blended_p, question.close_time)
        final_p = self.apply_bayesian_calibration(final_p * 100) / 100.0

        # cap near-impossible
        if any(x in research.lower() for x in ["physically impossible", "logically impossible", "violates known laws"]):
            final_p = min(final_p, 0.03)

        self._recent_predictions.append((question, final_p))

        comment = self._brief_binary_comment(
            forecast_map=forecast_map,
            raw_p=raw_p,
            red_teamed_p=red_teamed_p,
            final_p=final_p,
            research=research,
        )

        return ReasonedPrediction(prediction_value=final_p, reasoning=comment)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        forecasters = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.5-sonnet",
        ]
        tasks = [self._get_model_forecast(m, question, research) for m in forecasters]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        forecast_map = {f"model_{i}": (r.model_dump() if r else {}) for i, r in enumerate(results)}

        critic_llm = self.get_llm("critic", "llm")
        example_opts = [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]
        schema_example = json.dumps({"predicted_options": example_opts})
        prompt = clean_indents(
            f"""
Question: {question.question_text}
Options: {question.options}
Research: {research}
Ensemble Forecasts: {json.dumps(forecast_map)}

Apply forecasting best practices:
- Start from general priors
- Avoid recency/salience bias
- Favor structural stability
- Leave some probability mass on unexpected outcomes
- Ensure probabilities sum to 1

OUTPUT ONLY VALID JSON:
{schema_example}
"""
        )
        critique = await critic_llm.invoke(prompt)
        final_list: PredictedOptionList = await structure_output(
            sanitize_llm_json(critique),
            PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )

        # Align to provided option order and normalize
        option_names = question.options
        current_options = {o.option_name: o.probability for o in final_list.predicted_options}
        aligned_options = [{"option_name": name, "probability": float(current_options.get(name, 0.0))} for name in option_names]
        total = float(sum(o["probability"] for o in aligned_options))
        if total == 0:
            uniform_p = 1.0 / len(aligned_options)
            for o in aligned_options:
                o["probability"] = uniform_p
        else:
            for o in aligned_options:
                o["probability"] /= total

        final_val = safe_model(PredictedOptionList, {"predicted_options": aligned_options})
        avg_prob = float(np.mean([opt["probability"] for opt in aligned_options])) if aligned_options else 0.0
        self._recent_predictions.append((question, avg_prob))

        return ReasonedPrediction(
            prediction_value=final_val,
            reasoning=self._brief_mcq_comment(research),
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Updated to match the template numeric format:
          - forecast models produce percentile block (10/20/40/60/80/90)
          - critic also produces the same block
          - parser extracts list[Percentile] with unit normalization instructions
        """
        forecasters = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.5-sonnet",
        ]
        tasks = [self._get_model_forecast(m, question, research) for m in forecasters]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # results are list[Percentile]
        forecast_map = {
            f"model_{i}": ([p.model_dump() for p in r] if r else [])
            for i, r in enumerate(results)
        }

        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        units = question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"

        critic_llm = self.get_llm("critic", "llm")
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {units}

            Research:
            {research}

            Ensemble Forecasts (already parsed to percentiles):
            {json.dumps(forecast_map)}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units.
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there.
              The value for percentile 10 should always be less than percentile 20, and so on.
            - Use a wide 90/10 interval to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        critique = await critic_llm.invoke(prompt)

        final_pcts: List[Percentile] = await structure_output(
            critique,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=self._numeric_parsing_instructions(question),
            num_validation_samples=self._structure_output_validation_samples,
        )

        final_pcts = self._require_standard_numeric_percentiles(final_pcts)
        final_pcts = self._enforce_numeric_percentiles_monotone(final_pcts)

        dist = NumericDistribution.from_question(final_pcts, question)

        # Update recent_predictions with a stable proxy based on pseudo-median (40/60 average)
        pseudo_median = self._pseudo_median_from_40_60(final_pcts)
        self._recent_predictions.append(
            (question, float(pseudo_median / (abs(pseudo_median) + 1.0)) if pseudo_median else 0.0)
        )

        return ReasonedPrediction(
            prediction_value=dist,
            reasoning=self._brief_numeric_comment(research),
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

        # test_questions
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTIONS]
        return await bot.forecast_questions(questions, return_exceptions=True)

    reports = asyncio.run(run_all())
    bot.log_report_summary(reports)
