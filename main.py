import argparse
import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Literal, List, Any, Dict, Union, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from tavily import TavilyClient
from asknews_sdk import AskNewsSDK

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
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

logger = logging.getLogger(__name__)


def get_model_id(base_name: str) -> str:
    """Resolve model ID with graceful fallbacks."""
    overrides = {
        "gpt-5.1": os.getenv("FORCE_GPT_5_1_MODEL"),
        "claude-sonnet-4.5": os.getenv("FORCE_CLAUDE_SONNET_4_5_MODEL"),
    }
    if overrides.get(base_name):
        return overrides[base_name]
    
    fallbacks = {
        "gpt-5.1": [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o-2024-11-20",
        ],
        "claude-sonnet-4.5": [
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/anthropic/claude-3-sonnet",
        ],
    }
    return fallbacks.get(base_name, [base_name])[0]


class ConfidenceWeightedEnsembleBot2025(ForecastBot):
    """
    Financial-Credentialed Ensemble Forecaster
    - Research: AskNews + Tavily
    - Models: gpt-5, gpt-5.1, claude-sonnet-4.5
    - Special handling for financial questions
    - Type-safe parsing with dedicated summarizer & parser LLMs
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    MAX_TAVILY_QUERIES = 200
    MAX_ASKNEWS_QUERIES = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 🔑 API clients
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.asknews = AskNewsSDK(
            client_id=os.getenv("ASKNEWS_CLIENT_ID"),
            client_secret=os.getenv("ASKNEWS_CLIENT_SECRET"),
        )

        # 📊 Query counters
        self._tavily_count = 0
        self._asknews_count = 0
        self._query_lock = asyncio.Lock()

        # 🤖 Forecast models
        self.FORECAST_MODELS = {
            "gpt-5": "openrouter/openai/gpt-5",
            "gpt-5.1": get_model_id("gpt-5.1"),
            "claude-sonnet-4.5": get_model_id("claude-sonnet-4.5"),
        }
        self.BASE_WEIGHTS = {
            "gpt-5": 1.0,
            "gpt-5.1": 1.3,
            "claude-sonnet-4.5": 1.1,
        }

        # ✅ Dedicated LLMs for roles
        self.researcher_llm = self.get_llm("researcher", "llm")
        self.summarizer_llm = self.get_llm("summarizer", "llm")
        self.parser_llm = self.get_llm("parser", "llm")

        self._current_question: Optional[MetaculusQuestion] = None

    # ✅ Override get_llm to ensure roles resolve properly
    def get_llm(self, role: str, guarantee_type: Literal["llm", "model_name"] = "llm") -> Any:
        if role in ("default", "researcher", "summarizer", "parser"):
            # Let parent handle, but ensure fallbacks
            try:
                return super().get_llm(role, guarantee_type)
            except Exception:
                # Fallback to reasonable defaults
                fallback_map = {
                    "researcher": GeneralLlm(model=get_model_id("gpt-5.1"), temperature=0.3, timeout=60),
                    "summarizer": GeneralLlm(model=get_model_id("gpt-5.1"), temperature=0.2, timeout=45),
                    "parser": GeneralLlm(model=get_model_id("gpt-5.1"), temperature=0.0, timeout=30),
                }
                if role in fallback_map:
                    logger.warning(f"Using fallback LLM for role '{role}'")
                    return fallback_map[role]
                raise
        return super().get_llm(role, guarantee_type)

    async def _increment_query_count(self, source: str) -> bool:
        async with self._query_lock:
            if source == "tavily" and self._tavily_count >= self.MAX_TAVILY_QUERIES:
                return False
            if source == "asknews" and self._asknews_count >= self.MAX_ASKNEWS_QUERIES:
                return False
            if source == "tavily":
                self._tavily_count += 1
            elif source == "asknews":
                self._asknews_count += 1
            return True

    def _is_financial_question(self, q: MetaculusQuestion) -> bool:
        text = " ".join([
            q.question_text,
            q.background_info or "",
            q.resolution_criteria or "",
        ]).lower()
        financial_terms = [
            r"\b(stock|equity|share|s&p|nasdaq|dow|djia|index|ticker|nyse|cpi|fed|interest rate|yield curve|vix|volatility)\b",
            r"\$[a-z]{1,5}\b",
            r"\b(?:[a-z]{1,4}\.?\s*[ou]n\s+\d{1,2}/\d{1,2})\b",
            r"\bby\s+\d{4}\b.*\b(s&p|nasdaq)",
        ]
        return any(re.search(pat, text) for pat in financial_terms)

    async def run_research(self, question: MetaculusQuestion) -> str:
        self._current_question = question
        async with self._concurrency_limiter:
            try:
                q_text = f"{question.question_text} {question.resolution_criteria}"[:250]
                is_financial = self._is_financial_question(question)

                asknews_query = f"Financial market update: {q_text}" if is_financial else q_text
                tavily_query = f"Long-term trends and base rates for: {q_text}" if is_financial else q_text

                tavily_res = await self._run_tavily_research(tavily_query)
                asknews_res = await self._run_asknews_research(asknews_query, financial=is_financial)

                combined = clean_indents(f"""
                ### Research Summary (as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
                {"[FINANCIAL QUESTION]" if is_financial else ""}

                {'[AskNews — Breaking Events]' + chr(10) + asknews_res if asknews_res else ''}
                {'[Tavily — Deep Context]' + chr(10) + tavily_res if tavily_res else ''}

                ⚠️ Forecaster Guidance:
                - Anchor to base rates.
                - High VIX → widen intervals.
                - Respect resolution criteria strictly.
                """)
                return combined
            except Exception as e:
                logger.error(f"Research failed for Q{question.id}: {e}")
                return f"⚠️ Research unavailable (error: {e})"

    async def _run_tavily_research(self, query: str) -> str:
        if not await self._increment_query_count("tavily"):
            return "[Skipped: Tavily quota exceeded]"
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    include_answer=True,
                    max_results=4,
                    days=365,
                ),
            )
            answer = res.get("answer", "").strip() or "[No summary]"
            snippets = "\n".join(
                f"• {r['title'][:60]}: {r['content'][:200]}..."
                for r in res.get("results", [])[:3]
            )
            return f"Summary: {answer}\nSources:\n{snippets or 'None'}"
        except Exception as e:
            return f"[Tavily error: {e}]"

    async def _run_asknews_research(self, query: str, financial: bool = False) -> str:
        if not await self._increment_query_count("asknews"):
            return "[Skipped: AskNews quota exceeded]"
        try:
            categories = ["business", "finance"] if financial else ["politics", "business", "tech"]
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.asknews.news.search_news(
                    query=query,
                    n_articles=4,
                    return_type="news",
                    categories=categories,
                    max_age_days=30 if financial else 90,
                ),
            )
            articles = response.data.articles
            if not articles:
                return "[No recent news]"
            snippets = "\n".join(
                f"• {a.headline[:70]} ({a.date[:10]}) — {a.snippet[:200]}..."
                for a in articles[:4]
            )
            sentiment = response.data.sentiment
            sent_str = f"Overall sentiment: {sentiment.overall:+.2f} (pos={sentiment.positive:.2f}, neg={sentiment.negative:.2f})"
            return f"{snippets}\n{sent_str}"
        except Exception as e:
            return f"[AskNews error: {e}]"

    # ===== QUESTION CATEGORIZATION & WEIGHTING =====
    def _categorize_question(self, q: MetaculusQuestion) -> str:
        # ✅ FIRST: use actual type — most reliable
        if isinstance(q, BinaryQuestion):
            return "binary"
        elif isinstance(q, MultipleChoiceQuestion):
            return "mcq"
        elif isinstance(q, NumericQuestion):
            return "numeric"
        # Fallback to heuristics
        text = (q.question_text + " " + (q.background_info or "")).lower()
        if "which" in text or "option" in text or hasattr(q, 'options'):
            return "mcq"
        if "probability" in text or "will it" in text:
            return "binary"
        return "numeric"

    def _get_dynamic_weights(self, question: MetaculusQuestion) -> Dict[str, float]:
        weights = self.BASE_WEIGHTS.copy()
        q_text = question.question_text.lower()
        q_type = self._categorize_question(question)
        is_financial = self._is_financial_question(question)

        if q_type == "binary":
            weights["gpt-5.1"] *= 1.4
        elif q_type == "numeric":
            if is_financial:
                weights["gpt-5"] *= 1.2
                weights["claude-sonnet-4.5"] *= 1.1
            else:
                weights["gpt-5"] *= 1.3
        elif q_type == "mcq":
            weights["claude-sonnet-4.5"] *= 1.3
        if is_financial:
            weights["gpt-5.1"] *= 1.1

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    # ===== ROBUST CONFIDENCE EXTRACTION =====
    def _extract_confidence(self, text: str, is_financial: bool = False) -> float:
        # Try multiple regex patterns
        patterns = [
            r"Confidence:\s*([\d.]+)%?",
            r"confidence level.*?([\d.]+)%?",
            r"I am ([\d.]+)%.*confiden",
            r"self-confidence:\s*([\d.]+)%?",
            r"(?:certainty|sureness):\s*([\d.]+)%?",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    conf = float(match.group(1)) / 100.0
                    conf = np.clip(conf, 0.3, 1.0)
                    if is_financial:
                        conf *= 0.9
                    return conf
                except (ValueError, TypeError):
                    continue

        # Semantic fallback
        text_lower = text.lower()
        low_conf_indicators = ["uncertain", "speculative", "hard to say", "guess", "low confidence"]
        high_conf_indicators = ["certain", "very likely", "clear", "definitive", "high confidence"]
        
        if any(word in text_lower for word in low_conf_indicators):
            return 0.5
        if any(word in text_lower for word in high_conf_indicators):
            return 0.85
        return 0.7  # default

    def _get_bounds(self, q: NumericQuestion) -> Tuple[str, str]:
        lo = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        hi = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        lo_str = f"≥{lo}" if q.open_lower_bound else f"={lo} (hard min)"
        hi_str = f"≤{hi}" if q.open_upper_bound else f"={hi} (hard max)"
        return lo_str, hi_str

    def _apply_financial_adjustment(self, dist: NumericDistribution, question: NumericQuestion) -> NumericDistribution:
        try:
            pcts = {int(p.percentile * 100): p.value for p in dist.declared_percentiles}
            needed = [10, 20, 40, 50, 60, 80, 90]

            if 50 not in pcts:
                if 40 in pcts and 60 in pcts:
                    pcts[50] = (pcts[40] + pcts[60]) / 2
                else:
                    pcts[50] = np.median(list(pcts.values()))

            median = pcts[50]
            if median <= 0:
                return dist

            log_pcts = {p: np.log(v) for p, v in pcts.items() if v > 0}
            new_log_pcts = {}
            for p in needed:
                if p in log_pcts:
                    shrink = 0.4 if abs(p - 50) >= 30 else 0.15
                    new_log = (1 - shrink) * log_pcts[p] + shrink * np.log(median)
                    new_log_pcts[p] = new_log
                else:
                    lower = max([k for k in log_pcts.keys() if k <= p], default=min(log_pcts.keys()))
                    upper = min([k for k in log_pcts.keys() if k >= p], default=max(log_pcts.keys()))
                    if lower == upper:
                        new_log_pcts[p] = log_pcts[lower]
                    else:
                        w = (p - lower) / (upper - lower)
                        new_log_pcts[p] = (1 - w) * log_pcts[lower] + w * log_pcts[upper]

            new_pcts = {p: np.exp(v) for p, v in new_log_pcts.items()}
            final_percentiles = [
                Percentile(percentile=p / 100.0, value=new_pcts[p])
                for p in [10, 20, 40, 60, 80, 90]
            ]
            return NumericDistribution.from_question(final_percentiles, question)
        except Exception as e:
            logger.warning(f"Financial adjustment failed: {e}")
            return dist

    # ===== CORE FORECASTING WITH TYPE-SAFE PARSING =====
    async def _run_forecast_with_confidence(
        self, question: MetaculusQuestion, research: str, model_key: str
    ) -> Tuple[ReasonedPrediction, float]:
        model_name = self.FORECAST_MODELS[model_key]
        forecaster_llm = GeneralLlm(model=model_name, temperature=0.2, timeout=60, allowed_tries=2)

        is_financial = self._is_financial_question(question)
        q_type = self._categorize_question(question)

        # Build prompt
        base_prompt = clean_indents(f"""
        You are a professional superforecaster with finance expertise where relevant.

        Question: {question.question_text}
        Resolution: {question.resolution_criteria}
        Background: {question.background_info or 'None'}
        Today: {datetime.now().strftime('%Y-%m-%d')}

        Research:
        {research}
        """)

        if q_type == "binary":
            prompt = base_prompt + clean_indents(f"""
            Analyze:
            (a) Time to resolution
            (b) Base rate
            {"(c) Market-implied probability" if is_financial else ""}
            (d) Status quo

            Output:
            Rationale: ...
            Probability: ZZ% (0–100)
            Confidence: WW% (50=guess, 100=certain)
            """)
        elif q_type == "mcq":
            options = getattr(question, "options", [])
            prompt = base_prompt + clean_indents(f"""
            Options: {options}

            Analyze:
            (a) Most likely option
            (b) Plausible alternatives
            (c) Why others are less likely

            Output probabilities (sum=100%), then:
            Confidence: WW%
            """)
        else:  # numeric
            lo, hi = self._get_bounds(question)
            prompt = base_prompt + clean_indents(f"""
            Units: {getattr(question, 'unit_of_measure', 'inferred')}
            Bounds: {lo} to {hi}
            {"→ Financial: assume log-normal" if is_financial else ""}

            Analyze:
            (a) Current level
            (b) Trend
            (c) Volatility regime
            (d) Low scenario (10th %ile)
            (e) High scenario (90th %ile)

            Output percentiles: 10, 20, 40, 60, 80, 90
            Confidence: WW%
            """)

        reasoning = await forecaster_llm.invoke(prompt)
        confidence = self._extract_confidence(reasoning, is_financial)

        try:
            if q_type == "binary":
                # ✅ Use parser_llm for structured output
                pred: BinaryPrediction = await structure_output(
                    reasoning, BinaryPrediction, model=self.parser_llm
                )
                value = np.clip(pred.prediction_in_decimal, 0.01, 0.99)
            elif q_type == "mcq":
                parsing_instructions = f"Valid options: {getattr(question, 'options', [])}"
                pred: PredictedOptionList = await structure_output(
                    reasoning,
                    PredictedOptionList,
                    model=self.parser_llm,
                    additional_instructions=parsing_instructions,
                )
                value = pred
            else:  # numeric
                pct_list: List[Percentile] = await structure_output(
                    reasoning,
                    list[Percentile],
                    model=self.parser_llm,
                )
                dist = NumericDistribution.from_question(pct_list, question)
                if is_financial:
                    dist = self._apply_financial_adjustment(dist, question)
                value = dist
        except Exception as e:
            logger.warning(f"Parsing failed for {model_key} on Q{question.id}: {e}. Using fallback.")
            # ✅ Type-safe fallbacks
            if q_type == "binary":
                value = 0.5
            elif q_type == "mcq":
                opts = getattr(question, "options", ["A", "B"])
                value = PredictedOptionList({o: round(100.0 / len(opts), 1) for o in opts})
            else:  # numeric
                lo = getattr(question, "lower_bound", 0)
                hi = getattr(question, "upper_bound", 100)
                fallback_pcts = [Percentile(p / 100, lo + (hi - lo) * p / 100) for p in [10, 20, 40, 60, 80, 90]]
                value = NumericDistribution.from_question(fallback_pcts, question)

        return ReasonedPrediction(prediction_value=value, reasoning=reasoning), confidence

    # ===== STATISTICALLY SOUND AGGREGATION WITH TYPE GUARD =====
    async def _run_generic_forecast(self, question: MetaculusQuestion, research: str) -> ReasonedPrediction:
        weights = self._get_dynamic_weights(question)
        raw_results: List[Tuple[ReasonedPrediction, float, str]] = []

        for model_key in self.FORECAST_MODELS:
            try:
                pred, conf = await self._run_forecast_with_confidence(question, research, model_key)
                effective_weight = weights[model_key] * conf
                raw_results.append((pred, effective_weight, model_key))
                logger.info(f"✓ {model_key}: conf={conf:.2f}, wt={effective_weight:.2f}")
            except Exception as e:
                logger.error(f"✗ {model_key} failed: {e}")

        if not raw_results:
            raise RuntimeError("All models failed.")

        q_type = self._categorize_question(question)
        is_financial = self._is_financial_question(question)

        # ✅ Filter to only type-consistent predictions
        valid_results = []
        for pred, wt, model_key in raw_results:
            pv = pred.prediction_value
            if q_type == "binary" and isinstance(pv, (int, float)):
                valid_results.append((pred, wt, model_key))
            elif q_type == "mcq" and isinstance(pv, PredictedOptionList):
                valid_results.append((pred, wt, model_key))
            elif q_type == "numeric" and isinstance(pv, NumericDistribution):
                valid_results.append((pred, wt, model_key))
            else:
                logger.warning(f"Discarding {model_key}: expected {q_type}, got {type(pv)}")

        if not valid_results:
            raise ValueError(f"No valid {q_type} predictions collected.")

        combined_reasoning = "\n\n".join(f"[{r[2]}] {r[0].reasoning}" for r in valid_results)

        if q_type == "binary":
            values = [float(r[0].prediction_value) for r in valid_results]
            weights_list = [r[1] for r in valid_results]
            final_pred = np.average(values, weights=weights_list)
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        elif q_type == "mcq":
            options = getattr(question, "options", [])
            if not options:
                raise ValueError("MCQ has no options")
            alpha = 0.5
            smoothed = np.full(len(options), alpha * sum(wt for _, wt, _ in valid_results))
            for pred, wt, _ in valid_results:
                pv = pred.prediction_value  # PredictedOptionList
                for i, opt in enumerate(options):
                    try:
                        p = pv.get_probability_for_option(opt)
                        smoothed[i] += p * wt
                    except:
                        smoothed[i] += (100.0 / len(options)) * wt
            final_probs = smoothed / smoothed.sum()
            final_pred = PredictedOptionList(dict(zip(options, final_probs.tolist())))
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        else:  # numeric
            target_percentiles = [10, 20, 40, 60, 80, 90]
            ensemble_pcts = []
            for p in target_percentiles:
                vals, wts = [], []
                for pred, wt, _ in valid_results:
                    dist = pred.prediction_value  # NumericDistribution
                    closest = min(
                        dist.declared_percentiles,
                        key=lambda x: abs(x.percentile - p / 100),
                    )
                    if abs(closest.percentile - p / 100) < 0.15:
                        vals.append(closest.value)
                        wts.append(wt)
                if vals:
                    if is_financial and all(v > 0 for v in vals):
                        log_vals = [np.log(v) for v in vals]
                        log_q = np.average(log_vals, weights=wts)
                        q = np.exp(log_q)
                    else:
                        q = np.average(vals, weights=wts)
                else:
                    lo = getattr(question, "lower_bound", 0)
                    hi = getattr(question, "upper_bound", 1)
                    q = lo + (hi - lo) * p / 100.0
                ensemble_pcts.append(Percentile(percentile=p / 100.0, value=q))
            final_dist = NumericDistribution.from_question(ensemble_pcts, question)
            return ReasonedPrediction(prediction_value=final_dist, reasoning=combined_reasoning)

    # ===== Routing =====
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        return await self._run_generic_forecast(question, research)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        return await self._run_generic_forecast(question, research)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_generic_forecast(question, research)


# ===== MAIN =====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["32916", "ACX2026", "minibench", "all"],
        default="all",
        help="Tournament: 32916, ACX2026, minibench, or all",
    )
    args = parser.parse_args()

    # Check env vars
    required = ["TAVILY_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_CLIENT_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    # ✅ Configure all roles explicitly
    bot = ConfidenceWeightedEnsembleBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "researcher": GeneralLlm(
                model=get_model_id("gpt-5.1"),
                temperature=0.3,
                timeout=60,
            ),
            "summarizer": GeneralLlm(
                model=get_model_id("gpt-5.1"),
                temperature=0.2,
                timeout=45,
            ),
            "parser": GeneralLlm(
                model=get_model_id("gpt-5.1"),  # deterministic parsing
                temperature=0.0,
                timeout=30,
            ),
        },
    )

    async def run():
        tournament_map = {
            "32916": 32916,
            "ACX2026": "ACX2026",
            "minibench": MetaculusApi.CURRENT_MINIBENCH_ID,
        }
        targets = list(tournament_map.values()) if args.mode == "all" else [tournament_map[args.mode]]

        reports = []
        for tid in targets:
            logger.info(f"▶ Forecasting tournament: {tid}")
            r = await bot.forecast_on_tournament(tid, return_exceptions=True)
            reports.extend(r)

        bot.log_report_summary(reports)
        logger.info(f"📊 Queries used — Tavily: {bot._tavily_count}, AskNews: {bot._asknews_count}")

    asyncio.run(run())
