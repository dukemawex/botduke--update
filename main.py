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
    - Research: AskNews (breaking financial events) + Tavily (deep context)
    - Models: gpt-5, gpt-5.1 (calibration), claude-sonnet-4.5 (nuance)
    - Special handling for stock/financial questions:
        • Log-normal ensemble for prices
        • Volatility-scaled confidence
        • Base-rate anchoring (S&P range, VIX regimes)
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

        # 🤖 Forecast models (with fallback resolution)
        self.FORECAST_MODELS = {
            "gpt-5": "openrouter/openai/gpt-5",
            "gpt-5.1": get_model_id("gpt-5.1"),
            "claude-sonnet-4.5": get_model_id("claude-sonnet-4.5"),
        }

        # 📈 Base weights — gpt-5.1 favored for calibration
        self.BASE_WEIGHTS = {
            "gpt-5": 1.0,
            "gpt-5.1": 1.3,  # ↑ Higher — expected best calibration
            "claude-sonnet-4.5": 1.1,
        }

        self._current_question: Optional[MetaculusQuestion] = None

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
            r"\$[a-z]{1,5}\b",  # e.g., $AAPL
            r"\b(?:[a-z]{1,4}\.?\s*[ou]n\s+\d{1,2}/\d{1,2})\b",  # e.g., TSLA on 12/15
            r"\bby\s+\d{4}\b.*\b(s&p|nasdaq)",
        ]
        return any(re.search(pat, text) for pat in financial_terms)

    async def run_research(self, question: MetaculusQuestion) -> str:
        self._current_question = question
        async with self._concurrency_limiter:
            try:
                q_text = f"{question.question_text} {question.resolution_criteria}"[:250]
                is_financial = self._is_financial_question(question)

                # Adjust query strategy for financial questions
                if is_financial:
                    # Prioritize AskNews for events, earnings, Fed actions
                    asknews_query = f"Financial market update: {q_text}"
                    tavily_query = f"Long-term trends and base rates for: {q_text}"
                else:
                    asknews_query = tavily_query = q_text

                # Run in parallel
                tavily_task = self._run_tavily_research(tavily_query)
                asknews_task = self._run_asknews_research(asknews_query, financial=is_financial)

                tavily_res, asknews_res = await asyncio.gather(tavily_task, asknews_task)

                # Combine with financial emphasis
                combined = clean_indents(f"""
                ### Research Summary (as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
                {"[FINANCIAL QUESTION]" if is_financial else ""}

                {'[AskNews — Breaking Events]' + chr(10) + asknews_res if asknews_res else ''}
                {'[Tavily — Deep Context]' + chr(10) + tavily_res if tavily_res else ''}

                ⚠️ Forecaster Guidance:
                - For financial questions: anchor to base rates (e.g., S&P 52-wk range, historical vol).
                - Adjust for regime: high VIX → widen intervals; low vol → narrow cautiously.
                - Never ignore resolution criteria — e.g., "closing price on YYYY-MM-DD".
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
        text = (q.question_text + " " + (q.background_info or "")).lower()
        if any(kw in text for kw in ["probability", "will", "by", "happen", "occur", "?"]):
            return "binary"
        if any(kw in text for kw in ["which", "who", "what", "option", "among"]):
            return "mcq"
        return "numeric"

    def _get_dynamic_weights(self, question: MetaculusQuestion) -> Dict[str, float]:
        weights = self.BASE_WEIGHTS.copy()
        q_text = question.question_text.lower()
        q_type = self._categorize_question(question)
        is_financial = self._is_financial_question(question)

        # Type-based
        if q_type == "binary":
            weights["gpt-5.1"] *= 1.4  # best calibrated
        elif q_type == "numeric":
            if is_financial:
                weights["gpt-5"] *= 1.2  # deeper financial reasoning
                weights["claude-sonnet-4.5"] *= 1.1
            else:
                weights["gpt-5"] *= 1.3
        elif q_type == "mcq":
            weights["claude-sonnet-4.5"] *= 1.3  # contrastive strength

        # Financial boost
        if is_financial:
            weights["gpt-5.1"] *= 1.1  # improved quant calibration

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    # ===== FORECASTING CORE =====
    async def _run_forecast_with_confidence(
        self, question: MetaculusQuestion, research: str, model_key: str
    ) -> Tuple[ReasonedPrediction, float]:
        model_name = self.FORECAST_MODELS[model_key]
        llm = GeneralLlm(model=model_name, temperature=0.2, timeout=60, allowed_tries=2)

        is_financial = self._is_financial_question(question)
        q_type = self._categorize_question(question)

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
            (a) Time to resolution — short-term (<30d) → higher noise
            (b) Base rate (e.g., % of similar bills passed, startups succeeded)
            {"(c) Market-implied probability (if options/futures exist)" if is_financial else ""}
            (d) Status quo trajectory

            Output:
            Rationale: ...
            Probability: ZZ% (0–100)
            Confidence: WW% (50=guess, 100=certain; ↓ if short-term/financial)
            """)
        elif q_type == "mcq":
            options = getattr(question, "options", [])
            prompt = base_prompt + clean_indents(f"""
            Options: {options}

            Analyze:
            (a) Most likely option (status quo)
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
            {"→ Financial: assume log-normal; anchor to historical ranges" if is_financial else ""}

            Analyze:
            (a) Current level (e.g., S&P 52-wk range)
            (b) Trend (growth/decay rate)
            (c) Volatility regime (high VIX → widen intervals)
            (d) Plausible low (10th %ile)
            (e) Plausible high (90th %ile)

            Output percentiles: 10, 20, 40, 60, 80, 90
            Confidence: WW%
            """)

        reasoning = await llm.invoke(prompt)
        confidence = self._extract_confidence(reasoning, is_financial)

        try:
            if q_type == "binary":
                pred: BinaryPrediction = await structure_output(
                    reasoning, BinaryPrediction, model=llm
                )
                value = np.clip(pred.prediction_in_decimal, 0.01, 0.99)
            elif q_type == "mcq":
                parsing_instructions = f"Valid options: {getattr(question, 'options', [])}"
                pred: PredictedOptionList = await structure_output(
                    reasoning, PredictedOptionList, model=llm,
                    additional_instructions=parsing_instructions,
                )
                value = pred
            else:  # numeric
                pct_list: List[Percentile] = await structure_output(
                    reasoning, list[Percentile], model=llm
                )
                dist = NumericDistribution.from_question(pct_list, question)
                # ✅ Apply financial adjustment if needed
                if is_financial:
                    dist = self._apply_financial_adjustment(dist, question)
                value = dist
        except Exception as e:
            logger.warning(f"Parsing failed for {model_key} on Q{question.id}: {e}")
            # Fallbacks
            if q_type == "binary":
                value = 0.5
            elif q_type == "mcq":
                opts = getattr(question, "options", ["A", "B"])
                value = PredictedOptionList({o: 100.0 / len(opts) for o in opts})
            else:
                lo = getattr(question, "lower_bound", 0)
                hi = getattr(question, "upper_bound", 100)
                fallback_pcts = [Percentile(p / 100, lo + (hi - lo) * p / 100) for p in [10, 20, 40, 60, 80, 90]]
                value = NumericDistribution.from_question(fallback_pcts, question)

        return ReasonedPrediction(prediction_value=value, reasoning=reasoning), confidence

    def _extract_confidence(self, text: str, is_financial: bool = False) -> float:
        match = re.search(r"Confidence:\s*([\d.]+)%?", text, re.IGNORECASE)
        base_conf = float(match.group(1)) / 100.0 if match else 0.7
        base_conf = np.clip(base_conf, 0.3, 1.0)

        # ↓ Downweight short-term financial questions (inherently noisy)
        if is_financial:
            base_conf *= 0.9
        return base_conf

    def _get_bounds(self, q: NumericQuestion) -> Tuple[str, str]:
        lo = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        hi = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        lo_str = f"≥{lo}" if q.open_lower_bound else f"={lo} (hard min)"
        hi_str = f"≤{hi}" if q.open_upper_bound else f"={hi} (hard max)"
        return lo_str, hi_str

    def _apply_financial_adjustment(self, dist: NumericDistribution, question: NumericQuestion) -> NumericDistribution:
        """
        For stock/financial numeric questions:
        - Enforce log-normality (prices can't be negative; multiplicative shocks)
        - Apply double-median stabilization
        - Anchor to plausible market regimes
        """
        try:
            # Get percentiles
            pcts = {int(p.percentile * 100): p.value for p in dist.declared_percentiles}
            needed = [10, 20, 40, 50, 60, 80, 90]

            # Ensure median (50) exists
            if 50 not in pcts:
                if 40 in pcts and 60 in pcts:
                    pcts[50] = (pcts[40] + pcts[60]) / 2
                else:
                    pcts[50] = np.median(list(pcts.values()))

            median = pcts[50]
            if median <= 0:
                return dist  # can't log-transform

            # Log-transform, apply double-median, transform back
            log_pcts = {p: np.log(v) for p, v in pcts.items() if v > 0}

            new_log_pcts = {}
            for p in needed:
                if p in log_pcts:
                    # Pull extremes toward median (robustify)
                    shrink = 0.4 if abs(p - 50) >= 30 else 0.15
                    new_log = (1 - shrink) * log_pcts[p] + shrink * np.log(median)
                    new_log_pcts[p] = new_log
                else:
                    # Interpolate
                    lower = max([k for k in log_pcts.keys() if k <= p], default=min(log_pcts.keys()))
                    upper = min([k for k in log_pcts.keys() if k >= p], default=max(log_pcts.keys()))
                    if lower == upper:
                        new_log_pcts[p] = log_pcts[lower]
                    else:
                        w = (p - lower) / (upper - lower)
                        new_log_pcts[p] = (1 - w) * log_pcts[lower] + w * log_pcts[upper]

            # Back to linear scale
            new_pcts = {p: np.exp(v) for p, v in new_log_pcts.items()}

            # Build final percentiles (standard set)
            final_percentiles = [
                Percentile(percentile=p / 100.0, value=new_pcts[p])
                for p in [10, 20, 40, 60, 80, 90]
            ]
            return NumericDistribution.from_question(final_percentiles, question)

        except Exception as e:
            logger.warning(f"Financial adjustment failed: {e}")
            return dist

    # ===== STATISTICALLY OPTIMAL AGGREGATION =====
    async def _run_generic_forecast(self, question: MetaculusQuestion, research: str) -> ReasonedPrediction:
        weights = self._get_dynamic_weights(question)
        results: List[Tuple[ReasonedPrediction, float, str]] = []

        for model_key in self.FORECAST_MODELS:
            try:
                pred, conf = await self._run_forecast_with_confidence(question, research, model_key)
                effective_weight = weights[model_key] * conf
                results.append((pred, effective_weight, model_key))
                logger.info(f"✓ {model_key}: conf={conf:.2f}, wt={effective_weight:.2f}")
            except Exception as e:
                logger.error(f"✗ {model_key} failed: {e}")

        if not results:
            raise RuntimeError("All models failed.")

        q_type = self._categorize_question(question)
        is_financial = self._is_financial_question(question)
        combined_reasoning = "\n\n".join(f"[{r[2]}] {r[0].reasoning}" for r in results)

        if q_type == "binary":
            # ✅ Brier-optimal: weighted *mean*
            values = []
            wts = []
            for pred, wt, _ in results:
                if isinstance(pred.prediction_value, (int, float)):
                    values.append(float(pred.prediction_value))
                    wts.append(wt)
            if not values:
                raise ValueError("No valid binary predictions")
            final_pred = np.average(values, weights=wts)
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        elif q_type == "mcq":
            options = getattr(question, "options", [])
            if not options:
                raise ValueError("MCQ has no options")

            # Dirichlet smoothing (α=0.5)
            alpha = 0.5
            smoothed = np.full(len(options), alpha * sum(wt for _, wt, _ in results))
            for pred, wt, _ in results:
                if isinstance(pred.prediction_value, PredictedOptionList):
                    for i, opt in enumerate(options):
                        try:
                            p = pred.prediction_value.get_probability_for_option(opt)
                            smoothed[i] += p * wt
                        except:
                            smoothed[i] += (1.0 / len(options)) * wt
            final_probs = smoothed / smoothed.sum()
            final_pred = PredictedOptionList(dict(zip(options, final_probs.tolist())))
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        else:  # numeric
            target_percentiles = [10, 20, 40, 60, 80, 90]
            ensemble_pcts = []

            for p in target_percentiles:
                vals = []
                wts = []
                for pred, wt, _ in results:
                    if isinstance(pred.prediction_value, NumericDistribution):
                        dist = pred.prediction_value
                        # Find closest declared percentile
                        closest = min(
                            dist.declared_percentiles,
                            key=lambda x: abs(x.percentile - p / 100),
                        )
                        if abs(closest.percentile - p / 100) < 0.15:  # within 15pp
                            vals.append(closest.value)
                            wts.append(wt)

                if vals:
                    if is_financial:
                        # Log-average for financial
                        log_vals = [np.log(v) for v in vals if v > 0]
                        log_wts = [w for v, w in zip(vals, wts) if v > 0]
                        if log_vals:
                            log_q = np.average(log_vals, weights=log_wts)
                            q = np.exp(log_q)
                        else:
                            q = np.average(vals, weights=wts)
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

    bot = ConfidenceWeightedEnsembleBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "researcher": GeneralLlm(
                model=get_model_id("gpt-5.1"),  # ✅ Upgraded
                temperature=0.3,
                timeout=60,
            ),
            "parser": GeneralLlm(
                model=get_model_id("gpt-5.1"),  # ✅ More reliable parsing
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
